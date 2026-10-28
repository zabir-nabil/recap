import os
import time
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from recap import config

logger = logging.getLogger(__name__)
_completed_call_capture = ContextVar("completed_call_capture", default=None)


@contextmanager
def capture_completed_calls():
    """Capture usage only, for billing unexpected failed question attempts."""
    calls = []
    token = _completed_call_capture.set(calls)
    try:
        yield calls
    finally:
        _completed_call_capture.reset(token)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int               # generated text tokens ONLY (excludes reasoning)
    reasoning_tokens: int = 0        # thinking tokens (Gemini 2.5) / reasoning tokens (o-series, gpt-5)

    def __post_init__(self):
        # guard against negative token counts from SDK bugs
        if self.input_tokens < 0 or self.output_tokens < 0 or self.reasoning_tokens < 0:
            logger.warning(
                "LLMResponse received negative token counts (in=%d out=%d reasoning=%d); "
                "clamping to 0 -- check provider SDK version",
                self.input_tokens, self.output_tokens, self.reasoning_tokens,
            )
            self.input_tokens = max(0, self.input_tokens)
            self.output_tokens = max(0, self.output_tokens)
            self.reasoning_tokens = max(0, self.reasoning_tokens)

    @property
    def billable_output_tokens(self) -> int:
        """total output for cost computation. reasoning is billed at output rate."""
        return self.output_tokens + self.reasoning_tokens


def call_llm(messages, model=None, provider=None, temperature=None, max_tokens=None,
             thinking_budget=None, seed=None):
    """call an llm, return response with token counts.

    temperature: None means "use config.TEMPERATURE", which itself defaults to
    None = "do not send a temperature; let the provider use its own default".
    A concrete value is passed through verbatim. Keeping None as the default
    across every call site is what makes sampling identical for all methods
    (see config.TEMPERATURE).
    """
    model = model or config.PLANNER_MODEL
    provider = provider or config.PLANNER_PROVIDER
    temperature = temperature if temperature is not None else config.TEMPERATURE
    max_tokens = max_tokens or config.MAX_TOKENS
    seed = config.EXPERIMENT_SEED if seed is None else int(seed)

    if provider == "openai":
        response = _call_openai(messages, model, temperature, max_tokens, seed=seed)
    elif provider == "deepseek":
        response = _call_openai_compatible(
            messages, model, temperature, max_tokens,
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
    elif provider == "gemini":
        response = _call_gemini(
            messages, model, temperature, max_tokens,
            thinking_budget=thinking_budget, seed=seed,
        )
    elif provider == "openrouter":
        response = _call_openai_compatible(
            messages, model, temperature, max_tokens,
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"unknown provider: {provider}")
    capture = _completed_call_capture.get()
    if capture is not None:
        capture.append({
            "model": model, "provider": provider, "role": "failed_attempt",
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "reasoning_tokens": response.reasoning_tokens,
        })
    return response


def _call_openai(messages, model, temperature, max_tokens, seed=None):
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # gpt-5+ and o-series are reasoning models:
    # - use max_completion_tokens instead of max_tokens
    # - temperature must be 1 (only value supported)
    is_reasoning = model.startswith("gpt-5") or model.startswith("o")
    params = {}
    if is_reasoning:
        params["max_completion_tokens"] = max_tokens
        # reasoning models only support temperature=1
    else:
        params["max_tokens"] = max_tokens
        # temperature=None -> omit the field so the API applies its own default
        if temperature is not None:
            params["temperature"] = temperature
    if seed is not None:
        params["seed"] = int(seed)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **params,
    )

    # reasoning models expose completion_tokens_details.reasoning_tokens;
    # subtract so output_tokens reports only generated text (billable at same rate)
    reasoning = 0
    total_completion = resp.usage.completion_tokens or 0
    details = getattr(resp.usage, "completion_tokens_details", None)
    if details is not None:
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
    output_only = max(0, total_completion - reasoning)

    return LLMResponse(
        text=resp.choices[0].message.content or "",
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=output_only,
        reasoning_tokens=reasoning,
    )


def _gemini_thinking_budget(model, max_tokens):
    """return (thinking_budget, max_output_tokens). model-aware.

    default policy (no env override): DYNAMIC thinking (budget=-1) with an extra
    8192-token answer headroom on top of max_tokens. This is the equivalent of
    GEMINI_THINKING_BUDGET=-1 and is the default because the alternative —
    leaving thinking unconstrained under a single max_output_tokens ceiling —
    lets a hard question spend the whole budget on thinking and hit the ceiling with no room
    left for the visible answer, yielding an empty MALFORMED_RESPONSE. Giving
    thinking its own dynamic budget plus a dedicated answer headroom keeps the
    answer from being starved. Gemini 3.1 Flash-Lite also supports thinking;
    its reasoning tokens must be included in output-token cost accounting.

    env override GEMINI_THINKING_BUDGET:
      - "0": disable thinking (pro clamps to 128; SDK rejects 0 for pro)
      - "-1" or "dynamic": dynamic thinking (same as the default)
      - integer >= 128: explicit budget
      - "none"/"unset": legacy behavior — no thinking_config, single max_tokens
        ceiling (kept for reproducing the pre-default runs / ablation)
    """
    model_l = (model or "").lower()
    is_pro = "2.5-pro" in model_l or "3.1-pro" in model_l

    raw = os.getenv("GEMINI_THINKING_BUDGET")
    if raw is None:
        # DEFAULT (no env override): dynamic thinking + answer headroom, so a
        # heavy-thinking call never starves the visible answer. Equivalent to
        # setting GEMINI_THINKING_BUDGET=-1.
        return -1, max_tokens + 8192

    s = raw.strip().lower()
    if s in ("none", "unset"):
        # explicit escape hatch: reproduce the legacy no-thinking_config path
        # (single max_output_tokens ceiling shared by thinking + answer).
        return None, max_tokens
    if s in ("-1", "dynamic"):
        budget = -1
    else:
        try:
            budget = int(s)
        except ValueError:
            budget = 0
    if is_pro and budget != -1 and 0 < budget < 128:
        budget = 128
    if budget == -1:
        return budget, max_tokens + 8192
    elif budget > 0:
        return budget, max_tokens + budget
    else:
        if is_pro:
            return 128, max_tokens + 128
        return 0, max_tokens


# capacity/transient-error hints (matched case-insensitively as substrings).
# NOTE: use specific tokens, never a bare "rate" — "rate" is a substring of
# "generate"/"generateContent" (which appear in many non-retriable Gemini error
# bodies, e.g. INVALID_ARGUMENT), so a bare "rate" would misclassify hard errors
# as capacity and retry them. "429" + "rate limit"/"rate_limit" capture the
# actual rate-limit case without that collision.
_GEMINI_CAPACITY_HINTS = (
    "503", "429", "resource_exhausted", "rate limit", "rate_limit",
    "deadline", "unavailable", "internal", "timeout", "timed out",
    # transient transport drops (e.g. a socket torn down by laptop sleep/wake or
    # a brief network blip); a retry opens a fresh connection and usually
    # succeeds. specific phrases, so they can't collide with non-retriable bodies.
    "connection reset", "reset by peer", "broken pipe", "connection aborted",
)


def _generate_with_retry(client, model, contents, gen_config, max_retries=3, base_delay=2.0):
    """call client.models.generate_content with capacity-only retry + backoff.

    retries on 503 / 429 / RESOURCE_EXHAUSTED / rate-limit / DEADLINE /
    UNAVAILABLE / INTERNAL / timeout. fails fast on INVALID_ARGUMENT / SAFETY /
    RECITATION (not retriable). failed attempts are not billed by Gemini.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            capacity = any(h in msg for h in _GEMINI_CAPACITY_HINTS)
            if not capacity or attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            logger.warning(
                "gemini capacity error (attempt %d/%d): %s; retry in %.1fs",
                attempt + 1, max_retries, e, wait,
            )
            time.sleep(wait)
    raise last_exc  # defensive (unreachable)


def _call_gemini(messages, model, temperature, max_tokens, thinking_budget=None,
                 seed=None):
    """call gemini via the google-genai SDK.

    gemini-2.5/3.x flash and pro are thinking models: max_output_tokens
    INCLUDES thinking tokens, so thinking needs its own budget plus answer
    headroom or a heavy-thinking call starves the visible answer. The default
    (see _gemini_thinking_budget) is dynamic thinking (-1) with +8192 headroom.
    safety filters are set to BLOCK_NONE so benchmark questions never get
    silently empty-text'd by safety blocks (RECITATION still cannot be configured).
    """
    from google import genai
    from google.genai import types

    # per-request timeout (ms) so a hung/stalled stream fails fast into the
    # capacity-retry loop below instead of blocking the whole run forever
    # (observed: a single pathological question can hang a stream indefinitely).
    # override with GEMINI_TIMEOUT_S; default 300s.
    try:
        timeout_s = float(os.getenv("GEMINI_TIMEOUT_S", "300"))
    except ValueError:
        timeout_s = 300.0
    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )

    # convert openai message format to gemini format
    system_text = None
    contents = []
    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_text = text
        elif role == "user":
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            ))
        elif role == "assistant":
            contents.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=text)],
            ))

    # model-aware thinking budget. default is dynamic thinking (-1) with answer
    # headroom so a heavy-thinking call never starves the visible answer; budget
    # is None only when GEMINI_THINKING_BUDGET is explicitly "none"/"unset"
    # (legacy single-ceiling behavior), in which case thinking_config is left off.
    #
    # per-call override (thinking_budget): lets cheap decision calls (judge /
    # format+verify) request a small fixed budget instead of the dynamic default,
    # cutting reasoning-token cost ~17x with no verdict change at 1024 (validated).
    # The global GEMINI_THINKING_BUDGET env still wins, for reproducibility/ablation.
    if thinking_budget is not None and os.getenv("GEMINI_THINKING_BUDGET") is None:
        budget = int(thinking_budget)
        max_tokens = max_tokens + max(budget, 0)
    else:
        budget, max_tokens = _gemini_thinking_budget(model, max_tokens)
    thinking = types.ThinkingConfig(thinking_budget=budget) if budget is not None else None

    # safety: disable all four configurable categories so benchmark
    # questions don't get silently blocked. RECITATION cannot be configured.
    safety_settings = [
        types.SafetySetting(category=cat, threshold="BLOCK_NONE")
        for cat in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        )
    ]

    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_text,
        thinking_config=thinking,
        safety_settings=safety_settings,
        seed=seed,
    )

    response = _generate_with_retry(client, model, contents, gen_config)

    usage = response.usage_metadata
    text_out = response.text or ""

    if not text_out.strip():
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        thought_tok = (getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0
        cand_tok = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        if thought_tok > 0 and cand_tok == 0:
            logger.warning(
                "gemini empty text after %d thinking tokens "
                "(model=%s finish_reason=%s); raise budget or set GEMINI_THINKING_BUDGET",
                thought_tok, model, finish_reason,
            )
        elif str(finish_reason or "").upper() in ("SAFETY", "RECITATION"):
            logger.error("gemini blocked response: finish_reason=%s model=%s", finish_reason, model)

    if usage is None:
        # defensive fallback — usage_metadata should always be present
        logger.warning(
            "gemini response missing usage_metadata; token cost for this call "
            "is unrecoverable (model=%s, text_len=%d)", model, len(text_out),
        )
        return LLMResponse(text=text_out, input_tokens=0, output_tokens=0)

    # Gemini 2.5 SDK reports these three fields independently:
    #   prompt_token_count    - input
    #   candidates_token_count - GENERATED text only (does NOT include thinking)
    #   thoughts_token_count   - thinking-only tokens
    #   total_token_count      = prompt + candidates + thoughts
    # Both generated and thinking tokens are billed at the output rate by Gemini.
    # We keep them separate so callers can report them independently.
    return LLMResponse(
        text=text_out,
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
        reasoning_tokens=getattr(usage, "thoughts_token_count", None) or 0,
    )


def _call_openai_compatible(messages, model, temperature, max_tokens, api_key, base_url):
    """works with any openai-compatible api (deepseek, openrouter, together, etc.)"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    params = {"max_tokens": max_tokens}
    # temperature=None -> omit the field so the API applies its own default
    if temperature is not None:
        params["temperature"] = temperature
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **params,
    )

    # match _call_openai: handle reasoning_tokens if the provider exposes them
    reasoning = 0
    total_completion = resp.usage.completion_tokens or 0
    details = getattr(resp.usage, "completion_tokens_details", None)
    if details is not None:
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
    output_only = max(0, total_completion - reasoning)

    return LLMResponse(
        text=resp.choices[0].message.content or "",
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=output_only,
        reasoning_tokens=reasoning,
    )
