"""Run a single RECAP configuration over an ordered question stream."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=[
        "financebench", "sql_api_bench", "gaia_l1", "gaia_l2",
    ])
    source.add_argument("--factory", help="module:function returning (questions, tools, preamble)")
    parser.add_argument("--dataset-name", default="custom", help="Plan-library namespace for a custom factory")
    parser.add_argument("--scorer", help="module:function accepting (result, question)")
    parser.add_argument("--agent", choices=["react", "plan_and_act"], default="react")
    parser.add_argument("--mode", choices=["lite", "verified"], default="verified")
    parser.add_argument("--groups", type=int, help="Number of complete families; omit or use 0 for all")
    parser.add_argument("--order", choices=["family", "file"], default="family")
    parser.add_argument("--tau", type=float, help="Retrieval cosine-similarity threshold")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--revision", help="Hugging Face dataset revision")
    parser.add_argument("--sql-data-dir", help="Directory containing the prepared Spider databases")
    parser.add_argument("--output", default="results/recap.jsonl", help="Per-question JSONL output")
    return parser


def _callable(spec):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("Expected module:function")
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise TypeError(f"{spec} is not callable")
    return value


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.groups is not None and args.groups < 0:
        parser.error("--groups must be nonnegative")
    if args.top_k < 1 or args.max_steps < 1:
        parser.error("--top-k and --max-steps must be positive")

    from recap import RecapEngine, run_with_recap, config
    from recap.datasets import load_benchmark, normalize_questions, order_questions
    from recap.scoring import score_simple

    if args.seed is not None:
        config.set_experiment_seed(args.seed)
    if args.factory:
        records, tools, preamble = _callable(args.factory)()
        questions = order_questions(
            normalize_questions(records), groups=args.groups, order=args.order,
        )
        dataset = args.dataset_name
    else:
        questions, tools, preamble = load_benchmark(
            args.dataset, groups=args.groups, order=args.order,
            sql_data_dir=args.sql_data_dir, revision=args.revision,
        )
        dataset = "gaia" if args.dataset.startswith("gaia_") else args.dataset
    custom_scorer = _callable(args.scorer) if args.scorer else None
    engine = RecapEngine(
        tools=tools, dataset=dataset, preamble=preamble,
        agent_type=args.agent, config_mode="direct" if args.mode == "lite" else "verified",
        tau=args.tau, top_k=args.top_k, max_steps=args.max_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = correct = scored = 0
    total_cost = 0.0
    with output.open("w", encoding="utf-8") as handle:
        for question in questions:
            result = run_with_recap(question, engine)
            result["variant_type"] = question.get("variant_type", "original")
            result["parent_id"] = question.get("parent_id")
            if custom_scorer is not None:
                score = custom_scorer(result, question)
            elif args.dataset and question.get("expected") not in (None, ""):
                score = score_simple(dataset, result, question)
            else:
                score = None
            if score is not None:
                accepted, reason = score if isinstance(score, tuple) else (score, "custom")
                result["correct"] = bool(accepted)
                result["score_reason"] = str(reason)
                correct += bool(accepted)
                scored += 1
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            count += 1
            total_cost += result.get("usd_cost", 0.0)
            print(f"[{count}/{len(questions)}] {question['id']}: "
                  f"{result.get('metadata', {}).get('recap_path', '')} "
                  f"${result.get('usd_cost', 0.0):.4f}")
    summary = {"questions": count, "model_cost_usd": round(total_cost, 6), "output": str(output)}
    if scored:
        summary.update(scored_questions=scored, accuracy=correct / scored)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
