import os
import math
import sqlite3
from pathlib import Path
from urllib.request import pathname2url
from recap.tools import BaseTool


def find_database_dir(data_dir):
    """find spider database directory inside the data dir.
    compatible with the structure created by the setup script."""
    data_path = Path(data_dir)
    candidates = [
        data_path / "spider" / "spider_data" / "database",
        data_path / "spider" / "database",
        data_path / "spider" / "spider" / "database",
        data_path / "database",
    ]
    for p in candidates:
        if p.exists() and "__MACOSX" not in str(p):
            return p

    # fallback: search recursively
    if data_path.exists():
        for root, dirs, _ in os.walk(data_path):
            if "__MACOSX" in root:
                continue
            if "database" in dirs:
                candidate = Path(root) / "database"
                if "__MACOSX" not in str(candidate):
                    return candidate
    return None


def _get_db_path(database_dir, db_name):
    """resolve the sqlite file for a database name. validates path stays inside database_dir."""
    db_real = os.path.realpath(database_dir)
    p = Path(database_dir) / db_name / f"{db_name}.sqlite"
    if p.exists() and os.path.realpath(str(p)).startswith(db_real + os.sep):
        return str(p)
    p = Path(database_dir) / f"{db_name}.sqlite"
    if p.exists() and os.path.realpath(str(p)).startswith(db_real + os.sep):
        return str(p)
    return None


# ---- database tools ----

class GetSchema(BaseTool):
    name = "get_schema"
    description = "get the schema (create table statements) for all tables in a database."

    def __init__(self, database_dir):
        self.database_dir = database_dir

    def execute(self, db_name):
        path = _get_db_path(self.database_dir, db_name)
        if not path:
            return {"status": "error", "output": f"database not found: {db_name}", "output_type": "text"}
        try:
            conn = sqlite3.connect(path)
            cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
            schemas = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            return {"status": "success", "output": schemas, "output_type": "structured"}
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text"}


class SQLExecute(BaseTool):
    name = "sql_execute"
    description = "execute a read-only sql query on a database. returns rows as list of dicts."

    # block statements that modify data
    _BLOCKED = {"insert", "update", "delete", "drop", "alter", "create", "replace",
                "truncate", "attach", "vacuum", "reindex", "analyze"}

    def __init__(self, database_dir):
        self.database_dir = database_dir

    @staticmethod
    def _strip_sql_comments(query):
        """strip SQL block comments /* ... */ and line comments -- ..."""
        import re
        # remove block comments (non-greedy to handle multiple)
        query = re.sub(r'/\*.*?\*/', ' ', query, flags=re.DOTALL)
        # remove line comments
        query = re.sub(r'--[^\n]*', ' ', query)
        return query

    @staticmethod
    def _strip_string_literals(query):
        """replace single/double quoted string literals with a placeholder so
        that words inside them (e.g. WHERE note LIKE '%update%') do not trip the
        write-keyword guard. handles doubled-quote escapes ('' and "")."""
        import re
        query = re.sub(r"'(?:[^']|'')*'", " '' ", query)
        query = re.sub(r'"(?:[^"]|"")*"', ' "" ', query)
        return query

    @classmethod
    def _has_blocked_keyword(cls, query, blocked):
        """check whether a data-modifying keyword appears as a bare word.

        operates on a query that has already had comments AND string literals
        stripped. `replace` is allowed when used as the read-only scalar
        function `replace(` — only the statement forms (REPLACE INTO / INSERT OR
        REPLACE) remain blocked. identifiers like `update_time` are safe because
        the tokenizer keeps them as one token distinct from `update`.
        """
        import re
        # neutralize the read-only REPLACE() function so only bare `replace`
        # (e.g. REPLACE INTO) can still trigger the guard.
        query = re.sub(r'\breplace\s*\(', ' func( ', query, flags=re.IGNORECASE)
        words = re.findall(r'[a-zA-Z_]+', query.lower())
        return any(w in blocked for w in words)

    def execute(self, db_name, query):
        # write protection: strip comments and string literals, then check for
        # any bare data-modifying keyword.
        cleaned = self._strip_string_literals(self._strip_sql_comments(query))
        if self._has_blocked_keyword(cleaned, self._BLOCKED):
            return {"status": "error", "output": f"write operations not allowed", "output_type": "text"}

        path = _get_db_path(self.database_dir, db_name)
        if not path:
            return {"status": "error", "output": f"database not found: {db_name}", "output_type": "text"}
        try:
            # open READ-ONLY so model SQL can never mutate the shared benchmark
            # DB file (the keyword denylist above is defense-in-depth).
            conn = sqlite3.connect(f"file:{pathname2url(path)}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            if cursor.description:
                columns = [d[0] for d in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            else:
                rows = []
            conn.close()
            return {"status": "success", "output": rows, "output_type": "structured"}
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text"}


# ---- lexical apis ----

class CountSyllables(BaseTool):
    name = "count_syllables"
    description = "count the number of syllables in a word or phrase."

    def __init__(self):
        self._cmu = None

    def _load_cmu(self):
        if self._cmu is not None:
            return
        try:
            from nltk.corpus import cmudict
            self._cmu = cmudict.dict()
        except Exception:
            self._cmu = {}

    def _count_word(self, word):
        word = word.lower().strip()
        if not word or not word.isalpha():
            return 0
        self._load_cmu()
        if self._cmu and word in self._cmu:
            return sum(1 for p in self._cmu[word][0] if p[-1].isdigit())
        return self._heuristic(word)

    def _heuristic(self, word):
        vowels = "aeiouy"
        count = 0
        prev = False
        for ch in word.lower():
            is_v = ch in vowels
            if is_v and not prev:
                count += 1
            prev = is_v
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def execute(self, text):
        try:
            total = sum(self._count_word(w) for w in text.strip().split())
            return {"status": "success", "output": total, "output_type": "numeric"}
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text"}


# ---- numeric apis ----

class IsPrime(BaseTool):
    name = "is_prime"
    description = "check if a number is a prime number."

    def execute(self, n):
        try:
            n = int(n)
            if n < 2:
                return {"status": "success", "output": False, "output_type": "boolean"}
            if n < 4:
                return {"status": "success", "output": True, "output_type": "boolean"}
            if n % 2 == 0 or n % 3 == 0:
                return {"status": "success", "output": False, "output_type": "boolean"}
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return {"status": "success", "output": False, "output_type": "boolean"}
                i += 6
            return {"status": "success", "output": True, "output_type": "boolean"}
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text"}


class IsFibonacci(BaseTool):
    name = "is_fibonacci"
    description = "check if a number is a fibonacci number."

    def execute(self, n):
        try:
            n = int(n)
            if n < 0:
                return {"status": "success", "output": False, "output_type": "boolean"}
            def is_sq(x):
                if x < 0:
                    return False
                s = int(math.isqrt(x))
                return s * s == x
            result = is_sq(5 * n * n + 4) or is_sq(5 * n * n - 4)
            return {"status": "success", "output": result, "output_type": "boolean"}
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text"}


class IsPerfectSquare(BaseTool):
    name = "is_perfect_square"
    description = "check if a number is a perfect square."

    def execute(self, n):
        try:
            n = int(n)
            if n < 0:
                return {"status": "success", "output": False, "output_type": "boolean"}
            s = int(math.isqrt(n))
            return {"status": "success", "output": s * s == n, "output_type": "boolean"}
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text"}


class IsEven(BaseTool):
    name = "is_even"
    description = "check if a number is even."

    def execute(self, n):
        try:
            return {"status": "success", "output": int(n) % 2 == 0, "output_type": "boolean"}
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text"}


class IsDivisibleBy(BaseTool):
    name = "is_divisible_by"
    description = "check if a number is divisible by a given divisor."

    def execute(self, n, divisor):
        try:
            n, divisor = int(n), int(divisor)
            if divisor == 0:
                return {"status": "error", "output": "division by zero", "output_type": "text"}
            return {"status": "success", "output": n % divisor == 0, "output_type": "boolean"}
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text"}


# ---- geo-spatial apis (geopy-backed with runtime cache) ----

# country -> capital mapping

CAPITALS = {
    "afghanistan": "Kabul", "albania": "Tirana", "algeria": "Algiers",
    "argentina": "Buenos Aires", "armenia": "Yerevan", "australia": "Canberra",
    "austria": "Vienna", "azerbaijan": "Baku", "bahrain": "Manama",
    "bangladesh": "Dhaka", "belarus": "Minsk", "belgium": "Brussels",
    "bolivia": "Sucre", "bosnia and herzegovina": "Sarajevo", "brazil": "Brasilia",
    "bulgaria": "Sofia", "cambodia": "Phnom Penh", "canada": "Ottawa",
    "chile": "Santiago", "china": "Beijing", "colombia": "Bogota",
    "costa rica": "San Jose", "croatia": "Zagreb", "cuba": "Havana",
    "cyprus": "Nicosia", "czech republic": "Prague", "czechia": "Prague",
    "denmark": "Copenhagen", "ecuador": "Quito", "egypt": "Cairo",
    "england": "London", "estonia": "Tallinn", "ethiopia": "Addis Ababa",
    "finland": "Helsinki", "france": "Paris", "georgia": "Tbilisi",
    "germany": "Berlin", "greece": "Athens", "guatemala": "Guatemala City",
    "hungary": "Budapest", "iceland": "Reykjavik", "india": "New Delhi",
    "indonesia": "Jakarta", "iran": "Tehran", "iraq": "Baghdad",
    "ireland": "Dublin", "israel": "Jerusalem", "italy": "Rome",
    "japan": "Tokyo", "jordan": "Amman", "kazakhstan": "Astana",
    "kenya": "Nairobi", "kuwait": "Kuwait City", "latvia": "Riga",
    "lebanon": "Beirut", "libya": "Tripoli", "lithuania": "Vilnius",
    "luxembourg": "Luxembourg", "malaysia": "Kuala Lumpur", "mexico": "Mexico City",
    "mongolia": "Ulaanbaatar", "morocco": "Rabat", "myanmar": "Naypyidaw",
    "nepal": "Kathmandu", "netherlands": "Amsterdam", "new zealand": "Wellington",
    "nigeria": "Abuja", "north korea": "Pyongyang", "norway": "Oslo",
    "oman": "Muscat", "pakistan": "Islamabad", "panama": "Panama City",
    "paraguay": "Asuncion", "peru": "Lima", "philippines": "Manila",
    "poland": "Warsaw", "portugal": "Lisbon", "qatar": "Doha",
    "romania": "Bucharest", "russia": "Moscow", "saudi arabia": "Riyadh",
    "scotland": "Edinburgh", "serbia": "Belgrade", "singapore": "Singapore",
    "slovakia": "Bratislava", "slovenia": "Ljubljana", "south africa": "Pretoria",
    "south korea": "Seoul", "spain": "Madrid", "sri lanka": "Colombo",
    "sweden": "Stockholm", "switzerland": "Bern", "syria": "Damascus",
    "taiwan": "Taipei", "thailand": "Bangkok", "turkey": "Ankara",
    "ukraine": "Kyiv", "united arab emirates": "Abu Dhabi",
    "united kingdom": "London", "uk": "London",
    "united states": "Washington", "usa": "Washington", "us": "Washington",
    "uruguay": "Montevideo", "uzbekistan": "Tashkent",
    "venezuela": "Caracas", "vietnam": "Hanoi",
}


class Geocode(BaseTool):
    name = "geocode"
    # not deterministic: makes a live Nominatim network call whose result can
    # vary with time, ranking, and availability.
    is_deterministic = False
    description = (
        "look up geographic info for a place, city, or country. "
        "returns lat, lon, and country. if input is a country, also returns its capital. "
        "handles common name variations (e.g. 'USA', 'United States', 'US' all work)."
    )

    def __init__(self):
        self._geocoder = None
        self._cache = {}  # runtime cache: normalized_name -> result dict

    def _get_geocoder(self):
        if self._geocoder is None:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter
            raw = Nominatim(user_agent="sql_api_bench_tools", timeout=10)
            self._geocoder = RateLimiter(raw.geocode, min_delay_seconds=1.0)
        return self._geocoder

    def execute(self, place):
        key = place.strip().lower()

        # check runtime cache first
        if key in self._cache:
            return {"status": "success", "output": self._cache[key], "output_type": "structured"}

        result = {}

        # check if this is a country name -> add capital info
        if key in CAPITALS:
            result["capital"] = CAPITALS[key]

        # geocode using geopy
        try:
            geocoder = self._get_geocoder()
            location = geocoder(place, language="en", addressdetails=True)
            if location is None:
                # if direct lookup fails and it's a country, try geocoding the capital
                if key in CAPITALS:
                    location = geocoder(CAPITALS[key], language="en", addressdetails=True)

            if location is None:
                return {"status": "error", "output": f"could not geocode: {place}", "output_type": "text"}

            result["lat"] = location.latitude
            result["lon"] = location.longitude

            # extract country from address details if available
            addr = location.raw.get("address", {})
            if "country" in addr:
                result["country"] = addr["country"]

            self._cache[key] = result
            return {"status": "success", "output": result, "output_type": "structured"}

        except Exception as e:
            return {"status": "error", "output": f"geocoding error for '{place}': {e}", "output_type": "text"}


class GeoDistance(BaseTool):
    name = "geo_distance"
    # not deterministic: geocodes both places via a live Nominatim call.
    is_deterministic = False
    description = "calculate distance in km between two places, cities, or countries."

    def __init__(self, geocode_tool):
        self.geocode_tool = geocode_tool

    def execute(self, place1, place2):
        r1 = self.geocode_tool.execute(place=place1)
        if r1["status"] != "success" or "lat" not in r1.get("output", {}):
            return {"status": "error", "output": f"cannot locate: {place1}", "output_type": "text"}
        r2 = self.geocode_tool.execute(place=place2)
        if r2["status"] != "success" or "lat" not in r2.get("output", {}):
            return {"status": "error", "output": f"cannot locate: {place2}", "output_type": "text"}

        try:
            from geopy.distance import geodesic
            coord1 = (r1["output"]["lat"], r1["output"]["lon"])
            coord2 = (r2["output"]["lat"], r2["output"]["lon"])
            dist = geodesic(coord1, coord2).kilometers
            return {"status": "success", "output": round(dist, 2), "output_type": "numeric"}
        except Exception as e:
            return {"status": "error", "output": f"distance error: {e}", "output_type": "text"}
