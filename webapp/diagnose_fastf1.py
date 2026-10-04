"""Prints what FastF1 can actually fetch from this machine (used to debug CI)."""
import logging
import os
import sys
import tempfile

import fastf1

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s", stream=sys.stdout)

cache_dir = tempfile.mkdtemp(prefix="ff1diag_")
fastf1.Cache.enable_cache(cache_dir)

print("=== loading 2025 R1 Race cold ===")
s = fastf1.get_session(2025, 1, "R")
s.load(telemetry=False, weather=False, messages=False)
print("results rows:", len(s.results))
try:
    print("laps rows:", len(s.laps))
except Exception as e:
    print("laps UNAVAILABLE:", type(e).__name__)

n = sum(len(f) for _, _, f in os.walk(cache_dir))
size = sum(os.path.getsize(os.path.join(d, x)) for d, _, f in os.walk(cache_dir) for x in f)
print(f"cache files written: {n}, bytes: {size}")
