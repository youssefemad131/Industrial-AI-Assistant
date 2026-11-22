"""Run every test-set row through the real engine and check route + key content.
   python tests/run_testset.py          (exit code 1 if any row fails)
   python tests/run_testset.py --show   (also print every reply)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.engine import FactoryLens, validate_numbers, Ledger
from tests.testset_def import ROWS

show, bad = "--show" in sys.argv, 0
bot = FactoryLens()
for r in ROWS:
    sid = "t-" + r["id"]
    for s in r["setup"]:
        bot.ask(s, sid)
    out = bot.ask(r["prompt"], sid)
    ok_route = out["intent"].startswith(r["intent"])
    missing = [m for m in r["must"] if m not in out["reply"]]
    ok = ok_route and not missing
    bad += not ok
    print(f"{r['id']} {'PASS' if ok else 'FAIL'}  {r['type']:<12} intent={out['intent']}" + ("" if ok else f"  expected={r['intent']} missing={missing}"))
    if show:
        print("  USER:", r["prompt"]); print("  BOT :", out["reply"].replace("\n", "\n        ")); print()
# validator must catch an invented number (E4)
L = Ledger(); L.c(179)
assert validate_numbers("Defects were 179 and 77.7%", L) == ["77.7"], "validator should flag the invented number"
print("validator check: PASS (invented number 77.7 is blocked)")
print(f"\n{len(ROWS) - bad}/{len(ROWS)} rows passed")
sys.exit(1 if bad else 0)
