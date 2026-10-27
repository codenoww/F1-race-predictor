#!/usr/bin/env python3
"""Builds public/data/dashboard.json for the F1 podium dashboard.

Runs hourly on GitHub Actions. Only public results data is used (the Ergast-compatible
API at api.jolpi.ca): qualifying times and race results. For the race being predicted,
only its qualifying session is read. Practice, sprint and the race itself are never used.

Model: weighted ridge regression (linear), trained on all completed races of the previous
and current season. Features per driver: qualifying gap (% of pole lap), gap to teammate
(% of pole lap), qualifying position, and average finish of the last 3 races.

A prediction is frozen the first time it is made, so the track record is honest.
Past races without a frozen prediction are re-predicted "as if the race had not happened"
(walk-forward backtest) and labelled as such.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.jolpi.ca/ergast/f1"
RIDGE = 0.5
CURRENT_WEIGHT, PREVIOUS_WEIGHT = 3.0, 1.0
RACE_WINDOW = timedelta(hours=3)  # a race counts as "in progress" for this long after the start
FEATURES = ["gap_pct", "tm_pct", "qpos", "form"]
FEATURE_LABELS = {
    "gap_pct": "Qualifying gap to pole (% of lap)",
    "tm_pct": "Gap to teammate in qualifying (% of lap)",
    "qpos": "Qualifying position",
    "form": "Average finish, last 3 races",
}
MIN_QUALI_ROWS = 15

_session = requests.Session()
_session.headers["User-Agent"] = "f1-podium-dashboard (github.com/codenoww/F1-race-predictor)"


# ----------------------------------------------------------------------------- data access
def api_get(path, tries=6):
    last = "no attempt"
    for attempt in range(tries):
        try:
            r = _session.get(f"{API}/{path}", timeout=30)
            if r.status_code == 200:
                time.sleep(0.3)
                return r.json()["MRData"]
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:
            last = type(e).__name__
        if attempt < tries - 1:
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"{path}: {last} after {tries} tries")


def parse_utc(date, time_):
    t = (time_ or "00:00:00Z").replace("Z", "")
    return datetime.fromisoformat(f"{date}T{t}").replace(tzinfo=timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_schedule(year):
    races = api_get(f"{year}.json?limit=100")["RaceTable"]["Races"]
    out = []
    for r in races:
        q = r.get("Qualifying")
        out.append({
            "key": f"{year}-{int(r['round'])}",
            "year": year,
            "round": int(r["round"]),
            "name": r["raceName"],
            "country": r["Circuit"]["Location"]["country"],
            "locality": r["Circuit"]["Location"]["locality"],
            "circuit": r["Circuit"]["circuitName"],
            "start": parse_utc(r["date"], r.get("time")),
            "quali_start": parse_utc(q["date"], q.get("time")) if q else None,
            "sprint": bool(r.get("Sprint")),
        })
    return sorted(out, key=lambda x: x["round"])


def driver_code(d):
    return d.get("code") or d["driverId"][:3].upper()


def compact_quali(rows):
    return [{
        "d": driver_code(x["Driver"]), "t": x["Constructor"]["constructorId"], "tn": x["Constructor"]["name"],
        "p": int(x["position"]), "q1": x.get("Q1") or "", "q2": x.get("Q2") or "", "q3": x.get("Q3") or "",
    } for x in rows]


def compact_results(rows):
    return [{
        "d": driver_code(x["Driver"]), "t": x["Constructor"]["constructorId"], "tn": x["Constructor"]["name"],
        "p": int(x["position"]),
    } for x in rows]


def fetch_quali(year, rnd):
    races = api_get(f"{year}/{rnd}/qualifying.json?limit=100")["RaceTable"]["Races"]
    return compact_quali(races[0]["QualifyingResults"]) if races else []


def fetch_results(year, rnd):
    races = api_get(f"{year}/{rnd}/results.json?limit=100")["RaceTable"]["Races"]
    return compact_results(races[0]["Results"]) if races else []


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path, obj, indent=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=indent, separators=(",", ":") if indent is None else None)
    Path(path).write_text(text, encoding="utf-8")


def key_tuple(k):
    y, r = k.split("-")
    return int(y), int(r)


# ----------------------------------------------------------------------------- features + model
def secs(s):
    if not s:
        return None
    if ":" in s:
        m, rest = s.split(":")
        return int(m) * 60 + float(rest)
    return float(s)


def usable_quali(q):
    return len(q) >= MIN_QUALI_ROWS and any(secs(x[c]) for x in q for c in ("q1", "q2", "q3"))


def race_features(qrows):
    best = {}
    for x in qrows:
        t = None
        for c in ("q3", "q2", "q1"):
            v = secs(x[c])
            if v:
                t = v
                break
        best[x["d"]] = t
    ref = None
    for c in ("q3", "q2", "q1"):
        vals = [secs(x[c]) for x in qrows if secs(x[c])]
        if vals:
            ref = min(vals)
            break
    if ref is None:
        return []
    by_team = {}
    for x in qrows:
        by_team.setdefault(x["t"], []).append(x["d"])
    rows = []
    for x in qrows:
        t = best[x["d"]]
        gap = (t - ref) if t else None
        mates = [best[m] for m in by_team[x["t"]] if m != x["d"] and best[m]]
        tm = (t - min(mates)) if (t and mates) else 0.0
        rows.append({"d": x["d"], "team": x["tn"], "qpos": x["p"], "gap": gap, "tm": tm, "ref": ref, "best": t})
    worst = max([r["gap"] for r in rows if r["gap"] is not None] or [1.5])
    for r in rows:
        if r["gap"] is None:
            r["gap"] = worst + 0.5
        r["gap_pct"] = 100.0 * r["gap"] / ref
        r["tm_pct"] = 100.0 * r["tm"] / ref
    return rows


def build_table(history, keys):
    """Per-race feature rows in chronological order, with pre-race 'form' (mean of the
    driver's last 3 finishing positions, default 10). Returns (table, form_history)."""
    form_hist, table = {}, {}
    for k in keys:
        rows = race_features(history[k]["q"])
        pos = {x["d"]: x["p"] for x in history[k]["res"]}
        for r in rows:
            h = form_hist.get(r["d"], [])
            r["form"] = float(np.mean(h[-3:])) if h else 10.0
            r["y"] = pos.get(r["d"])
        table[k] = rows
        for r in rows:
            if r["y"] is not None:
                form_hist.setdefault(r["d"], []).append(r["y"])
    return table, form_hist


def fit(rows, weights, lam=RIDGE):
    X = np.array([[r[f] for f in FEATURES] for r in rows], float)
    y = np.array([r["y"] for r in rows], float)
    w = np.array(weights, float)
    mn = X.min(0)
    rg = np.where(X.max(0) - X.min(0) == 0, 1, X.max(0) - X.min(0))
    Z = np.hstack([np.ones((len(X), 1)), (X - mn) / rg])
    sw = np.sqrt(w)[:, None]
    penalty = np.eye(Z.shape[1]) * lam
    penalty[0, 0] = 0  # do not shrink the intercept
    beta = np.linalg.solve((Z * sw).T @ (Z * sw) + penalty, (Z * sw).T @ (y * np.sqrt(w)))
    return beta, mn, rg


def score(model, rows):
    beta, mn, rg = model
    X = np.array([[r[f] for f in FEATURES] for r in rows], float)
    return np.hstack([np.ones((len(X), 1)), (X - mn) / rg]) @ beta


def training_set(table, keys_before, target_year):
    rows, weights, used = [], [], []
    for k in keys_before:
        y = key_tuple(k)[0]
        if y not in (target_year - 1, target_year):
            continue
        wt = CURRENT_WEIGHT if y == target_year else PREVIOUS_WEIGHT
        got = [r for r in table[k] if r["y"] is not None]
        if got:
            rows += got
            weights += [wt] * len(got)
            used.append(k)
    return rows, weights, used


def rank_drivers(model, rows):
    s = score(model, rows)
    order = np.argsort(s, kind="stable")
    return [{"pos": i + 1, "driver": rows[j]["d"], "team": rows[j]["team"], "qpos": rows[j]["qpos"],
             "score": round(float(s[j]), 2)} for i, j in enumerate(order)]


def model_summary(model, used, n_rows):
    beta = model[0]
    return {
        "name": "Weighted ridge regression (linear)",
        "ridge_penalty": RIDGE,
        "training_races": len(used),
        "training_rows": int(n_rows),
        "intercept": round(float(beta[0]), 3),
        "weights": {FEATURE_LABELS[f]: round(float(b), 3) for f, b in zip(FEATURES, beta[1:])},
    }


def metrics(order_codes, actual):
    act = sorted(actual, key=actual.get)
    od = [d for d in order_codes if d in actual]
    if not od or not act:
        return None
    return {"winner": od[0] == act[0], "podium": len(set(od[:3]) & set(act[:3])),
            "exact": od[:3] == act[:3], "pred": od[:3], "actual": act[:3]}


# ----------------------------------------------------------------------------- main build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", help="pretend 'now' is this UTC time (ISO, for testing)")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "public" / "data"))
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    if args.as_of:
        now = datetime.fromisoformat(args.as_of.replace("Z", "+00:00")).astimezone(timezone.utc)
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    print(f"now (UTC): {iso(now)}")

    # which season is "current"? first one with a race that has not finished yet
    year = now.year
    schedule = get_schedule(year)
    upcoming = [r for r in schedule if r["start"] + RACE_WINDOW > now]
    if not upcoming:
        try:
            nxt = get_schedule(year + 1)
        except RuntimeError:
            nxt = []
        if nxt:
            year, schedule, upcoming = year + 1, nxt, [r for r in nxt if r["start"] + RACE_WINDOW > now]
    target = upcoming[0] if upcoming else None
    schedules = {year: schedule}
    try:
        schedules[year - 1] = get_schedule(year - 1)
    except RuntimeError as e:
        print("warning: previous season schedule unavailable:", e)

    # ---- history: only fetch races we do not have yet
    history_path = data_dir / "history.json"
    history = load_json(history_path, {})
    starts = {r["key"]: r["start"] for s in schedules.values() for r in s}
    fetched = 0
    for k, st in starts.items():
        if k in history:
            history[k]["start"] = iso(st)
    for s in schedules.values():
        for r in s:
            if r["start"] + RACE_WINDOW > now or r["key"] in history:
                continue
            try:
                q, res = fetch_quali(r["year"], r["round"]), fetch_results(r["year"], r["round"])
            except RuntimeError as e:
                print(f"  could not fetch {r['key']} yet: {e}")
                continue
            if q and res:
                history[r["key"]] = {"name": r["name"], "start": iso(r["start"]), "q": q, "res": res}
                fetched += 1
                print(f"  added {r['key']} {r['name']}")
    history = {k: v for k, v in history.items() if key_tuple(k)[0] >= year - 1}
    print(f"history: {len(history)} races ({fetched} newly fetched)")

    # races that count as completed *as of now*, with usable qualifying times
    visible = sorted(
        (k for k, v in history.items()
         if datetime.fromisoformat(v["start"].replace("Z", "+00:00")) + RACE_WINDOW <= now
         and usable_quali(v["q"]) and v["res"]),
        key=key_tuple)
    table, form_hist = build_table(history, visible)

    # ---- frozen predictions
    pred_path = data_dir / "predictions.json"
    predictions = load_json(pred_path, {})

    status, prediction, qualifying = "season_complete", None, []
    if target:
        k = target["key"]
        if now >= target["start"]:
            status = "in_progress"
        elif target["quali_start"] and now >= target["quali_start"]:
            status = "waiting_for_qualifying"
        else:
            status = "waiting_for_qualifying"
        if k not in predictions and (target["quali_start"] is None or now >= target["quali_start"]):
            try:
                q = fetch_quali(target["year"], target["round"])
            except RuntimeError as e:
                q = []
                print("could not fetch target qualifying:", e)
            if usable_quali(q):
                rows = race_features(q)
                for r in rows:
                    h = form_hist.get(r["d"], [])
                    r["form"] = float(np.mean(h[-3:])) if h else 10.0
                tr, w, used = training_set(table, visible, target["year"])
                if tr:
                    model = fit(tr, w)
                    quali_sorted = sorted(rows, key=lambda r: r["qpos"])
                    predictions[k] = {
                        "generated_at": iso(now),
                        "live": bool(now < target["start"]),
                        "order": rank_drivers(model, rows),
                        "qualifying": [{"pos": r["qpos"], "driver": r["d"], "team": r["team"],
                                        "gap_pct": round(r["gap_pct"], 3)} for r in quali_sorted[:10]],
                        "model": model_summary(model, used, len(tr)),
                    }
                    print(f"  froze prediction for {k}: {[o['driver'] for o in predictions[k]['order'][:3]]}")
        if k in predictions:
            prediction, qualifying = predictions[k], predictions[k]["qualifying"]
            status = "in_progress" if now >= target["start"] else "predicted"

    # ---- track record for the current season (live if frozen, else walk-forward backtest)
    rows_out = []
    for k in [x for x in visible if key_tuple(x)[0] == year]:
        idx = visible.index(k)
        actual = {r["d"]: r["y"] for r in table[k] if r["y"] is not None}
        base = [r["d"] for r in sorted(table[k], key=lambda r: r["qpos"])]
        frozen = predictions.get(k)
        if frozen and frozen.get("live"):
            order, kind = [o["driver"] for o in frozen["order"]], "live"
        else:
            tr, w, _ = training_set(table, visible[:idx], year)
            if not tr:
                continue
            order = [o["driver"] for o in rank_drivers(fit(tr, w), table[k])]
            kind = "backtest"
        m, b = metrics(order, actual), metrics(base, actual)
        if not m or not b:
            continue
        rows_out.append({"key": k, "round": key_tuple(k)[1], "name": history[k]["name"], "type": kind,
                         "pred_top3": m["pred"], "actual_top3": m["actual"], "base_top3": b["pred"],
                         "winner_hit": m["winner"], "podium": m["podium"], "exact": m["exact"],
                         "base_winner_hit": b["winner"], "base_podium": b["podium"]})

    def summarize(rs):
        n = len(rs)
        if not n:
            return None
        return {"races": n,
                "model_winners": sum(r["winner_hit"] for r in rs), "base_winners": sum(r["base_winner_hit"] for r in rs),
                "model_podium_avg": round(sum(r["podium"] for r in rs) / n, 2),
                "base_podium_avg": round(sum(r["base_podium"] for r in rs) / n, 2)}

    track = {"rows": sorted(rows_out, key=lambda r: -r["round"]),
             "summary_all": summarize(rows_out),
             "summary_live": summarize([r for r in rows_out if r["type"] == "live"])}

    # ---- standings (current season)
    standings = {"drivers": [], "constructors": []}
    try:
        d = api_get(f"{year}/driverStandings.json")["StandingsTable"]["StandingsLists"]
        if d:
            standings["drivers"] = [{"pos": int(x["position"]), "driver": driver_code(x["Driver"]),
                                     "team": x["Constructors"][0]["name"] if x["Constructors"] else "",
                                     "points": float(x["points"]), "wins": int(x["wins"])}
                                    for x in d[0]["DriverStandings"]]
            standings["after_round"] = int(d[0]["round"])
        c = api_get(f"{year}/constructorStandings.json")["StandingsTable"]["StandingsLists"]
        if c:
            standings["constructors"] = [{"pos": int(x["position"]), "team": x["Constructor"]["name"],
                                          "points": float(x["points"]), "wins": int(x["wins"])}
                                         for x in c[0]["ConstructorStandings"]]
    except RuntimeError as e:
        print("warning: standings unavailable:", e)
        old = load_json(out_dir / "dashboard.json", {}).get("standings")
        if old:
            standings = old

    # ---- model description (trained on everything completed so far)
    tr, w, used = training_set(table, visible, year)
    model_info = model_summary(fit(tr, w), used, len(tr)) if tr else None

    next_race = None
    if target:
        next_race = {"round": target["round"], "name": target["name"], "country": target["country"],
                     "circuit": target["circuit"], "locality": target["locality"],
                     "race_start": iso(target["start"]),
                     "quali_start": iso(target["quali_start"]) if target["quali_start"] else None,
                     "sprint_weekend": target["sprint"], "status": status}
    dashboard = {
        "season": year,
        "next_race": next_race,
        "prediction": ({"generated_at": prediction["generated_at"], "live": prediction["live"],
                        "order": prediction["order"]} if prediction else None),
        "qualifying": qualifying,
        "track_record": track,
        "standings": standings,
        "model": model_info,
        "sources": "Qualifying times, race results and standings: api.jolpi.ca (Ergast-compatible)",
    }

    # ---- write (only when something changed, so the repo is not spammed with commits)
    out_path = out_dir / "dashboard.json"
    old = load_json(out_path, None)

    def strip(d):
        return {k: v for k, v in (d or {}).items() if k != "generated_at"}

    changed = strip(old) != strip(dashboard)
    dashboard["generated_at"] = iso(now) if changed or not old else old.get("generated_at", iso(now))
    if changed or not old:
        save_json(out_path, dashboard)
    save_json(history_path, history)
    save_json(pred_path, predictions, indent=1)
    print(f"status: {status} | dashboard.json {'updated' if changed else 'unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
