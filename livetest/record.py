#!/usr/bin/env python3
"""Records how OpenF1 behaves around a real F1 session, to answer: can a dashboard show live data?

Modes
  probe   one reachability pass against the most recent finished session (tests the network path)
  watch   poll a session while it runs and for a while after: is data available DURING the session,
          how far behind real time is it, does it need credentials, and when do the final
          results appear on OpenF1 versus the public results database (jolpi)?
  lookup  single check of whether results are published yet (run the day after a session)

Everything is written to livetest/results/ as small JSON files. Nothing here uses credentials; if
OpenF1 requires them for live data, the recorder just logs the refusal.
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OF1 = "https://api.openf1.org/v1"
JOLPI = "https://api.jolpi.ca/ergast/f1"
UA = {"User-Agent": "f1-podium-dashboard live-test (github.com/codenoww/F1-race-predictor)"}

# endpoint -> name of its timestamp field
LIVE = {"position": "date", "laps": "date_start", "pit": "date", "race_control": "date", "weather": "date"}
RACE_EXTRA = {"intervals": "date"}
RESULT_POLL = {"Qualifying": ["session_result", "starting_grid"], "Race": ["session_result"]}
TAIL_MIN = {"Practice": 20, "Qualifying": 180, "Race": 120}


def utcnow():
    return datetime.now(timezone.utc)


def parse(s):
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def iso(d):
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def naive(d):
    """Timestamp for OpenF1 filters: no offset, because '+' would be read as a space in a URL."""
    return d.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="microseconds")


def call(base, path, query=""):
    url = f"{base}{path}" + (f"?{query}" if query else "")
    t = time.time()
    try:
        r = requests.get(url, headers=UA, timeout=45)
        ms = round((time.time() - t) * 1000)
        try:
            body = r.json()
        except ValueError:
            body = None
        return {"status": r.status_code, "ms": ms, "body": body, "text": None if body is not None else r.text[:160]}
    except requests.RequestException as e:
        return {"status": None, "ms": round((time.time() - t) * 1000), "body": None, "text": type(e).__name__}


def egress():
    r = call("https://ipinfo.io", "/json")
    b = r["body"] if isinstance(r["body"], dict) else {}
    return {"org": b.get("org"), "country": b.get("country")}


def sessions(year, country=None):
    q = f"year={year}" + (f"&country_name={country}" if country else "")
    r = call(OF1, "/sessions", q)
    return r["body"] if isinstance(r["body"], list) else []


def refusal(r):
    """Short note when the server answered but not with data."""
    if r["status"] in (None, 200):
        return None
    b = r["body"] if isinstance(r["body"], dict) else {}
    if r["status"] == 404 and "No results found" in str(b.get("detail", "")):
        return None  # just "no rows yet", not a refusal
    if r["status"] is None:
        return f"request failed: {r['text']}"
    return f"HTTP {r['status']}: {str(b.get('detail') or r['text'] or '')[:120]}"


def newest_ts(rows, field):
    vals = [parse(x[field]) for x in rows if isinstance(x, dict) and x.get(field)]
    return max(vals) if vals else None


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------------------- probe
def probe():
    now = utcnow()
    done = [s for s in sessions(now.year) if s.get("date_end") and parse(s["date_end"]) < now]
    done.sort(key=lambda s: parse(s["date_end"]))
    pick = next((s for s in reversed(done) if s["session_name"] == "Race"), done[-1] if done else None)
    out = {"mode": "probe", "at": iso(now), "egress": egress(), "sessions_listed": len(done), "endpoints": {}}
    if not pick:
        out["error"] = "no finished session found (sessions endpoint unreachable?)"
        write(f"probe-{now:%Y%m%dT%H%M%S}.json", out)
        print(out)
        return 1
    sk = pick["session_key"]
    out["session"] = {"key": sk, "name": pick["session_name"], "country": pick.get("country_name"), "end": pick["date_end"]}
    for ep, field in {**LIVE, **RACE_EXTRA, "stints": None, "session_result": None}.items():
        r = call(OF1, f"/{ep}", f"session_key={sk}")
        rows = r["body"] if isinstance(r["body"], list) else []
        entry = {"status": r["status"], "ms": r["ms"], "rows": len(rows), "note": refusal(r)}
        if rows and field:
            entry["newest"] = iso(newest_ts(rows, field))
        out["endpoints"][ep] = entry
        time.sleep(0.4)
    if out["endpoints"]["position"]["rows"]:
        mid = parse(call(OF1, "/position", f"session_key={sk}")["body"][100]["date"])
        r = call(OF1, "/position", f"session_key={sk}&date>={naive(mid)}")
        out["date_filter_test"] = {"status": r["status"], "rows": len(r["body"]) if isinstance(r["body"], list) else None}
    cors = requests.get(f"{OF1}/sessions?year={now.year}&session_name=Race", headers={**UA, "Origin": "https://f1-race-predictor-vercel.vercel.app"}, timeout=30)
    out["cors_allow_origin"] = cors.headers.get("access-control-allow-origin")
    write(f"probe-{now:%Y%m%dT%H%M%S}.json", out)
    ok = [e for e in out["endpoints"].values() if e["status"] == 200]
    print(json.dumps(out, indent=1)[:1500])
    print(f"probe: {len(ok)}/{len(out['endpoints'])} endpoints answered 200")
    return 0


# --------------------------------------------------------------------------------------- results lookup
def jolpi_round(date):
    r = call(JOLPI, f"/{date.year}.json", "limit=100")
    races = r["body"]["MRData"]["RaceTable"]["Races"] if isinstance(r["body"], dict) else []
    for x in races:
        d = parse(f"{x['date']}T00:00:00Z")
        if abs((d - date).total_seconds()) <= 3 * 86400:
            return int(x["round"])
    return None


def results_state(session):
    """Is the final result of this session published yet, on OpenF1 and on the public database?"""
    sk, name = session["session_key"], session["session_name"]
    state = {}
    for ep in RESULT_POLL.get(name, []):
        r = call(OF1, f"/{ep}", f"session_key={sk}")
        state[f"openf1_{ep}"] = isinstance(r["body"], list) and len(r["body"]) > 0
    rnd = jolpi_round(parse(session["date_start"]))
    if rnd and name in ("Qualifying", "Race"):
        ep = "qualifying" if name == "Qualifying" else "results"
        r = call(JOLPI, f"/{parse(session['date_start']).year}/{rnd}/{ep}.json", "limit=100")
        races = r["body"]["MRData"]["RaceTable"]["Races"] if isinstance(r["body"], dict) else []
        state["jolpi_results"] = bool(races)
    return state


def lookup(args):
    now = utcnow()
    cand = [s for s in sessions(now.year, args.country) if s["session_name"] == args.session]
    if not cand:
        print("session not found")
        return 1
    s = cand[0]
    out = {"mode": "lookup", "at": iso(now), "session": s["session_name"], "ended": s["date_end"],
           "hours_after_end": round((now - parse(s["date_end"])).total_seconds() / 3600, 1), **results_state(s)}
    write(f"lookup-{args.session}-{now:%Y%m%dT%H%M%S}.json", out)
    print(json.dumps(out, indent=1))
    return 0


# --------------------------------------------------------------------------------------- watch
def pick_session(args):
    now = utcnow()
    ss = sessions(args.year, args.country)
    if args.session_key:
        return next((s for s in ss if s["session_key"] == args.session_key), None)
    for s in sorted(ss, key=lambda s: parse(s["date_start"])):
        kind = "Practice" if s["session_name"].startswith("Practice") else s["session_name"]
        if kind not in TAIL_MIN:
            continue
        if parse(s["date_start"]) - timedelta(minutes=args.lead_minutes) <= now <= parse(s["date_end"]) + timedelta(minutes=TAIL_MIN[kind]):
            return s
    return None


def watch(args):
    s = pick_session(args)
    if not s:
        print("no session of interest is running or about to start - nothing to record")
        return 0
    sk, name = s["session_key"], s["session_name"]
    kind = "Practice" if name.startswith("Practice") else name
    start, end = parse(s["date_start"]), parse(s["date_end"])
    deadline = end + timedelta(minutes=args.tail_minutes if args.tail_minutes is not None else TAIL_MIN[kind])
    stop = utcnow() + timedelta(minutes=args.max_minutes) if args.max_minutes else None
    if stop:  # replaying an already-finished session for testing: its natural deadline is in the past
        deadline = max(deadline, stop)
    eps = dict(LIVE)
    if kind == "Race":
        eps.update(RACE_EXTRA)
    print(f"watching {name} (key {sk}) {iso(start)} -> {iso(end)}, until {iso(deadline)}; endpoints: {list(eps)}")

    stats = {ep: {"polls": 0, "statuses": Counter(), "polls_with_new_rows": 0, "rows_new_live": 0, "rows_new_after": 0,
                  "first_data_wall": None, "lags_live_s": [], "ms": [], "notes": Counter()} for ep in eps}
    last_seen, events = {}, []
    first_seen = {}
    jolpi_name = {"Qualifying": "jolpi_results", "Race": "jolpi_results"}.get(kind)
    started = utcnow()
    last_save = time.time()

    def summary(final):
        res = {k: {"seen_at": iso(v), "after_end_s": round((v - end).total_seconds())} for k, v in first_seen.items()}
        live_ok = any(st["lags_live_s"] and statistics.median(st["lags_live_s"]) < 90 for st in stats.values())
        any_live_rows = any(st["rows_new_live"] > 0 for st in stats.values())
        blocked = {ep: dict(st["notes"]) for ep, st in stats.items() if st["notes"]}
        return {
            "mode": "watch", "final": final, "session": {"key": sk, "name": name, "start": iso(start), "end": iso(end)},
            "watched_from": iso(started), "watched_until": iso(utcnow()), "egress": egress_info,
            "verdict": {
                "data_arrived_during_session": any_live_rows,
                "near_real_time (median lag < 90 s)": live_ok,
                "refusals_or_errors": blocked or None,
            },
            "endpoints": {ep: {"polls": st["polls"], "statuses": dict(st["statuses"]),
                               "polls_with_new_rows": st["polls_with_new_rows"],
                               "rows_new_while_live": st["rows_new_live"], "rows_new_after_end": st["rows_new_after"],
                               "first_data_wall": st["first_data_wall"],
                               "median_lag_live_s": round(statistics.median(st["lags_live_s"]), 1) if st["lags_live_s"] else None,
                               "max_lag_live_s": round(max(st["lags_live_s"]), 1) if st["lags_live_s"] else None,
                               "median_ms": round(statistics.median(st["ms"])) if st["ms"] else None}
                          for ep, st in stats.items()},
            "results_first_seen": res,
            "events_logged": len(events),
        }

    egress_info = egress()
    while utcnow() < deadline and not (stop and utcnow() > stop):
        now = utcnow()
        phase = "pre" if now < start else "live" if now <= end else "post"
        for ep, field in eps.items():
            # only rows newer than the last one seen ('>' not '>=', or the last row would repeat)
            q = f"session_key={sk}" + (f"&{field}>{naive(last_seen[ep])}" if ep in last_seen else "")
            r = call(OF1, f"/{ep}", q)
            st = stats[ep]
            st["polls"] += 1
            st["statuses"][str(r["status"])] += 1
            st["ms"].append(r["ms"])
            note = refusal(r)
            if note:
                st["notes"][note] += 1
            rows = r["body"] if isinstance(r["body"], list) else []
            new_ts = newest_ts(rows, field)
            if rows and new_ts:
                last_seen[ep] = new_ts
                st["polls_with_new_rows"] += 1
                st["first_data_wall"] = st["first_data_wall"] or iso(now)
                if phase == "live":
                    st["rows_new_live"] += len(rows)
                    st["lags_live_s"].append((now - new_ts).total_seconds())
                elif phase == "post":
                    st["rows_new_after"] += len(rows)
                if len(events) < 400:
                    events.append({"t": iso(now), "ep": ep, "phase": phase, "rows": len(rows), "newest": iso(new_ts),
                                   "lag_s": round((now - new_ts).total_seconds(), 1), "ms": r["ms"]})
            elif note and len(events) < 400:
                events.append({"t": iso(now), "ep": ep, "phase": phase, "note": note})
            time.sleep(0.5)
        if phase == "post":
            for k, v in results_state(s).items():
                if v and k not in first_seen:
                    first_seen[k] = now
                    print(f"{iso(now)} first seen: {k} ({round((now - end).total_seconds() / 60)} min after end)")
            expected = {f"openf1_{e}" for e in RESULT_POLL.get(kind, [])} | ({jolpi_name} if jolpi_name else set())
            if expected and expected <= set(first_seen):
                print("all result sources have published - done")
                break
        if time.time() - last_save > 600:
            write(f"watch-{sk}-{name.replace(' ', '')}.json", {**summary(False), "events": events})
            last_save = time.time()
        if phase == "pre":  # the job may start long before the session: poll gently, wake just before it begins
            nap = min(args.pre_interval, max(5, (start - utcnow()).total_seconds() - 5))
        else:
            nap = args.live_interval if phase == "live" else args.post_interval
        limit = min(deadline, stop) if stop else deadline
        time.sleep(max(1, min(nap, (limit - utcnow()).total_seconds())))

    final = {**summary(True), "events": events}
    write(f"watch-{sk}-{name.replace(' ', '')}.json", final)
    print(json.dumps(final["verdict"], indent=1))
    print("results first seen:", json.dumps(final["results_first_seen"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "watch", "lookup"], required=True)
    ap.add_argument("--year", type=int, default=utcnow().year)
    ap.add_argument("--country", default="Mexico")
    ap.add_argument("--session", default="Qualifying", help="lookup mode: Qualifying or Race")
    ap.add_argument("--session-key", type=int, help="watch this session regardless of the clock (testing)")
    ap.add_argument("--lead-minutes", type=int, default=20)
    ap.add_argument("--tail-minutes", type=int)
    ap.add_argument("--max-minutes", type=float, help="hard stop (testing)")
    ap.add_argument("--live-interval", type=int, default=30)
    ap.add_argument("--pre-interval", type=int, default=180)
    ap.add_argument("--post-interval", type=int, default=120)
    args = ap.parse_args()
    return {"probe": lambda: probe(), "watch": lambda: watch(args), "lookup": lambda: lookup(args)}[args.mode]()


if __name__ == "__main__":
    sys.exit(main())
