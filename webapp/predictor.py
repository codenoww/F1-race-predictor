"""
F1 podium predictor — core engine.
Generalized: automatically finds whatever the *next* race on the calendar is
(any season, any circuit) instead of being hardcoded to one GP.

Pipeline: weighted least-squares regression (with a small ridge term for
stability) trained on the full previous season + all completed races of the
current season, applied to the upcoming race's qualifying + practice data.
"""
import json
import logging
import os
import time
import warnings
from collections import defaultdict, deque
from datetime import date

import fastf1
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "..", "f1_cache")  # reuse cache already warmed from earlier runs
PRED_CACHE_FILE = os.path.join(BASE_DIR, "prediction_cache.json")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

for lg in ["fastf1", "fastf1.core", "fastf1.req", "fastf1._api",
           "fastf1.ergast", "urllib3", "requests"]:
    l = logging.getLogger(lg)
    l.setLevel(logging.CRITICAL)
    l.propagate = False

FEATURE_NAMES = [
    "Quali Gap (s)",
    "Tire Factor",
    "Env. Index",
    "Mean/Practice Lap (s)",
    "Pitstop (s)",
    "Recent Form (avg pos)",
    "Teammate Quali Delta (s)",
    "Grid Penalty (positions)",
    "Overtake Difficulty",
]

COMPOUND_MAP = {"SOFT": 1.0, "MEDIUM": 2.0, "HARD": 3.0, "INTERMEDIATE": 2.5, "WET": 3.5}

HIST_PIT = {
    "Red Bull Racing": 23.1, "Mercedes": 23.8, "Ferrari": 23.5, "McLaren": 23.4,
    "Aston Martin": 24.6, "Williams": 25.1, "Alpine": 24.9, "Haas F1 Team": 25.3,
    "RB": 24.8, "Racing Bulls": 24.8, "Kick Sauber": 25.0, "Audi": 25.5, "Cadillac": 26.0,
}

# Static per-circuit overtaking-difficulty index: 0 = easy to pass, 1 = track position is king.
# Matched against the event's Country/Location/EventName (case-insensitive substring).
OVERTAKE_DIFFICULTY = {
    "monaco": 0.95, "hungar": 0.80, "singapore": 0.85, "netherlands": 0.75,
    "zandvoort": 0.75, "spain": 0.65, "barcelona": 0.65, "italy": 0.25, "monza": 0.25,
    "belgium": 0.35, "spa": 0.35, "austria": 0.45, "britain": 0.50, "silverstone": 0.50,
    "australia": 0.55, "china": 0.50, "japan": 0.60, "canada": 0.55, "miami": 0.55,
    "bahrain": 0.40, "saudi": 0.55, "qatar": 0.60, "united states": 0.50, "austin": 0.50,
    "mexico": 0.60, "brazil": 0.45, "são paulo": 0.45, "las vegas": 0.45,
    "abu dhabi": 0.60, "azerbaijan": 0.40, "baku": 0.40,
}
DEFAULT_OVERTAKE_DIFFICULTY = 0.55

RIDGE_LAMBDA = 0.5  # small Tikhonov term — stabilises (AtWA)^-1 now that we have 9, possibly-correlated, features
RECENT_FORM_LOOKBACK = 3

# Training-window size. Each race costs two FastF1 session loads (Race + Quali) plus
# per-driver lap parsing, so this is the main lever on runtime: previously the whole
# previous season (~24 rounds) + all completed current-season rounds were loaded
# (~35 races, ~10 min). Capping both windows trades a bit of training-set size for a
# much faster run — 9 features only needs a few hundred rows to fit well.
PREV_SEASON_LOOKBACK_ROUNDS = 8
CURRENT_SEASON_LOOKBACK_ROUNDS = 8

def safe_f(v, default):
    try:
        r = float(v)
        return default if r != r else r
    except Exception:
        return default

def _with_retry(fn, *args, retries=4, base_delay=15, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as ex:
            is_rate_limit = "RateLimitExceeded" in type(ex).__name__ or "500 calls" in str(ex)
            if not is_rate_limit or attempt == retries - 1:
                raise
            wait = base_delay * (attempt + 1)
            print(f"  rate limited, retrying in {wait}s ({attempt + 1}/{retries})")
            time.sleep(wait)

def overtake_difficulty_for(event_name, country):
    text = f"{event_name} {country}".lower()
    for key, val in OVERTAKE_DIFFICULTY.items():
        if key in text:
            return val
    return DEFAULT_OVERTAKE_DIFFICULTY

def _teammate_deltas(q_res, drivers, teams):
    """Signed quali gap of each driver vs. their teammate's best quali time.
    Positive = slower than teammate, negative = faster."""
    team_to_drivers = defaultdict(list)
    for d, t in zip(drivers, teams):
        team_to_drivers[t].append(d)

    def best_time(d):
        for qc in ["Q3", "Q2", "Q1"]:
            if qc in q_res.columns and d in q_res.index:
                raw = q_res.loc[d, qc]
                if pd.notna(raw):
                    return safe_f(raw.total_seconds() if hasattr(raw, "total_seconds") else raw, None)
        return None

    out = []
    for d, t in zip(drivers, teams):
        mates = [m for m in team_to_drivers[t] if m != d]
        my_t = best_time(d)
        mate_times = [x for x in (best_time(m) for m in mates) if x is not None]
        if my_t is None or not mate_times:
            out.append(0.0)
        else:
            out.append(round(my_t - min(mate_times), 4))
    return out

def _grid_penalty_from_results(race_results, quali_results, drivers):
    """positive = lost grid spots to a penalty, 0 = none / unknown. Reuses session
    objects already loaded by the caller instead of re-fetching them."""
    try:
        grid = race_results.set_index("Abbreviation")["GridPosition"]
        qpos = quali_results.set_index("Abbreviation")["Position"]
        out = []
        for d in drivers:
            if d in grid.index and d in qpos.index:
                out.append(round(safe_f(grid.loc[d], 0) - safe_f(qpos.loc[d], 0), 1))
            else:
                out.append(0.0)
        return out
    except Exception:
        return [0.0] * len(drivers)

def _grid_penalty_for_future_round(season, rnd, quali_results, drivers):
    """For a race that hasn't run yet: try to read the finalised starting grid if
    it's already been published (e.g. race weekend under way); default to 0 (unknown)."""
    try:
        r = fastf1.get_session(season, rnd, "R")
        r.load(telemetry=False, laps=False, weather=False, messages=False)
        return _grid_penalty_from_results(r.results, quali_results, drivers)
    except Exception:
        return [0.0] * len(drivers)

def _recent_form(history, drivers):
    out = []
    for d in drivers:
        h = history.get(d)
        out.append(round(float(np.mean(h)), 3) if h else 6.0)  # 6.0 = neutral mid-field prior
    return out

def _update_history(history, drivers, positions):
    for d, p in zip(drivers, positions):
        history[d].append(p)

def _load_real_round(season, rnd, top_n, history, event_meta):
    sess = fastf1.get_session(season, rnd, "R")
    sess.load(telemetry=False, weather=True, messages=False)
    quali = fastf1.get_session(season, rnd, "Q")
    quali.load(telemetry=False)

    res = sess.results.sort_values("Position").dropna(subset=["Position"]).head(top_n)
    drivers = res["Abbreviation"].tolist()
    teams = res["TeamName"].tolist()
    positions = res["Position"].astype(float).tolist()

    q_res = quali.results.set_index("Abbreviation")
    best_t = None
    for qc in ["Q3", "Q2", "Q1"]:
        if qc in q_res.columns:
            ts = q_res[qc].dropna()
            if not ts.empty:
                best_t = ts.dt.total_seconds().min(); break
    best_t = best_t or 90.0
    q_gaps = []
    for d in drivers:
        t = None
        for qc in ["Q3", "Q2", "Q1"]:
            if qc in q_res.columns and d in q_res.index:
                raw = q_res.loc[d, qc]
                if pd.notna(raw):
                    t = safe_f(raw.total_seconds() if hasattr(raw, "total_seconds") else raw, None)
                    if t: break
        q_gaps.append(round((t - best_t) if t else 1.5, 4))

    tire_f = []
    for d in drivers:
        try:
            laps = sess.laps.pick_drivers(d)
            if laps.empty: tire_f.append(2.0); continue
            last = laps.iloc[-1]
            cmp = COMPOUND_MAP.get(str(last["Compound"]).upper(), 2.0)
            age = safe_f(last["TyreLife"], 20)
            tire_f.append(round(cmp * (1 + age / 60), 3))
        except Exception:
            tire_f.append(2.0)

    try:
        wx = sess.weather_data
        tt = safe_f(wx["TrackTemp"].mean(), 38.0) if not wx.empty else 38.0
        rain = 1.0 if (not wx.empty and bool(wx["Rainfall"].any())) else 0.0
        env_v = round(tt / 60 + rain, 3)
    except Exception:
        env_v = 0.65
    env = [env_v] * len(drivers)

    mean_l = []
    for d in drivers:
        try:
            laps = sess.laps.pick_drivers(d).pick_quicklaps()
            if laps.empty: mean_l.append(93.0); continue
            lt = laps["LapTime"].dt.total_seconds().dropna()
            mean_l.append(round(float(lt.mean()), 3) if not lt.empty else 93.0)
        except Exception:
            mean_l.append(93.0)

    pit_t = []
    for d, team in zip(drivers, teams):
        try:
            laps = sess.laps.pick_drivers(d)
            pi = laps["PitInTime"].dropna(); po = laps["PitOutTime"].dropna()
            if pi.empty or po.empty: pit_t.append(HIST_PIT.get(team, 25.0)); continue
            n = min(len(pi), len(po))
            dur = [(po.iloc[i] - pi.iloc[i]).total_seconds() for i in range(n)]
            dur = [x for x in dur if 15 < x < 60]
            pit_t.append(round(float(np.mean(dur)), 2) if dur else HIST_PIT.get(team, 25.0))
        except Exception:
            pit_t.append(HIST_PIT.get(team, 25.0))

    recent_form = _recent_form(history, drivers)
    teammate_delta = _teammate_deltas(q_res, drivers, teams)
    grid_penalty = _grid_penalty_from_results(sess.results, quali.results, drivers)
    ev = fastf1.get_event(season, rnd)
    overtake = [overtake_difficulty_for(ev.get("EventName", ""), ev.get("Country", ""))] * len(drivers)

    feats = np.array([q_gaps, tire_f, env, mean_l, pit_t, recent_form,
                       teammate_delta, grid_penalty, overtake], dtype=float).T
    for col in range(feats.shape[1]):
        m = np.isnan(feats[:, col])
        if m.any():
            med = float(np.nanmedian(feats[:, col])) if not np.all(m) else 0.5
            feats[m, col] = med

    _update_history(history, drivers, positions)
    return drivers, teams, feats, np.array(positions)

def _load_future_quali(season, rnd, top_n, history):
    quali = fastf1.get_session(season, rnd, "Q")
    quali.load(telemetry=False)

    practice = None
    practice_label = None
    for stype in ["FP2", "Practice 1", "Sprint"]:
        try:
            cand = fastf1.get_session(season, rnd, stype)
            cand.load(telemetry=False, weather=True)
            practice = cand
            practice_label = stype
            break
        except Exception:
            continue
    if practice is None:
        raise RuntimeError("no practice/sprint session available for this event")

    res = quali.results.sort_values("Q3").head(top_n)
    if res.empty or res["Q3"].isna().all():
        res = quali.results.head(top_n)
    drivers = res["Abbreviation"].tolist()
    teams = res["TeamName"].tolist()

    q_res = quali.results.set_index("Abbreviation")
    best_t = None
    for qc in ["Q3", "Q2", "Q1"]:
        if qc in q_res.columns:
            ts = q_res[qc].dropna()
            if not ts.empty: best_t = ts.dt.total_seconds().min(); break
    best_t = best_t or 90.0
    q_gaps = []
    for d in drivers:
        t = None
        for qc in ["Q3", "Q2", "Q1"]:
            if qc in q_res.columns and d in q_res.index:
                raw = q_res.loc[d, qc]
                if pd.notna(raw):
                    t = safe_f(raw.total_seconds() if hasattr(raw, "total_seconds") else raw, None)
                    if t: break
        q_gaps.append(round((t - best_t) if t else 1.5, 4))

    tire_f = []
    for d in drivers:
        try:
            laps = practice.laps.pick_drivers(d)
            if laps.empty: tire_f.append(2.0); continue
            last = laps.iloc[-1]
            cmp = COMPOUND_MAP.get(str(last["Compound"]).upper(), 2.0)
            age = safe_f(last["TyreLife"], 15)
            tire_f.append(round(cmp * (1 + age / 60), 3))
        except Exception:
            tire_f.append(2.0)

    try:
        wx = practice.weather_data
        tt = safe_f(wx["TrackTemp"].mean(), 38.0) if not wx.empty else 38.0
        rain = 1.0 if (not wx.empty and bool(wx["Rainfall"].any())) else 0.0
        env_v = round(tt / 60 + rain, 3)
    except Exception:
        env_v = 0.65
    env = [env_v] * len(drivers)

    pace = []
    for d in drivers:
        try:
            laps = practice.laps.pick_drivers(d).pick_quicklaps()
            if laps.empty: pace.append(93.0); continue
            lt = laps["LapTime"].dt.total_seconds().dropna()
            pace.append(round(float(lt.median()), 3) if not lt.empty else 93.0)
        except Exception:
            pace.append(93.0)

    pit_t = [HIST_PIT.get(t, 25.0) for t in teams]
    recent_form = _recent_form(history, drivers)
    teammate_delta = _teammate_deltas(q_res, drivers, teams)
    grid_penalty = _grid_penalty_for_future_round(season, rnd, quali.results, drivers)  # usually [0,0,...] pre-race
    ev = fastf1.get_event(season, rnd)
    overtake = [overtake_difficulty_for(ev.get("EventName", ""), ev.get("Country", ""))] * len(drivers)

    feats = np.array([q_gaps, tire_f, env, pace, pit_t, recent_form,
                       teammate_delta, grid_penalty, overtake], dtype=float).T
    for col in range(feats.shape[1]):
        m = np.isnan(feats[:, col])
        if m.any():
            med = float(np.nanmedian(feats[:, col])) if not np.all(m) else 0.5
            feats[m, col] = med
    return drivers, teams, feats, practice_label

def normalise(M):
    mn, mx = M.min(0), M.max(0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return (M - mn) / rng, mn, rng

def norm_apply(M, mn, rng):
    return (M - mn) / rng

def weighted_ridge_least_squares(A, b, W_vec, lam=RIDGE_LAMBDA):
    sqrt_w = np.sqrt(W_vec)
    A_w = A * sqrt_w[:, np.newaxis]
    b_w = b * sqrt_w
    n_feat = A.shape[1]
    lhs = A_w.T @ A_w + lam * np.eye(n_feat)
    rhs = A_w.T @ b_w
    x_hat = np.linalg.solve(lhs, rhs)
    b_hat = A @ x_hat
    resid = b - b_hat
    rmse = float(np.sqrt(np.average(resid ** 2, weights=W_vec)))
    return x_hat, b_hat, resid, rmse

def get_next_race():
    """Find the next race on the calendar relative to today, any season."""
    today = date.today()
    for year in (today.year, today.year + 1):
        sched = fastf1.get_event_schedule(year, include_testing=False)
        sched = sched.sort_values("RoundNumber")
        upcoming = sched[sched["EventDate"].dt.date >= today]
        if not upcoming.empty:
            row = upcoming.iloc[0]
            return {
                "season": int(year),
                "round": int(row["RoundNumber"]),
                "name": str(row["EventName"]),
                "country": str(row["Country"]),
                "date": str(row["EventDate"].date()),
                "format": str(row.get("EventFormat", "conventional")),
            }
    raise RuntimeError("could not find an upcoming race in this or next season's schedule")

def _season_round_count(season):
    sched = fastf1.get_event_schedule(season, include_testing=False)
    return int(sched["RoundNumber"].max())

def run_prediction(top_n=10, force=False):
    if not force and os.path.exists(PRED_CACHE_FILE):
        try:
            cached = json.load(open(PRED_CACHE_FILE, encoding="utf-8"))
            next_race = get_next_race()
            if cached.get("season") == next_race["season"] and cached.get("round") == next_race["round"]:
                cached["cached"] = True
                cached.setdefault("out_drivers", [])
                return with_live_view(cached)
        except Exception:
            pass

    next_race = get_next_race()
    target_season, target_round = next_race["season"], next_race["round"]
    prev_season = target_season - 1
    n_prev = _season_round_count(prev_season)
    completed_current = target_round - 1

    # Only train on the most recent rounds of each season, not the full history —
    # keeps runtime bounded regardless of how far into the season we are.
    prev_start = max(1, n_prev - PREV_SEASON_LOOKBACK_ROUNDS + 1)
    cur_start = max(1, completed_current - CURRENT_SEASON_LOOKBACK_ROUNDS + 1)

    history = defaultdict(lambda: deque(maxlen=RECENT_FORM_LOOKBACK))
    A_rows, b_rows, w_rows = [], [], []
    log_lines = []

    for rnd in range(prev_start, n_prev + 1):
        try:
            drv, teams_r, feats, pos = _with_retry(_load_real_round, prev_season, rnd, top_n, history, None)
            A_rows.append(feats); b_rows.extend(pos.tolist())
            w_rows.extend([1.0] * len(drv))
            log_lines.append(f"{prev_season} R{rnd} ok ({len(drv)} drivers, w=1.0)")
        except Exception as e:
            log_lines.append(f"{prev_season} R{rnd} failed: {e}")

    for rnd in range(cur_start, completed_current + 1):
        try:
            drv, teams_r, feats, pos = _with_retry(_load_real_round, target_season, rnd, top_n, history, None)
            A_rows.append(feats); b_rows.extend(pos.tolist())
            w_rows.extend([3.0] * len(drv))
            log_lines.append(f"{target_season} R{rnd} ok ({len(drv)} drivers, w=3.0)")
        except Exception as e:
            log_lines.append(f"{target_season} R{rnd} failed: {e}")

    if not A_rows:
        raise RuntimeError("no training data could be loaded")

    A_train_raw = np.vstack(A_rows)
    b_train = np.array(b_rows, dtype=float)
    W_vec = np.array(w_rows, dtype=float)

    A_train, mn, rng = normalise(A_train_raw)
    x_hat, b_hat_train, resid, rmse_train = weighted_ridge_least_squares(A_train, b_train, W_vec)

    try:
        drv, teams_t, feat_raw, practice_label = _with_retry(
            _load_future_quali, target_season, target_round, top_n, history)
        feat_norm = norm_apply(feat_raw, mn, rng)
        b_pred = feat_norm @ x_hat
        ranked = sorted(zip(drv, teams_t, b_pred.tolist()), key=lambda x: x[2])
    except Exception as e:
        # No quali/practice data yet (race hasn't happened) or the fetch got
        # rate-limited/blocked (seen repeatedly on shared CI runner IPs) —
        # either way, degrade to an empty prediction rather than crash the
        # whole run. The caller (generate_site.py) treats an empty "ranked"
        # as "nothing to publish yet" and leaves prior good output alone.
        practice_label = None
        ranked = []
        log_lines.append(f"target race ({target_season} R{target_round}) fetch failed: {e}")
    result = {
        "season": target_season,
        "round": target_round,
        "race_name": next_race["name"],
        "country": next_race["country"],
        "race_date": next_race["date"],
        "event_format": next_race["format"],
        "practice_session_used": practice_label,
        "trained_on": {
            "prev_season": prev_season,
            "prev_season_rounds_used": f"R{prev_start}-R{n_prev}",
            "current_season_rounds_used": f"R{cur_start}-R{completed_current}" if completed_current >= cur_start else "none",
            "training_rows": int(A_train_raw.shape[0]),
        },
        "features": FEATURE_NAMES,
        "coefficients": {f: round(float(c), 4) for f, c in zip(FEATURE_NAMES, x_hat)},
        "rmse": round(rmse_train, 4),
        "ranked": [
            {"pos": i + 1, "driver": d, "team": t, "score": round(s, 3)}
            for i, (d, t, s) in enumerate(ranked)
        ],
        "log": log_lines,
        "cached": False,
    }
    result["out_drivers"] = []
    json.dump(result, open(PRED_CACHE_FILE, "w", encoding="utf-8"), indent=2)
    return with_live_view(result)

def with_live_view(result):
    """Re-rank around any drivers marked out (crashed/retired/DNF) without retraining.
    The regression only ever modeled pre-race form, so a live incident can't be
    re-scored — the honest adjustment is: drop them, keep everyone else's relative
    order (which is still whatever the model predicted pre-race), renumber."""
    out = {o["driver"] for o in result.get("out_drivers", [])}
    running = [r for r in result["ranked"] if r["driver"] not in out]
    retired = [r for r in result["ranked"] if r["driver"] in out]
    live_ranked = []
    for i, r in enumerate(running):
        live_ranked.append({**r, "pos": i + 1, "status": "RUNNING"})
    for r in retired:
        meta = next(o for o in result["out_drivers"] if o["driver"] == r["driver"])
        live_ranked.append({**r, "pos": None, "status": "OUT", "reason": meta.get("reason", "")})
    out_result = dict(result)
    out_result["live_ranked"] = live_ranked
    return out_result

def mark_driver_out(driver, reason=""):
    if not os.path.exists(PRED_CACHE_FILE):
        raise RuntimeError("no prediction has been run yet")
    result = json.load(open(PRED_CACHE_FILE, encoding="utf-8"))
    driver = driver.upper().strip()
    if not any(r["driver"] == driver for r in result["ranked"]):
        raise RuntimeError(f"'{driver}' is not in the predicted field")
    result.setdefault("out_drivers", [])
    if not any(o["driver"] == driver for o in result["out_drivers"]):
        result["out_drivers"].append({
            "driver": driver, "reason": reason,
            "marked_at": time.strftime("%H:%M:%S"),
        })
    json.dump(result, open(PRED_CACHE_FILE, "w", encoding="utf-8"), indent=2)
    return with_live_view(result)

def mark_driver_in(driver):
    """Undo a mark-out (e.g. entered in error)."""
    if not os.path.exists(PRED_CACHE_FILE):
        raise RuntimeError("no prediction has been run yet")
    result = json.load(open(PRED_CACHE_FILE, encoding="utf-8"))
    driver = driver.upper().strip()
    result["out_drivers"] = [o for o in result.get("out_drivers", []) if o["driver"] != driver]
    json.dump(result, open(PRED_CACHE_FILE, "w", encoding="utf-8"), indent=2)
    return with_live_view(result)

def get_cached_live_view():
    """Only returns a cached prediction if it's still for the *current* next race —
    once the calendar rolls over to a new race, a stale cached podium (with retirements
    marked against the old race's field) must not be shown as if it's current."""
    if not os.path.exists(PRED_CACHE_FILE):
        return None
    result = json.load(open(PRED_CACHE_FILE, encoding="utf-8"))
    next_race = get_next_race()
    if result.get("season") != next_race["season"] or result.get("round") != next_race["round"]:
        return None
    result.setdefault("out_drivers", [])
    return with_live_view(result)
