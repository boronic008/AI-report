from __future__ import annotations
import streamlit as st
import json, datetime
import pandas as pd
from pathlib import Path
from .utils import _now_stamp, _horizon_tag, _disp

from .paths import (
    LAST_MONITOR_JSON, LAST_MONITOR_SUMMARY_PATH, LAST_MONITOR_TXT,
    get_monitor_dir, OUT_DIR, FINAL_DIR
)

def _round5(v):
    try: return round(float(v), 5)
    except Exception: return v

def _metric_text(x):
    try: return f"{float(x):.5f}"
    except Exception: return "—"

def summarize_and_autotune(train_json: str | None, bt_json: str | None, final_csv: str | None, params_used: dict | None, weekly: bool=False) -> tuple[dict, dict]:
    """
    Summarize train/backtest/final outputs into a monitoring summary dict and
    optionally propose new defaults (new_defaults).
    """
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "ece": None, "brier": None,
        "top_bin_n": None, "top_bin_pred": None, "top_bin_hit": None, "top_bin_deviation_pct": None,
        "trades": None, "peak20_hit_rate": None,
        "final_today_count": None,
        "params_used": params_used or {},
    }
    new_defaults = {}
    try:
        if train_json and Path(train_json).exists():
            tr = json.loads(Path(train_json).read_text(encoding="utf-8"))
            # support multiple key layouts
            cal = tr.get("calibration") or tr.get("calib") or tr.get("cal") or {}
            summary["ece"] = cal.get("ece") or cal.get("ECE") or tr.get("ece")
            summary["brier"] = cal.get("brier") or cal.get("Brier") or tr.get("brier")
            rel = cal.get("reliability") or cal.get("reliabilities") or tr.get("reliability") or []
            if rel:
                try:
                    top = max(rel, key=lambda r: float(r.get("pred_mean", r.get("pred", 0) or 0)))
                except Exception:
                    top = rel[0]
                n = int(_num_or_none(top.get("n") or top.get("count") or 0) or 0)
                pm = _num_or_none(top.get("pred_mean") or top.get("pred") or top.get("p") or 0) or 0.0
                hr = _num_or_none(top.get("hit_rate") or top.get("hit") or top.get("actual_rate") or 0) or 0.0
                dev = (abs(hr - pm) / max(pm, 1e-6)) if (pm is not None and pm > 0) else None
                dev_pct = None if dev is None else round(dev * 100, 5)
                summary.update({"top_bin_n": n, "top_bin_pred": pm, "top_bin_hit": hr, "top_bin_deviation_pct": dev_pct})
                # propose stronger calibration if top bin too small or dev large
                try:
                    if (n < 300) or (dev is not None and dev * 100.0 > 10.0):
                        new_defaults["calib_min_bin_size"] = max(int((params_used or {}).get("calib_min_bin_size", 250)), 800)
                        new_defaults["final_bins"] = min(int((params_used or {}).get("final_bins", 30)), 20)
                except Exception:
                    pass
    except Exception:
        # swallow to allow partial summary
        pass

    try:
        if bt_json and Path(bt_json).exists():
            bt = json.loads(Path(bt_json).read_text(encoding="utf-8"))
            trades = bt.get("trades") or bt.get("trade_records") or []
            # trades may be a dict or list
            summary["trades"] = _normalize_trades(trades)
            summary["peak20_hit_rate"] = bt.get("peak20_hit_rate") or _get_nested(bt, [["metrics","peak20_hit_rate"], ["peak20","hit_rate"]])
    except Exception:
        pass

    try:
        if final_csv and Path(final_csv).exists():
            df_final = pd.read_csv(final_csv)
            summary["final_today_count"] = int(len(df_final))
            if weekly and len(df_final) == 0:
                cur_min_prob = float((params_used or {}).get("final_min_prob", 0.32))
                cur_p = float((params_used or {}).get("min_daily_p", 0.93))
                if cur_min_prob > 0.20:
                    new_defaults["final_min_prob"] = max(0.20, round(cur_min_prob - 0.02, 2))
                elif cur_p > 0.50:
                    new_defaults["min_daily_p"] = round(max(0.50, cur_p - 0.01), 2)
    except Exception:
        pass

    # ensure types
    return summary, new_defaults


# helpers used by pages (robust monitor summary load/write)

def _num_or_none(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            if isinstance(x, float) and (pd.isna(x) or x == float("inf") or x == float("-inf")):
                return None
            return x
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return None
            return float(s) if ("." in s or "e" in s.lower()) else int(s)
    except Exception:
        return None
    return None


def _normalize_trades(val):
    if val is None:
        return None
    if isinstance(val, list):
        return len(val)
    if isinstance(val, dict):
        for k in ("count","n","total","size","trades_count"):
            v = val.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        try:
            return len(val)
        except Exception:
            return None
    if isinstance(val, (int, float, str)):
        v = _num_or_none(val)
        return int(v) if v is not None else None
    return None


def _get_nested(d: dict, paths: list[list[str]]):
    for path in paths:
        cur = d
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok:
            return cur
    return None


def _compute_top_dev_from_deciles(decile_csv: Path | None) -> float | None:
    if not decile_csv or not decile_csv.exists():
        return None
    try:
        df = pd.read_csv(decile_csv)
        cols_lower = [c.lower() for c in df.columns]
        pred_cols = [c for c in df.columns if c.lower() in ("pred","p","expected","expected_prob","avg_pred","mean_pred","avg_score","pred_prob")]
        hit_cols  = [c for c in df.columns if c.lower() in ("hit","rate","actual","actual_rate","avg_hit","mean_hit","hit_rate")]
        if "decile" in cols_lower and not df.empty:
            # pick highest decile row
            try:
                top = df.loc[df["decile"].idxmax()]
            except Exception:
                top = df.iloc[-1]
        elif "bin" in cols_lower and not df.empty:
            try:
                top = df.loc[df["bin"].idxmax()]
            except Exception:
                top = df.iloc[-1]
        else:
            top = df.iloc[0] if not df.empty else None
        if top is None:
            return None
        pred = None
        hit = None
        for c in pred_cols:
            if c in top.index:
                pred = _num_or_none(top.get(c)); break
        for c in hit_cols:
            if c in top.index:
                hit = _num_or_none(top.get(c)); break
        if pred and pred > 0 and (hit is not None):
            return abs(hit - pred) / pred * 100.0
    except Exception:
        return None
    return None


def _extract_from_latest_log() -> tuple[float | None, float | None, Path | None]:
    try:
        logs = sorted(Path(OUT_DIR).glob("web_run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            return None, None, None
        txt = logs[0].read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"ECE\s*[=:]\s*([0-9.]+).*?Brier\s*[=:]\s*([0-9.]+)", txt, flags=re.IGNORECASE | re.S)
        e = float(m.group(1)) if m else None
        b = float(m.group(2)) if m else None
        m2 = re.search(r"decile.*?([^\s,;\'\"]+decile[^\s,;\'\"]*\.csv)", txt, flags=re.IGNORECASE)
        decile_path = None
        if m2:
            name = Path(m2.group(1)).name
            cands = list(Path(OUT_DIR).rglob(name))
            if cands:
                decile_path = cands[0]
        return e, b, decile_path
    except Exception:
        return None, None, None


def _load_monitor_summary_latest() -> tuple[dict | None, Path | None]:
    # priority: LAST_MONITOR_JSON -> LAST_MONITOR_SUMMARY_PATH -> newest monitor-like json under OUT_DIR
    try:
        p = Path(LAST_MONITOR_JSON)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")), p
            except Exception:
                try:
                    return json.loads(p.read_text(encoding="utf-8", errors="ignore")), p
                except Exception:
                    pass
    except Exception:
        pass

    try:
        p2 = Path(LAST_MONITOR_SUMMARY_PATH)
        if p2.exists():
            try:
                return json.loads(p2.read_text(encoding="utf-8")), p2
            except Exception:
                try:
                    return json.loads(p2.read_text(encoding="utf-8", errors="ignore")), p2
                except Exception:
                    pass
    except Exception:
        pass

    newest_p, newest_ts = None, 0.0
    try:
        for q in Path(OUT_DIR).rglob("*.json"):
            name = q.name.lower()
            if any(k in name for k in ("monitor","train_report","bt_report","backtest")):
                try:
                    ts = q.stat().st_mtime
                    if ts > newest_ts:
                        newest_ts, newest_p = ts, q
                except Exception:
                    continue
        if newest_p:
            try:
                return json.loads(newest_p.read_text(encoding="utf-8")), newest_p
            except Exception:
                try:
                    return json.loads(newest_p.read_text(encoding="utf-8", errors="ignore")), newest_p
                except Exception:
                    return None, newest_p
    except Exception:
        pass
    return None, None


def _write_monitor_summary_from_reports(train_path: str | Path | None, bt_path: str | Path | None, final_csv: str | Path | None = None) -> tuple[dict | None, Path | None, str]:
    try:
        train_p = Path(train_path) if train_path else None
        bt_p = Path(bt_path) if bt_path else None
        have_train = bool(train_p and train_p.exists())
        have_bt = bool(bt_p and bt_p.exists())
        if not (have_train or have_bt):
            return None, None, "沒有可用的 train/bt 報表，略過寫入監控摘要"

        summary = None
        try:
            if have_train and have_bt:
                summary, _ = summarize_and_autotune(str(train_p), str(bt_p), str(final_csv) if final_csv else None, params_used={})
        except Exception:
            summary = None

        def _jload(p: Path) -> dict:
            try:
                return json.loads(p.read_text(encoding="utf-8")) if p else {}
            except Exception:
                try:
                    return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    return {}

        if not isinstance(summary, dict):
            tr = _jload(train_p) if have_train else {}
            bt = _jload(bt_p) if have_bt else {}

            def _get(d: dict, paths: list[list[str]]):
                for path in paths:
                    cur = d
                    ok = True
                    for k in path:
                        if not isinstance(cur, dict) or k not in cur:
                            ok = False
                            break
                        cur = cur[k]
                    if ok:
                        return cur
                return None

            ece = _get(tr, [["ece"],["ECE"],["calib","ece"],["calibration","ece"],["metrics","ece"]]) or _get(bt, [["ece"],["ECE"],["calib","ece"],["calibration","ece"],["metrics","ece"]])
            brier = _get(tr, [["brier"],["Brier"],["calib","brier"],["calibration","brier"],["metrics","brier"]]) or _get(bt, [["brier"],["Brier"],["calib","brier"],["calibration","brier"],["metrics","brier"]])

            top_dev = _get(tr, [["top_bin_deviation_pct"],["top_bin","deviation_pct"],["top_bin","dev_pct"]]) or _get(bt, [["top_bin_deviation_pct"],["top_bin","deviation_pct"],["top_bin","dev_pct"]])
            if top_dev is None:
                pred = _get(tr, [["top_bin_pred"],["top_bin","pred"]) or _get(bt, [["top_bin_pred"],["top_bin","pred"]])
                hit  = _get(tr, [["top_bin_hit"],["top_bin","hit"]]) or _get(bt, [["top_bin_hit"],["top_bin","hit"]])
                pred = _num_or_none(pred); hit = _num_or_none(hit)
                if pred and pred > 0 and hit is not None:
                    top_dev = abs(hit - pred) / pred * 100.0

            trades = _normalize_trades((bt.get("trades") if isinstance(bt, dict) else None) or (tr.get("trades") if isinstance(tr, dict) else None))
            peak20 = _get(bt, [["peak20_hit_rate"],["metrics","peak20_hit_rate"]])
            final_today = _num_or_none(_get(tr, [["final_today_count"],["final","today_count"]])) or 0
            params_used = (tr.get("params_used") if isinstance(tr, dict) else None) or (bt.get("params_used") if isinstance(bt, dict) else {}) or {}

            # try extract from latest log / decile csv
            if (ece is None) or (brier is None) or (top_dev is None):
                try:
                    e_log, b_log, decile_path = _extract_from_latest_log()
                    if ece is None and e_log is not None:
                        ece = e_log
                    if brier is None and b_log is not None:
                        brier = b_log
                    if top_dev is None and decile_path:
                        top_dev = _compute_top_dev_from_deciles(decile_path)
                except Exception:
                    pass

            if top_dev is None:
                try:
                    for p in Path(OUT_DIR).rglob("*decile*.csv"):
                        td = _compute_top_dev_from_deciles(p)
                        if td is not None:
                            top_dev = td
                            break
                except Exception:
                    pass

            # if final count still missing, try to find final/picked csv in FINAL_DIR or OUT_DIR
            if (not final_today) or int(final_today) == 0:
                try:
                    candidates = []
                    try:
                        candidates += list(Path(FINAL_DIR).glob("*.csv"))
                    except Exception:
                        pass
                    try:
                        candidates += [p for p in Path(OUT_DIR).glob("*.csv") if ("final" in p.name.lower() or "picked" in p.name.lower())]
                    except Exception:
                        pass
                    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                    if candidates:
                        for p in candidates[:3]:
                            try:
                                df = pd.read_csv(p)
                                if not df.empty:
                                    final_today = int(len(df))
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass

            summary = {
                "timestamp": datetime.datetime.now().isoformat(),
                "ece": _num_or_none(ece),
                "brier": _num_or_none(brier),
                "top_bin_deviation_pct": _num_or_none(top_dev),
                "trades": trades,
                "peak20_hit_rate": _num_or_none(peak20),
                "final_today_count": int(final_today),
                "params_used": params_used,
            }

        dst = Path(LAST_MONITOR_JSON)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary, dst, f"已覆寫監控摘要：{dst}"
    except Exception as e:
        return None, None, f"生成/寫入監控摘要失敗：{e}"