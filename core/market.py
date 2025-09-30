# modified: add get_market_indicators_pair (macro + strategy) and coarse filtering helpers
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

try:
    from price_cache import load_many as load_price_cache
except Exception:
    load_price_cache = None

from .universe import _load_universe_raw, build_base_universe

def _pct_rank(series: pd.Series, value: float) -> float | None:
    try:
        s = pd.Series(series).dropna()
        if s.empty or value is None or np.isnan(value):
            return None
        return float((s.le(value).mean()) * 100.0)
    except Exception:
        return None

def _choose_proxy_symbol(candidates: list[str]) -> str | None:
    prefs = ["0050", "2330", "1101", "2412"]
    for p in prefs:
        if p in candidates:
            return p
    return candidates[0] if candidates else None

def _calc_atr_pctile(dfp: pd.DataFrame, win_atr: int = 14, lookback: int = 252) -> tuple[float | None, float | None]:
    try:
        df = dfp.copy()
        cols = {c.lower(): c for c in df.columns}
        c = df[cols.get("close","close")].astype(float)
        h = df[cols.get("high","high")].astype(float) if "high" in cols else None
        l = df[cols.get("low","low")].astype(float) if "low" in cols else None
        if c.dropna().shape[0] < win_atr + 1:
            return None, None
        if h is None or l is None:
            tr = c.diff().abs()
        else:
            prev_c = c.shift(1)
            tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(win_atr).mean()
        atrp = (atr / c).dropna()
        base = atrp.dropna().iloc[-lookback:] if atrp.dropna().shape[0] >= lookback else atrp.dropna()
        last = float(atrp.dropna().iloc[-1])
        pctile = _pct_rank(base, last)
        return last, pctile
    except Exception:
        return None, None

def _calc_bbw_pctile(dfp: pd.DataFrame, win: int = 20, lookback: int = 252) -> tuple[float | None, float | None]:
    try:
        df = dfp.copy()
        cols = {c.lower(): c for c in df.columns}
        c = df[cols.get("close","close")].astype(float)
        if c.dropna().shape[0] < win + 1:
            return None, None
        ma = c.rolling(win).mean()
        sd = c.rolling(win).std()
        mid = ma
        up = ma + 2 * sd
        lo = ma - 2 * sd
        bbw = ((up - lo) / mid).replace([np.inf, -np.inf], np.nan).dropna()
        base = bbw.iloc[-lookback:] if bbw.dropna().shape[0] >= lookback else bbw
        last = float(bbw.iloc[-1])
        pctile = _pct_rank(base, last)
        return last, pctile
    except Exception:
        return None, None

def _calc_breadth20(price_map: dict[str, pd.DataFrame]) -> float | None:
    try:
        count = 0
        above = 0
        for s, dfp in price_map.items():
            if dfp is None or getattr(dfp, "empty", True):
                continue
            cols = {c.lower(): c for c in dfp.columns}
            if "close" not in cols:
                continue
            c = dfp[cols["close"].astype(float)]
            if c.dropna().shape[0] < 20:
                continue
            ma20 = c.rolling(20).mean()
            if pd.notna(c.iloc[-1]) and pd.notna(ma20.iloc[-1]):
                count += 1
                if c.iloc[-1] > ma20.iloc[-1]:
                    above += 1
        if count <= 0:
            return None
        return float(above / count)
    except Exception:
        return None

# ------------------------------
# 原本的單一版本（舊介面，保留相容性）
# ------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def get_market_indicators(inc_tse: bool, inc_otc: bool, include_emerging: bool) -> dict | None:
    """
    原始介面：基於板別（build_base_universe）計算市場指標（breadth、proxy ATR/BBW）。
    為相容性保留。
    """
    if load_price_cache is None:
        return None
    uni_df = _load_universe_raw()
    base_df = build_base_universe(uni_df, inc_tse, inc_otc, include_emerging)
    syms = base_df["symbol"].astype(str).tolist()
    if not syms:
        return None
    proxy = _choose_proxy_symbol(syms)
    try:
        price_map = load_price_cache(syms)
        breadth20 = _calc_breadth20(price_map)
        atrp_last = atrp_pct = bbw_last = bbw_pct = None
        if proxy and proxy in price_map and price_map[proxy] is not None and not price_map[proxy].empty:
            atrp_last, atrp_pct = _calc_atr_pctile(price_map[proxy], 14, 252)
            bbw_last, bbw_pct = _calc_bbw_pctile(price_map[proxy], 20, 252)
        cond_low_vol = (atrp_pct is not None and atrp_pct <= 40.0)
        cond_low_bbw = (bbw_pct is not None and bbw_pct <= 40.0)
        cond_breadth_mid = (breadth20 is not None and 0.35 <= breadth20 <= 0.60)
        majority = sum([cond_low_vol, cond_low_bbw, cond_breadth_mid])
        regime = "綠"
        reason = []
        if breadth20 is not None and breadth20 > 0.60:
            regime = "綠"
            reason.append("Breadth>60%")
        if majority >= 2:
            regime = "黃"
            reason.append(f"三條件成立 {majority}/3")
        if breadth20 is not None and breadth20 < 0.35 and (atrp_pct is not None and atrp_pct >= 60.0):
            regime = "紅"
            reason = ["Breadth<35% 且 ATR%分位≥60%"]
        return {
            "regime": regime,
            "reason": "；".join(reason) if reason else "",
            "proxy_symbol": proxy,
            "atrp": atrp_last,
            "atrp_pctile": atrp_pct,
            "bbw": bbw_last,
            "bbw_pctile": bbw_pct,
            "breadth20": breadth20,
        }
    except Exception:
        return None

# ------------------------------
# 新增：pair 介面（同時回傳 macro + strategy）
# ------------------------------
def _extract_last_close_and_adv(dfp: pd.DataFrame) -> Tuple[float|None, float|None]:
    """
    從 price dataframe 嘗試抽出最後收盤價與 ADV（若可得）。
    ADV 欄位可能名稱不一，嘗試幾個候選欄位名稱。
    """
    try:
        cols = {c.lower(): c for c in dfp.columns}
        last_close = None
        adv_val = None
        # last close
        if "close" in cols:
            s = dfp[cols["close"].dropna()]
            if not s.empty:
                last_close = float(s.iloc[-1])
        # adv 候選
        cand_adv = ["adv10_m","adv10","adv","avg_volume","volume_avg","volume","turnover","成交金額","成交量"]
        for cand in cand_adv:
            if cand in cols:
                try:
                    v = dfp[cols[cand]].dropna()
                    if not v.empty:
                        adv_val = float(v.iloc[-1])
                        break
                except Exception:
                    continue
        return last_close, adv_val
    except Exception:
        return None, None

def _build_indicators_from_syms(syms: List[str], price_map: Dict[str, pd.DataFrame]) -> dict | None:
    """
    給定 symbol list（可能只包含 price_map 中的 subset），用內部 helpers 建立單一指標 dict。
    """
    if not syms:
        return None
    available_syms = [s for s in syms if s in price_map and price_map[s] is not None and not price_map[s].empty]
    if not available_syms:
        return None
    proxy = _choose_proxy_symbol(available_syms)
    pm = {s: price_map[s] for s in available_syms}
    bread = _calc_breadth20(pm)
    atrp_last = atrp_pct = bbw_last = bbw_pct = None
    if proxy and proxy in pm:
        try:
            atrp_last, atrp_pct = _calc_atr_pctile(pm[proxy], 14, 252)
            bbw_last, bbw_pct = _calc_bbw_pctile(pm[proxy], 20, 252)
        except Exception:
            atrp_last = atrp_pct = bbw_last = bbw_pct = None
    cond_low_vol = (atrp_pct is not None and atrp_pct <= 40.0)
    cond_low_bbw = (bbw_pct is not None and bbw_pct <= 40.0)
    cond_breadth_mid = (bread is not None and 0.35 <= bread <= 0.60)
    majority = sum([cond_low_vol, cond_low_bbw, cond_breadth_mid])
    regime = "綠"
    reason = []
    if bread is not None and bread > 0.60:
        regime = "綠"
        reason.append("Breadth>60%")
    if majority >= 2:
        regime = "黃"
        reason.append(f"三條件成立 {majority}/3")
    if bread is not None and bread < 0.35 and (atrp_pct is not None and atrp_pct >= 60.0):
        regime = "紅"
        reason = ["Breadth<35% 且 ATR%分位≥60%"]
    return {
        "regime": regime,
        "reason": "；".join(reason) if reason else "",
        "proxy_symbol": proxy,
        "atrp": atrp_last,
        "atrp_pctile": atrp_pct,
        "bbw": bbw_last,
        "bbw_pctile": bbw_pct,
        "breadth20": bread,
        "pool_count": len(available_syms),
        "pool_symbols_sample": available_syms[:200],
    }

@st.cache_data(show_spinner=False, ttl=300)
def get_market_indicators_pair(inc_tse: bool, inc_otc: bool, include_emerging: bool,
                               min_price: float | None = None, min_adv: float | None = None) -> dict | None:
    """
    回傳一個 dict 包含 macro 與 strategy 兩組指標：
      {"macro": {...}, "strategy": {...}}。
    strategy 是在 macro base pool 篩完 price/min_adv 後的子池（若提供 min_price/min_adv）。
    若 strategy 無法計算（例如沒有 price cache），strategy 為 None。
    """
    if load_price_cache is None:
        return None
    uni_df = _load_universe_raw()
    base_df = build_base_universe(uni_df, inc_tse, inc_otc, include_emerging)
    syms = base_df["symbol"].astype(str).tolist()
    if not syms:
        return None
    try:
        price_map = load_price_cache(syms)
        # macro indicators (use existing builder)
        macro = _build_indicators_from_syms(syms, price_map)
        # strategy (apply coarse filters if provided)
        if min_price is None and min_adv is None:
            strategy = macro  # same as macro if no filters
        else:
            filtered_syms = []
            for s in syms:
                if s not in price_map:
                    continue
                dfp = price_map[s]
                if dfp is None or getattr(dfp, "empty", True):
                    continue
                last_close, adv_val = _extract_last_close_and_adv(dfp)
                if min_price is not None:
                    if last_close is None or last_close < float(min_price):
                        continue
                if min_adv is not None:
                    # NOTE: min_adv is in 百萬台幣 on UI; data adv_val may be in same unit or not.
                    # We leave the caller responsible for unit consistency. Here we compare directly.
                    if adv_val is None or adv_val < float(min_adv):
                        continue
                filtered_syms.append(s)
            # if no symbols pass, strategy None
            strategy = _build_indicators_from_syms(filtered_syms, price_map) if filtered_syms else None
        return {"macro": macro, "strategy": strategy}
    except Exception:
        return None
