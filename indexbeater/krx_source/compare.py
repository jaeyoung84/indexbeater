"""KRX·네이버·야후를 같은 종목, 같은 날짜로 맞춰 비교하는 함수 모음 (notebooks/02_외부자료비교 가 쓴다).

# 비교의 기본 생각
- 세 자료는 '가격을 보정한 정도'가 서로 다르다. 그래서 같은 기준끼리만 비교한다.
  - KRX 분할보정(close_s)  ↔ 네이버 종가  (둘 다 '분할만' 보정)
  - KRX 분할+배당보정(adj_close) ↔ 야후 종가 (둘 다 '분할+배당' 보정)
- 정답은 KRX 원자료와 공시 사실(분할·배당)이다. 네이버·야후가 우리 값과 다르면 '왜 다른지'를 밝힌다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import external

MATCH_TOL = 0.005      # 종가가 0.5% 이내로 같으면 '일치'
RET_TOL = 0.002        # 하루 수익률이 0.2%p 이내로 같으면 '일치' (야후 비교용)


def select_sample(meta: pd.DataFrame, split_codes: set[str], div_codes: set[str], n_each: int = 12, seed: int = 42) -> pd.DataFrame:
    """대표 표본을 뽑는다. meta: code, market, first, last, n(거래일 수), mc(시가총액 중앙값).

    - 상장 중인 종목: 시장(코스피·코스닥) x 크기(대·중·소) 6칸에서 n_each 개씩
    - 상장폐지 종목: 시장별 n_each 개씩 (거래일 250일 이상인 것만)
    - 분할을 겪은 종목, 배당을 많이 준 종목: 따로 추가로 뽑는다
    같은 종목이 겹치면 한 번만 넣고 이유(why)를 이어 붙인다.
    """
    rng = np.random.default_rng(seed)
    m = meta[meta["n"] >= 250].copy()
    active = m[m["last"] >= "2026-11-01"].copy()
    active["size"] = active.groupby("market")["mc"].transform(lambda s: pd.qcut(s, 3, labels=["소형", "중형", "대형"]))
    picks = []
    for (mk, sz), g in active.groupby(["market", "size"], observed=True):
        picks.append(g.sample(min(n_each, len(g)), random_state=int(rng.integers(1e9))).assign(why=f"상장중 {mk} {sz}"))
    for mk, g in m[m["last"] < "2026-11-01"].groupby("market"):
        picks.append(g.sample(min(n_each, len(g)), random_state=int(rng.integers(1e9))).assign(why=f"상장폐지 {mk}"))
    sp = active[active["code"].isin(split_codes)]
    picks.append(sp.sample(min(2 * n_each, len(sp)), random_state=int(rng.integers(1e9))).assign(why="분할 경험"))
    dv = active[active["code"].isin(div_codes)]
    picks.append(dv.sample(min(2 * n_each, len(dv)), random_state=int(rng.integers(1e9))).assign(why="배당 다수"))
    s = pd.concat(picks)
    s = s.groupby("code", as_index=False).agg({**{c: "first" for c in s.columns if c not in ("code", "why")}, "why": " / ".join})
    return s.drop(columns=["size"], errors="ignore").reset_index(drop=True)


def compare_naver(k: pd.DataFrame, code: str) -> pd.DataFrame | None:
    """한 종목: KRX 분할보정 종가(close_s) 와 네이버 종가를 날짜별로 맞춰 본다. 네이버에 없으면 None."""
    n = external.load_naver(code)
    if n is None:
        return None
    j = k.set_index("date")[["close_s"]].join(n[["close"]].rename(columns={"close": "naver"}), how="outer")
    j["code"] = code
    return j


def compare_yahoo(k: pd.DataFrame, code: str) -> pd.DataFrame | None:
    """한 종목: KRX 분할+배당보정 종가(adj_close) 와 야후 종가를 날짜별로 맞춰 본다. 야후에 없으면 None."""
    y = external.load_yahoo(code)
    if y is None:
        return None
    j = k.set_index("date")[["close", "close_s", "adj_close"]].join(y[["close"]].rename(columns={"close": "yahoo"}), how="outer")
    j["code"] = code
    return j


def match_stats(j: pd.DataFrame, ours: str, theirs: str, tol: float = MATCH_TOL) -> dict:
    """겹치는 날짜에서 일치율과 날짜 빠짐을 센다. ours/theirs 는 열 이름."""
    both = j.dropna(subset=[ours, theirs])
    only_ours = j[j[ours].notna() & j[theirs].isna()]
    only_theirs = j[j[theirs].notna() & j[ours].isna()]
    rel = (both[theirs] / both[ours] - 1).abs()
    return {"days_ours": int(j[ours].notna().sum()), "days_both": len(both), "days_only_ours": len(only_ours),
            "days_only_theirs": len(only_theirs), "match_days": int((rel <= tol).sum()),
            "match_rate": float((rel <= tol).mean()) if len(both) else np.nan,
            "median_abs_diff": float(rel.median()) if len(both) else np.nan}


def return_match(j: pd.DataFrame, ours: str, theirs: str, tol: float = RET_TOL) -> pd.DataFrame:
    """두 가격의 '하루 수익률'을 비교한다. 수준(level)이 달라도 수익률이 같으면 같은 보정을 한 것이다."""
    both = j.dropna(subset=[ours, theirs])
    r = pd.DataFrame({"ours": both[ours].pct_change(), "theirs": both[theirs].pct_change()}).dropna()
    r["diff"] = (r["theirs"] - r["ours"]).abs()
    r["ok"] = r["diff"] <= tol
    return r
