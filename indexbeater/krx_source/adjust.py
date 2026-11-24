"""분할 → 배당 순서로 과거 가격을 고쳐 주는 파일 (계획서 2.2 ③·④, 2.3 ③·④).

# 왜 과거 가격을 고치나
- 주식을 50개로 쪼개면 가격이 1/50이 된다. 과거 가격도 1/50로 맞춰 줘야 그래프가 끊기지 않는다.
- 배당을 주면 그만큼 가격이 빠진다(배당락). 이것도 과거 가격에 반영해야 "배당 받은 만큼의 수익"까지 비교할 수 있다.

# 어떻게 고치나
- 이벤트가 일어난 날(ex_date)보다 '이전' 날짜의 줄에만 계수를 곱한다.
- 이벤트가 여러 번이면 계수가 계속 곱해진다. (최근 것부터 거꾸로 하나씩 적용한 것과 결과가 같다)
  - 분할: 가격 x (1/분할비율), 거래량 x 분할비율 (주식 수가 늘면 거래되는 주식 수도 늘어나 보여서)
  - 배당: 가격 x (1 - 배당금/전일종가), 거래량은 그대로
  - 기준가 변동(권리락·감자·합병 등): 가격 x (기준가/전일종가), 거래량은 그대로 (base_factor)
  - 유상증자 권리락: 가격 x (권리락 이론가/전일종가), 거래량은 그대로 (확정된 것만, rights_factor)

# 결과 열의 뜻
- split_factor, div_factor, rights_factor, base_factor : 각 줄에 곱해진 누적 계수
- close_s, open_s ...      : 분할만 고친 가격 → 네이버·야후 종가와 비교할 때 쓴다
- adj_close, adj_open ...  : 분할+배당+권리락을 모두 고친 가격 → 야후 Adj Close와 비교, 백테스트에 쓴다
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import paths

PRICE_COLS = ["open", "high", "low", "close"]
ALL_CONF = ("confirmed", "yahoo", "estimated", "low")   # 배당락일을 얼마나 믿는지 (확정 > 야후 실제 날짜 > 추정 > 낮음)


def cumulative_factor(dates: pd.Series, events: pd.DataFrame, factor_col: str) -> np.ndarray:
    """날짜마다 곱해야 할 계수를 만든다. 처음엔 모두 1이고, 이벤트 날짜 이전 줄마다 계수를 곱한다."""
    f = np.ones(len(dates))                 # 1은 곱해도 변하지 않는 수라서 시작값으로 쓴다
    d = dates.to_numpy()
    for ex, fac in zip(events["ex_date"], events[factor_col]):
        f[d < np.datetime64(ex)] *= fac     # 이벤트 날짜보다 이전인 줄에만 곱한다
    return f


def apply_adjustments(raw: pd.DataFrame, splits: pd.DataFrame, divs: pd.DataFrame,
                      confidences: tuple[str, ...] = ALL_CONF, rights: pd.DataFrame | None = None, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """원래 가격표(raw)에 분할·배당·권리락 계수를 곱해서 보정된 가격 열을 붙인다. rights 는 confidence 가 confirmed 인 것만 쓴다."""
    rt = pd.DataFrame(columns=["code", "ex_date", "factor"]) if rights is None or not len(rights) else rights[rights["confidence"] == "confirmed"].copy()
    rt["ex_date"] = pd.to_datetime(rt["ex_date"])
    r_by = {c: g for c, g in rt.groupby("code")}
    bs = pd.DataFrame(columns=["code", "ex_date", "factor"]) if base is None or not len(base) else base.copy()
    bs["ex_date"] = pd.to_datetime(bs["ex_date"])
    b_by = {c: g for c, g in bs.groupby("code")}
    # 기준가 보정은 그날 일어난 일을 거래소가 직접 계산한 값이라 가장 정확하다.
    # 같은 일을 다시 보정하지 않도록, 유상증자는 3일 안, 배당은 같은 날에 기준가 이벤트가 있으면 뺀다.
    def _drop_near(ev: pd.DataFrame, days: int = 3) -> pd.DataFrame:
        if not len(ev) or not len(bs):
            return ev
        keep = []
        for r in ev.itertuples():
            g = b_by.get(r.code)
            keep.append(g is None or not ((g["ex_date"] - r.ex_date).abs() <= pd.Timedelta(days=days)).any())
        return ev[keep]
    rt = _drop_near(rt)
    r_by = {c: g for c, g in rt.groupby("code")}
    s = splits.copy()                       # 원본 표를 망가뜨리지 않게 복사해서 쓴다
    s["ex_date"] = pd.to_datetime(s["ex_date"])
    s["factor"] = 1.0 / s["share_ratio"].astype(float)   # 50배로 쪼개졌으면 가격 계수는 1/50
    dv = divs.copy()
    if len(dv):
        dv["ex_date"] = pd.to_datetime(dv["ex_date"])
        dv = dv[dv["confidence"].isin(confidences)]       # 믿을 만한 정도가 허용된 배당만 쓴다
        dv = _drop_near(dv, 0)      # 배당은 같은 날일 때만 뺀다
    # 종목코드 → 그 종목의 이벤트 표, 로 빨리 찾을 수 있게 사전(dict)으로 만든다
    s_by = {c: g for c, g in s.groupby("code")}
    d_by = {c: g for c, g in dv.groupby("code")} if len(dv) else {}
    empty = pd.DataFrame({"ex_date": pd.to_datetime([]), "factor": []})   # 이벤트가 없는 종목용 빈 표
    parts = []
    for code, g in raw.sort_values(["code", "date"]).groupby("code", sort=False):
        g = g.copy()
        g["split_factor"] = cumulative_factor(g["date"], s_by.get(code, empty), "factor")
        g["div_factor"] = cumulative_factor(g["date"], d_by.get(code, empty), "factor")
        g["rights_factor"] = cumulative_factor(g["date"], r_by.get(code, empty), "factor")
        g["base_factor"] = cumulative_factor(g["date"], b_by.get(code, empty), "factor")
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    for c in PRICE_COLS:
        out[f"{c}_s"] = out[c] * out["split_factor"]          # 분할만 고친 가격
        out[f"adj_{c}"] = out[f"{c}_s"] * out["div_factor"] * out["rights_factor"] * out["base_factor"]   # 거기에 배당·권리락·기준가 변동까지 고친 가격
    out["volume_s"] = out["volume"] / out["split_factor"]     # 거래량은 가격과 반대로 늘려 준다
    return out


def main(argv=None):
    """터미널에서 `python -m indexbeater.krx_source adjust` 로 실행할 때 쓰는 입구."""
    ap = argparse.ArgumentParser(prog="krx_source adjust")
    ap.add_argument("--raw", default=str(paths.RAW_DAILY))
    ap.add_argument("--split-only", action="store_true", help="분할만 보정해서 krx_adjusted_split_only.parquet 로 저장 (네이버 비교용)")
    ap.add_argument("--exclude-low", action="store_true", help="2023 사업연도 이후 추정 배당락일(low)은 적용 제외")
    a = ap.parse_args(argv)
    raw = pd.read_parquet(a.raw)
    splits = pd.read_csv(paths.SPLIT_EVENTS, dtype={"code": str}, parse_dates=["ex_date"])   # 종목코드는 글자로 읽어야 앞의 0이 안 사라진다
    divs = (pd.read_csv(paths.DIV_EVENTS, dtype={"code": str}, parse_dates=["ex_date"])
            if paths.DIV_EVENTS.exists() else pd.DataFrame())
    if a.split_only:                        # 배당 표를 비워서 넘기면 div_factor 가 모두 1이 된다
        divs = pd.DataFrame()
    conf = ("confirmed", "yahoo", "estimated") if a.exclude_low else ALL_CONF
    if len(divs) and paths.YAHOO_DIVS.exists():   # 2011~2014 배당은 DART에 없어서 야후 배당으로 보충한 파일을 이어 붙인다
        y = pd.read_csv(paths.YAHOO_DIVS, dtype={"code": str}, parse_dates=["ex_date"])
        divs = pd.concat([divs, y], ignore_index=True)
    rights = (pd.read_csv(paths.RIGHTS_EVENTS, dtype={"code": str}, parse_dates=["ex_date"])
              if paths.RIGHTS_EVENTS.exists() and not a.split_only else None)
    from .base import build_base_events, dedupe_splits, drop_unsupported_big_dividends
    splits = dedupe_splits(splits, raw)         # 같은 날 분할이 둘로 잡힌 경우 하나만 (두 번 보정 방지)
    base = None
    if not a.split_only:                    # 기준가 이벤트는 raw 에서 바로 만든다 (키 불필요)
        base = build_base_events(raw, splits)
        base.to_csv(paths.BASE_EVENTS, index=False, encoding="utf-8-sig")
    divs = drop_unsupported_big_dividends(divs, raw)   # 실제로 가격이 안 떨어진 큰 배당은 오류로 보고 뺀다
    out = apply_adjustments(raw, splits, divs, conf, rights, base)
    dest = paths.ADJUSTED_SPLIT_ONLY if a.split_only else paths.ADJUSTED
    out.to_parquet(dest, index=False)
    print(f"저장: {dest}  행 {len(out):,}")


if __name__ == "__main__":
    main()
