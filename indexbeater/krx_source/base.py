"""KRX '기준가'가 전날 종가와 크게 다른 날을 찾아, 과거 가격을 이어 붙일 계수를 만드는 파일.

# 기준가란
- 거래소가 그날 시작 가격으로 정해 주는 값. 보통 전날 종가와 같다.
- 권리락(무상증자 등)·감자·분할합병처럼 '가격이 바뀌는 게 정상인 일'이 있으면 거래소가 기준가를 바꿔 준다.
- 그래서 (기준가 / 전날 종가) 가 곧 '과거 가격에 곱해 줄 계수'다. 네이버가 이 날들의 90퍼센트 이상을 같은 방식으로 보정했다 (notebooks/02_외부자료비교 3절).

# 어떤 날을 쓰나
- 계수가 0.95 미만이거나 1.01 초과인 날만 쓴다. (0.95~0.99 는 배당락 등 작은 조정이라 배당 표에서 따로 다룬다)
- 분할로 이미 처리한 날(split 이벤트)은 뺀다. 같은 일을 두 번 보정하지 않기 위해서다.
"""
from __future__ import annotations

import pandas as pd

LOW, HIGH = 0.95, 1.01


def build_base_events(raw: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """raw: date, code, close, base_price. 돌려주는 표: code, ex_date, factor, 종류."""
    r = raw.sort_values(["code", "date"])[["date", "code", "close", "base_price"]].copy()
    r["pc"] = r.groupby("code")["close"].shift()
    r["factor"] = r["base_price"] / r["pc"]
    ev = r[(r["factor"] < LOW) | (r["factor"] > HIGH)].dropna(subset=["factor"])
    sd = set(zip(splits["code"], pd.to_datetime(splits["ex_date"])))
    ev = ev[[(c, t) not in sd for c, t in zip(ev["code"], ev["date"])]]
    return ev.rename(columns={"date": "ex_date"})[["code", "ex_date", "factor"]].reset_index(drop=True)


def dedupe_splits(splits: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """같은 종목·같은 권리락일에 분할 이벤트가 둘 이상이면(신주가 두 번에 나눠 상장되면 생긴다) 하나만 남긴다.

    그날 거래소 기준가/전일종가 와 가장 가까운 '1/비율' 을 가진 이벤트를 고른다. 둘 다 곱하면 두 번 보정되기 때문이다.
    (거래소 기준가가 없으면 가격이 이론과 가장 잘 맞는 price_dev 가 작은 쪽을 고른다.)
    """
    import numpy as np
    s = splits.copy()
    s["ex_date"] = pd.to_datetime(s["ex_date"])
    dup = s.duplicated(["code", "ex_date"], keep=False)
    if not dup.any():
        return s
    r = raw.sort_values(["code", "date"])[["date", "code", "close", "base_price"]].copy()
    r["pc"] = r.groupby("code")["close"].shift()
    obs = r.set_index(["code", "date"])
    keep = list(s.index[~dup])
    for (code, ex), g in s[dup].groupby(["code", "ex_date"]):
        o = np.nan
        if (code, ex) in obs.index:
            row = obs.loc[(code, ex)]
            o = row["base_price"] / row["pc"] if row["pc"] else np.nan
        if np.isfinite(o) and o > 0:
            best = (np.log(1.0 / g["share_ratio"].astype(float) / o)).abs().idxmin()
        else:
            best = g["price_dev"].astype(float).idxmin()
        keep.append(best)
    return s.loc[sorted(keep)].reset_index(drop=True)


BIG_DIV = 0.10          # 배당이 전일종가의 10퍼센트를 넘으면 '큰 배당'으로 보고 실제 가격이 떨어졌는지 확인한다


def drop_unsupported_big_dividends(divs: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """큰 배당(10퍼센트 초과) 중, 배당락일에 실제 가격이 이론 하락의 1/3 도 안 떨어진 것은 뺀다.

    왜: 이런 이벤트는 배당금 단위 오류나 배당락일 오류일 가능성이 높다. 그대로 두면 과거 가격이 통째로 틀어진다.
    (확인: 86건 중 71건에서 실제 하락이 거의 없었다.) 작은 배당은 건드리지 않는다.
    """
    if not len(divs):
        return divs
    r = raw.sort_values(["code", "date"])[["date", "code", "close"]].copy()
    r["ret"] = r.groupby("code")["close"].pct_change()
    d = divs.copy()
    d["ex_date"] = pd.to_datetime(d["ex_date"])
    d = d.merge(r.rename(columns={"date": "ex_date"})[["code", "ex_date", "ret"]], on=["code", "ex_date"], how="left")
    drop = 1.0 - d["factor"]
    bad = (drop > BIG_DIV) & (d["ret"] > -drop / 3)
    return d.loc[~bad].drop(columns="ret").reset_index(drop=True)
