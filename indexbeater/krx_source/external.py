"""네이버·야후 파일을 읽는 파일. 비교할 때 '종목코드 하나 → 날짜별 가격표'로 바로 꺼내 쓰게 해 준다.

# 두 자료의 성격 (삼성전자로 확인함)
- 네이버: 분할만 보정된 가격. 배당은 보정하지 않는다. → 우리 `close_s`(분할만 보정)와 비교한다.
- 야후  : 분할 + 배당이 모두 보정된 가격. → 우리 `adj_close`(분할+배당 보정)와 비교한다.
"""
from __future__ import annotations

import functools

import pandas as pd

from . import paths


def load_naver(code: str) -> pd.DataFrame | None:
    """네이버 파일 1개를 읽는다. 파일이 없으면 None (= 네이버에 없는 종목)."""
    f = paths.NAVER_DIR / f"{code}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["Date"], dtype={"Code": str})   # utf-8-sig: 맨 앞 보이지 않는 표시(BOM)를 없앤다
    return d.rename(columns=str.lower).drop(columns="code").set_index("date")


@functools.lru_cache(maxsize=1)
def yahoo_index() -> dict[str, "pathlib.Path"]:   # noqa: F821
    """야후 파일 이름은 '005930.KS_삼성전자.csv' 라서, 앞의 6자리 코드로 파일을 찾을 수 있게 목록을 만든다."""
    return {p.name[:6]: p for p in paths.YAHOO_DIR.glob("*.csv")}


def load_yahoo(code: str) -> pd.DataFrame | None:
    """야후 파일 1개를 읽는다. 없으면 None."""
    p = yahoo_index().get(code)
    if p is None:
        return None
    d = pd.read_csv(p)
    d["Date"] = pd.to_datetime(d["Date"].str[:10])      # 시간대 표시(+09:00)가 붙어 있어도 날짜만 쓴다
    return d.rename(columns=str.lower).rename(columns={"stock splits": "splits"}).set_index("date")


@functools.lru_cache(maxsize=1)
def yf_index() -> dict[str, "pathlib.Path"]:   # noqa: F821
    """새 야후 폴더(yf)의 파일 목록. 파일 이름 맨 앞 6자리가 종목코드다. (fail_list.csv 같은 파일은 뺀다. 코드에는 0001A0 처럼 영문자도 섞여 있다)"""
    return {p.name[:6]: p for p in paths.YF_DIR.glob("*.csv") if p.name[6:7] == "."}   # 6자리 코드 바로 뒤에 점(.KS/.KQ)이 오는 파일만


def load_yf(code: str) -> pd.DataFrame | None:
    """새 야후 파일 1개. Close=분할만 보정(우리 close_s 와 비교), Adj Close=분할+배당 보정(adj_close 와 비교). 없으면 None."""
    p = yf_index().get(code)
    if p is None:
        return None
    d = pd.read_csv(p, encoding="utf-8-sig")
    d["Date"] = pd.to_datetime(d["Date"].astype(str).str[:10])
    return d.rename(columns=lambda c: c.lower().replace(" ", "_")).set_index("date")
