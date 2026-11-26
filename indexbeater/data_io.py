"""데이터 입출력: long(한 줄 = 하루·한 종목) → wide(행 = 날짜, 열 = 종목) 변환.

wide 데이터는 dict[str, DataFrame] 이고 키는 다음과 같다 (11/20 결정 1A: 파일 열 이름 그대로).
  adj_open adj_high adj_low adj_close  — 분할·배당·권리락·기준가 보정 가격 (신호·수익률 계산용, 기준 2장)
  value                                — 거래대금 (16 거래량 돌파, 유동성 필터)
  volume_s                             — 분할 보정 거래량 (17 OBV 전용, 기준 5.2)
  open close volume                    — 원래 값 (유동성 필터·거래정지 판단용, 기준 5.1-1)
  mktcap                               — 시가총액 (대조군 5분위용, 기준 5.4)
  market                               — 그날 시장 이름 (KOSPI·KOSDAQ). 유니버스 밖이면 빈칸
  value20                              — 그날까지 20일 평균 거래대금 (매수 순서, 기준 3)
  tradable                             — 그날 유동성 필터를 통과했나 (그날까지의 자료로만 계산)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indexbeater import config as C

PRICE_COLS = ["adj_open", "adj_high", "adj_low", "adj_close"]
WIDE_COLS = PRICE_COLS + ["value", "volume_s", "open", "close", "volume", "mktcap"]


def load_long(end=C.DESIGN_END, allow_test: bool = False, path=C.KRX_ADJUSTED) -> pd.DataFrame:
    """krx_adjusted.parquet 을 end 날짜까지만 읽는다.

    시험 구간(2020-01 이후)은 allow_test=True 없이 열지 않는다 (기준 4장, 계획서 4.4).
    """
    end = pd.Timestamp(end)
    if end >= C.TEST_START and not allow_test:
        # 실수로 시험 구간을 보는 일을 막는 잠금장치다
        raise ValueError(f"{end.date()} 는 시험 구간이다. 열려면 allow_test=True 가 필요하다 (기준 4장)")
    cols = ["date", "market", "code", "name"] + WIDE_COLS
    # filters 는 '파일을 읽을 때 이 조건에 맞는 줄만 꺼내라'는 뜻이다.
    # 그래서 2020년 이후 줄은 메모리에 아예 올라오지 않는다.
    df = pd.read_parquet(path, columns=cols, filters=[("date", "<=", end)])
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def universe_mask(df: pd.DataFrame) -> pd.Series:
    """long 표의 줄마다 유니버스에 들어가는지 True/False (11/20 결정 2A).

    - 그날 시장이 KOSPI·KOSDAQ 인 줄만 (ETF 빼기)
    - 우선주 빼기: 종목코드 끝자리가 0이 아니면 우선주다 (예: 005935 삼성전자우)
    - 스팩 빼기: 한 번이라도 이름에 '스팩'이 들어간 종목코드는 전부 뺀다
    """
    in_market = df["market"].isin(C.UNIVERSE_MARKETS)
    common = df["code"].str[-1] == C.PREFERRED_LAST_DIGIT
    spac_codes = df.loc[df["name"].str.contains(C.SPAC_NAME_KEYWORD, na=False), "code"].unique()
    not_spac = ~df["code"].isin(spac_codes)
    return in_market & common & not_spac


def to_wide(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """long 표를 wide 표 묶음으로 바꾼다. 유니버스 밖의 줄은 미리 버린다."""
    df = df[universe_mask(df)]
    # pivot 은 '날짜를 행, 종목코드를 열로 세운 표'를 만드는 명령이다 (엑셀 피벗과 같음).
    # 모든 날짜를 행으로 쓰기 위해 전체 거래일 목록을 먼저 만든다.
    dates = pd.DatetimeIndex(sorted(df["date"].unique()), name="date")
    codes = pd.Index(sorted(df["code"].unique()), name="code")
    out = {}
    for col in WIDE_COLS:
        out[col] = df.pivot(index="date", columns="code", values=col).reindex(index=dates, columns=codes)
    out["market"] = df.pivot(index="date", columns="code", values="market").reindex(index=dates, columns=codes)

    # 거래정지일(거래량 0)은 시가·고가·저가가 0으로 들어 있다 (95,585줄, 종가는 전날 종가).
    # 0원을 그대로 두면 '−100% 폭락'처럼 계산되므로, 그날은 시가=고가=저가=종가인 '평평한 날'로 둔다.
    # (줄은 지우지 않는다. 날짜 칸 수를 세는 보유기간 H가 달라지지 않게 하기 위해서다)
    # mask(조건, 값) 은 '조건이 True 인 칸만 값으로 바꿔라'라는 명령이다.
    halt = out["volume"].eq(0)
    out["open"] = out["open"].mask(halt & out["open"].eq(0), out["close"])
    for adj in ("adj_open", "adj_high", "adj_low"):
        out[adj] = out[adj].mask(halt & out[adj].eq(0), out["adj_close"])

    # 20일 평균 거래대금: 그날 포함 지난 20거래일 평균. 20일이 안 차면 빈칸 (기준 2장 유동성 필터).
    # rolling(20) 은 '20칸짜리 창문을 한 칸씩 밀면서' 계산하는 것이다. 창문은 항상 과거 쪽만 본다.
    out["value20"] = out["value"].rolling(C.VALUE_AVG_DAYS, min_periods=C.VALUE_AVG_DAYS).mean()
    out["tradable"] = tradable_mask(out)
    return out


def tradable_mask(w: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """그날 유동성 필터를 통과했나 (기준 2장·5.1-1). 모두 그날까지의 값만 쓴다.

    - 유니버스 안 (market 칸이 비어 있지 않음)
    - 그날 거래가 있었음 (거래량 > 0, 거래정지일 제외)
    - 원래 종가 ≥ 1,000원
    - 20일 평균 거래대금 ≥ 10억 원
    """
    ok = w["market"].notna()
    ok &= w["volume"].gt(0)
    ok &= w["close"].ge(C.MIN_PRICE)
    ok &= w["value20"].ge(C.MIN_VALUE_20D)
    return ok


def load_wide(end=C.DESIGN_END, allow_test: bool = False) -> dict[str, pd.DataFrame]:
    """설계 구간(기본 2019-12-31까지) wide 데이터 묶음을 만든다."""
    return to_wide(load_long(end=end, allow_test=allow_test))


def last_valid_index(frame: pd.DataFrame) -> np.ndarray:
    """종목마다 자료가 있는 마지막 행 번호. 자료가 하나도 없으면 −1."""
    has = frame.notna().to_numpy()
    n = has.shape[0]
    # 뒤집어서 처음 True 가 나오는 곳을 찾으면 '마지막 True' 위치를 알 수 있다
    last = n - 1 - np.argmax(has[::-1], axis=0)
    last[~has.any(axis=0)] = -1
    return last
