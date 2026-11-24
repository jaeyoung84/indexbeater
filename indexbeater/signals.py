"""매수 신호 18개 (계획서 부록 A, 기준 5.2).

형식 (모든 신호 공통)
- 입력: wide 데이터 dict (KRX `adj_open·adj_high·adj_low·adj_close` = 분할·배당·권리락·기준가 보정 가격,
  거래량 대신 `value`(거래대금). 17번 OBV만 `volume_s`(분할 보정 거래량)) — 11/20 결정 1A
- 출력: entry — 날짜×종목 True/False 표. entry[t] = t일 종가까지 보고 "내일 시가에 사라"
- 모든 매수 신호는 이벤트형: 조건이 "처음 성립한 날"만 True (기준 5.2)
  · 이벤트 = 오늘 조건이 참이고 어제 조건이 거짓 (어제 값이 계산되지 않았으면 켜지 않음)
  · 교차(상향 돌파) = 오늘은 위, 어제는 같거나 아래
- 파라미터 후보는 config.PARAM_GRID 에서만 가져온다 (결과를 보고 넓히지 않는다)
- 4번 일목은 후행스팬을 쓰지 않는다 (미래 가격이 들어가기 때문)
- 15번 변동성 돌파는 일봉이라 '종가 ≥ 시가 + k×전일 범위'로 바꿨다 (원래 규칙과 다름, 기준 5.2)
- 신호는 2012-07-01(SIGNAL_START)부터 켠다 (기준 5.1-2). 유동성 필터는 엔진이 t일 값으로 건다.
- 부록 A의 '매도 조건' 열은 아직 만들지 않았다 (11/20 결정 7A: STEP 4 전에 정의 확인 후 구현)

11/20 결정 6A로 정한 세부 정의는 각 함수 설명에 적었다.
새 신호를 추가하면 SIGNALS에 등록하고 `pytest tests/test_lookahead.py` 를 통과해야 한다.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from indexbeater import config as C
from indexbeater import indicators as I


# ---------------------------------------------------------------- 공통 도구
def event(cond: pd.DataFrame) -> pd.DataFrame:
    """조건이 '처음 성립한 날'만 True (기준 5.2 이벤트형).

    cond 는 1.0(참) / 0.0(거짓) / 빈칸(계산 불가) 표다.
    오늘 1이고 어제 0인 칸만 True. 어제가 빈칸이면 '처음'인지 알 수 없어서 켜지 않는다.
    """
    return cond.eq(1.0) & cond.shift(1).eq(0.0)


def as_cond(mask: pd.DataFrame, *inputs: pd.DataFrame) -> pd.DataFrame:
    """True/False 표를 1/0 표로 바꾸고, 재료(inputs) 중 하나라도 빈칸인 날은 빈칸으로 둔다."""
    ok = None
    for x in inputs:
        ok = x.notna() if ok is None else ok & x.notna()
    return mask.astype(float).where(ok)


def cross_up(a: pd.DataFrame, b) -> pd.DataFrame:
    """a 가 b 를 아래에서 위로 넘은 날: 오늘 a > b, 어제 a ≤ b (계획서 STEP 3 예시 코드)."""
    b_prev = b.shift(1) if isinstance(b, pd.DataFrame) else b
    return (a > b) & (a.shift(1) <= b_prev)


def _o(d):
    return d["adj_open"]


def _h(d):
    return d["adj_high"]


def _l(d):
    return d["adj_low"]


def _c(d):
    return d["adj_close"]


# ---------------------------------------------------------------- 1~6 추세형
def sig_ma_cross(d, short, long):
    """A-01 골든크로스: 단기 이평이 장기 이평을 상향 돌파한 날."""
    return cross_up(I.sma(_c(d), short), I.sma(_c(d), long))


def sig_alignment(d, periods):
    """A-02 정배열 시작 (6A): 단기 > 중기 > 장기가 처음 성립한 날."""
    a, b, c = (I.sma(_c(d), p) for p in periods)
    return event(as_cond((a > b) & (b > c), a, b, c))


def sig_macd(d, fast, slow, signal):
    """A-03 MACD 가 시그널을 상향 돌파한 날, 그날 MACD 가 0 아래."""
    line, sig = I.macd(_c(d), fast, slow, signal)
    return cross_up(line, sig) & (line < C.MACD_ZERO)


def sig_ichimoku(d, conv, base):
    """A-04 (6A): 전환선 > 기준선 이면서 종가 > 구름 위가 처음 성립한 날.
    구름은 t−26일까지 자료로 계산해 앞으로 민 값 (후행스팬 안 씀, 기준 5.2)."""
    tenkan, kijun, top, _ = I.ichimoku(_h(d), _l(d), _c(d), conv, base)
    return event(as_cond((tenkan > kijun) & (_c(d) > top), tenkan, kijun, top))


def sig_adx_di(d, period, adx_min):
    """A-05 +DI 가 −DI 를 상향 돌파한 날, 그날 ADX > 기준."""
    pdi, mdi, adx = I.adx_di(_h(d), _l(d), _c(d), period)
    return cross_up(pdi, mdi) & (adx > adx_min)


def sig_sar(d, step, max_step):
    """A-06 SAR 이 가격 아래로 전환한 날 (내림 추세 → 오름 추세로 바뀐 날)."""
    _, trend = I.parabolic_sar(_h(d), _l(d), _c(d), step, max_step)
    return trend.eq(1.0) & trend.shift(1).eq(-1.0)


# ---------------------------------------------------------------- 7~11 오실레이터
def sig_rsi(d, period, low):
    """A-07 RSI 가 하단(low)을 아래에서 위로 넘은 날 (계획서 STEP 3 예시 코드 그대로)."""
    r = I.rsi(_c(d), period)
    return cross_up(r, low)


def sig_stochastic(d, k, d_, low, high=None):
    """A-08 (6A): %K 가 %D 를 상향 돌파한 날, 그날 %K ≤ low(20). high 는 매도용(STEP 4)."""
    pk, pd_ = I.stochastic(_h(d), _l(d), _c(d), k, d_)
    return cross_up(pk, pd_) & (pk <= low)


def sig_disparity(d, ma, thr):
    """A-09 종가 ÷ 20일선 ≤ 기준(예: 95%)이 처음 성립한 날."""
    r = I.disparity(_c(d), ma)
    return event(as_cond(r <= thr, r))


def sig_cci(d, period):
    """A-10 CCI 가 −100 을 상향 돌파한 날."""
    return cross_up(I.cci(_h(d), _l(d), _c(d), period), C.CCI_BUY_LEVEL)


def sig_williams_r(d, period):
    """A-11 윌리엄스 %R 이 −80 을 상향 돌파한 날."""
    return cross_up(I.williams_r(_h(d), _l(d), _c(d), period), C.WILLR_BUY_LEVEL)


# ---------------------------------------------------------------- 12~15 변동성·돌파
def sig_bollinger(d, period, k):
    """A-12 (6A): 어제 종가 < 하단, 오늘 종가 ≥ 하단 (하단 이탈 후 밴드 안으로 복귀한 날)."""
    lower, _, _ = I.bollinger(_c(d), period, k)
    c = _c(d)
    return (c.shift(1) < lower.shift(1)) & (c >= lower)


def sig_squeeze(d, period, lookback):
    """A-13 (6A): 어제 밴드폭이 최근 lookback 일 중 최저, 오늘 종가가 상단을 상향 돌파 (k=2).
    밴드폭 = (상단 − 하단) ÷ 중심."""
    lower, mid, upper = I.bollinger(_c(d), period, C.SQUEEZE_K)
    bw = (upper - lower) / mid
    was_min = bw.shift(1) <= bw.shift(1).rolling(lookback, min_periods=lookback).min()
    return was_min & cross_up(_c(d), upper)


def _new_high(d, n):
    """오늘 종가 > 어제까지 n일 고가의 최고값 (6A, 오늘 고가는 비교 대상에 넣지 않음)."""
    hh = _h(d).rolling(n, min_periods=n).max().shift(1)
    return _c(d) > hh, hh


def sig_donchian(d, n):
    """A-14 N일 신고가 돌파가 처음 성립한 날."""
    brk, hh = _new_high(d, n)
    return event(as_cond(brk, hh, _c(d)))


def sig_vol_breakout(d, k):
    """A-15 종가 ≥ 시가 + k × 전일 범위(전일 고가 − 저가)가 처음 성립한 날 (일봉용으로 바꾼 규칙)."""
    rng = (_h(d) - _l(d)).shift(1)
    lvl = _o(d) + k * rng
    return event(as_cond(_c(d) >= lvl, lvl, _c(d)))


def sig_volume_breakout(d, n, m):
    """A-16 (6A): N일 신고가 돌파 + 거래대금 ≥ m × 어제까지 20일 평균 거래대금이 처음 성립한 날."""
    brk, hh = _new_high(d, n)
    avg = d["value"].rolling(C.VALUE_AVG_DAYS, min_periods=C.VALUE_AVG_DAYS).mean().shift(1)
    return event(as_cond(brk & (d["value"] >= m * avg), hh, avg, _c(d)))


# ---------------------------------------------------------------- 16~18 거래량·패턴
def sig_obv(d, ma):
    """A-17 OBV 가 OBV 이평을 상향 돌파한 날 (volume_s 사용, 기준가 이벤트 미보정 한계 있음)."""
    o = I.obv(_c(d), d["volume_s"])
    return cross_up(o, I.sma(o, ma))


def sig_candle(d, trend, tail_ratio):
    """A-18 (6A): 하락 추세에서 망치형 또는 상승장악형이 처음 나온 날.

    - 하락 추세: 오늘 종가 < trend 일 전 종가
    - 망치형: 아래꼬리 ≥ tail_ratio × 몸통, 위꼬리 ≤ 몸통
    - 상승장악형: 어제 음봉, 오늘 양봉, 오늘 몸통이 어제 몸통을 감쌈
    - 고가 = 저가인 날(거래정지일 등 움직임이 없는 날)은 캔들 모양이 없어서 넣지 않는다
    """
    o, h, l, c = _o(d), _h(d), _l(d), _c(d)
    body = (c - o).abs()
    lower_tail = np.minimum(o, c) - l
    upper_tail = h - np.maximum(o, c)
    hammer = (lower_tail >= tail_ratio * body) & (upper_tail <= C.CANDLE_UPPER_MAX * body)
    po, pc = o.shift(1), c.shift(1)
    engulf = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    down = c < c.shift(trend)
    cond = down & (hammer | engulf) & (h > l)
    return event(as_cond(cond, c.shift(trend), po, pc, o, h, l, c))


# ---------------------------------------------------------------- 등록표
SIGNALS = {   # 키는 config.PARAM_GRID 와 같다. 번호는 부록 A 순서.
    "ma_cross": sig_ma_cross,               # 1
    "alignment": sig_alignment,             # 2
    "macd": sig_macd,                       # 3
    "ichimoku": sig_ichimoku,               # 4
    "adx_di": sig_adx_di,                   # 5
    "sar": sig_sar,                         # 6
    "rsi": sig_rsi,                         # 7
    "stochastic": sig_stochastic,           # 8
    "disparity": sig_disparity,             # 9
    "cci": sig_cci,                         # 10
    "williams_r": sig_williams_r,           # 11
    "bollinger": sig_bollinger,             # 12
    "squeeze": sig_squeeze,                 # 13
    "donchian": sig_donchian,               # 14
    "vol_breakout": sig_vol_breakout,       # 15
    "volume_breakout": sig_volume_breakout,  # 16
    "obv": sig_obv,                         # 17
    "candle": sig_candle,                   # 18
}


def iter_params(name: str):
    """PARAM_GRID 의 모든 짝을 하나씩 돌려준다 (ma_cross 는 단기 < 장기만, 기준 5.3)."""
    grid = C.PARAM_GRID[name]
    for combo in product(*grid.values()):
        p = dict(zip(grid.keys(), combo))
        if name == "ma_cross" and p["short"] >= p["long"]:
            continue
        yield p


def raw_signal(name: str, data: dict, params: dict) -> pd.DataFrame:
    """날짜 자르기 전의 신호 표 (미래 정보 테스트용. 매매에는 entry_signal 을 쓴다)."""
    # 스토캐스틱의 d 는 데이터 묶음 이름(d)과 겹쳐서 함수 안에서는 d_ 로 받는다
    p = dict(params)
    if name == "stochastic":
        p["d_"] = p.pop("d")
    out = SIGNALS[name](data, **p)
    # 빈칸 비교 결과는 False 로 보고, True/False 표로 통일한다
    return out.fillna(False).astype(bool)


def entry_signal(name: str, data: dict, params: dict) -> pd.DataFrame:
    """신호 하나의 매수 표. 2012-07-01(SIGNAL_START) 이전은 모두 False (기준 5.1-2)."""
    e = raw_signal(name, data, params)
    keep = np.asarray(e.index >= C.SIGNAL_START)
    return e & keep[:, None]
