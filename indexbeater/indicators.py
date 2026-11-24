"""기술적 지표. 모든 함수는 t일까지의 값만 쓴다 (center=True, shift(-n) 금지).  # lookahead-ok

입력·출력 모두 wide DataFrame (행=날짜, 열=종목).
계산식은 11/20 팀장 결정 5A (표준식)를 따른다:
  RSI·ATR·ADX·SAR = Wilder 방식, MACD = EMA, 스토캐스틱 %D = 3일 단순평균, CCI = 평균편차.
EMA·Wilder 평균은 '처음 n개 값의 단순평균'을 출발점으로 삼는다 (교과서 방식).

주의: shift(1) 은 '어제 값을 오늘 칸으로 옮겨 오는 것'(과거 보기)이라 써도 된다.
shift(-1) 은 '내일 값을 오늘 칸으로 당겨 오는 것'(미래 보기)이라 쓰면 안 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indexbeater import config as C


# ---------------------------------------------------------------- 평균 종류
def sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """단순 이동평균: 지난 n일(오늘 포함) 값을 더해서 n으로 나눈다. n일이 안 차면 빈칸."""
    return x.rolling(n, min_periods=n).mean()


def _seeded_ewm(x: pd.DataFrame, n: int, alpha: float) -> pd.DataFrame:
    """처음 n개 값의 단순평균에서 출발하는 지수평균 (EMA·Wilder 공통).

    지수평균 = 어제 평균 × (1 − alpha) + 오늘 값 × alpha.
    종목마다 상장일이 달라서, 종목마다 '처음 n개가 찬 날'을 따로 찾아 출발점으로 쓴다.
    """
    start = sma(x, n)
    has = start.notna().to_numpy()
    first = np.where(has.any(axis=0), np.argmax(has, axis=0), -1)   # 종목별 출발 행 번호
    rows = np.arange(len(x))[:, None]
    before = rows < first[None, :]                                    # 출발 전 칸
    y = x.to_numpy(dtype=float, copy=True)
    y[before] = np.nan
    cols = np.flatnonzero(first >= 0)
    y[first[cols], cols] = start.to_numpy()[first[cols], cols]        # 출발 칸 = 단순평균
    y[:, first < 0] = np.nan
    seeded = pd.DataFrame(y, index=x.index, columns=x.columns)
    # ewm(alpha, adjust=False) 는 위 '어제 평균 × (1 − alpha) + 오늘 값 × alpha' 를 그대로 계산한다.
    out = seeded.ewm(alpha=alpha, adjust=False).mean()
    return out.where(x.notna() & ~pd.DataFrame(before, index=x.index, columns=x.columns))


def ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """지수 이동평균 (MACD용). alpha = 2 / (n + 1)."""
    return _seeded_ewm(x, n, 2.0 / (n + 1))


def wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder 평균 (RSI·ATR·ADX용). alpha = 1 / n. 어제 평균에 오늘 값을 1/n 만큼 섞는다."""
    return _seeded_ewm(x, n, 1.0 / n)


def _hh(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """지난 n일(오늘 포함) 최고값."""
    return x.rolling(n, min_periods=n).max()


def _ll(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """지난 n일(오늘 포함) 최저값."""
    return x.rolling(n, min_periods=n).min()


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """b가 0인 칸은 나누지 않고 빈칸으로 둔다 (0으로 나누면 무한대가 되기 때문)."""
    return a / b.where(b != 0)


# ---------------------------------------------------------------- 추세형
def macd(close: pd.DataFrame, fast: int, slow: int, signal: int):
    """MACD 선 = 빠른 EMA − 느린 EMA, 시그널 선 = MACD 의 EMA (부록 A-03)."""
    line = ema(close, fast) - ema(close, slow)
    return line, ema(line, signal)


def true_range(high, low, close) -> pd.DataFrame:
    """하루 실제 움직임 = (고가−저가), |고가−어제 종가|, |저가−어제 종가| 중 가장 큰 것.
    어제 종가가 없는 첫날은 빈칸."""
    pc = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - pc).abs(), (low - pc).abs()))
    return tr.where(pc.notna())


def atr(high, low, close, n: int = C.ATR_PERIOD) -> pd.DataFrame:
    """ATR = 실제 움직임의 Wilder 평균 (부록 B-S2)."""
    return wilder(true_range(high, low, close), n)


def adx_di(high, low, close, n: int):
    """+DI, −DI, ADX (Wilder, 부록 A-05).

    +DM = 오늘 고가가 어제보다 오른 폭 (내린 폭보다 클 때만), −DM 은 반대.
    DI = DM 평균 ÷ ATR × 100, DX = |+DI − −DI| ÷ (+DI + −DI) × 100, ADX = DX 의 Wilder 평균.
    """
    up = high - high.shift(1)
    dn = low.shift(1) - low
    first = high.shift(1).isna()                      # 어제가 없는 첫날은 빈칸
    pdm = up.where((up > dn) & (up > 0), 0.0).mask(first)
    mdm = dn.where((dn > up) & (dn > 0), 0.0).mask(first)
    tr_s = wilder(true_range(high, low, close), n)
    pdi = 100 * _safe_div(wilder(pdm, n), tr_s)
    mdi = 100 * _safe_div(wilder(mdm, n), tr_s)
    s = pdi + mdi
    # +DI 와 −DI 가 둘 다 0이면 방향이 없다는 뜻이라 DX 를 0으로 둔다
    dx = (100 * (pdi - mdi).abs() / s.where(s != 0)).mask((s == 0) & pdi.notna(), 0.0)
    return pdi, mdi, wilder(dx, n)


def ichimoku(high, low, close, conv: int, base: int):
    """일목균형표의 전환선·기준선·구름 위·아래 (부록 A-04, 기준 5.2).

    구름(선행스팬)은 'base 일 전까지의 자료로 계산한 값을 base 일 앞으로 민 것'이다.
    shift(base) 는 과거 값을 오늘로 가져오는 것이라 미래 정보가 아니다.
    후행스팬은 미래 가격이 들어가서 쓰지 않는다.
    """
    tenkan = (_hh(high, conv) + _ll(low, conv)) / 2
    kijun = (_hh(high, base) + _ll(low, base)) / 2
    span_a = ((tenkan + kijun) / 2).shift(base)
    span_b = ((_hh(high, C.ICHIMOKU_SPAN_B) + _ll(low, C.ICHIMOKU_SPAN_B)) / 2).shift(base)
    top = np.maximum(span_a, span_b).where(span_a.notna() & span_b.notna())
    bottom = np.minimum(span_a, span_b).where(span_a.notna() & span_b.notna())
    return tenkan, kijun, top, bottom


def parabolic_sar(high, low, close, step: float, max_step: float):
    """Wilder 파라볼릭 SAR (부록 A-06). (SAR 값, 추세: 1 = 오름 / −1 = 내림) 을 돌려준다.

    하루씩 앞으로 가며 계산한다 (오늘 값은 어제까지의 상태와 오늘 가격으로만 정해짐).
    - 출발: 이틀 자료가 생긴 날, 오늘 종가 ≥ 어제 종가면 오름 추세로 시작.
      오름이면 SAR = 어제 저가, 극점(EP) = 오늘 고가. 내림이면 SAR = 어제 고가, EP = 오늘 저가.
    - 다음 날 SAR = 어제 SAR + 가속(af) × (EP − 어제 SAR).
      오름 추세면 SAR 이 지난 이틀 저가보다 높으면 안 된다 (내림이면 반대로 고가).
    - 가격이 SAR 을 뚫으면 추세를 뒤집고, SAR = 그동안의 EP, af 는 처음 값으로.
    - 새 극점이 나오면 af 를 step 만큼 올린다 (max_step 까지).
    """
    H = high.to_numpy(float)
    L = low.to_numpy(float)
    Cl = close.to_numpy(float)
    n, m = H.shape
    sar_out = np.full((n, m), np.nan)
    trend_out = np.full((n, m), np.nan)
    sar = np.full(m, np.nan)
    ep = np.full(m, np.nan)
    af = np.full(m, np.nan)
    up = np.zeros(m, dtype=bool)
    live = np.zeros(m, dtype=bool)          # 계산이 시작된 종목
    for i in range(1, n):
        valid = ~np.isnan(H[i]) & ~np.isnan(H[i - 1])
        live &= valid                       # 자료가 끊기면(상장폐지) 계산을 멈춘다
        # 1) 새로 출발하는 종목
        start = valid & ~live & ~np.isnan(Cl[i]) & ~np.isnan(Cl[i - 1])
        if start.any():
            u = Cl[i] >= Cl[i - 1]
            up[start] = u[start]
            sar[start] = np.where(u, L[i - 1], H[i - 1])[start]
            ep[start] = np.where(u, H[i], L[i])[start]
            af[start] = step
        # 2) 이미 계산 중인 종목
        go = live.copy()
        if go.any():
            prev2_ok = np.zeros(m, dtype=bool)
            if i >= 2:
                prev2_ok = ~np.isnan(H[i - 2])
            low2 = np.where(prev2_ok, np.fmin(L[i - 1], L[i - 2] if i >= 2 else L[i - 1]), L[i - 1])
            high2 = np.where(prev2_ok, np.fmax(H[i - 1], H[i - 2] if i >= 2 else H[i - 1]), H[i - 1])
            new = sar + af * (ep - sar)
            new = np.where(up, np.minimum(new, low2), np.maximum(new, high2))
            flip_dn = go & up & (L[i] < new)
            flip_up = go & ~up & (H[i] > new)
            flip = flip_dn | flip_up
            stay = go & ~flip
            # 추세가 뒤집히는 종목
            new = np.where(flip, ep, new)
            ep_new = np.where(flip_dn, L[i], np.where(flip_up, H[i], ep))
            af_new = np.where(flip, step, af)
            # 추세가 이어지는 종목: 새 극점이면 af 를 올린다
            better = stay & np.where(up, H[i] > ep, L[i] < ep)
            ep_new = np.where(better, np.where(up, H[i], L[i]), ep_new)
            af_new = np.where(better, np.minimum(af + step, max_step), af_new)
            sar[go] = new[go]
            ep[go] = ep_new[go]
            af[go] = af_new[go]
            up[flip] = ~up[flip]
        live |= start
        sar_out[i, live] = sar[live]
        trend_out[i, live] = np.where(up, 1.0, -1.0)[live]
    idx, cols = high.index, high.columns
    return pd.DataFrame(sar_out, idx, cols), pd.DataFrame(trend_out, idx, cols)


# ---------------------------------------------------------------- 오실레이터
def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """RSI (Wilder, 부록 A-07). 오른 폭 평균 ÷ (오른 폭 평균 + 내린 폭 평균) × 100.

    diff() 는 '오늘 값 − 어제 값'이다. 첫날은 어제가 없어서 빈칸.
    """
    d = close.diff()
    gain = d.clip(lower=0).where(d.notna())
    loss = (-d).clip(lower=0).where(d.notna())
    ag, al = wilder(gain, n), wilder(loss, n)
    return 100 * _safe_div(ag, ag + al)


def stochastic(high, low, close, k: int, d: int):
    """스토캐스틱 %K, %D (부록 A-08). %K = (종가 − k일 최저) ÷ (k일 최고 − k일 최저) × 100,
    %D = %K 의 d일 단순평균."""
    hh, ll = _hh(high, k), _ll(low, k)
    pk = 100 * _safe_div(close - ll, hh - ll)
    return pk, sma(pk, d)


def disparity(close: pd.DataFrame, ma: int) -> pd.DataFrame:
    """이격도 = 종가 ÷ ma일 이동평균 (부록 A-09). 0.95 면 평균보다 5% 아래."""
    return close / sma(close, ma)


def cci(high, low, close, n: int) -> pd.DataFrame:
    """CCI (부록 A-10). (TP − TP 평균) ÷ (0.015 × 평균편차), TP = (고가+저가+종가) ÷ 3.

    평균편차 = 지난 n일 각각의 TP 가 '오늘의 TP 평균'에서 얼마나 떨어졌는지의 평균.
    """
    tp = (high + low + close) / 3
    ma = sma(tp, n)
    md = sum((tp.shift(i) - ma).abs() for i in range(n)) / n
    return _safe_div(tp - ma, C.CCI_CONSTANT * md)


def williams_r(high, low, close, n: int) -> pd.DataFrame:
    """윌리엄스 %R (부록 A-11) = (n일 최고 − 종가) ÷ (n일 최고 − n일 최저) × (−100). 0 ~ −100."""
    hh, ll = _hh(high, n), _ll(low, n)
    return -100 * _safe_div(hh - close, hh - ll)


# ---------------------------------------------------------------- 변동성·거래량
def bollinger(close: pd.DataFrame, n: int, k: float):
    """볼린저 밴드 (부록 A-12·13): 중심 = n일 평균, 위·아래 = 중심 ± k × 표준편차(나누기 n)."""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=C.BOLL_DDOF)
    return mid - k * sd, mid, mid + k * sd


def obv(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """OBV (부록 A-17): 오른 날은 거래량을 더하고 내린 날은 뺀 누적합.

    cumsum 은 '처음부터 오늘까지 더한 값'이다. 과거만 더하므로 미래 정보가 없다.
    출발 값이 종목마다 달라도, OBV 와 OBV 이평을 비교할 때는 같은 값이 양쪽에 더해져서 결과가 같다.
    """
    sign = np.sign(close.diff()).fillna(0.0)
    step = (sign * volume.fillna(0.0)).where(close.notna())
    return step.cumsum().where(close.notna())
