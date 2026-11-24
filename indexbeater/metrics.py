"""신호 성적 지표 — 대조군·Edge·t값·카이제곱 (docs/프로젝트_기준.md 5.3·5.4).

엔진(engine.py)은 바꾸지 않고, 엔진이 만든 매매 표만 읽어서 계산한다.

대조군 (5.4, 11/24 팀장 결정 Q2·Q3)
- 대조군 수익 = 같은 엔진으로 '그날 유동성을 통과한 모든 종목'을 같은 H로 사고판 결과 (중복 무시 없이).
- 양쪽 모두 '체결'된 매매만 쓴다.
- 짝 칸 = 같은 신호일 T · 같은 시장 · 같은 시가총액 5분위 · 같은 거래대금(value20) 5분위.
  5분위는 T일마다, 시장마다, 유동성 통과 종목 안에서 순위로 나눈다.
- 그날 신호가 켜진 종목(중복_무시 포함)은 대조군에서 뺀다 → 칸 합계에서 신호 종목 몫을 빼서 바로 평균을 낸다.
  (랜덤으로 뽑는 것의 기댓값과 같으므로 뽑지 않고 평균으로 계산한다 — 5.4)
- 짝 칸에 대조 종목이 하나도 없으면 그 매매는 Edge 계산에서 빼고 수를 적는다.

비용은 신호·대조군 양쪽에서 똑같이 빼므로 Edge(차이)는 비용과 상관없이 같다. 승률은 비용마다 다르다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from indexbeater import config as C
from indexbeater import engine as E

MARKET_CODE = {mk: k for k, mk in enumerate(C.UNIVERSE_MARKETS)}   # KOSPI 0, KOSDAQ 1
Q = C.MATCH_QUANTILES                                               # 5
CELLS_PER_DAY = len(C.UNIVERSE_MARKETS) * Q * Q                     # 50
COSTS = {"0.5%": C.COST_SELECT, "0.25%": C.COST_ROUNDTRIP}


# ---------------------------------------------------------------- 짝 칸 (시장 × 시총 5분위 × 거래대금 5분위)
def quintile(values: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """날짜마다(가로 한 줄마다) mask 안 종목의 순위를 1~5로 나눈다. mask 밖은 0."""
    v = values.where(mask)
    pct = v.rank(axis=1, pct=True).to_numpy(float)       # 같은 값은 평균 순위
    q = np.ceil(pct * Q)
    q = np.where(np.isnan(q), 0, np.clip(q, 1, Q))
    return q.astype(np.int16)


def match_cells(data: dict) -> np.ndarray:
    """날짜×종목 칸마다 짝 칸 번호 (없으면 −1).

    번호 = T행 × 50 + 시장 × 25 + (시총분위−1) × 5 + (거래대금분위−1)
    """
    tradable = data["tradable"].to_numpy(bool)
    market = data["market"].to_numpy(object)
    n, m = tradable.shape
    cell = np.full((n, m), -1, dtype=np.int64)
    rows = np.arange(n)[:, None]
    for mk, k in MARKET_CODE.items():
        mask = tradable & (market == mk)
        mq = quintile(data["mktcap"], mask)
        vq = quintile(data["value20"], mask)
        ok = mask & (mq > 0) & (vq > 0)
        key = rows * CELLS_PER_DAY + k * Q * Q + (mq.astype(np.int64) - 1) * Q + (vq.astype(np.int64) - 1)
        cell = np.where(ok, key, cell)
    return cell


# ---------------------------------------------------------------- 대조군 바탕 (H마다 한 번)
class ControlBase:
    """유동성 통과 모든 종목-일을 같은 H로 사고판 결과를 칸별 합계로 들고 있는 상자."""

    def __init__(self, data: dict, rule, m: E._Market, cell: np.ndarray):
        self.hold = rule.hold
        shape = cell.shape
        allowed = data["tradable"].to_numpy(bool) & np.asarray(m.dates >= C.SIGNAL_START)[:, None]
        entry = pd.DataFrame(allowed, index=m.dates, columns=m.codes)
        tr = E.simulate_trades(entry, data, rule, dedupe=False, _m=m)
        f = tr[tr["status"] == E.FILLED]
        j = pd.Index(m.codes).get_indexer(f["code"])
        t = f["t_idx"].to_numpy(int)
        self.gross = np.full(shape, np.nan)
        self.gross[t, j] = f["ret_gross"].to_numpy(float)
        self.delist = np.zeros(shape, bool)
        self.delist[t, j] = f["reason"].eq(E.DELIST).to_numpy()
        self.cell = cell
        self.n_cells = shape[0] * CELLS_PER_DAY
        self.n_base_trades = len(f)
        self._tot = {w: self._sums(self.variant(w), np.ones(shape, bool)) for w in (False, True)}

    def variant(self, worst: bool) -> np.ndarray:
        """worst=True 면 상장폐지 정산을 −100%로 바꾼 수익 (engine.delisting_worst_case 와 같은 규칙)."""
        if not worst:
            return self.gross
        return np.where(self.delist, -1.0, self.gross)

    def _sums(self, g: np.ndarray, where: np.ndarray):
        ok = where & ~np.isnan(g) & (self.cell >= 0)
        k = self.cell[ok]
        gv = g[ok]
        out = {"n": np.bincount(k, minlength=self.n_cells).astype(float),
               "sum": np.bincount(k, weights=gv, minlength=self.n_cells)}
        for name, c in COSTS.items():
            out[f"win{name}"] = np.bincount(k, weights=(gv - c > 0).astype(float), minlength=self.n_cells)
        return out

    def matched(self, sig_on: np.ndarray, t: np.ndarray, j: np.ndarray, worst: bool = False) -> dict:
        """신호 매매 (t, j) 마다 짝 칸 대조군 평균 수익(비용 전)과 대조군 승률(비용별).

        sig_on : 그날 신호가 켜진 칸 (유동성·날짜 통과, 중복_무시 포함) → 대조군에서 뺀다
        """
        tot = self._tot[worst]
        sub = self._sums(self.variant(worst), sig_on)
        k = self.cell[t, j]
        has = k >= 0
        kk = np.where(has, k, 0)
        n = tot["n"][kk] - sub["n"][kk]
        has &= n > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            out = {"ctrl_n": np.where(has, n, 0.0),
                   "ctrl_gross": np.where(has, (tot["sum"][kk] - sub["sum"][kk]) / n, np.nan)}
            for name in COSTS:
                w = tot[f"win{name}"][kk] - sub[f"win{name}"][kk]
                out[f"ctrl_win{name}"] = np.where(has, w / n, np.nan)
        return out


# ---------------------------------------------------------------- 통계
def block_t(daily_edge: pd.Series, calendar: pd.DatetimeIndex, block: int = C.BLOCK_DAYS) -> float:
    """일별 평균 Edge → 20거래일씩 겹치지 않는 블록 평균 → 블록 평균들의 t값 (5.4).

    - 블록은 calendar(그 기간의 거래일 달력) 첫날부터 20거래일씩 자른다.
    - 블록 평균 = 그 블록 안에서 매매가 있는 날들의 일별 평균 Edge 의 평균. 매매가 없는 블록은 뺀다.
    - t = 블록 평균의 평균 ÷ (표준편차 ÷ √블록 수). 블록이 2개 미만이거나 표준편차 0이면 NaN.
    """
    if daily_edge.empty:
        return float("nan")
    pos = calendar.get_indexer(daily_edge.index)
    if (pos < 0).any():
        raise ValueError("일별 Edge 날짜가 달력에 없습니다")
    b = pd.Series(daily_edge.to_numpy(float)).groupby(pos // block).mean()
    if len(b) < 2:
        return float("nan")
    sd = b.std(ddof=1)
    if not sd > 0:
        return float("nan")
    return float(b.mean() / (sd / math.sqrt(len(b))))


def chi_square(win: np.ndarray, p_ctrl: np.ndarray) -> tuple[float, float]:
    """신호 승률 vs 짝 대조군 승률 (5.3). 자유도 1.

    관측 = 이긴 수·진 수, 기대 = 매매마다 짝 칸 대조군 승률을 더한 값.
    """
    n = len(win)
    if n == 0:
        return float("nan"), float("nan")
    obs_w = float(np.sum(win))
    exp_w = float(np.sum(p_ctrl))
    exp_l = n - exp_w
    if exp_w <= 0 or exp_l <= 0:
        return float("nan"), float("nan")
    chi = (obs_w - exp_w) ** 2 / exp_w + ((n - obs_w) - exp_l) ** 2 / exp_l
    return float(chi), float(stats.chi2.sf(chi, 1))


def win_loss(ret: np.ndarray) -> dict:
    """승률·평균 이익·평균 손실(크기)·손익비. 이긴 거래 = 비용 뺀 수익 > 0, 진 거래 = 0 이하 (5.17)."""
    ret = np.asarray(ret, float)
    n = len(ret)
    w = ret[ret > 0]
    lo = ret[ret <= 0]
    avg_win = float(w.mean()) if len(w) else 0.0
    avg_loss = float(-lo.mean()) if len(lo) else 0.0
    if avg_loss > 0:
        payoff = avg_win / avg_loss
    else:
        payoff = math.inf if avg_win > 0 else 0.0
    return {"승률": len(w) / n if n else float("nan"), "평균이익": avg_win,
            "평균손실": avg_loss, "손익비": payoff}


def top5_share(code: pd.Series, ret: pd.Series) -> float:
    """상위 5종목 수익 기여 = 종목별 순수익 합 상위 5개의 합 ÷ 전체 순수익 합 (전체 합 > 0 일 때만)."""
    total = float(ret.sum())
    if not total > 0:
        return float("nan")
    by = ret.groupby(code.to_numpy()).sum().sort_values(ascending=False)
    return float(by.head(5).sum() / total)


def unfillable_ratio(status: pd.Series) -> float:
    """체결 불가 비율 = (취소_갭 + 매수불가) ÷ (체결 + 취소_갭 + 매수불가). 중복_무시·미완료는 세지 않는다."""
    vc = status.value_counts()
    bad = vc.get(E.GAP_CANCEL, 0) + vc.get(E.NO_BUY, 0)
    den = bad + vc.get(E.FILLED, 0)
    return float(bad / den) if den else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    """최대 낙폭 (음수, 예: −0.25 = −25%)."""
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1).min())
