"""점수표 계산 부품(metrics.py)이 규칙대로 되는지 작은 손 계산 예제로 확인한다 (기준 5.3·5.4, 11/24 결정 D2·D3·D7)."""
import math

import numpy as np
import pandas as pd
import pytest

from indexbeater import engine as E
from indexbeater import metrics as M


def test_quintile_rank_per_day_and_mask():
    v = pd.DataFrame([list(range(1, 11))], dtype=float)
    mask = np.ones((1, 10), bool)
    assert M.quintile(v, mask).tolist() == [[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]
    mask[0, 9] = False                                    # 빠진 종목은 0, 나머지 9개로 다시 나눔
    q = M.quintile(v, mask)[0]
    assert q[9] == 0 and q[0] == 1 and q[8] == 5


def test_quintile_ties_get_same_bucket():
    v = pd.DataFrame([[5.0, 5.0, 5.0, 5.0, 5.0]])
    q = M.quintile(v, np.ones((1, 5), bool))[0]
    assert len(set(q)) == 1                               # 같은 값 = 평균 순위 = 같은 분위


def _toy_base(gross, cell, delist=None):
    """엔진을 돌리지 않고 대조군 바탕 상자만 손으로 만든다."""
    b = M.ControlBase.__new__(M.ControlBase)
    b.gross = np.asarray(gross, float)
    b.delist = np.zeros(b.gross.shape, bool) if delist is None else np.asarray(delist, bool)
    b.cell = np.asarray(cell, np.int64)
    b.n_cells = b.gross.shape[0] * M.CELLS_PER_DAY
    b._tot = {w: b._sums(b.variant(w), np.ones(b.gross.shape, bool)) for w in (False, True)}
    return b


def test_matched_excludes_signal_stocks():
    # 하루, 종목 4개. 0·1·2 는 같은 칸(7), 3 은 다른 칸(9)
    b = _toy_base([[0.10, 0.20, 0.30, 0.50]], [[7, 7, 7, 9]])
    sig_on = np.array([[True, False, False, False]])
    out = b.matched(sig_on, np.array([0]), np.array([0]))
    assert out["ctrl_n"][0] == 2
    assert out["ctrl_gross"][0] == pytest.approx(0.25)    # 신호 종목(0.10)은 빼고 0.20·0.30 평균
    # 승률(비용 0.5%): 0.20·0.30 둘 다 이김 → 1.0
    assert out["ctrl_win0.5%"][0] == pytest.approx(1.0)


def test_matched_signal_on_also_removes_duplicate_ignored():
    # 1번 종목도 그날 신호가 켜졌으면(중복_무시여도) 대조군에서 빠진다
    b = _toy_base([[0.10, 0.20, 0.30]], [[7, 7, 7]])
    sig_on = np.array([[True, True, False]])
    out = b.matched(sig_on, np.array([0]), np.array([0]))
    assert out["ctrl_n"][0] == 1 and out["ctrl_gross"][0] == pytest.approx(0.30)


def test_matched_no_control_gives_nan():
    b = _toy_base([[0.10, 0.50]], [[7, 9]])
    out = b.matched(np.array([[True, False]]), np.array([0]), np.array([0]))
    assert out["ctrl_n"][0] == 0 and math.isnan(out["ctrl_gross"][0])


def test_matched_worst_case_uses_minus_100():
    b = _toy_base([[0.10, 0.20, 0.30]], [[7, 7, 7]], delist=[[False, True, False]])
    sig = np.array([[True, False, False]])
    assert b.matched(sig, np.array([0]), np.array([0]))["ctrl_gross"][0] == pytest.approx(0.25)
    assert b.matched(sig, np.array([0]), np.array([0]), worst=True)["ctrl_gross"][0] == pytest.approx(-0.35)


def test_block_t_hand_calc():
    cal = pd.bdate_range("2015-01-01", periods=60)
    daily = pd.Series([0.01] * 20 + [0.03] * 20 + [0.02] * 20, index=cal)
    # 블록 평균 0.01, 0.03, 0.02 → 평균 0.02, 표준편차 0.01 → t = 0.02 / (0.01/√3)
    assert M.block_t(daily, cal) == pytest.approx(0.02 / (0.01 / math.sqrt(3)))


def test_block_t_skips_empty_blocks_and_needs_two():
    cal = pd.bdate_range("2015-01-01", periods=60)
    one = pd.Series([0.01, 0.02], index=cal[[0, 5]])      # 블록 1개뿐
    assert math.isnan(M.block_t(one, cal))
    two = pd.Series([0.01, 0.03], index=cal[[0, 45]])     # 가운데 블록은 매매 없음 → 뺌
    assert M.block_t(two, cal) == pytest.approx(0.02 / (math.sqrt(2) * 0.01 / math.sqrt(2)))


def test_chi_square_zero_when_as_expected():
    chi, p = M.chi_square(np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert chi == pytest.approx(0.0) and p == pytest.approx(1.0)


def test_chi_square_hand_calc():
    win = np.array([1] * 70 + [0] * 30)
    chi, p = M.chi_square(win, np.full(100, 0.5))
    assert chi == pytest.approx((70 - 50) ** 2 / 50 * 2)
    assert p < 0.001


def test_win_loss():
    r = M.win_loss(np.array([0.10, -0.05, 0.0, 0.20]))
    assert r["승률"] == 0.5                                # 0 은 진 거래
    assert r["평균이익"] == pytest.approx(0.15)
    assert r["평균손실"] == pytest.approx(0.025)
    assert r["손익비"] == pytest.approx(6.0)


def test_top5_share_and_unfillable_and_mdd():
    code = pd.Series(list("ABCDEFG"))
    ret = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1, -0.2, -0.3])
    assert M.top5_share(code, ret) == pytest.approx(1.5 / 1.0)
    assert math.isnan(M.top5_share(code, -ret.abs()))
    st = pd.Series([E.FILLED] * 8 + [E.GAP_CANCEL, E.NO_BUY, E.DUPLICATE, E.OPEN_END])
    assert M.unfillable_ratio(st) == pytest.approx(2 / 10)
    assert M.max_drawdown(pd.Series([100, 120, 90, 130, 117])) == pytest.approx(-0.25)
