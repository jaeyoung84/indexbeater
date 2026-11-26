"""신호 점수표(5.17절) 계산이 기준대로 되는지 손 계산과 비교한다."""
import math

from indexbeater import config as C
from indexbeater.scoring import breakeven_win_rate, signal_score, success_ref_score


def test_weights_sum_to_100():
    assert sum(C.SCORE_WEIGHTS.values()) == 100
    assert C.SCORE_STAB_YEARS < C.SCORE_WEIGHTS["stability"]


def test_full_marks():
    s = signal_score(edge=0.02, hold=20, pos_years=5, t_value=2.5,
                     avg_win=0.06, avg_loss=0.03, win_rate=0.65, n_trades=1000)
    assert s["점수"] == 100.0
    assert s["표시"] == ""


def test_zero_marks_and_flags():
    s = signal_score(edge=-0.01, hold=10, pos_years=0, t_value=-1.0,
                     avg_win=0.01, avg_loss=0.03, win_rate=0.30, n_trades=100)
    assert s["점수"] == 0.0
    assert "경고(승률)" in s["표시"] and "표본 부족" in s["표시"]


def test_hand_calculation():
    # Edge 0.5%를 H=5로 → 20일 기준 2% → 40점
    # 플러스 3해 → 15×3/5 = 9점, t값 1.0 → 10×0.5 = 5점 → 안정 14점
    # 손익비 0.04/0.02 = 2.0 → 20점, 승률 50% → 15×0.5 = 7.5점
    s = signal_score(edge=0.005, hold=5, pos_years=3, t_value=1.0,
                     avg_win=0.04, avg_loss=0.02, win_rate=0.50, n_trades=500)
    assert s["수익점수"] == 40.0
    assert s["안정점수"] == 14.0
    assert s["손익비점수"] == 20.0
    assert s["승률점수"] == 7.5
    assert s["점수"] == 81.5


def test_low_win_rate_high_payoff_is_warned_not_removed():
    # 승률은 낮아도 손익비가 커서 돈을 버는 신호: 점수는 받고 경고만 붙는다
    s = signal_score(edge=0.01, hold=20, pos_years=4, t_value=2.0,
                     avg_win=0.12, avg_loss=0.03, win_rate=0.35, n_trades=800)
    assert s["승률점수"] == 0.0
    assert s["점수"] > 50
    assert s["표시"] == "경고(승률)"


def test_breakeven_and_success_ref():
    assert breakeven_win_rate(1.0) == 0.5
    assert math.isclose(breakeven_win_rate(3.0), 0.25)
    assert success_ref_score([True] * 5 + [False] * 2) == round(500 / 7, 1)
