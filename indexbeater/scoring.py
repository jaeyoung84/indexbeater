"""신호 점수표 (100점) — docs/프로젝트_기준.md 5.17절.

점수는 '어느 신호가 더 좋은지 한눈에 보는 순위표'와 발표용이다.
채택 여부는 4.1절 성공 기준(통과/실패)으로만 정한다. 점수로 정하지 않는다.

점수 = 수익(Edge) 40 + 안정성 25 + 손익비 20 + 승률 15
- 수익: 대조군보다 얼마나 더 벌었나 (비용 0.5% 뺀 뒤, 20일 보유 기준으로 맞춤)
- 안정성: 2015~2019 다섯 해 중 몇 해나 플러스였나 + t값
- 손익비: 이긴 거래 평균 이익 ÷ 진 거래 평균 손실
- 승률: 이긴 거래 비율. 40%보다 낮으면 '경고(승률)'
"""
from __future__ import annotations

import math

from indexbeater import config as C


def _line(x: float, lo: float, hi: float) -> float:
    """x가 lo면 0, hi면 1, 그 사이는 곧은 선으로 나눈다. 밖으로 나가면 0이나 1로 자른다."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def signal_score(edge: float, hold: int, pos_years: int, t_value: float,
                 avg_win: float, avg_loss: float, win_rate: float,
                 n_trades: int, n_years: int = 5) -> dict:
    """신호 하나(또는 조합 하나)의 100점 점수와 항목별 점수를 돌려준다.

    edge      : 거래 1건당 Edge (예: 0.01 = 1%), 비용 0.5% 뺀 값
    hold      : 보유기간 H (5·10·20)
    pos_years : Edge가 플러스인 해의 수 (0~5)
    t_value   : 20일 블록 t값 (5.4절)
    avg_win   : 이긴 거래의 평균 이익 (플러스 숫자)
    avg_loss  : 진 거래의 평균 손실 (크기만, 플러스 숫자로)
    win_rate  : 승률 (0~1)
    n_trades  : 거래 수 (300건 미만이면 '표본 부족'으로 순위에서 뺀다)
    """
    W = C.SCORE_WEIGHTS
    edge20 = edge * 20 / hold                      # 보유기간이 달라도 같은 20일 기준으로 비교
    s_edge = W["edge"] * _line(edge20, 0.0, C.SCORE_EDGE_FULL)

    year_part = C.SCORE_STAB_YEARS * max(0, min(pos_years, n_years)) / n_years
    t_part = (W["stability"] - C.SCORE_STAB_YEARS) * _line(t_value, 0.0, C.SCORE_STAB_T_FULL)
    s_stab = year_part + t_part

    if avg_loss and avg_loss > 0:
        payoff = avg_win / avg_loss
    else:
        payoff = math.inf if avg_win and avg_win > 0 else 0.0   # 진 거래가 없으면 손익비 최고
    s_pay = W["payoff"] * _line(min(payoff, 1e9), *C.SCORE_PAYOFF_RANGE)

    s_win = W["win"] * _line(win_rate, *C.SCORE_WIN_RANGE)

    flags = []
    if win_rate < C.WIN_RATE_WARN:
        flags.append("경고(승률)")
    if n_trades < C.MIN_TRADES:
        flags.append("표본 부족")

    total = s_edge + s_stab + s_pay + s_win
    return {
        "점수": round(total, 1),
        "수익점수": round(s_edge, 1),
        "안정점수": round(s_stab, 1),
        "손익비점수": round(s_pay, 1),
        "승률점수": round(s_win, 1),
        "Edge20": edge20,
        "손익비": payoff,
        "손익분기승률": breakeven_win_rate(payoff),
        "표시": ",".join(flags),
    }


def breakeven_win_rate(payoff: float) -> float:
    """손익비가 b일 때 본전이 되는 승률 = 1 ÷ (1 + b). 승률이 이보다 높아야 돈을 번다 (비용 뺀 뒤)."""
    if payoff is None or payoff <= 0:
        return 1.0
    if math.isinf(payoff):
        return 0.0
    return 1.0 / (1.0 + payoff)


def success_ref_score(passed: list[bool]) -> float:
    """성공 기준 7개 중 통과한 비율을 100점으로 바꾼 참고 점수 (발표용, 판정에는 쓰지 않음)."""
    return round(100 * sum(bool(p) for p in passed) / len(passed), 1)
