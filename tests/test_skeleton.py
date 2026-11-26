"""뼈대 점검: 모든 모듈이 import 되고 공통 기준이 서로 맞는지 확인한다.

각 STEP을 구현하면 그 모듈의 테스트 파일을 이 폴더에 추가한다
(예: test_lookahead.py — 18개 신호 미래 정보 테스트, test_engine.py — 매매 규칙 손 계산 비교).
"""
import importlib
import pkgutil

import pandas as pd

import indexbeater
from indexbeater import config as C


def test_all_modules_import():
    for m in pkgutil.walk_packages(indexbeater.__path__, "indexbeater."):
        importlib.import_module(m.name)


def test_periods_do_not_overlap():
    assert C.DATA_START < C.INIT_TRAIN_END < C.DESIGN_START <= C.DESIGN_END < C.TEST_START <= C.DATA_END


def test_purge_covers_max_hold():
    assert C.PURGE_DAYS >= C.MAX_HOLD >= max(C.HOLDS)


def test_test_periods_inside_test_range():
    for a, b, _ in C.TEST_PERIODS.values():
        assert C.TEST_START <= pd.Timestamp(a) <= pd.Timestamp(b) <= C.DATA_END


def test_cost_and_tax_table():
    taxes = list(C.SELL_TAX_HISTORY.items())
    assert [k for k, _ in taxes] == sorted(k for k, _ in taxes)
    assert C.COST_SELECT >= max(v for _, v in taxes)   # 선택 비용은 세금보다 커야 한다


def test_triple_combo_count():
    from math import comb
    assert comb(C.TRIPLE_TOP_N, 3) == 20


def test_param_grid_total():
    # 그리드 세트 수를 세어서 문서(5.3 표)의 72세트와 같은지 확인한다
    from itertools import product
    total = 0
    for name, grid in C.PARAM_GRID.items():
        for combo in product(*grid.values()):
            p = dict(zip(grid.keys(), combo))
            if name == "ma_cross" and p["short"] >= p["long"]:
                continue                     # 단기 < 장기만 쓴다
            total += 1
    assert len(C.PARAM_GRID) == 18
    assert total == C.PARAM_GRID_TOTAL == 72


def test_cell_counts():
    # 시장별 칸 수: (18 + 153 + 20) × H 3개 × 비용 2개 = 1,146
    from math import comb
    per_set = len(C.HOLDS) * 2
    assert (18 + comb(18, 2) + comb(C.TRIPLE_TOP_N, 3)) * per_set == 1146
    assert len(C.S1_STOP) * len(C.S1_TAKE) == 20


def test_exit_hold_within_purge():
    # 어떤 매도 규칙을 붙여도 보유는 H(최대 20일) 안이어야 한다
    assert max(C.HOLDS) <= C.PURGE_DAYS
    assert C.WF_CUT_OFFSET == C.PURGE_DAYS + 1
