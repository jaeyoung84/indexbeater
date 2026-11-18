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
