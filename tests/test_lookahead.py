"""18개 신호 미래 정보 테스트 (계획서 부록 D). 새 신호를 추가하면 이 파일이 통과해야 한다."""
import numpy as np
import pandas as pd
import pytest

from indexbeater import config as C
from indexbeater import lookahead as LA
from indexbeater import signals as S


@pytest.fixture(scope="module")
def data():
    return LA.make_synthetic()


def test_all_18_signals_registered():
    assert list(S.SIGNALS) == list(C.PARAM_GRID)
    assert sum(1 for name in S.SIGNALS for _ in S.iter_params(name)) == C.PARAM_GRID_TOTAL


@pytest.mark.parametrize("name", list(C.PARAM_GRID))
def test_no_lookahead_every_param_set(name, data):
    for p in S.iter_params(name):
        assert LA.check_no_lookahead(name, p, data) == [], f"{name} {p} 미래 정보 사용"


@pytest.mark.parametrize("name", list(C.PARAM_GRID))
def test_every_param_set_fires(name, data):
    # 가짜 주가에서 한 번도 안 켜지면 테스트가 아무것도 검사하지 못한 것이다
    for p in S.iter_params(name):
        assert S.raw_signal(name, data, p).to_numpy().sum() > 0, f"{name} {p} 신호 0개"


def test_checker_catches_a_leak(data, monkeypatch):
    # 일부러 '내일 종가'를 보는 나쁜 신호를 넣으면 검사가 잡아내야 한다
    def leaky(d, period):
        return d["adj_close"].shift(-1) > d["adj_close"]
    monkeypatch.setitem(S.SIGNALS, "rsi", leaky)
    assert LA.check_no_lookahead("rsi", {"period": 14}, data) != []


def test_entry_signal_starts_at_signal_start(data):
    e = S.entry_signal("rsi", data, {"period": 14, "low": 30})
    assert not e.loc[e.index < C.SIGNAL_START].to_numpy().any()
    assert e.loc[e.index >= C.SIGNAL_START].to_numpy().any()


def test_event_helper_needs_yesterday():
    cond = pd.DataFrame({"a": [np.nan, 1.0, 1.0, 0.0, 1.0]})
    assert S.event(cond)["a"].tolist() == [False, False, False, False, True]
