"""기간 설정 점검: 점수표가 '아직 오지 않은 날'의 가격을 보지 않는지 확인한다 (기준 4장·5.5, 계획서 4.4).

막는 장치 3겹
1. 시험 구간 잠금: data_io.load_long/load_wide 는 2020-01-01 이후를 allow_test=True 없이 열지 않는다.
   잠금은 파일을 읽기 전에 걸린다 (2020년 줄은 메모리에 올라오지도 않는다).
2. 설계 구간만 불러오기: optimize 는 load_wide() 를 기본값(end=DESIGN_END, 2019-12-31) 그대로 부른다.
   기간은 config.py 에 있고 config.py 는 지문으로 고정돼 있어서, 기간을 바꾸면 optimize 가 실행을 거부한다.
3. 연도마다 자르기(5.5): 검증 연도 Y의 파라미터는 cut(= Y년 첫 거래일 − 21거래일)까지의 신호로만 고른다.
   H 최대 20 이면 마지막 학습 매매는 Y년 첫 거래일 시가에 판다 → Y년 첫날 '다음' 가격은 쓰지 않는다.
   거래정지로 매도가 늦어져 Y년 첫날 뒤에 판 학습 매매는 신호·대조군 양쪽에서 뺀다 (11/24 팀장 결정).

이 파일은 기본 pytest 모음(test_*.py)에 넣지 않았다. 노트북 04 가 'pytest 통과 수 130'을 엔진 v2 기준값과
비교하기 때문이다 (기준값 파일은 덮어쓰지 않는다). 노트북 05 의 2-2 칸이 이 파일을 따로 돌린다:
    python -m pytest tests/check_period.py

고치기 전 실제 건수 (노트북 05 2-2): 늦게 판 학습 매매 최대 0.02%. 지금은 학습에서 빠진다.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from indexbeater import config as C
from indexbeater import data_io
from indexbeater import engine as E
from indexbeater import lookahead as LA
from indexbeater import metrics as M
from indexbeater import optimize as O
from indexbeater import scorecard as SC
from indexbeater import signals as S
from indexbeater.exits import ExitRule

SIGNALS = ("ma_cross", "volume_breakout", "candle")   # 이동평균·거래량·캔들: 계산 방식이 서로 다른 3개
YEAR = 2017                                           # 가운데 검증 연도 (앞뒤로 학습·검증이 다 있다)
PX = ["adj_open", "adj_high", "adj_low", "adj_close", "open", "close", "mktcap"]
VOL = ["volume", "volume_s", "value"]


# ---------------------------------------------------------------- 1. 시험 구간 잠금
def test_test_period_is_locked_before_reading():
    # 없는 파일 이름을 줘도 '파일 없음'이 아니라 '시험 구간' 오류가 먼저 난다 = 읽기 전에 막는다
    with pytest.raises(ValueError, match="시험 구간"):
        data_io.load_long(end=C.TEST_START, path="없는파일.parquet")
    with pytest.raises(ValueError, match="시험 구간"):
        data_io.load_wide(end="2020-06-30")


def test_default_end_is_design_end():
    for f in (data_io.load_long, data_io.load_wide):
        assert inspect.signature(f).parameters["end"].default == C.DESIGN_END
    assert C.DESIGN_END < C.TEST_START


def test_optimize_loads_design_period_only(monkeypatch, tmp_path):
    """optimize 가 load_wide 를 어떤 값으로 부르는지 가로채서 본다 (실제 데이터는 읽지 않는다)."""
    called = {}

    class Stop(Exception):
        pass

    def fake_load_wide(*args, **kwargs):
        called["args"], called["kwargs"] = args, kwargs
        raise Stop

    monkeypatch.setattr(SC, "fingerprints", lambda with_data=True: {})
    monkeypatch.setattr(O, "reference_fingerprints", lambda path=None: {})
    monkeypatch.setattr(O, "check_frozen", lambda now, ref: [])
    monkeypatch.setattr(data_io, "load_wide", fake_load_wide)
    with pytest.raises(Stop):
        O.run(["ichimoku"], tmp_path)
    assert called == {"args": (), "kwargs": {}}       # 기본값 = DESIGN_END, allow_test=False


@pytest.mark.skipif(not C.KRX_ADJUSTED.exists(), reason="데이터 파일 없음 (구글 드라이브에서 받기)")
def test_real_data_stops_before_test_period():
    d = data_io.load_long()
    assert d["date"].max() < C.TEST_START


# ---------------------------------------------------------------- 2·3. 가짜 주가로 점수표 전체 돌려 보기
@pytest.fixture(scope="module")
def synthetic_long():
    """2011-11-17 ~ 2019-12-31 가짜 주가 60종목 (long 표). make_synthetic 이 to_wide 에 넘기는 표를 가로챈다."""
    cap = {}
    orig = data_io.to_wide
    data_io.to_wide = lambda df: cap.setdefault("long", df.copy())
    try:
        LA.make_synthetic(n_days=len(pd.bdate_range(C.DATA_START, C.DESIGN_END)), n_codes=60, seed=0)
    finally:
        data_io.to_wide = orig
    return cap["long"]


def _bend(long: pd.DataFrame, after: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    """after 다음 날부터 가격·시총·거래량을 0.5~1.5배로 마구 바꾼 사본 (거래정지일 0 은 0 그대로)."""
    rng = np.random.default_rng(seed)
    d = long.copy()
    m = (d["date"] > after).to_numpy()
    f, g = rng.uniform(0.5, 1.5, m.sum()), rng.uniform(0.5, 1.5, m.sum())
    for c in PX:
        d.loc[m, c] = d.loc[m, c].to_numpy() * f
    for c in VOL:
        d.loc[m, c] = d.loc[m, c].to_numpy() * g
    return d


def _run(long):
    ctx = SC.Context(data_io.to_wide(long), log=lambda *a: None)
    return ctx, {n: SC.run_signal(n, ctx, log=lambda *a: None, portfolio=False) for n in SIGNALS}


def _upto(df, year):
    return df[df["검증연도"] <= year].reset_index(drop=True)


@pytest.fixture(scope="module")
def base_run(synthetic_long):
    return _run(synthetic_long)


def _year_start(ctx, year):
    return ctx.dates[ctx.dates >= pd.Timestamp(f"{year}-01-01")][0]


def test_last_training_trade_sells_on_year_start(base_run):
    """cut + 1(다음 날 매수) + H 최대 20 = 검증 연도 첫 거래일. 2020 은 첫날이 자료 밖(= 읽지 않음)."""
    ctx, _ = base_run
    for y, ci in ctx.cut_idx.items():
        start_idx = int(np.searchsorted(ctx.dates, pd.Timestamp(f"{y}-01-01")))
        assert ci + 1 + max(C.HOLDS) == start_idx
    assert int(np.searchsorted(ctx.dates, pd.Timestamp("2020-01-01"))) == len(ctx.dates)


def test_prices_after_year_start_do_not_change_choice(synthetic_long, base_run):
    """Y년 첫날 다음부터 가격을 마구 바꿔도 Y년까지의 학습 Edge·고른 세트는 한 칸도 안 바뀐다.

    '불안정' 칸은 뺀다: 2015~2020 여섯 해에 고른 세트가 얼마나 같은지 붙이는 표시라서 뒤 연도를 본다.
    (고르는 데는 쓰지 않는다 — 점수표 '표시' 칸용)
    """
    ctx, base = base_run
    _, new = _run(_bend(synthetic_long, _year_start(ctx, YEAR)))
    for n in SIGNALS:
        pd.testing.assert_frame_equal(_upto(base[n]["train"], YEAR), _upto(new[n]["train"], YEAR))
        a = _upto(base[n]["selection"], YEAR).drop(columns="불안정")
        b = _upto(new[n]["selection"], YEAR).drop(columns="불안정")
        pd.testing.assert_frame_equal(a, b)


def test_check_is_sensitive_on_year_start(synthetic_long, base_run):
    """하루 앞(Y년 첫날 시가)부터 바꾸면 학습 Edge 가 바뀌어야 한다 = 이 점검이 진짜로 잡아낸다는 증거."""
    ctx, base = base_run
    _, new = _run(_bend(synthetic_long, _year_start(ctx, YEAR) - pd.Timedelta(days=1)))
    changed = [n for n in SIGNALS if not _upto(base[n]["train"], YEAR).equals(_upto(new[n]["train"], YEAR))]
    assert changed


# ---------------------------------------------------------------- 4. 거래정지로 늦게 판 학습 매매 (11/24 팀장 결정)
def test_late_control_trade_is_left_out_by_hand():
    """손 계산: 하루, 같은 칸(7) 종목 4개. 0번 = 신호 종목, 3번 = Y년 첫날(10) 뒤 50일에 판 대조군."""
    b = M.ControlBase.__new__(M.ControlBase)
    b.gross = np.array([[0.10, 0.20, 0.30, 0.60]])
    b.delist = np.zeros(b.gross.shape, bool)
    b.cell = np.array([[7, 7, 7, 7]], np.int64)
    b.n_cells = M.CELLS_PER_DAY
    b._tot = {w: b._sums(b.variant(w), np.ones(b.gross.shape, bool)) for w in (False, True)}
    b._late = (np.array([0, 0]), np.array([0, 3]), np.array([50, 50]))   # 0번(신호 종목)도 늦게 팜 → 두 번 빼면 안 됨
    sig_on = np.array([[True, False, False, False]])
    t, j = np.array([0]), np.array([0])
    mt = b.matched(sig_on, t, j)
    assert mt["ctrl_gross"][0] == pytest.approx((0.2 + 0.3 + 0.6) / 3)       # 보통 대조군: 1·2·3번
    assert b.matched_train(sig_on, t, j, mt, cut=0, start=10)[0] == pytest.approx(0.25)   # 학습: 3번도 뺌
    assert b.matched_train(sig_on, t, j, mt, cut=0, start=60)[0] == mt["ctrl_gross"][0]   # 첫날 전에 팔았으면 그대로


@pytest.fixture(scope="module")
def halted_long(synthetic_long, base_run):
    """종목 3개 중 1개를 YEAR년 첫날 10거래일 전 ~ 15거래일 뒤까지 거래정지로 바꾼 가짜 주가.

    그 사이에 팔 차례였던 학습 매매는 정지가 풀린 뒤(= YEAR년 안) 팔린다.
    """
    ctx, _ = base_run
    s = ctx.start_idx[YEAR]
    d = synthetic_long.sort_values(["code", "date"]).reset_index(drop=True)
    codes = sorted(d["code"].unique())[::3]
    m = d["date"].between(ctx.dates[s - 10], ctx.dates[s + 15]) & d["code"].isin(codes)
    prev = d[d["date"] == ctx.dates[s - 11]].set_index("code")["adj_close"]
    px = d.loc[m, "code"].map(prev).to_numpy()
    for c in ("adj_open", "adj_high", "adj_low", "open", "volume", "volume_s", "value"):
        d.loc[m, c] = 0.0
    d.loc[m, "adj_close"] = d.loc[m, "close"] = px
    d.loc[m, "mktcap"] = px * 1e7
    return d


def test_halted_setup_really_has_late_trades(halted_long, base_run):
    """준비가 제대로 됐는지: 신호 쪽·대조군 쪽 모두 'cut 전 신호 · YEAR년 첫날 뒤 매도' 매매가 있어야 한다."""
    ctx0, _ = base_run
    ci, s = ctx0.cut_idx[YEAR], ctx0.start_idx[YEAR]
    data = data_io.to_wide(halted_long)
    h = max(C.HOLDS)
    sig_late = 0
    for n in SIGNALS:
        tr = E.simulate_trades(S.entry_signal(n, data, SC.param_sets(n)[0]), data, ExitRule(hold=h))
        f = tr[tr["status"] == E.FILLED]
        sig_late += int(((f["t_idx"] <= ci) & (f["s_idx"] > s)).sum())
    assert sig_late > 0
    ctx = SC.Context(data, log=lambda *a: None)
    lt, _, ls = ctx.base[h]._late
    assert ((lt <= ci) & (ls > s)).sum() > 0


def test_late_training_trades_do_not_see_year_prices(halted_long, base_run):
    """긴 거래정지가 있어도 YEAR년 첫날 뒤 가격을 마구 바꾸면 학습 Edge·고른 세트가 그대로여야 한다."""
    ctx0, _ = base_run
    _, base = _run(halted_long)
    _, new = _run(_bend(halted_long, _year_start(ctx0, YEAR)))
    for n in SIGNALS:
        pd.testing.assert_frame_equal(_upto(base[n]["train"], YEAR), _upto(new[n]["train"], YEAR))
        a = _upto(base[n]["selection"], YEAR).drop(columns="불안정")
        b = _upto(new[n]["selection"], YEAR).drop(columns="불안정")
        pd.testing.assert_frame_equal(a, b)
