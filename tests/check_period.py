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

맨 끝 덧붙임 2개(11/24): optimize 의 --who(돌린사람)·끝 안내 문구 — 기간과 상관없지만 같은 이유로 여기에 둔다.
맨 끝 덧붙임 4개(11/24): D4 벤치마크(분배금 반영 ETF) — 시험 구간 잠금·분배금 재투자·분할 멈춤·상장한 해 비움.
맨 끝 덧붙임 3개(11/24): 코스닥 상장 전 대신 쓰는 지수(KQ11) — 잇는 날짜·빠진 거래일 멈춤·시험 구간 잠금.
맨 끝 덧붙임 4개(11/25): 점수표 지수 초과수익 열 — 그냥 빼기·원래 열 그대로·빠진 해 멈춤·시험 구간 잠금.
맨 끝 덧붙임 4개(11/25): 신호 조합 AND(W=3, 기준 5.6) — 날짜 창 경계·자기 자신과 조합 = 단독·3신호·미래 엿보기.
  + 2개(11/25): 153개 목록·5.5 고른 세트 읽기, 다른 지문 묶음은 섞지 않음.
  + 3개(11/25): 3신호 고르기(5.7-1~3) — 토막은 cut 까지만·거래일 3등분, 순위·동점 규칙, 20개·816개 목록.
  + 1개(11/25): 조합 포트폴리오 — 살 후보 0건이면 엔진을 부르지 않음 (후보 조건 = 엔진과 같음).
맨 끝 덧붙임 9개(11/25): 4단계 합격 시험(stage4) — 정상 부트스트랩·SPA/RC·Holm·개선 판정(엄격)·상위 3개 규칙·
  옛 코드 0.0 칸 빈칸·포트폴리오 다시 계산 = 점수표·랜덤 매수 같은 수·지수 비교.
  + 1개(11/25): 시험 5 — combo_tables 에 준 W 가 AND 합치기까지 그대로 간다 (기본 = config.AND_WINDOW).
  + 1개(11/25): 시험 1 — 시도 수표에서 고른 20개 칸(816개 칸 안에 있음)을 겹침 줄로 빼서 두 번 세지 않는다.
맨 끝 덧붙임 4개(11/25): 대표 일봉차트 조합 8개(classic, 별개 비교분석·참고) — 미래 엿보기·정의(③ W=3·⑤ ⊆ ③·
  2012-07-01 전 끔·켜진 날 조건)·무작위/비용 시험 신호 표 = 점수표 포트폴리오·⑥ 참고 열과 빈 포트폴리오 칸.
맨 끝 덧붙임 5개(11/26): 다음 날 종가 매도 비교(close_exit, 참고) — 5거래일 뒤 시가로 돌리면 엔진·점수표와 끝자리까지
  같음·종가 매도 규칙(팔 수 있는 첫 날 종가·상장폐지 정산)·거래정지 날 미룸·미래 엿보기·종가 매도 날 자리는 차 있음.
맨 끝 덧붙임 4개(11/26): 팀원 최적 비교(team_compare, 참고) — 미래 엿보기·정의(김승찬 W=3·%K ⊆·김동겸 03 띠)·
  07 학습 표본 창·날짜별 상위 30%·김동겸 표 24줄 읽기.
맨 끝 덧붙임 4개(11/26): TOP3 매수 신호 CSV(buy_export, 팀원 매도 ML·DL 용) — 최종 모델 cut(2020 첫날 − 21)·
  2020 자료 있으면 멈춤·다-2 최종 모델 학습 창(cut 까지·예정 매도일 2020 전)과 점수 매긴 구간(2015 전)·
  2012~2014 = 최종 모델 / 2015~ = 해마다 이어 붙이기·CSV 다시 읽어 엔진에 넣으면 같은 매수.
맨 끝 덧붙임 9개(11/26): 매도 ML·DL 공통 시작 코드(sell_common) — 고정 H5·H10·H20 = 엔진·포트폴리오 = 엔진·
  매도 정산(팔 수 있는 첫 시가·상장폐지·미완료)·라벨·정화 40·미래 엿보기·최대 보유 없음·가중치·기준 확률·모델 7종·
  CSV 멈춤·칸 포트폴리오.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from indexbeater import combo as CB
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


# ---------------------------------------------------------------- 덧붙임: 실행 안내 (기간과는 상관없음, 11/24)
# 기본 pytest 수(130)를 바꾸지 않으려고 여기에 둔다 (위 설명 참고).
def test_who_is_written_to_run(monkeypatch):
    """--all --who 이름 → run() 에 그 이름이 간다 (meta '돌린사람' 칸). 안 주면 빈칸."""
    got = {}
    monkeypatch.setattr(O, "run", lambda sigs, out_dir, portfolio=True, who="": got.update(sigs=sigs, who=who) or 0)
    assert O.main(["--all", "--who", "송재영"]) == 0
    assert got == {"sigs": O.SIGNAL_ORDER, "who": "송재영"}
    O.main(["--signals", "rsi"])
    assert got == {"sigs": ["rsi"], "who": ""}


def test_next_step_message_points_to_merge(tmp_path):
    """나눠 돌리지 않으므로(D6 취소) 끝 안내는 '드라이브에 올리기'가 아니라 합치기다."""
    msg = O.next_step_message(SC.PART_DIR)
    assert "--merge" in msg and "드라이브" not in msg
    assert "--compare" in O.next_step_message(tmp_path)


# ---------------------------------------------------------------- 덧붙임: D4 벤치마크 (11/24)
# 배당(분배금) 반영 ETF 비교 지수. 기본 pytest 수(130)를 바꾸지 않으려고 여기에 둔다 (위 설명 참고).
from indexbeater import benchmark as B  # noqa: E402


def _etf(close, base):
    d = pd.bdate_range("2016-04-25", periods=len(close))
    return pd.DataFrame({"date": d, "code": "069500", "name": "KODEX 200", "close": close, "base_price": base})


def test_benchmark_test_period_is_locked_before_reading():
    # 없는 파일 이름을 줘도 '시험 구간' 오류가 먼저 난다 = 읽기 전에 막는다
    with pytest.raises(ValueError, match="시험 구간"):
        B.load_raw(end=C.TEST_START, path="없는파일.parquet")
    with pytest.raises(ValueError, match="시험 구간"):
        B.build(end="2020-06-30", path="없는파일.parquet")
    assert inspect.signature(B.load_raw).parameters["end"].default == C.DESIGN_END


def test_benchmark_distribution_is_reinvested():
    """3일째 분배락: 전일 종가 100, 기준가 98 → 분배금 2. 종가 98 이면 가격은 −2% 지만 수익은 0."""
    tr = B.total_return(_etf([100, 100, 98, 99], [np.nan, 100, 98, 98]))
    assert tr["dist"].tolist() == [0, 0, 2, 0]
    assert tr["ret_tr"].round(12).tolist() == [0, 0, 0, round(1 / 98, 12)]
    assert tr["tr_index"].iloc[-1] == pytest.approx(100 * 99 / 98)
    assert tr["price_index"].iloc[-1] == pytest.approx(99)


def test_benchmark_stops_on_split_like_event():
    """기준가가 전일 종가의 절반 = 분할 같은 다른 사건 → 분배금으로 넣지 않고 멈춘다."""
    with pytest.raises(ValueError, match="분할"):
        B.total_return(_etf([100, 100, 50], [np.nan, 100, 50]))
    with pytest.raises(ValueError, match="다른 사건"):
        B.total_return(_etf([100, 100, 103], [np.nan, 100, 102]))   # 기준가 > 전일 종가


def test_benchmark_listing_year_is_blank():
    """상장한 해(1년 전체가 아님)는 연간 수익을 비운다. 다음 해는 전년 마지막 종가부터 잰다."""
    d = pd.to_datetime(["2015-10-01", "2015-12-30", "2016-06-30", "2016-12-29"])
    raw = pd.DataFrame({"date": d, "code": "229200", "name": "KODEX 코스닥150",
                        "close": [100.0, 110, 100, 121], "base_price": [np.nan, 100, 110, 100]})
    y = B.yearly(B.total_return(raw)).set_index("year")
    assert np.isnan(y.loc[2015, "수익률_분배포함"])
    assert y.loc[2016, "수익률_가격만"] == pytest.approx(0.1)
    assert y.loc[2016, "분배금_합"] == 0
    assert (y["bench_market"] == "KOSDAQ").all()


# 코스닥 상장 전 대신 쓰는 지수 (11/24 팀장 결정 1번: KQ11 로 채운다)
def _two_etfs():
    d = pd.bdate_range("2015-09-28", periods=5)                  # 229200 은 3일째(d[2]) 상장이라고 친다
    k = pd.DataFrame({"date": d, "code": "069500", "name": "KODEX 200",
                      "close": [100.0, 101, 102, 103, 104], "base_price": [np.nan, 100, 101, 102, 103]})
    q = pd.DataFrame({"date": d[2:], "code": "229200", "name": "KODEX 코스닥150",
                      "close": [10.0, 11, 11], "base_price": [np.nan, 10, 11]})
    sub = pd.DataFrame({"date": d, "symbol": "KQ11", "close": [500.0, 510, 520.2, 0, 0]})   # 상장 뒤 값은 쓰면 안 된다
    return d, B.total_return(pd.concat([k, q])), sub


def test_benchmark_kosdaq_before_listing_uses_sub_index():
    """상장 첫날까지는 KQ11 하루 수익률, 다음 날부터 229200. 상장 뒤 KQ11 값(0)은 쓰지 않는다."""
    d, tr, sub = _two_etfs()
    ms = B.market_series(tr, sub)
    kq = ms[ms["market"] == "KOSDAQ"].set_index("date")
    assert kq["ret"].round(12).tolist() == [0, 0.02, 0.02, 0.1, 0]
    assert kq["source"].tolist() == ["KQ11 대신 (가격만)"] * 3 + ["229200 분배 포함"] * 2
    assert kq["index"].iloc[-1] == pytest.approx(100 * 1.02 * 1.02 * 1.1)
    ks = ms[ms["market"] == "KOSPI"]
    assert ks["index"].iloc[-1] == pytest.approx(104) and (ks["source"] == "069500 분배 포함").all()
    y = B.market_yearly(ms)
    assert y["수익률"].isna().all()                                # 첫해(1년이 안 됨)는 비운다
    assert y.loc[y["market"] == "KOSDAQ", "출처"].item() == "KQ11 대신 (가격만) + 229200 분배 포함"


def test_benchmark_sub_index_missing_day_stops():
    """KQ11 에 거래일이 하루 빠지면 조용히 건너뛰지 않고 멈춘다."""
    d, tr, sub = _two_etfs()
    with pytest.raises(ValueError, match="거래일"):
        B.market_series(tr, sub.drop(index=1))


def test_benchmark_sub_index_test_period_locked():
    """대신 쓰는 지수도 2020년 이후는 읽기 전에·받기 전에 막는다 (없는 파일·인터넷 없이도 오류가 먼저)."""
    with pytest.raises(ValueError, match="시험 구간"):
        B.load_sub_index("없는파일.parquet", end=C.TEST_START)
    with pytest.raises(ValueError, match="시험 구간"):
        B.fetch_sub_index(out="없는파일.parquet", end="2020-01-02")
    assert B.SUB_END < C.TEST_START


# ---------------------------------------------------------------- 덧붙임: 점수표 지수 초과수익 열 (11/25)
# 11/25 팀장 결정: 초과수익 = 포트 연수익 − 같은 시장 비교 지수 연수익 (그냥 빼기), scorecard.csv 에 열을 더한다.
def _my():
    """가짜 비교 지수 연수익: KOSPI 는 매년 10%, KOSDAQ 은 매년 20% (2015 는 KQ11 대신 씀)."""
    rows = []
    for y in C.WF_YEARS:
        rows.append({"market": "KOSPI", "year": y, "수익률": 0.10, "출처": "069500 분배 포함"})
        src = "KQ11 대신 (가격만) + 229200 분배 포함" if y == 2015 else "229200 분배 포함"
        rows.append({"market": "KOSDAQ", "year": y, "수익률": 0.20, "출처": src})
    return pd.DataFrame(rows)


def _score():
    """점수표 3줄: KOSPI 0.5%, KOSDAQ 0.5%, KOSPI 0.25%(포트폴리오 없음)."""
    s = pd.DataFrame({"신호": ["a", "a", "a"], "시장": ["KOSPI", "KOSDAQ", "KOSPI"], "비용": ["0.5%", "0.5%", "0.25%"]})
    for y, a, b in zip(C.WF_YEARS, [0.15, 0.05, 0.10, 0.30, 0.00], [0.10, 0.25, 0.30, 0.20, 0.15]):
        s[f"포트_수익_{y}"] = [a, b, np.nan]
    s["포트_MDD"] = [-0.1, -0.2, np.nan]
    return s


def test_excess_is_simple_difference():
    out = B.add_excess(_score(), _my())
    ex = out[[f"포트_초과수익_{y}" for y in C.WF_YEARS]]
    assert ex.iloc[0].round(12).tolist() == [0.05, -0.05, 0.0, 0.2, -0.1]       # KOSPI: 포트 − 10%
    assert ex.iloc[1].round(12).tolist() == [-0.1, 0.05, 0.1, 0.0, -0.05]       # KOSDAQ: 포트 − 20%
    assert ex.iloc[2].isna().all()                                              # 0.25% 줄은 빈칸
    assert out["포트_지수수익_2017"].tolist()[:2] == [0.10, 0.20]
    assert out["포트_초과수익_중앙값"].round(12).tolist()[:2] == [0.0, 0.0]
    assert out["포트_초과음수해수"].tolist()[:2] == [2, 2] and pd.isna(out["포트_초과음수해수"].iloc[2])
    assert out["포트_지수대신쓴해"].tolist()[:2] == ["없음", "2015"] and pd.isna(out["포트_지수대신쓴해"].iloc[2])
    assert list(out.columns) == list(_score().columns) + B.EXCESS_COLS           # 맨 뒤에 붙는다


def test_excess_keeps_original_columns_and_repeats_same(tmp_path):
    s = _score()
    once = B.add_excess(s, _my())
    pd.testing.assert_frame_equal(once[list(s.columns)], s)                     # 원래 열은 그대로
    pd.testing.assert_frame_equal(B.add_excess(once, _my()), once)              # 두 번 붙여도 같다
    # 파일로 붙이기: 원래 열 그대로 + 같은 내용이면 덮어쓰지 않아도 된다
    sc, mk = tmp_path / "scorecard.csv", tmp_path / "market.csv"
    s.to_csv(sc, index=False, encoding="utf-8-sig")
    d = pd.bdate_range("2014-12-31", "2019-12-31")
    pd.DataFrame({"date": d, "market": "KOSPI", "ret": 0.0, "index": 100.0, "source": "x"}).to_csv(mk, index=False)
    with pytest.raises(ValueError, match="KOSDAQ"):                              # 시장이 빠지면 멈춤
        B.attach_scorecard(sc, mk)


def test_excess_missing_benchmark_year_stops():
    my = _my()
    with pytest.raises(ValueError, match="2018"):
        B.add_excess(_score(), my[~((my["market"] == "KOSDAQ") & (my["year"] == 2018))])
    half = _score()
    half.loc[0, "포트_수익_2019"] = np.nan                                       # 한 해만 빈 줄 → 멈춤
    with pytest.raises(ValueError, match="일부"):
        B.add_excess(half, _my())


def test_excess_benchmark_file_test_period_locked(tmp_path):
    p = tmp_path / "market.csv"
    pd.DataFrame({"date": pd.to_datetime(["2019-12-30", "2020-01-02"]), "market": "KOSPI",
                  "ret": 0.0, "index": 100.0, "source": "x"}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="시험 구간"):
        B.load_market(p)


# ---------------------------------------------------------------- 신호 조합 AND (11/25, 기준 5.6·5.7)
def _onoff(days, n=10):
    a = np.zeros((n, 1), bool)
    a[list(days), 0] = True
    return pd.DataFrame(a, index=pd.bdate_range("2015-01-01", periods=n), columns=["000010"])


def test_combo_window_edges():
    assert C.AND_WINDOW == 3
    a = _onoff([5])
    got = lambda b: CB.and_entry([a, _onoff(b)])["000010"].to_numpy().nonzero()[0].tolist()
    assert got([5]) == [5]                         # 같은 날
    assert got([3]) == [5]                         # 다른 신호가 T−2 → T일에 켜짐
    assert got([2]) == []                          # T−3 은 창 밖
    assert got([7]) == [7]                         # 반대 순서: 나중 신호가 켜진 날(7)에 켜짐, 5일에는 안 켜짐
    assert got([6]) == [6]                         # 다음 날 켜진 신호를 5일에 미리 보지 않는다
    assert got([8]) == []
    b = _onoff([3])
    assert CB.and_entry([a, b]).equals(CB.and_entry([b, a]))   # 순서 바꿔도 같다


def test_combo_with_itself_is_single():
    d = LA.make_synthetic()
    e = S.entry_signal("rsi", d, {"period": 14, "low": 30})
    assert e.to_numpy().sum() > 0
    assert CB.and_entry([e, e]).equals(e)
    assert CB.and_entry([e, e, e]).equals(e)


def test_combo_three_needs_all_three():
    a, b = _onoff([5]), _onoff([4])
    got = lambda c: CB.and_entry([a, b, _onoff(c)])["000010"].to_numpy().nonzero()[0].tolist()
    assert got([3]) == [5]                         # 3·4·5 모두 창 안
    assert got([2]) == []                          # 셋째가 T−3 → 안 켜짐
    assert got([6]) == [6]                         # 6일: 6 켜짐, 4·5 가 6−2~6 안
    assert got([]) == []                           # 하나라도 없으면 안 켜짐


def test_combo_no_lookahead():
    """k일 이후 가격을 바꿔도 k일 전 조합 신호는 그대로 (lookahead.py 와 같은 방식, 2·3신호)."""
    d = LA.make_synthetic()
    combos = [[("rsi", {"period": 14, "low": 30}), ("macd", {"fast": 12, "slow": 26, "signal": 9})],
              [("ma_cross", {"short": 5, "long": 20}), ("volume_breakout", {"n": 20, "m": 2}),
               ("candle", {"trend": 5, "tail_ratio": 2})]]
    for parts in combos:
        for name, p in parts:
            assert p in list(S.iter_params(name)), (name, p)
        make = lambda dd: CB.and_entry([S.entry_signal(n, dd, p) for n, p in parts])
        base = make(d)
        n = len(base)
        rng = np.random.default_rng(1)
        for k in (n - 1, n // 4, n // 2, (3 * n) // 4, 30):
            assert base.iloc[:k].equals(make(LA.perturb(d, k, rng)).iloc[:k]), (parts, k)
    # 실제 신호는 드물어서 바꾼 날 바로 앞에 조합이 안 켜질 수 있다 → 촘촘한 무작위 신호표로 한 번 더
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=300)
    for m in (2, 3):
        es = [pd.DataFrame(rng.random((300, 20)) < 0.2, index=idx) for _ in range(m)]
        base = CB.and_entry(es)
        for k in (1, 2, 50, 150, 299):
            ch = [e.copy() for e in es]
            for e in ch:
                e.iloc[k:] = rng.random((300 - k, 20)) < 0.5
            assert base.iloc[:k].equals(CB.and_entry(ch).iloc[:k]), (m, k)


def test_combo_pairs_and_picks():
    import json
    from indexbeater import signals as S
    pairs = CB.pair_list()
    order = list(S.SIGNALS)
    assert len(pairs) == len(set(pairs)) == 153
    assert all(order.index(a) < order.index(b) for a, b in pairs)
    picks, flags = CB.load_picks()
    assert len(picks) == len(C.UNIVERSE_MARKETS) * 18 * len(C.HOLDS) * 6        # 2015~2019 + 2020용
    # 표시는 단독 점수표의 선택 표시와 같아야 한다 (조합 줄에 그대로 옮기므로)
    sc = pd.read_csv(C.RESULTS / "scorecard.csv", encoding="utf-8-sig")
    for r in sc[sc["비용"] == "0.5%"].to_dict("records"):
        got = flags[(r["시장"], r["신호"], int(r["H"]))]
        shown = [x for x in str(r["표시"]).split(",") if x in ("표본 부족", "불안정", "기본값 사용")]
        assert set(got) <= set(shown), (r["시장"], r["신호"], r["H"], got, shown)
        if "불안정" in shown or "기본값 사용" in shown:
            assert ("불안정" in got) == ("불안정" in shown) and ("기본값 사용" in got) == ("기본값 사용" in shown)
        assert json.loads(r["2020용세트"]) is not None


def test_combo_part_with_other_fingerprint_stops(tmp_path):
    import json
    d = tmp_path / "a+b"
    d.mkdir()
    assert not CB.part_done(d, {"x": "1"})
    (d / "meta.json").write_text(json.dumps({"지문": {"x": "1"}}), encoding="utf-8")
    assert CB.part_done(d, {"x": "1"})
    with pytest.raises(SystemExit):
        CB.part_done(d, {"x": "2"})


def test_triple_segments_stop_at_cut():
    """토막 = 신호 시작 ~ 2015 cut (끝 포함), 거래일 수로 3등분, 겹침·빈틈 없음. 2015년 날짜는 들어가지 않는다."""
    from types import SimpleNamespace
    dates = pd.bdate_range("2011-11-17", "2015-06-30")         # 가짜 달력 (실제 597일은 --triple-pick 이 출력)
    first = int(np.searchsorted(dates, pd.Timestamp("2015-01-01")))
    ctx = SimpleNamespace(dates=dates, cut_idx={2015: first - 21})
    b = CB.segment_bounds(ctx, 3, 2015)
    assert len(b) == 3 and b[-1][1] == first - 21
    assert dates[b[0][0]] >= C.SIGNAL_START and dates[b[0][0] - 1] < C.SIGNAL_START
    assert all(b[i][1] + 1 == b[i + 1][0] for i in range(2))
    sizes = [e - a + 1 for a, e in b]
    assert max(sizes) - min(sizes) <= 1
    assert dates[b[-1][1]] < pd.Timestamp("2015-01-01")


def test_triple_rank_rule():
    """순위: (시장, 토막, H) 칸마다 Edge 큰 쪽 1위, Edge 없음 = 꼴찌(신호 수) → 9개 중앙값 → 평균 → 부록 A 번호."""
    names = list(S.SIGNALS)
    rows = []
    for seg in (1, 2, 3):
        for h in C.HOLDS:
            for i, n in enumerate(names):
                e = -i * 1e-3                                  # 기본: 부록 A 순서대로 1, 2, 3 ... 위
                if n == names[17] and seg == 1:
                    e = 1.0                                    # 18번이 한 토막에서만 1위 → 중앙값은 그대로 아래
                if n == names[5]:
                    e = np.nan                                 # 거래 0건 → 꼴찌 18
                if n in (names[3], names[4]):
                    e = -3.5e-3                                # 4·5번 동점 → 같은 순위, 부록 A 번호로 가름
                rows.append({"시장": "KOSPI", "신호": n, "H": h, "토막": seg, "Edge": e})
    seg = pd.DataFrame(rows)
    top = CB.rank_top(pd.concat([seg, seg.assign(토막="전체")]))   # '전체' 줄은 순위에 안 쓴다
    assert (top["순위수"] == 9).all()
    r = top.set_index("신호")
    assert r.loc[names[5], "순위중앙값"] == 18
    assert r.loc[names[3], "순위중앙값"] == r.loc[names[4], "순위중앙값"]
    assert r.loc[names[3], "등수"] < r.loc[names[4], "등수"]
    assert r.loc[names[17], "순위중앙값"] == 17                 # 9개 중 3개만 1위 → 중앙값은 아래쪽
    assert top[top["상위"]]["신호"].tolist() == [names[i] for i in (0, 1, 2, 3, 4, 6)]
    trip = CB.triple_list(top, "KOSPI")
    assert len(trip) == 20 and len(set(trip)) == 20
    assert all(names.index(a) < names.index(b) < names.index(c) for a, b, c in trip)


def test_triple_all_list():
    t = CB.all_triples()
    order = list(S.SIGNALS)
    assert len(t) == len(set(t)) == 816
    assert all(order.index(a) < order.index(b) < order.index(c) for a, b, c in t)


def test_combo_portfolio_candidate():
    """포트폴리오 후보 = 신호 켜짐 & 그날 tradable & 2012-07 이후 & 그날 그 시장 (engine.simulate_trades 와 같은 조건).
    후보 0건이면 엔진을 부르지 않는다 (11/25: 3신호 stochastic+disparity+volume_breakout 에서 엔진이 빈 표로 멈춤)."""
    from types import SimpleNamespace
    dates = pd.bdate_range("2014-12-29", periods=10)
    codes = ["A", "B"]
    market = np.array([["KOSPI", "KOSDAQ"]] * len(dates), dtype=object)
    allowed = np.ones((len(dates), 2), bool)
    ctx = SimpleNamespace(dates=dates, years=dates.year.to_numpy(), m=SimpleNamespace(codes=codes, market=market),
                          allowed=allowed)
    ysets = {y: "k" for y in C.WF_YEARS}
    on = pd.DataFrame(False, index=dates, columns=codes)
    assert not CB.has_portfolio_candidate(ctx, "KOSPI", ysets, {"k": on})           # 신호 없음
    b = on.copy(); b.iloc[5, 1] = True
    assert CB.has_portfolio_candidate(ctx, "KOSDAQ", ysets, {"k": b})
    assert not CB.has_portfolio_candidate(ctx, "KOSPI", ysets, {"k": b})            # 다른 시장 종목만
    a = on.copy(); a.iloc[1, 0] = True                                               # 2014년 날짜 → 2015~2019 밖
    assert not CB.has_portfolio_candidate(ctx, "KOSPI", ysets, {"k": a})
    a.iloc[5, 0] = True
    ctx.allowed = allowed.copy(); ctx.allowed[5, 0] = False                          # 그날 거래 못 하는 종목
    assert not CB.has_portfolio_candidate(ctx, "KOSPI", ysets, {"k": a})


# ---------------------------------------------------------------- 4단계 합격 시험 (기준 5.8·5.11·5.12, 11/25)
from indexbeater import stage4 as S4   # noqa: E402


def test_stage4_stationary_indices():
    """정상 부트스트랩: 번호는 0~n−1, 새 출발점 비율 ≈ 1/block, 끝 다음은 처음 (원형), 같은 SEED → 같은 결과."""
    n, block = 500, 20
    idx = S4.stationary_indices(n, 300, block, np.random.default_rng(0))
    assert idx.shape == (300, n) and idx.min() >= 0 and idx.max() < n
    jump = (idx[:, 1:] - idx[:, :-1]) % n != 1
    assert 0.8 / block < jump.mean() < 1.2 / block
    assert ((idx[:, :-1] == n - 1) & (idx[:, 1:] == 0)).any()
    assert (S4.stationary_indices(n, 300, block, np.random.default_rng(0)) == idx).all()


def test_stage4_spa_and_rc():
    """평균 0 인 칸만 있으면 p 가 크고, 하루 0.3% 좋은 칸이 하나 있으면 p 가 작다.
    아주 나쁜 칸을 많이 더하면 RC p 는 커지고 SPA p 는 (거의) 그대로 = SPA 가 나쁜 칸에 덜 흔들린다 (Hansen 2005)."""
    rng = np.random.default_rng(1)
    n, K = 1000, 30
    X = rng.normal(0, 0.01, (n, K))
    X -= X.mean(axis=0)
    idx = S4.stationary_indices(n, 500, 20, np.random.default_rng(2))
    r0 = S4.spa_test(X, idx)
    assert r0["SPA_p"] > 0.2 and r0["RC_p"] > 0.2
    good = X.copy()
    good[:, 3] += 0.003
    r1 = S4.spa_test(good, idx)
    assert r1["SPA_p"] < 0.01 and r1["RC_p"] < 0.01 and r1["칸별_p"][3] < 0.01
    mid = X.copy()
    mid[:, 3] += 0.0006
    bad = np.hstack([mid, rng.normal(-0.003, 0.01, (n, 50))])
    a, b = S4.spa_test(mid, idx), S4.spa_test(bad, idx)
    assert b["RC_p"] > a["RC_p"] + 0.05
    assert abs(b["SPA_p"] - a["SPA_p"]) < 0.03


def test_stage4_holm():
    assert np.allclose(S4.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(S4.holm([0.5, 0.6]), [1.0, 1.0])


def _cells(rows):
    return pd.DataFrame(rows, columns=["시장", "종류", "신호", "H", "점수", "거래수", "구성1", "구성2"])


def test_stage4_improved_is_strict():
    """개선 = 300건 이상 & 점수 > 단독 1위 & 점수 > 구성 신호 각각의 같은 H 점수 (같으면 개선 아님)."""
    rows = []
    for mk in C.UNIVERSE_MARKETS:
        for h in (5, 10):
            rows += [(mk, "단독", "x", h, 5.0, 500, None, None), (mk, "단독", "y", h, 6.0, 500, None, None),
                     (mk, "단독", "z", h, 7.0, 500, None, None), (mk, "단독", "w", h, 99.0, 100, None, None)]
        rows += [(mk, "2신호", "x+z", 5, 7.5, 500, "x", "z"),       # 개선
                 (mk, "2신호", "y+z", 5, 7.0, 500, "y", "z"),       # 단독 1위와 같음 → 아님
                 (mk, "2신호", "x+y", 5, 6.5, 500, "x", "y"),       # 구성보다는 낫지만 단독 1위보다 못함
                 (mk, "2신호", "x+z", 10, 8.0, 100, "x", "z"),      # 300건 미만
                 (mk, "2신호", "w+z", 5, 50.0, 500, "w", "z")]      # 구성 w(99점, 표본 부족 단독)보다 못함
    j = S4.judge_improved(_cells(rows), kinds=("2신호",))
    assert (j["단독1위점수"] == 7.0).all() and (j["단독1위"] == "z H=5").all()     # 300건 미만 단독은 1위가 못 된다
    got = sorted(set(zip(j.loc[j["개선"], "신호"], j.loc[j["개선"], "H"])))
    assert got == [("x+z", 5)]
    assert j["개선"].sum() == len(C.UNIVERSE_MARKETS)


def test_stage4_pick_top_rule():
    """최고 점수 − 1 안에서 초과수익 중앙값 큰 순 → 고른 신호의 다른 H 칸은 지움 → 다시 최고 점수부터."""
    p = pd.DataFrame({"신호": ["a", "a", "b", "c", "d"], "H": [5, 10, 5, 5, 5],
                      "점수": [10.0, 9.5, 9.2, 8.0, 6.0], "포트_초과수익_중앙값": [0.01, 0.05, 0.02, 0.10, 0.20],
                      "포트_초과음수해수": [0, 0, 0, 0, 0], "거래수": [500] * 5})
    t = S4.pick_top(p, n=3)
    assert list(zip(t["신호"], t["H"])) == [("a", 10), ("b", 5), ("c", 5)]
    assert list(t["1점안칸수"]) == [3, 1, 1]                        # 3번째: c(8점)만, d(6점)는 1점 밖
    q = pd.DataFrame({"신호": ["e", "f"], "H": [5, 5], "점수": [9.0, 9.5], "포트_초과수익_중앙값": [0.03, 0.03],
                      "포트_초과음수해수": [0, 1], "거래수": [500, 900]})
    assert S4.pick_top(q, n=1)["신호"].iloc[0] == "e"              # 중앙값 같으면 지수에 진 해가 적은 쪽


def test_stage4_blank_old_zero():
    """거래 0건 + 포트폴리오 0.0 칸은 빈칸으로. 알려진 8칸 말고 또 있으면 멈춘다."""
    def df(name, mk):
        d = {"시장": [mk, mk], "신호": [name, name], "H": [5, 10], "거래수": [0, 400], "포트_MDD": [0.0, -0.2],
             "포트_마지막평가금액": [1e8, 1.3e8], "포트_초과수익_중앙값": [-0.05, 0.02]}
        d.update({c: [0.0, 0.05] for c in S4.YEAR_RET})
        return pd.DataFrame(d)
    out, found = S4.blank_old_zero(df("ichimoku+rsi", "KOSPI"), "2신호")
    assert found == {("2신호", "ichimoku+rsi", "KOSPI", 5)}
    assert out.loc[0, S4.PORT_COLS + ["포트_초과수익_중앙값"]].isna().all()
    assert out.loc[1, S4.PORT_COLS].notna().all() and out.loc[1, "포트_초과수익_중앙값"] == 0.02
    with pytest.raises(ValueError, match="8칸"):
        S4.blank_old_zero(df("ichimoku+rsi", "KOSDAQ"), "2신호")


@pytest.fixture(scope="module")
def s4_ctx(synthetic_long):
    return SC.Context(data_io.to_wide(synthetic_long), log=lambda *a: None)


def test_stage4_portfolio_same_as_scorecard(s4_ctx):
    """다시 만든 포트폴리오(비용 0.5%) = scorecard._portfolio 값 그대로. 0.25% 면 마지막 평가금액이 더 크거나 같다.
    일간 수익을 다 곱하면 마지막 평가금액이 된다 (SPA 에 쓰는 일간 수익의 첫날 처리 확인)."""
    ctx = s4_ctx
    sig = pd.DataFrame(np.random.default_rng(3).random((len(ctx.dates), len(ctx.m.codes))) < 0.02,
                       index=ctx.dates, columns=ctx.m.codes)
    ysets = {y: "k" for y in C.WF_YEARS}
    for mk in C.UNIVERSE_MARKETS:
        ref, _ = SC._portfolio("t", mk, 10, {y: {"set": "k"} for y in C.WF_YEARS}, {"k": sig}, ctx)
        out, eq = S4.portfolio_at_cost(S4.year_signal(ctx, ysets, {"k": sig}), ctx, mk, 10)
        assert out == ref
        assert S4.same_as_saved(out, {**ref, "시장": mk, "신호": "t", "H": 10}) == 0.0
        cheap, _ = S4.portfolio_at_cost(S4.year_signal(ctx, ysets, {"k": sig}), ctx, mk, 10, C.COST_ROUNDTRIP)
        assert cheap["포트_마지막평가금액"] >= out["포트_마지막평가금액"]
        assert np.isclose((1 + S4.daily_returns(eq)).prod() * C.INIT_CAPITAL, eq.iloc[-1], rtol=1e-10)


def test_stage4_random_signals_same_count(s4_ctx):
    """랜덤 매수: A 후보가 있는 날마다, 그 시장에서 그날 살 수 있는 종목 중 같은 수만 켠다. 2015~2019 밖은 끈다."""
    ctx = s4_ctx
    sig = pd.DataFrame(np.random.default_rng(4).random((len(ctx.dates), len(ctx.m.codes))) < 0.05,
                       index=ctx.dates, columns=ctx.m.codes)
    for mk in C.UNIVERSE_MARKETS:
        pool = ctx.allowed & (ctx.m.market == mk)
        rows = np.isin(ctx.years, C.WF_YEARS)
        r = S4.random_signals(sig, ctx, mk, np.random.default_rng([42, 0])).to_numpy(bool)
        assert not (r & ~pool).any() and not r[~rows].any()
        assert ((r.sum(axis=1) == (sig.to_numpy(bool) & pool).sum(axis=1)) | ~rows).all()
        again = S4.random_signals(sig, ctx, mk, np.random.default_rng([42, 0])).to_numpy(bool)
        assert (again == r).all()


def test_stage4_index_compare():
    """시험 7: 5년 합친 수익 = 해마다 (1+수익) 곱 − 1, 지수 이긴 해 = 초과수익 > 0 인 해 (빈 해가 있으면 비움)."""
    d = {f"포트_수익_{y}": [0.10, 0.10] for y in C.WF_YEARS}
    d.update({f"포트_지수수익_{y}": [0.05, 0.05] for y in C.WF_YEARS})
    d.update({f"포트_초과수익_{y}": [0.05, -0.01 if y == 2016 else 0.05] for y in C.WF_YEARS})
    d["포트_초과수익_2019"] = [0.05, np.nan]
    out = S4.add_index_compare(pd.DataFrame(d))
    assert np.isclose(out.loc[0, "포트_5년수익"], 1.1 ** 5 - 1) and np.isclose(out.loc[0, "지수_5년수익"], 1.05 ** 5 - 1)
    assert out.loc[0, "지수이긴해수"] == 5 and bool(out.loc[0, "지수이김_5년"])
    assert pd.isna(out.loc[1, "지수이긴해수"])


def test_combo_tables_passes_window(monkeypatch):
    """시험 5(W=2·4): combo_tables(window=W) 의 W 가 and_entry 에 그대로 간다. 안 주면 config.AND_WINDOW(3)."""
    seen = []

    class Stop(Exception):
        pass

    def fake_and(entries, window=None):
        seen.append(window)
        raise Stop

    monkeypatch.setattr(CB, "and_entry", fake_and)
    picks = {(mk, n, h, y): 0 for mk in C.UNIVERSE_MARKETS for n in ("rsi", "macd") for h in C.HOLDS
             for y in range(2010, 2021)}
    for w, want in ((None, C.AND_WINDOW), (2, 2), (4, 4)):
        kw = {} if w is None else {"window": w}
        with pytest.raises(Stop):
            CB.combo_tables(("rsi", "macd"), None, picks, {}, lambda n, k: None, log=lambda *a, **k: None, **kw)
    assert seen == [C.AND_WINDOW, 2, 4] and C.AND_WINDOW == 3


def test_tries_rows_subtracts_overlap():
    """시험 1: 고른 20개 칸이 816개 칸 안에도 있으면 '겹침 빼기' 줄로 한 번만 센다 (11/25 사용자 "1번")."""
    from indexbeater import stage4 as S4
    cells = pd.DataFrame({"종류": ["3신호20", "3신호20", "3신호816", "3신호816", "3신호816"],
                          "신호": ["a+b+c", "a+b+d", "a+b+c", "a+b+d", "a+c+d"],
                          "H": [5, 5, 5, 5, 5]})
    seg = pd.DataFrame({"토막": ["1", "2", "3", "전체"]})
    rows = {r["항목"]: r["수"] for r in S4.tries_rows("KOSPI", 10, 4, 6, seg, cells)}
    assert rows["3신호 고른 20개 칸 (20 × H3 × 비용2)"] == 4
    assert rows["3신호 816개 칸 (비교만, 816 × H3 × 비용2)"] == 6
    assert rows["겹침 빼기 (고른 20개 칸은 816개 칸 안에 있음)"] == -4
    assert rows["합계 (팀장 몫)"] == 10 + 4 + 6 + 4 + 4 + 6 - 4      # 겹친 2칸 × 비용2 = 4 를 한 번만
    # H 가 다르면 다른 칸 → 겹침 아님
    cells2 = cells.assign(H=[5, 10, 5, 5, 5])
    rows2 = {r["항목"]: r["수"] for r in S4.tries_rows("KOSPI", 0, 0, 0, seg.iloc[:0], cells2)}
    assert rows2["겹침 빼기 (고른 20개 칸은 816개 칸 안에 있음)"] == -2


# ---------------------------------------------------------------- 대표 일봉차트 조합 8개 (classic, 별개 비교분석·참고, 11/25)
@pytest.fixture(scope="module")
def classic_data():
    return LA.make_synthetic(n_days=1200, n_codes=20)


def test_classic_no_lookahead(classic_data):
    """k일 이후 가격·거래량을 바꿔도 k일 전 8개 조합 신호는 그대로 (lookahead.py 와 같은 방식)."""
    from indexbeater import classic as K
    d = classic_data
    for k_ in range(1, 9):
        base = K.entry(k_, d)
        n = len(base)
        rng = np.random.default_rng(10 + k_)
        for k in (n - 1, n // 4, n // 2, (3 * n) // 4, 30):
            assert base.iloc[:k].equals(K.entry(k_, LA.perturb(d, k, rng)).iloc[:k]), (k_, k)
    assert all(K.entry(k_, d).to_numpy().any() for k_ in range(1, 9))   # 8개 모두 한 번 이상 켜짐 (빈 시험 아님)


def test_classic_definitions(classic_data):
    """③ = 구성 3개 W=3 AND, ⑤ = ③ & 추세 상태 (⊆ ③), 2012-07-01 전 끔, 켜진 날은 정의 조건을 모두 만족."""
    from indexbeater import classic as K
    from indexbeater import indicators as I
    d = classic_data
    e3, e5 = K.entry(3, d), K.entry(5, d)
    assert e3.equals(CB.and_entry(K.three_parts(d), window=3))
    assert e5.equals(e3 & K.trend(d).state()) and not (e5 & ~e3).to_numpy().any()
    before = d["adj_close"].index < C.SIGNAL_START
    assert K.raw(6, d).fillna(False).astype(bool)[before].to_numpy().any()       # 자르기 전엔 켜진 날이 있다
    assert not any(K.entry(k, d)[before].to_numpy().any() for k in range(1, 9))
    c, h, lo, v = d["adj_close"], d["adj_high"], d["adj_low"], d["volume_s"]
    r = I.rsi(c, 14)
    tr = (c > I.sma(c, 200)) & (I.sma(c, 50) > I.sma(c, 200))
    on = lambda k: K.entry(k, d).to_numpy()
    ok = lambda m, k: m.to_numpy()[on(k)].all()
    # ① 추세 + 어제 RSI 40~50·어제 < 그제·오늘 > 어제
    assert ok(tr & (r.shift(1) >= 40) & (r.shift(1) <= 50) & (r.shift(1) < r.shift(2)) & (r > r.shift(1)), 1)
    # ②④ 20일 신고가(어제까지) + 거래량 1.5배(어제까지 평균) + ADX≥25·+DI>−DI, ④ 는 추세도
    pdi, mdi, adx = I.adx_di(h, lo, c, 14)
    brk = ((c > h.rolling(20).max().shift(1)) & (v >= 1.5 * v.rolling(20).mean().shift(1)) & (adx >= 25) & (pdi > mdi))
    assert ok(brk, 2) and ok(brk & tr, 4)
    # ⑥ 종가 < 볼린저 하단 & RSI ≤ 30, 어제는 (계산되고) 아니었음
    lower, _, _ = I.bollinger(c, 20, 2)
    c6 = (c < lower) & (r <= 30)
    assert ok(c6, 6) and ok(~c6.shift(1, fill_value=True) & lower.shift(1).notna() & r.shift(1).notna(), 6)
    # ⑦ RSI 30 상향 돌파 + OBV > 9일 평균·어제보다 큼
    o = I.obv(c, v)
    assert ok((r > 30) & (r.shift(1) <= 30) & (o > I.sma(o, 9)) & (o > o.shift(1)), 7)
    # ⑧ 최근 3일 저가 ≤ EMA10, 종가 > EMA10·어제 종가, RSI 50~70, ADX > 20, MACD 선 ≥ 0, EMA5 > EMA10
    e10 = I.ema(c, 10)
    touch = (lo <= e10) | (lo.shift(1) <= e10.shift(1)) | (lo.shift(2) <= e10.shift(2))
    line, _ = I.macd(c, 12, 26, 9)
    assert ok(tr & touch & (c > e10) & (c > c.shift(1)) & (r >= 50) & (r <= 70) & (adx > 20) & (line >= 0)
              & (I.ema(c, 5) > e10), 8)


def test_classic_portfolio_same_path(s4_ctx):
    """무작위·비용 시험에 넘기는 신호 표 = 점수표 포트폴리오(scorecard._portfolio)가 쓰는 것 (0.5% 결과 끝자리까지 같음)."""
    from indexbeater import classic as K
    from indexbeater import stage4 as S4
    ctx = s4_ctx
    done = 0
    for k in range(1, 9):
        ent = K.entry(k, ctx.data)
        for mk in C.UNIVERSE_MARKETS:
            if not CB.has_portfolio_candidate(ctx, mk, {y: 0 for y in C.WF_YEARS}, {0: ent}):
                continue
            ref, _ = SC._portfolio("t", mk, 5, {y: {"set": 0} for y in C.WF_YEARS}, {0: ent}, ctx)
            out, _ = S4.portfolio_at_cost(K.signal_table(ctx, k), ctx, mk, 5)
            assert out == ref, (k, mk)
            done += 1
    assert done >= 8


def test_classic_index_above_200_and_blank(monkeypatch):
    """⑥ 참고 열: 지수가 200일 평균 위인가 (평균이 없는 앞 199일은 빈칸). 포트폴리오 없는 칸의 5년 수익은 0 이 아니라 빈칸."""
    from indexbeater import benchmark as B
    from indexbeater import classic as K
    days = pd.bdate_range("2014-01-01", periods=260)
    val = np.r_[np.linspace(100, 200, 230), np.full(30, 50.0)]
    fake = pd.DataFrame({"date": days, "market": "KOSPI", "index": val, "ret": 0.0, "source": "t"})
    monkeypatch.setattr(B, "load_market", lambda *a, **k: fake)
    s = K.index_above_200("KOSPI")
    assert s.iloc[:199].isna().all() and bool(s.iloc[199]) and bool(s.iloc[229]) and not bool(s.iloc[-1])
    row = {"비용": "0.5%", **{f"포트_수익_{y}": np.nan for y in C.WF_YEARS},
           **{f"포트_지수수익_{y}": np.nan for y in C.WF_YEARS}, **{f"포트_초과수익_{y}": np.nan for y in C.WF_YEARS}}
    good = {**row, **{f"포트_수익_{y}": 0.1 for y in C.WF_YEARS}, **{f"포트_지수수익_{y}": 0.0 for y in C.WF_YEARS},
            **{f"포트_초과수익_{y}": 0.1 for y in C.WF_YEARS}}
    out = K.cells(pd.DataFrame([row, good, {**good, "비용": "0.25%"}]))
    assert len(out) == 2 and pd.isna(out.loc[0, "포트_5년수익"]) and pd.isna(out.loc[0, "지수이긴해수"])
    assert np.isclose(out.loc[1, "포트_5년수익"], 1.1 ** 5 - 1)


# ---------------------------------------------------------------- 다음 날 종가 매도 비교 (close_exit, 11/26 참고)
def _rand_sig(ctx, seed, p=0.03):
    return pd.DataFrame(np.random.default_rng(seed).random((len(ctx.dates), len(ctx.m.codes))) < p,
                        index=ctx.dates, columns=ctx.m.codes)


def test_close_exit_open_same_as_engine(s4_ctx):
    """옮겨 적은 계산을 '5거래일 뒤 시가'로 돌리면 엔진 매매 표·대조군 H5·점수표 매매 표·포트폴리오와 끝자리까지 같다."""
    from indexbeater import close_exit as X
    ctx = s4_ctx
    sig = _rand_sig(ctx, 7)
    for dd in (True, False):
        ref = E.simulate_trades(sig, ctx.data, ExitRule(hold=5), dedupe=dd, _m=ctx.m)
        got = X.simulate(sig, ctx.data, X.OPEN, 5, dedupe=dd, m=ctx.m)
        pd.testing.assert_frame_equal(got, ref, check_dtype=False)
    assert X.same_control(X.Control(ctx.data, ctx.m, ctx.cell, X.OPEN, 5), ctx.base[5])
    ref_ft, ref_st, _ = SC._trade_table(ctx, sig, 5)
    ft, st = X.trade_table(ctx, sig, X.OPEN, 5, ctx.base[5])
    pd.testing.assert_frame_equal(ft[ref_ft.columns], ref_ft, check_dtype=False)
    pd.testing.assert_frame_equal(st.reset_index(drop=True), ref_st.reset_index(drop=True), check_dtype=False)
    for mk in C.UNIVERSE_MARKETS:
        ref_d, _ = E.run_portfolio(sig, ctx.data, ExitRule(hold=5), market=mk)
        got_d, _ = X.run_portfolio(sig, ctx.data, mk, X.OPEN, 5, m=ctx.m)
        pd.testing.assert_frame_equal(got_d, ref_d)


def test_close_exit_close_rule(s4_ctx):
    """종가 매도: 매수 다음 날부터 팔 수 있는 첫 날의 종가에 판다. 그 사이 날은 모두 못 파는 날. 못 찾고 자료가 끝나면
    마지막 종가 정산 (상장폐지). 매수 쪽(체결·취소·매수불가·매수가)은 시가 매도와 같다."""
    from indexbeater import close_exit as X
    ctx, m = s4_ctx, s4_ctx.m
    sig = _rand_sig(ctx, 8, 0.05)
    cl = X.simulate(sig, ctx.data, X.CLOSE, 1, dedupe=False, m=m)
    op = X.simulate(sig, ctx.data, X.OPEN, 5, dedupe=False, m=m)
    for c in ("code", "t_idx", "b_idx", "buy_px"):
        assert np.array_equal(cl[c].to_numpy(), op[c].to_numpy(), equal_nan=True) if c == "buy_px" else \
            (cl[c].to_numpy() == op[c].to_numpy()).all()
    same_buy = cl["status"].isin([E.GAP_CANCEL, E.NO_BUY])
    assert (same_buy == op["status"].isin([E.GAP_CANCEL, E.NO_BUY])).all()
    f = cl[cl["status"] == E.FILLED]
    j = pd.Index(m.codes).get_indexer(f["code"])
    b, s = f["b_idx"].to_numpy(int), f["s_idx"].to_numpy(int)
    sold = (f["reason"] == X.CLOSE_REASON).to_numpy()
    assert sold.sum() > 100 and not f["sell_at_open"].any()
    assert (s[sold] >= b[sold] + 1).all() and m.sellable[s[sold], j[sold]].all()
    assert np.array_equal(f["sell_px"].to_numpy()[sold], m.Cl[s[sold], j[sold]])
    assert (f["delay"].to_numpy()[sold] == s[sold] - b[sold] - 1).all()
    for q in np.nonzero(sold & (s > b + 1))[0]:                    # 미룬 매매: 그 사이 날은 못 파는 날
        assert not m.sellable[b[q] + 1:s[q], j[q]].any()
    dl = (f["reason"] == E.DELIST).to_numpy()
    assert (s[dl] == m.last[j[dl]]).all() and np.array_equal(f["sell_px"].to_numpy()[dl], m.Cl[s[dl], j[dl]])
    assert not (sold & dl).any() and (sold | dl).all()


def test_close_exit_halt_moves_sale(synthetic_long):
    """매수 다음 날 거래정지(거래량 0)면 그날 못 팔고 다음 거래일 종가에 판다 (delay 1)."""
    from indexbeater import close_exit as X
    data = data_io.to_wide(synthetic_long)
    m = E._Market(data)
    sig = pd.DataFrame(np.random.default_rng(9).random((m.n, len(m.codes))) < 0.03, index=m.dates, columns=m.codes)
    tr = X.simulate(sig, data, X.CLOSE, 1, dedupe=False, m=m)
    q = tr[(tr["status"] == E.FILLED) & (tr["delay"] == 0) & (tr["reason"] == X.CLOSE_REASON)]
    jj = pd.Index(m.codes).get_indexer(q["code"])
    s = q["s_idx"].to_numpy(int)
    ok = (s + 1 < m.n - 1) & m.sellable[np.minimum(s + 1, m.n - 1), jj]
    r = q[ok].iloc[len(q[ok]) // 2]
    jx, sx = int(pd.Index(m.codes).get_loc(r["code"])), int(r["s_idx"])
    d2 = dict(data)
    vol = data["volume"].copy()
    vol.iloc[sx, jx] = 0.0
    d2["volume"] = vol
    m2 = E._Market(d2)
    one = sig.copy() & False
    one.iloc[int(r["t_idx"]), jx] = True
    got = X.simulate(one, d2, X.CLOSE, 1, m=m2)
    g = got[got["status"] == E.FILLED].iloc[0]
    assert int(g["s_idx"]) == sx + 1 and int(g["delay"]) == 1 and g["sell_px"] == m2.Cl[sx + 1, jx]


def test_close_exit_no_lookahead(classic_data):
    """k일부터 가격·거래량을 바꿔도 k일 전에 판 매매(신호·매수·매도·가격)는 그대로 (두 매도 방식 모두)."""
    from indexbeater import close_exit as X
    d = classic_data
    n = len(d["adj_close"])
    sig = pd.DataFrame(np.random.default_rng(11).random(d["adj_close"].shape) < 0.05,
                       index=d["adj_close"].index, columns=d["adj_close"].columns)
    cols = ["code", "t_idx", "b_idx", "s_idx", "status", "reason", "buy_px", "sell_px", "delay"]
    for mode, k_ in ((X.CLOSE, 1), (X.OPEN, 5)):
        base = X.simulate(sig, d, mode, k_, dedupe=False)
        rng = np.random.default_rng(12)
        for k in (n // 4, n // 2, (3 * n) // 4, n - 3):
            new = X.simulate(sig, LA.perturb(d, k, rng), mode, k_, dedupe=False)
            done = ((base["s_idx"] >= 0) & (base["s_idx"] < k)).to_numpy()
            assert done.sum() > 0
            a = base.loc[done, cols].reset_index(drop=True)
            b = new.set_index(["code", "t_idx"]).loc[list(zip(a["code"], a["t_idx"]))].reset_index()[cols]
            pd.testing.assert_frame_equal(a, b, check_dtype=False, obj=f"{mode} k={k}")


def test_close_exit_portfolio_slot_held_on_sale_day(s4_ctx):
    """종가 매도 날에는 그 자리가 아직 차 있다: 자리 1개, A 를 T 신호로 사서 T+2 종가에 팔면, T+1 신호의 B 는
    T+2 시가에 못 산다. 같은 상황에서 시가 매도(k=1)면 A 를 T+2 시가에 먼저 팔아 B 를 산다."""
    from indexbeater import close_exit as X
    ctx, m = s4_ctx, s4_ctx.m
    mk = C.UNIVERSE_MARKETS[0]
    cand = X.simulate(_rand_sig(ctx, 13, 0.05), ctx.data, X.CLOSE, 1, dedupe=False, m=m)
    cand = cand[(cand["market"] == mk) & (cand["status"] == E.FILLED)]
    a = cand[(cand["delay"] == 0) & (cand["reason"] == X.CLOSE_REASON)]
    pair = None
    for r in a.itertuples(index=False):
        bb = cand[(cand["t_idx"] == r.t_idx + 1) & (cand["code"] != r.code)]
        if len(bb):
            pair = (r, bb.iloc[0])
            break
    assert pair is not None
    A, Bq = pair
    sig = pd.DataFrame(False, index=ctx.dates, columns=m.codes)
    sig.loc[ctx.dates[A.t_idx], A.code] = True
    sig.loc[ctx.dates[Bq["t_idx"]], Bq["code"]] = True
    ja = int(pd.Index(m.codes).get_loc(A.code))
    if not m.sellable[A.b_idx + 1, ja]:
        pytest.skip("A 가 T+2 시가에 못 파는 날")
    _, log_c = X.run_portfolio(sig, ctx.data, mk, X.CLOSE, 1, max_positions=1, m=m)
    _, log_o = X.run_portfolio(sig, ctx.data, mk, X.OPEN, 1, max_positions=1, m=m)
    bought = lambda lg: set(lg.loc[lg["shares"] > 0, "code"])
    assert bought(log_c) == {A.code}
    assert bought(log_o) == {A.code, Bq["code"]}


# ---------------------------------------------------------------- 팀원 최적 전략 동일 조건 비교 (team_compare, 참고, 11/26)
def test_team_compare_no_lookahead(classic_data):
    """k일 이후 가격·거래량을 바꿔도 k일 전 김동겸 03·김승찬 두 신호는 그대로."""
    from indexbeater import team_compare as T
    d = classic_data
    for f in (T.entry_03, T.entry_d3, T.entry_d4):
        base = f(d)
        n = len(base)
        rng = np.random.default_rng(21)
        for k in (n - 1, n // 4, n // 2, (3 * n) // 4, 30):
            assert base.iloc[:k].equals(f(LA.perturb(d, k, rng)).iloc[:k]), (f.__name__, k)
    assert T.entry_d3(d).to_numpy().any()


def test_team_compare_definitions(classic_data):
    """다-3 = rsi·bollinger W=3 AND, 다-4 = 다-3 & 그날 %K ≤ 20 (⊆ 다-3), 03 켜진 날 = 0.80 ≤ 이격도 < 0.85·모멘텀 ≥ 0."""
    from indexbeater import indicators as I
    from indexbeater import signals as S
    from indexbeater import team_compare as T
    d = classic_data
    d3, d4 = T.entry_d3(d), T.entry_d4(d)
    want = CB.and_entry([S.entry_signal("rsi", d, {"period": 14, "low": 25}),
                         S.entry_signal("bollinger", d, {"period": 10, "k": 1.5})], window=3)
    assert d3.equals(want.fillna(False).astype(bool) & (d["adj_close"].index >= C.SIGNAL_START)[:, None])
    pk = I.stochastic(S._h(d), S._l(d), S._c(d), 14, 3)[0]
    assert not (d4 & ~d3).to_numpy().any()
    assert d4.equals(d3 & (pk <= 20))
    e03 = T.entry_03(d)
    disp, mom = T.disparity_momentum(d)
    on = e03.to_numpy()
    assert (disp.to_numpy()[on] >= 0.80).all() and (disp.to_numpy()[on] < 0.85).all() and (mom.to_numpy()[on] >= 0).all()
    assert not (e03 & ~S.entry_signal("disparity", d, {"ma": 20, "thr": 0.85}).fillna(False)).to_numpy().any()


def test_team_compare_07_window_and_rank():
    """07 학습 표본 = 검증 해 첫날 전에 팔았고 예정 매도일도 그 전인 것만. 날짜마다 점수순 상위 ceil(30%)."""
    from indexbeater import team_compare as T
    idx = pd.MultiIndex.from_tuples([(90, "A"), (93, "B"), (94, "C"), (80, "D")], names=["t_idx", "code"])
    learn = pd.DataFrame({"s_idx": [96, 99, 99, 101], "ret_gross": 0.0}, index=idx)
    kept = T.keep_learn(learn, start=100, hold=5)          # 90+6=96 ✓ · 93+6=99 ✓ · 94+6=100 ✗ · s_idx 101 ✗
    assert list(kept.index.get_level_values("code")) == ["A", "B"]
    pred = pd.DataFrame({"t_idx": [1] * 7 + [2] * 2, "code": list("abcdefgxy"),
                         "score": [.1, .9, .5, .5, .3, .2, .8, .4, .6], "value20": [1, 1, 2, 1, 1, 1, 1, 1, 1]})
    r = T.rank_day(pred)
    d1 = r[r["t_idx"] == 1]
    assert list(d1["code"]) == ["b", "g", "c", "d", "e", "f", "a"]       # 같은 점수 → 거래대금 큰 c 먼저
    assert list(d1["priority"]) == [7, 6, 5, 4, 3, 2, 1]
    assert d1["selected"].sum() == 3 and r[r["t_idx"] == 2]["selected"].sum() == 1   # ceil(2.1)=3, ceil(0.6)=1


def test_team_compare_reported_table():
    """김동겸 표 24줄 = 시장 2 × 비용 2 × H 3 × (03, 07), 겹치는 줄 없음, % 열은 ÷100."""
    from indexbeater import team_compare as T
    rep = T.reported()
    assert len(rep) == 24 and not rep.duplicated(["시장", "비용", "H", "ID"]).any()
    r = rep[(rep["시장"] == "KOSPI") & (rep["비용"] == 0.005) & (rep["H"] == 5) & (rep["ID"] == "03")].iloc[0]
    assert r["점수"] == 79.2 and r["청산거래"] == 1246 and abs(r["포트MDD"] + 0.2415) < 1e-12


# ---------------------------------------------------------------- TOP3 매수 신호 CSV (buy_export, 11/26)
def test_top3csv_final_cut(s4_ctx):
    """최종 모델 cut = 2020 첫날(= 자료 끝) − 21거래일 = scorecard 의 2020 cut. 2020 자료가 있으면 멈춘다."""
    from indexbeater import buy_export as BX
    start, cut = BX.final_cut(s4_ctx.dates)
    assert start == len(s4_ctx.dates) == s4_ctx.start_idx[2020]
    assert cut == start - C.WF_CUT_OFFSET == s4_ctx.cut_idx[2020]
    with pytest.raises(SystemExit):
        BX.final_cut(s4_ctx.dates.append(pd.DatetimeIndex([pd.Timestamp("2020-01-02")])))


def test_top3csv_final_07_window(s4_ctx, monkeypatch):
    """다-2 최종 모델: 학습 매매는 cut 까지 신호 · 2020 첫날 전에 팔았고 예정 매도일도 그 전.
    점수 매긴 후보 = 2015 전 · 03|04|05 띠 안. 2015 이후 줄은 0."""
    from indexbeater import buy_export as BX
    from indexbeater import team_compare as T
    seen = []
    real = T.fit_07

    def spy(train, holdout, truth):
        seen.append((train.copy(), holdout.copy()))
        return real(train, holdout, truth)

    monkeypatch.setattr(T, "fit_07", spy)
    order, folds = BX.final_07(s4_ctx, log=lambda *a: None)
    start, cut = BX.final_cut(s4_ctx.dates)
    assert len(seen) == len(C.UNIVERSE_MARKETS) == len(folds)
    for (train, hold_out), mk in zip(seen, C.UNIVERSE_MARKETS):
        t = train["t_idx"].to_numpy()
        assert len(train) and (t <= cut).all() and (train["s_idx"].to_numpy() < start).all()
        assert (t + 1 + BX.HOLD < start).all()
        assert train["market"].eq(mk).all() and hold_out["market"].eq(mk).all()
        assert (s4_ctx.years[hold_out["t_idx"].to_numpy()] < C.WF_YEARS[0]).all()
    bands = T.band_rules(s4_ctx.data)
    mask = (bands["03"] | bands["04"] | bands["05"]).to_numpy()
    on = order.to_numpy() > 0
    assert on.any() and not (on & ~mask).any()
    assert not on[s4_ctx.years >= C.WF_YEARS[0]].any()


def test_top3csv_splice(s4_ctx):
    """2012~2014 줄 = 최종 모델 표, 2015~ = 해마다 표, 다른 시장 종목·엔진이 안 보는 칸은 0."""
    from indexbeater import buy_export as BX
    ctx = s4_ctx
    shape = (len(ctx.dates), len(ctx.m.codes))
    wf = pd.DataFrame(np.full(shape, 3), index=ctx.dates, columns=ctx.m.codes)
    fin = pd.DataFrame(np.full(shape, 7), index=ctx.dates, columns=ctx.m.codes)
    v, drop = BX.splice(wf, fin, ctx, "KOSPI")
    v = v.to_numpy()
    early = (ctx.years < C.WF_YEARS[0])[:, None]
    ok = ctx.allowed & (ctx.m.market == "KOSPI")
    assert (v[ok & early] == 7).all() and (v[ok & ~early] == 3).all()
    assert (v[~ok] == 0).all()
    assert drop == int((~ctx.allowed & (ctx.m.market == "KOSPI")).sum())


def test_top3csv_rows_roundtrip(s4_ctx):
    """신호 줄: 매수가 = T+1 수정 시가, 상태 = 엔진 simulate_trades. CSV 글자로 바꿨다 다시 읽어 엔진에 넣으면
    같은 종목을 같은 날 산다. '실제로샀다_2015부터' 는 2015 전 빈칸."""
    from indexbeater import buy_export as BX
    ctx = s4_ctx
    e = S.entry_signal("rsi", ctx.data, {"period": 14, "low": 30}).fillna(False).astype(int)
    val, _ = BX.splice(e, e, ctx, "KOSDAQ")
    meta = {"시장": "KOSDAQ", "TOP3순서": 1, "PDF순위": 1, "구분": "시험", "전략": "rsi"}
    df = BX.rows_for(val, ctx, meta, log=lambda *a: None)
    assert len(df) == int((val.to_numpy() > 0).sum()) > 0
    tr = E.simulate_trades(val > 0, ctx.data, ExitRule(hold=BX.HOLD), dedupe=False, _m=ctx.m)
    assert sorted(df["매수상태"]) == sorted(tr["status"])
    f = df[df["매수상태"].eq(E.FILLED)]
    t = ctx.dates.get_indexer(pd.to_datetime(f["신호일"]))
    j = pd.Index(ctx.m.codes).get_indexer(f["종목코드"])
    assert (f["매수가_수정시가"].to_numpy() == ctx.m.O[t + 1, j]).all()
    early = pd.to_datetime(df["신호일"]).dt.year < C.WF_YEARS[0]
    assert df.loc[early, "실제로샀다_2015부터"].isna().all() and df.loc[~early, "실제로샀다_2015부터"].notna().all()
    back = BX.read_back(df.to_csv(index=False, lineterminator=chr(10)))
    assert back["종목코드"].str.len().eq(6).all()                 # 앞자리 0 이 안 없어진다
    v = back.pivot(index="신호일", columns="종목코드", values="매수순서값")
    v.index = pd.to_datetime(v.index)
    v = v.reindex(index=ctx.dates, columns=ctx.m.codes).fillna(0).astype(int)
    assert v.equals(val)
    got, eq = BX.bought_set(v, ctx, "KOSDAQ")
    want, eq0 = BX.bought_set(val, ctx, "KOSDAQ")
    assert got == want and eq.equals(eq0)
    b = df[df["실제로샀다_2012부터"]]
    assert len(want) and set(zip(ctx.dates.get_indexer(pd.to_datetime(b["신호일"])), b["종목코드"])) == want
    assert b["매수상태"].isin([E.FILLED, E.OPEN_END]).all()


# ---------------------------------------------------------------- 매도 ML·DL 공통 시작 코드 (sell_common, 11/26)
def _sell_sig(data, seed=3, p=0.02):
    """가짜 TOP3 CSV: 시장마다 2칸, 엔진이 보는 칸(유동성·2012-07 이후·그 시장)에서만 신호."""
    m = E._Market(data)
    rng = np.random.default_rng(seed)
    ok = data["tradable"].to_numpy(bool) & np.asarray(m.dates >= C.SIGNAL_START)[:, None]
    rows = []
    for mk in C.UNIVERSE_MARKETS:
        for cell in ("가", "나"):
            t, j = np.nonzero(ok & (m.market == mk) & (rng.random(ok.shape) < p))
            rows.append(pd.DataFrame({"시장": mk, "구분": cell, "전략": f"s_{cell}", "신호일": m.dates[t],
                                      "종목코드": m.codes.to_numpy()[j], "매수순서값": rng.integers(1, 4, len(t))}))
    return pd.concat(rows, ignore_index=True)


def _sell_index(data):
    return {mk: data["adj_close"].where(data["market"] == mk).mean(axis=1) for mk in C.UNIVERSE_MARKETS}


def _sell_ctx(data, sig=None):
    from indexbeater import sell_common as SL
    return SL.SellCtx(data, _sell_sig(data) if sig is None else sig, index_level=_sell_index(data), log=lambda *a: None)


def _same_row(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        (x != x and y != y) or x == y for x, y in ((a[k], b[k]) for k in a))   # nan 끼리는 같다고 본다


@pytest.fixture(scope="module")
def sell_ctx(s4_ctx):
    return _sell_ctx(s4_ctx.data)


def test_sellml_fixed_and_portfolio_same_as_engine(sell_ctx):
    """옮겨 적은 매도 정산을 고정 H5·H10·H20 으로 돌리면 엔진 매매 표와 끝자리까지 같다.
    옮겨 적은 포트폴리오에 그 표를 넣으면 engine.run_portfolio 와 날마다 같다 (매수 순서 값 = 신호 개수)."""
    from indexbeater import sell_common as SL
    ctx, m = sell_ctx, sell_ctx.m
    s = ctx.sig
    for k in SL.FIXED_H:
        for mk in C.UNIVERSE_MARKETS:
            part = s[s["시장"] == mk]
            ref = E.simulate_trades(SL.entry_table(m, part["t"], part["j"]), ctx.data, ExitRule(hold=k),
                                    dedupe=False, _m=m)
            ref = ref[ref["market"] == mk].reset_index(drop=True)
            tj = np.unique(np.column_stack([part["t"], part["j"]]), axis=0)
            got = SL.to_table(SL.fixed_record(m, tj[:, 0], tj[:, 1], k), m).drop(columns="j")
            pd.testing.assert_frame_equal(got, ref, check_dtype=False)
    cnt = np.zeros((m.n, len(m.codes)), dtype=int)
    np.add.at(cnt, (s["t"].to_numpy(), s["j"].to_numpy()), 1)
    cnt = pd.DataFrame(cnt, index=m.dates, columns=m.codes)
    for mk in C.UNIVERSE_MARKETS:
        part = s[s["시장"] == mk]
        tj = np.unique(np.column_stack([part["t"], part["j"]]), axis=0)
        tab = SL.to_table(SL.fixed_record(m, tj[:, 0], tj[:, 1], 5), m)
        tab["n_sig"] = cnt.to_numpy()[tab["t_idx"], tab["j"]]
        tab["v20"] = m.value20[tab["t_idx"], tab["j"]]
        for cost in (C.COST_SELECT, C.COST_ROUNDTRIP):
            ref_d, ref_l = E.run_portfolio(cnt, ctx.data, ExitRule(hold=5), market=mk, cost=cost)
            got_d, got_l = SL.run_portfolio_cand(tab, m, ctx.close_ff, cost=cost)
            pd.testing.assert_frame_equal(got_d, ref_d)
            assert len(got_l) == len(ref_l) > 0


def test_sellml_settle_rule(sell_ctx):
    """d일 장 끝에 판다 → d 다음 '팔 수 있는 첫 날' 시가 (거래정지일은 건너뜀). 없으면 상장폐지 = 마지막 종가,
    자료 끝까지 없으면 미완료. 하나씩 따라가며 찾은 답과 같다."""
    from indexbeater import sell_common as SL
    m = sell_ctx.m
    rng = np.random.default_rng(0)
    j = rng.integers(0, len(m.codes), 3000)
    d = rng.integers(0, m.n, 3000)
    st = SL.settle(m, j, d)
    halted = 0
    for q in range(len(j)):
        nxt = [i for i in range(d[q] + 1, m.n) if m.sellable[i, j[q]]]
        if nxt:
            assert st["s_idx"][q] == nxt[0] and st["sell_px"][q] == m.O[nxt[0], j[q]] and st["sell_at_open"][q]
            halted += nxt[0] > d[q] + 1
        elif m.last[j[q]] < m.n - 1:
            assert st["reason"][q] == E.DELIST and st["s_idx"][q] == m.last[j[q]]
            assert st["sell_px"][q] == m.Cl[m.last[j[q]], j[q]]
        else:
            assert st["status"][q] == E.OPEN_END and st["s_idx"][q] == -1
    assert halted > 0                                              # 거래정지로 미룬 경우가 실제로 있었다


def test_sellml_label_purge(sell_ctx):
    """라벨 = (d+20 판단 매도가 ÷ d 판단 매도가 − 1) < −0.5%. 라벨 끝 = d+20 판단의 매도일 (상장폐지 아니면 > d+20).
    시험 해 Y 의 학습 줄은 모두 라벨 끝 + 20 < Y 첫날, 앞 80% 는 검증 첫날 전, 검증 줄은 검증 첫날부터."""
    from indexbeater import sell_common as SL
    ctx = sell_ctx
    for mk in C.UNIVERSE_MARKETS:
        P = SL.build_panel(ctx, mk)
        assert len(P["y"]) > 1000 and 0.2 < P["y"].mean() < 0.8
        a, b = SL.settle(ctx.m, P["j"], P["d"]), SL.settle(ctx.m, P["j"], P["d"] + SL.LOOK)
        assert np.allclose(P["diff"], b["sell_px"] / a["sell_px"] - 1)
        assert (P["y"] == (P["diff"] < -SL.MARGIN)).all()
        assert (P["label_end"] == b["s_idx"]).all() and (P["label_end"] >= P["d"]).all()
        assert ((P["label_end"] > P["d"] + SL.LOOK) | (b["reason"] == E.DELIST)).all()
        held = P["d"] - ctx.pool[mk]["b"][P["trade"]]
        assert held.min() == 0 and held.max() <= SL.PANEL_DAYS - 1
        for y in ctx.test_years:
            sp = SL.split_rows(ctx, P, y)
            le = P["label_end"]
            assert sp["pool"].any() and sp["fit"].sum() < sp["pool"].sum()
            assert (le[sp["pool"]] + SL.GAP < sp["sY"]).all() and (le[sp["fit"]] + SL.GAP < sp["val_start"]).all()
            assert (P["d"][sp["val"]] >= sp["val_start"]).all() and not (sp["fit"] & sp["val"]).any()
            assert sp["sY"] - sp["val_start"] == round(SL.VAL_FRAC * (sp["sY"] - ctx.ystart[ctx.first_year]))


def test_sellml_no_lookahead(s4_ctx):
    """k일부터 가격·거래량을 마구 바꿔도 (1) k 전 날의 특징 (2) 라벨 끝이 k 전인 학습 줄 (3) k = Y 첫날일 때 Y 해의
    학습 결과(고른 기준·검증 성적·모델 확률) 는 그대로다."""
    from indexbeater import sell_common as SL
    sig = _sell_sig(s4_ctx.data)
    a = _sell_ctx(s4_ctx.data, sig)
    y = a.test_years[1]
    k = a.ystart[y]
    b = _sell_ctx(LA.perturb(s4_ctx.data, k, np.random.default_rng(5)), sig)
    for f in SL.DAY_FEATS:
        assert np.array_equal(a.feat[f][:k], b.feat[f][:k], equal_nan=True), f
    for mk in C.UNIVERSE_MARKETS:
        assert np.array_equal(a.mkt[mk][:k], b.mkt[mk][:k], equal_nan=True)
        Pa, Pb = SL.build_panel(a, mk), SL.build_panel(b, mk)

        def keys(ctx, P):
            t = ctx.pool[mk]["t"][P["trade"]]
            return {(t[q], P["j"][q], P["d"][q]): q for q in np.nonzero(P["label_end"] < k)[0]}
        ka, kb = keys(a, Pa), keys(b, Pb)
        assert ka.keys() == kb.keys() and len(ka) > 1000
        qa, qb = np.array(list(ka.values())), np.array([kb[x] for x in ka])
        for c in ("X", "y", "diff", "label_end"):
            assert np.array_equal(Pa[c][qa], Pb[c][qb], equal_nan=True), c
        ma, ta, ra = SL.fit_year(a, mk, Pa, y, "logistic", SL.GRIDS["logistic"][1])
        mb, tb, rb = SL.fit_year(b, mk, Pb, y, "logistic", SL.GRIDS["logistic"][1])
        assert ta == tb and _same_row(ra, rb)
        R = SL.rows_of(Pa, Pa["label_end"] < k)
        assert np.array_equal(ma.prob(R), mb.prob(R))


class _HeldModel:
    """보유 거래일 ≥ at 이면 판다고 하는 가짜 모델."""
    def __init__(self, at):
        self.at = at

    def prob(self, R):
        from indexbeater import sell_common as SL
        return (R["X"][:, SL.FEATURES.index("held")] >= self.at).astype(float)


def test_sellml_long_hold_and_chunks(sell_ctx):
    """최대 보유 한도가 없다: '안 판다' 모델이면 자료 끝까지 들고(미완료·상장폐지), 75일째 판다 모델이면 60일 묶음을
    넘어서도 75일째(그날 값이 비면 그 뒤 첫 날) 판단한다. 묶어 계산한 특징 = 한 번에 계산한 특징 (최고가 이어 받기 포함)."""
    from indexbeater import sell_common as SL
    ctx, m = sell_ctx, sell_ctx.m
    mk = C.UNIVERSE_MARKETS[0]
    T = ctx.test[mk]
    yrs = list(ctx.test_years)
    never = SL.ml_decide(ctx, mk, {y: _HeldModel(10 ** 6) for y in yrs}, {y: 0.5 for y in yrs})
    assert (never == m.n).all()
    tab = SL.to_table(SL.trade_record(m, T["t"], T["j"], never), m)
    bought = tab[tab["buy_px"].notna()]
    assert len(bought) > 100 and bought["status"].isin([E.OPEN_END, E.FILLED]).all()
    assert (bought.loc[bought["status"] == E.FILLED, "reason"] == E.DELIST).all()
    dec = SL.ml_decide(ctx, mk, {y: _HeldModel(75) for y in yrs}, {y: 0.5 for y in yrs})
    a = np.nonzero(T["ok"])[0]
    W = 240
    X, ok, d, _ = SL.trade_block(ctx, mk, T["b"][a], T["j"][a], T["buy_px"][a], 0, W, T["buy_px"][a])
    yr = ctx.years[np.minimum(d, m.n - 1)]
    hit = ok & (d - T["b"][a][:, None] >= 75) & np.isin(yr, yrs)
    has = hit.any(axis=1)
    want = d[np.arange(len(a)), hit.argmax(axis=1)]
    assert has.sum() > 50 and (dec[a][has] == want[has]).all()
    assert (dec[a][~has] >= np.minimum(T["b"][a][~has] + W, m.n)).all()      # 240일 안에 없으면 그 뒤 (또는 끝까지)
    X1, ok1, _, pk1 = SL.trade_block(ctx, mk, T["b"][a], T["j"][a], T["buy_px"][a], 0, 60, T["buy_px"][a])
    X2, ok2, _, _ = SL.trade_block(ctx, mk, T["b"][a], T["j"][a], T["buy_px"][a], 60, 60, pk1)
    both = np.concatenate([ok1, ok2], axis=1)
    assert np.array_equal(both, ok[:, :120])
    assert np.array_equal(np.concatenate([X1, X2], axis=1)[both], X[:, :120][both])


def test_sellml_weights_and_threshold(sell_ctx):
    """가중치: 평균 1, 같은 매매 안에서는 |차이| 에 비례, 매매마다 합 ∝ 평균 |차이| (줄 수로 나눔). balance 면 두 쪽 합이 같다.
    기준 확률: 검증 평균 순수익이 가장 큰 것, 같으면 앞 (모두 '안 판다'면 세 기준 값이 같아 0.55)."""
    from indexbeater import sell_common as SL
    ctx = sell_ctx
    mk = C.UNIVERSE_MARKETS[1]
    P = SL.build_panel(ctx, mk)
    sp = SL.split_rows(ctx, P, ctx.test_years[2])
    rows = np.nonzero(sp["pool"])[0]
    w = SL.weights(P, rows, balance=False)
    assert np.isclose(w.mean(), 1)
    df = pd.DataFrame({"tr": P["trade"][rows], "w": w, "a": np.abs(P["diff"][rows])})
    g = df.groupby("tr").agg(w=("w", "sum"), a=("a", "mean"))
    g = g[g["a"] > 0]
    assert np.allclose(g["w"] / g["a"], (g["w"] / g["a"]).iloc[0])
    wb = SL.weights(P, rows, balance=True)
    y = P["y"][rows]
    assert np.isclose(wb.mean(), 1) and np.isclose(wb[y == 1].sum(), wb[y == 0].sum())
    thr, info = SL.pick_threshold(ctx, mk, _HeldModel(10 ** 6), sp)
    vals = [info[f"검증평균_{t:.2f}"] for t in SL.THRESHOLDS]
    assert thr == SL.THRESHOLDS[0] and info["검증매매수"] > 0 and vals[0] == vals[1] == vals[2]
    thr2, info2 = SL.pick_threshold(ctx, mk, _HeldModel(0), sp)          # 늘 판다 → 세 기준 같음 → 앞
    assert thr2 == SL.THRESHOLDS[0] and info2[f"검증평균_{SL.THRESHOLDS[0]:.2f}"] != vals[0]


def test_sellml_models_fit_predict(sell_ctx):
    """모델 7종 (계획서 5 + xgboost·catboost) 이 같은 틀로 학습·확률을 낸다. 설정 수 = 계획서 표.
    한쪽 라벨만 있으면 그 비율을 그대로 준다. 저장소 폴더에 파일을 만들지 않는다."""
    import os
    from indexbeater import sell_common as SL
    ctx = sell_ctx
    mk = C.UNIVERSE_MARKETS[0]
    P = SL.build_panel(ctx, mk)
    q = np.sort(np.random.default_rng(0).choice(len(P["y"]), 3000, replace=False))
    sub = {k: v[q] for k, v in P.items()}
    R, y, w = SL.rows_of(sub, np.ones(len(q), bool)), sub["y"], SL.weights(sub, np.arange(len(q)))
    assert {k: len(v) for k, v in SL.GRIDS.items() if k in C.ML_GRID_SIZES} == dict(C.ML_GRID_SIZES)
    assert len(SL.GRIDS["xgboost"]) == len(SL.GRIDS["catboost"]) == 5
    before = set(os.listdir("."))
    for kind in ("logistic", "random_forest", "lightgbm", "xgboost", "catboost", "lstm", "cnn1d"):
        mdl = SL.Model(kind, SL.GRIDS[kind][0], seq=ctx.seq, dl_max_rows=2000, epochs=1).fit(R, y, w, R, y, w)
        p = mdl.prob(R)
        assert p.shape == (len(q),) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), kind
        if kind not in SL.DL_KINDS:
            assert len(mdl.importance()) == len(SL.FEATURES), kind
    assert set(os.listdir(".")) == before
    one = SL.Model("lightgbm", SL.GRIDS["lightgbm"][0]).fit(R, np.zeros(len(q), int), w)
    assert (one.prob(R) == 0).all()


def test_sellml_csv_checks(s4_ctx):
    """신호 CSV: 엔진이 안 보는 칸(유동성 못 넘음)·내 데이터에 없는 종목·같은 줄 두 번이면 멈춘다. 2015 전 줄은 버린다."""
    from indexbeater import sell_common as SL
    data = s4_ctx.data
    sig = _sell_sig(data)
    ctx = _sell_ctx(data, sig)
    assert len(ctx.sig) == int((sig["신호일"].dt.year >= SL.FIRST_YEAR).sum()) < len(sig)
    m = ctx.m
    t, j = np.nonzero(~data["tradable"].to_numpy(bool) & m.has & np.asarray(m.dates.year >= 2016)[:, None])
    bad = sig.iloc[:1].assign(신호일=m.dates[t[0]], 종목코드=m.codes[j[0]])
    for wrong in (pd.concat([sig, bad]), pd.concat([sig, sig.iloc[-1:]]),
                  pd.concat([sig, sig.iloc[-1:].assign(종목코드="999999")])):
        with pytest.raises(SystemExit):
            _sell_ctx(data, wrong)


def test_sellml_cell_portfolio(sell_ctx):
    """칸 포트폴리오: 후보 = 그 칸 신호 (시험 첫 해부터) × 시장 매매 표. 평가금액은 시험 첫 해부터, 해마다 수익을 잇는다."""
    from indexbeater import sell_common as SL
    ctx = sell_ctx
    mk = C.UNIVERSE_MARKETS[0]
    T = ctx.test[mk]
    tab = SL.to_table(SL.fixed_record(ctx.m, T["t"], T["j"], 5), ctx.m)
    s = ctx.sig[(ctx.sig["시장"] == mk) & (ctx.sig["구분"] == "가")]
    cand = SL.cell_candidates(ctx, tab, mk, "가", ctx.test_years[0])
    assert len(cand) == int((ctx.years[s["t"]] >= ctx.test_years[0]).sum()) > 0
    rows, daily = SL.score_cells(ctx, tab, mk, None, {"모델": "시험"})
    assert len(rows) == 2 * len(SL.COSTS) and all(r["산매매수"] > 0 for r in rows)
    r = rows[0]
    eq = daily[0].set_index("date")["equity"]
    assert eq.index.year.min() == ctx.test_years[0]
    tot = np.prod([1 + r[f"포트_수익_{y}"] for y in ctx.test_years])
    assert np.isclose(tot, eq.iloc[-1] / C.INIT_CAPITAL)
