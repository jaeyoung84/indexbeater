"""엔진 매매 규칙 테스트 (계획서 STEP 3, 기준 3·4.2). 손으로 만든 작은 주가 표로 답을 미리 정해 두고 비교한다.

가격은 모두 '눈으로 계산할 수 있는' 숫자다. 날짜 번호는 0부터 센다 (0일, 1일, ...).
"""
import numpy as np
import pandas as pd
import pytest

from indexbeater import config as C
from indexbeater.engine import delisting_worst_case, run_portfolio, simulate_trades
from indexbeater.exits import ExitRule


# ---------------------------------------------------------------- 가짜 표 만들기
def flat(p, v=1000):
    """시가=고가=저가=종가=p 인 하루"""
    return (p, p, p, p, v)


def bar(o, h, l, c, v=1000):
    return (o, h, l, c, v)


def wide(bars: dict, start="2016-01-04", v20=None):
    """종목별 하루 목록 → 엔진이 받는 wide 묶음. None 은 '자료 없음'(상장 전·상장폐지 뒤)."""
    n = max(len(v) for v in bars.values())
    dates = pd.bdate_range(start, periods=n)
    codes = sorted(bars)
    keys = ["adj_open", "adj_high", "adj_low", "adj_close", "volume"]
    arr = {k: np.full((n, len(codes)), np.nan) for k in keys}
    for j, c in enumerate(codes):
        for i, b in enumerate(bars[c]):
            if b is not None:
                for k, x in zip(keys, b):
                    arr[k][i, j] = x
    d = {k: pd.DataFrame(a, index=dates, columns=codes) for k, a in arr.items()}
    has = d["adj_close"].notna()
    d["open"], d["close"], d["volume_s"] = d["adj_open"], d["adj_close"], d["volume"]
    d["value"] = d["adj_close"] * d["volume"]
    d["market"] = pd.DataFrame(np.where(has, "KOSPI", None), index=dates, columns=codes)
    v20 = v20 or {}
    d["value20"] = pd.DataFrame({c: np.where(has[c], v20.get(c, 5e9), np.nan) for c in codes}, index=dates)
    d["tradable"] = has & d["volume"].gt(0)
    return d


def sig(data, *points):
    """(날짜 번호, 종목) 칸에 매수 신호 True"""
    e = pd.DataFrame(False, index=data["adj_close"].index, columns=data["adj_close"].columns)
    for i, c in points:
        e.iloc[i, e.columns.get_loc(c)] = True
    return e


def rising(n, base=100.0):
    """매일 1씩 오르는 주가: 시가 = 종가 − 0.5"""
    return [bar(base + i - 0.5, base + i + 0.5, base + i - 1, base + i) for i in range(n)]


# ---------------------------------------------------------------- 매수·S3
def test_buy_next_open_sell_after_h():
    d = wide({"000010": rising(30)})
    tr = simulate_trades(sig(d, (2, "000010")), d, ExitRule(hold=5))
    r = tr.iloc[0]
    assert r.status == "체결" and r.reason == "S3"
    assert (r.b_idx, r.s_idx) == (3, 8)                     # 신호 2일 → 3일 시가 매수 → 3+5=8일 시가 매도
    assert r.buy_px == 102.5 and r.sell_px == 107.5
    assert r.ret_gross == pytest.approx(107.5 / 102.5 - 1)
    assert r.ret_net == pytest.approx(r.ret_gross - 0.005)
    assert r.ret_net_025 == pytest.approx(r.ret_gross - 0.0025)
    assert r.delay == 0


@pytest.mark.parametrize("start, open_px, status", [
    ("2015-01-05", 114.0, "취소_갭"),      # 2015.6.15 전: +14% 이상이면 못 산다
    ("2015-01-05", 113.9, "체결"),
    ("2016-01-04", 129.0, "취소_갭"),      # 그 뒤: +29% 이상이면 못 산다
    ("2016-01-04", 128.9, "체결"),
    ("2016-01-04", 114.0, "체결"),
])
def test_gap_cancel(start, open_px, status):
    days = [flat(100)] * 20
    days[3] = bar(open_px, open_px, open_px, open_px)
    d = wide({"000010": days}, start=start)
    tr = simulate_trades(sig(d, (2, "000010")), d, ExitRule(hold=5))
    assert tr.iloc[0].status == status


def test_halt_on_buy_day_is_no_buy():
    days = [flat(100)] * 20
    days[3] = flat(100, v=0)                                 # 매수일 거래정지
    d = wide({"000010": days})
    assert simulate_trades(sig(d, (2, "000010")), d, ExitRule(hold=5)).iloc[0].status == "매수불가"


def test_zero_open_day_is_no_buy_v2():
    # 엔진 v2: 시가·고가·저가 0, 거래량 > 0 인 날(자료 결함)은 사지 않는다. v1 은 0원에 사서 수익이 무한대였다
    days = [flat(100)] * 20
    days[3] = bar(0, 0, 0, 100)
    d = wide({"000010": days})
    r = simulate_trades(sig(d, (2, "000010")), d, ExitRule(hold=5)).iloc[0]
    assert r.status == "매수불가" and np.isnan(r.buy_px)


def test_zero_open_day_is_not_sold_on_v2():
    # 매도 예정일이 시가 0 인 날 → 그날은 못 팔고 다음 날 시가에 판다
    days = [flat(100)] * 20
    days[8] = bar(0, 0, 0, 100)
    d = wide({"000010": days})
    r = simulate_trades(sig(d, (2, "000010")), d, ExitRule(hold=5)).iloc[0]
    assert (r.status, r.s_idx, r.delay, r.sell_px) == ("체결", 9, 1, 100)
    assert np.isfinite(r.ret_gross)


def test_zero_low_on_zero_open_day_does_not_trigger_stop_v2():
    # 시가 0 인 날의 저가 0 은 진짜 가격이 아니다 → 손절로 보지 않고 S3 로 판다
    days = [flat(100)] * 20
    days[5] = bar(0, 0, 0, 100)
    d = wide({"000010": days})
    r = simulate_trades(sig(d, (2, "000010")), d, ExitRule(hold=5, stop=0.05, take=0.10)).iloc[0]
    assert (r.reason, r.s_idx, r.sell_px) == ("S3", 8, 100)


def test_portfolio_skips_zero_open_buy_v2():
    days = [flat(100)] * 20
    days[3] = bar(0, 0, 0, 100)
    d = wide({"000010": days})
    daily, log = run_portfolio(sig(d, (2, "000010")).astype(int), d, ExitRule(hold=5), "KOSPI")
    assert np.isfinite(daily["equity"]).all()
    assert (log["shares"] == 0).all()                        # 후보였지만 매수불가 → 0주


def test_sell_delayed_by_halt_and_limit_down():
    days = [flat(100)] * 20
    days[8] = flat(100, v=0)                                 # 원래 매도일: 거래정지
    days[9] = bar(71, 71, 71, 71)                            # 다음 날: 시가 −29% (하한가 근처) → 못 판다
    d = wide({"000010": days})
    r = simulate_trades(sig(d, (2, "000010")), d, ExitRule(hold=5)).iloc[0]
    assert r.s_idx == 10 and r.delay == 2 and r.sell_px == 100   # 10일 시가 100에 판다


def test_delisting_settles_at_last_close():
    d = wide({"000010": [flat(100)] * 6 + [flat(90)], "000020": [flat(50)] * 20})
    r = simulate_trades(sig(d, (2, "000010")), d, ExitRule(hold=10)).iloc[0]
    assert r.reason == "상장폐지" and r.s_idx == 6 and r.sell_px == 90


def test_delisting_worst_case_only_changes_delisted_trades():
    # 상장폐지 매매만 −100%가 되고, 보통 매매(S3)는 그대로여야 한다
    d = wide({"000010": [flat(100)] * 6 + [flat(90)], "000020": [flat(50)] * 20})
    tr = simulate_trades(sig(d, (2, "000010"), (2, "000020")), d, ExitRule(hold=10))
    worst = delisting_worst_case(tr)
    gone = worst["reason"].eq("상장폐지")
    assert gone.sum() == 1
    assert worst.loc[gone, "ret_gross"].iloc[0] == -1.0
    assert worst.loc[gone, "ret_net"].iloc[0] == pytest.approx(-1.0 - C.COST_SELECT)
    pd.testing.assert_frame_equal(worst[~gone], tr[~gone])
    assert tr.loc[gone, "sell_px"].iloc[0] == 90           # 원래 표는 바뀌지 않는다 (사본만 바꿈)


def test_exit_past_end_is_incomplete():
    d = wide({"000010": [flat(100)] * 10})
    tr = simulate_trades(sig(d, (5, "000010"), (9, "000010")), d, ExitRule(hold=5), dedupe=False)
    assert list(tr.status) == ["미완료", "미완료"]          # 5일 신호는 11일 매도(끝 넘음), 9일 신호는 매수일이 없음


def test_dedupe_and_resignal_on_sell_day():
    d = wide({"000010": rising(30)})
    e = sig(d, (2, "000010"), (4, "000010"), (8, "000010"))
    tr = simulate_trades(e, d, ExitRule(hold=5))
    # 2일 신호 → 3일 매수, 8일 매도. 4일은 보유 중이라 무시. 8일(매도일) 신호는 다시 산다
    assert list(tr.status) == ["체결", "중복_무시", "체결"]
    assert tr.iloc[2].b_idx == 9
    tr2 = simulate_trades(e, d, ExitRule(hold=5), dedupe=False)
    assert list(tr2.status) == ["체결", "체결", "체결"]


def test_signal_ignored_when_not_tradable_or_before_start():
    d = wide({"000010": [flat(100)] * 20})
    d["tradable"].iloc[2, 0] = False                         # 2일은 유동성 필터 탈락
    assert simulate_trades(sig(d, (2, "000010")), d, ExitRule(hold=5)).empty
    d2 = wide({"000010": [flat(100)] * 20}, start="2012-06-20")   # 2012-07-01(SIGNAL_START) 전 신호
    assert simulate_trades(sig(d2, (2, "000010")), d2, ExitRule(hold=5)).empty


# ---------------------------------------------------------------- S1 손절·익절
RULE_S1 = ExitRule(hold=5, stop=0.05, take=0.10)


def _s1(day4, day3=None, extra=None):
    """2일 신호, 3일 시가 100에 매수. 4일 모양을 바꿔 가며 본다."""
    days = [flat(100)] * 20
    days[3] = day3 or flat(100)
    days[4] = day4
    for i, b in (extra or {}).items():
        days[i] = b
    d = wide({"000010": days})
    return simulate_trades(sig(d, (2, "000010")), d, RULE_S1).iloc[0]


def test_s1_stop_at_level():
    r = _s1(bar(99, 101, 94, 96))
    assert (r.reason, r.s_idx, r.sell_px, r.sell_at_open) == ("S1손절", 4, 95.0, False)


def test_s1_take_at_level():
    r = _s1(bar(101, 111, 100, 105))
    assert (r.reason, r.sell_px) == ("S1익절", pytest.approx(110.0))


def test_s1_gap_down_sells_at_open():
    r = _s1(bar(90, 92, 88, 91))
    assert (r.reason, r.sell_px, r.sell_at_open) == ("S1손절", 90, True)


def test_s1_both_hit_same_day_is_stop():
    r = _s1(bar(100, 112, 94, 100))
    assert r.reason == "S1손절" and r.sell_px == 95.0


def test_s1_not_checked_on_buy_day():
    r = _s1(flat(100), day3=bar(100, 100, 80, 100))          # 매수일 장중 −20% 는 보지 않는다
    assert r.reason == "S3" and r.s_idx == 8


def test_open_sale_before_stop_same_day():
    # 8일: S3 시가 매도일. 그날 장중 −10% 여도 시가에 먼저 판다 (결정 4A)
    r = _s1(flat(100), extra={8: bar(100, 100, 90, 92)})
    assert (r.reason, r.s_idx, r.sell_px) == ("S3", 8, 100)


def test_stop_on_limit_down_day_waits_for_next_open():
    # 4일 시가 −30% (팔 수 없는 날)에 손절선 닿음 → 5일 시가에 판다
    r = _s1(bar(70, 70, 70, 70), extra={5: bar(72, 75, 71, 74)})
    assert (r.reason, r.s_idx, r.sell_px) == ("S1손절", 5, 72)


def test_exit_rule_hold_limit():
    with pytest.raises(ValueError):
        ExitRule(hold=C.MAX_HOLD + 1)


# ---------------------------------------------------------------- 포트폴리오
def test_portfolio_cash_not_reused_same_day():
    # 2종목 한도, 종목당 50%. A는 1일 매수 → 6일 시가 매도. 5일에 B·C 신호.
    # 6일 아침 빈 자리는 2개지만 현금은 50뿐(A 판 돈은 장 마감 뒤 들어옴) → B만 산다
    n = 15
    d = wide({"000010": [flat(100)] * n, "000020": [flat(100)] * n, "000030": [flat(100)] * n},
             v20={"000020": 9e9, "000030": 8e9})
    e = sig(d, (0, "000010"), (5, "000020"), (5, "000030")).astype(int)
    daily, trades = run_portfolio(e, d, ExitRule(hold=5), "KOSPI", capital=100.0, max_positions=2, weight=0.5)
    bought = trades[trades["notional"] > 0]
    assert sorted(bought["code"]) == ["000010", "000020"]
    assert bought.set_index("code").loc["000020", "notional"] == pytest.approx(50.0)
    assert daily["n_pos"].iloc[6] == 1


def test_portfolio_no_double_buy_and_order():
    n = 20
    d = wide({c: [flat(100)] * n for c in ("000010", "000020", "000030", "000040")},
             v20={"000020": 9e9, "000030": 8e9, "000040": 8e9})
    cnt = sig(d).astype(int)
    cnt.iloc[0, 0] = 1                                       # A: 0일 신호 → 1일 매수
    cnt.iloc[3, 0] = 1                                       # A: 보유 중 재신호 → 사지 않음
    cnt.iloc[3, 1] = 1                                       # B: 신호 1개, 거래대금 9e9
    cnt.iloc[3, 2] = 2                                       # C: 신호 2개 → B보다 먼저
    cnt.iloc[3, 3] = 2                                       # D: 신호 2개, 거래대금 같음 → 코드 순서로 C 다음
    daily, trades = run_portfolio(cnt, d, ExitRule(hold=5), "KOSPI", capital=100.0, max_positions=3, weight=0.2)
    got = trades[trades["notional"] > 0].sort_values(["b_idx", "code"])
    assert list(zip(got["b_idx"], got["code"])) == [(1, "000010"), (4, "000030"), (4, "000040")]


def test_portfolio_gap_cancel_slot_not_refilled_and_cost():
    n = 20
    days_b = [flat(100)] * n
    days_b[2] = flat(140)                                    # B: 1일 신호, 2일 시가 +40% → 취소
    d = wide({"000010": [flat(100)] * n, "000020": days_b}, v20={"000020": 9e9})
    e = sig(d, (1, "000010"), (1, "000020")).astype(int)
    daily, trades = run_portfolio(e, d, ExitRule(hold=5), "KOSPI", capital=100.0, max_positions=1, weight=0.5)
    # 자리는 1개, 1순위 B가 갭으로 취소 → A로 채우지 않는다
    assert (trades["notional"] > 0).sum() == 0
    # 비용: A만 1자리로 사서 그대로 팔면 0.5% × 50 = 0.25 손해
    daily2, _ = run_portfolio(sig(d, (1, "000010")).astype(int), d, ExitRule(hold=5), "KOSPI",
                              capital=100.0, max_positions=1, weight=0.5)
    assert daily2["equity"].iloc[-1] == pytest.approx(100.0 - 0.25)
