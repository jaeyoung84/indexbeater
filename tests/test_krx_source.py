"""KRX 수정주가 로직 단위 테스트 (합성 데이터 — 키 없이 실행 가능)."""
import numpy as np
import pandas as pd

from indexbeater.krx_source.adjust import apply_adjustments
from indexbeater.krx_source.collect import rows_to_frame
from indexbeater.krx_source.dividends import build_events, estimate_ex_date, parse_dps
from indexbeater.krx_source.splits import detect_splits

PX = ["close", "open", "high", "low", "base_price"]
NO_SPLITS = pd.DataFrame(columns=["code", "ex_date", "share_ratio"])


def _series(n=40, price=1000.0, shares=1_000_000.0, code="000001"):
    d = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"date": d, "market": "KOSPI", "code": code, "name": "T", "close": price,
                         "open": price, "high": price, "low": price, "volume": 100.0,
                         "listed_shares": shares, "base_price": price})


def test_split_same_day_is_continuous():
    g = _series()
    g.loc[20:, PX] = 20.0                       # 1/50 (삼성전자형)
    g.loc[20:, "listed_shares"] = 50_000_000.0
    g.loc[20, "base_price"] = 20.0
    ev, rv = detect_splits(g)
    assert len(ev) == 1 and rv.empty
    assert ev.iloc[0]["ex_date"] == g["date"].iloc[20] and ev.iloc[0]["share_ratio"] == 50
    adj = apply_adjustments(g, ev, pd.DataFrame())
    assert np.allclose(adj["close_s"], 20.0)
    assert np.allclose(adj["volume_s"].iloc[:20], 5000.0) and np.allclose(adj["volume_s"].iloc[20:], 100.0)


def test_bonus_issue_price_moves_before_shares_listed():
    # 권리락(15일): 기준가 조정(전일종가/기준가=2), 신주상장(18일)에 주식수 2배
    g = _series()
    g.loc[15:, PX] = 500.0
    g.loc[18:, "listed_shares"] = 2_000_000.0
    ev, _ = detect_splits(g)
    assert len(ev) == 1 and ev.iloc[0]["ex_date"] == g["date"].iloc[15]


def test_rights_offering_goes_to_review():
    g = _series()
    g.loc[20:, "listed_shares"] = 2_000_000.0   # 주식수만 2배, 가격 불변
    ev, rv = detect_splits(g)
    assert ev.empty and len(rv) == 1


def test_two_splits_accumulate():
    g = _series(60)
    g.loc[20:, PX] = 500.0
    g.loc[20:, "listed_shares"] = 2e6
    g.loc[40:, PX] = 100.0
    g.loc[40:, "listed_shares"] = 1e7
    ev, _ = detect_splits(g)
    adj = apply_adjustments(g, ev, pd.DataFrame())
    assert len(ev) == 2 and np.allclose(adj["close_s"], 100.0)


def test_dividend_factor_and_order():
    g = _series(60)
    div = pd.DataFrame({"code": ["000001"], "ex_date": [g["date"].iloc[30]], "factor": [0.98],
                        "confidence": ["estimated"]})
    adj = apply_adjustments(g, NO_SPLITS, div)
    assert np.allclose(adj["adj_close"].iloc[:30], 980.0) and np.allclose(adj["adj_close"].iloc[30:], 1000.0)
    assert np.allclose(adj["close_s"], 1000.0)


def test_estimate_ex_date_second_to_last_trading_day():
    days = pd.DatetimeIndex(["2022-12-27", "2022-12-28", "2022-12-29", "2023-01-02"])
    assert estimate_ex_date(days, 2022) == pd.Timestamp("2022-12-28")
    assert estimate_ex_date(days, 2025) is None


def test_build_dividend_events():
    d = pd.bdate_range("2022-12-20", "2023-01-05")
    raw = pd.DataFrame({"date": d, "code": "000001", "close": 10000.0})
    div = pd.DataFrame({"code": ["000001", "000001"], "fiscal_year": [2022, 2021], "dps": [500.0, np.nan],
                        "acc_mt": ["12", "12"]})
    ev, rv = build_events(div, raw)
    assert len(ev) == 1 and abs(ev.iloc[0]["factor"] - 0.95) < 1e-9 and ev.iloc[0]["confidence"] == "estimated"
    ev2, rv2 = build_events(div.assign(acc_mt="3"), raw)
    assert ev2.empty and len(rv2) == 1


def test_parse_dps_and_rows():
    j = {"list": [{"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444"},
                  {"se": "주당 현금배당금(원)", "stock_knd": "우선주", "thstrm": "1,445"}]}
    assert parse_dps(j) == 1444
    f = rows_to_frame([{"ISU_CD": "005930", "ISU_NM": "삼성전자", "TDD_CLSPRC": "51,900",
                        "CMPPREVDD_PRC": "-1,000", "LIST_SHRS": "6,419,420,000"}],
                      "KOSPI", pd.Timestamp("2018-05-04"))
    assert f.iloc[0]["close"] == 51900 and f.iloc[0]["base_price"] == 52900
    assert f.iloc[0]["listed_shares"] == 6_419_420_000


def test_etf_is_excluded_from_split_detection():
    # ETF는 주식 수(상장좌수)가 설정·환매로 바뀌는 게 정상이라서, 분할 후보에서 빼야 한다.
    g = _series().assign(market="ETF")
    g.loc[20:, "listed_shares"] = 2_000_000.0
    ev, rv = detect_splits(g)
    assert ev.empty and rv.empty
    ev2, rv2 = detect_splits(g, markets=("ETF",))   # 일부러 ETF를 고르면 다시 후보로 잡힌다
    assert len(rv2) == 1


def test_parse_dps_single_class_company_uses_dash():
    # 주식 종류가 하나뿐인 회사는 DART가 종류를 '-' 로 적는다. 첫 줄이 진짜 배당금, 둘째 줄은 빈 값이다.
    j = {"list": [{"se": "주당 현금배당금(원)", "stock_knd": "-", "thstrm": "150"},
                  {"se": "주당 현금배당금(원)", "stock_knd": "-", "thstrm": "-"}]}
    assert parse_dps(j) == 150
    # 우선주 줄만 있으면 보통주 배당은 없는 것으로 본다
    assert np.isnan(parse_dps({"list": [{"se": "주당 현금배당금(원)", "stock_knd": "우선주", "thstrm": "100"}]}))


def test_rights_factor_applies_only_confirmed():
    g = _series(60)
    rt = pd.DataFrame({"code": ["000001", "000001"], "ex_date": [g["date"].iloc[30], g["date"].iloc[45]],
                       "factor": [0.9, 0.5], "confidence": ["confirmed", "review"]})
    adj = apply_adjustments(g, NO_SPLITS, pd.DataFrame(), rights=rt)
    assert np.allclose(adj["adj_close"].iloc[:30], 900.0) and np.allclose(adj["adj_close"].iloc[30:], 1000.0)
    assert np.allclose(adj["close_s"], 1000.0)          # 네이버 비교용 분할 보정값은 그대로


def test_find_ex_date_confirms_with_share_jump():
    from indexbeater.krx_source.rights import find_ex_date
    days = pd.bdate_range("2020-01-01", periods=80)
    close = np.full(80, 1000.0)
    N, n, Pi = 1_000_000, 250_000, 500.0
    f = (1000 * N + Pi * n) / (N + n) / 1000            # 이론 권리락 계수
    close[40:] = 1000 * f
    shares = np.full(80, float(N))
    shares[60:] = N + n   # 신주는 권리락 20일 뒤 상장
    x = pd.DataFrame({"date": days, "code": "000001", "close": close, "listed_shares": shares})
    cand = pd.Series(dict(bfic_tisstk_ostk=N, nstk_ostk_cnt=n, price=Pi, rdate=days[10], stock_code="000001", rcept_no="x", ic_mthn="주주배정증자"))
    r = find_ex_date(cand, x, set())
    assert r["ex_date"] == days[40] and r["confidence"] == "confirmed" and abs(r["factor"] - f) < 1e-9


def test_base_events_and_adjust_skip_duplicates():
    from indexbeater.krx_source.base import build_base_events
    d = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    raw = pd.DataFrame({"date": d, "code": "000001", "open": 100.0, "high": 100.0, "low": 100.0,
                        "close": [100.0, 100.0, 80.0, 80.0], "volume": 1,
                        "base_price": [100.0, 100.0, 90.0, 80.0]})        # 1/6 에 기준가 90 (전일종가 100 → 계수 0.9)
    ev = build_base_events(raw, pd.DataFrame({"code": [], "ex_date": []}))
    assert len(ev) == 1 and abs(ev.factor.iloc[0] - 0.9) < 1e-9
    rt = pd.DataFrame({"code": ["000001"], "ex_date": [pd.Timestamp("2020-01-06")], "factor": [0.9], "confidence": ["confirmed"]})
    out = apply_adjustments(raw, NO_SPLITS, pd.DataFrame(), rights=rt, base=ev)
    assert abs(out.loc[0, "adj_close"] - 90.0) < 1e-9      # 0.9 가 한 번만 곱해진다 (유상증자와 중복 안 됨)
    assert out.loc[2, "adj_close"] == 80.0


def test_dedupe_splits_keeps_event_matching_base_price():
    from indexbeater.krx_source.base import dedupe_splits
    d = pd.to_datetime(["2020-01-02", "2020-01-03"])
    raw = pd.DataFrame({"date": d, "code": "000001", "close": [900.0, 600.0], "base_price": [900.0, 600.0]})   # 기준가/전일종가 = 0.667
    sp = pd.DataFrame({"code": ["000001"] * 2, "ex_date": [d[1]] * 2, "share_ratio": [1.37, 1.5], "price_dev": [0.09, 0.0]})
    out = dedupe_splits(sp, raw)
    assert len(out) == 1 and out.share_ratio.iloc[0] == 1.5


def test_drop_unsupported_big_dividends():
    from indexbeater.krx_source.base import drop_unsupported_big_dividends
    d = pd.to_datetime(["2020-01-02", "2020-01-03"])
    raw = pd.DataFrame({"date": d, "code": "000001", "close": [1000.0, 995.0]})   # 가격이 거의 안 떨어짐
    dv = pd.DataFrame({"code": ["000001"], "ex_date": [d[1]], "factor": [0.7], "confidence": ["estimated"]})
    assert len(drop_unsupported_big_dividends(dv, raw)) == 0
    raw.loc[1, "close"] = 700.0                                                     # 실제로 30퍼센트 떨어짐
    assert len(drop_unsupported_big_dividends(dv, raw)) == 1
