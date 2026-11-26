"""D4 벤치마크 — 분배금(ETF 배당)을 다시 넣은 ETF 수익 지수 (기준 3장 4.5·5.8 '비교 지수는 배당 반영 ETF').

왜 따로 만드나
- krx_adjusted.parquet 의 ETF 가격은 분배금이 빠진 그대로다 (adj_close = close). ETF 는 배당 공시(DART) 대상이
  아니라서 배당 보정 단계에서 빠졌다 (krx_source/dividends.py).
- 전략은 배당 반영 가격(adj_*)으로 계산하므로, 비교 지수도 분배금을 다시 넣어야 공정하다.
  빼고 비교하면 전략이 해마다 분배금만큼(코스피200 약 1~2%) 유리해 보인다.

어떻게 계산하나 (KRX 원자료만 쓴다, 새로 받는 자료 없음)
- KRX 는 분배락 날에 기준가를 '전일 종가 − 분배금'으로 낮춘다. 원자료 표(krx_raw_daily.parquet)의
  base_price(= 종가 − 전일대비)가 그 기준가다.
- 분배금 = 전일 종가 − 그날 기준가 (평소엔 0)
- 그날 수익률(분배금 재투자) = (종가 + 분배금) / 전일 종가 − 1 = (종가 − 기준가) / 전일 종가
- 수익 지수 = 첫날 100 에서 그날 수익률을 차례로 곱한다.
- 분배금이 전일 종가의 MAX_DIST_PCT 를 넘으면 분할 같은 다른 사건일 수 있어서 멈춘다 (조용히 틀리지 않게).

코스닥 2015-10 이전 (11/24 팀장 결정 1번: 지수를 받아서 채운다)
- 229200 은 2015-10-01 상장이라 그 전 날짜가 없다. 기준 5.8 '코스닥150 지수(없으면 코스닥 종합)를 대신 쓰고 표시'.
- 코스닥150(KQ150)은 FinanceDataReader 로 받아지지 않았다(404). KRX 지수 화면은 열쇠 없이 막힌다.
  → 5.8 순서대로 코스닥 종합(KQ11)을 쓴다. 가격 지수라 배당이 빠져 있다 → 출처 열에 그대로 적는다.
- 받는 기간: 자료 첫날 ~ 2015-12-31 (상장 뒤 3달은 229200 과 겹쳐 보려고). 2020년 이후는 받지 않는다.
- 잇는 방법: 229200 상장 첫날(2015-10-01)까지는 KQ11 하루 수익률, 다음 날부터 229200 분배 포함 수익률.
  상장 첫날은 229200 의 전일 종가가 없어서 그날 수익률을 KQ11 로 채운다.
- ETF 표(benchmark_tr.csv)는 그대로 두고, 시장별 비교 지수를 따로 만든다(benchmark_market.csv).

시험 구간 잠금: data_io 와 같다. 2020-01-01 이후는 allow_test=True 없이 열지 않는다 (읽기 전에 막는다).

    python -m indexbeater benchmark --fetch-index   # KQ11 받기 → data/index_kq11.parquet (처음 한 번)
    python -m indexbeater benchmark                 # → results/benchmark_tr.csv, results/benchmark_market.csv
    python -m indexbeater benchmark --scorecard     # 이미 만든 results/scorecard.csv 에 지수 초과수익 열만 붙인다

점수표 초과수익 (11/25 팀장 결정, 기준 5.6·5.8·5.17): 포트 연수익 − 같은 시장 비교 지수 연수익 (그냥 빼기).
optimize --merge 도 합칠 때 같은 함수(add_excess)로 붙인다. 조합 점수표도 같은 함수를 쓴다.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

RAW_DAILY = C.DATA / "krx_raw_daily.parquet"   # KRX 원자료를 합친 표 (krx_source/paths.RAW_DAILY 와 같은 파일)
OUT = C.RESULTS / "benchmark_tr.csv"
MAX_DIST_PCT = 0.05     # 분배금이 전일 종가의 5% 를 넘으면 멈춘다 (2011~2019 069500 최대 약 1.7%)
MARKET_OF = {"069500": "KOSPI", "229200": "KOSDAQ"}   # 어느 시장 전략과 비교하는가 (C.BENCH_ETFS 순서와 같다)

MARKET_OUT = C.RESULTS / "benchmark_market.csv"
SUB_SYMBOL = "KQ11"                      # 5.8: 코스닥150 이 없으면 코스닥 종합 (KQ150 은 404, 11/24 확인)
SUB_FOR = "229200"                       # 이 ETF 상장 전을 대신 채운다
SUB_FILE = C.DATA / "index_kq11.parquet"
SUB_END = pd.Timestamp("2015-12-31")     # 상장(2015-10-01) 뒤 3달까지만 받는다 (겹치는 기간 대조용)


def _check_end(end, allow_test: bool) -> pd.Timestamp:
    end = pd.Timestamp(end)
    if end >= C.TEST_START and not allow_test:
        raise ValueError(f"{end.date()} 는 시험 구간이다. 열려면 allow_test=True 가 필요하다 (기준 4장)")
    return end


def load_raw(end=C.DESIGN_END, allow_test: bool = False, path=RAW_DAILY, codes=tuple(C.BENCH_ETFS)) -> pd.DataFrame:
    """원자료에서 벤치마크 ETF 줄만 읽는다. 날짜·종목 조건을 읽을 때 걸어서 2020년 줄은 메모리에 올라오지 않는다."""
    end = _check_end(end, allow_test)
    df = pd.read_parquet(path, columns=["date", "market", "code", "name", "close", "base_price"],
                         filters=[("date", "<=", end), ("code", "in", list(codes))])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def total_return(raw: pd.DataFrame) -> pd.DataFrame:
    """종목마다 분배금과 분배금 재투자 수익 지수를 만든다. raw 열: date, code, name, close, base_price."""
    out = []
    for code, g in raw.sort_values("date").groupby("code", sort=True):
        g = g.dropna(subset=["close"]).copy()
        if g["base_price"].isna().iloc[1:].any():
            raise ValueError(f"{code}: 기준가가 빈 날이 있다 → 분배금을 계산할 수 없다")
        prev = g["close"].shift(1)
        dist = (prev - g["base_price"]).round(6)
        dist.iloc[0] = 0.0                        # 첫날은 전일이 없다
        if (dist < -1e-9).any():
            bad = g.loc[dist < -1e-9, "date"].dt.date.tolist()
            raise ValueError(f"{code}: 기준가가 전일 종가보다 높은 날 {bad[:5]} → 분배금이 아닌 다른 사건")
        if (dist > MAX_DIST_PCT * prev).any():
            bad = g.loc[dist > MAX_DIST_PCT * prev, "date"].dt.date.tolist()
            raise ValueError(f"{code}: 분배금이 전일 종가의 {MAX_DIST_PCT:.0%} 를 넘는 날 {bad[:5]} → 분할 등 다른 사건인지 확인")
        dist = dist.clip(lower=0.0)
        ret = ((g["close"] + dist) / prev - 1).fillna(0.0)
        g["dist"] = dist
        g["ret_tr"] = ret
        g["tr_index"] = 100 * (1 + ret).cumprod()
        g["price_index"] = 100 * g["close"] / g["close"].iloc[0]   # 분배금 뺀 가격만 (참고용)
        g["bench_market"] = MARKET_OF.get(code, "")
        out.append(g)
    cols = ["date", "code", "name", "bench_market", "close", "base_price", "dist", "ret_tr", "tr_index", "price_index"]
    return pd.concat(out, ignore_index=True)[cols]


def yearly(tr: pd.DataFrame) -> pd.DataFrame:
    """연도별 수익률 (분배금 재투자 / 가격만). 그해 1월 첫 거래일 전에 상장한 해만 채운다.

    상장한 해(예: 229200 의 2015년)는 1년 전체가 아니므로 NaN. 그해 수익률은 '전년 마지막 종가 → 그해 마지막 종가'.
    """
    rows = []
    for code, g in tr.groupby("code", sort=True):
        g = g.set_index("date").sort_index()
        last = g.groupby(g.index.year).tail(1)
        first_year = g.index[0].year
        for i in range(len(last)):
            y = last.index[i].year
            if i == 0:
                tr_y = pr_y = np.nan                 # 첫해: 전년 말 값이 없다
            else:
                tr_y = last["tr_index"].iloc[i] / last["tr_index"].iloc[i - 1] - 1
                pr_y = last["price_index"].iloc[i] / last["price_index"].iloc[i - 1] - 1
            rows.append({"code": code, "bench_market": g["bench_market"].iloc[0], "year": y,
                         "수익률_분배포함": tr_y, "수익률_가격만": pr_y,
                         "분배금_합": g.loc[g.index.year == y, "dist"].sum(),
                         "첫해": y == first_year})
    return pd.DataFrame(rows)


def build(end=C.DESIGN_END, allow_test: bool = False, path=RAW_DAILY) -> pd.DataFrame:
    return total_return(load_raw(end=end, allow_test=allow_test, path=path))


def fetch_sub_index(out=SUB_FILE, end=SUB_END) -> pd.DataFrame:
    """KQ11(코스닥 종합)을 받아 저장한다. 인터넷이 필요하다. 2020년 이후는 받지 않는다(받기 전에 막는다)."""
    end = _check_end(end, False)
    import FinanceDataReader as fdr  # 받을 때만 쓴다 (테스트·계산은 저장한 파일만 읽는다)
    d = fdr.DataReader(SUB_SYMBOL, C.DATA_START.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    d = d.reset_index().rename(columns={"Date": "date", "Close": "close"})[["date", "close"]]
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= C.DATA_START) & (d["date"] <= end)].dropna(subset=["close"])
    d.insert(1, "symbol", SUB_SYMBOL)
    d.to_parquet(out, index=False)
    return d.reset_index(drop=True)


def load_sub_index(path=SUB_FILE, end=C.DESIGN_END, allow_test: bool = False) -> pd.DataFrame:
    end = _check_end(end, allow_test)
    d = pd.read_parquet(path, filters=[("date", "<=", end)])
    return d.sort_values("date").reset_index(drop=True)


def market_series(tr: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """시장별 비교 지수 (하루 수익률 → 100 에서 곱한 지수).

    KOSPI  = 069500 분배 포함 (자료 첫날부터 있다)
    KOSDAQ = 229200 상장 첫날까지 KQ11 가격 수익률 + 다음 날부터 229200 분배 포함
    대신 쓴 날의 거래일이 069500 거래일과 하나라도 다르면 멈춘다 (빠진 날을 조용히 건너뛰지 않게).
    """
    out = []
    for code, mk in MARKET_OF.items():
        g = tr.loc[tr["code"] == code, ["date", "ret_tr"]].rename(columns={"ret_tr": "ret"})
        g = g.sort_values("date").assign(source=f"{code} 분배 포함")
        if code == SUB_FOR:
            first = g["date"].min()
            s = sub.sort_values("date")
            s = s.assign(ret=s["close"].pct_change().fillna(0.0))
            pre = s.loc[s["date"] <= first, ["date", "ret"]].assign(source=f"{SUB_SYMBOL} 대신 (가격만)")
            cal = tr.loc[(tr["code"] == "069500") & (tr["date"] <= first), "date"]
            if not pre["date"].reset_index(drop=True).equals(cal.sort_values().reset_index(drop=True)):
                miss = sorted(set(cal) ^ set(pre["date"]))
                raise ValueError(f"{SUB_SYMBOL} 거래일이 069500 과 다르다 {[d.date() for d in miss[:5]]} → 자료 확인")
            g = pd.concat([pre, g[g["date"] > first]], ignore_index=True)
        g["market"] = mk
        g["index"] = 100 * (1 + g["ret"]).cumprod()
        out.append(g)
    return pd.concat(out, ignore_index=True)[["date", "market", "ret", "index", "source"]]


def market_yearly(ms: pd.DataFrame) -> pd.DataFrame:
    """시장별 연도 수익률. 첫해(2011, 1년이 안 됨)는 NaN. 출처가 섞인 해는 출처를 '+' 로 잇는다."""
    rows = []
    for mk, g in ms.groupby("market", sort=False):
        g = g.set_index("date").sort_index()
        last = g.groupby(g.index.year)["index"].last()
        for i, (y, v) in enumerate(last.items()):
            src = " + ".join(dict.fromkeys(g.loc[g.index.year == y, "source"]))
            rows.append({"market": mk, "year": y, "수익률": np.nan if i == 0 else v / last.iloc[i - 1] - 1, "출처": src})
    return pd.DataFrame(rows)


def build_market(end=C.DESIGN_END, allow_test: bool = False, path=RAW_DAILY, sub_path=SUB_FILE) -> pd.DataFrame:
    tr = build(end=end, allow_test=allow_test, path=path)
    return market_series(tr, load_sub_index(sub_path, end=end, allow_test=allow_test))


def load_market(path=MARKET_OUT) -> pd.DataFrame:
    """저장한 시장별 비교 지수(benchmark_market.csv)를 읽는다. 2020년 이후 날짜가 들어 있으면 멈춘다."""
    ms = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"], float_precision="round_trip")
    if (ms["date"] >= C.TEST_START).any():
        raise ValueError(f"{path} 에 시험 구간 날짜가 있다 → 설계 구간 파일만 쓴다 (기준 4장)")
    return ms


# ---------------------------------------------------------------- 점수표에 지수 초과수익 붙이기
# 기준 5.6·5.8·5.17 '포트폴리오 성적(연도별 지수 초과수익, 중앙값, 음수 연도 수)', D4 의 남은 일.
# 11/25 팀장 결정: 초과수익 = 포트 연수익 − 같은 시장 비교 지수 연수익 (그냥 빼기, 기준 4장 지표 정의 IR '(전략 − 지수)' 와 같은 방식)
#                  열은 results/scorecard.csv 에 더한다 (원래 열은 그대로).
EXCESS_YEARS = C.WF_YEARS
EXCESS_COLS = ([f"포트_지수수익_{y}" for y in EXCESS_YEARS] + [f"포트_초과수익_{y}" for y in EXCESS_YEARS]
               + ["포트_초과수익_중앙값", "포트_초과음수해수", "포트_지수대신쓴해"])
SCORECARD = C.RESULTS / "scorecard.csv"


def add_excess(score: pd.DataFrame, my: pd.DataFrame) -> pd.DataFrame:
    """점수표(score)에 지수 초과수익 열을 맨 뒤에 붙인다. my = market_yearly() 결과.

    포트폴리오가 있는 줄(비용 0.5%, D4)만 채우고 나머지 줄은 빈칸. 이미 붙어 있으면 새로 계산해 바꾼다.
    음수 연도 = 초과수익 < 0 인 해. 대신쓴해 = 비교 지수에 대신 쓴 지수(KQ11)가 섞인 해 (5.8 '표시한다'), 없으면 '없음'
    (빈칸으로 두면 파일을 다시 읽을 때 '2015' 가 숫자로 읽힌다).
    """
    out = score.drop(columns=[c for c in EXCESS_COLS if c in score.columns]).copy()
    port = out[[f"포트_수익_{y}" for y in EXCESS_YEARS]]
    has = port.notna().all(axis=1)
    if (port.notna().any(axis=1) & ~has).any():
        raise ValueError("포트폴리오 연수익이 일부 해만 빈 줄이 있다 → 포트폴리오 결과 확인")
    ret = my.pivot(index="market", columns="year", values="수익률")
    src = my.pivot(index="market", columns="year", values="출처")
    markets = out.loc[has, "시장"].unique()
    for mk in markets:
        if mk not in ret.index:
            raise ValueError(f"비교 지수에 {mk} 시장이 없다")
        miss = [y for y in EXCESS_YEARS if y not in ret.columns or pd.isna(ret.loc[mk, y])]
        if miss:
            raise ValueError(f"{mk} 비교 지수에 {miss} 년 수익이 없다 → benchmark_market.csv 확인")
    for y in EXCESS_YEARS:
        out[f"포트_지수수익_{y}"] = out["시장"].map(ret[y]).where(has)
    for y in EXCESS_YEARS:
        out[f"포트_초과수익_{y}"] = (out[f"포트_수익_{y}"] - out[f"포트_지수수익_{y}"]).where(has)
    ex = out[[f"포트_초과수익_{y}" for y in EXCESS_YEARS]]
    out["포트_초과수익_중앙값"] = ex.median(axis=1).where(has)
    out["포트_초과음수해수"] = (ex < 0).sum(axis=1).astype("Int64").where(has)
    sub = {mk: ",".join(str(y) for y in EXCESS_YEARS if "대신" in str(src.loc[mk, y])) or "없음" for mk in markets}
    out["포트_지수대신쓴해"] = out["시장"].map(sub).where(has)
    return out


def attach_scorecard(path=SCORECARD, market_path=MARKET_OUT, overwrite: bool = False) -> str:
    """이미 만든 점수표 파일에 초과수익 열을 붙여 다시 쓴다. 원래 열은 숫자 하나도 바뀌지 않았는지 확인한다."""
    score = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    new = add_excess(score, market_yearly(load_market(market_path)))
    keep = [c for c in score.columns if c not in EXCESS_COLS]
    pd.testing.assert_frame_equal(new[keep], score[keep])          # 원래 열은 그대로
    return _save(new, path, overwrite)


def _save(df: pd.DataFrame, out, overwrite: bool) -> str:
    """이미 있는 결과 파일과 내용이 같으면 그대로 두고, 다르면 --overwrite 없이는 멈춘다 (결과 덮어쓰기 금지)."""
    from pathlib import Path
    out = Path(out)
    text = df.to_csv(index=False, lineterminator="\n")
    if out.exists() and not overwrite:
        if out.read_text(encoding="utf-8-sig").replace("\r\n", "\n") == text:
            return "그대로 (내용 같음)"
        raise SystemExit(f"{out} 이 이미 있고 내용이 다르다. 덮어쓰려면 --overwrite (팀장 승인 뒤)")
    out.write_text(text, encoding="utf-8-sig", newline="\n")
    return "저장"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m indexbeater benchmark", description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--market-out", default=str(MARKET_OUT))
    ap.add_argument("--fetch-index", action="store_true", help=f"{SUB_SYMBOL} 를 인터넷에서 받아 {SUB_FILE.name} 에 저장")
    ap.add_argument("--overwrite", action="store_true", help="내용이 다른 결과 파일을 덮어쓴다")
    ap.add_argument("--scorecard", nargs="?", const=str(SCORECARD), metavar="점수표",
                    help=f"이미 만든 점수표에 지수 초과수익 열만 붙인다 (기본 {SCORECARD.name}). 비교 지수는 다시 만들지 않는다")
    a = ap.parse_args(argv)
    if a.scorecard:
        print(f"{attach_scorecard(a.scorecard, a.market_out, a.overwrite)}: {a.scorecard} (+ 초과수익 열 {len(EXCESS_COLS)}개)")
        return 0
    if a.fetch_index:
        d = fetch_sub_index()
        print(f"받음: {SUB_FILE} ({len(d)}줄, {d['date'].min().date()} ~ {d['date'].max().date()})")
    C.RESULTS.mkdir(exist_ok=True)
    tr = build()
    print(f"{_save(tr, a.out, a.overwrite)}: {a.out} ({len(tr)}줄, {tr['date'].min().date()} ~ {tr['date'].max().date()})")
    print(yearly(tr).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not SUB_FILE.exists():
        print(f"{SUB_FILE.name} 가 없어 시장별 비교 지수는 건너뜀 → --fetch-index 로 먼저 받는다")
        return 0
    ms = market_series(tr, load_sub_index())
    print(f"{_save(ms, a.market_out, a.overwrite)}: {a.market_out} ({len(ms)}줄)")
    print(market_yearly(ms).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0
