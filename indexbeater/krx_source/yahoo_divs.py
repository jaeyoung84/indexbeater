"""야후 파일의 배당(Dividends 열)으로 2011~2014 배당 이벤트를 만드는 파일.

# 왜 필요한가
- OpenDART 배당 자료는 2015 사업연도부터라서 2011~2014년 배당은 비어 있다. 이 구간을 야후 배당으로 채운다.
- 야후는 배당락일(실제 날짜)과 1주당 배당금을 직접 준다. DART에서 연말로 '추정'한 것보다 날짜가 정확하다.

# 주의: 야후 배당금은 분할이 이미 반영된 값이다 (삼성전자 2013-12-27: 13,800원 ÷ 50 = 276)
- 그래서 계수는 '분할만 보정한 전일 종가(close_s)'로 나눈다: 계수 = 1 - 배당금 / 전일 close_s.
  (분할 전의 진짜 가격과 진짜 배당금으로 계산해도 같은 값이 나온다.)
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import external, paths

FIRST, LAST = pd.Timestamp("2011-11-17"), pd.Timestamp("2014-12-31")   # 야후 배당으로 채울 기간 (DART가 없는 구간)


def build_yahoo_events(adj_split_only: pd.DataFrame, first=FIRST, last=LAST) -> pd.DataFrame:
    """분할만 보정한 KRX 표에서 코스피·코스닥 종목마다 야후 배당을 찾아 이벤트 표로 만든다."""
    k = adj_split_only[adj_split_only["market"].isin(["KOSPI", "KOSDAQ"])]
    out = []
    for code, g in k.groupby("code", sort=False):
        y = external.load_yahoo(code)
        if y is None:
            continue
        d = y[(y["dividends"] > 0) & (y.index >= first) & (y.index <= last)]
        if d.empty:
            continue
        cs = g.sort_values("date").set_index("date")["close_s"]
        for ex, div in d["dividends"].items():
            prev = cs[cs.index < ex]            # 배당락일 전의 거래일들
            if prev.empty:                      # KRX에 그 전 가격이 없으면 계산할 수 없다
                continue
            f = 1 - div / prev.iloc[-1]
            if 0.5 < f < 1:                     # 배당이 주가의 절반을 넘는 건 말이 안 되니 뺀다
                out.append({"code": code, "fiscal_year": ex.year, "dps": div, "ex_date": ex,
                            "prev_close": prev.iloc[-1], "factor": f, "confidence": "yahoo"})
    return pd.DataFrame(out)


def main(argv=None):
    """터미널에서 `python -m indexbeater.krx_source yahoo_divs` 로 실행한다."""
    argparse.ArgumentParser(prog="krx_source yahoo_divs").parse_args(argv)
    adj = pd.read_parquet(paths.ADJUSTED_SPLIT_ONLY, columns=["date", "market", "code", "close_s"])
    ev = build_yahoo_events(adj)
    ev.to_csv(paths.YAHOO_DIVS, index=False, encoding="utf-8-sig")
    print(f"야후 배당 이벤트 {len(ev):,}건 ({ev['code'].nunique()}종목) → {paths.YAHOO_DIVS}")


if __name__ == "__main__":
    main()
