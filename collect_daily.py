"""
15년치 일봉 데이터 수집 (KOSPI + KOSDAQ 보통주, 무료 소스)

- 종목 목록 : FinanceDataReader StockListing('KRX')
- 개별 종목 : yfinance (Yahoo)  -> 2000년부터 제공, 액면분할 소급 반영, Adj Close = 배당까지 반영
- 지수      : FinanceDataReader (KS11, KQ11, KS200)

제외: KONEX, 스팩, 우선주(종목코드 끝자리 != 0)
한계: 현재 상장 종목만 받음 -> 상장폐지 종목 빠짐(생존자 편향). 상폐 목록은 data/delisted.csv 로 따로 저장.

실행:  python collect_daily.py            (기본 15년)
       python collect_daily.py --years 10
결과:  data/daily_ohlcv.parquet  (Date, Code, Open, High, Low, Close, AdjClose, Volume)
       data/index_ohlcv.parquet, data/universe.csv, data/failed.csv, data/delisted.csv
"""
import argparse
import time
import warnings
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

DATA = Path(__file__).parent / "data"
SUFFIX = {"KOSPI": ".KS", "KOSDAQ": ".KQ", "KOSDAQ GLOBAL": ".KQ"}


def build_universe() -> pd.DataFrame:
    lst = fdr.StockListing("KRX")
    u = lst[lst["Market"].isin(SUFFIX)].copy()
    u = u[~u["Name"].str.contains("스팩")]
    u = u[u["Code"].str.endswith("0")]  # 우선주 제외
    u["Ticker"] = u["Code"] + u["Market"].map(SUFFIX)
    return u[["Code", "Name", "Market", "Ticker"]].reset_index(drop=True)


def download_chunk(tickers, start, end, retries=3):
    for i in range(retries):
        try:
            return yf.download(tickers, start=start, end=end, auto_adjust=False,
                               group_by="ticker", threads=True, progress=False)
        except Exception as e:  # 네트워크 오류 시 재시도
            print("  retry", i + 1, e)
            time.sleep(5 * (i + 1))
    return pd.DataFrame()


def to_long(raw: pd.DataFrame, tickers) -> list:
    out = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        d = raw[t].dropna(subset=["Close"])
        if d.empty:
            continue
        d = d.rename(columns={"Adj Close": "AdjClose"})
        d = d[["Open", "High", "Low", "Close", "AdjClose", "Volume"]].copy()
        d["Code"] = t.split(".")[0]
        out.append(d.reset_index())
    return out


def main(years: int, chunk: int):
    DATA.mkdir(exist_ok=True)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=years)
    print(f"기간: {start.date()} ~ {end.date()}")

    uni = build_universe()
    uni.to_csv(DATA / "universe.csv", index=False, encoding="utf-8-sig")
    print("대상 종목:", len(uni), uni["Market"].value_counts().to_dict())

    frames, tickers = [], uni["Ticker"].tolist()
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        frames += to_long(download_chunk(part, start, end), part)
        print(f"  {min(i + chunk, len(tickers))}/{len(tickers)} 완료")
        time.sleep(1)

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
    df.to_parquet(DATA / "daily_ohlcv.parquet", index=False)

    got = set(df["Code"])
    uni[~uni["Code"].isin(got)].to_csv(DATA / "failed.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {len(df):,} 행, {len(got)} 종목 (실패 {len(uni) - len(got)})")

    idx = []
    for sym in ["KS11", "KQ11", "KS200"]:
        d = fdr.DataReader(sym, start, end)[["Open", "High", "Low", "Close", "Volume"]]
        d["Symbol"] = sym
        idx.append(d.rename_axis("Date").reset_index())
    pd.concat(idx).to_parquet(DATA / "index_ohlcv.parquet", index=False)

    try:
        fdr.StockListing("KRX-DELISTING").to_csv(DATA / "delisted.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print("상폐 목록 실패:", e)
    print("끝")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--chunk", type=int, default=100)
    a = ap.parse_args()
    main(a.years, a.chunk)
