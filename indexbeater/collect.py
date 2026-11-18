"""STEP 1 데이터 수집 — 15년 일봉 (KOSPI·KOSDAQ 보통주) + 지수 + 보조 벤치마크 ETF.

실행 (한 명만, 결과 파일을 공유 드라이브로 공유):
    python -m indexbeater collect                  # 전체
    python -m indexbeater collect --limit 30       # 시험 삼아 30종목만
    python -m indexbeater collect --retry-failed   # failed.csv 종목만 다시

출처
- 종목 목록 : FinanceDataReader StockListing('KRX')
- 종목 일봉 : yfinance (코스피 = 코드.KS, 코스닥 = 코드.KQ). Close = 분할 소급, Adj Close = 배당까지
- 지수      : FinanceDataReader (KS200, KQ11, KS11, 코스닥150 후보)
- ETF       : yfinance 069500.KS, 229200.KS (배당 반영 수정주가)

결과: data/daily_ohlcv.parquet  (Date, Code, Open, High, Low, Close, AdjClose, Volume)
      data/index_ohlcv.parquet  (Date, Symbol, Open, High, Low, Close, AdjClose, Volume)
      data/universe.csv, data/failed.csv, data/delisted.csv, data/collect_meta.json
"""

# TODO(STEP 1): 계획서 해당 STEP의 '할 일'에 따라 구현한다.


def main(argv=None):
    raise NotImplementedError("STEP 1 구현 전입니다 (collect.py)")
