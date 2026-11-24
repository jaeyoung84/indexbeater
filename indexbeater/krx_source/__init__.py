"""KRX 기준 수정주가를 만드는 꾸러미 (4인판 계획서 2.2~2.3절).

순서대로 네 단계로 되어 있다.
  collect    KRX Open API에서 일별 전종목 원주가를 받는다   → data/krx_raw_daily.parquet
  splits     상장주식수가 갑자기 늘어난 날로 분할·무상증자를 찾는다 → data/krx_split_events.csv (+ 사람이 볼 목록)
  dividends  OpenDART에서 배당을 받아 배당 이벤트를 만든다   → data/krx_dividend_events.csv
  adjust     분할 → 배당 순서로 과거 가격을 고친다           → data/krx_adjusted.parquet

터미널 실행:  python -m indexbeater.krx_source <명령> [옵션]
설명을 보면서 단계별로 실행하려면 notebooks/ 폴더의 노트북을 쓴다.

키(비밀번호)는 환경변수로만 받는다. 코드나 노트북, 문서에 적지 않는다:  KRX_AUTH_KEY, OPENDART_KEY
"""
