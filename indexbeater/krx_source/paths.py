"""이 폴더의 코드가 쓰는 파일 위치를 한곳에 모아 둔 파일.

경로를 여기에만 적어 두면, 위치를 바꿀 때 이 파일 하나만 고치면 된다.
(DATA 폴더는 indexbeater/config.py 에서 정한다. 데이터 파일은 용량이 커서 git에 올리지 않는다.)
"""
from __future__ import annotations

from indexbeater.config import DATA

RAW_CACHE = DATA / "krx_raw"                  # 날짜·시장별로 받은 KRX 원본 답(JSON). 중간에 멈춰도 이어받으려고 저장
DART_CACHE = DATA / "dart_raw"                # OpenDART 원본 답 저장 폴더
RAW_DAILY = DATA / "krx_raw_daily.parquet"    # 전 종목 일별 원주가를 하나로 합친 표
SPLIT_EVENTS = DATA / "krx_split_events.csv"  # 찾아낸 분할 목록
SPLIT_REVIEW = DATA / "krx_split_review.csv"  # 자동으로 판단 못 해서 사람이 확인할 목록
DIV_EVENTS = DATA / "krx_dividend_events.csv"
DIV_REVIEW = DATA / "krx_dividend_review.csv"
DIV_OVERRIDE = DATA / "dividend_ex_dates_override.csv"   # 사람이 직접 확정한 배당락일 (code, fiscal_year, ex_date)
ADJUSTED = DATA / "krx_adjusted.parquet"      # 분할·배당까지 보정이 끝난 최종 표

# ---- 비교용 외부 자료 (사용자가 직접 내려받아 넣어 둔 파일) ----
NAVER_DIR = DATA / "naver_data" / "csv_full_20261118"   # 종목별 CSV: Date,Code,Open,High,Low,Close,Volume (분할만 보정된 값)
YAHOO_DIR = DATA / "yahoo_data" / "krx_daily_data_from_2000" / "krx_daily_data_from_2000"   # 005930.KS_삼성전자.csv 식 (분할+배당 보정)
YF_DIR = DATA / "yahoo_data" / "yf"   # 새 야후: Close(분할만) + Adj Close(분할+배당) + Dividends + Stock Splits
ADJUSTED_SPLIT_ONLY = DATA / "krx_adjusted_split_only.parquet"   # 분할만 보정한 표
JUMP_REVIEW = DATA / "krx_jump_review.csv"     # 분할 보정 후에도 남은 큰 변동 목록(종류 표시)
YAHOO_DIVS = DATA / "yahoo_dividends_2011_2014.csv"   # 야후에서 가져온 2011~2014 배당
BASE_EVENTS = DATA / "krx_base_events.csv"   # 기준가 ≠ 전일종가인 큰 변동일 (분할일 제외)
RIGHTS_EVENTS = DATA / "krx_rights_events.csv"   # 주주배정 유상증자 권리락 이벤트 (confirmed 만 보정에 쓴다)
