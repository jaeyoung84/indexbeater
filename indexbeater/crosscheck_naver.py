"""STEP 1 교차 검증 — 무작위 10종목의 최근 1년 종가를 네이버 금융 일별 시세와 비교.

네이버 일별 시세는 분할을 소급하지 않은 실제 가격이므로, 최근 1년 안에 분할이 없던 종목만 비교한다.
(yfinance Close 의 1년 내 하루 변동이 ±40%를 넘으면 분할 의심으로 보고 다른 종목을 고른다.)

실행: python -m indexbeater crosscheck [--n 10] [--seed 42]
결과: results/crosscheck_naver.csv
"""

# TODO(STEP 1): 계획서 해당 STEP의 '할 일'에 따라 구현한다.


def main(argv=None):
    raise NotImplementedError("STEP 1 구현 전입니다 (crosscheck_naver.py)")
