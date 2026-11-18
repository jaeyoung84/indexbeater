"""데이터 입출력: long(Date, Code, ...) ↔ wide(행=날짜, 열=종목) 변환.

wide 데이터는 dict[str, DataFrame] 이고 키는 다음과 같다.
  Open High Low Close AdjClose Volume  — 원본 (Close는 분할 소급 반영, AdjClose는 배당까지)
  adj                                   — 수정 계수 AdjClose / Close
  AO AH AL AC                           — 수정 시가·고가·저가·종가 (수익률 계산용)
  tradable                              — 유동성 필터 통과 여부 (그날까지의 데이터로만 계산)
"""

# TODO(STEP 1~2): 계획서 해당 STEP의 '할 일'에 따라 구현한다.
