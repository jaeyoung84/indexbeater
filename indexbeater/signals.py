"""매수·매도 신호 18개 (계획서 부록 A).

형식 (모든 신호 공통)
- 입력: wide 데이터 dict (Open·High·Low·Close·Volume, 분할 반영 원본 가격)
- 출력: (entry, exit) — 날짜×종목 True/False 표 두 개
- entry[t] = t일 종가까지 보고 "내일 시가에 사라", exit[t] = "내일 시가에 팔아라"
- 15번 변동성 돌파는 자체 매도 조건이 없어 exit가 None (매도 전용 규칙 S1~S3 사용)

새 신호를 추가하면 SIGNALS에 등록하고 `pytest tests/test_lookahead.py` 를 통과해야 한다.
"""

# TODO(STEP 3): 계획서 해당 STEP의 '할 일'에 따라 구현한다.
