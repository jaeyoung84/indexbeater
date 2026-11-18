# 협업 규칙 (4명)

1. `main`에 직접 push하지 않습니다. `이름/작업` 브랜치에서 작업하고 PR을 올립니다. 예: `jaeyoung/s07-rsi`
2. PR을 올리기 전에 `python -m indexbeater check`와 `python -m pytest`를 실행합니다. 둘 다 통과해야 합니다.
3. 다른 팀원 1명 이상이 리뷰합니다. 리뷰할 때는 **shift 방향**과 **rolling center=True 사용 여부**를 먼저 봅니다.
4. `config.py`(공통 기준)와 `engine.py`는 4명이 합의한 뒤에만 수정합니다. 합의 내용은 `회의기록.md`와 `실험기록.csv`에 남깁니다.
5. 새로 시도할 때마다 `실험기록.csv`에 한 줄을 적습니다. `optimize`는 구현할 때 자동 기록하게 만듭니다.
6. 데이터(parquet)는 git에 올리지 않고 구글드라이브로 공유합니다.
7. 새 신호를 추가하면 `signals.py`의 `SIGNALS`에 등록합니다. 미래 정보 테스트(`tests/test_lookahead.py`)를 통과해야 합니다.
