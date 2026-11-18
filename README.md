# IndexBeater (인덱스비터)

15년 일봉 백데이터로 **매수·매도 신호 18개**를 규칙·머신러닝·딥러닝으로 조합하고 최적화합니다. 그 결과가 미래 정보를 보지 않는 시험에서 **코스피200·코스닥150 지수**를 이기는지 검증하는 파이썬 퀀트 시스템입니다.

> ⚠️ 학습·연구용 코드입니다. 투자 권유가 아니며, 이 코드로 생긴 손익에 책임지지 않습니다.

- 팀: **시그널랩 (Signal Lab)**, 4명 ([역할 분담](docs/역할분담.md))
- 기간: 2026.11.18 ~ 2026.12.02
- 상세 계획: `프로젝트계획서_팀원공유용.docx`

## 빠른 시작 (VS Code + Anaconda)

```bash
conda env create -f environment.yml
conda activate indexbeater
python -m pytest                 # 뼈대 점검 (모듈 import·공통 기준)
```

VS Code에서 폴더를 열고 `Python: Select Interpreter`로 `indexbeater` 환경을 고릅니다. 그러면 **실행 및 디버그(F5)** 메뉴에 단계별 실행 항목이 나타납니다.

## 명령 (각 STEP에서 구현 예정)

| 명령 | 단계 | 하는 일 |
|---|---|---|
| `python -m indexbeater check` | 공통 | 환경·폴더·데이터·미래 정보 테스트 자가 점검 |
| `python -m indexbeater collect` | STEP 1 | 15년 일봉·지수·ETF 수집 (한 명만 실행해서 공유) |
| `python -m indexbeater crosscheck` | STEP 1 | 네이버 금융 일별 시세와 종가 교차 검증 |
| `python -m indexbeater preprocess` | STEP 2 | 거래정지 제거, OHLC 보정, 이상치 목록 |
| `python -m indexbeater optimize` | STEP 3 | 18개 신호 그리드 × H(5·10·20), 설계 구간만 사용 |

## 폴더 구조

```
indexbeater/        패키지
  config.py         공통 기준 (계획서 4장). 바꿀 때는 4명 합의 필요
  collect.py        STEP 1 수집        crosscheck_naver.py  STEP 1 교차 검증
  preprocess.py     STEP 2 전처리      regime.py            장세 라벨 (부록 C)
  indicators.py     지표               signals.py           매수·매도 신호 18개 (부록 A)
  exits.py          매도 규칙 S1~S4    engine.py            공통 백테스트 엔진 (합의 없이 수정 금지)
  metrics.py        평가 지표 (4.5)    lookahead.py         미래 정보 테스트 (부록 D)
  optimize.py       STEP 3 최적화      features.py          STEP 5 피처·라벨
  ml/walkforward.py STEP 5 ML (purge 20일)   ml/dl.py       STEP 5 LSTM·1D-CNN
tests/              pytest (가짜 데이터로 신호·엔진·ML을 검사)
notebooks/          02_EDA.ipynb ~ 07_시각화.ipynb
data/ results/ figures/ models/ dashboard/
실험기록.csv        모든 시도 기록      회의기록.md
```

## 미래 정보를 막는 원칙 (구현할 때 지킬 것)

- `load_wide()`는 기본적으로 **2019-12-31까지만** 불러오게 만든다. 시험 구간(2020~)은 `allow_test=True`일 때만 연다.
- 18개 신호는 모두 **과거 가격 조작 테스트**를 통과해야 한다. 중간 날짜 가격을 바꿨을 때 그 이전 신호가 하나라도 바뀌면 실패다.
- 워크포워드 학습은 학습 끝과 적용 시작 사이를 **20거래일** 비운다.
- `check` 명령은 코드에서 `shift(-n)`, `center=True`, `shuffle=True` 같은 위험한 패턴을 찾아 준다.

## 개발 로드맵

| 단계 | 기간 | 상태 |
|---|---|---|
| STEP 0 환경·기준 | 11/18 | ✅ 뼈대 v0.1 (폴더·설정·공통 기준 `config.py`) |
| STEP 1 수집 | 11/18~19 | 예정 |
| STEP 2 전처리·EDA | 11/19~20 | 예정 |
| STEP 3 신호 18개·매수 최적화 | 11/20~23 | 예정 |
| STEP 4 규칙 조합 | 11/24~25 | 예정 |
| STEP 5 ML·DL | 11/25~27 | 예정 |
| STEP 6 포트폴리오·최종 시험 | 11/28~30 | 예정 (`portfolio.py`) |
| STEP 7·8 시각화·발표 | 11/30~12/02 | 예정 |

## 기여

[CONTRIBUTING.md](CONTRIBUTING.md)를 보세요. 라이선스: [MIT](LICENSE)
