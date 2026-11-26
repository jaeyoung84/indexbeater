# 매도 ML·DL 공통 시작 코드 사용법 (11/26 팀장)

대상: 매도 ML·DL 을 각자 만드는 팀원 (김동겸·김승찬·김정철)

모두가 **같은 매수 신호, 같은 매도 계산, 같은 점수**로 비교하려고 만든 시작 코드입니다.
파일: `indexbeater/sell_common.py` · 실행: `python -m indexbeater sellml ...`

---

## 1. 준비

1) 팀장 브랜치 `jaeyoung/engine-v1` 의 최신 코드를 받습니다 (팀장이 올린 뒤).
2) 패키지를 설치합니다 (버전은 그대로 맞춰 주세요).

```
pip install lightgbm==4.7.0 xgboost==3.4.1 catboost==1.2.10 "tensorflow==2.19.1"
```

3) 아래 파일이 `results/` 폴더에 있어야 합니다.
   - `top3_buy_signals.csv` (TOP3 매수 신호)
   - `team_compare_portfolio_daily.csv` (점검에서 회의 PDF 와 맞춰 보는 데 씀)
   - `benchmark_market.csv` (비교 지수)
4) 데이터(parquet)는 각자 수집한 것을 씁니다. 2020년 이후는 열지 않습니다.

## 2. 먼저 점검 (반드시)

```
python -m indexbeater sellml --check
```

- 약 20초 걸립니다. 아무것도 저장하지 않습니다.
- 줄마다 ✓ 가 나와야 합니다: CSV 매수가 · 갭취소 · 고정 H5/H10/H20 매매 표 = 엔진 · H5 포트폴리오 6칸 = 회의 PDF.
- **✗ 가 하나라도 있으면 학습하지 말고 팀장에게 알려 주세요.** 데이터가 서로 다르다는 뜻입니다.

## 3. 정해진 규칙 (11/26, 바꾸지 않음)

| 항목 | 규칙 |
|---|---|
| 모델 수 | 시장(KOSPI·KOSDAQ)마다 1개, 전략 이름은 입력에 안 넣음 |
| 학습 신호 | 2015년 신호부터 |
| 시험 방식 | 해마다 앞으로 넓혀 가며 학습 → 2016·2017·2018·2019 시험 |
| 정화 40 | 판단한 날 + 20 = 라벨 끝, 그 뒤 20거래일 빈칸. 라벨 끝 + 20거래일이 시험 해 첫날보다 앞인 줄만 학습 |
| 판단 | d일 장 끝에 '판다/들고 있다' 판단 → 다음 날부터 팔 수 있는 첫 시가에 팖 |
| 최대 보유 | **없음.** 해가 바뀌면 그해 모델이 이어서 판단. 2019-12-30 까지 못 팔면 평가만 |
| 손절·익절 | 고정 손절·익절 없음 (고정 H5·H10·H20 은 비교용) |
| 라벨 | 'd 다음 첫 시가' 와 'd+20 다음 첫 시가' 비교 → 들고 있는 쪽이 0.5% 넘게 나쁘면 판다(1) |
| 가중치 | 두 가격 차이의 크기 ÷ 그 매매의 줄 수 |
| 학습 줄 | 매수 뒤 120거래일까지 |
| 기준 확률 | 0.55 / 0.60 / 0.65 중 학습 구간 마지막 20% 에서 평균 순수익이 가장 큰 것 |
| 점수 | TOP3 6칸마다 포트폴리오, 첫째 순위 기준 = 지수 초과수익 중앙값 |

## 4. 명령

| 명령 | 하는 일 |
|---|---|
| `--check` | 점검 (저장 안 함) |
| `--list` | 모델 7종 · 설정 · 입력 17개 목록 |
| `--baseline` | 고정 H5·H10·H20 비교 결과 → `results/sell_ml_baseline_*.csv` |
| `--run 모델이름` | 그 모델의 모든 설정 → `results/sell_ml_모델이름_*.csv` |
| `--run 모델이름 --setting 0 2` | 설정 0번·2번만 |
| `--overwrite` | 이미 있는 결과 파일 덮어쓰기 (팀장 승인 뒤에만) |

모델 이름: `logistic`(3) · `random_forest`(4) · `lightgbm`(5) · `xgboost`(5) · `catboost`(5) · `lstm`(3) · `cnn1d`(3).
괄호 = 설정 수. **설정 하나 = 시도 하나**로 셉니다.

예:
```
python -m indexbeater sellml --run xgboost
python -m indexbeater sellml --run catboost --setting 1
```

결과 파일: `score`(칸별 점수) · `rank`(순위) · `years`(해마다 학습 정보) · `importance`(입력 중요도) · `daily`(포트폴리오 날마다) · `trades`(매매 표).

## 5. 내 모델 넣기 (11/27 전에 정한 것만)

sklearn 모양(`fit(X, y, sample_weight=w)`, `predict_proba(X)`)이면 됩니다.

```python
from indexbeater import sell_common as SL
from sklearn.ensemble import ExtraTreesClassifier

SL.add_model("my_extra", lambda p: ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, **p),
             [{"max_depth": 8}, {"max_depth": 12}])
ctx = SL.load_ctx()
out = SL.run_model(ctx, "my_extra")
print(SL.rank_table(out["score"]))
```

- 모델과 설정 목록은 **11/27 전에 팀장에게 알려 주세요.** 시험 결과를 보고 설정을 더하면 안 됩니다.
- 입력 17개(`--list`)는 모두 같은 것을 씁니다. 입력을 바꾸고 싶으면 먼저 팀장과 상의합니다.

## 6. 주의

- 학습 줄이 많습니다 (2019 시험 해: 시장마다 100만 줄 넘음). 랜덤포레스트·LSTM·1D CNN 은 오래 걸릴 수 있습니다.
  LSTM·1D CNN 은 기본으로 20만 줄만 뽑아 학습합니다.
- LSTM·1D CNN(텐서플로)은 같은 seed 라도 결과가 조금씩 다를 수 있습니다.
- 결과를 보고 규칙을 바꾸지 않습니다. 바꿔야 할 것 같으면 먼저 팀장에게 말합니다.
- 2020년 이후 자료(시험 8)는 팀 전체가 한 번만 씁니다. 각자 열지 않습니다.
