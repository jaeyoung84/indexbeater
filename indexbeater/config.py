"""공통 기준 (계획서 4장) — 모든 코드가 이 값만 쓴다.

여기 값을 바꾸려면 4명 합의 → 실험기록.csv 기록 → PR 리뷰 순서를 지킨다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
EXPERIMENT_LOG = ROOT / "실험기록.csv"

# ---------------------------------------------------------------- 4.1 데이터 기준
DATA_START = pd.Timestamp("2011-11-17")
DATA_END = pd.Timestamp("2026-11-17")

YF_SUFFIX = {"KOSPI": ".KS", "KOSDAQ": ".KQ", "KOSDAQ GLOBAL": ".KQ"}

INDEX_SYMBOLS = {            # FinanceDataReader 심볼
    "KS200": "코스피200",
    "KQ11": "코스닥 종합 (코스닥150 대체·2015.10 이전)",
    "KS11": "코스피 종합",
}
KOSDAQ150_CANDIDATES = ["KQ150"]   # 수집 시 받아지는지 확인, 실패하면 ETF로 대체
BENCH_ETFS = {               # 보조 벤치마크 (배당 반영 수정주가)
    "069500": "KODEX 200",
    "229200": "KODEX 코스닥150",
}

MIN_VALUE_20D = 1e9          # 20일 평균 거래대금 10억 원 이상
MIN_PRICE = 1_000            # 종가 1,000원 이상
OUTLIER_RET = 0.30           # 하루 ±30% 초과 → 이상치 점검 대상

# ---------------------------------------------------------------- 4.2 매매 기준
COST_ROUNDTRIP = 0.0025      # 왕복 0.25% (민감도 0.5%)
COST_SENSITIVITY = 0.005
LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-15")
UPPER_GAP_BEFORE = 0.14      # 가격제한폭 ±15% 시절: +14% 이상 갭이면 매수 불가
UPPER_GAP_AFTER = 0.29       # ±30% 시절: +29% 이상 갭이면 매수 불가

# ---------------------------------------------------------------- 4.3 포트폴리오 기준
INIT_CAPITAL = 100_000_000
MAX_POSITIONS = 20
WEIGHT_PER_POSITION = 0.05

# ---------------------------------------------------------------- 4.4 검증 기준
INIT_TRAIN_END = pd.Timestamp("2014-12-31")    # 최초 학습 2011.11 ~ 2014
DESIGN_START = pd.Timestamp("2015-01-01")      # 설계 확정 2015 ~ 2019
DESIGN_END = pd.Timestamp("2019-12-31")        # ← 선택(신호·파라미터·모델)은 여기까지
TEST_START = pd.Timestamp("2020-01-01")        # 시험 2020.01 ~ 2026.11 (단 한 번)
PURGE_DAYS = 20                                # = 최대 보유기간 H
MAX_HOLD = 20
HOLDS = (5, 10, 20)                            # S3 시간 청산 H 후보

TEST_PERIODS = {   # 부록 C — STEP 2에서 코스피200 차트로 확정 후 수정 (전략 성과를 본 뒤에는 수정 금지)
    "T1": ("2020-01-20", "2020-03-19", "코로나 폭락"),
    "T2": ("2020-03-20", "2021-06-30", "유동성 폭등"),
    "T3": ("2021-07-01", "2022-09-30", "금리인상 약세장"),
    "T4": ("2022-10-01", "2024-12-31", "일반장(회복·횡보)"),
    "T5": ("2025-01-01", "2026-06-30", "폭등장"),
    "T6": ("2026-07-01", "2026-11-17", "최근"),
}

# ---------------------------------------------------------------- 4.5 성공 기준
RANDOM_TRIALS = 1_000
RANDOM_TOP_PCT = 0.05
MIN_PERIODS_WIN = 4          # T1~T6 중 4개 이상
MAX_SHARPE_DROP = 0.50       # 설계 → 시험 샤프 하락 50% 이내

MIN_TRADES = 300             # 신호 성적표에서 최소 거래수 (이보다 적으면 선택 제외)
SEED = 42
