"""매도 ML·DL 공통 시작 코드 — 팀원 각자 매도 모델을 같은 규칙으로 학습·채점하는 틀 (11/26 팀장)

성격: 매수는 TOP3 매수 신호 CSV (results/top3_buy_signals.csv, buy_export.py) 그대로. 이 파일은 '언제 팔까'만 배운다.
      엔진(engine.py·signals.py·exits.py·indicators.py·data_io.py·config.py)과 점수표(scorecard.py·metrics.py·
      scoring.py)는 고치지 않는다. 엔진 매도는 '정해진 날(최대 20일)'만 되어서, 매도 한 건 정산과 포트폴리오 하루
      돌리기를 이 파일에 엔진과 똑같이 옮겨 적었다. 옮겨 적은 것이 맞는지: 고정 H5·H10·H20 으로 돌리면 엔진과
      끝자리까지 같아야 한다 (--check, 테스트).

11/26 팀장 결정 (매도 부분만 바뀜)
  - 시장마다 모델 1개 (전략 이름은 입력에 안 넣는다). 2015 이후 신호만 쓴다. 칸마다 다른 대체 규칙 없음.
  - 최대 보유 기간 없음. 연말 강제 매도 없음. 해가 바뀌면 그 해 모델이 판단한다. 2019-12-30 은 평가만 한다.
  - 고정 손절·익절 없음 (S1·S2·S3 는 비교용 기준선만).
  - 라벨: d일 장 끝에 판단. '지금 판다' = d 다음 팔 수 있는 첫 시가, '20거래일 더 들고 판단' = d+20 다음 팔 수 있는 첫 시가.
          더 들고 있는 쪽이 0.5% 넘게 나쁘면 y = 1 (판다). 20 은 내다보는 거리일 뿐 보유 한도가 아니다.
  - 가중치: |두 값 차이| ÷ (그 매매의 학습 줄 수). 평균 1 로 맞춘다. (선택) 두 쪽 합을 같게 (balance).
  - 정화(purge) 40 = 라벨 끝(20) + 빈칸 20. 시험 해 Y 의 학습 줄은 '라벨 끝 매도일 + 20 < Y년 첫날' 만.
  - 학습 줄: 매수일부터 120거래일까지의 날 (PANEL_DAYS).
  - 기준 확률 (0.55·0.60·0.65) 은 학습 구간 마지막 20% 로 고른다 (그 구간 매매의 평균 순수익이 가장 큰 것, 같으면 앞).
    고른 뒤 학습 구간 전부로 다시 학습한다 (refit). DL 은 처음 학습 때 가장 좋았던 에폭 수로 다시 학습.
  - 모델: 계획서 5종 (logistic·random_forest·lightgbm·lstm·cnn1d) + xgboost·catboost (각 5설정).
          다른 모델도 11/27 전에 정해 두면 add_model() 로 넣어 같은 방법으로 채점한다 (설정 하나 = 시도 하나).
  - 채점: TOP3 6칸 (시장 3칸씩) 포트폴리오. 2016 신호부터, 평가금액 2016~2019. 1억·20자리·5%, 비용 0.5%·0.25%.
          순위 첫 기준 = 해마다 지수 초과수익의 중앙값 (0.5% 줄).

모델이 보는 것 (모두 d일 장 끝까지 알 수 있는 값, 수정 가격)
  종목: r1·r5·r20 수익, 5·20·60일 이동평균과의 거리, RSI14, ATR14÷종가, 거래대금÷20일 평균, 20일 고가와의 거리,
        r20 − 그 시장 지수 r20
  시장: 지수 r20, 지수 > 120일·200일 평균 (1/0)
  매매: 산 값 대비 수익, 산 뒤 가장 높았던 종가 대비 하락, 산 뒤 거래일 수
  DL 은 위 '종목' 값의 60일 묶음 + 전체 값 (그날 한 줄) 을 같이 본다.
  값 하나라도 빈 날(예: 상장 60일 안 된 종목의 60일 평균)은 학습에서 빼고, 시험에서는 그날 판단 안 함(그대로 보유).

실행 (설계 구간만, 2020 이후는 열지 않는다)
  python -m indexbeater sellml --check              옮겨 적은 계산 = 엔진·PDF 포트폴리오 점검 + 학습 줄 수 (저장 안 함)
  python -m indexbeater sellml --list               모델·설정·입력 값 목록
  python -m indexbeater sellml --baseline           비교용 고정 H5·H10·H20 → results/sell_ml_baseline_*.csv
  python -m indexbeater sellml --run lightgbm       그 모델 모든 설정 → results/sell_ml_lightgbm_*.csv
       --setting 0                                  설정 하나만 (0부터)
       --overwrite                                  이미 있는 결과 파일이 다를 때 덮어쓰기 (팀장 승인 뒤)
  --run · --baseline 은 실험기록.csv 에 '규칙' 줄을 먼저 적고 돌린다.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from indexbeater import config as C
from indexbeater import engine as E

LOOK = 20                         # 라벨: 20거래일 더 들고 있었으면 어땠나
GAP = 20                          # 라벨 끝 뒤 빈칸 (정화 = LOOK + GAP = 40)
MARGIN = 0.005                    # 더 들고 있는 쪽이 0.5% 넘게 나쁘면 판다
PANEL_DAYS = 120                  # 학습 줄: 매수일부터 120거래일
VAL_FRAC = 0.2                    # 학습 구간 마지막 20% 로 기준 확률을 고른다
FIRST_YEAR = 2015                 # 2015 이후 신호만
TEST_YEARS = (2016, 2017, 2018, 2019)
CHUNK = 60                        # 시험 때 한 번에 보는 날 수 (계산 묶음일 뿐, 결과와 상관없음)
THRESHOLDS = tuple(C.PROB_THRESHOLDS)
COSTS = {"0.5%": C.COST_SELECT, "0.25%": C.COST_ROUNDTRIP}
ML_REASON = "ML"
FIXED_H = (5, 10, 20)             # 비교용 고정 보유일

DAY_FEATS = ["r1", "r5", "r20", "ma5_gap", "ma20_gap", "ma60_gap", "rsi14", "atr14_pct", "value_ratio",
             "hi20_gap", "ex20"]
MKT_FEATS = ["idx_r20", "idx_up120", "idx_up200"]
TRADE_FEATS = ["ret_buy", "dd_peak", "held"]
FEATURES = DAY_FEATS + MKT_FEATS + TRADE_FEATS

SIG_CSV = C.RESULTS / "top3_buy_signals.csv"


def out_path(model: str, part: str) -> Path:
    return C.RESULTS / f"sell_ml_{model}_{part}.csv"


# ================================================================ 준비: 데이터·특징·CSV 신호
class SellCtx:
    """매도 학습에 매번 쓰는 것을 한 번만 만들어 두는 상자.

    data        data_io.load_wide() 결과 (설계 구간)
    sig         TOP3 CSV (FIRST_YEAR 이후 줄, t·j 열 붙임)
    index_level 시장 → 지수 값 Series (기본: benchmark_market.csv)
    """

    def __init__(self, data: dict, sig: pd.DataFrame, index_level: dict | None = None,
                 first_year: int = FIRST_YEAR, test_years=TEST_YEARS, log=print):
        self.data = data
        self.m = E._Market(data)
        self.dates = self.m.dates
        self.years = np.asarray(self.dates.year)
        self.first_year = int(first_year)
        self.test_years = tuple(int(y) for y in test_years)
        self.close_ff = pd.DataFrame(self.m.Cl).ffill().to_numpy()
        self.ystart = {int(y): int(np.argmax(self.years == y)) for y in np.unique(self.years)}
        if index_level is None:
            index_level = load_index_level()
        self.mkt = market_features(index_level, self.dates)
        self.feat = day_features(data, self.m, self.mkt)
        self._seq = None
        self.sig = attach_tj(sig, self, log=log)
        self.pool = {mk: trades_of(self, mk, self.first_year) for mk in self.markets()}
        self.test = {mk: trades_of(self, mk, self.test_years[0], filled_only=False) for mk in self.markets()}

    def markets(self):
        return list(dict.fromkeys(self.sig["시장"]))

    @property
    def seq(self) -> np.ndarray:
        """DL 용 (날짜, 종목, 종목 특징) float32 묶음. 처음 쓸 때 만든다."""
        if self._seq is None:
            self._seq = np.stack([self.feat[k] for k in DAY_FEATS], axis=-1).astype(np.float32)
        return self._seq


def load_index_level(path=None) -> dict:
    """benchmark_market.csv 의 시장별 지수 값 (2020 이후 줄이 있으면 benchmark.load_market 이 멈춘다)."""
    from indexbeater import benchmark as B
    ms = B.load_market() if path is None else B.load_market(path)
    return {mk: g.set_index("date")["index"].astype(float) for mk, g in ms.groupby("market", sort=False)}


def market_features(index_level: dict, dates: pd.DatetimeIndex) -> dict:
    """시장 → (날짜 수, 3) 배열: 지수 r20, 지수 > 120일 평균, 지수 > 200일 평균. 지수 쉬는 날은 앞 값."""
    out = {}
    for mk, s in index_level.items():
        s = s[~s.index.duplicated()].sort_index()
        s = s.reindex(s.index.union(dates)).ffill().reindex(dates)
        r20 = s / s.shift(20) - 1
        ma120 = s.rolling(120, min_periods=120).mean()
        ma200 = s.rolling(200, min_periods=200).mean()
        up120 = (s > ma120).astype(float).where(ma120.notna())
        up200 = (s > ma200).astype(float).where(ma200.notna())
        out[mk] = np.column_stack([r20.to_numpy(float), up120.to_numpy(float), up200.to_numpy(float)])
    return out


def day_features(data: dict, m: E._Market, mkt: dict) -> dict:
    """종목 특징 (날짜 × 종목, float32). 모두 그날 장 끝까지의 값으로만 계산한다 (미래 엿보기 테스트 있음)."""
    from indexbeater import indicators as I
    px, hi, lo = data["adj_close"], data["adj_high"], data["adj_low"]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = {
            "r1": px / px.shift(1) - 1,
            "r5": px / px.shift(5) - 1,
            "r20": px / px.shift(20) - 1,
            "ma5_gap": px / I.sma(px, 5) - 1,
            "ma20_gap": px / I.sma(px, 20) - 1,
            "ma60_gap": px / I.sma(px, 60) - 1,
            "rsi14": I.rsi(px, 14) / 100,
            "atr14_pct": I.atr(hi, lo, px, 14) / px,
            "value_ratio": data["value"] / data["value20"],
            "hi20_gap": px / hi.rolling(20, min_periods=20).max() - 1,
        }
        out = {k: v.to_numpy(float).astype(np.float32) for k, v in f.items()}
        ex = np.full(out["r20"].shape, np.nan, dtype=np.float32)
        for mk, arr in mkt.items():
            here = m.market == mk
            ex = np.where(here, out["r20"] - arr[:, [0]].astype(np.float32), ex)
        out["ex20"] = ex.astype(np.float32)
    for k in out:
        out[k][~np.isfinite(out[k])] = np.nan
    return out


def read_signals(path=SIG_CSV) -> pd.DataFrame:
    """TOP3 매수 신호 CSV 를 읽는다 (종목코드는 글자, 앞자리 0 유지)."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"종목코드": str}, parse_dates=["신호일"],
                     float_precision="round_trip")             # 소수 끝자리까지 쓴 그대로 읽기 (매수가 점검)
    need = {"시장", "구분", "전략", "신호일", "종목코드", "매수순서값"}
    if not need <= set(df.columns):
        raise SystemExit(f"✗ {path}: 열이 모자란다 {sorted(need - set(df.columns))}")
    if (df["신호일"] >= C.TEST_START).any():
        raise SystemExit("✗ 신호 CSV 에 2020 이후 줄이 있다 (시험 구간)")
    return df


def attach_tj(sig: pd.DataFrame, ctx: SellCtx, log=print) -> pd.DataFrame:
    """FIRST_YEAR 이후 줄만 남기고 날짜 번호 t·종목 번호 j 를 붙인다. 엔진이 처음부터 안 보는 칸이 있으면 멈춘다."""
    m = ctx.m
    s = sig[pd.to_datetime(sig["신호일"]).dt.year >= ctx.first_year].copy()
    t = m.dates.get_indexer(pd.to_datetime(s["신호일"]))
    j = pd.Index(m.codes).get_indexer(s["종목코드"])
    if (t < 0).any() or (j < 0).any():
        raise SystemExit(f"✗ 신호 CSV 의 날짜·종목 {int(((t < 0) | (j < 0)).sum())}줄이 내 데이터에 없다 → 데이터가 다르다")
    ok = (ctx.data["tradable"].to_numpy(bool)[t, j] & np.asarray(m.dates[t] >= C.SIGNAL_START)
          & (m.market[t, j] == s["시장"].to_numpy()))
    if not ok.all():
        raise SystemExit(f"✗ 신호 CSV {int((~ok).sum())}줄이 엔진이 안 보는 칸 (유동성·시장) → 데이터가 다르다")
    if s.duplicated(["시장", "구분", "신호일", "종목코드"]).any():
        raise SystemExit("✗ 신호 CSV 에 같은 칸·날·종목 줄이 두 번 있다")
    s["t"], s["j"] = t, j
    log(f"  신호 {len(s):,}줄 ({ctx.first_year}년부터) · 칸 {s.groupby(['시장', '구분']).ngroups}개")
    return s.reset_index(drop=True)


def trades_of(ctx: SellCtx, mk: str, from_year: int, filled_only: bool = True) -> dict:
    """시장 하나의 매매 (t, j) — 그 시장 3칸 신호를 합쳐 같은 (날, 종목)은 한 번만. filled_only 면 실제로 산 것만."""
    s = ctx.sig[(ctx.sig["시장"] == mk) & (ctx.years[ctx.sig["t"].to_numpy()] >= from_year)]
    tj = np.unique(np.column_stack([s["t"].to_numpy(np.int64), s["j"].to_numpy(np.int64)]), axis=0)
    t, j = (tj[:, 0], tj[:, 1]) if len(tj) else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    buy = buy_arrays(ctx.m, t, j)
    if filled_only:
        k = buy["ok"]
        return {"t": t[k], "j": j[k], "b": buy["b"][k], "buy": buy["buy_px"][k]}
    return {"t": t, "j": j, **buy}


# ================================================================ 사기·팔기 (엔진 _one_trade 와 같은 규칙)
def buy_arrays(m: E._Market, t, j) -> dict:
    """T일 신호 → T+1 시가 매수. 엔진과 같은 상태 (내일 자료 없음 = 미완료, 거래정지 = 매수불가, 상한가 갭 = 취소)."""
    n = m.n
    t = np.asarray(t, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    b = t + 1
    status = np.full(len(t), E.FILLED, dtype=object)
    buy = np.full(len(t), np.nan)
    end = b >= n
    bb = np.minimum(b, n - 1)
    no_buy = ~end & ~m.traded[bb, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = m.O[bb, j] / m.Cl[t, j] - 1
    gap_cancel = ~end & ~no_buy & (gap >= m.thr[bb] - E.FLOAT_EPS)
    ok = ~end & ~no_buy & ~gap_cancel
    buy[ok] = m.O[bb[ok], j[ok]]
    status[end] = E.OPEN_END
    status[no_buy] = E.NO_BUY
    status[gap_cancel] = E.GAP_CANCEL
    return {"b": b, "ok": ok, "status": status, "buy_px": buy}


def settle(m: E._Market, j, d) -> dict:
    """d일 장 끝에 '판다'고 정했을 때: d 다음 팔 수 있는 첫 날 시가.
    그런 날이 없고 상장폐지면 마지막 종가 (s = 마지막 날), 자료 끝(2019-12-30)까지 없으면 미완료 (s = −1, 평가만)."""
    n = m.n
    j = np.asarray(j, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    s = m.nxt[np.clip(d + 1, 0, n), j]
    px = np.full(len(d), np.nan)
    at_open = s < n
    px[at_open] = m.O[s[at_open], j[at_open]]
    last = m.last[j]
    dl = ~at_open & (last < n - 1)
    px[dl] = m.Cl[last[dl], j[dl]]
    s_idx = np.where(at_open, s, np.where(dl, last, -1))
    status = np.where(at_open | dl, E.FILLED, E.OPEN_END).astype(object)
    reason = np.where(at_open, ML_REASON, np.where(dl, E.DELIST, None)).astype(object)
    return {"s_idx": s_idx, "sell_px": px, "sell_at_open": at_open, "status": status, "reason": reason}


def trade_record(m: E._Market, t, j, decide) -> dict:
    """매매 기록 (엔진 TRADE_COLS 와 같은 열). decide = 매수일 번호 b → 판단한 날 d (판단 없으면 m.n 이상)."""
    t = np.asarray(t, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    buy = buy_arrays(m, t, j)
    ok = buy["ok"]
    N = len(t)
    rec = {"t_idx": t, "b_idx": buy["b"], "s_idx": np.full(N, -1, np.int64), "buy_px": buy["buy_px"],
           "sell_px": np.full(N, np.nan), "reason": np.full(N, None, dtype=object), "delay": np.zeros(N, np.int64),
           "sell_at_open": np.zeros(N, bool), "status": buy["status"].copy()}
    if ok.any():
        d = np.asarray(decide, dtype=np.int64)[ok] if np.ndim(decide) else np.full(int(ok.sum()), decide)
        st = settle(m, j[ok], d)
        for k in ("s_idx", "sell_px", "sell_at_open", "reason"):
            rec[k][ok] = st[k]
        rec["status"][ok] = st["status"]
    rec["code"] = m.codes.to_numpy()[j]
    rec["market"] = m.market[t, j]
    return rec


def fixed_record(m: E._Market, t, j, k: int) -> dict:
    """고정 H 거래일 보유 = 엔진 S3 (매수일 + k 거래일 시가, 못 팔면 팔 수 있는 첫 날). d = b + k − 1 장 끝에 판단."""
    t = np.asarray(t, dtype=np.int64)
    rec = trade_record(m, t, j, t + 1 + k - 1)
    sold = (rec["status"] == E.FILLED) & rec["sell_at_open"]
    s0 = rec["b_idx"] + k
    rec["delay"] = np.where(sold, rec["s_idx"] - s0, 0)
    dl = (rec["status"] == E.FILLED) & ~rec["sell_at_open"]
    rec["delay"] = np.where(dl, np.maximum(rec["s_idx"] - s0, 0), rec["delay"])
    rec["reason"] = np.where(sold, "S3", rec["reason"]).astype(object)
    return rec


def entry_table(m: E._Market, t, j) -> pd.DataFrame:
    """(t, j) 칸만 True 인 날짜 × 종목 표 (엔진 simulate_trades 에 넣는 모양)."""
    a = np.zeros((m.n, len(m.codes)), dtype=bool)
    a[np.asarray(t, np.int64), np.asarray(j, np.int64)] = True
    return pd.DataFrame(a, index=m.dates, columns=m.codes)


def to_table(rec: dict, m: E._Market) -> pd.DataFrame:
    """엔진 _to_table 로 같은 모양의 매매 표 + 종목 번호 j."""
    df = E._to_table(rec, m)
    df["j"] = pd.Index(m.codes).get_indexer(df["code"])
    return df


# ================================================================ 학습 줄 (매매 × 날)
def trade_block(ctx: SellCtx, mk: str, b, j, buy, start: int, width: int, peak0) -> tuple:
    """매매 N개의 매수일 + start ~ + start+width−1 날 특징.
    돌려주는 값: X (N, width, 특징 수) float32, ok (N, width) 쓸 수 있는 날, d (N, width) 날짜 번호, 끝의 최고 종가."""
    m = ctx.m
    n = m.n
    b = np.asarray(b, np.int64)
    j = np.asarray(j, np.int64)
    buy = np.asarray(buy, float)
    d = b[:, None] + start + np.arange(width)[None, :]
    lim = np.minimum(m.last[j], n - 1)[:, None]
    inside = d <= lim
    dc = np.minimum(d, n - 1)
    jj = j[:, None]
    c = ctx.close_ff[dc, jj]
    cm = np.where(inside, c, -np.inf)
    peak = np.maximum(np.asarray(peak0, float)[:, None], np.maximum.accumulate(cm, axis=1))
    X = np.empty(d.shape + (len(FEATURES),), dtype=np.float32)
    for q, k in enumerate(DAY_FEATS):
        X[..., q] = ctx.feat[k][dc, jj]
    mf = ctx.mkt[mk][dc]
    X[..., len(DAY_FEATS):len(DAY_FEATS) + len(MKT_FEATS)] = mf
    base = len(DAY_FEATS) + len(MKT_FEATS)
    with np.errstate(invalid="ignore", divide="ignore"):
        X[..., base] = c / buy[:, None] - 1
        X[..., base + 1] = c / peak - 1
    X[..., base + 2] = d - b[:, None]
    ok = inside & m.has[dc, jj] & np.isfinite(X).all(axis=-1)
    return X, ok, d, peak[:, -1]


def build_panel(ctx: SellCtx, mk: str) -> dict:
    """학습 줄: 그 시장 매매(2015 이후 신호, 실제로 산 것)의 매수일부터 120거래일. 라벨·라벨 끝 날·가중치 바탕."""
    m = ctx.m
    T = ctx.pool[mk]
    if not len(T["b"]):
        return {"X": np.zeros((0, len(FEATURES)), np.float32), "j": np.zeros(0, np.int64), "d": np.zeros(0, np.int64),
                "trade": np.zeros(0, np.int64), "y": np.zeros(0, int), "diff": np.zeros(0), "label_end": np.zeros(0, np.int64)}
    X, ok, d, _ = trade_block(ctx, mk, T["b"], T["j"], T["buy"], 0, PANEL_DAYS, T["buy"])
    tr, off = np.nonzero(ok)
    X, d = X[tr, off], d[tr, off]
    j = T["j"][tr]
    p0 = settle(m, j, d)
    p20 = settle(m, j, d + LOOK)
    keep = (p0["status"] != E.OPEN_END) & (p20["status"] != E.OPEN_END)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = p20["sell_px"] / p0["sell_px"] - 1
    keep &= np.isfinite(diff)
    return {"X": X[keep], "j": j[keep], "d": d[keep], "trade": tr[keep], "y": (diff[keep] < -MARGIN).astype(int),
            "diff": diff[keep], "label_end": p20["s_idx"][keep].astype(np.int64)}


def split_rows(ctx: SellCtx, panel: dict, year: int) -> dict:
    """시험 해 Y 의 줄 나누기. pool = 라벨 끝 + GAP < Y 첫날, fit = 라벨 끝 + GAP < 검증 첫날, val = pool 중 검증 첫날부터."""
    sY = ctx.ystart[year]
    s0 = ctx.ystart[ctx.first_year]
    val_start = sY - int(round(VAL_FRAC * (sY - s0)))
    le = panel["label_end"]
    pool = le + GAP < sY
    fit = le + GAP < val_start
    val = pool & (panel["d"] >= val_start)
    return {"sY": sY, "val_start": val_start, "pool": pool, "fit": fit, "val": val}


def weights(panel: dict, rows: np.ndarray, balance: bool = True) -> np.ndarray:
    """가중치 = |차이| ÷ (그 매매의 줄 수, 이 줄 묶음 안에서). 평균 1. balance 면 판다·든다 두 쪽 합을 같게."""
    tr = panel["trade"][rows]
    _, inv, cnt = np.unique(tr, return_inverse=True, return_counts=True)
    w = np.abs(panel["diff"][rows]) / cnt[inv]
    if balance:
        y = panel["y"][rows]
        for c in (0, 1):                                      # 두 쪽 합을 각각 1 로 → 아래에서 평균 1 로
            s = w[y == c].sum()
            if s > 0:
                w[y == c] /= s
    tot = w.sum()
    return w * (len(w) / tot) if tot > 0 else np.ones(len(w))


def rows_of(panel: dict, mask: np.ndarray) -> dict:
    return {"X": panel["X"][mask], "j": panel["j"][mask], "d": panel["d"][mask]}


# ================================================================ 모델
GRIDS = {
    "logistic": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "random_forest": [{"max_depth": a, "min_samples_leaf": b} for a in (6, 10) for b in (500, 2000)],
    "lightgbm": [{"num_leaves": a, "learning_rate": b} for a, b in ((15, .05), (31, .05), (63, .05), (31, .02), (31, .1))],
    "xgboost": [{"max_depth": a, "learning_rate": b} for a, b in ((3, .05), (4, .05), (6, .05), (4, .02), (4, .1))],
    "catboost": [{"depth": a, "learning_rate": b} for a, b in ((4, .05), (6, .05), (8, .05), (6, .02), (6, .1))],
    "lstm": [{"units": u} for u in (16, 32, 64)],
    "cnn1d": [{"filters": f} for f in (16, 32, 64)],
}
assert all(len(GRIDS[k]) == v for k, v in C.ML_GRID_SIZES.items()), "계획서 설정 수와 다르다"
DL_KINDS = ("lstm", "cnn1d")
CUSTOM: dict = {}


def add_model(name: str, make, grid: list[dict]):
    """팀원 모델 넣기. make(params) → fit(X, y, sample_weight=w) 와 predict_proba(X) 가 있는 물건 (sklearn 모양).
    11/27 전에 정해 둔 모델·설정만. 설정 하나가 시도 하나로 세어진다."""
    if name in GRIDS:
        raise ValueError(f"{name} 은 이미 있는 이름")
    CUSTOM[name] = make
    GRIDS[name] = list(grid)


def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras
    return keras


class Model:
    """모델 하나 (설정 하나). fit(R, y, w, Rv, yv, wv) → prob(R). R = {"X": 특징, "j": 종목 번호, "d": 날짜 번호}."""

    def __init__(self, kind: str, params: dict, seq: np.ndarray | None = None, seed: int = C.SEED,
                 dl_max_rows: int = 200_000, epochs: int = 10, batch: int = 1024):
        if kind not in GRIDS:
            raise ValueError(f"모델 이름은 {list(GRIDS)} 중 하나: {kind}")
        self.kind, self.params, self.seed = kind, dict(params), seed
        self.seq_src, self.dl_max_rows, self.epochs, self.batch = seq, dl_max_rows, epochs, batch
        self.const = None
        self.best_epoch = None
        self.est = None

    # ---------------------------------------------------------------- 표 모델
    def _make(self):
        k, p, s = self.kind, self.params, self.seed
        if k in CUSTOM:
            return CUSTOM[k](p)
        if k == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=1000))
        if k == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s, **p)
        if k == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(n_estimators=300, random_state=s, verbose=-1, n_jobs=-1, **p)
        if k == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(n_estimators=300, tree_method="hist", random_state=s, n_jobs=-1,
                                     eval_metric="logloss", **p)
        if k == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=300, random_seed=s, verbose=0, allow_writing_files=False, **p)
        raise ValueError(k)

    def fit(self, R, y, w, Rv=None, yv=None, wv=None, epochs: int | None = None):
        y = np.asarray(y, int)
        if len(y) == 0 or len(np.unique(y)) < 2:              # 한쪽 라벨만 있으면 그 비율을 그대로 준다
            self.const = float(y.mean()) if len(y) else 0.0
            return self
        self.const = None
        if self.kind in DL_KINDS:
            return self._fit_dl(R, y, w, Rv, yv, wv, epochs)
        self.est = self._make()
        if self.kind == "logistic":
            self.est.fit(R["X"], y, logisticregression__sample_weight=w)
        else:
            self.est.fit(R["X"], y, sample_weight=w)
        return self

    def prob(self, R) -> np.ndarray:
        if self.const is not None:
            return np.full(len(R["X"]), self.const)
        if self.kind in DL_KINDS:
            return self._prob_dl(R)
        return np.asarray(self.est.predict_proba(R["X"]))[:, 1]

    def importance(self) -> pd.Series | None:
        if self.const is not None or self.est is None or self.kind in DL_KINDS:
            return None
        e = self.est
        if self.kind == "logistic":
            v = np.abs(e[-1].coef_[0])
        elif self.kind == "catboost":
            v = np.asarray(e.get_feature_importance(), float)
        elif hasattr(e, "feature_importances_"):
            v = np.asarray(e.feature_importances_, float)
        else:
            return None
        return pd.Series(v, index=FEATURES[:len(v)])

    # ---------------------------------------------------------------- DL (LSTM·1D CNN, Keras 3)
    def _seq(self, R) -> np.ndarray:
        S = self.seq_src
        L = C.DL_INPUT_DAYS
        d, j = np.asarray(R["d"]), np.asarray(R["j"])
        idx = d[:, None] - (L - 1) + np.arange(L)[None, :]
        out = S[np.clip(idx, 0, None), j[:, None], :]
        out = (out - self.mu_s) / self.sd_s
        out[idx < 0] = 0.0
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _tab(self, R) -> np.ndarray:
        return np.nan_to_num((R["X"] - self.mu_t) / self.sd_t, nan=0.0).astype(np.float32)

    def _dataset(self, R, y, w, shuffle):
        keras = _keras()
        owner = self

        class Batches(keras.utils.PyDataset):
            def __init__(self):
                super().__init__()
                self.order = np.arange(len(y))
                self.rng = np.random.default_rng(owner.seed)
                if shuffle:
                    self.rng.shuffle(self.order)

            def __len__(self):
                return int(np.ceil(len(y) / owner.batch))

            def __getitem__(self, i):
                q = self.order[i * owner.batch:(i + 1) * owner.batch]
                r = {"X": R["X"][q], "j": R["j"][q], "d": R["d"][q]}
                return ({"seq": owner._seq(r), "tab": owner._tab(r)}, y[q].astype(np.float32),
                        w[q].astype(np.float32))

            def on_epoch_end(self):
                if shuffle:
                    self.rng.shuffle(self.order)

        return Batches()

    def _fit_dl(self, R, y, w, Rv, yv, wv, epochs):
        keras = _keras()
        keras.utils.set_random_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        if len(y) > self.dl_max_rows:                         # 줄이 너무 많으면 같은 seed 로 일부만
            q = np.sort(rng.choice(len(y), self.dl_max_rows, replace=False))
            R = {k: v[q] for k, v in R.items()}
            y, w = y[q], w[q]
        self.mu_t = np.nanmean(R["X"], axis=0)
        self.sd_t = np.nanstd(R["X"], axis=0) + 1e-6
        cur = self.seq_src[R["d"], R["j"], :]
        self.mu_s = np.nanmean(cur, axis=0)
        self.sd_s = np.nanstd(cur, axis=0) + 1e-6
        L, K, F = C.DL_INPUT_DAYS, self.seq_src.shape[-1], R["X"].shape[1]
        seq = keras.Input((L, K), name="seq")
        tab = keras.Input((F,), name="tab")
        if self.kind == "lstm":
            h = keras.layers.LSTM(self.params["units"])(seq)
        else:
            h = keras.layers.Conv1D(self.params["filters"], 5, activation="relu")(seq)
            h = keras.layers.GlobalAveragePooling1D()(h)
        z = keras.layers.Concatenate()([h, tab])
        z = keras.layers.Dense(32, activation="relu")(z)
        out = keras.layers.Dense(1, activation="sigmoid")(z)
        self.est = keras.Model(inputs={"seq": seq, "tab": tab}, outputs=out)
        self.est.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
        train = self._dataset(R, y, np.asarray(w, float), shuffle=True)
        if epochs is None and Rv is not None and len(yv) and len(np.unique(yv)) == 2:
            val = self._dataset(Rv, np.asarray(yv, int), np.asarray(wv, float), shuffle=False)
            stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
            h = self.est.fit(train, validation_data=val, epochs=self.epochs, callbacks=[stop], verbose=0)
            self.best_epoch = int(np.argmin(h.history["val_loss"])) + 1
        else:
            self.est.fit(train, epochs=epochs or self.epochs, verbose=0)
            self.best_epoch = epochs or self.epochs
        return self

    def _prob_dl(self, R) -> np.ndarray:
        out, B = [], 4096                                     # 묶음 크기를 늘 같게 (모자라면 앞 줄로 채우고 버림)
        for i in range(0, len(R["X"]), B):
            r = {k: v[i:i + B] for k, v in R.items()}
            k = len(r["X"])
            if k < B:
                r = {q: np.concatenate([v, np.repeat(v[:1], B - k, axis=0)]) for q, v in r.items()}
            p = np.asarray(self.est.predict_on_batch({"seq": self._seq(r), "tab": self._tab(r)})).ravel()
            out.append(p[:k])
        return np.concatenate(out) if out else np.zeros(0)


# ================================================================ 기준 확률 고르기 (학습 구간 마지막 20%)
def first_hit(p: np.ndarray, ok: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """줄마다 처음으로 확률 ≥ 기준인 칸. (있는가, 몇 번째 칸)."""
    hit = ok & (np.nan_to_num(p, nan=-1.0) >= thr)
    return hit.any(axis=1), hit.argmax(axis=1)


def pick_threshold(ctx: SellCtx, mk: str, model: Model, sp: dict) -> tuple[float, dict]:
    """검증 구간(학습 구간 마지막 20%)에 산 매매를 처음 학습한 모델로 팔아 본다. 평균 순수익(−0.5%)이 가장 큰 기준.
    Y년 첫날 전 마지막 날(sY−1) 뒤에 팔게 되면 그날 종가로 평가. 120일 안에 판단이 없으면 b+120(또는 sY−1) 종가."""
    m = ctx.m
    T = ctx.pool[mk]
    sY, v0 = sp["sY"], sp["val_start"]
    sel = (T["b"] >= v0) & (T["b"] <= sY - 2)
    res = {thr: np.nan for thr in THRESHOLDS}
    if not sel.any():
        return THRESHOLDS[0], {"검증매매수": 0, **{f"검증평균_{t:.2f}": v for t, v in res.items()}}
    b, j, buy = T["b"][sel], T["j"][sel], T["buy"][sel]
    X, ok, d, _ = trade_block(ctx, mk, b, j, buy, 0, PANEL_DAYS, buy)
    ok &= d <= sY - 2
    p = np.full(ok.shape, np.nan)
    if ok.any():
        p[ok] = model.prob({"X": X[ok], "j": np.broadcast_to(j[:, None], ok.shape)[ok], "d": d[ok]})
    end_px = ctx.close_ff[np.minimum(b + PANEL_DAYS, sY - 1), j]
    for thr in THRESHOLDS:
        has, at = first_hit(p, ok, thr)
        px = end_px.copy()
        if has.any():
            dd = d[np.arange(len(b)), at][has]
            st = settle(m, j[has], dd)
            late = (st["status"] == E.OPEN_END) | (st["s_idx"] > sY - 1)
            px_h = np.where(late, ctx.close_ff[sY - 1, j[has]], st["sell_px"])
            px[has] = px_h
        res[thr] = float(np.mean(px / buy - 1 - C.COST_SELECT))
    best = max(THRESHOLDS, key=lambda t: (res[t], -THRESHOLDS.index(t)))
    return best, {"검증매매수": int(sel.sum()), **{f"검증평균_{t:.2f}": v for t, v in res.items()}}


# ================================================================ 해마다 학습 → 시험 매도
def fit_year(ctx: SellCtx, mk: str, panel: dict, year: int, kind: str, params: dict,
             refit: bool = True, balance: bool = True, model_kw: dict | None = None):
    """시험 해 하나: 앞 80% 로 학습 → 마지막 20% 로 기준 고르기 → (refit) 전부로 다시 학습."""
    from sklearn.metrics import roc_auc_score
    kw = dict(model_kw or {})
    seq = ctx.seq if kind in DL_KINDS else None
    sp = split_rows(ctx, panel, year)
    fit, val, pool = sp["fit"], sp["val"], sp["pool"]
    wf = weights(panel, np.nonzero(fit)[0], balance)
    wv = weights(panel, np.nonzero(val)[0], balance) if val.any() else np.zeros(0)
    m1 = Model(kind, params, seq=seq, **kw).fit(rows_of(panel, fit), panel["y"][fit], wf,
                                               rows_of(panel, val), panel["y"][val], wv)
    thr, info = pick_threshold(ctx, mk, m1, sp)
    auc = np.nan
    if val.any() and len(np.unique(panel["y"][val])) == 2:
        auc = float(roc_auc_score(panel["y"][val], m1.prob(rows_of(panel, val))))
    model = m1
    if refit and pool.sum() > fit.sum():
        wp = weights(panel, np.nonzero(pool)[0], balance)
        ep = m1.best_epoch if kind in DL_KINDS else None
        model = Model(kind, params, seq=seq, **kw).fit(rows_of(panel, pool), panel["y"][pool], wp, epochs=ep)
    row = {"시장": mk, "시험해": year, "학습줄_앞80%": int(fit.sum()), "검증줄": int(val.sum()),
           "학습줄_전부": int(pool.sum()), "판다비율_전부": float(panel["y"][pool].mean()) if pool.any() else np.nan,
           "고른기준": thr, "검증AUC": auc, "DL에폭": m1.best_epoch, **info,
           "검증첫날": ctx.dates[sp["val_start"]].date(), "시험첫날": ctx.dates[sp["sY"]].date()}
    return model, thr, row


def ml_decide(ctx: SellCtx, mk: str, models: dict, thrs: dict) -> np.ndarray:
    """시험 매매(2016 이후 신호)마다 판단한 날 d. 그날 해의 모델·기준을 쓴다. 판단이 없으면 m.n (끝까지 보유)."""
    m = ctx.m
    T = ctx.test[mk]
    N = len(T["t"])
    dec = np.full(N, m.n, dtype=np.int64)
    active = np.nonzero(T["ok"])[0]
    peak = np.where(T["ok"], T["buy_px"], np.nan)
    lim = np.minimum(m.last[T["j"]], m.n - 1)
    off = 0
    while active.size:
        a = active
        X, ok, d, pk = trade_block(ctx, mk, T["b"][a], T["j"][a], T["buy_px"][a], off, CHUNK, peak[a])
        p = np.full(ok.shape, np.nan)
        yr = ctx.years[np.minimum(d, m.n - 1)]
        jj = np.broadcast_to(T["j"][a][:, None], ok.shape)
        hit = np.zeros(ok.shape, bool)
        for y, mdl in models.items():
            sel = ok & (yr == y)
            if sel.any():
                p[sel] = mdl.prob({"X": X[sel], "j": jj[sel], "d": d[sel]})
                hit |= sel & (p >= thrs[y])
        has = hit.any(axis=1)
        at = hit.argmax(axis=1)
        dec[a[has]] = d[np.arange(len(a)), at][has]
        peak[a] = pk
        more = ~has & (T["b"][a] + off + CHUNK <= lim[a])
        active = a[more]
        off += CHUNK
    return dec


def walk_forward(ctx: SellCtx, mk: str, kind: str, params: dict, refit: bool = True, balance: bool = True,
                 model_kw: dict | None = None, panel: dict | None = None, log=print):
    """시장 하나 · 설정 하나: 해마다 학습 → 시험 매매 표 (엔진 모양) · 해마다 정보 · 특징 중요도."""
    panel = build_panel(ctx, mk) if panel is None else panel
    models, thrs, rows, imps = {}, {}, [], []
    for y in ctx.test_years:
        t0 = time.time()
        mdl, thr, row = fit_year(ctx, mk, panel, y, kind, params, refit, balance, model_kw)
        models[y], thrs[y] = mdl, thr
        rows.append(row)
        imp = mdl.importance()
        if imp is not None:
            imps.append(pd.DataFrame({"시장": mk, "시험해": y, "특징": imp.index, "중요도": imp.to_numpy()}))
        log(f"    {mk} {y}: 학습 {row['학습줄_전부']:,}줄 · 기준 {thr:.2f} · 검증AUC {row['검증AUC']:.3f} "
            f"({time.time() - t0:.0f}초)")
    dec = ml_decide(ctx, mk, models, thrs)
    T = ctx.test[mk]
    table = to_table(trade_record(ctx.m, T["t"], T["j"], dec), ctx.m)
    return table, pd.DataFrame(rows), (pd.concat(imps, ignore_index=True) if imps else pd.DataFrame())


# ================================================================ 포트폴리오 (engine.run_portfolio 와 같은 규칙, 후보 매매만 미리 받음)
def cell_candidates(ctx: SellCtx, table: pd.DataFrame, mk: str, cell: str, from_year: int) -> pd.DataFrame:
    """칸 하나의 후보: 그 칸 신호(from_year 부터) × 시장 매매 표. 매수 순서 값·20일 평균 거래대금 붙임."""
    s = ctx.sig[(ctx.sig["시장"] == mk) & (ctx.sig["구분"] == cell) & (ctx.years[ctx.sig["t"].to_numpy()] >= from_year)]
    key = s[["t", "j", "매수순서값"]].rename(columns={"t": "t_idx", "매수순서값": "n_sig"})
    cand = table.merge(key, on=["t_idx", "j"], how="inner", validate="one_to_one")
    if len(cand) != len(key):
        raise SystemExit(f"✗ {mk} {cell}: 매매 표에 없는 신호 {len(key) - len(cand)}개")
    cand["v20"] = ctx.m.value20[cand["t_idx"].to_numpy(int), cand["j"].to_numpy(int)]
    return cand


def run_portfolio_cand(cand: pd.DataFrame, m: E._Market, close_ff: np.ndarray, cost: float = C.COST_SELECT,
                       capital: float = C.INIT_CAPITAL, max_positions: int = C.MAX_POSITIONS,
                       weight: float = C.WEIGHT_PER_POSITION):
    """engine.run_portfolio 를 그대로 옮겨 적고 후보 매매만 미리 받는다. 돌려주는 값도 같다 (daily, log)."""
    cand = cand.sort_values(["b_idx", "n_sig", "v20", "code"], ascending=[True, False, False, True])
    by_day = {b: g for b, g in cand.groupby("b_idx", sort=False)}
    cash = float(capital)
    equity_prev = float(capital)
    held: dict[str, dict] = {}
    log, daily = [], []
    for i in range(m.n):
        proceeds = 0.0
        held_at_T = set(held)
        # 1) 시가 매도
        for code in [c for c, p in held.items() if p["s_idx"] == i and p["sell_at_open"]]:
            proceeds += E._close_position(held.pop(code), cost, log)
        # 2) 시가 매수 (t = i−1 의 신호)
        budget = cash
        free = max_positions - len(held)
        if i in by_day and free > 0:
            picks = by_day[i][~by_day[i]["code"].isin(held_at_T)].head(free)
            for r in picks.itertuples(index=False):
                if r.status in (E.GAP_CANCEL, E.NO_BUY):
                    log.append(dict(r._asdict(), shares=0.0, notional=0.0, pnl=0.0))
                    continue
                size = min(weight * equity_prev, budget)
                if size <= 0:
                    continue
                budget -= size
                held[r.code] = dict(r._asdict(), shares=size / r.buy_px, notional=size)
        cash = budget
        # 3) 장중·종가 매도 (상장폐지 정산은 여기)
        for code in [c for c, p in held.items() if p["s_idx"] == i]:
            proceeds += E._close_position(held.pop(code), cost, log)
        # 4) 판 돈은 장 마감 뒤 현금으로
        cash += proceeds
        value = sum(p["shares"] * close_ff[i, p["j"]] for p in held.values())
        equity_prev = cash + value
        daily.append(dict(date=m.dates[i], cash=cash, equity=equity_prev, n_pos=len(held)))
    for p in held.values():
        log.append(dict(p, pnl=np.nan))
    return pd.DataFrame(daily).set_index("date"), pd.DataFrame(log)


def yearly_out(eq: pd.Series, years) -> dict:
    """stage4.yearly_out 와 같은 계산 (해 목록만 받음). eq = 첫 시험 해부터의 평가금액."""
    from indexbeater import metrics as M
    out = {}
    prev = float(C.INIT_CAPITAL)
    for y in years:
        e = eq[eq.index.year == y]
        out[f"포트_수익_{y}"] = float(e.iloc[-1] / prev - 1) if len(e) else np.nan
        prev = float(e.iloc[-1]) if len(e) else prev
    out["포트_MDD"] = M.max_drawdown(pd.concat([pd.Series([float(C.INIT_CAPITAL)]), eq.reset_index(drop=True)]))
    out["포트_마지막평가금액"] = float(eq.iloc[-1])
    return out


def index_yearly() -> pd.DataFrame:
    from indexbeater import benchmark as B
    return B.market_yearly(B.load_market())


def score_cells(ctx: SellCtx, table: pd.DataFrame, mk: str, my: pd.DataFrame | None, head: dict,
                from_year: int | None = None, costs=None):
    """시장 하나 매매 표 → 그 시장 3칸 × 비용 2개 포트폴리오 점수 줄 + 날마다 평가금액."""
    from_year = ctx.test_years[0] if from_year is None else from_year
    years = [y for y in sorted(set(ctx.years)) if y >= from_year]
    costs = COSTS if costs is None else costs
    rows, daily = [], []
    cells = list(dict.fromkeys(ctx.sig.loc[ctx.sig["시장"] == mk, "구분"]))
    for cell in cells:
        cand = cell_candidates(ctx, table, mk, cell, from_year)
        strat = ctx.sig.loc[(ctx.sig["시장"] == mk) & (ctx.sig["구분"] == cell), "전략"].iloc[0]
        for cname, cost in costs.items():
            dly, lg = run_portfolio_cand(cand, ctx.m, ctx.close_ff, cost=cost)
            eq = dly["equity"]
            eq = eq[eq.index.year >= from_year]
            r = {**head, "시장": mk, "구분": cell, "전략": strat, "비용": cname, **yearly_out(eq, years)}
            bought = lg[lg["shares"] > 0] if len(lg) else lg
            sold = bought[bought["s_idx"] >= 0] if len(bought) else bought
            r["산매매수"] = int(len(bought))
            r["평균보유거래일"] = float((sold["s_idx"] - sold["b_idx"]).mean()) if len(sold) else np.nan
            r["가장긴보유거래일"] = int((sold["s_idx"] - sold["b_idx"]).max()) if len(sold) else 0
            r["끝까지보유수"] = int(lg["pnl"].isna().sum()) if len(lg) else 0
            if my is not None:
                ex = []
                for y in years:
                    iy = my[(my["market"] == mk) & (my["year"] == y)]["수익률"]
                    iv = float(iy.iloc[0]) if len(iy) else np.nan
                    r[f"지수_수익_{y}"] = iv
                    r[f"초과_{y}"] = r[f"포트_수익_{y}"] - iv
                    ex.append(r[f"초과_{y}"])
                ex = np.asarray(ex, float)
                r["초과_중앙값"] = float(np.nanmedian(ex)) if np.isfinite(ex).any() else np.nan
                r["초과음수해수"] = int((ex < 0).sum())
            rows.append(r)
            daily.append(pd.DataFrame({**{k: v for k, v in head.items()}, "date": eq.index, "시장": mk, "구분": cell,
                                       "비용": cname, "equity": eq.to_numpy(), "n_pos": dly.loc[eq.index, "n_pos"].to_numpy()}))
    return rows, daily


def rank_table(score: pd.DataFrame) -> pd.DataFrame:
    """0.5% 줄 순위: 초과수익 중앙값 큰 순 → 초과 음수 해 적은 순 (칸마다)."""
    s = score[score["비용"] == "0.5%"].copy()
    s = s.sort_values(["시장", "구분", "초과_중앙값", "초과음수해수"], ascending=[True, True, False, True])
    s["칸안순위"] = s.groupby(["시장", "구분"]).cumcount() + 1
    return s


# ================================================================ 실행
def run_model(ctx: SellCtx, kind: str, settings=None, refit=True, balance=True, model_kw=None, log=print):
    """모델 하나의 설정들 → 점수 줄·날마다 평가금액·해마다 정보·특징 중요도·시험 매매 표."""
    my = index_yearly()
    settings = range(len(GRIDS[kind])) if settings is None else settings
    panels = {mk: build_panel(ctx, mk) for mk in ctx.markets()}
    rows, daily, info, imps, trades = [], [], [], [], []
    for q in settings:
        params = GRIDS[kind][q]
        head = {"모델": kind, "설정번호": q, "설정": str(params)}
        log(f"  {kind} 설정 {q} {params}")
        for mk in ctx.markets():
            table, yr, imp = walk_forward(ctx, mk, kind, params, refit, balance, model_kw, panels[mk], log)
            r, d = score_cells(ctx, table, mk, my, head)
            rows += r
            daily += d
            info.append(yr.assign(**head))
            if len(imp):
                imps.append(imp.assign(**head))
            trades.append(table.assign(**head))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return {"score": pd.DataFrame(rows), "daily": cat(daily), "years": cat(info), "importance": cat(imps),
            "trades": cat(trades)}


def run_baseline(ctx: SellCtx, log=print):
    """비교용 고정 H5·H10·H20 (엔진 S3 와 같은 매도). 같은 6칸·같은 기간."""
    my = index_yearly()
    rows, daily = [], []
    for k in FIXED_H:
        head = {"모델": "기준_고정H", "설정번호": k, "설정": f"H{k}"}
        for mk in ctx.markets():
            T = ctx.test[mk]
            table = to_table(fixed_record(ctx.m, T["t"], T["j"], k), ctx.m)
            r, d = score_cells(ctx, table, mk, my, head)
            rows += r
            daily += d
        log(f"  고정 H{k} 끝")
    return {"score": pd.DataFrame(rows), "daily": pd.concat(daily, ignore_index=True)}


# ================================================================ 점검 (실제 데이터에서 이것만 돌려 봄)
def self_check(ctx: SellCtx, log=print, tol: float = 1e-9) -> dict:
    """1) CSV 매수가·갭 = 내 계산  2) 고정 H5·H10·H20 매매 표 = 엔진 simulate_trades
    3) H5·0.5%·2015 신호부터 포트폴리오 = team_compare 저장 평가금액(PDF) 6칸  4) 학습 줄 수·판다 비율."""
    from indexbeater import stage4 as S4
    from indexbeater import team_compare as T
    from indexbeater.exits import ExitRule
    m = ctx.m
    out = {}
    s = ctx.sig
    buy = buy_arrays(m, s["t"], s["j"])
    rows = []
    if "매수가_수정시가" in s:
        csv_px = s["매수가_수정시가"].to_numpy(float)
        same_px = np.array_equal(np.where(buy["ok"], buy["buy_px"], np.nan), np.where(buy["ok"], csv_px, np.nan),
                                 equal_nan=True)
        same_gap = np.array_equal(buy["status"] == E.GAP_CANCEL, s["갭취소"].astype(bool).to_numpy())
        rows.append({"점검": "CSV 매수가 = T+1 수정 시가", "통과": bool(same_px)})
        rows.append({"점검": "CSV 갭취소 = 내 계산", "통과": bool(same_gap)})
    for k in FIXED_H:
        for mk in ctx.markets():
            part = s[s["시장"] == mk]
            eng = E.simulate_trades(entry_table(m, part["t"], part["j"]), ctx.data, ExitRule(hold=k), dedupe=False, _m=m)
            eng = eng[eng["market"] == mk].reset_index(drop=True)
            tj = np.unique(np.column_stack([part["t"], part["j"]]), axis=0)
            mine = to_table(fixed_record(m, tj[:, 0], tj[:, 1], k), m).drop(columns="j")
            ok = len(eng) == len(mine)
            if ok:
                for c in ("t_idx", "b_idx", "s_idx", "status", "sell_at_open", "code", "delay", "reason"):
                    ok &= bool((eng[c].fillna("-").astype(str) == mine[c].fillna("-").astype(str)).all())
                for c in ("buy_px", "sell_px"):
                    ok &= np.array_equal(eng[c].to_numpy(float), mine[c].to_numpy(float), equal_nan=True)
            rows.append({"점검": f"고정 H{k} 매매 표 = 엔진 ({mk}, {len(mine):,}건)", "통과": bool(ok)})
    saved = pd.read_csv(T.OUT_DAILY, encoding="utf-8-sig", parse_dates=["date"])
    saved = saved[(saved["H"] == 5) & (saved["비용"] == S4.COST_ROW)]
    for mk in ctx.markets():
        T5 = to_table(fixed_record(m, *np.unique(np.column_stack([s.loc[s["시장"] == mk, "t"], s.loc[s["시장"] == mk, "j"]]),
                                                    axis=0).T, 5), m)
        for cell in dict.fromkeys(s.loc[s["시장"] == mk, "구분"]):
            cand = cell_candidates(ctx, T5, mk, cell, ctx.first_year)
            dly, _ = run_portfolio_cand(cand, m, ctx.close_ff, cost=C.COST_SELECT)
            eq = dly["equity"]
            eq = eq[eq.index.year >= ctx.first_year]
            ref = saved[(saved["시장"] == mk) & (saved["구분"] == cell)].set_index("date")["equity"]
            same = len(ref) == len(eq) and bool((ref.index == eq.index).all())
            diff = float(np.abs(eq.to_numpy() / ref.to_numpy() - 1).max()) if same else np.inf
            rows.append({"점검": f"H5·0.5% 포트폴리오 = PDF ({mk} {cell}, 차이 {diff:.1e})", "통과": diff <= tol})
    out["점검"] = pd.DataFrame(rows)
    for mk in ctx.markets():
        P = build_panel(ctx, mk)
        yrs = ctx.years[P["d"]]
        by_year, by_test = [], []
        for y in sorted(set(yrs)):
            sel = yrs == y
            by_year.append({"시장": mk, "판단날의해": int(y), "학습줄": int(sel.sum()),
                        "매매수": int(len(np.unique(P["trade"][sel]))), "판다비율": float(P["y"][sel].mean())})
        for y in ctx.test_years:
            sp = split_rows(ctx, P, y)
            by_test.append({"시장": mk, "시험해": y, "앞80%줄": int(sp["fit"].sum()), "검증줄": int(sp["val"].sum()),
                        "전부줄": int(sp["pool"].sum()), "검증첫날": ctx.dates[sp["val_start"]].date(),
                        "가장늦은라벨끝": ctx.dates[int(P["label_end"][sp["pool"]].max())].date() if sp["pool"].any() else None})
        out[f"학습줄_판단날의해_{mk}"] = pd.DataFrame(by_year)
        out[f"학습줄_시험해마다_{mk}"] = pd.DataFrame(by_test)
    for _, r in out["점검"].iterrows():
        log(f"  {'✓' if r['통과'] else '✗'} {r['점검']}")
    return out


def load_ctx(path=SIG_CSV, log=print) -> SellCtx:
    from indexbeater import data_io
    t0 = time.time()
    data = data_io.load_wide()                    # 기본값 end = 2019-12-31 (시험 구간 잠금)
    ctx = SellCtx(data, read_signals(path), log=log)
    log(f"  데이터 {len(ctx.dates):,}일 × {len(ctx.m.codes):,}종목 ({time.time() - t0:.0f}초)")
    return ctx


def main(argv=None):
    import argparse
    from indexbeater import benchmark as B
    ap = argparse.ArgumentParser(prog="python -m indexbeater sellml", description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="옮겨 적은 계산 = 엔진·PDF 점검 + 학습 줄 수 (저장 안 함)")
    ap.add_argument("--list", action="store_true", help="모델·설정·입력 값 목록")
    ap.add_argument("--baseline", action="store_true", help="고정 H5·H10·H20 → results/sell_ml_baseline_*.csv")
    ap.add_argument("--run", metavar="모델", help=f"모델 하나의 설정 전부: {', '.join(GRIDS)}")
    ap.add_argument("--setting", type=int, nargs="*", help="설정 번호만 (0부터)")
    ap.add_argument("--signals", default=str(SIG_CSV), help="TOP3 매수 신호 CSV")
    ap.add_argument("--overwrite", action="store_true", help="결과 파일 덮어쓰기 (팀장 승인 뒤)")
    a = ap.parse_args(argv)
    if a.list:
        for k, g in GRIDS.items():
            print(f"  {k:14s} {len(g)}개: {g}")
        print(f"  입력 {len(FEATURES)}개: {FEATURES}")
        return 0
    if not (a.check or a.baseline or a.run):
        ap.print_help()
        return 0
    t0 = time.time()
    ctx = load_ctx(a.signals)
    if a.check:
        res = self_check(ctx)
        for k, v in res.items():
            print(f"\n[{k}]\n{v.to_string(index=False)}")
        if not res["점검"]["통과"].all():
            print("\n✗ 점검 실패 → 학습을 돌리지 말고 팀장에게 알린다")
            return 1
    if a.baseline:
        out = run_baseline(ctx)
        for part in ("score", "daily"):
            print(f"  {out_path('baseline', part).name}: {B._save(out[part], out_path('baseline', part), a.overwrite)}")
    if a.run:
        if a.run not in GRIDS:
            raise SystemExit(f"모델 이름은 {list(GRIDS)} 중 하나")
        out = run_model(ctx, a.run, a.setting)
        out["rank"] = rank_table(out["score"])
        for part in ("score", "rank", "years", "importance", "daily", "trades"):
            if len(out[part]):
                print(f"  {out_path(a.run, part).name}: {B._save(out[part], out_path(a.run, part), a.overwrite)}")
        show = ["모델", "설정번호", "시장", "구분", "초과_중앙값", "초과음수해수", "포트_MDD", "산매매수", "평균보유거래일"]
        print(out["rank"][show].to_string(index=False))
    print(f"\n끝 ({time.time() - t0:.0f}초)")
    return 0
