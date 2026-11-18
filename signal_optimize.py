"""
일봉 매수신호 파라미터 최적화 (1차 베이스라인)

규칙
- 신호는 t일 종가 기준으로 판단 -> t+1일 시가에 매수 -> t+H일 종가에 매도 (미래 정보 사용 금지)
- 가격은 수정주가 사용 (Adj 계수 = AdjClose / Close 를 시가에도 적용)
- 거래비용 왕복 0.25% (수수료 + 매도세 근사)
- 유동성 필터: 20일 평균 거래대금 >= 10억, 주가 >= 1,000원, 거래량 0인 날 제외
- 학습(Train) 구간에서 그리드서치로 최적 파라미터 선택 -> 검증(Test) 구간에서 그대로 성능 확인
- 비교 기준(Base): 같은 조건의 '아무 날이나 산' 평균 수익률. Edge = 신호 평균 - Base

실행:  python signal_optimize.py
결과:  results/grid_train.csv, results/best_train_vs_test.csv
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA, RES = ROOT / "data", ROOT / "results"

TRAIN_END = "2020-12-31"   # 학습: 시작 ~ 2020, 검증: 2021 ~ 현재
COST = 0.0025
MIN_TRADES = 300
HOLDS = [5, 10, 20]


# ---------------- 데이터 ----------------
def load_wide():
    df = pd.read_parquet(DATA / "daily_ohlcv.parquet")
    df = df[df["Volume"] > 0]
    w = {c: df.pivot(index="Date", columns="Code", values=c).astype("float32")
         for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]}
    adj = w["AdjClose"] / w["Close"]
    w["AO"] = w["Open"] * adj          # 수정 시가
    w["AC"] = w["AdjClose"]            # 수정 종가
    value20 = (w["Close"] * w["Volume"]).rolling(20).mean()
    w["tradable"] = (value20 >= 1e9) & (w["Close"] >= 1000)
    return w


def forward_returns(w, H):
    """t일 신호 -> t+1 시가 매수, t+H 종가 매도 수익률 (비용 차감)"""
    entry = w["AO"].shift(-1)
    exit_ = w["AC"].shift(-H)
    return (exit_ / entry - 1 - COST).astype("float32")


# ---------------- 지표 ----------------
def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


# ---------------- 신호 정의 (이름: (함수, 파라미터 그리드)) ----------------
def sig_ma_cross(w, short, long):
    c = w["Close"]
    return cross_up(c.rolling(short).mean(), c.rolling(long).mean())


def sig_rsi_rebound(w, n, th):
    r = rsi(w["Close"], n)
    return cross_up(r, pd.DataFrame(th, index=r.index, columns=r.columns))


def sig_bb_rebound(w, n, k):
    c = w["Close"]
    lower = c.rolling(n).mean() - k * c.rolling(n).std()
    return (c.shift(1) < lower.shift(1)) & (c > lower)


def sig_macd_cross(w, fast, slow):
    c = w["Close"]
    macd = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    return cross_up(macd, sig) & (macd < 0)


def sig_breakout(w, n, vol_mult):
    c, v = w["Close"], w["Volume"]
    return (c > w["High"].rolling(n).max().shift(1)) & (v > vol_mult * v.rolling(20).mean())


SIGNALS = {
    "MA골든크로스": (sig_ma_cross, {"short": [5, 10, 20], "long": [20, 60, 120]}),
    "RSI과매도반등": (sig_rsi_rebound, {"n": [7, 14], "th": [20, 25, 30, 35]}),
    "볼린저하단반등": (sig_bb_rebound, {"n": [10, 20, 40], "k": [1.5, 2.0, 2.5]}),
    "MACD음수권크로스": (sig_macd_cross, {"fast": [8, 12], "slow": [21, 26, 34]}),
    "거래량돌파": (sig_breakout, {"n": [20, 60, 120], "vol_mult": [2.0, 3.0, 5.0]}),
}


# ---------------- 평가 ----------------
def stats(r: pd.Series):
    r = r.dropna()
    if len(r) == 0:
        return dict(trades=0, mean=np.nan, win=np.nan, pf=np.nan, t=np.nan)
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    return dict(trades=len(r), mean=r.mean(), win=(r > 0).mean(),
                pf=gain / loss if loss > 0 else np.nan,
                t=r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan)


def evaluate(mask, fwd, period):
    m = mask.loc[period]
    return stats(fwd.loc[period][m].stack())


def main():
    RES.mkdir(exist_ok=True)
    w = load_wide()
    dates = w["Close"].index
    train = slice(dates[0], pd.Timestamp(TRAIN_END))
    test = slice(pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1), dates[-1])
    print(f"종목 {w['Close'].shape[1]}, 기간 {dates[0].date()} ~ {dates[-1].date()}")

    fwds = {H: forward_returns(w, H) for H in HOLDS}
    base = {}
    for H in HOLDS:
        for name, per in [("train", train), ("test", test)]:
            base[(H, name)] = evaluate(w["tradable"], fwds[H], per)["mean"]
    print("Base(무작위 매수) 평균수익률:", {k: round(v, 4) for k, v in base.items()})

    rows = []
    for sname, (fn, grid) in SIGNALS.items():
        keys = list(grid)
        for vals in itertools.product(*grid.values()):
            p = dict(zip(keys, vals))
            if "short" in p and p["short"] >= p["long"]:
                continue
            if "fast" in p and p["fast"] >= p["slow"]:
                continue
            mask = fn(w, **p).fillna(False) & w["tradable"]
            for H in HOLDS:
                for per_name, per in [("train", train), ("test", test)]:
                    s = evaluate(mask, fwds[H], per)
                    rows.append(dict(signal=sname, params=str(p), hold=H, period=per_name, **s,
                                     edge=s["mean"] - base[(H, per_name)]))
            print(f"  {sname} {p} 완료")

    res = pd.DataFrame(rows)
    tr = res[res.period == "train"].copy()
    tr.to_csv(RES / "grid_train.csv", index=False, encoding="utf-8-sig")
    res.to_csv(RES / "grid_all.csv", index=False, encoding="utf-8-sig")

    # 학습 구간에서 신호별 최적 (거래수 MIN_TRADES 이상, edge 최대) -> 검증 구간 성능 붙이기
    tr = tr[tr.trades >= MIN_TRADES]
    best = tr.loc[tr.groupby("signal")["edge"].idxmax()]
    te = res[res.period == "test"].set_index(["signal", "params", "hold"])
    out = best.set_index(["signal", "params", "hold"])[["trades", "mean", "win", "pf", "edge"]]
    out = out.join(te[["trades", "mean", "win", "pf", "edge"]], lsuffix="_train", rsuffix="_test")
    out = out.sort_values("edge_train", ascending=False)
    out.to_csv(RES / "best_train_vs_test.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n=== 신호별 학습구간 최적 파라미터 vs 검증구간 ===")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
