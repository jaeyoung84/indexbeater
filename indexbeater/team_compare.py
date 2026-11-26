"""팀원 최적 전략 동일 조건 비교 — 참고용 별개 시험 (실험기록 34번째 줄 규칙, 11/26 팀장)

성격: 시험 8 후보를 정하지 않는다. 4단계 상위 3개에 넣지 않는다. 시도 수표에 더하지 않는다.
      팀원이 고른 전략을 우리 엔진·점수표·대조군·포트폴리오(1억·20자리·5%)·비교 지수(D4) 그대로 놓고 본다.

엔진(engine.py·signals.py·exits.py·indicators.py·data_io.py·config.py)과 점수표(scorecard.py·metrics.py·scoring.py)는
고치지 않는다. 팀원 코드는 돌리지 않았다 — 노트북 글을 읽고 이 파일에 다시 짰다.

대상 (두 시장 모두, 비용 0.5% 본 표 · 0.25% 참고)
  가   기존 1등: 코스피 disparity H5 · 코스닥 rsi+stochastic H5 (W=3, 해마다 고른 세트 그대로) — 저장값과 같은지 먼저 확인
  나   대표조합 ⑥ 볼린저(20,2) 하단 + RSI(14) ≤ 30 H5 — 저장값과 같은지 먼저 확인
  다-1 김동겸 03: 이격도(20일선) 0.85 아래로 처음 내려간 날 + 이격도 ≥ 0.80 + 12−1 모멘텀 ≥ 0
  다-2 김동겸 07: 03·04·05 띠 합친 후보 중 로지스틱 점수 시장별·날짜별 상위 30% (올림). 포트폴리오 매수 순서 = 모델 점수 순
  다-3 김승찬: RSI(14) 25 상향 돌파 + 볼린저(10,1.5) 하단 이탈 후 복귀, W=3 (combo.and_entry)
  다-4 김승찬: 다-3 켜진 날 그날 스토캐스틱 %K(14,3) ≤ 20
  다 전략은 H 5·10·20 모두 계산하고 stage4.pick_top(n=1) 로 1개 고른다
  (고르는 칸 = 비용 0.5% 줄, 거래 300건 이상, 포트폴리오 있음 — classic.top3 와 같은 규칙).
  시장별 1등 = 가·나·다-1~4 고른 칸 6개를 같은 규칙으로 줄 세운 1등.

실행 (설계 구간만)
  python -m indexbeater teamcmp --reproduce   김동겸 표 24줄 다시 계산 → results/team_compare_reproduce.csv
  python -m indexbeater teamcmp --run         가·나 저장값 확인 → 56줄 → results/team_compare_*.csv
  python -m indexbeater teamcmp --random      시장별 1등 랜덤 매수 1,000번 (1등이 가·나면 이미 있는 결과를 가져옴)
  --overwrite                                 이미 있는 결과 파일이 다를 때 덮어쓰기 (팀장 승인 뒤)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from indexbeater import config as C
from indexbeater import engine as E
from indexbeater import indicators as I
from indexbeater import metrics as M
from indexbeater import signals as S

OUT_SCORE = C.RESULTS / "team_compare_scorecard.csv"
OUT_RANK = C.RESULTS / "team_compare_rank.csv"
OUT_DAILY = C.RESULTS / "team_compare_portfolio_daily.csv"
OUT_CHECK = C.RESULTS / "team_compare_check.csv"
OUT_REPRO = C.RESULTS / "team_compare_reproduce.csv"
OUT_FOLDS = C.RESULTS / "team_compare_07_folds.csv"
OUT_RANDOM = C.RESULTS / "team_compare_random.csv"
OUT_RANDOM_TRIALS = C.RESULTS / "team_compare_random_trials.csv"

D03, D07, D3, D4 = "다-1", "다-2", "다-3", "다-4"
STRATS = {   # 구분 → (신호 이름, 설명, 표시)
    D03: ("김동겸03_이격도80~85+모멘텀", "김동겸 03: 이격도 0.85 아래 처음 + 이격도≥0.80 + 12−1 모멘텀≥0",
          ["설계구간안에서고름", "우리격자밖(0.85·0.80)"]),
    D07: ("김동겸07_로지스틱상위30%", "김동겸 07: 이격도 80~95% 띠 후보 중 로지스틱 점수 상위 30%",
          ["설계구간안에서고름", "매수순서다름(모델점수)"]),
    D3: ("김승찬_RSI25+볼린저10", "김승찬: RSI(14) 25 상향 돌파 + 볼린저(10,1.5) 하단 복귀 (W=3)",
         ["설계구간안에서고름"]),
    D4: ("김승찬_RSI25+볼린저10+%K20", "김승찬: 위 + 그날 스토캐스틱 %K(14,3) ≤ 20",
         ["설계구간안에서고름"]),
}
TOP_SHARE = 0.30                 # 07: 날짜·시장별 상위 30% (올림)
BANDS = [("03", 0.80, 0.85), ("04", 0.85, 0.90), ("05", 0.90, 0.95)]
CORE_X = ["disparity20", "momentum12_1", "rsi14"]
FULL_X = CORE_X + ["dist_sma60", "dist_sma200", "ret5", "ret20", "vol20", "atr14_pct",
                   "bb_width20", "adx14", "value_ratio20", "mktcap_rank"]


# ================================================================ 매수 표
def _allowed(d) -> pd.DataFrame:
    dates = d["adj_close"].index
    return d["tradable"] & np.asarray(dates >= C.SIGNAL_START)[:, None]


def disparity_momentum(d):
    """20일 이격도 (종가 ÷ 20일 평균), 12−1 모멘텀 (20일 전 종가 ÷ 250일 전 종가 − 1)."""
    close = S._c(d)
    return I.disparity(close, 20), close.shift(20) / close.shift(250) - 1


def band_rules(d) -> dict[str, pd.DataFrame]:
    """03·04·05 띠 (김동겸 노트북 셀 3 그대로). 03 만 아래 끝 포함(≥), 04·05 는 초과(>)."""
    disp, mom = disparity_momentum(d)
    allowed = _allowed(d)
    out = {}
    for name, lower, upper in BANDS:
        cross = S.entry_signal("disparity", d, {"ma": 20, "thr": upper})
        inside = disp.ge(lower) if name == "03" else disp.gt(lower)
        out[name] = cross & inside & mom.ge(0) & allowed
    return out


def _mask(e: pd.DataFrame) -> pd.DataFrame:
    from indexbeater import classic as K
    return K._mask(e)


def entry_03(d) -> pd.DataFrame:
    return _mask(band_rules(d)["03"])


def entry_d3(d) -> pd.DataFrame:
    from indexbeater import combo as CB
    parts = [S.entry_signal("rsi", d, {"period": 14, "low": 25}),
             S.entry_signal("bollinger", d, {"period": 10, "k": 1.5})]
    return _mask(CB.and_entry(parts, window=C.AND_WINDOW))


def entry_d4(d, d3: pd.DataFrame | None = None) -> pd.DataFrame:
    d3 = entry_d3(d) if d3 is None else d3
    pk = I.stochastic(S._h(d), S._l(d), S._c(d), 14, 3)[0]
    return _mask(d3 & (pk <= 20))


# ================================================================ 07 로지스틱 (김동겸 노트북 셀 3·5·6 을 글로 다시 짬)
def candidates(d, mask: pd.DataFrame) -> pd.DataFrame:
    """후보 칸 (t, j) 마다 특징 13개. 모든 값은 그날 종가까지의 자료."""
    close = S._c(d)
    codes = close.columns
    disp, mom = disparity_momentum(d)
    allowed = _allowed(d)
    t, j = np.nonzero(mask.to_numpy(bool))

    def v(frame):
        return frame.to_numpy()[t, j]

    out = pd.DataFrame({"t_idx": t, "j": j, "code": codes[j], "market": v(d["market"]), "value20": v(d["value20"]),
                        "disparity20": v(disp), "momentum12_1": v(mom), "rsi14": v(I.rsi(close, 14))})
    for n in (60, 200):
        out[f"dist_sma{n}"] = v(close / I.sma(close, n) - 1)
    for n in (5, 20):
        out[f"ret{n}"] = v(close / close.shift(n) - 1)
    out["vol20"] = v((close / close.shift(1) - 1).rolling(20).std())
    high, low = S._h(d), S._l(d)
    out["atr14_pct"] = v(I.atr(high, low, close, 14) / close)
    lo, mid, up = I.bollinger(close, 20, 2.0)
    out["bb_width20"] = v((up - lo) / mid)
    out["adx14"] = v(I.adx_di(high, low, close, 14)[2])
    out["value_ratio20"] = v(d["value"] / d["value20"])
    out["mktcap_rank"] = np.nan
    for mk in C.UNIVERSE_MARKETS:
        inm = out["market"].eq(mk).to_numpy()
        rank = d["mktcap"].where(allowed & d["market"].eq(mk)).rank(axis=1, pct=True)
        out.loc[inm, "mktcap_rank"] = v(rank)[inm]
    out[FULL_X] = out[FULL_X].replace([np.inf, -np.inf], np.nan)
    out = out.set_index(["t_idx", "code"], drop=False)
    if not out.index.is_unique:
        raise ValueError("07 후보 (날짜, 종목) 가 겹친다")
    return out


def independent_labels(mask, data, rule, m) -> pd.DataFrame:
    """후보마다 따로 사고판 결과 (중복 매수 금지 없이). 체결된 것만, (t_idx, code) 색인."""
    from indexbeater import scorecard as SC
    tr = E.simulate_trades(mask, data, rule, dedupe=False, _m=m)
    f = tr.loc[tr["status"].eq(E.FILLED), ["t_idx", "code", "s_idx", "ret_gross"]]
    SC.check_finite(f["ret_gross"].to_numpy(float), "07 학습·예측 라벨")
    return f.set_index(["t_idx", "code"])


def keep_learn(learn: pd.DataFrame, start: int, hold: int) -> pd.DataFrame:
    """학습 표본 = 검증 해 첫날(start) 전에 실제로 팔았고, 예정 매도일(t+1+hold)도 start 전인 것만."""
    t = learn.index.get_level_values("t_idx").to_numpy()
    return learn.loc[(learn["s_idx"].to_numpy() < start) & (t + 1 + hold < start)]


def rank_day(pred: pd.DataFrame, share: float = TOP_SHARE) -> pd.DataFrame:
    """날짜마다 점수 내림차순 (같으면 20일 거래대금 내림차순 → 종목코드 오름차순).
    priority = 후보 수 − 순번 (클수록 먼저 삼), selected = 순번 < ceil(share × 후보 수)."""
    o = pred.sort_values(["score", "value20", "code"], ascending=[False, False, True])
    g = o.groupby("t_idx", sort=False)
    size = g["score"].transform("size")
    pos = g.cumcount()
    return o.assign(priority=size - pos, selected=pos.lt(np.ceil(share * size)))


def fit_07(train: pd.DataFrame, holdout: pd.DataFrame, truth: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    clf = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                        StandardScaler(),
                        LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, class_weight="balanced",
                                           random_state=C.SEED))
    y = train["ret_gross"].gt(C.COST_SELECT)
    if y.nunique() != 2:
        raise ValueError("07 학습 표본에 이긴 거래와 진 거래가 모두 있어야 한다")
    with threadpool_limits(limits=2):
        clf.fit(train[FULL_X], y)
        pred = clf.predict_proba(holdout[FULL_X])[:, 1]
    ranked = rank_day(holdout.reset_index(drop=True).assign(score=pred)).set_index(["t_idx", "code"], drop=False)
    lab = ranked.join(truth[["ret_gross"]], how="inner")
    wins = lab["ret_gross"].gt(C.COST_SELECT)
    diag = {"학습표본": len(y), "검증후보": len(ranked), "실제선택비율": float(ranked["selected"].mean()),
            "AUC": roc_auc_score(wins, lab["score"]) if wins.nunique() == 2 else np.nan,
            "전체후보승률": float(wins.mean()), "상위후보승률": float(wins.loc[lab["selected"]].mean())}
    return ranked.loc[ranked["selected"]], diag


def orders_07(ctx, log=print, holds=C.HOLDS) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """H 마다 07 매수 순서 표 (int16, 0 = 안 삼, 클수록 먼저). 2015~2019 해마다 그 해 시작 21거래일 전까지로 학습.
    holds = 계산할 H (기본 5·10·20; buy_export 는 H5 만)."""
    from indexbeater.exits import ExitRule
    t0 = time.time()
    d, dates = ctx.data, ctx.dates
    bands = band_rules(d)
    mask = bands["03"] | bands["04"] | bands["05"]
    cands = candidates(d, mask)
    log(f"  07 후보 {len(cands):,}개 ({', '.join(f'{k} {v:,}' for k, v in cands.groupby('market').size().items())})")
    oos = np.isin(ctx.years, C.WF_YEARS)
    cyear = ctx.years[cands["t_idx"].to_numpy()]
    out, folds = {}, []
    for h in holds:
        rule = ExitRule(hold=h)
        truth = independent_labels(mask & oos[:, None], d, rule, ctx.m)
        order = np.zeros(mask.shape, dtype="int16")
        for y in C.WF_YEARS:
            start = int(dates.searchsorted(pd.Timestamp(y, 1, 1)))
            cut = start - C.WF_CUT_OFFSET
            prior = {k: f.iloc[:start] for k, f in d.items()}
            tmask = mask.iloc[:start] & (np.arange(start) <= cut)[:, None]
            learn = independent_labels(tmask, prior, rule, E._Market(prior))
            train = cands.join(keep_learn(learn, start, h), how="inner")
            hold_out = cands.loc[cyear == y]
            for mk in C.UNIVERSE_MARKETS:
                chosen, diag = fit_07(train.loc[train["market"].eq(mk)], hold_out.loc[hold_out["market"].eq(mk)],
                                      truth)
                order[chosen["t_idx"].to_numpy(), chosen["j"].to_numpy()] = chosen["priority"].to_numpy()
                folds.append({"시장": mk, "H": h, "연도": y, **diag})
            del prior
        out[h] = pd.DataFrame(order, index=dates, columns=ctx.m.codes)
        log(f"  07 H={h} 학습 끝 ({time.time() - t0:.0f}초)")
    return out, pd.DataFrame(folds)


# ================================================================ 김동겸 표 다시 계산 (팀원 방법 그대로)
REPORTED = """
KOSPI 0.50%
5 03 79.2 40.0 17.0 9.3 12.9 1246 57.14 1.79 1.16 1.01 4 6.40 -24.15 +0.67
5 07 81.5 40.0 25.0 8.4 8.2 3925 50.88 0.42 0.72 2.73 5 2.02 -25.19 -3.70
10 03 64.6 34.2 12.0 9.6 8.8 1122 51.78 1.27 0.85 -0.31 4 3.88 -24.40 -1.84
10 07 73.0 32.2 25.0 9.7 6.1 3351 48.11 0.44 0.80 2.71 5 -1.55 -36.08 -7.27
20 03 50.6 21.1 12.0 9.8 7.7 1012 50.30 1.35 1.05 -0.31 4 5.72 -24.11 -0.00
20 07 52.1 15.6 21.9 10.2 4.3 2804 45.76 0.31 0.78 1.98 4 12.44 -26.01 +6.72
KOSDAQ 0.50%
5 03 88.2 40.0 23.6 10.2 14.3 2936 59.13 2.48 1.53 1.73 5 10.32 -22.46 +7.26
5 07 82.8 40.0 25.0 8.9 8.9 6649 51.81 0.74 0.99 4.13 5 7.22 -36.64 +0.45
10 03 76.5 40.0 15.4 10.4 10.8 2648 54.34 2.14 1.11 0.67 4 -7.00 -52.48 -10.52
10 07 60.9 26.2 19.6 8.9 6.2 5434 48.23 0.35 0.66 1.52 4 -6.76 -52.01 -12.22
20 03 53.2 20.1 13.6 10.5 9.0 2398 51.96 2.24 1.00 0.32 4 -15.28 -72.68 -21.04
20 07 42.5 12.6 15.9 10.1 3.9 4571 45.20 0.22 0.63 0.78 4 -3.22 -45.63 -8.29
KOSPI 0.25%
5 03 80.3 40.0 17.0 9.4 13.8 1246 58.43 2.04 1.16 1.01 4 8.78 -21.63 +3.06
5 07 83.2 40.0 25.0 8.6 9.6 3925 52.87 0.67 0.72 2.73 5 9.66 -23.52 +3.94
10 03 65.6 34.2 12.0 9.6 9.8 1122 53.12 1.52 0.85 -0.31 4 5.76 -22.65 +0.04
10 07 74.3 32.2 25.0 9.9 7.2 3351 49.60 0.69 0.80 2.71 5 3.26 -29.40 -2.46
20 03 51.3 21.1 12.0 10.0 8.2 1012 50.99 1.60 1.05 -0.31 4 7.19 -24.04 +1.47
20 07 53.0 15.6 21.9 10.4 5.1 2804 46.75 0.56 0.78 1.98 4 15.59 -25.77 +9.86
KOSDAQ 0.25%
5 03 88.9 40.0 23.6 10.3 15.0 2936 60.49 2.73 1.53 1.73 5 15.40 -21.12 +12.18
5 07 84.2 40.0 25.0 9.1 10.1 6649 53.48 0.99 0.99 4.13 5 17.24 -22.82 +9.84
10 03 77.4 40.0 15.4 10.4 11.7 2648 55.55 2.39 1.11 0.67 4 -3.86 -46.88 -7.49
10 07 62.0 26.2 19.6 9.0 7.1 5434 49.52 0.60 0.66 1.52 4 -1.85 -43.27 -7.59
20 03 53.9 20.1 13.6 10.4 9.8 2398 53.04 2.49 1.00 0.32 4 -13.54 -70.30 -19.42
20 07 43.3 12.6 15.9 10.4 4.4 4571 45.90 0.47 0.63 0.78 4 -0.53 -40.35 -5.69
"""
REPORTED_COLS = ["점수", "수익점수", "안정점수", "손익비점수", "승률점수", "청산거래", "승률", "평균순수익", "Edge",
                 "t값", "플러스해수", "포트CAGR", "포트MDD", "ETF차"]
PERCENT_COLS = {"승률", "평균순수익", "Edge", "포트CAGR", "포트MDD", "ETF차"}   # 팀원 표는 % 로 적힘 → ÷100
TOL = {**{c: 0.05 + 1e-9 for c in ["점수", "수익점수", "안정점수", "손익비점수", "승률점수"]},
       **{c: 5e-5 + 1e-12 for c in PERCENT_COLS}, "t값": 0.005 + 1e-9, "청산거래": 0, "플러스해수": 0}


def reported() -> pd.DataFrame:
    rows, mk, cost = [], None, None
    for line in REPORTED.strip().splitlines():
        p = line.split()
        if len(p) == 2:
            mk, cost = p[0], float(p[1].rstrip("%")) / 100
            continue
        r = {"시장": mk, "비용": cost, "H": int(p[0]), "ID": p[1]}
        for c, x in zip(REPORTED_COLS, p[2:]):
            v = float(x)
            r[c] = v / 100 if c in PERCENT_COLS else (int(v) if c in ("청산거래", "플러스해수") else v)
        rows.append(r)
    return pd.DataFrame(rows)


def _cagr(s: pd.Series) -> float:
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan


def etf_index() -> pd.DataFrame:
    from indexbeater import benchmark as B
    raw = B.load_raw(end=C.DESIGN_END, allow_test=False)
    return B.total_return(raw).pivot(index="date", columns="bench_market", values="tr_index")


def evaluate_teammate(ctx, name: str, order: pd.DataFrame, hold: int, etf: pd.DataFrame) -> list[dict]:
    """김동겸 노트북 evaluate() 를 글로 다시 짠 것 (청산거래·Edge·t·플러스해·점수·CAGR·MDD·ETF 차)."""
    from indexbeater import scorecard as SC
    from indexbeater.exits import ExitRule
    from indexbeater.scoring import signal_score
    rule = ExitRule(hold=hold)
    entry = order.gt(0)
    tr = E.simulate_trades(entry, ctx.data, rule, _m=ctx.m)
    f = tr.loc[tr["status"].eq(E.FILLED)].copy()
    ti = f["t_idx"].to_numpy(int)
    ji = pd.Index(ctx.m.codes).get_indexer(f["code"])
    f["edge"] = f["ret_gross"].to_numpy(float) - ctx.base[hold].matched(entry.to_numpy(), ti, ji)["ctrl_gross"]
    f["year"] = ctx.years[ti]
    cal = ctx.dates[np.isin(ctx.years, C.WF_YEARS)]
    rows = []
    for mk in C.UNIVERSE_MARKETS:
        mf = f.loc[f["market"].eq(mk)]
        eb = SC._edge_block(mf, ctx, cal)
        ann = mf.groupby("year").agg(edge=("edge", "mean"), matched=("edge", "count")).reindex(C.WF_YEARS)
        pos = int(((ann["edge"] > 0) & (ann["matched"] >= C.MIN_TRADES_PER_YEAR)).sum())
        for cost in (C.COST_SELECT, C.COST_ROUNDTRIP):
            net = mf["ret_gross"].to_numpy(float) - cost
            daily, _ = E.run_portfolio(order, ctx.data, rule, market=mk, cost=cost)
            eq = daily["equity"].loc[cal]
            ov = pd.concat([eq.rename("s"), etf[mk].rename("e")], axis=1).dropna()
            wl = M.win_loss(net)
            sc = signal_score(eb["edge"], hold, pos, eb["t"], wl["평균이익"], wl["평균손실"], wl["승률"], len(net),
                              n_years=len(C.WF_YEARS))
            rows.append({"시장": mk, "비용": cost, "H": hold, "ID": name, "점수": sc["점수"], "수익점수": sc["수익점수"],
                         "안정점수": sc["안정점수"], "손익비점수": sc["손익비점수"], "승률점수": sc["승률점수"],
                         "청산거래": len(net), "승률": float((net > 0).mean()), "평균순수익": float(net.mean()),
                         "Edge": eb["edge"], "t값": eb["t"], "플러스해수": pos, "포트CAGR": _cagr(eq),
                         "포트MDD": M.max_drawdown(eq),
                         "ETF차": _cagr(ov["s"]) - _cagr(ov["e"]) if len(ov) > 1 else np.nan,
                         "ETF비교시작": ov.index[0].date().isoformat() if len(ov) else ""})
    return rows


def reproduce(ctx, o7: dict, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """김동겸 24줄 다시 계산 → (긴 비교 표, 다시 계산한 넓은 표)."""
    t0 = time.time()
    etf = etf_index()
    oos = np.isin(ctx.years, C.WF_YEARS)
    r03 = (band_rules(ctx.data)["03"] & oos[:, None]).astype("int16")
    mine = []
    for h in C.HOLDS:
        for name, order in (("03", r03), ("07", o7[h])):
            mine += evaluate_teammate(ctx, name, order, h, etf)
        log(f"  김동겸 방법 H={h} 끝 ({time.time() - t0:.0f}초)")
    mine = pd.DataFrame(mine)
    rep = reported()
    keys = ["시장", "비용", "H", "ID"]
    both = rep.merge(mine, on=keys, suffixes=("_팀원", "_재현"), validate="one_to_one")
    if len(both) != 24:
        raise SystemExit(f"✗ 김동겸 표 24줄과 다시 계산한 줄이 다 맞물리지 않는다 ({len(both)}줄)")
    long = []
    for r in both.to_dict("records"):
        for c in REPORTED_COLS:
            a, b = r[f"{c}_팀원"], r[f"{c}_재현"]
            d = abs(a - b) if pd.notna(b) else np.inf
            long.append({**{k: r[k] for k in keys}, "열": c, "팀원값": a, "재현값": b, "차이": b - a,
                         "허용": TOL[c], "통과": bool(d <= TOL[c]),
                         "비고": "팀원은 ETF 겹친 기간만" if c == "ETF차" else ""})
    return pd.DataFrame(long), mine


# ================================================================ 우리 조건으로 점수표·포트폴리오
def score_rows(ctx, mk, h, ysets, ents, name, flags, made=None) -> list[dict]:
    """해마다 그 해 세트의 매매만 이어 점수표 2줄 (scorecard._score_rows 그대로). made = (세트, H) → 매매표 저장소."""
    from indexbeater import scorecard as SC
    made = {} if made is None else made
    for key in sorted(set(ysets.values())):
        if (key, h) not in made:
            ft, st, _ = SC._trade_table(ctx, ents[key], h)
            made[(key, h)] = (ft, st)
    parts, sparts = [], []
    for y in C.WF_YEARS:
        ft, st = made[(ysets[y], h)]
        parts.append(ft[(ft["market"] == mk) & (ft["year"] == y)])
        sparts.append(st[(st["market"] == mk) & (st["year"] == y)])
    oos = pd.concat(parts, ignore_index=True)
    st_oos = pd.concat(sparts, ignore_index=True)
    return SC._score_rows(name, mk, h, oos, st_oos, flags, ctx)


def add_portfolio(rows, ctx, mk, h, sig, has_cand: bool, meta: dict, daily: list):
    """점수표 2줄 각각에 그 비용의 포트폴리오 (stage4.portfolio_at_cost). 후보 0건이면 빈칸."""
    from indexbeater import scorecard as SC
    from indexbeater import stage4 as S4
    for r in rows:
        r.update(meta)
        if not has_cand:
            r["표시"] = ",".join(x for x in [r["표시"], "포트폴리오없음(후보0건)"] if x)
            continue
        cost = C.COST_SELECT if r["비용"] == S4.COST_ROW else C.COST_ROUNDTRIP
        pr, eq = S4.portfolio_at_cost(sig, ctx, mk, h, cost)
        SC.check_finite(eq.to_numpy(float), f"포트폴리오 {mk} {r['신호']} H={h}")
        r.update(pr)
        daily.append(pd.DataFrame({"date": eq.index, "시장": mk, "구분": meta["구분"], "신호": r["신호"], "H": h,
                                   "비용": r["비용"], "equity": eq.to_numpy()}))
    return rows


def team_inputs(ctx, o7: dict) -> dict:
    """다 전략 매수 표: 구분 → H → (ysets, ents, 포트폴리오 신호 표)."""
    from indexbeater import stage4 as S4
    d = ctx.data
    one = {y: 0 for y in C.WF_YEARS}
    d3 = entry_d3(d)
    ent = {D03: entry_03(d), D3: d3, D4: entry_d4(d, d3)}
    out = {}
    for g in (D03, D07, D3, D4):
        out[g] = {}
        for h in C.HOLDS:
            if g == D07:
                out[g][h] = (one, {0: o7[h].gt(0)}, o7[h])          # 매수 순서 = 모델 점수 순 (priority)
            else:
                ents = {0: ent[g]}
                out[g][h] = (one, ents, S4.year_signal(ctx, one, ents))
    return out


def pool_rule(df: pd.DataFrame) -> pd.Series:
    """classic.top3 와 같은 고르는 칸 규칙: 거래 300건 이상 + 포트폴리오 있음."""
    return (df["거래수"] >= C.MIN_TRADES) & df["포트_수익_2015"].notna()


def rank_table(sc: pd.DataFrame) -> pd.DataFrame:
    """다 전략마다 H 1개 (pick_top n=1) → 시장마다 6칸 → 같은 규칙으로 줄 세움. 0.25% 값도 옆에 붙인다."""
    from indexbeater import stage4 as S4
    main = sc[sc["비용"] == S4.COST_ROW]
    low = sc[sc["비용"] != S4.COST_ROW].set_index(["시장", "신호", "H"])
    out = []
    for mk in C.UNIVERSE_MARKETS:
        m = main[main["시장"] == mk]
        six = [m[m["구분"].isin(["가", "나"])].assign(H고르기="고정(H5)")]
        for g in STRATS:
            cand = m[m["구분"] == g]
            ok = cand[pool_rule(cand)]
            how = "pick_top"
            if ok.empty:                                          # 규칙 통과 칸이 없다 → 포트폴리오 있는 칸 중에서, 표시
                ok, how = cand[cand["포트_수익_2015"].notna()], "규칙미달_포트있는칸에서"
            if ok.empty:
                ok, how = cand, "규칙미달_전체에서"
            p = S4.pick_top(ok, n=1).drop(columns=["등수", "그때최고점수", "1점안칸수"])
            six.append(p.assign(H고르기=how))
        six = pd.concat(six, ignore_index=True)
        good = six[pool_rule(six)]
        ranked = S4.pick_top(good, n=len(good)) if len(good) else good.assign(등수=pd.NA)
        rest = six[~six["신호"].isin(ranked["신호"])].assign(등수=pd.NA)
        out.append(pd.concat([ranked, rest], ignore_index=True))
    df = pd.concat(out, ignore_index=True)
    for c in ["점수", "포트_5년수익", "5년초과수익", "포트_MDD", "지수이긴해수"]:
        df[f"{c}_0.25%"] = [low.loc[(r["시장"], r["신호"], r["H"]), c] for r in df.to_dict("records")]
    df["표본부족"] = df["거래수"] < C.MIN_TRADES
    df["시장1등"] = df["등수"] == 1
    return df


def run(log=print, tol: float = 1e-12):
    from indexbeater import benchmark as B
    from indexbeater import close_exit as CE
    from indexbeater import combo as CB
    from indexbeater import stage4 as S4
    t0 = time.time()
    _, ctx, picks, flags, cache = CB.prepare(log=log)
    rows, daily, checks = [], [], []
    # 가·나: 다시 계산 → 저장된 H5 0.5% 줄과 같은지
    for row in CE.targets().to_dict("records"):
        mk = row["시장"]
        ysets, ents, fl = CE.cell_inputs(row, ctx, picks, flags, cache)
        g = "가" if row["구분"] == "기존1등" else "나"
        sr = score_rows(ctx, mk, 5, ysets, ents, row["신호"], fl)
        sig = S4.year_signal(ctx, ysets, ents)
        add_portfolio(sr, ctx, mk, 5, sig, CB.has_portfolio_candidate(ctx, mk, ysets, ents),
                      {"구분": g, "설명": f"{row['구분']} {row['신호']} H5", "종류": row["종류"]}, daily)
        r = next(x for x in sr if x["비용"] == S4.COST_ROW)
        meta = {"구분", "설명", "종류"}                        # 이 파일이 붙인 이름표 (계산 값 아님) → 비교 안 함
        cols = [c for c in list(r) + S4.PORT_COLS if c in row and c not in CE.NOT_CHECKED | meta]
        d = {c: CE._diff(r[c], row[c]) for c in dict.fromkeys(cols)}
        worst = max(d.values())
        checks.append({"시장": mk, "구분": g, "신호": row["신호"], "비교열수": len(d), "가장큰차이": worst,
                       "다른열": ",".join(c for c, v in d.items() if v > tol), "통과": worst <= tol})
        if worst > tol:
            raise SystemExit(f"✗ {mk} {row['신호']}: 다시 계산 ≠ 저장된 H5 ({checks[-1]['다른열']}, {worst:.2e})")
        log(f"  ✓ {g} {mk} {row['신호']} H5 = 저장값 ({len(d)}열, 가장 큰 차이 {worst:.1e}, {time.time() - t0:.0f}초)")
        rows += sr
    # 다: 팀원 전략 4개 × H 3개 × 두 시장
    o7, folds = orders_07(ctx, log=log)
    inp = team_inputs(ctx, o7)
    for g, (name, desc, fl) in STRATS.items():
        made = {}
        for h in C.HOLDS:
            ysets, ents, sig = inp[g][h]
            for mk in C.UNIVERSE_MARKETS:
                sr = score_rows(ctx, mk, h, ysets, ents, name, fl, made)
                add_portfolio(sr, ctx, mk, h, sig, CB.has_portfolio_candidate(ctx, mk, ysets, ents),
                              {"구분": g, "설명": desc, "종류": "팀원"}, daily)
                rows += sr
            made.clear()
            log(f"  {g} {name} H={h} 끝 ({time.time() - t0:.0f}초)")
    sc = pd.DataFrame(rows)
    sc = sc.reindex(columns=list(dict.fromkeys(list(sc.columns) + S4.PORT_COLS)))
    sc = B.add_excess(sc, B.market_yearly(B.load_market()))
    sc = S4.add_index_compare(sc)
    no = sc["포트_수익_2015"].isna()
    for col in ("포트_5년수익", "지수_5년수익", "5년초과수익", "지수이김_5년"):
        sc[col] = sc[col].mask(no)
    first = ["시장", "구분", "종류", "신호", "설명", "H", "비용", "점수", "거래수", "Edge", "t값", "플러스해수", "승률",
             "평균수익", "포트_5년수익", "지수_5년수익", "5년초과수익", "지수이긴해수", "포트_MDD", "표시"]
    sc = sc[first + [c for c in sc.columns if c not in first]]
    rk = rank_table(sc)
    return sc, rk, pd.concat(daily, ignore_index=True), pd.DataFrame(checks), folds


# ================================================================ 시장별 1등 랜덤 매수 1,000번
def random_step(log=print):
    from indexbeater import classic as K
    from indexbeater import combo as CB
    from indexbeater import stage4 as S4
    rk = pd.read_csv(OUT_RANK, encoding="utf-8-sig", float_precision="round_trip")
    top = rk[rk["등수"] == 1]
    summ, trials, ctx, o7 = [], [], None, None
    for row in top.to_dict("records"):
        mk, g = row["시장"], row["구분"]
        if g in ("가", "나"):
            src = S4.OUT_RANDOM if g == "가" else K.OUT_RANDOM
            s = pd.read_csv(src, encoding="utf-8-sig")
            summ.append(s[s["시장"] == mk].assign(구분=g, 출처=src.name))
            log(f"  {mk} 1등 = {g} {row['신호']} → 이미 있는 {src.name} 를 씀")
            continue
        if ctx is None:
            _, ctx, _, _, _ = CB.prepare(log=log)
        if g == D07:
            if o7 is None:
                o7, _ = orders_07(ctx, log=log)
            sig = o7[int(row["H"])]                                   # 매수 순서 = 모델 점수 순 그대로
        else:
            sig = _team_one(ctx, g)[2]
        log(f"  {mk} 1등 = {g} {row['신호']} H{int(row['H'])} → 랜덤 매수 {C.RANDOM_TRIALS:,}번 새로 계산")
        s, tr = S4.random_test_sig(sig, row, ctx, log=log)
        summ.append(s.assign(구분=g, 출처="새로 계산"))
        trials.append(tr.assign(구분=g, 신호=row["신호"], H=int(row["H"])))
    return (pd.concat(summ, ignore_index=True),
            pd.concat(trials, ignore_index=True) if trials else pd.DataFrame())


def _team_one(ctx, g):
    from indexbeater import stage4 as S4
    d = ctx.data
    e = {D03: entry_03, D3: entry_d3, D4: entry_d4}[g](d)
    one = {y: 0 for y in C.WF_YEARS}
    return one, {0: e}, S4.year_signal(ctx, one, {0: e})


# ================================================================ 명령줄
def main(argv=None):
    import argparse
    from indexbeater import benchmark as B
    from indexbeater import combo as CB
    ap = argparse.ArgumentParser(prog="python -m indexbeater teamcmp", description=__doc__.split("\n")[0])
    ap.add_argument("--reproduce", action="store_true", help="김동겸 표 24줄 다시 계산 → team_compare_reproduce.csv")
    ap.add_argument("--run", action="store_true", help="가·나 확인 → 56줄 → team_compare_*.csv")
    ap.add_argument("--random", action="store_true", help="시장별 1등 랜덤 매수 1,000번")
    ap.add_argument("--overwrite", action="store_true", help="결과 파일 덮어쓰기 (팀장 승인 뒤)")
    a = ap.parse_args(argv)
    if not (a.reproduce or a.run or a.random):
        ap.print_help()
        return 0
    t0 = time.time()
    pd.set_option("display.width", 250)
    if a.reproduce:
        _, ctx, _, _, _ = CB.prepare()
        o7, folds = orders_07(ctx)
        long, mine = reproduce(ctx, o7)
        bad = long[~long["통과"]]
        print(f"\n김동겸 표 대조: {len(long)}칸 중 통과 {int(long['통과'].sum())}칸")
        if len(bad):
            print(bad.to_string(index=False))
        fm = folds.groupby(["시장", "H"])[["AUC", "실제선택비율", "전체후보승률", "상위후보승률"]].mean()
        print(fm.to_string())
        for d, path in ((long, OUT_REPRO), (folds, OUT_FOLDS)):
            print(f"  {path.name}: {B._save(d, path, a.overwrite)}")
    if a.run:
        sc, rk, daily, checks, folds = run()
        show = ["시장", "등수", "구분", "신호", "H", "H고르기", "점수", "거래수", "포트_5년수익", "지수_5년수익",
                "5년초과수익", "지수이긴해수", "포트_MDD", "포트_5년수익_0.25%", "표시"]
        print(rk[show].to_string(index=False))
        print(checks.to_string(index=False))
        outs = [(sc, OUT_SCORE), (rk, OUT_RANK), (daily, OUT_DAILY), (checks, OUT_CHECK)]
        if not OUT_FOLDS.exists():
            outs.append((folds, OUT_FOLDS))
        else:
            old = pd.read_csv(OUT_FOLDS, encoding="utf-8-sig", float_precision="round_trip")
            pd.testing.assert_frame_equal(old, folds.reset_index(drop=True), check_dtype=False, rtol=1e-12)
            print("  ✓ 07 학습 진단 = --reproduce 때와 같음")
        for d, path in outs:
            print(f"  {path.name}: {B._save(d, path, a.overwrite)}")
    if a.random:
        s, tr = random_step()
        print(s.to_string(index=False))
        print(f"  {OUT_RANDOM.name}: {B._save(s, OUT_RANDOM, a.overwrite)}")
        if len(tr):
            print(f"  {OUT_RANDOM_TRIALS.name}: {B._save(tr, OUT_RANDOM_TRIALS, a.overwrite)}")
    print(f"\n끝 ({time.time() - t0:.0f}초)")
    return 0
