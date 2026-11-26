"""대표 일봉차트 조합 8개 — 별개 비교분석용 시험 (참고, 실험기록 30번째 줄, 11/25 팀장)

성격: 4단계 상위 3개·매도 규칙 단계·시험 8 에 넣지 않는다. 기존 결과와 '같은 조건'으로 나란히 놓고 보기만 한다.
      채택 여부는 팀이 정한다.

엔진(engine.py·signals.py·exits.py·indicators.py·data_io.py·config.py)과 점수표(scorecard.py·metrics.py·
scoring.py)는 고치지 않고 부르기만 한다. 지표는 indicators.py 것을 그대로 쓴다.

정의 (파라미터 고정, 해마다 고르지 않음. 가격 = 수정주가, 거래량 = volume_s)
  추세 상태 = 종가 > 200일 평균 이고 50일 평균 > 200일 평균
  ① 추세 상태 + 어제 RSI(14) 40~50, 어제 RSI < 그제 RSI, 오늘 RSI > 어제 RSI
  ② 종가 > 어제까지 20일 최고 고가 + 거래량 ≥ 1.5 × 어제까지 20일 평균 거래량 + ADX(14) ≥ 25 이고 +DI > −DI
  ③ MACD(12,26,9) 0 아래 골든크로스 · 느린 스토캐스틱(14,3,3) %K↑%D 그날 %K ≤ 20 · 볼린저(20,2) 하단 복귀,
     셋이 W=3 안에 (combo.and_entry 그대로)
  ④ 추세 상태 + ②          ⑤ ③ 이 켜진 날 추세 상태
  ⑥ 종가 < 볼린저(20,2) 하단 이고 RSI(14) ≤ 30
  ⑦ RSI(14) 30 상향 돌파 + OBV > OBV 9일 평균 이고 오늘 OBV > 어제 OBV
  ⑧ 추세 상태 + EMA5 > EMA10 + 최근 3일(오늘 포함) 저가 ≤ 그날 EMA10 + 종가 > EMA10 이고 > 어제 종가
     + RSI(14) 50~70 + ADX(14) > 20 + MACD 선(12,26) ≥ 0
  켜지는 날: ①②④⑥⑦⑧ = 조건이 모두 처음 함께 성립한 날 (signals.event, 기존 16번과 같은 방식).
            ③ = W=3 규칙, ⑤ = ③ & 추세 상태. 2012-07-01 전은 모두 끔 (signals.entry_signal 과 같음).

실행 (설계 구간만, 2020년 이후 자료는 열지 않는다)
  python -m indexbeater classic --score      8개 × H 3 점수표·포트폴리오 → classic_scorecard.csv·classic_portfolio_daily.csv
  python -m indexbeater classic --compare    기존 결과와 나란히 + 대표 조합 상위 3개 + 시도 수
  python -m indexbeater classic --random     대표 조합 1등 랜덤 매수 1,000번 (stage4 시험 4 와 같은 계산)
  python -m indexbeater classic --cost       대표 조합 상위 3개 0.25% (stage4 시험 6 과 같은 계산)
  python -m indexbeater classic --spa        SPA (참고): 기존 칸 + 대표 조합 칸 / 대표 조합 칸만
  --overwrite                                이미 있는 결과 파일이 다를 때 덮어쓰기 (팀장 승인 뒤)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from indexbeater import benchmark as B
from indexbeater import combo as CB
from indexbeater import config as C
from indexbeater import indicators as I
from indexbeater import signals as S
from indexbeater import stage4 as S4

KIND = "대표조합"
NAMES = {1: "c1_trend_rsi_hook", 2: "c2_breakout_vol_adx", 3: "c3_macd_stoch_boll", 4: "c4_trend_breakout",
         5: "c5_trend_reversal3", 6: "c6_boll_rsi_oversold", 7: "c7_rsi_obv", 8: "c8_ema10_pullback"}
DESC = {1: "① 상승추세 + RSI 40~50 눌림 후 반등", 2: "② 20일 신고가 + 거래량 1.5배 + ADX≥25·+DI>−DI",
        3: "③ MACD 0 아래 골든 · 스토캐스틱 20 아래 · 볼린저 하단 복귀 (W=3)", 4: "④ 상승추세 + ②",
        5: "⑤ 상승추세 + ③", 6: "⑥ 볼린저 하단 이탈 + RSI ≤ 30", 7: "⑦ RSI 30 상향 돌파 + OBV 9일 평균 위·증가",
        8: "⑧ 상승추세 EMA10 눌림 후 회복 (RSI 50~70·ADX>20·MACD≥0)"}

OUT_SCORE = C.RESULTS / "classic_scorecard.csv"
OUT_DAILY = C.RESULTS / "classic_portfolio_daily.csv"
OUT_COMPARE = C.RESULTS / "classic_compare.csv"
OUT_TOP = C.RESULTS / "classic_top3.csv"
OUT_TRIES = C.RESULTS / "classic_tries.csv"
OUT_RANDOM = C.RESULTS / "classic_random.csv"
OUT_RANDOM_TRIALS = C.RESULTS / "classic_random_trials.csv"
OUT_COST = C.RESULTS / "classic_cost.csv"
OUT_SPA = C.RESULTS / "classic_spa.csv"
OUT_SPA_CELLS = C.RESULTS / "classic_spa_cells.csv"
REF_COL = "참고_지수200일위비율"                 # ⑥ 만: 신호 매매 중 그날 비교 지수가 200일 평균 위였던 비율


# ================================================================ 조건 (1/0/빈칸 표와 그 재료)
class Cond:
    """True/False 표 + 재료 목록. 재료가 하나라도 빈칸이면 그날은 '계산 불가'(빈칸)로 둔다 (signals.as_cond)."""

    def __init__(self, mask, *inputs):
        self.mask, self.inputs = mask, list(inputs)

    def __and__(self, other: "Cond") -> "Cond":
        return Cond(self.mask & other.mask, *self.inputs, *other.inputs)

    def first_day(self) -> pd.DataFrame:
        """조건이 모두 처음 함께 성립한 날 (signals.event)."""
        return S.event(S.as_cond(self.mask, *self.inputs))

    def state(self) -> pd.DataFrame:
        """그날 성립 여부 (빈칸 = 거짓)."""
        return S.as_cond(self.mask, *self.inputs).eq(1.0)


def trend(d) -> Cond:
    c = S._c(d)
    s50, s200 = I.sma(c, 50), I.sma(c, 200)
    return Cond((c > s200) & (s50 > s200), c, s50, s200)


def rsi_hook(d) -> Cond:
    r = I.rsi(S._c(d), 14)
    r1, r2 = r.shift(1), r.shift(2)
    return Cond((r1 >= 40) & (r1 <= 50) & (r1 < r2) & (r > r1), r, r1, r2)


def breakout(d) -> Cond:
    brk, hh = S._new_high(d, 20)
    v = d["volume_s"]
    vavg = v.rolling(20, min_periods=20).mean().shift(1)
    pdi, mdi, adx = I.adx_di(S._h(d), S._l(d), S._c(d), 14)
    return Cond(brk & (v >= 1.5 * vavg) & (adx >= 25) & (pdi > mdi), S._c(d), hh, v, vavg, pdi, mdi, adx)


def slow_stoch(d) -> pd.DataFrame:
    """느린 스토캐스틱(14,3,3): slowK = 빠른 %K 의 3일 평균, slowD = slowK 의 3일 평균. slowK↑slowD 그날 slowK ≤ 20."""
    _, slow_k = I.stochastic(S._h(d), S._l(d), S._c(d), 14, 3)
    slow_d = I.sma(slow_k, 3)
    return S.cross_up(slow_k, slow_d) & (slow_k <= 20)


def three_parts(d) -> list[pd.DataFrame]:
    """③ 의 구성 신호 3개 (2012-07-01 전 끔, signals.entry_signal 과 같은 손질)."""
    return [_mask(S.sig_macd(d, 12, 26, 9)), _mask(slow_stoch(d)), _mask(S.sig_bollinger(d, 20, 2))]


def boll_rsi(d) -> Cond:
    c = S._c(d)
    lower, _, _ = I.bollinger(c, 20, 2)
    r = I.rsi(c, 14)
    return Cond((c < lower) & (r <= 30), c, lower, r)


def rsi_obv(d) -> Cond:
    c = S._c(d)
    r = I.rsi(c, 14)
    o = I.obv(c, d["volume_s"])
    o9, o1 = I.sma(o, 9), o.shift(1)
    return Cond(S.cross_up(r, 30) & (o > o9) & (o > o1), r, r.shift(1), o, o9, o1)


def ema10_pullback(d) -> Cond:
    c, lo = S._c(d), S._l(d)
    e5, e10 = I.ema(c, 5), I.ema(c, 10)
    touch = (lo <= e10) | (lo.shift(1) <= e10.shift(1)) | (lo.shift(2) <= e10.shift(2))
    r = I.rsi(c, 14)
    _, _, adx = I.adx_di(S._h(d), lo, c, 14)
    line, _ = I.macd(c, 12, 26, 9)
    c1 = c.shift(1)
    mask = ((e5 > e10) & touch & (c > e10) & (c > c1) & (r >= 50) & (r <= 70) & (adx > 20) & (line >= 0))
    return Cond(mask, c, c1, lo, lo.shift(1), lo.shift(2), e5, e10, e10.shift(1), e10.shift(2), r, adx, line)


def _mask(e: pd.DataFrame) -> pd.DataFrame:
    """빈칸 → False, 2012-07-01 전 → False (signals.raw_signal + entry_signal 과 같은 손질)."""
    e = e.fillna(False).astype(bool)
    keep = np.asarray(e.index >= C.SIGNAL_START)
    return e & keep[:, None]


def raw(k: int, d) -> pd.DataFrame:
    """조합 k 의 켜지는 날 표 (2012-07-01 자르기 전, ③⑤ 는 구성 신호가 이미 잘려 있음)."""
    if k == 1:
        return (trend(d) & rsi_hook(d)).first_day()
    if k == 2:
        return breakout(d).first_day()
    if k == 3:
        return CB.and_entry(three_parts(d), window=C.AND_WINDOW)
    if k == 4:
        return (trend(d) & breakout(d)).first_day()
    if k == 5:
        return CB.and_entry(three_parts(d), window=C.AND_WINDOW) & trend(d).state()
    if k == 6:
        return boll_rsi(d).first_day()
    if k == 7:
        return rsi_obv(d).first_day()
    if k == 8:
        return (trend(d) & ema10_pullback(d)).first_day()
    raise ValueError(f"조합 번호는 1~8: {k}")


def entry(k: int, d) -> pd.DataFrame:
    """조합 k 의 매수 표 (2012-07-01 전 모두 False)."""
    return _mask(raw(k, d))


# ================================================================ 점수표 (combo.combo_tables 와 같은 계산, 세트 하나 고정)
def index_above_200(mk: str) -> pd.Series:
    """그 시장 비교 지수(benchmark_market.csv)가 그날 200일 평균 위인가 (날짜 → True/False, 평균 없으면 빈칸)."""
    ms = B.load_market()
    g = ms[ms["market"] == mk].set_index("date").sort_index()["index"]
    s200 = g.rolling(200, min_periods=200).mean()
    return (g > s200).where(s200.notna())


def score(ctx, log=print) -> dict[str, pd.DataFrame]:
    """8개 × H 3 → 점수표 줄 (비용 2줄) · 포트폴리오 일별. 단독 점수표와 같은 함수(_trade_table·_score_rows·_portfolio)."""
    from indexbeater import scorecard as SC
    above = {mk: index_above_200(mk) for mk in C.UNIVERSE_MARKETS}
    scores, port = [], []
    for k, name in NAMES.items():
        t0 = time.time()
        ent = entry(k, ctx.data)
        ents = {0: ent}
        ysets = {y: 0 for y in C.WF_YEARS}
        for h in C.HOLDS:
            ft, st, _ = SC._trade_table(ctx, ent, h)
            for mk in C.UNIVERSE_MARKETS:
                parts = [ft[(ft["market"] == mk) & (ft["year"] == y)] for y in C.WF_YEARS]
                sparts = [st[(st["market"] == mk) & (st["year"] == y)] for y in C.WF_YEARS]
                oos = pd.concat(parts, ignore_index=True)
                st_oos = pd.concat(sparts, ignore_index=True)
                rows = SC._score_rows(name, mk, h, oos, st_oos, [], ctx)
                ref = np.nan
                if k == 6 and len(oos):
                    on = above[mk].reindex(ctx.dates[oos["t_idx"].to_numpy(int)])
                    ref = float(on.dropna().astype(bool).mean()) if on.notna().any() else np.nan
                for r in rows:
                    r.update({"종류": KIND, "번호": k, "설명": DESC[k], REF_COL: ref})
                if CB.has_portfolio_candidate(ctx, mk, ysets, ents):
                    pr, pdaily = SC._portfolio(name, mk, h, {y: {"set": 0} for y in C.WF_YEARS}, ents, ctx)
                    rows[0].update(pr)         # 포트폴리오는 0.5% 줄에만 (기존과 같음)
                    port.append(pdaily)
                else:
                    log(f"  {name} {mk} H={h}: 살 후보 0건 → 포트폴리오 빈칸")
                scores += rows
        log(f"  {name}: {time.time() - t0:.0f}초")
    sc = pd.DataFrame(scores)
    sc = sc.reindex(columns=list(dict.fromkeys(list(sc.columns) + S4.PORT_COLS)))
    sc = B.add_excess(sc, B.market_yearly(B.load_market()))
    return {"scorecard": sc, "portfolio_daily": pd.concat(port, ignore_index=True) if port else pd.DataFrame()}


def load_score() -> pd.DataFrame:
    if not OUT_SCORE.exists():
        raise SystemExit(f"✗ {OUT_SCORE.name} 이 없다 → 먼저 --score")
    return pd.read_csv(OUT_SCORE, encoding="utf-8-sig", float_precision="round_trip")


def cells(sc: pd.DataFrame) -> pd.DataFrame:
    """비용 0.5% 칸 (포트폴리오가 있는 줄) + 지수 비교 열."""
    c = S4.add_index_compare(sc[sc["비용"] == S4.COST_ROW].reset_index(drop=True))
    no = c["포트_수익_2015"].isna()            # 포트폴리오 없는 칸: 곱하기가 0 을 만들지 않게 빈칸으로
    for col in ("포트_5년수익", "지수_5년수익", "5년초과수익", "지수이김_5년"):
        c[col] = c[col].mask(no)               # 참/거짓 열도 빈칸을 담을 수 있게 (mask 가 형식을 넓힌다)
    return c


# ================================================================ 기존 결과와 나란히
COMPARE_COLS = ["시장", "구분", "종류", "신호", "설명", "H", "점수", "거래수", "표본부족", "포트_5년수익", "지수_5년수익",
                "5년초과수익", "지수이긴해수", "포트_MDD", "포트_초과수익_중앙값", "표시", "단독1위점수", "기존1등점수",
                "점수차_단독1위", "점수차_기존1등", REF_COL]


def compare(cl: pd.DataFrame, top_old: pd.DataFrame, single: pd.DataFrame) -> pd.DataFrame:
    """대표 조합 24칸(시장마다) + 기존 상위 3개. 단독 1위·기존 1등 점수와의 차이를 붙인다."""
    out = []
    for mk in C.UNIVERSE_MARKETS:
        best = S4.single_best(single, mk)
        old = top_old[top_old["시장"] == mk].sort_values("등수")
        o1 = float(old.iloc[0]["점수"])
        a = old.assign(구분=[f"기존 상위 {int(x)}등" for x in old["등수"]], 설명="")
        b = cl[cl["시장"] == mk].sort_values(["번호", "H"]).assign(구분="대표 조합")
        for g in (a, b):
            g = g.copy()
            g["표본부족"] = g["거래수"] < C.MIN_TRADES
            g["단독1위점수"] = float(best["점수"])
            g["기존1등점수"] = o1
            g["점수차_단독1위"] = g["점수"] - float(best["점수"])
            g["점수차_기존1등"] = g["점수"] - o1
            out.append(g.reindex(columns=COMPARE_COLS))
    return pd.concat(out, ignore_index=True)


def top3(cl: pd.DataFrame) -> pd.DataFrame:
    """대표 조합 칸 중 거래 300건 이상 → stage4.pick_top (1점 안 규칙) → 시장마다 상위 3개. 1등 = 무작위 시험 대상."""
    tops = []
    for mk in C.UNIVERSE_MARKETS:
        pool = cl[(cl["시장"] == mk) & (cl["거래수"] >= C.MIN_TRADES) & cl["포트_수익_2015"].notna()]
        if pool.empty:
            continue
        t = S4.pick_top(pool)
        t["일관성부족"] = t["포트_초과수익_중앙값"] < 0
        t["후보A"] = t["등수"] == 1
        tops.append(S4.add_index_compare(t))
    top = pd.concat(tops, ignore_index=True)
    first = [c for c in S4.SHOW + ["설명", REF_COL] if c in top.columns]
    return top[first + [c for c in top.columns if c not in first]]


def tries(sc: pd.DataFrame) -> pd.DataFrame:
    """시도 수: 기존 합계 (stage4_tries.csv) + 대표 조합 줄 (8 × H3 × 비용2). 기존 파일은 건드리지 않는다."""
    old = pd.read_csv(S4.OUT_TRIES, encoding="utf-8-sig")
    rows = []
    for mk in C.UNIVERSE_MARKETS:
        o = int(old[(old["시장"] == mk) & (old["항목"] == "합계 (팀장 몫)")]["수"].iloc[0])
        n = int((sc["시장"] == mk).sum())
        rows += [{"시장": mk, "항목": "기존 합계 (stage4_tries.csv)", "수": o},
                 {"시장": mk, "항목": "대표 조합 칸 (8 × H3 × 비용2)", "수": n},
                 {"시장": mk, "항목": "합계 (대표 조합 포함)", "수": o + n}]
    return pd.DataFrame(rows)


# ================================================================ 무작위·비용 시험 (stage4 와 같은 계산)
def signal_table(ctx, k: int) -> pd.DataFrame:
    """scorecard._portfolio 에 넘기는 것과 같은 표 (2015~2019 만 켬)."""
    return S4.year_signal(ctx, {y: 0 for y in C.WF_YEARS}, {0: entry(k, ctx.data)})


def _load_top() -> pd.DataFrame:
    if not OUT_TOP.exists():
        raise SystemExit(f"✗ {OUT_TOP.name} 이 없다 → 먼저 --compare")
    return pd.read_csv(OUT_TOP, encoding="utf-8-sig", float_precision="round_trip")


# ================================================================ SPA (참고)
def classic_daily(cl: pd.DataFrame, mk: str) -> pd.DataFrame:
    d = pd.read_csv(OUT_DAILY, encoding="utf-8-sig", parse_dates=["date"], float_precision="round_trip")
    d = d[d["시장"] == mk]
    d["칸"] = d["신호"] + "|H" + d["H"].astype(str)
    wide = d.pivot(index="date", columns="칸", values="equity")
    want = cl[(cl["시장"] == mk) & cl["포트_수익_2015"].notna()]
    keys = list(dict.fromkeys(want["신호"] + "|H" + want["H"].astype(str)))
    if [k for k in keys if k not in wide.columns] or wide[keys].isna().any().any():
        raise ValueError(f"{mk}: 대표 조합 일별 평가금액이 빠졌다")
    return wide[keys]


def run_spa(old_cells: pd.DataFrame, cl: pd.DataFrame, top_cl: pd.DataFrame, log=print):
    """SPA·RC (stage4.spa_test 그대로, 같은 SEED·부트스트랩). '기존+대표' 와 '대표만' 두 번. stage4_spa.csv 는 안 건드림."""
    saved = pd.read_csv(S4.OUT_SPA, encoding="utf-8-sig").set_index("시장")
    summ, per = [], []
    for mk in C.UNIVERSE_MARKETS:
        t0 = time.time()
        Xo = S4.excess_daily(S4.load_daily(old_cells, mk), mk)
        if Xo.shape[1] != int(saved.loc[mk, "칸수"]):
            raise ValueError(f"{mk}: 기존 칸 수 {Xo.shape[1]} ≠ stage4_spa.csv {saved.loc[mk, '칸수']}")
        Xc = S4.excess_daily(classic_daily(cl, mk), mk)
        idx = S4.stationary_indices(len(Xo), C.BOOTSTRAP_TRIALS, C.BLOCK_DAYS, np.random.default_rng(C.SEED))
        tk = top_cl[top_cl["시장"] == mk].assign(칸=lambda d: d["신호"] + "|H" + d["H"].astype(str))
        for what, X in (("기존+대표조합", pd.concat([Xo, Xc], axis=1)), ("대표조합만", Xc)):
            r = S4.spa_test(X.to_numpy(float), idx)
            pc = pd.DataFrame({"시장": mk, "묶음": what, "칸": X.columns,
                               "종류": np.where(X.columns.isin(Xc.columns), KIND, "기존"),
                               "일평균초과": r["dbar"], "표준화": r["t"], "칸별_p": r["칸별_p"]})
            if what == "대표조합만" and len(tk):
                sel = pc.set_index("칸").loc[tk["칸"]]
                pc["상위3_Holm_p"] = pc["칸"].map(dict(zip(tk["칸"], S4.holm(sel["칸별_p"].to_numpy()))))
                pc["상위3등수"] = pc["칸"].map(dict(zip(tk["칸"], tk["등수"])))
            best = pc.loc[pc["표준화"].idxmax()]
            summ.append({"시장": mk, "묶음": what, "칸수": r["K"], "날짜수": r["n"],
                         "칸수_대표조합": int(Xc.shape[1]), "SPA통계": r["SPA통계"], "SPA_p": r["SPA_p"],
                         "RC통계": r["RC통계"], "RC_p": r["RC_p"], "표준화1위칸": best["칸"],
                         "기존SPA_p(stage4)": float(saved.loc[mk, "SPA_p"]), "표시": "참고 (별개 비교분석)"})
            if what == "대표조합만":
                per.append(pc)
            log(f"  {mk} {what}: 칸 {r['K']} → SPA p={r['SPA_p']:.3f}, RC p={r['RC_p']:.3f}, "
                f"표준화 1위 {best['칸']} ({time.time() - t0:.0f}초)")
    return pd.DataFrame(summ), pd.concat(per, ignore_index=True)


# ================================================================ 명령
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m indexbeater classic", description=__doc__.split("\n")[0])
    for f in ("score", "compare", "random", "cost", "spa"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="결과 파일 덮어쓰기 (팀장 승인 뒤)")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.score:
        _, ctx, _, _, _ = CB.prepare()
        r = score(ctx)
        print(f"  {OUT_SCORE.name}: {B._save(r['scorecard'], OUT_SCORE, a.overwrite)}")
        print(f"  {OUT_DAILY.name}: {B._save(r['portfolio_daily'], OUT_DAILY, a.overwrite)}")
    if a.compare:
        sc = load_score()
        cl = cells(sc)
        single = pd.read_csv(S4.SINGLE, encoding="utf-8-sig", float_precision="round_trip").assign(종류="단독")
        single = single[single["비용"] == S4.COST_ROW]
        cmp_ = compare(cl, S4._top_from_file(), single)
        top = top3(cl)
        tr = tries(sc)
        print(cmp_.to_string(index=False))
        print(top[[c for c in S4.SHOW + ["설명"] if c in top.columns]].to_string(index=False))
        print(tr.to_string(index=False))
        for df, out in ((cmp_, OUT_COMPARE), (top, OUT_TOP), (tr, OUT_TRIES)):
            print(f"  {out.name}: {B._save(df, out, a.overwrite)}")
    if a.random or a.cost:
        top = _load_top()
        _, ctx, _, _, _ = CB.prepare()
        if a.random:
            ss, ts = [], []
            for row in top[top["후보A"]].to_dict("records"):
                s, tr = S4.random_test_sig(signal_table(ctx, int(row["번호"])), row, ctx)
                print(s.to_string(index=False))
                ss.append(s)
                ts.append(tr)
            print(f"  {OUT_RANDOM.name}: {B._save(pd.concat(ss, ignore_index=True), OUT_RANDOM, a.overwrite)}")
            print(f"  {OUT_RANDOM_TRIALS.name}: "
                  f"{B._save(pd.concat(ts, ignore_index=True), OUT_RANDOM_TRIALS, a.overwrite)}")
        if a.cost:
            idx = S4.index_yearly()
            c = pd.DataFrame([S4.cost_row(signal_table(ctx, int(r["번호"])), r, ctx, idx)
                              for r in top.to_dict("records")])
            print(f"  {OUT_COST.name}: {B._save(c, OUT_COST, a.overwrite)}")
    if a.spa:
        s, p = run_spa(S4.load_cells(triples_all=True), cells(load_score()), _load_top())
        print(s.to_string(index=False))
        print(f"  {OUT_SPA.name}: {B._save(s, OUT_SPA, a.overwrite)}")
        print(f"  {OUT_SPA_CELLS.name}: {B._save(p, OUT_SPA_CELLS, a.overwrite)}")
    if not any(vars(a)[f] for f in ("score", "compare", "random", "cost", "spa")):
        ap.print_help()
    print(f"\n끝 ({time.time() - t0:.0f}초)")
    return 0
