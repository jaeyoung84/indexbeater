"""4단계 — 조합 합격 시험 1~7 (기준 5.7-5·5.8·5.11·5.12, 설계 구간만)

규칙은 순위를 보기 전에 실험기록.csv 22번째 줄에 먼저 적었다 (11/25 팀장). 이 파일은 그 규칙을 그대로 옮긴 것이다.

  python -m indexbeater stage4 --rank      시험 3(개선 판정)·5.7-5 표·시장별 상위 3개 + 시험 7(지수 비교)
  python -m indexbeater stage4 --tries     시험 1: 시도 수표 (팀장 몫)
  python -m indexbeater stage4 --spa       시험 2: SPA·Reality Check [중간] (3신호 816개가 다 끝난 뒤)
  python -m indexbeater stage4 --random    시험 4: 후보 A 랜덤 매수 1,000번
  python -m indexbeater stage4 --cost      시험 6: 상위 3개 0.25% 포트폴리오
  python -m indexbeater stage4 --window    시험 5: 상위 3개 W=2·W=4 (보고만)
  --overwrite                              이미 있는 결과 파일이 다를 때 덮어쓰기 (팀장 승인 뒤)

모든 시험: 2015~2019 OOS, 비용 0.5%, 시장별로 따로. 2020년 이후 자료는 열지 않는다.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from indexbeater import benchmark as B
from indexbeater import combo as CB
from indexbeater import config as C

COST_ROW = "0.5%"                                 # 점수표 '비용' 열 글자 (포트폴리오는 이 줄에만 있다)
SCORE_BAND = 1.0                                  # 5.8-2: 점수 차이 1점 이내면 포트폴리오 성적으로 고른다
YEAR_RET = [f"포트_수익_{y}" for y in C.WF_YEARS]
PORT_COLS = YEAR_RET + ["포트_MDD", "포트_마지막평가금액"]
SINGLE = C.RESULTS / "scorecard.csv"
SINGLE_DAILY = C.RESULTS / "scorecard_portfolio_daily.csv"
KINDS = ("단독", "2신호", "3신호20", "3신호816")
PART_KIND = {"2신호": "pairs", "3신호20": "triples", "3신호816": "triples"}

OUT_IMPROVED = C.RESULTS / "stage4_improved.csv"
OUT_TRIPLE_VS = C.RESULTS / "stage4_triple_vs_parts.csv"
OUT_POOL = C.RESULTS / "stage4_pool.csv"
OUT_TOP = C.RESULTS / "stage4_top3.csv"
OUT_TRIES = C.RESULTS / "stage4_tries.csv"
OUT_SPA = C.RESULTS / "stage4_spa.csv"
OUT_SPA_CELLS = C.RESULTS / "stage4_spa_cells.csv"
OUT_RANDOM = C.RESULTS / "stage4_random.csv"
OUT_RANDOM_TRIALS = C.RESULTS / "stage4_random_trials.csv"
OUT_COST = C.RESULTS / "stage4_cost.csv"
OUT_WINDOW = C.RESULTS / "stage4_window.csv"

# 11/25 확인 (노트북 07): 후보 0건 고치기(실험기록 21번째 줄) 전 코드로 만든 묶음 중, 그 시장 후보가 0건인데
# 다른 시장에 후보가 있어 엔진이 멈추지 않고 포트폴리오를 0.0 으로 적은 칸. 지금 코드는 빈칸으로 둔다.
# 읽을 때 빈칸으로 바꾼다 (거래 0건 = 300건 미만이라 순위·SPA 에는 원래 안 들어간다).
OLD_ZERO_CELLS = ({("2신호", "ichimoku+rsi", "KOSPI", h) for h in C.HOLDS}
                  | {("3신호", "alignment+stochastic+disparity", "KOSPI", h) for h in C.HOLDS}
                  | {("3신호", "rsi+disparity+squeeze", "KOSDAQ", h) for h in (5, 20)})


# ================================================================ 읽기
def components(row) -> list[str]:
    """칸의 구성 신호 (단독이면 자기 하나)."""
    if row["종류"] == "단독":
        return [row["신호"]]
    return [row[f"구성{i}"] for i in (1, 2, 3) if f"구성{i}" in row and isinstance(row[f"구성{i}"], str)]


def blank_old_zero(df: pd.DataFrame, kind_group: str) -> tuple[pd.DataFrame, set]:
    """거래 0건인데 포트폴리오가 0.0 으로 적힌 칸(옛 코드) → 포트폴리오·초과수익 열을 빈칸으로. 바꾼 칸 목록도 돌려준다."""
    df = df.copy()
    cols = [c for c in PORT_COLS if c in df.columns]
    if not cols:
        return df, set()
    flat = ((df["거래수"] == 0) & (df[YEAR_RET] == 0).all(axis=1) & (df["포트_MDD"] == 0))
    found = {(kind_group, n, mk, int(h)) for n, mk, h in df.loc[flat, ["신호", "시장", "H"]].itertuples(index=False)}
    if not found <= OLD_ZERO_CELLS:
        raise ValueError(f"알려진 8칸 말고도 포트폴리오 0.0 칸이 있다: {sorted(found - OLD_ZERO_CELLS)}")
    df.loc[flat, cols + [c for c in B.EXCESS_COLS if c in df.columns]] = np.nan
    return df, found


def read_triple_parts(combos) -> pd.DataFrame:
    """3신호 묶음 점수표를 바로 읽는다 (합친 파일 전에도 쓸 수 있게) + 지수 초과수익 열 (merge_triples 와 같은 계산)."""
    d = CB.PARTS / "triples"
    now = CB.engine_fingerprints()
    missing = [CB.combo_name(c) for c in combos if not CB.part_done(d / CB.combo_name(c), now)]
    if missing:
        raise SystemExit(f"✗ 아직 안 끝난 3신호 묶음 {len(missing)}개 (예: {missing[:3]})")
    df = pd.concat([pd.read_csv(d / CB.combo_name(c) / "scorecard.csv", encoding="utf-8-sig",
                                float_precision="round_trip") for c in combos], ignore_index=True)
    df = df.reindex(columns=list(dict.fromkeys(list(df.columns) + PORT_COLS)))   # 포트 열이 아예 없는 묶음 대비
    return B.add_excess(df, B.market_yearly(B.load_market()))


def load_cells(triples_all: bool = False, log=print) -> pd.DataFrame:
    """비용 0.5% 칸 전부: 단독 + 2신호 153개 + 시장별 고른 3신호 20개 (+ triples_all 이면 816개 비교용)."""
    single = pd.read_csv(SINGLE, encoding="utf-8-sig", float_precision="round_trip").assign(종류="단독", 신호수=1)
    pairs = pd.read_csv(CB.PAIRS_OUT, encoding="utf-8-sig", float_precision="round_trip").assign(종류="2신호")
    pairs, f2 = blank_old_zero(pairs, "2신호")
    picked = CB.picked_triples()
    chosen = sorted({c for cs in picked.values() for c in cs}, key=CB.all_triples().index)
    t = read_triple_parts(CB.all_triples() if triples_all else chosen)
    t, f3 = blank_old_zero(t, "3신호")
    names20 = {(mk, CB.combo_name(c)) for mk, cs in picked.items() for c in cs}
    t20 = t[[(m, n) in names20 for m, n in zip(t["시장"], t["신호"])]].assign(종류="3신호20")
    parts = [single, pairs, t20] + ([t.assign(종류="3신호816")] if triples_all else [])
    cells = pd.concat(parts, ignore_index=True)
    cells = cells[cells["비용"] == COST_ROW].reset_index(drop=True)
    log(f"  읽은 칸 (0.5%): " + ", ".join(f"{k} {int((cells['종류'] == k).sum())}" for k in KINDS if (cells['종류'] == k).any())
        + f" / 옛 코드 0.0 → 빈칸: {len(f2 | f3)}칸")
    return cells


# ================================================================ 시험 3: 개선 판정 (5.8, 실험기록 22 ②③)
def single_best(cells: pd.DataFrame, mk: str) -> pd.Series:
    """그 시장 단독 54칸 중 거래 300건 이상 칸의 점수 1위."""
    s = cells[(cells["시장"] == mk) & (cells["종류"] == "단독") & (cells["거래수"] >= C.MIN_TRADES)]
    if s.empty:
        raise ValueError(f"{mk}: 거래 300건 이상 단독 칸이 없다")
    return s.sort_values(["점수", "신호", "H"], ascending=[False, True, True]).iloc[0]


def judge_improved(cells: pd.DataFrame, kinds=("2신호", "3신호20")) -> pd.DataFrame:
    """조합 칸마다: 단독1위점수, 구성 신호 같은 H 점수, 개선 여부. 개선 = 300건 이상 & 점수 > 단독 1위 & 점수 > 구성 각각."""
    single = cells[cells["종류"] == "단독"].set_index(["시장", "신호", "H"])["점수"]
    out = []
    for mk in C.UNIVERSE_MARKETS:
        best = single_best(cells, mk)
        g = cells[(cells["시장"] == mk) & cells["종류"].isin(kinds)].copy()
        comp = [[float(single.loc[(mk, n, h)]) for n in components(r)] for r, h in zip(g.to_dict("records"), g["H"])]
        g["단독1위"] = f"{best['신호']} H={best['H']}"
        g["단독1위점수"] = float(best["점수"])
        g["구성점수"] = [",".join(f"{x:.2f}" for x in c) for c in comp]
        g["구성최고점수"] = [max(c) for c in comp]
        g["표본부족"] = g["거래수"] < C.MIN_TRADES
        g["개선"] = (~g["표본부족"]) & (g["점수"] > g["단독1위점수"]) & (g["점수"] > g["구성최고점수"])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def triple_vs_parts(judged: pd.DataFrame, kind: str = "3신호20") -> pd.DataFrame:
    """5.7-5 '3신호 조합 vs 그 구성 신호들': 3신호 점수가 구성 신호 셋의 같은 H 점수보다 모두 높은가 (300건 이상만)."""
    g = judged[judged["종류"] == kind].copy()
    g["구성보다나음"] = (~g["표본부족"]) & (g["점수"] > g["구성최고점수"])
    g["점수차_구성최고"] = g["점수"] - g["구성최고점수"]
    cols = ["시장", "신호", "H", "거래수", "점수", "구성점수", "구성최고점수", "점수차_구성최고", "표본부족", "구성보다나음",
            "단독1위점수", "개선"]
    return g[cols].reset_index(drop=True)


# ================================================================ 상위 3개 (5.8 선택 규칙, 실험기록 22 ④)
TIE_ORDER = [("포트_초과수익_중앙값", False), ("포트_초과음수해수", True), ("거래수", False), ("점수", False),
             ("신호", True), ("H", True)]


def pick_top(pool: pd.DataFrame, n: int = C.EXIT_STAGE_TOP_N, band: float = SCORE_BAND) -> pd.DataFrame:
    """남은 칸 중 최고 점수 s → 점수 ≥ s − 1 인 칸 중 TIE_ORDER 첫째 → 그 신호·조합의 다른 H 칸은 지움 → n 개까지."""
    left = pool.copy()
    picked = []
    while len(picked) < n and len(left):
        s = left["점수"].max()
        near = left[left["점수"] >= s - band]
        near = near.sort_values([c for c, _ in TIE_ORDER], ascending=[a for _, a in TIE_ORDER], na_position="last")
        top = near.iloc[0].copy()
        top["등수"] = len(picked) + 1
        top["그때최고점수"] = s
        top["1점안칸수"] = len(near)
        picked.append(top)
        left = left[left["신호"] != top["신호"]]
    out = pd.DataFrame(picked).reset_index(drop=True)
    out["등수"] = out["등수"].astype(int)
    return out


def compound(df: pd.DataFrame, prefix: str) -> pd.Series:
    return (1 + df[[f"{prefix}_{y}" for y in C.WF_YEARS]]).prod(axis=1) - 1


def add_index_compare(df: pd.DataFrame) -> pd.DataFrame:
    """시험 7: 5년 합친 수익 vs 지수, 이긴 해 수 (초과수익 > 0 인 해)."""
    df = df.copy()
    df["포트_5년수익"] = compound(df, "포트_수익")
    df["지수_5년수익"] = compound(df, "포트_지수수익")
    df["5년초과수익"] = df["포트_5년수익"] - df["지수_5년수익"]
    ex = df[[f"포트_초과수익_{y}" for y in C.WF_YEARS]]
    df["지수이긴해수"] = (ex > 0).sum(axis=1).where(ex.notna().all(axis=1)).astype("Int64")
    df["지수이김_5년"] = (df["5년초과수익"] > 0).where(ex.notna().all(axis=1))
    return df


SHOW = ["시장", "등수", "종류", "신호", "H", "점수", "거래수", "그때최고점수", "1점안칸수", "포트_초과수익_중앙값",
        "포트_초과음수해수", "포트_MDD", "포트_5년수익", "지수_5년수익", "5년초과수익", "지수이긴해수", "포트_지수대신쓴해",
        "표시", "일관성부족", "후보A"]


def rank(cells: pd.DataFrame, log=print) -> dict[str, pd.DataFrame]:
    judged = judge_improved(cells)
    single = cells[(cells["종류"] == "단독") & (cells["거래수"] >= C.MIN_TRADES)]
    tops, pools = [], []
    for mk in C.UNIVERSE_MARKETS:
        pool = pd.concat([single[single["시장"] == mk],
                          judged[(judged["시장"] == mk) & judged["개선"]][cells.columns]], ignore_index=True)
        top = pick_top(pool)
        top["일관성부족"] = top["포트_초과수익_중앙값"] < 0
        top["후보A"] = top["등수"] == 1
        tops.append(add_index_compare(top))
        p = add_index_compare(pool).sort_values(["점수", "신호", "H"], ascending=[False, True, True])
        p = p.merge(top[["신호", "H", "등수"]], on=["신호", "H"], how="left")
        pools.append(p)
        log(f"  {mk}: 후보 칸 {len(pool)} (단독 {int((pool['종류'] == '단독').sum())} + 개선 조합 "
            f"{int((pool['종류'] != '단독').sum())})")
    top = pd.concat(tops, ignore_index=True)
    pool = pd.concat(pools, ignore_index=True)
    keep = [c for c in ["시장", "종류", "신호", "H", "등수", "점수", "거래수", "표시", *YEAR_RET, "포트_MDD",
                        *[f"포트_초과수익_{y}" for y in C.WF_YEARS], "포트_초과수익_중앙값", "포트_초과음수해수",
                        "포트_5년수익", "지수_5년수익", "5년초과수익", "지수이긴해수", "포트_지수대신쓴해"] if c in pool.columns]
    return {"judged": judged, "triple_vs": triple_vs_parts(judged), "pool": pool[keep],
            "top": top[[c for c in SHOW if c in top.columns] + [c for c in top.columns if c not in SHOW]]}


# ================================================================ 시험 1: 시도 수표 (5.11, 실험기록 22 ⑤)
def tries_rows(mk: str, n_param: int, n_single: int, n_pair: int, seg_mk: pd.DataFrame,
               cells_mk: pd.DataFrame) -> list[dict]:
    """한 시장의 시도 수 줄 (파일을 읽지 않는 부분). 같은 계산은 한 번만 센다 (11/25 사용자 "1번").

    - 고른 20개 칸은 816개 칸 안에 그대로 들어 있다 → '겹침' 줄로 빼서 합계에 두 번 들어가지 않게 한다
    - 0.5% 칸 하나 = 비용 두 줄 (0.5%·0.25%)
    """
    key = lambda d: set(zip(d["신호"], d["H"].astype(int)))
    c20 = key(cells_mk[cells_mk["종류"] == "3신호20"])
    c816 = key(cells_mk[cells_mk["종류"] == "3신호816"])
    items = [("파라미터 시도 (세트 × H)", n_param, "5.5 연도별 순차 고르기 — 그 해 전 자료로만 고름"),
             ("단독 칸 (18 × H3 × 비용2)", n_single, "5.11"),
             ("2신호 칸 (153 × H3 × 비용2)", n_pair, "5.6"),
             ("3신호 고르기 토막 Edge (18 × 토막3 × H3)", len(seg_mk), "5.7-1~3, 실험기록 18"),
             ("3신호 고른 20개 칸 (20 × H3 × 비용2)", len(c20) * 2, "5.7-4"),
             ("3신호 816개 칸 (비교만, 816 × H3 × 비용2)", len(c816) * 2, "5.7-6, 실험기록 17·22"),
             ("겹침 빼기 (고른 20개 칸은 816개 칸 안에 있음)", -len(c20 & c816) * 2, "같은 계산을 두 번 세지 않는다 (11/25)"),
             ("팀원 시도", np.nan, "6단계(팀원과 비교)에서 더한다")]
    rows = [{"시장": mk, "항목": k, "수": v, "근거": why} for k, v, why in items]
    rows.append({"시장": mk, "항목": "합계 (팀장 몫)", "수": sum(v for _, v, _ in items if pd.notna(v)), "근거": ""})
    return rows


def tries_table(cells_all: pd.DataFrame) -> pd.DataFrame:
    from indexbeater import scorecard as SC
    from indexbeater import signals as S
    seg = pd.read_csv(CB.TRIPLE_SEG_OUT, encoding="utf-8-sig")
    n_param = sum(len(SC.param_sets(n)) for n in S.SIGNALS) * len(C.HOLDS)
    score_all = {"단독": pd.read_csv(SINGLE, encoding="utf-8-sig"), "2신호": pd.read_csv(CB.PAIRS_OUT, encoding="utf-8-sig")}
    rows = []
    for mk in C.UNIVERSE_MARKETS:
        rows += tries_rows(mk, n_param, int((score_all["단독"]["시장"] == mk).sum()),
                           int((score_all["2신호"]["시장"] == mk).sum()), seg[seg["시장"] == mk],
                           cells_all[cells_all["시장"] == mk])
    out = pd.DataFrame(rows)
    out["수"] = out["수"].astype("Int64")
    return out


# ================================================================ 시험 2: SPA·Reality Check (5.11, 실험기록 22 ⑥)
def stationary_indices(n: int, trials: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis·Romano 정상 부트스트랩 날짜 번호 (trials × n). 매일 1/block 확률로 새 출발점, 아니면 다음 날 (끝 → 처음)."""
    p = 1.0 / block
    idx = np.empty((trials, n), np.int64)
    idx[:, 0] = rng.integers(0, n, trials)
    new = rng.random((trials, n)) < p
    starts = rng.integers(0, n, (trials, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def boot_means(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """부트스트랩 평균 (trials × K) = 날짜별 뽑힌 횟수 표 @ X / n."""
    n = X.shape[0]
    W = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(float)
    return W @ X / n


def spa_test(X: np.ndarray, idx: np.ndarray) -> dict:
    """X = 날짜 × 칸 (일간 초과수익). Hansen(2005) SPA (consistent, 표준화) + White(2000) RC + 칸별 p값.

    ω̂ = 부트스트랩으로 잰 √n·평균의 표준편차. SPA 는 많이 나쁜 칸(평균 < −√(ω̂²/n · 2loglog n))을 0 으로 두고 가운데 맞춤.
    """
    n, K = X.shape
    rn = np.sqrt(n)
    dbar = X.mean(axis=0)
    mb = boot_means(X, idx)
    dev = rn * (mb - dbar)
    omega = np.sqrt((dev ** 2).mean(axis=0))
    if (omega <= 0).any():
        raise ValueError("부트스트랩 표준편차가 0 인 칸이 있다 (값이 하나뿐인 줄)")
    stat = rn * dbar / omega
    t_spa = max(0.0, float(stat.max()))
    gc = np.where(dbar >= -np.sqrt(omega ** 2 / n * 2 * np.log(np.log(n))), dbar, 0.0)
    t_b = np.maximum(0.0, (rn * (mb - gc) / omega).max(axis=1))
    v = float(rn * dbar.max())
    v_b = dev.max(axis=1)
    return {"n": n, "K": K, "dbar": dbar, "omega": omega, "t": stat,
            "SPA통계": t_spa, "SPA_p": float((t_b >= t_spa).mean()),
            "RC통계": v, "RC_p": float((v_b >= v).mean()),
            "칸별_p": (dev >= rn * dbar).mean(axis=0)}


def holm(p) -> np.ndarray:
    """Holm 보정 p값 (같은 순서로 돌려준다)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


def daily_returns(eq: pd.Series) -> pd.Series:
    """평가금액 → 일간 수익. 첫날 = 첫날 평가금액 / 시작 금액 − 1 (2015 전에는 신호를 켜지 않아 시작 금액 그대로)."""
    r = eq.pct_change()
    r.iloc[0] = eq.iloc[0] / float(C.INIT_CAPITAL) - 1
    return r


def load_daily(cells: pd.DataFrame, mk: str) -> pd.DataFrame:
    """포트폴리오가 있는 칸의 일간 평가금액 (날짜 × '신호|H')."""
    want = cells[(cells["시장"] == mk) & cells["포트_수익_2015"].notna()]
    frames = []
    sd = pd.read_csv(SINGLE_DAILY, encoding="utf-8-sig", parse_dates=["date"], float_precision="round_trip")
    frames.append(sd[sd["시장"] == mk])
    for kind in ("pairs", "triples"):
        names = sorted(set(want.loc[want["종류"].map(PART_KIND) == kind, "신호"]))
        for n in names:
            d = pd.read_parquet(CB.PARTS / kind / n / "portfolio_daily.parquet")
            frames.append(d[d["시장"] == mk])
    d = pd.concat(frames, ignore_index=True)
    d["칸"] = d["신호"] + "|H" + d["H"].astype(str)
    wide = d.pivot(index="date", columns="칸", values="equity")
    keys = list(dict.fromkeys(want["신호"] + "|H" + want["H"].astype(str)))
    miss = [k for k in keys if k not in wide.columns]
    if miss:
        raise ValueError(f"{mk}: 일별 평가금액이 없는 칸 {len(miss)}개 (예: {miss[:3]})")
    wide = wide[keys]
    if wide.isna().any().any():
        raise ValueError(f"{mk}: 날짜가 빠진 칸이 있다")
    return wide


def excess_daily(wide: pd.DataFrame, mk: str) -> pd.DataFrame:
    """일간 (포트폴리오 수익 − 그 시장 비교 지수 수익). 날짜가 하루라도 안 맞으면 멈춘다."""
    ms = B.load_market()
    ix = ms[(ms["market"] == mk) & (ms["date"].dt.year.isin(C.WF_YEARS))].set_index("date")["ret"]
    if not ix.index.equals(wide.index):
        raise ValueError(f"{mk}: 포트폴리오 날짜와 비교 지수 날짜가 다르다 ({len(wide)} vs {len(ix)})")
    r = wide.apply(daily_returns)
    return r.sub(ix, axis=0)


def run_spa(cells: pd.DataFrame, top: pd.DataFrame, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ, per = [], []
    for mk in C.UNIVERSE_MARKETS:
        t0 = time.time()
        X = excess_daily(load_daily(cells, mk), mk)
        idx = stationary_indices(len(X), C.BOOTSTRAP_TRIALS, C.BLOCK_DAYS, np.random.default_rng(C.SEED))
        r = spa_test(X.to_numpy(float), idx)
        kinds = cells[cells["시장"] == mk].assign(칸=lambda d: d["신호"] + "|H" + d["H"].astype(str))
        kinds = kinds.drop_duplicates("칸").set_index("칸")["종류"]
        pc = pd.DataFrame({"시장": mk, "칸": X.columns, "종류": kinds.reindex(X.columns).to_numpy(),
                           "일평균초과": r["dbar"], "표준화": r["t"], "칸별_p": r["칸별_p"]})
        tk = top[top["시장"] == mk].assign(칸=lambda d: d["신호"] + "|H" + d["H"].astype(str))
        sel = pc.set_index("칸").loc[tk["칸"]]
        pc["상위3_Holm_p"] = pc["칸"].map(dict(zip(tk["칸"], holm(sel["칸별_p"].to_numpy()))))
        pc["상위3등수"] = pc["칸"].map(dict(zip(tk["칸"], tk["등수"])))
        best = pc.loc[pc["표준화"].idxmax()]
        summ.append({"시장": mk, "칸수": r["K"], "날짜수": r["n"],
                     **{f"칸수_{k}": int((pc["종류"] == k).sum()) for k in KINDS if k != "3신호20"},
                     "SPA통계": r["SPA통계"], "SPA_p": r["SPA_p"], "RC통계": r["RC통계"], "RC_p": r["RC_p"],
                     "표준화1위칸": best["칸"], "표시": "중간 (매도 단계·ML 뒤 다시)"})
        per.append(pc)
        log(f"  {mk}: 칸 {r['K']} × 날짜 {r['n']} → SPA p={r['SPA_p']:.3f}, RC p={r['RC_p']:.3f} "
            f"({time.time() - t0:.0f}초)")
    return pd.DataFrame(summ), pd.concat(per, ignore_index=True)


# ================================================================ 칸 → 신호 표 → 포트폴리오 (시험 4·5·6)
def cell_entries(row, picks, cache, window: int = C.AND_WINDOW):
    """칸 하나의 해마다 세트와 매수 표 (단독 = 고른 세트의 신호, 조합 = 구성 신호 AND)."""
    names, mk, h = components(row), row["시장"], int(row["H"])
    ysets = {y: tuple(picks[(mk, n, h, y)] for n in names) for y in C.WF_YEARS}
    ents = {}
    for key in sorted(set(ysets.values())):
        es = [cache(n, k) for n, k in zip(names, key)]
        ents[key] = es[0] if len(es) == 1 else CB.and_entry(es, window=window)
    return ysets, ents


def year_signal(ctx, ysets, ents) -> pd.DataFrame:
    """scorecard._portfolio 와 같은 방법: 2015~2019 해마다 그 해 세트의 신호만 켠 표."""
    sig = np.zeros((len(ctx.dates), len(ctx.m.codes)), bool)
    for y in C.WF_YEARS:
        rows = ctx.years == y
        e = ents[ysets[y]].reindex(index=ctx.dates, columns=ctx.m.codes, fill_value=False).to_numpy(bool)
        sig[rows] = e[rows]
    return pd.DataFrame(sig, index=ctx.dates, columns=ctx.m.codes)


def yearly_out(eq: pd.Series) -> dict:
    """scorecard._portfolio 와 같은 연수익·MDD·마지막 평가금액 (eq = 2015 이후 평가금액)."""
    from indexbeater import metrics as M
    out = {}
    prev = float(C.INIT_CAPITAL)
    for y in C.WF_YEARS:
        e = eq[eq.index.year == y]
        out[f"포트_수익_{y}"] = float(e.iloc[-1] / prev - 1) if len(e) else np.nan
        prev = float(e.iloc[-1]) if len(e) else prev
    out["포트_MDD"] = M.max_drawdown(pd.concat([pd.Series([float(C.INIT_CAPITAL)]), eq.reset_index(drop=True)]))
    out["포트_마지막평가금액"] = float(eq.iloc[-1])
    return out


def portfolio_at_cost(sig: pd.DataFrame, ctx, mk: str, hold: int, cost: float = C.COST_SELECT):
    """scorecard._portfolio 를 비용만 바꿔 돌린다 (엔진 run_portfolio 의 cost 인자). 0.5% 면 _portfolio 와 같아야 한다."""
    from indexbeater import engine as E
    from indexbeater.exits import ExitRule
    daily, _ = E.run_portfolio(sig, ctx.data, ExitRule(hold=hold), market=mk, cost=cost)
    eq = daily["equity"]
    eq = eq[eq.index.year >= C.WF_YEARS[0]]
    return yearly_out(eq), eq


def same_as_saved(out: dict, row, tol: float = 1e-12) -> float:
    """다시 계산한 포트폴리오와 저장된 값의 가장 큰 차이 (상대). tol 넘으면 멈춘다."""
    worst = 0.0
    for c in PORT_COLS:
        a, b = out[c], float(row[c])
        worst = max(worst, abs(a - b) / max(1.0, abs(b)))
    if worst > tol:
        raise ValueError(f"{row['시장']} {row['신호']} H={row['H']}: 다시 계산한 포트폴리오가 저장값과 다르다 ({worst:.2e})")
    return worst


# ================================================================ 시험 4: 랜덤 매수 (5.8 STEP 4, 실험기록 22 ⑦)
def random_signals(sig: pd.DataFrame, ctx, mk: str, rng: np.random.Generator) -> pd.DataFrame:
    """A 후보가 있는 날마다, 그 시장에서 그날 살 수 있는 종목 중 A 후보 수만큼 무작위로 켠 표 (2015~2019)."""
    pool = ctx.allowed & (ctx.m.market == mk)
    rows = np.isin(ctx.years, C.WF_YEARS)
    cnt = (sig.to_numpy(bool) & pool).sum(axis=1)
    out = np.zeros_like(pool)
    for t in np.nonzero(rows & (cnt > 0))[0]:
        out[t, rng.choice(np.flatnonzero(pool[t]), int(cnt[t]), replace=False)] = True
    return pd.DataFrame(out, index=sig.index, columns=sig.columns)


def random_test(row, ctx, picks, cache, trials: int = C.RANDOM_TRIALS, seed: int = C.SEED, log=print):
    ysets, ents = cell_entries(row, picks, cache)
    return random_test_sig(year_signal(ctx, ysets, ents), row, ctx, trials, seed, log)


def random_test_sig(sig: pd.DataFrame, row, ctx, trials: int = C.RANDOM_TRIALS, seed: int = C.SEED, log=print):
    """random_test 의 계산 부분 (신호 표를 바로 받는다. classic.py 도 이걸 부른다)."""
    mk, h = row["시장"], int(row["H"])
    a_out, _ = portfolio_at_cost(sig, ctx, mk, h)
    diff = same_as_saved(a_out, row)
    log(f"  {mk} 후보 A = {row['신호']} H={h}: 다시 계산 = 저장값 (가장 큰 차이 {diff:.1e})")
    res = []
    t0 = time.time()
    for i in range(trials):
        rs = random_signals(sig, ctx, mk, np.random.default_rng([seed, i]))
        out, _ = portfolio_at_cost(rs, ctx, mk, h)
        res.append({"시장": mk, "번호": i, **{c: out[c] for c in YEAR_RET}, "포트_MDD": out["포트_MDD"]})
        if (i + 1) % 50 == 0:
            log(f"    … {i + 1}/{trials} ({time.time() - t0:.0f}초)")
    tr = pd.DataFrame(res)
    tr["포트_5년수익"] = compound(tr, "포트_수익")
    a5 = float(np.prod([1 + a_out[c] for c in YEAR_RET]) - 1)
    summ = []
    for c, a in [*((c, a_out[c]) for c in YEAR_RET), ("포트_5년수익", a5)]:
        r = tr[c].to_numpy(float)
        summ.append({"시장": mk, "후보A": f"{row['신호']} H={h}", "구간": c.replace("포트_수익_", "").replace("포트_", ""),
                     "A수익": a, "랜덤중앙값": float(np.median(r)), "A가이김": bool(a > np.median(r)),
                     "백분위": float((r < a).mean() * 100), "상위5%": bool((r >= a).mean() <= C.RANDOM_TOP_PCT)})
    s = pd.DataFrame(summ)
    wins = int(s.loc[s["구간"] != "5년수익", "A가이김"].sum())
    s["이긴해수"] = wins
    s["통과(4해이상)"] = wins >= 4
    return s, tr


# ================================================================ 시험 6: 0.25% (5.12 '비용 가정 의존', 실험기록 22 ⑨)
def index_yearly() -> pd.DataFrame:
    """시장 × 해 비교 지수 수익률 표."""
    my = B.market_yearly(B.load_market())
    return my.pivot(index="market", columns="year", values="수익률")


def cost_row(sig: pd.DataFrame, r, ctx, idx: pd.DataFrame, log=print) -> dict:
    """칸 하나의 0.5%·0.25% 비교 줄 (신호 표를 바로 받는다. classic.py 도 이걸 부른다)."""
    o5, _ = portfolio_at_cost(sig, ctx, r["시장"], int(r["H"]), C.COST_SELECT)
    same_as_saved(o5, r)
    o25, _ = portfolio_at_cost(sig, ctx, r["시장"], int(r["H"]), C.COST_ROUNDTRIP)
    ix5 = float(np.prod([1 + idx.loc[r["시장"], y] for y in C.WF_YEARS]) - 1)
    p5 = float(np.prod([1 + o5[c] for c in YEAR_RET]) - 1)
    p25 = float(np.prod([1 + o25[c] for c in YEAR_RET]) - 1)
    log(f"  {r['시장']} {r['등수']}위 {r['신호']} H={r['H']}: 5년 0.5% {p5:+.1%} / 0.25% {p25:+.1%} / 지수 {ix5:+.1%}")
    return {"시장": r["시장"], "등수": r["등수"], "신호": r["신호"], "H": r["H"],
            **{f"0.25%_{c}": o25[c] for c in PORT_COLS},
            "0.5%_5년수익": p5, "0.25%_5년수익": p25, "지수_5년수익": ix5,
            "0.5%_지수이김": p5 > ix5, "0.25%_지수이김": p25 > ix5,
            "비용가정의존": (p25 > ix5) and not (p5 > ix5)}


def cost_test(top: pd.DataFrame, ctx, picks, cache, log=print) -> pd.DataFrame:
    idx = index_yearly()
    rows = []
    for r in top.to_dict("records"):
        ysets, ents = cell_entries(r, picks, cache)
        rows.append(cost_row(year_signal(ctx, ysets, ents), r, ctx, idx, log))
    return pd.DataFrame(rows)


# ================================================================ 시험 5: 상위 3개 W=2·W=4 (실험기록 22 ⑧, 보고만)
WINDOW_TRY = (2, 4)
WINDOW_SHOW = ["점수", "거래수", "표시", *PORT_COLS, *[f"포트_초과수익_{y}" for y in C.WF_YEARS], "포트_초과수익_중앙값",
               "포트_5년수익", "지수_5년수익", "5년초과수익", "지수이긴해수"]


def window_test(top: pd.DataFrame, ctx, picks, flags, cache, log=print) -> pd.DataFrame:
    """조합 칸만 W=2·4 로 combo_tables 를 다시 돌린다 (단독은 W 가 없어 '상관없음'). 합격선 없음, 고르기에 안 씀.

    먼저 W=3 으로 다시 돌려 저장된 값(stage4_top3.csv)과 같은지 본다 — 다르면 멈춘다.
    """
    my = B.market_yearly(B.load_market())
    rows = []
    for r in top.to_dict("records"):
        base = {"시장": r["시장"], "등수": r["등수"], "종류": r["종류"], "신호": r["신호"], "H": r["H"]}
        if r["종류"] == "단독":
            rows.append({**base, "W": "상관없음 (단독)", **{c: r.get(c) for c in WINDOW_SHOW}})
            log(f"  {r['시장']} {r['등수']}위 {r['신호']} H={r['H']}: 단독 — W 상관없음")
            continue
        for w in (C.AND_WINDOW, *WINDOW_TRY):
            t = CB.combo_tables(components(r), ctx, picks, flags, cache, window=w, log=lambda *a, **k: None)
            sc = t["scorecard"]
            one = sc[(sc["시장"] == r["시장"]) & (sc["H"] == r["H"]) & (sc["비용"] == COST_ROW)]
            if len(one) != 1:
                raise ValueError(f"{r['신호']} W={w}: 칸이 {len(one)}줄")
            one = add_index_compare(B.add_excess(one.copy(), my)).iloc[0]
            if w == C.AND_WINDOW:
                diff = max([abs(float(one[c]) - float(r[c])) for c in ["점수", "거래수", *PORT_COLS]])
                if not diff <= 1e-12:
                    raise ValueError(f"{r['신호']} W=3 다시 계산 ≠ 저장값 (차이 {diff})")
                log(f"  {r['시장']} {r['등수']}위 {r['신호']} H={r['H']}: W=3 다시 계산 = 저장값 (차이 {diff:.1e})")
            rows.append({**base, "W": str(w), **{c: one.get(c) for c in WINDOW_SHOW}})
            log(f"    W={w}: 점수 {one['점수']:.1f} / 거래 {int(one['거래수'])}건 / 5년 {one['포트_5년수익']:+.1%} "
                f"(지수 {one['지수_5년수익']:+.1%}) / 지수 이긴 해 {one['지수이긴해수']}")
    return pd.DataFrame(rows)


# ================================================================ 명령
def _top_from_file() -> pd.DataFrame:
    if not OUT_TOP.exists():
        raise SystemExit(f"✗ {OUT_TOP.name} 이 없다 → 먼저 --rank")
    return pd.read_csv(OUT_TOP, encoding="utf-8-sig", float_precision="round_trip")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m indexbeater stage4", description=__doc__.split("\n")[0])
    for f in ("rank", "tries", "spa", "random", "cost", "window"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="결과 파일 덮어쓰기 (팀장 승인 뒤)")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.rank:
        r = rank(load_cells())
        for key, out in (("judged", OUT_IMPROVED), ("triple_vs", OUT_TRIPLE_VS), ("pool", OUT_POOL), ("top", OUT_TOP)):
            print(f"  {out.name}: {B._save(r[key], out, a.overwrite)}")
        tv = r["triple_vs"]
        for mk in C.UNIVERSE_MARKETS:
            j = r["judged"][r["judged"]["시장"] == mk]
            print(f"\n{mk}: 단독 1위 {j['단독1위'].iloc[0]} ({j['단독1위점수'].iloc[0]:.2f}점) / 개선 2신호 "
                  f"{int(j[j['종류'] == '2신호']['개선'].sum())}칸, 개선 3신호 {int(j[j['종류'] == '3신호20']['개선'].sum())}칸 / "
                  f"5.7-5 구성보다 나은 3신호 {int(tv[tv['시장'] == mk]['구성보다나음'].sum())}칸")
            print(r["top"][r["top"]["시장"] == mk][SHOW].to_string(index=False))
    if a.tries:
        t = tries_table(load_cells(triples_all=True))
        print(t.to_string(index=False))
        print(f"  {OUT_TRIES.name}: {B._save(t, OUT_TRIES, a.overwrite)}")
    if a.spa:
        s, p = run_spa(load_cells(triples_all=True), _top_from_file())
        print(s.to_string(index=False))
        print(f"  {OUT_SPA.name}: {B._save(s, OUT_SPA, a.overwrite)}")
        print(f"  {OUT_SPA_CELLS.name}: {B._save(p, OUT_SPA_CELLS, a.overwrite)}")
    if a.random or a.cost:
        top = _top_from_file()
        _, ctx, picks, _, cache = CB.prepare()
        if a.random:
            ss, ts = [], []
            for row in top[top["후보A"]].to_dict("records"):
                s, tr = random_test(row, ctx, picks, cache)
                print(s.to_string(index=False))
                ss.append(s)
                ts.append(tr)
            print(f"  {OUT_RANDOM.name}: {B._save(pd.concat(ss, ignore_index=True), OUT_RANDOM, a.overwrite)}")
            print(f"  {OUT_RANDOM_TRIALS.name}: "
                  f"{B._save(pd.concat(ts, ignore_index=True), OUT_RANDOM_TRIALS, a.overwrite)}")
        if a.cost:
            c = cost_test(top, ctx, picks, cache)
            print(f"  {OUT_COST.name}: {B._save(c, OUT_COST, a.overwrite)}")
    if a.window:
        top = _top_from_file()
        _, ctx, picks, flags, cache = CB.prepare()
        w = window_test(top, ctx, picks, flags, cache)
        print(f"  {OUT_WINDOW.name}: {B._save(w, OUT_WINDOW, a.overwrite)}")
    if not any(vars(a)[f] for f in ("rank", "tries", "spa", "random", "cost", "window")):
        ap.print_help()
    print(f"\n끝 ({time.time() - t0:.0f}초)")
    return 0
