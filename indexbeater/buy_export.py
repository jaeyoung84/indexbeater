"""TOP3 매수 신호 CSV — 팀원 매도 ML·DL 용 (실험기록 36번째 줄 규칙, 11/26 팀장)

11/26 회의 결정: 계획서 5.8·5.9 를 바꿔, 매도 최적화는 팀원이 각자 ML·DL 로 한다.
  학습은 확장 창 워크포워드 — 2011~2015 학습 → 2016 시험, 2011~2016 → 2017, … , 2011~2018 → 2019. 2020 이후 안 씀.
이 파일은 그때 쓸 매수 신호(TOP3, 시장마다 3개 = 6개)를 한 CSV 로 내보낸다. 새로 고르는 것은 없다.

TOP3 (팀원최적_비교_회의자료.pdf 순위 = results/team_compare_rank.csv, 모두 H5)
  코스피  1위 나 ⑥ 볼린저(20,2) 하단 + RSI(14) ≤ 30 · 2위 다-2 김동겸07 로지스틱 상위 30% · 4위 가 disparity
  코스닥  2위 다-2 김동겸07 · 3위 나 ⑥ · 4위 가 rsi+stochastic (W=3)

신호를 만든 방법
  2015~2019  PDF 와 같은 신호 (해마다 그 해 시작 21거래일 전까지의 자료로 고른 세트·모델) — team_compare 와 같은 함수
  2012~2014  '최종 모델' = 2020 에 쓸 것 (2020-01-01 전 21거래일 = 2019-11-29 까지로 고름)
             가 = param_selection.csv 의 2020 줄 세트, 나 = 고정 정의, 다-2 = 그 날까지·예정 매도일도 그 전인 매매로
             학습한 로지스틱으로 시장·날짜별 상위 30%. 2020 자료는 열지 않는다 (load_wide 기본값).
  ※ 누설: 2012~2014 신호는 2016~2019 를 본 세트·모델이 고른 것. 다-2 의 2012~2014 는 그 후보로 학습한 모델이 다시
     점수를 매김 (학습 안 자료). TOP3 도 2015~2019 결과로 골랐다 → 팀원 시험 해(2016~2019)와 겹친다.

넣는 것 (가격 OHLCV 는 넣지 않음): 20자리 제한 전에 켜진 신호 전부 + 엔진 매수 순서 값 + '실제로 샀다' 참고 열 2개.
엔진(engine.py 등)·점수표는 고치지 않는다.

실행 (설계 구간만)
  python -m indexbeater top3csv --run          → results/top3_buy_signals.csv (+ _check.csv, _07_final.csv)
  --overwrite                                  이미 있는 결과 파일이 다를 때 덮어쓰기 (팀장 승인 뒤)
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from indexbeater import config as C
from indexbeater import engine as E

OUT = C.RESULTS / "top3_buy_signals.csv"
OUT_CHECK = C.RESULTS / "top3_buy_signals_check.csv"
OUT_FINAL07 = C.RESULTS / "top3_buy_signals_07_final.csv"
DESKTOP_NAME = "TOP3_매수신호_2012-2019.csv"

HOLD = 5
FINAL_YEAR = 2020                          # 최종 모델 = 2020 에 쓸 것 (param_selection 의 2020 줄과 같은 cut)
TOP3 = {"KOSPI": [(1, "나"), (2, "다-2"), (4, "가")],
        "KOSDAQ": [(2, "다-2"), (3, "나"), (4, "가")]}
SRC_WF = "해마다(그해전까지로고름)"
SRC_FINAL = "최종모델(2019-11-29까지로고름)"
COLS = ["시장", "TOP3순서", "PDF순위", "구분", "전략", "H(고를때)", "신호일", "종목코드", "매수일", "매수가_수정시가",
        "매수상태", "갭취소", "매수순서값", "20일평균거래대금", "신호출처", "실제로샀다_2012부터", "실제로샀다_2015부터"]


# ================================================================ 대상 6칸
def top3_rows() -> pd.DataFrame:
    """team_compare_rank.csv 에서 TOP3 6칸을 꺼낸다. 순위·구분·H 가 회의 결정과 다르면 멈춘다."""
    from indexbeater import team_compare as T
    rk = pd.read_csv(T.OUT_RANK, encoding="utf-8-sig", float_precision="round_trip")
    out = []
    for mk, want in TOP3.items():
        for k, (rank, g) in enumerate(want, 1):
            r = rk[(rk["시장"] == mk) & (rk["등수"] == rank)]
            if len(r) != 1 or r["구분"].iloc[0] != g or int(r["H"].iloc[0]) != HOLD:
                raise SystemExit(f"✗ {mk} {rank}위가 회의 결정({g} H{HOLD})과 다르다: {r[['구분', '신호', 'H']].to_dict('records')}")
            out.append({"시장": mk, "TOP3순서": k, "PDF순위": rank, "구분": g, "전략": r["신호"].iloc[0]})
    return pd.DataFrame(out)


def final_cut(dates: pd.DatetimeIndex) -> tuple[int, int]:
    """최종 모델의 start(= 2020 첫날 자리) 와 cut (= start − 21). 데이터에 2020 이후가 있으면 멈춘다."""
    start = int(dates.searchsorted(pd.Timestamp(FINAL_YEAR, 1, 1)))
    if start != len(dates):
        raise SystemExit(f"✗ 데이터에 {FINAL_YEAR} 년 이후가 있다 — 설계 구간만 써야 한다")
    return start, start - C.WF_CUT_OFFSET


# ================================================================ 최종 모델 (2012~2014 신호)
def final_entry(row: dict, picks, cache) -> tuple[tuple, pd.DataFrame]:
    """가: param_selection 2020 줄의 세트로 만든 매수 표 (조합이면 W=3 AND, stage4.cell_entries 와 같은 방법)."""
    from indexbeater import combo as CB
    from indexbeater import stage4 as S4
    names, mk, h = S4.components(row), row["시장"], int(row["H"])
    key = tuple(picks[(mk, n, h, FINAL_YEAR)] for n in names)
    es = [cache(n, k) for n, k in zip(names, key)]
    return key, (es[0] if len(es) == 1 else CB.and_entry(es, window=C.AND_WINDOW))


def final_07(ctx, h: int = HOLD, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """다-2 최종 모델: cut(2019-11-29) 까지의 후보 중 2020 첫날 전에 팔았고 예정 매도일도 그 전인 매매로 학습
    (team_compare.orders_07 의 한 해와 같은 방법, start = 2020 첫날) → 2012~2014 후보를 시장·날짜별 상위 30%."""
    from indexbeater import team_compare as T
    from indexbeater.exits import ExitRule
    t0 = time.time()
    d, dates = ctx.data, ctx.dates
    start, cut = final_cut(dates)
    bands = T.band_rules(d)
    mask = bands["03"] | bands["04"] | bands["05"]
    cands = T.candidates(d, mask)
    rule = ExitRule(hold=h)
    early = ctx.years < C.WF_YEARS[0]
    learn = T.independent_labels(mask & (np.arange(len(dates)) <= cut)[:, None], d, rule, ctx.m)
    train = cands.join(T.keep_learn(learn, start, h), how="inner")
    truth = T.independent_labels(mask & early[:, None], d, rule, ctx.m)
    hold_out = cands.loc[early[cands["t_idx"].to_numpy()]]
    order = np.zeros(mask.shape, dtype="int16")
    folds = []
    for mk in C.UNIVERSE_MARKETS:
        tr = train.loc[train["market"].eq(mk)]
        chosen, diag = T.fit_07(tr, hold_out.loc[hold_out["market"].eq(mk)], truth)
        order[chosen["t_idx"].to_numpy(), chosen["j"].to_numpy()] = chosen["priority"].to_numpy()
        folds.append({"시장": mk, "H": h, "학습cut": dates[cut].date().isoformat(),
                      "학습마지막신호일": dates[int(tr["t_idx"].max())].date().isoformat(),
                      "점수매긴구간": f"{dates[int(hold_out['t_idx'].min())].date()}~{dates[early].max().date()}",
                      **diag, "주의": "점수매긴 후보가 학습에도 들어감(학습 안 자료)"})
    log(f"  다-2 최종 모델 학습 끝 (cut {dates[cut].date()}, {time.time() - t0:.0f}초)")
    return pd.DataFrame(order, index=dates, columns=ctx.m.codes), pd.DataFrame(folds)


# ================================================================ 6칸 매수 순서 값 표
def splice(wf: pd.DataFrame, fin: pd.DataFrame, ctx, mk: str) -> tuple[pd.DataFrame, int]:
    """2012~2014 줄 = 최종 모델 표(fin), 2015~2019 줄 = 해마다 표(wf). 그 시장 종목만 남기고,
    엔진 simulate_trades 도 버리는 칸 (유동성 필터 tradable · 2012-07-01 전) 은 0 → 엔진이 보는 신호만 남는다."""
    early = ctx.years < C.WF_YEARS[0]
    a = wf.reindex(index=ctx.dates, columns=ctx.m.codes, fill_value=0).fillna(0).to_numpy().astype(np.int64)
    b = fin.reindex(index=ctx.dates, columns=ctx.m.codes, fill_value=0).fillna(0).to_numpy().astype(np.int64)
    v = np.where(early[:, None], b, a) * (ctx.m.market == mk)
    drop = int(((v != 0) & ~ctx.allowed).sum())
    v[~ctx.allowed] = 0
    if (v < 0).any():
        raise SystemExit(f"✗ {mk}: 매수 순서 값이 0 보다 작다")
    return pd.DataFrame(v, index=ctx.dates, columns=ctx.m.codes), drop


def value_tables(ctx, picks, flags, cache, log=print) -> tuple[dict, pd.DataFrame]:
    """(시장, 구분) → 날짜×종목 int 표 (0 = 신호 없음, 가·나 = 1, 다-2 = 우선순위) 와 출처 표 (True = 최종 모델)."""
    from indexbeater import classic as K
    from indexbeater import close_exit as CE
    from indexbeater import stage4 as S4
    from indexbeater import team_compare as T

    def join(wf: pd.DataFrame, fin: pd.DataFrame, mk: str, g: str) -> pd.DataFrame:
        v, drop = splice(wf, fin, ctx, mk)
        log(f"  {mk} {g}: 엔진이 안 보는 칸(유동성 필터·2012-07 전) {drop:,}개 뺌")
        return v

    out, notes = {}, []
    want = top3_rows()
    for row in CE.targets().to_dict("records"):
        mk = row["시장"]
        g = "가" if row["구분"] == "기존1등" else "나"
        ysets, ents, _ = CE.cell_inputs(row, ctx, picks, flags, cache)
        wf = S4.year_signal(ctx, ysets, ents)
        if g == "나":
            key, fin = "고정", K.entry(6, ctx.data)
        else:
            key, fin = final_entry(row, picks, cache)
        out[(mk, g)] = join(wf, fin, mk, g)
        notes.append({"시장": mk, "구분": g, "전략": row["신호"], "해마다세트": str(ysets), "최종모델세트": str(key)})
    o7, _ = T.orders_07(ctx, log=log, holds=(HOLD,))
    f7, folds = final_07(ctx, log=log)
    for mk in C.UNIVERSE_MARKETS:
        out[(mk, "다-2")] = join(o7[HOLD], f7, mk, "다-2")
    got = {(r["시장"], r["구분"]) for r in want.to_dict("records")}
    if not got <= set(out):
        raise SystemExit(f"✗ 만들지 못한 칸: {sorted(got - set(out))}")
    log("  세트: " + " / ".join(f"{n['시장']} {n['구분']} 해마다 {n['해마다세트']} 최종 {n['최종모델세트']}" for n in notes))
    return {k: out[k] for k in sorted(got)}, folds


def bought_set(val: pd.DataFrame, ctx, mk: str, from_year: int | None = None):
    """엔진 run_portfolio (H5·0.5%·1억·20자리) 로 실제로 산 (t_idx, 종목) 집합과 평가금액 (from_year 부터만 신호를 켬)."""
    from indexbeater.exits import ExitRule
    if from_year is None:
        v = val
    else:
        on = np.broadcast_to(np.asarray(ctx.years >= from_year)[:, None], val.shape)
        v = pd.DataFrame(np.where(on, val.to_numpy(), 0), index=val.index, columns=val.columns)
    daily, log = E.run_portfolio(v, ctx.data, ExitRule(hold=HOLD), market=mk, cost=C.COST_SELECT)
    got = set() if log.empty else set(map(tuple, log.loc[log["notional"] > 0, ["t_idx", "code"]].itertuples(index=False)))
    return got, daily["equity"]


def rows_for(val: pd.DataFrame, ctx, meta: dict, log=print) -> pd.DataFrame:
    """칸 하나의 신호 줄 (켜진 칸 전부) + 매수 상태 (엔진 simulate_trades) + 실제로 샀다 2개."""
    from indexbeater.exits import ExitRule
    m, dates, n = ctx.m, ctx.dates, ctx.m.n
    v = val.to_numpy()
    t, j = np.nonzero(v > 0)
    tr = E.simulate_trades(val > 0, ctx.data, ExitRule(hold=HOLD), dedupe=False, _m=m)
    tr = tr.set_index(["t_idx", "code"])
    codes = np.asarray(m.codes)[j]
    key = pd.MultiIndex.from_arrays([t, codes])
    if len(tr) != len(t) or not tr.index.is_unique:
        raise SystemExit(f"✗ {meta['시장']} {meta['구분']}: 엔진 후보 수 {len(tr)} ≠ 신호 칸 {len(t)}")
    st = tr.reindex(key)
    b = t + 1
    has_b = b < n
    bb = np.minimum(b, n - 1)
    open_px = np.where(has_b, m.O[bb, j], np.nan)
    filled = st["status"].isin([E.FILLED, E.OPEN_END]).to_numpy() & st["buy_px"].notna().to_numpy()
    if not np.allclose(open_px[filled], st["buy_px"].to_numpy(float)[filled], rtol=0, atol=0):
        raise SystemExit("✗ 매수가(T+1 수정 시가) ≠ 엔진 매수가")
    all_b, _ = bought_set(val, ctx, meta["시장"])
    b15, _ = bought_set(val, ctx, meta["시장"], from_year=C.WF_YEARS[0])
    pairs = list(zip(t.tolist(), codes.tolist()))
    early = ctx.years[t] < C.WF_YEARS[0]
    df = pd.DataFrame({
        **{k: meta[k] for k in ("시장", "TOP3순서", "PDF순위", "구분", "전략")}, "H(고를때)": HOLD,
        "신호일": dates[t].strftime("%Y-%m-%d"), "종목코드": codes,
        "매수일": np.where(has_b, dates[bb].strftime("%Y-%m-%d"), ""), "매수가_수정시가": open_px,
        "매수상태": st["status"].to_numpy(), "갭취소": st["status"].eq(E.GAP_CANCEL).to_numpy(),
        "매수순서값": v[t, j], "20일평균거래대금": m.value20[t, j],
        "신호출처": np.where(early, SRC_FINAL, SRC_WF),
        "실제로샀다_2012부터": [p in all_b for p in pairs],
        "실제로샀다_2015부터": pd.array([None if e else (p in b15) for p, e in zip(pairs, early)], dtype="boolean"),
    })
    log(f"  {meta['시장']} {meta['구분']} {meta['전략']}: 신호 {len(df):,}줄 (2012~2014 {int(early.sum()):,}) · "
        f"샀다 2012부터 {len(all_b):,} · 2015부터 {len(b15):,}")
    return df


def table(ctx, picks, flags, cache, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    vals, folds = value_tables(ctx, picks, flags, cache, log=log)
    want = top3_rows()
    parts = [rows_for(vals[(r["시장"], r["구분"])], ctx, r, log=log) for r in want.to_dict("records")]
    df = pd.concat(parts, ignore_index=True)[COLS]
    df = df.sort_values(["시장", "TOP3순서", "신호일", "매수순서값", "20일평균거래대금", "종목코드"],
                        ascending=[False, True, True, False, False, True], kind="mergesort").reset_index(drop=True)
    return df, folds


# ================================================================ 확인: CSV 만 다시 읽어 엔진에 넣으면 PDF 포트폴리오와 같은가
def read_back(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), dtype={"종목코드": str}, float_precision="round_trip")


def check(df_csv: pd.DataFrame, ctx, tol: float = 1e-9, log=print) -> pd.DataFrame:
    """CSV 의 2015~2019 줄 → 날짜×종목 매수 순서 값 표 → run_portfolio(H5·0.5%) → team_compare 저장 평가금액과 비교."""
    from indexbeater import stage4 as S4
    from indexbeater import team_compare as T
    saved = pd.read_csv(T.OUT_DAILY, encoding="utf-8-sig", parse_dates=["date"])
    saved = saved[(saved["H"] == HOLD) & (saved["비용"] == S4.COST_ROW)]
    out = []
    for (mk, g), part in df_csv.groupby(["시장", "구분"], sort=False):
        p = part[pd.to_datetime(part["신호일"]).dt.year >= C.WF_YEARS[0]]
        v = p.pivot(index="신호일", columns="종목코드", values="매수순서값")
        v.index = pd.to_datetime(v.index)
        v = v.reindex(index=ctx.dates, columns=ctx.m.codes).fillna(0).astype(int)
        _, eq = bought_set(v, ctx, mk)
        eq = eq[eq.index.year >= C.WF_YEARS[0]]
        ref = saved[(saved["시장"] == mk) & (saved["구분"] == g)].set_index("date")["equity"]
        same_days = len(ref) == len(eq) and (ref.index == eq.index).all()
        diff = float((eq.to_numpy() / ref.to_numpy() - 1).__abs__().max()) if same_days else np.inf
        row = {"시장": mk, "구분": g, "전략": part["전략"].iloc[0], "2015~2019신호줄": len(p), "날짜수": len(eq),
               "마지막평가금액_CSV": float(eq.iloc[-1]), "마지막평가금액_PDF": float(ref.iloc[-1]) if len(ref) else np.nan,
               "가장큰상대차이": diff, "통과": diff <= tol}
        out.append(row)
        log(f"  {'✓' if row['통과'] else '✗'} {mk} {g} CSV→엔진 = PDF 포트폴리오 (가장 큰 차이 {diff:.1e})")
    res = pd.DataFrame(out)
    if not res["통과"].all():
        raise SystemExit("✗ CSV 로 다시 돌린 포트폴리오가 PDF 와 다르다 → CSV 를 저장하지 않는다")
    return res


def run(log=print):
    from indexbeater import combo as CB
    t0 = time.time()
    _, ctx, picks, flags, cache = CB.prepare(log=log)
    final_cut(ctx.dates)
    df, folds = table(ctx, picks, flags, cache, log=log)
    text = df.to_csv(index=False, lineterminator="\n")
    back = read_back(text)
    chk = check(back, ctx, log=log)
    log(f"  신호 {len(df):,}줄 ({time.time() - t0:.0f}초)")
    return df, chk, folds


def main(argv=None):
    import argparse
    from indexbeater import benchmark as B
    ap = argparse.ArgumentParser(prog="python -m indexbeater top3csv", description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true", help="TOP3 매수 신호 CSV 만들기 → results/top3_buy_signals.csv")
    ap.add_argument("--overwrite", action="store_true", help="결과 파일 덮어쓰기 (팀장 승인 뒤)")
    a = ap.parse_args(argv)
    if not a.run:
        ap.print_help()
        return 0
    t0 = time.time()
    pd.set_option("display.width", 250)
    df, chk, folds = run()
    print(chk.to_string(index=False))
    print(folds.to_string(index=False))
    print(df.groupby(["시장", "TOP3순서", "구분", "전략", "신호출처"]).size().to_string())
    for d, path in ((df, OUT), (chk, OUT_CHECK), (folds, OUT_FINAL07)):
        print(f"  {path.name}: {B._save(d, path, a.overwrite)}")
    print(f"\n끝 ({time.time() - t0:.0f}초)")
    return 0
