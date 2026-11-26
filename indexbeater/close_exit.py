"""다음 날 종가 매도 비교 — 참고용 별개 시험 (실험기록 32번째 줄, 11/26 팀장)

성격: 4단계 상위 3개·매도 규칙 단계·시험 8 에 넣지 않는다. 랜덤 매수 1,000번도 하지 않는다.
      '5거래일 뒤 시가 매도(지금 규칙)' 와 '다음 날 종가 매도' 를 같은 매수 신호로 나란히 놓고 매도 시점 차이만 본다.

엔진(engine.py·signals.py·exits.py·indicators.py·data_io.py·config.py)과 점수표(scorecard.py·metrics.py·
scoring.py)는 고치지 않는다. 엔진은 '정해진 날 시가 매도'만 할 수 있어서, 매매 한 건을 따라가는 부분과
포트폴리오 하루 돌리기만 이 파일에 엔진과 똑같이 옮겨 적고 매도 가격 한 군데(시가 → 종가)만 바꿨다.
옮겨 적은 것이 맞는지: mode="open", k=5 로 돌리면 엔진·저장된 결과와 끝자리까지 같아야 한다 (테스트 + 실행 첫 단계).

대상 4칸 (매수 신호·해마다 세트는 H5 칸 그대로, 다시 고르지 않음)
  코스피 기존 1등 disparity H5 · 코스닥 기존 1등 rsi+stochastic H5 (W=3) · ⑥ 볼린저 하단+RSI≤30 H5 두 시장

매도 방식
  open  (지금 규칙) : T일 신호 → T+1 시가 매수 → 매수일 + k 거래일 (k=5) 시가 매도
  close (비교)      : T일 신호 → T+1 시가 매수 → 매수일 + k 거래일 (k=1, 매수 다음 날) 종가 매도
  공통: 그날 못 팔면 (거래정지·가격 없음·시가가 하한가 근처 = 엔진 _Market.sellable 그대로) 팔 수 있는 첫 날 같은 시점
        (시가/종가). 팔기 전에 상장폐지 = 마지막 종가 정산. 상한가 갭 매수 취소·중복 매수 금지는 엔진 그대로.
점수: 점수표 함수(scorecard._score_rows) 그대로. H 자리에는 '매수 날부터 판 날까지 거래일 수' = open 5, close 1.
      (Edge20 = Edge × 20 / H 이라 close 는 Edge × 20 → 수익점수가 커지기 쉽다. Edge 원값도 같이 본다.)
대조군: 같은 날 유동성 통과 모든 종목-일을 같은 매도 방식으로 사고판 짝 칸 평균 (metrics.ControlBase 와 같은 계산).
포트폴리오: 1억·20자리·5%, 비용 0.5% · 0.25%. 종가 매도는 그날 시가 매수 뒤에 하므로 그날 자리는 차 있고,
      판 돈은 장 마감 뒤 현금 (엔진 run_portfolio 의 3)·4) 단계 그대로).

실행 (설계 구간만)
  python -m indexbeater closeexit --run          점검(open H5 = 저장값) → 4칸 × 2방식 → results/close_exit_*.csv
  --overwrite                                   이미 있는 결과 파일이 다를 때 덮어쓰기 (팀장 승인 뒤)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from indexbeater import config as C
from indexbeater import engine as E
from indexbeater import metrics as M

OPEN, CLOSE = "open", "close"
MODES = {OPEN: ("5거래일뒤시가", 5), CLOSE: ("다음날종가", 1)}   # 이름, k (매수일 + k 거래일에 판다)
CLOSE_REASON = "종가"

OUT_COMPARE = C.RESULTS / "close_exit_compare.csv"
OUT_DAILY = C.RESULTS / "close_exit_portfolio_daily.csv"
OUT_CHECK = C.RESULTS / "close_exit_check.csv"

NOT_CHECKED = {"시장", "신호", "H", "비용"}          # 점검에서 뺄 열 (H 는 점수용H 로 바뀌므로 따로 본다)


# ================================================================ 매매 한 건 (엔진 _one_trade 의 S3 부분과 같은 규칙)
def trade_arrays(m: E._Market, t, j, mode: str, k: int) -> dict:
    """신호 칸 (t, j) 여러 개를 한 번에 따라간다. 돌려주는 값은 엔진 매매 기록과 같은 열 (배열)."""
    if mode not in MODES:
        raise ValueError(f"매도 방식은 {list(MODES)} 중 하나: {mode}")
    n = m.n
    t = np.asarray(t, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    N = len(t)
    b = t + 1
    status = np.full(N, E.FILLED, dtype=object)
    reason = np.full(N, None, dtype=object)
    buy = np.full(N, np.nan)
    sell = np.full(N, np.nan)
    s_idx = np.full(N, -1, dtype=np.int64)
    delay = np.zeros(N, dtype=np.int64)
    at_open = np.zeros(N, dtype=bool)

    end = b >= n                                              # 내일 자료가 없다 → 미완료
    bb = np.minimum(b, n - 1)
    no_buy = ~end & ~m.traded[bb, j]                          # 거래정지·상장폐지·가격 없음 → 못 산다
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = m.O[bb, j] / m.Cl[t, j] - 1
    gap_cancel = ~end & ~no_buy & (gap >= m.thr[bb] - E.FLOAT_EPS)
    ok = ~end & ~no_buy & ~gap_cancel
    buy[ok] = m.O[bb[ok], j[ok]]

    s0 = b + k
    s = m.nxt[np.minimum(s0, n), j]                           # s0 일 또는 그 뒤 팔 수 있는 첫 날 (없으면 n)
    sold = ok & (s < n)
    ss = s[sold]
    sell[sold] = (m.O if mode == OPEN else m.Cl)[ss, j[sold]]
    s_idx[sold] = ss
    delay[sold] = ss - s0[sold]
    reason[sold] = "S3" if mode == OPEN else CLOSE_REASON
    at_open[sold] = mode == OPEN

    rest = ok & (s >= n)                                      # 팔 날을 못 찾음
    last = m.last[j]
    dl = rest & (last < n - 1)                                # 상장폐지 → 마지막 종가 정산
    s_idx[dl] = last[dl]
    sell[dl] = m.Cl[last[dl], j[dl]]
    reason[dl] = E.DELIST
    delay[dl] = np.maximum(last[dl] - s0[dl], 0)
    status[rest & ~dl] = E.OPEN_END                           # 산 뒤 자료 끝까지 못 팖 (buy_px 는 남긴다)
    status[end] = E.OPEN_END
    status[no_buy] = E.NO_BUY
    status[gap_cancel] = E.GAP_CANCEL
    return {"t_idx": t, "b_idx": b, "s_idx": s_idx, "buy_px": buy, "sell_px": sell, "reason": reason,
            "delay": delay, "sell_at_open": at_open, "status": status}


def simulate(entry: pd.DataFrame, data: dict, mode: str, k: int, dedupe: bool = True,
             m: E._Market | None = None) -> pd.DataFrame:
    """engine.simulate_trades 와 같은 매매 표 (mode="open", k=5 면 ExitRule(hold=5) 결과와 같다)."""
    m = m or E._Market(data)
    e = entry.reindex(index=m.dates, columns=m.codes, fill_value=False).fillna(False).astype(bool)
    keep = data["tradable"].to_numpy(bool) & np.asarray(m.dates >= C.SIGNAL_START)[:, None]
    ti, ji = np.nonzero(e.to_numpy(bool) & keep)
    if not len(ti):
        return pd.DataFrame(columns=E.TRADE_COLS)
    order = np.lexsort((ti, ji))                              # 종목별 → 날짜 순서 (엔진과 같음)
    ti, ji = ti[order], ji[order]
    rec = trade_arrays(m, ti, ji, mode, k)
    if dedupe:
        held_until = np.full(len(m.codes), -1)
        dup = np.zeros(len(ti), bool)
        for q in range(len(ti)):
            t, j = ti[q], ji[q]
            if held_until[j] > t:                             # t일 장 마감 때 아직 보유 중 → 무시
                dup[q] = True
            elif rec["status"][q] == E.FILLED:
                held_until[j] = rec["s_idx"][q]
            elif rec["status"][q] == E.OPEN_END and not np.isnan(rec["buy_px"][q]):
                held_until[j] = m.n
        rec["b_idx"] = np.where(dup, -1, rec["b_idx"])
        rec["s_idx"] = np.where(dup, -1, rec["s_idx"])
        rec["buy_px"] = np.where(dup, np.nan, rec["buy_px"])
        rec["sell_px"] = np.where(dup, np.nan, rec["sell_px"])
        rec["delay"] = np.where(dup, 0, rec["delay"])
        rec["sell_at_open"] = np.where(dup, False, rec["sell_at_open"])
        rec["reason"] = np.where(dup, None, rec["reason"])
        rec["status"] = np.where(dup, E.DUPLICATE, rec["status"])
    rec["code"] = m.codes.to_numpy()[ji]
    rec["market"] = m.market[ti, ji]
    return E._to_table(rec, m)                               # 열 이름 → 배열 사전 (pd.DataFrame 으로 바로 표가 된다)


# ================================================================ 대조군 바탕 (metrics.ControlBase 와 같은 계산)
class Control(M.ControlBase):
    """유동성 통과 모든 종목-일을 같은 매도 방식으로 사고판 결과. 짝 칸 평균(matched)은 ControlBase 것을 그대로 쓴다."""

    def __init__(self, data: dict, m: E._Market, cell: np.ndarray, mode: str, k: int):
        self.hold = k
        shape = cell.shape
        allowed = data["tradable"].to_numpy(bool) & np.asarray(m.dates >= C.SIGNAL_START)[:, None]
        t, j = np.nonzero(allowed)
        r = trade_arrays(m, t, j, mode, k)
        f = r["status"] == E.FILLED
        t, j, s = t[f], j[f], r["s_idx"][f]
        with np.errstate(invalid="ignore", divide="ignore"):
            g = r["sell_px"][f] / r["buy_px"][f] - 1
        self.gross = np.full(shape, np.nan)
        self.gross[t, j] = g
        self.delist = np.zeros(shape, bool)
        self.delist[t, j] = r["reason"][f] == E.DELIST
        late = s > t + 1 + k
        self._late = (t[late], j[late], s[late])
        self.cell = cell
        self.n_cells = shape[0] * M.CELLS_PER_DAY
        self.n_base_trades = int(f.sum())
        self._tot = {w: self._sums(self.variant(w), np.ones(shape, bool)) for w in (False, True)}


def same_control(a: M.ControlBase, b: M.ControlBase) -> bool:
    """두 대조군 바탕이 같은가 (수익·폐지 표시·칸별 합계 모두, 소수 끝자리까지)."""
    if not (np.array_equal(a.gross, b.gross, equal_nan=True) and np.array_equal(a.delist, b.delist)
            and a.n_base_trades == b.n_base_trades):
        return False
    return all(np.array_equal(a._tot[w][key], b._tot[w][key]) for w in (False, True) for key in a._tot[w])


# ================================================================ 점수표 줄 (scorecard._trade_table 의 OOS 부분)
def trade_table(ctx, entry: pd.DataFrame, mode: str, k: int, base: M.ControlBase):
    """체결 매매 표(대조군 붙임)·상태 표. scorecard._trade_table 와 같은 계산 (학습 매매는 안 만든다)."""
    from indexbeater import scorecard as SC
    tr = simulate(entry, ctx.data, mode, k, dedupe=True, m=ctx.m)
    st = tr.loc[tr["status"].isin([E.FILLED, E.GAP_CANCEL, E.NO_BUY]), ["market", "t_idx", "status"]].copy()
    st["year"] = ctx.years[st["t_idx"].to_numpy(int)]
    f = tr[tr["status"] == E.FILLED]
    t = f["t_idx"].to_numpy(int)
    j = pd.Index(ctx.m.codes).get_indexer(f["code"])
    sig_on = entry.reindex(index=ctx.dates, columns=ctx.m.codes, fill_value=False).to_numpy(bool) & ctx.allowed
    SC.check_finite(f["ret_gross"].to_numpy(float), "신호 매매")
    out = pd.DataFrame({"market": f["market"].to_numpy(), "code": f["code"].to_numpy(),
                        "t_idx": t, "year": ctx.years[t], "gross": f["ret_gross"].to_numpy(float)})
    out["gross_w"] = np.where(f["reason"].eq(E.DELIST).to_numpy(), -1.0, out["gross"])
    for worst, sfx in ((False, ""), (True, "_w")):
        mt = base.matched(sig_on, t, j, worst=worst)
        out[f"ctrl_gross{sfx}"] = mt["ctrl_gross"]
        for name in M.COSTS:
            out[f"ctrl_win{name}{sfx}"] = mt[f"ctrl_win{name}"]
    out["edge"] = out["gross"] - out["ctrl_gross"]
    out["edge_w"] = out["gross_w"] - out["ctrl_gross_w"]
    # 비교용 덧붙임 (점수 계산에는 안 씀)
    out["delay"] = f["delay"].to_numpy(int)
    out["delist"] = f["reason"].eq(E.DELIST).to_numpy()
    out["held_days"] = f["s_idx"].to_numpy(int) - f["b_idx"].to_numpy(int)
    return out, st


def score_cell(ctx, mk: str, ysets: dict, ents: dict, mode: str, base: M.ControlBase, name: str, flags: list):
    """칸 하나 × 매도 방식 하나 → 점수표 2줄 (비용 0.5%·0.25%) + OOS 매매 표. 해마다 그 해 세트의 매매만 잇는다."""
    from indexbeater import scorecard as SC
    k = MODES[mode][1]
    made = {key: trade_table(ctx, ents[key], mode, k, base) for key in sorted(set(ysets.values()))}
    parts, sparts = [], []
    for y in C.WF_YEARS:
        ft, st = made[ysets[y]]
        parts.append(ft[(ft["market"] == mk) & (ft["year"] == y)])
        sparts.append(st[(st["market"] == mk) & (st["year"] == y)])
    oos = pd.concat(parts, ignore_index=True)
    st_oos = pd.concat(sparts, ignore_index=True)
    rows = SC._score_rows(name, mk, k, oos, st_oos, flags, ctx)
    return rows, oos


# ================================================================ 포트폴리오 (engine.run_portfolio 와 같은 규칙, 매매만 이 파일 것)
def run_portfolio(signal_count: pd.DataFrame, data: dict, market: str, mode: str, k: int,
                  capital: float = C.INIT_CAPITAL, max_positions: int = C.MAX_POSITIONS,
                  weight: float = C.WEIGHT_PER_POSITION, cost: float = C.COST_SELECT, m: E._Market | None = None):
    """engine.run_portfolio 를 그대로 옮겨 적고 후보 매매만 simulate(mode, k) 로 바꿨다. 돌려주는 값도 같다."""
    m = m or E._Market(data)
    cnt = signal_count.reindex(index=m.dates, columns=m.codes, fill_value=0).fillna(0).astype(int)
    cand = simulate(cnt > 0, data, mode, k, dedupe=False, m=m)
    cand = cand[cand["market"] == market].copy()
    cnt_np = cnt.to_numpy()
    jpos = {c: q for q, c in enumerate(m.codes)}
    cand["j"] = cand["code"].map(jpos)
    cand["n_sig"] = cnt_np[cand["t_idx"].to_numpy(int), cand["j"].to_numpy(int)]
    cand["v20"] = m.value20[cand["t_idx"].to_numpy(int), cand["j"].to_numpy(int)]
    cand = cand.sort_values(["b_idx", "n_sig", "v20", "code"], ascending=[True, False, False, True])
    by_day = {b: g for b, g in cand.groupby("b_idx", sort=False)}

    close_ff = pd.DataFrame(m.Cl).ffill().to_numpy()
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
        # 3) 장중·종가 매도 (종가 매도·상장폐지 정산은 여기)
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


def portfolio_cell(ctx, sig: pd.DataFrame, mk: str, mode: str, cost: float):
    """stage4.portfolio_at_cost 와 같은 출력 (연수익·MDD·마지막 평가금액, 2015 이후 평가금액)."""
    from indexbeater import stage4 as S4
    daily, _ = run_portfolio(sig, ctx.data, mk, mode, MODES[mode][1], cost=cost, m=ctx.m)
    from indexbeater import scorecard as SC
    eq = daily["equity"]
    SC.check_finite(eq.to_numpy(float), f"포트폴리오 {mk} {mode}")
    eq = eq[eq.index.year >= C.WF_YEARS[0]]
    return S4.yearly_out(eq), eq, daily.loc[eq.index, "n_pos"]


# ================================================================ 실제 실행 (4칸)
def targets() -> pd.DataFrame:
    """대상 4칸: stage4_top3 의 시장별 1등 (disparity H5 · rsi+stochastic H5) + classic_top3 의 ⑥ H5 두 시장."""
    from indexbeater import classic as K
    from indexbeater import stage4 as S4
    old = pd.read_csv(S4.OUT_TOP, encoding="utf-8-sig", float_precision="round_trip")
    old = old[old["등수"] == 1].assign(구분="기존1등")
    cl = pd.read_csv(K.OUT_TOP, encoding="utf-8-sig", float_precision="round_trip")
    cl = cl[(cl["신호"] == K.NAMES[6]) & (cl["H"] == 5)].assign(구분="대표조합⑥")
    want = {("KOSPI", "disparity"), ("KOSDAQ", "rsi+stochastic"), ("KOSPI", K.NAMES[6]), ("KOSDAQ", K.NAMES[6])}
    df = pd.concat([old, cl], ignore_index=True)
    got = set(zip(df["시장"], df["신호"]))
    if got != want or len(df) != 4 or (df["H"] != 5).any():
        raise SystemExit(f"✗ 대상 4칸이 실험기록 32번째 줄과 다르다: {sorted(got)}")
    return df


def cell_inputs(row, ctx, picks, flags, cache):
    """칸의 해마다 세트·매수 표·표시 (stage4·classic 과 같은 방법)."""
    from indexbeater import classic as K
    from indexbeater import stage4 as S4
    if row["구분"] == "대표조합⑥":
        return {y: 0 for y in C.WF_YEARS}, {0: K.entry(6, ctx.data)}, []
    ysets, ents = S4.cell_entries(row, picks, cache)
    fl = list(dict.fromkeys(x for n in S4.components(row) for x in flags[(row["시장"], n, int(row["H"]))]))
    return ysets, ents, fl


def _diff(a, b) -> float:
    blank = lambda x: x is None or x == "" or (isinstance(x, float) and np.isnan(x))
    if blank(a) and blank(b):                                 # 빈 글자 "" = CSV 로 다시 읽은 빈칸(NaN) — 둘 다 '표시 없음'
        return 0.0
    if isinstance(b, str) or isinstance(a, str):
        return 0.0 if str(a) == str(b) else np.inf
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return 0.0
    return abs(a - b) / max(1.0, abs(b))


def extra_cols(oos: pd.DataFrame) -> dict:
    return {"미룬매매수": int((oos["delay"] > 0).sum()), "상장폐지정산수": int(oos["delist"].sum()),
            "평균보유거래일": float(oos["held_days"].mean()) if len(oos) else np.nan,
            "Edge_원값_비용전": float(np.nanmean(oos["edge"].to_numpy())) if len(oos) else np.nan,
            "평균수익_비용전": float(oos["gross"].mean()) if len(oos) else np.nan}


def run(log=print, tol: float = 1e-12):
    """점검(open H5 = 저장값, 대조군 = ctx.base[5]) 을 먼저 하고, 통과하면 4칸 × 2방식을 계산한다."""
    from indexbeater import benchmark as B
    from indexbeater import combo as CB
    from indexbeater import scorecard as SC
    from indexbeater import stage4 as S4
    t0 = time.time()
    _, ctx, picks, flags, cache = CB.prepare(log=log)
    rows_t = targets()
    base = {OPEN: Control(ctx.data, ctx.m, ctx.cell, OPEN, MODES[OPEN][1])}
    if not same_control(base[OPEN], ctx.base[5]):
        raise SystemExit("✗ 옮겨 적은 대조군(5거래일 뒤 시가) ≠ 점수표 대조군 H5 → 종가 매도는 돌리지 않는다")
    log(f"  ✓ 대조군 5거래일 뒤 시가 = 점수표 대조군 H5 ({time.time() - t0:.0f}초)")
    base[CLOSE] = Control(ctx.data, ctx.m, ctx.cell, CLOSE, MODES[CLOSE][1])
    SC.check_finite(base[CLOSE].gross[~np.isnan(base[CLOSE].gross)], "대조군 바탕 다음날종가")
    log(f"  대조군 다음날종가: 체결 {base[CLOSE].n_base_trades:,}건 ({time.time() - t0:.0f}초)")

    out, daily, checks = [], [], []
    for row in rows_t.to_dict("records"):
        mk = row["시장"]
        ysets, ents, fl = cell_inputs(row, ctx, picks, flags, cache)
        sig = S4.year_signal(ctx, ysets, ents)
        for mode in (OPEN, CLOSE):
            label, k = MODES[mode]
            sc_rows, oos = score_cell(ctx, mk, ysets, ents, mode, base[mode], row["신호"], fl)
            ex = extra_cols(oos)
            for r in sc_rows:
                score_keys = [c for c in r if c not in NOT_CHECKED]
                cost = C.COST_SELECT if r["비용"] == S4.COST_ROW else C.COST_ROUNDTRIP
                pr, eq, npos = portfolio_cell(ctx, sig, mk, mode, cost)
                r.update(pr)
                r.update({"구분": row["구분"], "종류": row["종류"], "H": 5, "매도방식": label, "점수용H": k, **ex})
                daily.append(pd.DataFrame({"date": eq.index, "시장": mk, "구분": row["구분"], "신호": row["신호"],
                                           "매도방식": label, "비용": r["비용"], "equity": eq.to_numpy(),
                                           "n_pos": npos.to_numpy()}))
                if mode == OPEN and r["비용"] == S4.COST_ROW:     # 점검: 저장된 H5 0.5% 줄과 같아야 한다
                    cols = [c for c in score_keys + S4.PORT_COLS if c in row]
                    d = {c: _diff(r[c], row[c]) for c in cols}
                    worst = max(d.values())
                    checks.append({"시장": mk, "구분": row["구분"], "신호": row["신호"], "비교열수": len(cols),
                                   "가장큰차이": worst, "다른열": ",".join(c for c, v in d.items() if v > tol),
                                   "통과": worst <= tol})
                    if worst > tol:
                        raise SystemExit(f"✗ {mk} {row['신호']}: 옮겨 적은 계산(5거래일 뒤 시가) ≠ 저장된 H5 결과 "
                                         f"({checks[-1]['다른열']}, {worst:.2e}) → 종가 매도는 믿을 수 없다")
                    log(f"  ✓ {mk} {row['신호']} 5거래일뒤시가 = 저장된 H5 ({len(cols)}열, 가장 큰 차이 {worst:.1e})")
                out.append(r)
            log(f"  {mk} {row['신호']} {label}: 거래 {len(oos):,}건 ({time.time() - t0:.0f}초)")
    df = pd.DataFrame(out)
    df = B.add_excess(df, B.market_yearly(B.load_market()))
    df = S4.add_index_compare(df)
    first = ["시장", "구분", "종류", "신호", "H", "매도방식", "점수용H", "비용", "점수", "Edge", "Edge20", "Edge_원값_비용전",
             "평균수익_비용전", "평균수익", "승률", "거래수", "미룬매매수", "상장폐지정산수", "평균보유거래일",
             "포트_5년수익", "지수_5년수익", "5년초과수익", "지수이긴해수", "포트_MDD", "포트_마지막평가금액"]
    df = df[first + [c for c in df.columns if c not in first]]
    return df, pd.concat(daily, ignore_index=True), pd.DataFrame(checks)


def main(argv=None):
    import argparse
    from indexbeater import benchmark as B
    ap = argparse.ArgumentParser(prog="python -m indexbeater closeexit", description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true", help="점검 → 4칸 × 매도 방식 2개 계산 → results/close_exit_*.csv")
    ap.add_argument("--overwrite", action="store_true", help="결과 파일 덮어쓰기 (팀장 승인 뒤)")
    a = ap.parse_args(argv)
    if not a.run:
        ap.print_help()
        return 0
    t0 = time.time()
    df, daily, checks = run()
    show = ["시장", "구분", "신호", "매도방식", "비용", "점수", "Edge", "평균수익", "승률", "거래수", "미룬매매수",
            "포트_5년수익", "지수_5년수익", "포트_MDD"]
    print(df[show].to_string(index=False))
    print(checks.to_string(index=False))
    for d, path in ((df, OUT_COMPARE), (daily, OUT_DAILY), (checks, OUT_CHECK)):
        print(f"  {path.name}: {B._save(d, path, a.overwrite)}")
    print(f"\n끝 ({time.time() - t0:.0f}초)")
    return 0
