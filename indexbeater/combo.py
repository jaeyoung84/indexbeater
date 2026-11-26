"""신호 조합 (AND) — 2신호 153개·3신호 20개 (docs/프로젝트_기준.md 5.6·5.7·5.8).

엔진(engine.py·signals.py·exits.py·indicators.py·data_io.py·config.py)과 점수표(scorecard.py·metrics.py·
scoring.py)는 고치지 않고 부르기만 한다. 그래서 지문은 그대로다.

규칙 (실행 전에 실험기록.csv 에 적음, 11/25 — 결과를 보고 바꾸지 않는다)
- AND 날짜 규칙 (5.6): T일에 한 신호가 켜지고, 다른 신호가 최근 W거래일(T−W+1 ~ T) 안에 켜졌으면
  조합 신호가 T일에 켜진다. W = config.AND_WINDOW = 3, 탐색하지 않는다.
  신호가 3개면: T일에 하나라도 켜지고, 세 신호 모두 T−2 ~ T 안에 켜졌어야 한다 (5.7 "AND 규칙은 5.6과 같고").
  2개일 때 이 식은 위 문장과 똑같다: (A_T 그리고 최근B) 또는 (B_T 그리고 최근A).
- '켜졌다' = signals.entry_signal 이 True (이벤트형, 2012-07-01 이후). 유동성 필터는 엔진이 T일 조합 신호에 건다.
- 구성 신호 파라미터 (5.6): 5.5에서 그 시장·그 H·그 해로 고른 세트 (results/param_selection.csv) 그대로.

실행
  python -m indexbeater combo --self-check rsi   # 점검: rsi+rsi 조합 = 단독 rsi 점수표 (같아야 한다)
  python -m indexbeater combo --pairs            # 2신호 153개 (끝난 묶음은 건너뜀 — 중간에 멈춰도 이어서)
  python -m indexbeater combo --pairs --only rsi+macd
  python -m indexbeater combo --merge-pairs      # 153개 묶음 → results/combo_pairs.csv (+ 지수 초과수익 열)
  python -m indexbeater combo --triple-pick      # 3신호: 토막 3개 순위 → 시장별 상위 6개 → 20개 (5.7-1~3)
  python -m indexbeater combo --triples          # 3신호: 시장별 고른 20개 (두 시장 합집합)
  python -m indexbeater combo --triples-all      # 3신호: 나머지까지 816개 전부 (비교·시도 수·SPA 용, 고르는 데 안 씀)
  python -m indexbeater combo --merge-triples    # 816개 묶음 → results/combo_triples.csv

계산 방법은 단독 점수표(scorecard.run_signal)와 똑같다 (같은 함수를 부른다):
  그 해 고른 세트 묶음의 조합 신호로 설계 구간 전체를 매매 → 그 시장·그 해 매매만 OOS 로 이어 붙인다.
  포트폴리오는 해마다 그 해 조합 신호만 켠 표로 1억·20종목·5%, 비용 0.5% (0.25%는 상위 후보만, 5.6).
  매도는 H일 시간 청산만 (구성 신호 매도 규칙은 6단계, 5.9②).
구성 신호의 '표본 부족·불안정·기본값 사용' 표시는 조합 줄에 그대로 옮긴다 (그 파라미터를 쓰므로).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from indexbeater import config as C

SEP = "+"                                       # 조합 이름: "rsi+macd"
PARTS = C.RESULTS / "combo_parts"
SELECTION = C.RESULTS / "param_selection.csv"
SCORE_META = C.RESULTS / "scorecard_meta.json"
PAIRS_OUT = C.RESULTS / "combo_pairs.csv"
TRIPLE_SEG_OUT = C.RESULTS / "combo_triple_segments.csv"   # 토막별 학습 Edge (3신호 고르기 근거)
TRIPLE_RANK_OUT = C.RESULTS / "combo_triple_rank.csv"      # 시장별 18개 순위 → 상위 6개
TRIPLES_OUT = C.RESULTS / "combo_triples.csv"              # 816개 점수표 (+ '고른20' 열)


# ================================================================ AND 합치기
def recent(on: np.ndarray, window: int = C.AND_WINDOW) -> np.ndarray:
    """recent[T] = T−window+1 ~ T 중 하루라도 켜졌다 (날짜 = 0번 축). 미래(T+1 이후)는 보지 않는다."""
    out = on.copy()
    for lag in range(1, window):
        out[lag:] |= on[:-lag]
    return out


def and_entry(entries: list[pd.DataFrame], window: int = C.AND_WINDOW) -> pd.DataFrame:
    """구성 신호 매수 표 여러 개 → AND 조합 매수 표 (5.6 날짜 규칙)."""
    if len(entries) < 2:
        raise ValueError("조합은 신호 2개 이상")
    idx, cols = entries[0].index, entries[0].columns
    for e in entries[1:]:
        if not (e.index.equals(idx) and e.columns.equals(cols)):
            raise ValueError("구성 신호 표의 날짜·종목이 서로 다르다")
    arrs = [e.to_numpy(bool) for e in entries]
    today = np.logical_or.reduce(arrs)
    every = np.logical_and.reduce([recent(a, window) for a in arrs])
    return pd.DataFrame(today & every, index=idx, columns=cols)


def combo_name(names) -> str:
    return SEP.join(names)


def pair_list(order=None) -> list[tuple[str, str]]:
    """C(18,2) = 153개 전수 (5.6). 순서 = 부록 A 번호 순."""
    from indexbeater import signals as S
    return list(itertools.combinations(order or list(S.SIGNALS), 2))


# ================================================================ 구성 신호 파라미터 (5.5 결과)
def load_picks(path=SELECTION):
    """param_selection.csv → picks[(시장, 신호, H, 연도)] = 세트 번호, flags[(시장, 신호, H)] = 표시 목록.

    표시는 scorecard.run_signal 과 같은 규칙: 표본 부족·기본값 사용 = 2015~2019 중 한 해라도, 불안정 = 그 칸 전체.
    파라미터 글자가 지금 그리드(scorecard.param_sets)와 다르면 멈춘다.
    """
    from indexbeater import scorecard as SC
    sel = pd.read_csv(path, encoding="utf-8-sig")
    picks, flags = {}, {}
    for r in sel.to_dict("records"):
        k = int(r["고른세트"])
        if json.loads(r["파라미터"]) != json.loads(json.dumps(SC.param_sets(r["신호"])[k], ensure_ascii=False)):
            raise ValueError(f"{path}: {r['신호']} 세트 {k} 파라미터가 지금 그리드와 다르다")
        picks[(r["시장"], r["신호"], int(r["H"]), int(r["검증연도"]))] = k
    for (mk, name, h), g in sel.groupby(["시장", "신호", "H"]):
        wf = g[g["검증연도"].isin(C.WF_YEARS)]
        flags[(mk, name, int(h))] = ((["표본 부족"] if wf["표본부족"].any() else []) +
                                     (["불안정"] if g["불안정"].any() else []) +
                                     (["기본값 사용"] if wf["기본값사용"].any() else []))
    return picks, flags


class EntryCache:
    """(신호, 세트) 매수 표를 한 번만 만든다."""

    def __init__(self, data):
        self.data, self._c = data, {}

    def __call__(self, name, k):
        from indexbeater import scorecard as SC
        from indexbeater import signals as S
        if (name, k) not in self._c:
            self._c[(name, k)] = S.entry_signal(name, self.data, SC.param_sets(name)[k])
        return self._c[(name, k)]


# ================================================================ 조합 하나 계산
def combo_tables(names, ctx, picks, flags, cache, portfolio=True, log=print,
                 window: int = C.AND_WINDOW) -> dict[str, pd.DataFrame]:
    """조합 하나 (신호 이름 2~3개) → 점수표 줄·OOS 매매·포트폴리오 일별 (scorecard.run_signal 과 같은 방법).

    window = AND 날짜 규칙의 W. 기본 3 (config.AND_WINDOW) 만 쓰고, 4단계 시험 5(W=2·4, 보고만)에서만 바꾼다.
    """
    from indexbeater import scorecard as SC
    t0 = time.time()
    names = tuple(names)
    cname = combo_name(names)
    scores, oos_all, port = {}, {}, {}           # (시장, H) → 나중에 단독 점수표와 같은 순서(시장 → H)로
    no_port = []                                 # 포트폴리오 후보 0건인 (시장, H)
    n_sim = 0
    for h in C.HOLDS:
        keys = {(mk, y): tuple(picks[(mk, n, h, y)] for n in names)
                for mk in C.UNIVERSE_MARKETS for y in SC.SELECT_YEARS}
        made = {}
        for key in sorted({keys[(mk, y)] for mk in C.UNIVERSE_MARKETS for y in C.WF_YEARS}):
            ent = and_entry([cache(n, k) for n, k in zip(names, key)], window=window)
            ft, st, _ = SC._trade_table(ctx, ent, h)
            made[key] = (ent, ft, st)
            n_sim += 1
        for mk in C.UNIVERSE_MARKETS:
            parts, sparts = [], []
            for y in C.WF_YEARS:
                key = keys[(mk, y)]
                _, ft, st = made[key]
                parts.append(ft[(ft["market"] == mk) & (ft["year"] == y)].assign(세트=",".join(map(str, key))))
                sparts.append(st[(st["market"] == mk) & (st["year"] == y)])
            oos = pd.concat(parts, ignore_index=True)
            st_oos = pd.concat(sparts, ignore_index=True)
            fl = list(dict.fromkeys(x for n in names for x in flags[(mk, n, h)]))
            rows = SC._score_rows(cname, mk, h, oos, st_oos, fl, ctx)
            extra = {"신호수": len(names), **{f"구성{i + 1}": n for i, n in enumerate(names)}}
            for y in SC.SELECT_YEARS:
                extra[f"구성세트_{y}"] = ",".join(map(str, keys[(mk, y)]))
            extra["2020용세트"] = json.dumps({n: SC.param_sets(n)[k] for n, k in zip(names, keys[(mk, 2020)])},
                                          ensure_ascii=False)
            for r in rows:
                r.update(extra)
            if portfolio:
                ents = {k: v[0] for k, v in made.items()}
                ysets = {y: keys[(mk, y)] for y in C.WF_YEARS}
                if has_portfolio_candidate(ctx, mk, ysets, ents):
                    pr, pdaily = SC._portfolio(cname, mk, h, {y: {"set": s} for y, s in ysets.items()}, ents, ctx)
                    rows[0].update(pr)         # 포트폴리오는 0.5% 줄에만 (0.25%는 상위 후보만 — 5.6)
                    port[(mk, h)] = pdaily
                else:                          # 살 후보 0건 → 엔진을 부르지 않고 포트폴리오 칸은 비운다 (11/25)
                    no_port.append(f"{mk} H={h}")
            scores[(mk, h)] = rows
            oos_all[(mk, h)] = (oos.assign(신호=cname, H=h, signal_date=ctx.dates[oos["t_idx"].to_numpy(int)]))
    log(f"  {cname}: 매매 계산 {n_sim}번 ({time.time() - t0:.0f}초)")
    order = [(mk, h) for mk in C.UNIVERSE_MARKETS for h in C.HOLDS]
    return {"scorecard": pd.DataFrame([r for o in order for r in scores[o]]),
            "oos_trades": pd.concat([oos_all[o] for o in order], ignore_index=True),
            "portfolio_daily": pd.concat([port[o] for o in order if o in port], ignore_index=True)
            if port else pd.DataFrame(),
            "no_portfolio": no_port}


def has_portfolio_candidate(ctx, mk, year_sets, entries) -> bool:
    """SC._portfolio 가 엔진에 넘길 신호 표에 이 시장의 살 후보가 하나라도 있는가.

    엔진(simulate_trades)은 '신호 켜짐 & 그날 tradable & 2012-07-01 이후' 칸마다 후보를 하나 만들고 그날 시장으로 거른다.
    후보가 0건이면 엔진(고정 파일)이 빈 표에서 멈추므로, 같은 조건을 미리 본다. 계산 결과는 바꾸지 않는다.
    """
    for y in C.WF_YEARS:
        rows = ctx.years == y
        e = entries[year_sets[y]].reindex(index=ctx.dates, columns=ctx.m.codes, fill_value=False).to_numpy(bool)
        if (e[rows] & ctx.allowed[rows] & (ctx.m.market[rows] == mk)).any():
            return True
    return False


# ================================================================ 3신호: 상위 6개 고르기 (5.7-1~3, 11/25 팀장 결정)
# 토막 구간 = 2015년용 최초 학습 구간 (신호 시작 ~ cut 2014-12-01, 597거래일) 을 거래일 수로 3등분.
# 매매·대조군 = 5.5 학습 규칙 그대로 (scorecard._train_trades 와 같은 계산, 매매 날짜만 남김).
# 순위 = 시장별로 한 번: 신호마다 (토막 3 × H 3) 순위 9개 → 중앙값 작은 순 → 동점이면 9개 평균 → 부록 A 번호 순.
FIRST_YEAR = C.WF_YEARS[0]                      # 2015


def segment_bounds(ctx, n_seg: int = C.TRIPLE_SEGMENTS, year: int = FIRST_YEAR) -> list[tuple[int, int]]:
    """학습 구간(신호 시작 ~ cut) 날짜 번호를 n_seg 개 연속 토막으로 (처음, 끝 — 끝 포함)."""
    days = np.nonzero((ctx.dates >= C.SIGNAL_START) & (np.arange(len(ctx.dates)) <= ctx.cut_idx[year]))[0]
    return [(int(p[0]), int(p[-1])) for p in np.array_split(days, n_seg)]


def train_trades(ctx, entry: pd.DataFrame, hold: int, year: int = FIRST_YEAR) -> pd.DataFrame:
    """year 용 학습 매매 (market, t_idx, edge). scorecard._trade_table → _train_trades 와 같은 계산이다."""
    from indexbeater import engine as E
    from indexbeater.exits import ExitRule
    tr = E.simulate_trades(entry, ctx.data, ExitRule(hold=hold), dedupe=True, _m=ctx.m)
    f = tr[tr["status"] == E.FILLED]
    t = f["t_idx"].to_numpy(int)
    s = f["s_idx"].to_numpy(int)
    j = pd.Index(ctx.m.codes).get_indexer(f["code"])
    sig_on = entry.reindex(index=ctx.dates, columns=ctx.m.codes, fill_value=False).to_numpy(bool) & ctx.allowed
    base = ctx.base[hold]
    mt = base.matched(sig_on, t, j, worst=False)
    ci, start = ctx.cut_idx[year], ctx.start_idx[year]
    w = (t <= ci) & (s <= start)
    ctrl = base.matched_train(sig_on, t[w], j[w], {k: v[w] for k, v in mt.items()}, ci, start)
    return pd.DataFrame({"market": f["market"].to_numpy()[w], "t_idx": t[w],
                         "edge": f["ret_gross"].to_numpy(float)[w] - ctrl})


def segment_table(ctx, picks, cache, names=None, year: int = FIRST_YEAR, log=print) -> pd.DataFrame:
    """신호 × 시장 × H × 토막 → 학습 Edge·거래 수. '전체' 줄 = 세 토막 합친 값 (param_selection 학습Edge 와 같아야 함)."""
    from indexbeater import signals as S
    names = list(names or S.SIGNALS)
    bounds = segment_bounds(ctx, year=year)
    rows = []
    t0 = time.time()
    for name in names:
        for h in C.HOLDS:
            done = {}
            for mk in C.UNIVERSE_MARKETS:
                k = picks[(mk, name, h, year)]
                if k not in done:
                    done[k] = train_trades(ctx, cache(name, k), h, year)
                tt = done[k]
                tt = tt[(tt["market"] == mk) & ~np.isnan(tt["edge"].to_numpy())]
                segs = [("전체", tt)] + [(i + 1, tt[(tt["t_idx"] >= a) & (tt["t_idx"] <= b)])
                                        for i, (a, b) in enumerate(bounds)]
                for seg, g in segs:
                    rows.append({"시장": mk, "신호": name, "H": h, "세트": k, "토막": seg,
                                 "처음": ctx.dates[bounds[0][0] if seg == "전체" else bounds[seg - 1][0]].date().isoformat(),
                                 "끝": ctx.dates[bounds[-1][1] if seg == "전체" else bounds[seg - 1][1]].date().isoformat(),
                                 "거래수": len(g), "Edge": float(g["edge"].mean()) if len(g) else np.nan})
        log(f"  {name} ({time.time() - t0:.0f}초)")
    return pd.DataFrame(rows)


def rank_top(seg: pd.DataFrame, top_n: int = C.TRIPLE_TOP_N) -> pd.DataFrame:
    """시장마다: (토막, H) 칸마다 18개 순위 (Edge 큰 쪽 1위, Edge 없음 = 꼴찌) → 순위 9개 중앙값·평균 → 상위 top_n."""
    from indexbeater import signals as S
    order = {n: i for i, n in enumerate(S.SIGNALS)}
    g = seg[seg["토막"] != "전체"].copy()
    n_sig = g["신호"].nunique()
    g["순위"] = (g.groupby(["시장", "토막", "H"])["Edge"].rank(ascending=False, method="average")
                 .fillna(float(n_sig)))
    out = (g.groupby(["시장", "신호"])["순위"].agg(순위중앙값="median", 순위평균="mean", 순위수="size")
           .reset_index())
    out["번호"] = out["신호"].map(order)
    out = out.sort_values(["시장", "순위중앙값", "순위평균", "번호"]).reset_index(drop=True)
    out["등수"] = out.groupby("시장").cumcount() + 1
    out["상위"] = out["등수"] <= top_n
    return out


def triple_list(top: pd.DataFrame, mk: str) -> list[tuple[str, str, str]]:
    """그 시장 상위 6개 → C(6,3)=20개 (이름은 부록 A 번호 순)."""
    t = top[(top["시장"] == mk) & top["상위"]].sort_values("번호")["신호"].tolist()
    return list(itertools.combinations(t, 3))


def all_triples() -> list[tuple[str, str, str]]:
    """C(18,3) = 816개 (비교용, 5.7-6). 고르는 데 쓰지 않는다."""
    from indexbeater import signals as S
    return list(itertools.combinations(list(S.SIGNALS), 3))


# ================================================================ 준비·지문·묶음
def engine_fingerprints() -> dict:
    """엔진·점수표 9개 + 데이터 지문 (단독 점수표 scorecard_meta.json 의 '지문'과 같은 모양)."""
    from indexbeater import scorecard as SC
    return SC.fingerprints(with_data=True)


def prepare(log=print):
    """지문 확인 → 데이터(설계 구간만) → 대조군 바탕. 엔진·데이터·점수표 지문이 단독 점수표 때와 같아야 돌린다."""
    from indexbeater import data_io
    from indexbeater import optimize as O
    from indexbeater import scorecard as SC
    now = engine_fingerprints()
    bad = O.check_frozen(now, O.reference_fingerprints())
    ref = json.loads(SCORE_META.read_text(encoding="utf-8"))["지문"]
    bad += [f"{f}: 지금 {now.get(f)} / 단독 점수표 {v}" for f, v in ref.items() if now.get(f) != v]
    if bad:
        raise SystemExit("✗ 지문이 다릅니다. 실행하지 않습니다:\n - " + "\n - ".join(bad))
    log("  ✓ 엔진·점수표·데이터 지문 = 단독 점수표 때와 같음")
    data = data_io.load_wide()                    # 기본값 end = 2019-12-31 (시험 구간 잠금)
    ctx = SC.Context(data, log=log)
    picks, flags = load_picks()
    return now, ctx, picks, flags, EntryCache(data)


def write_part(cname, tables, meta, part_dir) -> Path:
    d = Path(part_dir) / cname
    d.mkdir(parents=True, exist_ok=True)
    tables["scorecard"].to_csv(d / "scorecard.csv", index=False, encoding="utf-8-sig")
    tables["oos_trades"].to_parquet(d / "oos_trades.parquet", index=False)
    tables["portfolio_daily"].to_parquet(d / "portfolio_daily.parquet", index=False)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def part_done(d: Path, now: dict) -> bool:
    """이미 끝난 묶음인가 (meta 가 있고 엔진·데이터 지문이 지금과 같다). 지문이 다르면 멈춘다 (섞이지 않게)."""
    mp = d / "meta.json"
    if not mp.exists():
        return False
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta["지문"] != now:
        raise SystemExit(f"✗ {d} 는 다른 지문으로 만든 묶음이다. 지우고 다시 돌릴지 팀장이 정한다.")
    return True


def run(combos, kind, portfolio=True) -> int:
    from indexbeater import optimize as O
    from indexbeater import scorecard as SC
    t0 = time.time()
    print("1) 준비 (지문 → 데이터 → 대조군 바탕)", flush=True)
    now, ctx, picks, flags, cache = prepare()
    code = SC.code_hash(Path(__file__))           # 시작할 때 한 번 (돌리는 중에 파일을 고쳐도 실제로 돈 판이 적히게)
    part_dir = PARTS / kind
    todo = [c for c in combos if not part_done(part_dir / combo_name(c), now)]
    print(f"2) 조합 {len(combos)}개 중 남은 {len(todo)}개", flush=True)
    for i, names in enumerate(todo, 1):
        t1 = time.time()
        tables = combo_tables(names, ctx, picks, flags, cache, portfolio=portfolio)
        no_port = tables.pop("no_portfolio")
        meta = {"조합": combo_name(names), "신호": list(names), "포트폴리오": portfolio,
                "포트폴리오없음(후보0건)": no_port, "지문": now,
                "combo.py": code, "환경": O.environment(), "git": O.git_commit(),
                "돌린시각": time.strftime("%Y-%m-%d %H:%M:%S"), "걸린초": round(time.time() - t1, 1),
                "행수": {k: len(v) for k, v in tables.items()}}
        write_part(combo_name(names), tables, meta, part_dir)
        print(f"  … {i}/{len(todo)} ({time.time() - t0:.0f}초)", flush=True)
    print(f"끝 ({time.time() - t0:.0f}초). 묶음 폴더: {part_dir}", flush=True)
    return 0


def self_check(name: str) -> int:
    """'name+name' 조합 = 단독 결과 (A AND A = A 이므로 점수표·매매 하나하나·포트폴리오 일별이 끝자리까지 같아야 한다).

    비교 대상 = 단독을 돌린 원본 묶음 results/scorecard_parts/<신호>/ (합친 scorecard.csv 는 숫자 끝자리가 잘려 있어 쓰지 않는다).
    """
    import io
    src = C.RESULTS / "scorecard_parts" / name
    print(f"점검: {name}+{name} 조합 vs {src}", flush=True)
    now, ctx, picks, flags, cache = prepare()
    got = combo_tables((name, name), ctx, picks, flags, cache)
    got.pop("no_portfolio")

    def as_saved(df):                                          # 저장했다 읽은 모양으로 맞춘다
        return pd.read_csv(io.StringIO(df.to_csv(index=False)), float_precision="round_trip")

    def same(a, b, what, skip):
        cols = [c for c in a.columns if c in b.columns and c not in skip]
        pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
        print(f"  ✓ {what} 같음: {len(a):,}줄 × {len(cols)}열 (안 본 열: {', '.join(skip)})")

    read = lambda f: pd.read_csv(src / f, encoding="utf-8-sig", float_precision="round_trip")
    same(as_saved(got["scorecard"]), read("scorecard.csv"), "점수표", ("신호", "2020용세트"))
    same(got["oos_trades"], pd.read_parquet(src / "oos_trades.parquet"), "OOS 매매", ("신호", "세트"))
    same(as_saved(got["portfolio_daily"]), read("portfolio_daily.csv"), "포트폴리오 일별", ("신호",))
    return 0


def merge_pairs(overwrite=False) -> int:
    from indexbeater import benchmark as B
    now = engine_fingerprints()
    pairs = pair_list()
    d = PARTS / "pairs"
    missing = [combo_name(p) for p in pairs if not part_done(d / combo_name(p), now)]
    if missing:
        print(f"✗ 합치지 않습니다: 안 끝난 조합 {len(missing)}개 (예: {missing[:3]})")
        return 1
    versions = {json.loads((d / combo_name(p) / "meta.json").read_text(encoding="utf-8"))["combo.py"]
                for p in pairs}
    print(f"  combo.py 판: {sorted(versions)} ({'하나' if len(versions) == 1 else '여러 개 — 확인 필요'})")
    df = pd.concat([pd.read_csv(d / combo_name(p) / "scorecard.csv", encoding="utf-8-sig",
                                float_precision="round_trip") for p in pairs], ignore_index=True)
    df = B.add_excess(df, B.market_yearly(B.load_market()))
    print(f"  {PAIRS_OUT}: {len(df):,}줄 → {B._save(df, PAIRS_OUT, overwrite)}")
    return 0


def check_segments(seg: pd.DataFrame) -> list[str]:
    """확인 장치: '전체' 줄 = 단독 점수표 2015 학습 거래 수·학습Edge (끝자리까지), 토막 거래 수 합 = 전체."""
    from indexbeater import signals as S
    bad = []
    whole = seg[seg["토막"] == "전체"]
    parts = seg[seg["토막"] != "전체"].groupby(["시장", "신호", "H"])["거래수"].sum()
    for name in whole["신호"].unique():
        sel = pd.read_csv(C.RESULTS / "scorecard_parts" / name / "selection.csv",
                          encoding="utf-8-sig", float_precision="round_trip")
        sel = sel[sel["검증연도"] == FIRST_YEAR].set_index(["시장", "H"])
        for r in whole[whole["신호"] == name].itertuples():
            ref = sel.loc[(r.시장, r.H)]
            if int(ref["고른세트"]) != r.세트 or int(ref["학습거래수"]) != r.거래수:
                bad.append(f"{r.시장} {name} H{r.H}: 세트·거래수 {r.세트},{r.거래수} / 단독 {ref['고른세트']},{ref['학습거래수']}")
            elif not (r.Edge == ref["학습Edge"] or (np.isnan(r.Edge) and np.isnan(ref["학습Edge"]))):
                bad.append(f"{r.시장} {name} H{r.H}: Edge {r.Edge!r} / 단독 {ref['학습Edge']!r}")
            if parts.loc[(r.시장, name, r.H)] != r.거래수:
                bad.append(f"{r.시장} {name} H{r.H}: 토막 거래수 합 {parts.loc[(r.시장, name, r.H)]} ≠ 전체 {r.거래수}")
    if len(whole) != len(C.UNIVERSE_MARKETS) * len(S.SIGNALS) * len(C.HOLDS):
        bad.append(f"칸 수 {len(whole)}")
    return bad


def triple_pick(overwrite=False) -> int:
    """5.7-1~3: 토막별 학습 Edge → 순위 → 시장별 상위 6개 → 20개. 확인 장치가 하나라도 틀리면 저장하지 않는다."""
    from indexbeater import benchmark as B
    t0 = time.time()
    _, ctx, picks, _, cache = prepare()
    bounds = segment_bounds(ctx)
    print("토막 (거래일 번호 기준 3등분):")
    for i, (a, b) in enumerate(bounds, 1):
        print(f"  {i}: {ctx.dates[a].date()} ~ {ctx.dates[b].date()} ({b - a + 1}거래일)")
    seg = segment_table(ctx, picks, cache)
    bad = check_segments(seg)
    if bad:
        print("✗ 확인 장치 실패 — 저장하지 않습니다:\n - " + "\n - ".join(bad[:20]))
        return 1
    n = int((seg["토막"] == "전체").sum())
    print(f"  ✓ {n}칸 모두: 세 토막 합친 학습 Edge·거래 수 = 단독 점수표 2015 학습Edge·학습거래수 (끝자리까지)")
    top = rank_top(seg)
    print(f"  {TRIPLE_SEG_OUT.name}: {B._save(seg, TRIPLE_SEG_OUT, overwrite)}")
    print(f"  {TRIPLE_RANK_OUT.name}: {B._save(top, TRIPLE_RANK_OUT, overwrite)}")
    for mk in C.UNIVERSE_MARKETS:
        t = top[top["시장"] == mk]
        print(f"\n{mk} 상위 {C.TRIPLE_TOP_N}개:")
        print(t[t["상위"]][["등수", "신호", "순위중앙값", "순위평균"]].to_string(index=False))
        print(f"  → 3신호 {len(triple_list(top, mk))}개")
    print(f"\n끝 ({time.time() - t0:.0f}초)")
    return 0


def picked_triples() -> dict[str, list[tuple[str, str, str]]]:
    """--triple-pick 결과 파일에서 시장별 20개."""
    top = pd.read_csv(TRIPLE_RANK_OUT, encoding="utf-8-sig")
    return {mk: triple_list(top, mk) for mk in C.UNIVERSE_MARKETS}


def merge_triples(overwrite=False) -> int:
    """816개 묶음 → 한 파일. '고른20' = 그 시장에서 5.7-3 으로 고른 20개 안에 드는 칸."""
    from indexbeater import benchmark as B
    now = engine_fingerprints()
    allt = all_triples()
    d = PARTS / "triples"
    missing = [combo_name(c) for c in allt if not part_done(d / combo_name(c), now)]
    if missing:
        print(f"✗ 합치지 않습니다: 안 끝난 조합 {len(missing)}개 (예: {missing[:3]})")
        return 1
    versions = {json.loads((d / combo_name(c) / "meta.json").read_text(encoding="utf-8"))["combo.py"] for c in allt}
    print(f"  combo.py 판: {sorted(versions)} ({'하나' if len(versions) == 1 else '여러 개 — 확인 필요'})")
    df = pd.concat([pd.read_csv(d / combo_name(c) / "scorecard.csv", encoding="utf-8-sig",
                                float_precision="round_trip") for c in allt], ignore_index=True)
    chosen = {(mk, combo_name(c)) for mk, cs in picked_triples().items() for c in cs}
    df.insert(2, "고른20", [(m, n) in chosen for m, n in zip(df["시장"], df["신호"])])
    df = B.add_excess(df, B.market_yearly(B.load_market()))
    print(f"  {TRIPLES_OUT}: {len(df):,}줄 → {B._save(df, TRIPLES_OUT, overwrite)}")
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m indexbeater combo", description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", metavar="신호")
    ap.add_argument("--pairs", action="store_true", help="2신호 153개")
    ap.add_argument("--only", help="조합 이름 하나만 (예: rsi+macd)")
    ap.add_argument("--merge-pairs", action="store_true")
    ap.add_argument("--triple-pick", action="store_true", help="3신호: 토막 순위 → 상위 6개 → 20개")
    ap.add_argument("--triples", action="store_true", help="3신호: 시장별 고른 20개")
    ap.add_argument("--triples-all", action="store_true", help="3신호: 816개 전부 (비교용)")
    ap.add_argument("--merge-triples", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="합친 결과 파일 덮어쓰기 (팀장 승인 뒤)")
    a = ap.parse_args(argv)
    if a.self_check:
        return self_check(a.self_check)
    if a.pairs:
        pairs = pair_list()
        if a.only:
            pairs = [p for p in pairs if combo_name(p) == a.only]
            if not pairs:
                raise SystemExit(f"153개 안에 없는 조합: {a.only}")
        return run(pairs, "pairs")
    if a.merge_pairs:
        return merge_pairs(a.overwrite)
    if a.triple_pick:
        return triple_pick(a.overwrite)
    if a.triples or a.triples_all:
        chosen = sorted({c for cs in picked_triples().values() for c in cs}, key=all_triples().index)
        todo = chosen if a.triples else chosen + [c for c in all_triples() if c not in chosen]
        if a.only:
            todo = [c for c in todo if combo_name(c) == a.only]
            if not todo:
                raise SystemExit(f"목록에 없는 조합: {a.only}")
        return run(todo, "triples")
    if a.merge_triples:
        return merge_triples(a.overwrite)
    ap.print_help()
    return 0
