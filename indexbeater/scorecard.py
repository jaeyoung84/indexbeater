"""1단계 단독 18개 성적표 계산 (docs/프로젝트_기준.md 5.3·5.4·5.5·5.17).

신호 하나를 받아 다음을 만든다. 엔진(engine.py)·신호(signals.py)·기준값(config.py)은 고치지 않고 부르기만 한다.

1. 그리드 성적표 (5.3): 시장 × 세트 × H 마다 설계 구간(2012-07 ~ 2019) 거래 수·승률·Edge·t값·연도별 Edge
2. 학습 구간 Edge (5.5): 검증 연도 Y마다 cut(= Y년 첫 거래일 − 21거래일)까지의 Edge·거래 수
3. 파라미터 고르기 (5.5): 300건 이상 세트 중 '안정 구간 점수'(자기 + 이웃 Edge 의 중앙값)가 가장 높은 세트
4. 점수표 (5.17): 2015~2019 표본 외(OOS) 매매를 이어 붙여 시장 × H × 비용마다 한 줄
5. 포트폴리오 (5.17 결정 ②): OOS 신호로 1억·20종목·5% 포트폴리오, 비용 0.5% (0.25%는 상위 후보만 — 부록 E)

팀장 해석 (11/24, 결과를 보기 전에 회의기록.md 에 적음 — 11/25 4명 확인)
- 학습 표본 = 신호일 ≤ cut. 승리 = 비용 뺀 수익 > 0. 진 거래 = 0 이하.
- 플러스 해 = 그 해 OOS 거래가 30건 이상이고 Edge > 0 인 해.
- 이웃 = 한 축에서만 한 칸 옆인 세트 (없는 짝은 건너뜀). 300건 미만 세트는 고르기·이웃 둘 다에서 뺀다.
- 동점 = 기본값(DEFAULT_PARAMS)과 칸 거리가 가까운 쪽 → 그래도 같으면 그리드 순서 앞.
- 불안정 = 6번(2015~2019용 5번 + 2020용 1번) 고른 세트 중 가장 많이 고른 세트가 3번 미만.
- 대조 종목이 없는 매매는 Edge·t값·카이제곱에서만 빼고 수를 적는다 (승률·평균 수익에는 넣는다).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from indexbeater import config as C
from indexbeater import engine as E
from indexbeater import metrics as M
from indexbeater import signals as S
from indexbeater.exits import ExitRule
from indexbeater.scoring import signal_score

PART_DIR = C.RESULTS / "scorecard_parts"
SELECT_YEARS = (*C.WF_YEARS, 2020)            # 2020 = 시험 구간에 쓸 파라미터 (2019년 말까지 자료로 고름)
UNSTABLE_MIN_SAME = 3                          # 6번 중 같은 세트가 3번 미만이면 '불안정'
FINGERPRINT_FILES = ("engine.py", "signals.py", "exits.py", "indicators.py", "data_io.py", "config.py",
                     "metrics.py", "scorecard.py", "scoring.py")

# 계획서 기본값 (5.5 동점·표본 부족 때 사용). 계획서에 표가 없어서 팀장 결정으로 정했다 (11/24 질문 2 → ②):
#   교과서 표준값(지표를 만든 사람이 쓴 값·널리 쓰는 값)이 있는 축은 그 값.
#   교과서 값이 없는 축만 이전 규칙: 부록 A 예시값, 없으면 그리드 가운데 값(짝수 개면 작은 쪽). ※ 표시.
#   모두 그리드 안의 값이다 (그리드 밖이면 default_index 가 멈춘다).
# 11/24 팀장 보완: 근거가 약하던 7개 축을 논문·대표 자료로 다시 확인했다 (회의기록 "기본값 근거 확인").
#   바뀐 것 3개: 거래량 돌파 m 3→2, OBV 10→20, 캔들 추세 5→10. 나머지 4개는 값 그대로, 근거만 보탬.
DEFAULT_PARAMS = {
    "ma_cross": {"short": 5, "long": 20},              # 기재부 시사경제용어사전: 단기 골든크로스 = 5일선이 20일선 돌파 (50·200 은 그리드 밖)
    "alignment": {"periods": (5, 20, 60)},             # 정배열 5·20·60 (부록 A 예시와 같음)
    "macd": {"fast": 12, "slow": 26, "signal": 9},     # 아펠(Appel) 12·26·9
    "ichimoku": {"conv": 9, "base": 26},               # 호소다 9·26
    "adx_di": {"period": 14, "adx_min": 25},           # 와일더(Wilder) 14, 추세 기준 25 (와일더 규칙: 25 위 = 추세, 20 아래 = 추세 없음)
    "sar": {"step": 0.02, "max_step": 0.2},            # 와일더 0.02·0.2
    "rsi": {"period": 14, "low": 30},                  # 와일더 14·30
    "stochastic": {"k": 14, "d": 3, "low": 20, "high": 80},   # 레인(Lane) 14·3, 20·80
    "disparity": {"ma": 20, "thr": 0.95},              # 20일 이격도 95% (부록 A 예시, 키움증권 지표 안내: 95% 밑돌면 매수)
    "cci": {"period": 20},                             # 램버트(Lambert, 1980) 20 — 주기의 1/3, 보통 쓰는 값
    "williams_r": {"period": 14},                      # 윌리엄스 14
    "bollinger": {"period": 20, "k": 2.0},             # 볼린저 20·2
    "squeeze": {"period": 20, "lookback": 120},        # 볼린저: 밴드폭이 6개월(약 120거래일) 최저
    "donchian": {"n": 20},                             # 돈치안 4주(20일) 규칙
    "vol_breakout": {"k": 0.5},                        # 래리 윌리엄스 k 0.5
    "volume_breakout": {"n": 60, "m": 2},              # n: 터틀 시스템 2 의 55일 돌파 → 가장 가까운 60. m: 오닐 "평균보다 40~50% 많은 거래" → 가장 가까운 2
    "obv": {"ma": 20},                                 # 그래빌 원래 방식엔 평균선 없음 → 바차트(Barchart) OBV 평균선 기본 20
    "candle": {"trend": 10, "tail_ratio": 2},          # 꼬리 ≥ 몸통 2배 (니슨 Nison). 추세 10: Marshall·Young·Rose(2006) 기본 = 10일 지수평균으로 앞선 추세
}


# ---------------------------------------------------------------- 지문
def code_hash(path) -> str:
    """노트북 04 와 같은 방식: 줄바꿈을 LF 로 맞춘 뒤 sha256 앞 12글자."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def fingerprints(with_data: bool = True) -> dict:
    pkg = Path(__file__).resolve().parent
    out = {f"indexbeater/{f}": code_hash(pkg / f) for f in FINGERPRINT_FILES}
    if with_data:
        out["data/krx_adjusted.parquet"] = file_hash(C.KRX_ADJUSTED)
    return out


# ---------------------------------------------------------------- 그리드 모양
def param_sets(name: str) -> list[dict]:
    return list(S.iter_params(name))


def grid_pos(name: str, p: dict) -> tuple:
    """세트가 그리드 각 축에서 몇 번째 칸인지 (이웃·거리 계산용)."""
    grid = C.PARAM_GRID[name]
    return tuple(list(vals).index(p[k]) for k, vals in grid.items())


def neighbors(positions: list[tuple]) -> list[list[int]]:
    """세트마다 '한 축에서만 한 칸 옆'인 세트 번호 목록."""
    out = []
    for a in positions:
        out.append([k for k, b in enumerate(positions)
                    if sum(abs(x - y) for x, y in zip(a, b)) == 1])
    return out


def default_index(name: str, sets: list[dict]) -> int:
    d = DEFAULT_PARAMS[name]
    for k, p in enumerate(sets):
        if all(p[a] == d[a] for a in d):
            return k
    raise ValueError(f"{name}: 기본값 {d} 이 그리드에 없습니다")


def select_set(edges: np.ndarray, counts: np.ndarray, nbrs: list[list[int]],
               positions: list[tuple], default: int) -> dict:
    """5.5 고르기 한 번. edges·counts = 세트별 학습 Edge·거래 수."""
    ok = counts >= C.MIN_TRADES
    if not ok.any():
        return {"set": default, "score": float("nan"), "eligible": 0, "short": True, "used_default": True}
    score = np.full(len(edges), -np.inf)
    for k in np.nonzero(ok)[0]:
        vals = [edges[k]] + [edges[q] for q in nbrs[k] if ok[q]]
        score[k] = float(np.median(vals))
    best = score.max()
    tied = [k for k in np.nonzero(ok)[0] if np.isclose(score[k], best, rtol=0, atol=1e-12)]
    dist = [sum(abs(x - y) for x, y in zip(positions[k], positions[default])) for k in tied]
    pick = tied[int(np.argmin(dist))]           # argmin 은 같은 거리면 앞(그리드 순서)을 고른다
    return {"set": int(pick), "score": float(best), "eligible": int(ok.sum()), "short": False,
            "used_default": len(tied) > 1 and pick == default}


class BadTradeError(RuntimeError):
    """체결 매매 수익이 무한대·숫자 아님 → 점수표를 만들지 않는다."""


def check_finite(gross: np.ndarray, where: str) -> None:
    """체결 수익에 무한대(inf)가 있으면 멈춘다.

    엔진 v1 은 '시가 0 · 거래량 > 0' 인 날(자료 53칸)을 살 수 있는 날로 보고 0원에 샀다 (11/24 발견).
    엔진 v2 에서 고쳤다. 이 검사는 안전장치로 남긴다: 그런 매매가 섞이면 평균·t값이 모두 망가지므로,
    숫자를 만들지 않고 멈춘다.
    """
    g = np.asarray(gross, float)
    bad = np.isinf(g).sum()
    if bad:
        raise BadTradeError(f"{where}: 수익이 무한대인 체결 매매 {bad}건 (0원 매수). "
                            "엔진 결함입니다 — 결과를 쓰지 말고 팀장에게 알려 주세요 (회의기록 11/24 '시가 0 매수').")


# ---------------------------------------------------------------- 공통 준비 (사람마다 한 번)
class Context:
    """데이터·엔진 배열·짝 칸·대조군 바탕(H 3개)을 한 번만 만든다."""

    def __init__(self, data: dict, log=print):
        t0 = time.time()
        self.data = data
        self.m = E._Market(data)
        self.dates = self.m.dates
        self.allowed = data["tradable"].to_numpy(bool) & np.asarray(self.dates >= C.SIGNAL_START)[:, None]
        self.cell = M.match_cells(data)
        self.base = {}
        for h in C.HOLDS:
            self.base[h] = M.ControlBase(data, ExitRule(hold=h), self.m, self.cell)
            check_finite(self.base[h].gross, f"대조군 바탕 H={h}")
            log(f"  대조군 바탕 H={h}: 체결 {self.base[h].n_base_trades:,}건 ({time.time() - t0:.0f}초)")
        self.cut_idx = {}
        for y in SELECT_YEARS:
            before = np.nonzero(self.dates < pd.Timestamp(f"{y}-01-01"))[0]
            self.cut_idx[y] = int(before[-C.WF_CUT_OFFSET])      # 계획서 코드 cut = days[days < start][-21]
        self.years = self.dates.year.to_numpy()
        self.oos_calendar = self.dates[(self.years >= C.WF_YEARS[0]) & (self.years <= C.WF_YEARS[-1])]
        self.full_calendar = self.dates[self.dates >= C.SIGNAL_START]


def _trade_table(ctx: Context, entry: pd.DataFrame, hold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """세트 하나 × H 하나: 체결 매매 표(대조군 붙임)와 상태 표."""
    tr = E.simulate_trades(entry, ctx.data, ExitRule(hold=hold), dedupe=True, _m=ctx.m)
    st = tr.loc[tr["status"].isin([E.FILLED, E.GAP_CANCEL, E.NO_BUY]), ["market", "t_idx", "status"]].copy()
    st["year"] = ctx.years[st["t_idx"].to_numpy(int)]
    f = tr[tr["status"] == E.FILLED]
    t = f["t_idx"].to_numpy(int)
    j = pd.Index(ctx.m.codes).get_indexer(f["code"])
    sig_on = entry.reindex(index=ctx.dates, columns=ctx.m.codes, fill_value=False).to_numpy(bool) & ctx.allowed
    base = ctx.base[hold]
    check_finite(f["ret_gross"].to_numpy(float), "신호 매매")
    out = pd.DataFrame({"market": f["market"].to_numpy(), "code": f["code"].to_numpy(),
                        "t_idx": t, "year": ctx.years[t],
                        "gross": f["ret_gross"].to_numpy(float)})
    out["gross_w"] = np.where(f["reason"].eq(E.DELIST).to_numpy(), -1.0, out["gross"])
    for worst, sfx in ((False, ""), (True, "_w")):
        mt = base.matched(sig_on, t, j, worst=worst)
        out[f"ctrl_gross{sfx}"] = mt["ctrl_gross"]
        for name in M.COSTS:
            out[f"ctrl_win{name}{sfx}"] = mt[f"ctrl_win{name}"]
    out["edge"] = out["gross"] - out["ctrl_gross"]
    out["edge_w"] = out["gross_w"] - out["ctrl_gross_w"]
    return out, st


def _edge_block(ft: pd.DataFrame, ctx: Context, calendar, sfx="") -> dict:
    v = ft[~np.isnan(ft[f"edge{sfx}"].to_numpy())]
    daily = v.groupby("t_idx")[f"edge{sfx}"].mean()
    daily.index = ctx.dates[daily.index.to_numpy()]
    return {"edge": float(v[f"edge{sfx}"].mean()) if len(v) else float("nan"),
            "t": M.block_t(daily, calendar), "n_valid": len(v)}


def _grid_rows(name, k, p, hold, ft, st, ctx) -> list[dict]:
    rows = []
    for mk in C.UNIVERSE_MARKETS:
        f = ft[(ft["market"] == mk)]
        eb = _edge_block(f, ctx, ctx.full_calendar)
        r = {"시장": mk, "신호": name, "세트": k, "파라미터": json.dumps(p, ensure_ascii=False), "H": hold,
             "거래수": len(f), "체결불가비율": M.unfillable_ratio(st.loc[st["market"] == mk, "status"]),
             "승률_0.5%": float((f["gross"] - C.COST_SELECT > 0).mean()) if len(f) else np.nan,
             "평균수익_0.5%": float(f["gross"].mean() - C.COST_SELECT) if len(f) else np.nan,
             "평균수익_0.25%": float(f["gross"].mean() - C.COST_ROUNDTRIP) if len(f) else np.nan,
             "대조군평균_0.5%": float(f["ctrl_gross"].mean() - C.COST_SELECT) if len(f) else np.nan,
             "Edge": eb["edge"], "t값": eb["t"], "대조군없음": len(f) - eb["n_valid"]}
        for y in range(C.SIGNAL_START.year, C.DESIGN_END.year + 1):
            fy = f[f["year"] == y]
            r[f"Edge_{y}"] = float(fy["edge"].mean()) if len(fy) else np.nan
            r[f"거래수_{y}"] = len(fy)
        rows.append(r)
    return rows


def _train_rows(name, k, hold, ft, ctx) -> list[dict]:
    rows = []
    v = ft[~np.isnan(ft["edge"].to_numpy())]
    for mk in C.UNIVERSE_MARKETS:
        vm = v[v["market"] == mk]
        for y, ci in ctx.cut_idx.items():
            tr = vm[vm["t_idx"] <= ci]
            rows.append({"시장": mk, "신호": name, "세트": k, "H": hold, "검증연도": y,
                         "cut": ctx.dates[ci].date().isoformat(), "학습거래수": len(tr),
                         "학습Edge": float(tr["edge"].mean()) if len(tr) else np.nan})
    return rows


def _score_rows(name, mk, hold, oos, st_oos, flags_sel, ctx) -> list[dict]:
    rows = []
    n = len(oos)
    for cname, c in M.COSTS.items():
        r = {"시장": mk, "신호": name, "H": hold, "비용": cname, "거래수": n}
        res = {}
        for sfx in ("", "_w"):
            eb = _edge_block(oos, ctx, ctx.oos_calendar, sfx)
            pos = 0
            for y in C.WF_YEARS:
                fy = oos[(oos["year"] == y) & ~np.isnan(oos[f"edge{sfx}"].to_numpy())]
                if len(fy) >= C.MIN_TRADES_PER_YEAR and fy[f"edge{sfx}"].mean() > 0:
                    pos += 1
            ret = oos[f"gross{sfx}"].to_numpy(float) - c
            wl = M.win_loss(ret)
            sc = signal_score(eb["edge"], hold, pos, eb["t"], wl["평균이익"], wl["평균손실"],
                              wl["승률"], n)
            res[sfx] = (eb, pos, ret, wl, sc)
        eb, pos, ret, wl, sc = res[""]
        v = ~np.isnan(oos["edge"].to_numpy())
        chi, pval = M.chi_square((ret > 0)[v], oos[f"ctrl_win{cname}"].to_numpy(float)[v])
        r.update({"점수": sc["점수"], "수익점수": sc["수익점수"], "안정점수": sc["안정점수"],
                  "손익비점수": sc["손익비점수"], "승률점수": sc["승률점수"],
                  "Edge": eb["edge"], "Edge20": sc["Edge20"], "t값": eb["t"], "플러스해수": pos,
                  "승률": wl["승률"], "평균수익": float(ret.mean()) if n else np.nan,
                  "대조군평균": float(np.nanmean(oos["ctrl_gross"].to_numpy()) - c) if n else np.nan,
                  "평균이익": wl["평균이익"], "평균손실": wl["평균손실"], "손익비": sc["손익비"],
                  "손익분기승률": sc["손익분기승률"], "카이제곱": chi, "카이제곱_p": pval,
                  "대조군없음": int((~v).sum())})
        for y in C.WF_YEARS:
            fy = oos[oos["year"] == y]
            r[f"Edge_{y}"] = float(np.nanmean(fy["edge"].to_numpy())) if len(fy) else np.nan
            r[f"평균수익_{y}"] = float(fy["gross"].mean() - c) if len(fy) else np.nan
            r[f"거래수_{y}"] = len(fy)
        unf = M.unfillable_ratio(st_oos["status"])
        top5 = M.top5_share(oos["code"], pd.Series(ret, index=oos.index))
        flags = [x for x in sc["표시"].split(",") if x]
        if unf > C.MAX_UNFILLABLE_RATIO:
            flags.append("보류")
        if top5 > C.TOP5_CONCENTRATION_WARN:
            flags.append("경고")
        flags += flags_sel
        eb_w, pos_w, _, _, sc_w = res["_w"]
        r.update({"체결불가비율": unf, "상위5종목기여": top5, "표시": ",".join(dict.fromkeys(flags)),
                  "Edge_폐지최악": eb_w["edge"], "t값_폐지최악": eb_w["t"],
                  "플러스해수_폐지최악": pos_w, "점수_폐지최악": sc_w["점수"]})
        rows.append(r)
    return rows


def run_signal(name: str, ctx: Context, log=print, portfolio: bool = True) -> dict[str, pd.DataFrame]:
    """신호 하나의 모든 표를 만든다."""
    t0 = time.time()
    sets = param_sets(name)
    positions = [grid_pos(name, p) for p in sets]
    nbrs = neighbors(positions)
    dflt = default_index(name, sets)
    entries = {k: S.entry_signal(name, ctx.data, p) for k, p in enumerate(sets)}
    grid, train, trades, status = [], [], {}, {}
    for k, p in enumerate(sets):
        for h in C.HOLDS:
            ft, st = _trade_table(ctx, entries[k], h)
            trades[(k, h)], status[(k, h)] = ft, st
            grid += _grid_rows(name, k, p, h, ft, st, ctx)
            train += _train_rows(name, k, h, ft, ctx)
        log(f"  {name} 세트 {k + 1}/{len(sets)} {p} ({time.time() - t0:.0f}초)")
    train_df = pd.DataFrame(train)

    selection, scores, oos_all, port = [], [], [], []
    for mk in C.UNIVERSE_MARKETS:
        for h in C.HOLDS:
            picks = {}
            for y in SELECT_YEARS:
                tt = train_df[(train_df["시장"] == mk) & (train_df["H"] == h) & (train_df["검증연도"] == y)]
                tt = tt.set_index("세트").reindex(range(len(sets)))
                sel = select_set(tt["학습Edge"].to_numpy(float), tt["학습거래수"].fillna(0).to_numpy(),
                                 nbrs, positions, dflt)
                picks[y] = sel
                selection.append({"시장": mk, "신호": name, "H": h, "검증연도": y,
                                  "cut": ctx.dates[ctx.cut_idx[y]].date().isoformat(),
                                  "고른세트": sel["set"], "파라미터": json.dumps(sets[sel["set"]], ensure_ascii=False),
                                  "안정구간점수": sel["score"],
                                  "학습거래수": int(np.nan_to_num(tt.loc[sel["set"], "학습거래수"])),
                                  "학습Edge": tt.loc[sel["set"], "학습Edge"], "300건이상세트수": sel["eligible"],
                                  "표본부족": sel["short"], "기본값사용": sel["used_default"]})
            most = pd.Series([v["set"] for v in picks.values()]).value_counts().iloc[0]
            unstable = most < UNSTABLE_MIN_SAME
            for row in selection[-len(SELECT_YEARS):]:
                row["불안정"] = unstable
            flags_sel = (["표본 부족"] if any(picks[y]["short"] for y in C.WF_YEARS) else []) + \
                        (["불안정"] if unstable else []) + \
                        (["기본값 사용"] if any(picks[y]["used_default"] for y in C.WF_YEARS) else [])
            parts, sparts = [], []
            for y in C.WF_YEARS:
                k = picks[y]["set"]
                ft, st = trades[(k, h)], status[(k, h)]
                parts.append(ft[(ft["market"] == mk) & (ft["year"] == y)].assign(세트=k))
                sparts.append(st[(st["market"] == mk) & (st["year"] == y)])
            oos = pd.concat(parts, ignore_index=True)
            st_oos = pd.concat(sparts, ignore_index=True)
            rows = _score_rows(name, mk, h, oos, st_oos, flags_sel, ctx)
            rows[0]["2020용세트"] = rows[1]["2020용세트"] = json.dumps(sets[picks[2020]["set"]], ensure_ascii=False)
            if portfolio:
                pr, pdaily = _portfolio(name, mk, h, picks, entries, ctx)
                rows[0].update(pr)                 # 포트폴리오는 0.5% 줄에만 (0.25%는 상위 후보만 — 부록 E)
                port.append(pdaily)
            scores += rows
            oos_all.append(oos.assign(신호=name, H=h, signal_date=ctx.dates[oos["t_idx"].to_numpy(int)]))
    log(f"  {name} 끝 ({time.time() - t0:.0f}초)")
    return {"grid": pd.DataFrame(grid), "train": train_df, "selection": pd.DataFrame(selection),
            "scorecard": pd.DataFrame(scores),
            "oos_trades": pd.concat(oos_all, ignore_index=True),
            "portfolio_daily": pd.concat(port, ignore_index=True) if port else pd.DataFrame()}


def _portfolio(name, mk, hold, picks, entries, ctx):
    """2015~2019 해마다 그 해에 고른 세트의 신호만 켠 표로 포트폴리오를 돌린다 (비용 0.5%)."""
    sig = np.zeros((len(ctx.dates), len(ctx.m.codes)), bool)
    for y in C.WF_YEARS:
        rows = ctx.years == y
        e = entries[picks[y]["set"]].reindex(index=ctx.dates, columns=ctx.m.codes, fill_value=False).to_numpy(bool)
        sig[rows] = e[rows]
    daily, _ = E.run_portfolio(pd.DataFrame(sig, index=ctx.dates, columns=ctx.m.codes), ctx.data,
                               ExitRule(hold=hold), market=mk)
    eq = daily["equity"]
    check_finite(eq.to_numpy(float), f"포트폴리오 {mk} H={hold}")
    eq = eq[eq.index.year >= C.WF_YEARS[0]]
    out = {}
    prev = float(C.INIT_CAPITAL)                  # 2015년 전에는 신호를 켜지 않았으므로 시작 금액 그대로
    for y in C.WF_YEARS:
        e = eq[eq.index.year == y]
        out[f"포트_수익_{y}"] = float(e.iloc[-1] / prev - 1) if len(e) else np.nan
        prev = float(e.iloc[-1]) if len(e) else prev
    out["포트_MDD"] = M.max_drawdown(pd.concat([pd.Series([float(C.INIT_CAPITAL)]), eq.reset_index(drop=True)]))
    out["포트_마지막평가금액"] = float(eq.iloc[-1])
    pdaily = pd.DataFrame({"date": eq.index, "시장": mk, "신호": name, "H": hold,
                           "equity": eq.to_numpy(), "n_pos": daily.loc[eq.index, "n_pos"].to_numpy()})
    return out, pdaily


def write_part(name: str, tables: dict, meta: dict, part_dir: Path = PART_DIR) -> Path:
    d = Path(part_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        if key == "oos_trades":
            df.to_parquet(d / f"{key}.parquet", index=False)
        else:
            df.to_csv(d / f"{key}.csv", index=False, encoding="utf-8-sig")
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return d
