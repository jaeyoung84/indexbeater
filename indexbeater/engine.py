"""공통 백테스트 엔진 (계획서 STEP 3) — 4명 합의 없이 수정 금지.

신호(True/False 표) → t+1 시가 매수 → 매도 규칙 → 매매별 수익률(비용 차감) 표.

매매 기준 (4.2)
- t일 장 마감 후 신호 판단, t+1일 수정 시가에 매수
- t+1 시가가 t일 종가 대비 +29%(2015.6.15 이전 +14%) 이상이면 상한가 근처로 보고 매수 불가
- 매도 신호(신호 매도·S2 트레일링·S3 시간 청산)는 다음 날 시가에 매도
- S1 손절·익절은 매수 다음 날부터 장중 고가·저가로 판정, 그 가격(갭이면 시가)에 매도
- 같은 날 시가 매도와 장중 손절이 겹치면 시가 매도가 먼저 (시간 순서, 11/20 결정 4A)
- 이미 보유 중인 종목은 신호가 다시 나와도 추가 매수하지 않는다 (dedupe=True)
- 매도일이 데이터 끝을 넘는 매매는 버린다 (미완료)

팀장 해석 (11/25 회의에서 확정할 것 — 회의기록.md 1단계 구현 질문)
- 갭은 수정 시가 ÷ 수정 종가로 잰다
- 하한가 근처(시가가 전날 종가 대비 −29%, 2015.6.15 이전 −14% 이하)인 날은 시가에 못 판다 (상한가 갭의 반대쪽)
- 거래정지일(거래량 0)도 팔 수 없다. 팔 수 있는 첫 날 시가로 미루고 미룬 날 수를 delay 에 적는다
- 시가가 0인 날(가격 없음, 거래량은 있음)은 거래정지일과 같이 사지도 팔지도 않는다 (엔진 v2, 11/24 발견 결함)
- 팔기 전에 상장폐지되면 마지막 거래일 종가로 정산한다 (reason=상장폐지).
  거래정지 중에 폐지되면 정지 전 가격으로 정산되어 덜 잃은 것으로 나올 수 있으므로,
  성적표는 delisting_worst_case()(상장폐지 정산 = −100%)로도 한 번 더 계산한다 (11/20 결정 ③)
- 비용은 수익률에서 뺀다: ret_net = ret_gross − 0.5% (ret_net_025 는 −0.25%, 세금 포함, 결정 3A)
- 중복 무시: t일 장 마감 때 아직 들고 있는(매도일 > t) 종목의 신호는 버린다. 매도일 = t 이면 다시 산다
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indexbeater import config as C
from indexbeater.data_io import last_valid_index
from indexbeater.exits import ExitRule

# 매매 상태 이름 (표의 status 열)
FILLED = "체결"
GAP_CANCEL = "취소_갭"
NO_BUY = "매수불가"
OPEN_END = "미완료"
DUPLICATE = "중복_무시"
DELIST = "상장폐지"          # 매도 이유(reason 열): 팔기 전에 자료가 끝나 마지막 종가로 정산

# 컴퓨터 소수 계산 오차를 없애는 아주 작은 값 (전략 숫자가 아니다).
# 예: 114 ÷ 100 − 1 을 컴퓨터는 0.1399999999999999 로 계산해서 '+14% 이상'을 놓친다.
FLOAT_EPS = 1e-9

TRADE_COLS = [
    "code", "market", "signal_date", "buy_date", "exit_date", "buy_px", "sell_px",
    "status", "reason", "delay", "sell_at_open", "ret_gross", "ret_net", "ret_net_025",
    "t_idx", "b_idx", "s_idx",
]


def gap_limit(dates: pd.DatetimeIndex) -> np.ndarray:
    """날짜마다 상한가 근처로 볼 갭 기준 (기준 4.2). 2015-06-15 전 0.14, 그 뒤 0.29."""
    return np.where(dates < C.LIMIT_CHANGE_DATE, C.UPPER_GAP_BEFORE, C.UPPER_GAP_AFTER)


class _Market:
    """엔진이 매번 쓰는 숫자 배열을 한 번만 만들어 두는 상자.

    to_numpy() 는 표(DataFrame)를 숫자 배열로 바꾸는 명령이다. 배열은 칸 하나 꺼내는 속도가 훨씬 빠르다.
    """

    def __init__(self, data: dict):
        px = data["adj_close"]
        self.dates = px.index
        self.codes = px.columns
        self.n = len(self.dates)
        self.O = data["adj_open"].to_numpy(float)
        self.H = data["adj_high"].to_numpy(float)
        self.L = data["adj_low"].to_numpy(float)
        self.Cl = px.to_numpy(float)
        vol = data["volume"].to_numpy(float)
        self.has = ~np.isnan(self.Cl)
        self.thr = gap_limit(self.dates)
        self.last = last_valid_index(px)          # 종목마다 자료가 있는 마지막 행
        self.market = data["market"].to_numpy(object)
        self.value20 = data["value20"].to_numpy(float)

        # 하한가 근처 시가: 시가 ÷ 전날 종가 − 1 ≤ −기준 (상한가 갭 기준의 반대쪽)
        prev_c = np.vstack([np.full((1, self.Cl.shape[1]), np.nan), self.Cl[:-1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            down = (self.O / prev_c - 1) <= -self.thr[:, None] + FLOAT_EPS
        # 가격이 있는 날 (엔진 v2, 11/24): 시가가 0보다 크다.
        #   데이터에 '시가·고가·저가 = 0, 거래량 > 0' 인 칸이 53개 있다 (2012~2019, 40종목).
        #   v1 은 그 날을 살 수 있는 날로 보고 0원에 사서 수익이 무한대가 되었다.
        #   이런 날은 거래정지일처럼 사지도 팔지도 않고, 장중 손절 판정에도 쓰지 않는다.
        self.priced = np.nan_to_num(self.O) > 0
        # 살 수 있는 날: 자료가 있고, 거래가 있었고(거래정지 아님), 가격이 있다
        # 팔 수 있는 날: 살 수 있는 날 중 하한가 근처 시가가 아닌 날
        self.traded = self.has & (np.nan_to_num(vol) > 0) & self.priced
        self.sellable = self.traded & ~down

        # nxt[i, j] = i일 또는 그 뒤에서 j 종목을 팔 수 있는 첫 날 (없으면 n)
        # 뒤에서부터 '지금까지 본 가장 작은 날짜 번호'를 적어 오는 방식이다 (minimum.accumulate).
        rows = np.arange(self.n)[:, None]
        idx = np.where(self.sellable, rows, self.n)
        nxt = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
        self.nxt = np.vstack([nxt, np.full((1, nxt.shape[1]), self.n)])   # 맨 끝에 '없음' 줄 추가


def _one_trade(m: _Market, t: int, j: int, rule: ExitRule) -> dict:
    """t일 신호 하나로 매매 한 건을 끝까지 따라간다."""
    n = m.n
    rec = dict(t_idx=t, b_idx=t + 1, s_idx=-1, buy_px=np.nan, sell_px=np.nan,
               reason=None, delay=0, sell_at_open=False)
    b = t + 1
    if b >= n:                                   # 내일 자료가 없다 → 끝을 넘는 매매
        return {**rec, "status": OPEN_END}
    if not m.traded[b, j]:                       # 내일 자료가 없거나(상장폐지) 거래정지 → 못 산다
        return {**rec, "status": NO_BUY}
    gap = m.O[b, j] / m.Cl[t, j] - 1
    if gap >= m.thr[b] - FLOAT_EPS:                        # 상한가 근처로 시작 → 살 수 없다 (기준 4.2)
        return {**rec, "status": GAP_CANCEL}
    buy = m.O[b, j]
    rec["buy_px"] = buy
    s0 = b + rule.hold                            # S3: 매수일부터 H 거래일 뒤 시가

    if rule.stop is None and rule.take is None:
        # S3만 쓸 때: s0일 또는 그 뒤 팔 수 있는 첫 날 시가에 판다
        s = m.nxt[min(s0, n), j]
        if s < n:
            return _sold(rec, s, m.O[s, j], "S3", s - s0, True)
        return _no_exit(m, rec, j, s0)

    stop_px = buy * (1 - rule.stop) if rule.stop is not None else None
    take_px = buy * (1 + rule.take) if rule.take is not None else None
    pending = False                               # 못 파는 날 손절선에 닿음 → 다음 팔 수 있는 시가에 판다
    for i in range(b + 1, min(n, m.last[j] + 1)):   # S1은 매수 다음 날부터 본다 (기준 4.2)
        ok = m.sellable[i, j]
        o, h, l = m.O[i, j], m.H[i, j], m.L[i, j]
        if pending:
            if ok:
                return _sold(rec, i, o, "S1손절", i - s0 if i > s0 else 0, True)
            continue
        if i >= s0 and ok:                        # 4A: 시가 매도(S3)가 장중 손절보다 먼저
            return _sold(rec, i, o, "S3", i - s0, True)
        if not ok:
            # 가격이 없는 날(시가 0)의 저가 0 은 진짜 가격이 아니다 → 손절선에 닿은 것으로 보지 않는다 (v2)
            if stop_px is not None and m.priced[i, j] and l <= stop_px:
                pending = True
            continue
        if stop_px is not None and o <= stop_px:  # 시가부터 손절선 아래 → 시가에 판다
            return _sold(rec, i, o, "S1손절", 0, True)
        if take_px is not None and o >= take_px:  # 시가부터 익절선 위 → 시가에 판다
            return _sold(rec, i, o, "S1익절", 0, True)
        if stop_px is not None and l <= stop_px:  # 둘 다 닿으면 손절 먼저 (S1_SAME_DAY="stop")
            return _sold(rec, i, stop_px, "S1손절", 0, False)
        if take_px is not None and h >= take_px:
            return _sold(rec, i, take_px, "S1익절", 0, False)
    return _no_exit(m, rec, j, s0)


def _sold(rec, s, px, reason, delay, at_open):
    return {**rec, "status": FILLED, "s_idx": s, "sell_px": px, "reason": reason,
            "delay": int(delay), "sell_at_open": at_open}


def _no_exit(m: _Market, rec, j, s0):
    """팔 날을 못 찾았을 때: 상장폐지면 마지막 종가 정산, 아니면 데이터 끝을 넘어서 미완료."""
    last = m.last[j]
    if last < m.n - 1:                            # 전체 마지막 날보다 먼저 자료가 끝남 = 상장폐지
        return _sold(rec, last, m.Cl[last, j], DELIST, max(last - s0, 0), False)
    return {**rec, "status": OPEN_END}


def simulate_trades(entry: pd.DataFrame, data: dict, rule: ExitRule, dedupe: bool = True,
                    _m: _Market | None = None) -> pd.DataFrame:
    """매수 신호 표 → 매매 표 (신호 단독 평가, 기준 5.3).

    - t일 유동성 필터(tradable)를 통과하고 2012-07-01 이후인 신호만 쓴다 (기준 5.1)
    - dedupe=True 면 보유 중인 종목의 신호는 '중복_무시'로 적고 사지 않는다
    """
    m = _m or _Market(data)
    e = entry.reindex(index=m.dates, columns=m.codes, fill_value=False).fillna(False).astype(bool)
    keep = data["tradable"].to_numpy(bool) & np.asarray(m.dates >= C.SIGNAL_START)[:, None]
    sig = e.to_numpy(bool) & keep
    # nonzero 는 True 인 칸의 (행, 열) 번호를 모두 꺼내는 명령이다
    ti, ji = np.nonzero(sig)
    order = np.lexsort((ti, ji))                  # 종목별 → 날짜 순서 (중복 판단을 앞에서부터 하기 위해)
    held_until = np.full(len(m.codes), -1)        # 종목마다 '이 날까지 들고 있다' (매도일 행 번호)
    rows = []
    for k in order:
        t, j = int(ti[k]), int(ji[k])
        if dedupe and held_until[j] > t:          # t일 장 마감 때 아직 보유 중 → 무시
            rec = dict(t_idx=t, b_idx=-1, s_idx=-1, buy_px=np.nan, sell_px=np.nan,   # 사려고 하지 않음
                       reason=None, delay=0, sell_at_open=False, status=DUPLICATE)
        else:
            rec = _one_trade(m, t, j, rule)
            if rec["status"] == FILLED:
                held_until[j] = rec["s_idx"]
            elif rec["status"] == OPEN_END and not np.isnan(rec["buy_px"]):
                held_until[j] = m.n               # 산 뒤 끝까지 못 팖 → 이후 신호는 모두 중복
        rec["code"] = m.codes[j]
        rec["market"] = m.market[t, j]
        rows.append(rec)
    return _to_table(rows, m)


def _to_table(rows, m: _Market) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TRADE_COLS)
    df = pd.DataFrame(rows)
    dates = np.append(m.dates.to_numpy(), np.datetime64("NaT", "ns"))
    t, b, s = (df[k].to_numpy(int) for k in ("t_idx", "b_idx", "s_idx"))
    df["signal_date"] = dates[t]
    df["buy_date"] = dates[np.where((b >= 0) & (b < m.n), b, m.n)]   # 끝을 넘거나 중복이면 빈 날짜(NaT)
    df["exit_date"] = dates[np.where(s >= 0, s, m.n)]
    df["ret_gross"] = df["sell_px"] / df["buy_px"] - 1
    df["ret_net"] = df["ret_gross"] - C.COST_SELECT        # 왕복 0.5% (세금 포함, 결정 3A)
    df["ret_net_025"] = df["ret_gross"] - C.COST_ROUNDTRIP  # 왕복 0.25% (참고)
    df = df[TRADE_COLS].sort_values(["signal_date", "code"]).reset_index(drop=True)
    return df


def delisting_worst_case(trades: pd.DataFrame) -> pd.DataFrame:
    """민감도 검사용 사본: 상장폐지로 정산한 매매를 모두 −100%(매도가 0원)로 바꾼다 (11/20 결정 ③).

    왜 필요한가:
    - 기본 엔진은 상장폐지 종목을 '마지막 거래일 종가'로 정산한다.
    - 그런데 거래정지 중에 폐지되면 마지막 종가는 '정지 전 가격'이라, 실제(정리매매에서 크게 떨어짐)보다
      덜 잃은 것으로 계산될 수 있다. 결과를 좋게 만드는 쪽의 오류다.
    - 합병(주식 교환, 손실 거의 없음)과 부실 폐지를 구분할 자료가 없어서, 모두 0원으로 보는 '가장 나쁜 경우'를 쓴다.
    - 두 계산에서 결론(통과·순위)이 같으면 상장폐지 처리 방식이 결론을 바꾸지 않았다고 본다.
    """
    out = trades.copy()
    hit = out["reason"].eq(DELIST)
    out.loc[hit, "sell_px"] = 0.0
    out.loc[hit, "ret_gross"] = -1.0
    out["ret_net"] = out["ret_gross"] - C.COST_SELECT
    out["ret_net_025"] = out["ret_gross"] - C.COST_ROUNDTRIP
    return out


# ---------------------------------------------------------------- 포트폴리오 (기준 3장)
def run_portfolio(signal_count: pd.DataFrame, data: dict, rule: ExitRule, market: str,
                  capital: float = C.INIT_CAPITAL, max_positions: int = C.MAX_POSITIONS,
                  weight: float = C.WEIGHT_PER_POSITION, cost: float = C.COST_SELECT):
    """시장 하나의 포트폴리오를 하루씩 돌린다 (기준 3장).

    signal_count : 날짜×종목, 그날 켜진 신호 개수 (신호 하나면 True/False 표도 된다)
    돌려주는 값  : (daily, trades)
      daily  — 날짜별 현금·평가금액·보유 종목 수
      trades — 실제로 산 매매 표 (+ 슬롯이 있었지만 갭·거래정지로 못 산 후보)

    규칙
    - t일 신호 → t+1 시가 매수. 매수 순서: 신호 개수 많은 순 → 20일 평균 거래대금 큰 순 → 종목코드 순
    - 빈 자리 = 최대 종목 수 − (그날 시가 매도 뒤 남은 보유 종목)
    - 한 종목 금액 = min(5% × t일 종가 기준 평가금액, t일 현금 중 남은 돈). 주식 수는 소수도 허용
    - 고른 후보가 갭·거래정지로 못 사면 그 자리는 다음 후보로 채우지 않는다
    - 판 돈은 그날 장이 끝난 뒤 현금에 들어온다 (같은 날 다시 쓰지 않음)
    - 비용(왕복)은 팔 때 산 금액 × cost 를 뺀다
    - 평가금액은 매일 수정 종가로 잰다
    """
    m = _Market(data)
    cnt = signal_count.reindex(index=m.dates, columns=m.codes, fill_value=0).fillna(0).astype(int)
    cand = simulate_trades(cnt > 0, data, rule, dedupe=False, _m=m)
    cand = cand[cand["market"] == market].copy()
    cnt_np = cnt.to_numpy()
    jpos = {c: k for k, c in enumerate(m.codes)}
    cand["j"] = cand["code"].map(jpos)
    cand["n_sig"] = cnt_np[cand["t_idx"].to_numpy(), cand["j"].to_numpy()]
    cand["v20"] = m.value20[cand["t_idx"].to_numpy(), cand["j"].to_numpy()]
    cand = cand.sort_values(["b_idx", "n_sig", "v20", "code"], ascending=[True, False, False, True])
    by_day = {b: g for b, g in cand.groupby("b_idx", sort=False)}

    # 평가용 종가: 거래정지 등으로 빈 날은 직전 종가를 쓴다 (ffill = 앞의 값으로 채우기)
    close_ff = pd.DataFrame(m.Cl).ffill().to_numpy()
    cash = float(capital)
    equity_prev = float(capital)
    held: dict[str, dict] = {}
    log, daily = [], []
    for i in range(m.n):
        proceeds = 0.0
        held_at_T = set(held)                     # t일 장 마감 때 들고 있던 종목 (중복 매수 금지)
        # 1) 시가 매도
        for code in [c for c, p in held.items() if p["s_idx"] == i and p["sell_at_open"]]:
            proceeds += _close_position(held.pop(code), cost, log)
        # 2) 시가 매수 (t = i−1 의 신호)
        budget = cash
        free = max_positions - len(held)
        if i in by_day and free > 0:
            picks = by_day[i][~by_day[i]["code"].isin(held_at_T)].head(free)
            for r in picks.itertuples(index=False):
                if r.status in (GAP_CANCEL, NO_BUY):
                    log.append(dict(r._asdict(), shares=0.0, notional=0.0, pnl=0.0))
                    continue
                size = min(weight * equity_prev, budget)
                if size <= 0:
                    continue
                budget -= size
                held[r.code] = dict(r._asdict(), shares=size / r.buy_px, notional=size)
        cash = budget
        # 3) 장중(S1)·종가(상장폐지) 매도
        for code in [c for c, p in held.items() if p["s_idx"] == i]:
            proceeds += _close_position(held.pop(code), cost, log)
        # 4) 판 돈은 장 마감 뒤 현금으로
        cash += proceeds
        value = sum(p["shares"] * close_ff[i, p["j"]] for p in held.values())
        equity_prev = cash + value
        daily.append(dict(date=m.dates[i], cash=cash, equity=equity_prev, n_pos=len(held)))
    for p in held.values():                       # 끝까지 들고 있는 종목 (미완료)
        log.append(dict(p, pnl=np.nan))
    return pd.DataFrame(daily).set_index("date"), pd.DataFrame(log)


def _close_position(p: dict, cost: float, log: list) -> float:
    gross = p["shares"] * p["sell_px"]
    fee = cost * p["notional"]
    log.append(dict(p, pnl=gross - fee - p["notional"]))
    return gross - fee
