"""액면분할·무상증자를 찾아내는 파일 (계획서 2.2절). KRX 데이터만 쓴다.

# 왜 필요한가
- 삼성전자는 2018년에 주식을 50개로 쪼갰다(액면분할). 그러면 가격이 265만 원 → 5만 원으로 보인다.
  그냥 보면 "폭락"처럼 보이지만 실제 가치는 그대로다. 그래서 과거 가격을 같은 비율로 맞춰 줘야 한다.
- 먼저 "언제, 몇 배로 쪼개졌는지"를 찾아야 한다. 이 파일이 그 일을 한다.

# 찾는 방법 (3단계)
1. 후보 찾기: 상장주식수(회사가 세상에 내놓은 주식 개수)가 하루 만에 1.3배 이상 늘었거나,
   1/1.3 이하로 줄어든 날을 찾는다. (줄어드는 건 주식 병합·감자)
2. 확인하기: 그 날짜 근처(앞 10거래일 ~ 뒤 3거래일, 오차 15%)에서 가격도 같은 비율로 줄었는지 본다.
   무상증자는 새 주식이 늦게 상장되므로 앞 60거래일까지는 오차 3%로 더 엄격하게 본다.
   - 종가 비율: 어제 종가 / 오늘 종가
   - 기준가 비율: 어제 종가 / 오늘 기준가  (KRX가 분할일에 조정해 준 기준가)
   둘 중 하나라도 주식 수 비율과 거의 같으면(오차 15% 이내) "진짜 분할"로 확정한다.
   보정을 시작하는 날(ex_date)은 주식 수가 바뀐 날이 아니라 '가격이 바뀐 날'로 한다.
   (무상증자는 가격이 먼저 바뀌고 새 주식은 며칠 뒤에 상장돼서 날짜가 어긋난다.)
3. 모르겠으면 멈추기: 주식 수만 늘고 가격은 그대로면(유상증자 등) 자동으로 넘기지 않고
   "사람이 확인할 목록(review)"으로 따로 뺀다.

# 알려진 한계
- 주식 수가 30% 미만으로 늘어난 작은 무상증자(예: 10%)는 후보에 오르지 않는다. (계획서 기준을 그대로 따름)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import paths

SHARE_JUMP = 1.3      # 주식 수가 이 배수 이상 바뀌면 후보로 본다
PRICE_TOL = 0.15      # 주식 수 비율과 가격 비율이 이 정도(15%)까지 달라도 같은 것으로 본다
WIN_BEFORE = 10       # 후보 날짜보다 며칠 앞까지 가격 변화를 찾아볼지
WIN_AFTER = 3         # 후보 날짜보다 며칠 뒤까지 찾아볼지
WIN_FAR = 60          # 무상증자는 새 주식이 가격 변화 한참 뒤에 상장돼서, 더 먼 과거(60거래일 전)까지도 본다
TOL_FAR = 0.03        # 단, 먼 곳에서 찾은 건 우연히 비슷할 수 있어서 훨씬 엄격하게(3%) 맞아야 인정한다
STOCK_MARKETS = ("KOSPI", "KOSDAQ")   # 분할을 찾을 시장. ETF는 지수 비교용이라 여기서는 뺀다


def _deviation(price_ratio: float, share_ratio: float) -> float:
    """가격 비율이 주식 수 비율과 얼마나 다른지(0이면 딱 같다)를 돌려준다."""
    if not np.isfinite(price_ratio) or price_ratio <= 0:   # 계산이 안 되는 값이면 '엄청 다르다'로 취급
        return np.inf
    return abs(price_ratio / share_ratio - 1.0)


def detect_one(g: pd.DataFrame, share_jump=SHARE_JUMP, tol=PRICE_TOL, before=WIN_BEFORE, after=WIN_AFTER,
               far=WIN_FAR, tol_far=TOL_FAR):
    """종목 1개(날짜 오름차순)에서 분할을 찾는다. (확정 목록, 사람 확인 목록)을 돌려준다."""
    # 표에서 열을 꺼내 numpy 배열로 바꾼다. 반복 계산할 때 pandas보다 훨씬 빠르기 때문이다.
    d = g["date"].to_numpy()
    sh = g["listed_shares"].to_numpy(float)
    cl = g["close"].to_numpy(float)
    base = g["base_price"].to_numpy(float) if "base_price" in g else np.full(len(g), np.nan)
    events, review = [], []
    if len(g) < 2:                          # 하루치뿐이면 비교할 어제가 없다
        return events, review
    with np.errstate(divide="ignore", invalid="ignore"):   # 0으로 나눠도 경고 대신 넘어가게 한다
        ratio = sh[1:] / sh[:-1]            # 오늘 주식 수 / 어제 주식 수
    # 1단계: 후보 날짜 찾기 (많이 늘었거나 많이 줄었거나)
    for k in np.flatnonzero((ratio >= share_jump) | (ratio <= 1 / share_jump)):
        i = k + 1                           # ratio는 하루씩 밀려 있어서 +1 해야 '바뀐 날'의 위치가 된다
        r = float(ratio[k])
        best = (np.inf, None, None)         # (가장 비슷한 정도, 그 날짜 위치, 종가/기준가 중 무엇인지) — 허용 오차 안에서만
        closest = np.inf                    # 허용 오차와 상관없이 가장 비슷했던 정도 (확인 목록에 참고로 적는다)
        # 2단계: 후보 날짜 주변에서 가격이 가장 비슷하게 움직인 날을 찾는다
        for t in range(max(1, i - max(before, far)), min(len(g), i + after + 1)):
            limit = tol if t >= i - before else tol_far     # 가까운 곳은 15%, 먼 곳은 3%까지만 인정
            for kind, den in (("close", cl[t]), ("base", base[t])):
                dev = _deviation(cl[t - 1] / den if den and np.isfinite(den) else np.nan, r)
                closest = min(closest, dev)
                if dev <= limit and dev < best[0]:
                    best = (dev, t, kind)
        row = {"share_change_date": pd.Timestamp(d[i]), "share_ratio": round(r, 6)}
        if best[1] is not None:             # 허용 오차 안에서 맞는 날을 찾았으면 진짜 분할로 확정
            _, t, kind = best
            events.append({**row, "ex_date": pd.Timestamp(d[t]), "evidence": kind,
                           "price_dev": round(best[0], 4), "kind": "split" if r > 1 else "reverse_split"})
        else:                               # 3단계: 가격 근거가 없으면 사람이 보게 따로 뺀다
            review.append({**row, "reason": "shares_jump_without_matching_price_move",
                           "best_price_dev": None if not np.isfinite(closest) else round(closest, 4)})
    return events, review


def detect_splits(raw: pd.DataFrame, markets=STOCK_MARKETS, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """선택한 시장의 종목마다 detect_one 을 돌리고 결과를 표 두 개로 모은다.
    기본은 코스피·코스닥 개별 종목만 본다. ETF는 설정·환매 때문에 주식 수(상장좌수)가 수시로 바뀌는데
    이건 분할이 아니라서, 넣으면 '사람이 확인할 목록'만 쓸데없이 불어난다."""
    ev, rv = [], []
    df = raw[raw["market"].isin(markets)]   # 볼 시장만 남긴다
    # 주식 수나 종가가 비었거나 0 이하인 줄은 계산이 안 되니 미리 뺀다
    df = df.dropna(subset=["listed_shares", "close"])
    df = df[(df["listed_shares"] > 0) & (df["close"] > 0)]
    # groupby = 종목코드가 같은 줄끼리 묶어서 한 종목씩 꺼내 주는 명령
    for code, g in df.sort_values(["code", "date"]).groupby("code", sort=False):
        e, r = detect_one(g.reset_index(drop=True), **kw)
        name = g["name"].iloc[-1]           # 종목명은 가장 최근 이름을 쓴다 (이름이 바뀐 회사도 있어서)
        ev += [{"code": code, "name": name, **x} for x in e]
        rv += [{"code": code, "name": name, **x} for x in r]
    events = pd.DataFrame(ev, columns=["code", "name", "ex_date", "share_change_date", "share_ratio",
                                       "evidence", "price_dev", "kind"])
    review = pd.DataFrame(rv, columns=["code", "name", "share_change_date", "share_ratio", "reason", "best_price_dev"])
    return events, review


def main(argv=None):
    """터미널에서 `python -m indexbeater.krx_source splits` 로 실행할 때 쓰는 입구."""
    ap = argparse.ArgumentParser(prog="krx_source splits")
    ap.add_argument("--raw", default=str(paths.RAW_DAILY))
    ap.add_argument("--share-jump", type=float, default=SHARE_JUMP)
    ap.add_argument("--price-tol", type=float, default=PRICE_TOL)
    ap.add_argument("--markets", nargs="+", default=list(STOCK_MARKETS), choices=["KOSPI", "KOSDAQ", "ETF"])
    a = ap.parse_args(argv)
    raw = pd.read_parquet(a.raw)
    events, review = detect_splits(raw, markets=tuple(a.markets), share_jump=a.share_jump, tol=a.price_tol)
    # utf-8-sig 로 저장해야 엑셀에서 한글이 안 깨진다
    events.to_csv(paths.SPLIT_EVENTS, index=False, encoding="utf-8-sig")
    review.to_csv(paths.SPLIT_REVIEW, index=False, encoding="utf-8-sig")
    print(f"분할 확정 {len(events)}건 → {paths.SPLIT_EVENTS}\n수동 확인 {len(review)}건 → {paths.SPLIT_REVIEW}")


if __name__ == "__main__":
    main()
