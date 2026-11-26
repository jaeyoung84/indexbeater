"""분할 보정 후에도 남은 '하루 큰 변동'을 찾아서 종류별로 나누는 파일. 데이터를 고치지 않고 표시만 한다.

# 왜 필요한가
- 분할을 보정했는데도 하루에 -50%, +300% 같은 변동이 남아 있으면 두 가지 중 하나다.
  (1) 진짜로 주가가 크게 움직였다 (2) 우리가 못 찾은 주식 수 변화(무상증자·감자 등)나 자료 이상이다.
- 한국 주식은 하루에 ±30%까지만 움직일 수 있다. 그래서 31%를 넘으면 '이상 신호'로 본다.

# 종류를 나누는 기준 (네이버를 '제2의 눈'으로 쓴다)
- 네이버는 무상증자·감자 같은 것도 보정해 놓은 것으로 보인다. 같은 날 KRX는 크게 변했는데 네이버는 그대로면
  → '우리가 못 찾은 주식 수 변화일 가능성'.
- 둘이 같이 크게 변했으면 → '진짜 시장 변동'(또는 둘 다 모르는 변화).
  A_1원 : 가격이 1원 이하 (자료에 넣어 둔 임시 값)
  B     : 거래정지(전날 거래량 0, 또는 5일 넘게 쉼) 후 재개 + 네이버도 같은 변동
  C     : 거래정지 아님 + 네이버도 같은 변동  → 실제 큰 변동
  D     : 거래정지 후 재개 + 네이버는 다른 값  → 정지 중 주식 수 변화 의심
  E     : 거래정지 아님 + 네이버는 다른 값    → 주식 수 변화 의심 (제일 중요)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import external, paths

LIMIT = 0.31          # 하루 상·하한(±30%)을 넘는 변동만 본다
AGREE_TOL = 0.02      # KRX 수익률과 네이버 수익률이 2%p 안이면 '같다'고 본다
HALT_DAYS = 5         # 이만큼(일) 넘게 거래 기록이 없으면 정지 후 재개로 본다
KINDS = {"A": "A_1원", "B": "B_정지후_네이버일치", "C": "C_실제변동_네이버일치",
         "D": "D_정지후_네이버만보정", "E": "E_네이버만보정"}


def find_jumps(adj: pd.DataFrame, limit: float = LIMIT) -> pd.DataFrame:
    """분할만 보정한 표(close_s 열 필요)에서 하루 변동이 limit 을 넘는 줄을 모은다."""
    k = adj[adj["market"].isin(["KOSPI", "KOSDAQ"])].sort_values(["code", "date"]).reset_index(drop=True)
    g = k.groupby("code")
    k["prev_close_s"] = g["close_s"].shift()          # 같은 종목의 어제 값 (shift = 한 칸 아래로 밀기)
    k["prev_volume"] = g["volume"].shift()
    k["prev_date"] = g["date"].shift()
    k["ret"] = k["close_s"] / k["prev_close_s"] - 1
    return k[k["ret"].abs() > limit][["date", "code", "name", "market", "close", "close_s", "prev_close_s",
                                      "prev_volume", "prev_date", "ret", "volume"]].reset_index(drop=True)


def classify(jumps: pd.DataFrame) -> pd.DataFrame:
    """각 큰 변동에 네이버 수익률을 붙이고 A~E 종류를 매긴다."""
    j = jumps.copy()
    nret = []
    for r in j.itertuples():
        n = external.load_naver(r.code)
        a = n["close"].get(r.date, np.nan) if n is not None else np.nan
        b = n["close"].get(r.prev_date, np.nan) if n is not None else np.nan
        nret.append(a / b - 1 if b == b and b else np.nan)
    j["naver_ret"] = nret
    j["gap_days"] = (j["date"] - j["prev_date"]).dt.days
    j["halt"] = (j["prev_volume"] == 0) | (j["gap_days"] > HALT_DAYS)
    j["placeholder"] = (j["close"] <= 1) | (j["prev_close_s"] <= 1)
    j["agree"] = (j["naver_ret"] - j["ret"]).abs() < AGREE_TOL
    def one(r):
        if r.placeholder:
            return KINDS["A"]
        if r.agree:
            return KINDS["B"] if r.halt else KINDS["C"]
        return KINDS["D"] if r.halt else KINDS["E"]
    j["kind"] = j.apply(one, axis=1)
    return j


def main(argv=None):
    """터미널에서 `python -m indexbeater.krx_source jumps` 로 실행. 결과를 csv 로 저장한다."""
    adj = pd.read_parquet(paths.ADJUSTED_SPLIT_ONLY, columns=["date", "market", "code", "name", "close", "close_s", "volume"])
    out = classify(find_jumps(adj))
    out.to_csv(paths.JUMP_REVIEW, index=False, encoding="utf-8-sig")
    print(out["kind"].value_counts().to_string(), f"\n→ {paths.JUMP_REVIEW}")


if __name__ == "__main__":
    main()
