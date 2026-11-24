"""KRX Open API에서 일별 전종목 원주가를 받아 오는 파일.

# 이 파일이 하는 일
- KRX에 "이 날짜 시장 전체 시세 줘" 하고 부탁(요청)한다. 하루 한 번 부탁하면 그날 시장 전체가 온다.
- 평일 x 3개 시장(코스피, 코스닥, ETF) = 약 1.2만 번 부탁해야 한다.
- 하루에 부탁할 수 있는 횟수에 한도가 있을 수 있다. 그래서 받은 답은 파일로 저장(캐시)해 둔다.
  중간에 멈춰도 다시 실행하면 이미 받은 날은 건너뛰고 이어서 받는다.
- 쉬는 날(휴장일)은 빈 답이 오는데, 이것도 저장해 둔다. 안 그러면 다음에 또 물어보게 된다.

# 키(비밀번호)는 코드에 적지 않는다
- 키는 환경변수 KRX_AUTH_KEY 에서만 읽는다. 코드나 노트북에 적으면 남에게 보일 수 있기 때문이다.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from indexbeater import config as C

from . import paths

# KRX 주소. 시장마다 뒤쪽 주소가 다르다.
BASE = "https://data-dbg.krx.co.kr/svc/apis/"
ENDPOINTS = {"KOSPI": "sto/stk_bydd_trd", "KOSDAQ": "sto/ksq_bydd_trd", "ETF": "etp/etf_bydd_trd"}

# KRX가 쓰는 영어 이름을 우리가 쓰기 쉬운 짧은 이름으로 바꾸는 표
FIELDS = {
    "ISU_CD": "code", "ISU_NM": "name",
    "TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low", "TDD_CLSPRC": "close",
    "CMPPREVDD_PRC": "chg",               # 전일 대비(어제보다 얼마나 올랐나). 아래 base_price 계산에 쓴다
    "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value", "MKTCAP": "mktcap", "LIST_SHRS": "listed_shares",
}
# 숫자로 바꿔야 하는 열. KRX는 "51,900" 처럼 쉼표 붙은 글자로 주기 때문이다.
NUMERIC = ["open", "high", "low", "close", "chg", "volume", "value", "mktcap", "listed_shares"]


def _num(x):
    """"51,900" 같은 글자를 51900 이라는 숫자로 바꾼다. 못 바꾸면 '빈 값(NaN)'으로 둔다."""
    if x is None:
        return np.nan
    s = str(x).replace(",", "").strip()   # 쉼표가 있으면 숫자로 못 바꾸니까 먼저 지운다
    try:
        return float(s)
    except ValueError:
        return np.nan


def fetch_day(session: requests.Session, key: str, market: str, d: pd.Timestamp, retries: int = 4) -> list[dict]:
    """한 날짜, 한 시장의 원본 데이터를 가져온다. 저장해 둔 파일이 있으면 인터넷을 쓰지 않는다."""
    cache = paths.RAW_CACHE / market / f"{d:%Y%m%d}.json"
    if cache.exists():                      # 전에 받아 둔 게 있으면 그걸 읽고 끝낸다 (시간·호출 횟수 절약)
        return json.loads(cache.read_text(encoding="utf-8"))
    url = BASE + ENDPOINTS[market]
    for attempt in range(retries):          # 인터넷이 잠깐 불안할 수 있어서 몇 번 다시 시도한다
        r = session.get(url, params={"basDd": f"{d:%Y%m%d}"}, headers={"AUTH_KEY": key}, timeout=30)
        if r.status_code == 200:            # 200 = 잘 받았다는 뜻
            rows = r.json().get("OutBlock_1", [])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            return rows
        if r.status_code in (401, 403):     # 키가 틀렸거나, 이 API 사용 승인을 안 받은 경우
            raise SystemExit(f"KRX 인증 실패 {r.status_code}: 키 또는 해당 API 이용 승인 여부 확인 ({market})")
        if r.status_code == 429:            # 오늘 부탁할 수 있는 횟수를 다 썼다는 뜻
            raise SystemExit("KRX 호출 한도 초과(429). 내일 같은 명령을 다시 실행하면 이어서 받는다.")
        time.sleep(2 ** attempt)            # 실패하면 1초, 2초, 4초... 점점 오래 쉬었다가 다시 한다
    raise RuntimeError(f"KRX 호출 실패 {market} {d:%Y-%m-%d}: HTTP {r.status_code}")


def rows_to_frame(rows: list[dict], market: str, d: pd.Timestamp) -> pd.DataFrame:
    """KRX가 준 목록(rows)을 pandas 표(DataFrame)로 바꾸고 깨끗하게 정리한다."""
    if not rows:                            # 휴장일은 빈 목록이 온다 → 빈 표를 돌려준다
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns=FIELDS)   # 이름을 짧은 이름으로 바꾼다
    for c in FIELDS.values():
        if c not in df:                     # 혹시 KRX가 빼먹은 열이 있어도 코드가 안 깨지게 빈 열을 만들어 둔다
            df[c] = np.nan
    df = df[list(FIELDS.values())].copy()   # 필요한 열만 순서대로 남긴다
    for c in NUMERIC:
        df[c] = df[c].map(_num)             # 글자 → 숫자
    # ETF는 쉬는 날에도 가격이 빈 줄이 온다. 종가가 비면 그날 거래가 없었다는 뜻이라 지운다.
    df = df.dropna(subset=["close"])
    if df.empty:
        return pd.DataFrame()
    df["code"] = df["code"].astype(str).str.strip()  # 종목코드 앞뒤 빈칸 제거
    df.insert(0, "market", market)          # 어느 시장인지 열을 추가
    df.insert(0, "date", d)                 # 어느 날짜인지 열을 추가
    # 기준가 = 종가 - 전일대비. 평소엔 '어제 종가'와 같다.
    # 그런데 액면분할·권리락 날에는 KRX가 기준가를 조정해 준다. 그 차이를 나중에 분할을 찾는 힌트로 쓴다.
    df["base_price"] = df["close"] - df["chg"]
    return df.drop(columns="chg")           # 계산이 끝났으니 chg 열은 버린다


def collect(start=C.DATA_START, end=C.DATA_END, markets=tuple(ENDPOINTS), limit_days=None, sleep=0.1) -> pd.DataFrame:
    """기간 전체를 날짜별·시장별로 받아서 표 하나로 합친다."""
    key = os.environ.get("KRX_AUTH_KEY")
    if not key:
        raise SystemExit("환경변수 KRX_AUTH_KEY 가 없습니다 (본인 명의로 발급받은 키).")
    days = pd.bdate_range(start, end)       # 평일(월~금)만 만든다. 주말엔 장이 안 열리니까 물어볼 필요가 없다
    if limit_days:
        days = days[:limit_days]            # 시험 삼아 앞의 며칠만 받고 싶을 때
    frames = []
    with requests.Session() as s:           # Session = 같은 연결을 계속 써서 더 빠르게 받는 방법
        for i, d in enumerate(days, 1):
            for m in markets:
                f = rows_to_frame(fetch_day(s, key, m, d), m, d)
                if len(f):
                    frames.append(f)
            time.sleep(sleep)               # 너무 빠르게 계속 부탁하면 KRX가 싫어할 수 있어서 조금 쉰다
            if i % 100 == 0:
                print(f"  {i}/{len(days)}일 ({d:%Y-%m-%d})", flush=True)   # 진행 상황 알림
    out = pd.concat(frames, ignore_index=True)       # 조각 표들을 위아래로 이어 붙인다
    out = out.sort_values(["code", "date"]).reset_index(drop=True)   # 종목별, 날짜순으로 정렬
    return out


def main(argv=None):
    """터미널에서 `python -m indexbeater.krx_source collect` 로 실행할 때 쓰는 입구."""
    ap = argparse.ArgumentParser(prog="krx_source collect")
    ap.add_argument("--start", default=str(C.DATA_START.date()))
    ap.add_argument("--end", default=str(C.DATA_END.date()))
    ap.add_argument("--limit-days", type=int, help="시험용: 앞에서 N일만")
    ap.add_argument("--markets", nargs="+", default=list(ENDPOINTS), choices=list(ENDPOINTS))
    a = ap.parse_args(argv)
    df = collect(pd.Timestamp(a.start), pd.Timestamp(a.end), tuple(a.markets), a.limit_days)
    # 시험용으로 받은 건 진짜 파일을 덮어쓰지 않게 이름을 따로 한다
    out = paths.RAW_DAILY if not a.limit_days else paths.RAW_DAILY.with_name("krx_raw_daily_sample.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"저장: {out}  행 {len(df):,}  종목 {df['code'].nunique():,}  {df['date'].min():%F} ~ {df['date'].max():%F}")


if __name__ == "__main__":
    main()
