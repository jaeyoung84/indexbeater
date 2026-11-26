"""유상증자(주주에게 돈을 받고 새 주식을 파는 것) 공시를 OpenDART에서 받아 오는 파일.

# 왜 필요한가
- 주주배정 유상증자는 '권리락일'에 주가가 이론값만큼 내려간다. 회사 가치가 줄어서가 아니라,
  싼 값에 새 주식을 살 권리가 빠져나가기 때문이다. 이 하락을 그냥 두면 가짜 하락으로 보인다.
- 확인해 보니 네이버도 이 하락을 과거 가격에 반영하지 않았다. 그래서 정답지로 쓸 수 없고 KRX 가격으로 직접 찾는다.
- 상장주식수 변화로는 알 수 없다(신주는 몇 주 뒤에야 상장된다). 그래서 공시를 따로 받는다.

# 받는 것 (원자료만 저장한다. 해석은 받은 파일을 눈으로 보고 정한다)
- piicDecsn.json   : 유상증자 결정
- pifricDecsn.json : 유무상증자 결정 (유상 + 무상을 함께 하는 경우)
- 회사 하나에 대해 2011-01-01 ~ 올해 끝까지 기간을 한 번에 물어본다 (회사당 2번).
- 받은 답은 data/dart_raw/rights/ 에 그대로 저장한다. 한도가 차면 다음 날 같은 명령으로 이어서 받는다.

# 키는 환경변수 OPENDART_KEY 에서만 읽는다 (코드에 적지 않는다)
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd
import requests

from . import paths
from .dividends import _get_json, load_corp_codes

ENDPOINTS = ("piicDecsn", "pifricDecsn")
SHAREHOLDER_TYPES = ("주주배정증자", "주주배정후 실권주 일반공모", "주주우선공모증자")   # 권리락일이 있는 방식만
FUND_COLS = ["fdpp_fclt", "fdpp_bsninh", "fdpp_op", "fdpp_dtrp", "fdpp_ocsa", "fdpp_etc"]
SEARCH_FROM, SEARCH_TO = 4, 150        # 공시일 며칠 뒤부터 며칠 뒤까지 권리락일을 찾을지
MIN_EFFECT = 0.97                      # 이론 하락이 3% 미만이면 무시한다
FIT_TOL = 0.02                         # 실제 등락과 이론값 차이가 2%p 이내여야 '확정'
SHARE_LO, SHARE_HI = 0.8, 1.2          # 상장주식수 증가분이 신주 수의 80~120% 이면 '신주 상장 확인'
DEDUP_DAYS = 20                        # 같은 종목에서 권리락일이 20일 안이면 같은 증자의 정정 공시로 본다
FROM, TO = "20110101", "20261231"


def fetch_rights(codes, progress_every=100) -> int:
    """종목마다 두 종류의 증자 공시를 받아 저장한다. 실제 받은(새로 물어본) 회사 수를 돌려준다."""
    n_new = 0
    with requests.Session() as s:
        corp = load_corp_codes(s).set_index("stock_code")["corp_code"]
        for n, code in enumerate(codes, 1):
            if code not in corp.index:      # DART에 없는 종목은 건너뛴다
                continue
            cc = corp[code]
            for ep in ENDPOINTS:
                f = paths.DART_CACHE / "rights" / f"{ep}_{cc}.json"
                if not f.exists():
                    n_new += 1
                _get_json(s, f"{ep}.json", {"corp_code": cc, "bgn_de": FROM, "end_de": TO}, f"rights/{ep}_{cc}.json")
            if n % progress_every == 0:
                print(f"  {n}/{len(codes)}종목", flush=True)
    return n_new


def load_candidates() -> pd.DataFrame:
    """받아 둔 piicDecsn 파일에서 주주배정 계열 유상증자만 모아 표로 만든다. 발행가는 '조달금액 / 신주 수'로 구한다."""
    rows = []
    for f in glob.glob(str(paths.DART_CACHE / "rights" / "piicDecsn_*.json")):
        with open(f, encoding="utf-8") as fh:
            rows += json.load(fh).get("list", []) or []
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d = d[d["ic_mthn"].isin(SHAREHOLDER_TYPES)].copy()
    for c in ["nstk_ostk_cnt", "bfic_tisstk_ostk"] + FUND_COLS:
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")   # '1,000' 같은 글자를 숫자로
    d["price"] = d[FUND_COLS].sum(axis=1, min_count=1) / d["nstk_ostk_cnt"]
    d["rdate"] = pd.to_datetime(d["rcept_no"].str[:8])
    corp = pd.read_csv(paths.DART_CACHE / "corpCode.csv", dtype=str)
    d = d.merge(corp[["corp_code", "stock_code"]], on="corp_code")
    return d.dropna(subset=["nstk_ostk_cnt", "bfic_tisstk_ostk", "price", "stock_code"])


def find_ex_date(cand, x: pd.DataFrame, split_days: set) -> dict | None:
    """한 증자의 권리락일을 찾는다. x: 그 종목의 원주가(date, close, listed_shares), 날짜순.

    공시 며칠 뒤~150일 안에서, 하루 등락이 이론값 f 와 가장 가까운 날을 고른다. (분할일은 뺀다)
    우연히 맞을 수 있으므로 '상장주식수가 신주 수만큼 늘었는지'를 따로 확인해 같이 기록한다.
    """
    N, n, Pi = cand.bfic_tisstk_ostk, cand.nstk_ostk_cnt, cand.price
    x = x.assign(pc=x["close"].shift())
    w = x[(x["date"] >= cand.rdate + pd.Timedelta(days=SEARCH_FROM)) & (x["date"] <= cand.rdate + pd.Timedelta(days=SEARCH_TO))]
    w = w[[(cand.stock_code, t) not in split_days for t in w["date"]]].copy()
    if w.empty:
        return None
    w["f"] = ((w["pc"] * N + Pi * n) / (N + n)) / w["pc"]      # 권리락 이론가 / 전일 종가
    w = w[w["f"] < MIN_EFFECT]
    if w.empty:
        return None
    w["err"] = (w["close"] / w["pc"] - w["f"]).abs()
    b = w.loc[w["err"].idxmin()]
    before = x.loc[x["date"] < b["date"], "listed_shares"]
    after = x.loc[(x["date"] >= b["date"]) & (x["date"] <= b["date"] + pd.Timedelta(days=120)), "listed_shares"]
    ratio = (after.max() - before.iloc[-1]) / n if len(before) and len(after) else np.nan   # 늘어난 주식 수 / 신주 수
    conf = "confirmed" if (SHARE_LO <= ratio <= SHARE_HI and b["err"] <= FIT_TOL) else ("review" if SHARE_LO <= ratio <= SHARE_HI else "unconfirmed")
    return dict(code=cand.stock_code, rcept_no=cand.rcept_no, ic_mthn=cand.ic_mthn, ex_date=b["date"], factor=float(b["f"]),
                obs_ratio=float(b["close"] / b["pc"]), fit_err=float(b["err"]), issue_price=float(Pi), share_ratio=float(ratio), confidence=conf)


def build_events(raw: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """모든 후보에 대해 권리락일을 찾고, 같은 증자의 정정 공시는 하나로 줄인다. raw: date, code, close, listed_shares."""
    cands = load_candidates()
    split_days = set(zip(splits["code"], pd.to_datetime(splits["ex_date"])))
    by = {c: g.sort_values("date").reset_index(drop=True) for c, g in raw.groupby("code")}
    out = []
    for cand in cands.itertuples():
        x = by.get(cand.stock_code)
        r = find_ex_date(cand, x, split_days) if x is not None else None
        if r:
            out.append(r)
    ev = pd.DataFrame(out)
    if ev.empty:
        return ev
    rank = {"confirmed": 0, "review": 1, "unconfirmed": 2}
    ev["_r"] = ev["confidence"].map(rank)
    ev = ev.sort_values(["code", "_r", "fit_err"])
    keep = []
    for _, g in ev.groupby("code"):      # 신뢰가 높은 것부터 보며, 20일 안에 이미 고른 것이 있으면 중복으로 버린다
        kept = []
        for r in g.itertuples():
            if all(abs((r.ex_date - k).days) > DEDUP_DAYS for k in kept):
                kept.append(r.ex_date)
                keep.append(r.Index)
    return ev.loc[keep].drop(columns="_r").sort_values(["code", "ex_date"]).reset_index(drop=True)


def build_main(argv=None):
    """`python -m indexbeater.krx_source rights --build` : 받아 둔 공시로 권리락 이벤트 파일을 만든다 (키 불필요)."""
    raw = pd.read_parquet(paths.RAW_DAILY, columns=["date", "code", "close", "listed_shares"])
    sp = pd.read_csv(paths.SPLIT_EVENTS, dtype={"code": str}, parse_dates=["ex_date"])
    ev = build_events(raw, sp)
    ev.to_csv(paths.RIGHTS_EVENTS, index=False, encoding="utf-8-sig")
    print(ev["confidence"].value_counts().to_string())
    print(f"저장: {paths.RIGHTS_EVENTS}")


def main(argv=None):
    """`python -m indexbeater.krx_source rights` 로 실행한다. 키가 필요하다. `--build` 를 붙이면 키 없이 이벤트 파일만 만든다."""
    if argv and "--build" in argv:
        return build_main()
    ap = argparse.ArgumentParser(prog="krx_source rights")
    ap.add_argument("--raw", default=str(paths.RAW_DAILY))
    ap.add_argument("--codes", nargs="+", help="이 종목코드만 받는다 (시험용)")
    a = ap.parse_args(argv)
    if a.codes:
        codes = a.codes
    else:
        raw = pd.read_parquet(a.raw, columns=["market", "code"])
        codes = sorted(raw.loc[raw["market"].isin(["KOSPI", "KOSDAQ"]), "code"].unique())
    n = fetch_rights(codes)
    print(f"완료. 새로 물어본 요청 {n}번. 저장 위치: {paths.DART_CACHE / 'rights'}")


if __name__ == "__main__":
    main()
