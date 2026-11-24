"""배당금을 OpenDART에서 받아 와서 '배당 이벤트'로 만드는 파일 (계획서 2.3절).

# 왜 필요한가
- 배당을 주는 날(배당락일) 주가는 배당금만큼 내려간다. 이걸 모르면 "주가가 떨어졌네" 하고 오해한다.
- 그래서 회사별로 "몇 년도에 주당 얼마를 줬고, 언제 가격이 빠졌는지"를 알아야 한다.

# 문제: 날짜를 모른다
- OpenDART의 배당 자료(alotMatter)는 '그 해에 주당 얼마'만 알려 주고, 배당락일(날짜)은 알려 주지 않는다.
- 그래서 날짜는 추정한다.
  - 12월 결산 회사: 그 해 마지막 거래일의 바로 전 거래일을 배당락일로 본다. (예: 12/29가 마지막 날이면 12/28)
  - 신뢰도(confidence): 2022 사업연도까지는 'estimated'(추정), 2023부터는 'low'(낮음).
    2023부터는 배당 기준일을 주주총회 뒤로 정하는 회사가 많아서 연말 가정이 틀릴 수 있다.
  - 12월 결산이 아닌 회사는 자동 처리하지 않고 사람이 볼 목록(review)으로 뺀다.
- 정확한 날짜를 아는 건 직접 CSV(DIV_OVERRIDE)에 적어 덮어쓴다. 그러면 'confirmed'(확정)가 된다.
- 중간배당(1년에 여러 번 주는 배당)은 연간 합계로 합쳐서 연말 한 번에 처리한다. 시점이 조금 어긋나는 근사이고, 한계로 기록해 둔다.

# 호출 횟수
- 회사 약 2,500개 x (기업개황 1번 + 연도 15번) = 약 4만 번. OpenDART는 하루 약 2만 번 한도다.
- 그래서 받은 답을 전부 파일로 저장(캐시)하고, 한도가 차면 멈췄다가 다음 날 같은 명령으로 이어서 받는다.

# 키는 환경변수 OPENDART_KEY 에서만 읽는다 (코드에 적지 않는다)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import requests

from . import paths

API = "https://opendart.fss.or.kr/api/"
REQUEST_GAP = 0.4                           # DART에 물어볼 때마다 쉬는 시간(초). 약 1분에 150번 이하로 천천히
ANNUAL = "11011"                    # 사업보고서(1년에 한 번 내는 보고서)를 뜻하는 DART의 번호
FIRST_LOW_CONFIDENCE_FY = 2023      # 이 사업연도부터는 배당락일 추정을 '낮음'으로 표시


def _key() -> str:
    """환경변수에서 OpenDART 키를 꺼낸다. 없으면 이유를 알려 주고 멈춘다."""
    k = os.environ.get("OPENDART_KEY")
    if not k:
        raise SystemExit("환경변수 OPENDART_KEY 가 없습니다 (본인 명의로 발급받은 키).")
    return k


def _num(x):
    """"1,444" 같은 글자를 숫자로 바꾼다. 못 바꾸면 빈 값(NaN)."""
    s = str(x).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_corp_codes(session: requests.Session) -> pd.DataFrame:
    """종목코드(6자리)를 DART가 쓰는 회사 번호(corp_code)로 바꿔 주는 표를 가져온다.
    DART는 우리가 쓰는 종목코드를 모르고 자기만의 회사 번호를 쓰기 때문에 이 표가 꼭 필요하다."""
    cache = paths.DART_CACHE / "corpCode.csv"
    if cache.exists():                      # 한 번 받아 두면 다시 안 받는다
        return pd.read_csv(cache, dtype=str)
    r = session.get(API + "corpCode.xml", params={"crtfc_key": _key()}, timeout=60)
    r.raise_for_status()                    # 실패했으면 여기서 바로 오류를 낸다
    # 잘 받으면 zip 파일이고, 키에 문제가 있으면 zip이 아니라 오류 설명 글이 온다.
    # zip 파일은 맨 앞 두 글자가 'PK'다. 아니면 DART가 보낸 설명을 그대로 보여 주고 멈춘다.
    if not r.content.startswith(b"PK"):
        raise SystemExit(f"DART가 회사 목록 대신 오류를 보냈습니다 (키는 출력하지 않음): {r.text[:300]}")
    # 답이 zip 파일로 오고 그 안에 XML이 있어서, 압축을 풀고 읽는다
    xml =zipfile.ZipFile(io.BytesIO(r.content)).read("CORPCODE.xml")
    rows = [{"corp_code": e.findtext("corp_code"), "stock_code": (e.findtext("stock_code") or "").strip(),
             "corp_name": e.findtext("corp_name")} for e in ET.fromstring(xml)]
    df = pd.DataFrame(rows)
    df = df[df["stock_code"] != ""]         # 종목코드가 없는 회사(비상장)는 필요 없어서 뺀다
    paths.DART_CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def _get_json(session, endpoint: str, params: dict, cache_name: str) -> dict:
    """DART에 한 번 물어보고 답(JSON)을 돌려준다. 저장해 둔 답이 있으면 그걸 쓴다."""
    f = paths.DART_CACHE / cache_name
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    j: dict = {}
    for attempt in range(6):                # 잠깐 실패할 수 있어서 최대 6번 시도
        try:
            r = session.get(API + endpoint, params={"crtfc_key": _key(), **params}, timeout=30)
            r.raise_for_status()
            j = r.json()
        except (requests.RequestException, ValueError) as e:
            # 인터넷이 잠깐 끊기거나 서버가 답 없이 연결을 닫은 경우. 멈추지 않고 쉬었다가 다시 한다.
            # (오류 글에는 주소가 들어 있어 키가 보일 수 있어서, 글은 찍지 않고 종류만 찍는다)
            wait = min(60, 5 * 2 ** attempt)
            print(f"  네트워크 오류({type(e).__name__}) → {wait}초 뒤 다시 시도 ({attempt + 1}/6)", flush=True)
            time.sleep(wait)
            continue
        if j.get("status") == "020":        # 020 = 오늘 쓸 수 있는 횟수를 다 썼다는 DART의 신호
            raise SystemExit("OpenDART 일일 호출 한도 초과(020). 내일 같은 명령을 다시 실행하세요.")
        if j.get("status") in ("000", "013"):   # 000 = 정상, 013 = 조회된 자료가 없음(이것도 정상 답)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
            time.sleep(REQUEST_GAP)         # DART에 실제로 물어본 뒤에는 매번 쉰다 (너무 빠르면 접속을 막는다)
            return j
        time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenDART 오류 {endpoint} {params}: {j}")


def parse_dps(j: dict) -> float:
    """배당 답(alotMatter)에서 '보통주 주당 현금배당금(원)'만 꺼낸다. 없으면 빈 값(NaN).
    (DPS = Dividend Per Share, 주식 1주당 받는 배당금)"""
    for it in j.get("list", []) or []:
        # 주식 종류가 하나뿐인 회사는 DART가 종류를 '보통주'가 아니라 '-' 로 적는다.
        # 그래서 '보통주'로 시작하거나 '-'(또는 빈칸)인 줄을 보통주 몫으로 본다. 우선주 줄은 건너뛴다.
        kind = (it.get("stock_knd") or "").strip()
        if "주당 현금배당금" in (it.get("se") or "") and (kind in ("", "-") or kind.startswith("보통주")):
            v = _num(it.get("thstrm"))
            if not np.isnan(v):          # 값이 '-' 라서 숫자가 아닌 줄이면 다음 줄을 계속 본다
                return v
    return np.nan


def fetch_dividends(codes, years, sleep=0.05) -> pd.DataFrame:
    """종목 × 연도마다 주당 배당금을 받아서 (code, fiscal_year, dps, acc_mt) 표로 만든다.
    acc_mt는 회사의 결산월(몇 월에 1년 장부를 마감하는지)이다."""
    out = []
    with requests.Session() as s:
        corp = load_corp_codes(s).set_index("stock_code")["corp_code"]
        for n, code in enumerate(codes, 1):
            if code not in corp.index:      # DART에 없는 종목(예: ETF)은 건너뛴다
                continue
            cc = corp[code]
            acc_mt = _get_json(s, "company.json", {"corp_code": cc}, f"company/{cc}.json").get("acc_mt")
            for y in years:
                j = _get_json(s, "alotMatter.json", {"corp_code": cc, "bsns_year": y, "reprt_code": ANNUAL},
                              f"alot/{cc}_{y}.json")
                out.append({"code": code, "fiscal_year": y, "dps": parse_dps(j), "acc_mt": acc_mt})
            time.sleep(sleep)               # 너무 빠르게 부탁하지 않으려고 잠깐 쉰다
            if n % 100 == 0:
                print(f"  {n}/{len(codes)}종목", flush=True)
    return pd.DataFrame(out)


def estimate_ex_date(trading_days: pd.DatetimeIndex, fiscal_year: int, month: int = 12):
    """배당락일을 추정한다: 결산월 말일 이전 마지막 거래일의 '바로 전' 거래일.
    우리가 가진 데이터가 그 말일을 덮지 못하면 추정할 수 없어서 None을 돌려준다."""
    end = pd.Timestamp(year=fiscal_year, month=month, day=1) + pd.offsets.MonthEnd(0)   # 그 달의 마지막 날
    if len(trading_days) == 0 or trading_days[0] > end or trading_days[-1] < end - pd.Timedelta(days=4):
        return None
    days = trading_days[trading_days <= end]
    return days[-2] if len(days) >= 2 else None    # [-2] = 뒤에서 두 번째 = 마지막 거래일의 전 거래일


def build_events(div: pd.DataFrame, raw: pd.DataFrame, override: pd.DataFrame | None = None):
    """배당금 표를 '날짜 + 보정계수'가 있는 이벤트 표로 바꾼다. 계산이 안 되는 건 확인 목록으로 뺀다."""
    cal = pd.DatetimeIndex(sorted(raw["date"].unique()))   # 거래가 있었던 날들의 달력
    ov = {}
    if override is not None and len(override):
        ov = {(r.code, int(r.fiscal_year)): pd.Timestamp(r.ex_date) for r in override.itertuples()}
    close = raw.set_index(["code", "date"])["close"]       # (종목, 날짜)로 종가를 빨리 찾기 위한 준비
    ev, rv = [], []
    for r in div.dropna(subset=["dps"]).query("dps > 0").itertuples():   # 배당금이 있는 줄만
        key = (r.code, int(r.fiscal_year))
        base = {"code": r.code, "fiscal_year": r.fiscal_year, "dps": r.dps}
        if key in ov:                       # 사람이 날짜를 확정해 둔 경우가 제일 우선
            ex, conf = ov[key], "confirmed"
        elif str(r.acc_mt) != "12":         # 12월 결산이 아니면 연말 가정이 안 맞으니 사람이 보게 뺀다
            rv.append({**base, "reason": f"non_december_fy(acc_mt={r.acc_mt})"})
            continue
        else:
            ex = estimate_ex_date(cal, int(r.fiscal_year))
            conf = "low" if r.fiscal_year >= FIRST_LOW_CONFIDENCE_FY else "estimated"
        if ex is None:
            rv.append({**base, "reason": "ex_date_outside_data_range"})
            continue
        prev = cal[cal < ex]                # 배당락일 전의 거래일들
        pc = close.get((r.code, prev[-1]), np.nan) if len(prev) else np.nan   # 그 중 마지막 날의 종가
        if not np.isfinite(pc) or pc <= 0:
            rv.append({**base, "reason": "no_prev_close"})
            continue
        # 계수 = 1 - 배당금/전일종가. 예) 종가 10,000원에 배당 500원이면 0.95.
        # 분할 전의 '실제' 가격으로 계산하기 때문에, 분할과 곱하는 순서를 바꿔도 결과가 같다.
        f = 1 - r.dps / pc
        if not (0.5 < f < 1):               # 배당이 주가의 절반을 넘는 건 말이 안 되니 오류로 보고 뺀다
            rv.append({**base, "reason": f"implausible_factor({f:.3f})"})
            continue
        ev.append({**base, "ex_date": ex, "prev_close": pc, "factor": f, "confidence": conf})
    return pd.DataFrame(ev), pd.DataFrame(rv)


def main(argv=None):
    """터미널에서 `python -m indexbeater.krx_source dividends` 로 실행할 때 쓰는 입구."""
    ap = argparse.ArgumentParser(prog="krx_source dividends")
    ap.add_argument("--raw", default=str(paths.RAW_DAILY))
    ap.add_argument("--years", nargs=2, type=int, default=[2011, 2025], metavar=("FROM", "TO"))
    ap.add_argument("--limit-codes", type=int)
    a = ap.parse_args(argv)
    raw = pd.read_parquet(a.raw)
    # ETF는 배당 공시 대상이 아니라서 코스피·코스닥 종목만 모은다
    codes = sorted(raw.loc[raw["market"].isin(["KOSPI", "KOSDAQ"]), "code"].unique())[: a.limit_codes]
    div = fetch_dividends(codes, range(a.years[0], a.years[1] + 1))
    div.to_csv(paths.DART_CACHE / "dps_table.csv", index=False)
    ov = pd.read_csv(paths.DIV_OVERRIDE, dtype={"code": str}) if paths.DIV_OVERRIDE.exists() else None
    ev, rv = build_events(div, raw, ov)
    ev.to_csv(paths.DIV_EVENTS, index=False, encoding="utf-8-sig")
    rv.to_csv(paths.DIV_REVIEW, index=False, encoding="utf-8-sig")
    print(f"배당 이벤트 {len(ev)}건 → {paths.DIV_EVENTS}\n확인 필요 {len(rv)}건 → {paths.DIV_REVIEW}")


if __name__ == "__main__":
    main()
