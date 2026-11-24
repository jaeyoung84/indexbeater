"""STEP 3 신호 18개 단독 점수표 — 4명이 나눠 동시에 돌리고, 지문을 확인해 합친다 (기준 5.3·5.5·5.17).

계산은 scorecard.py 에 있다. 이 파일은 '누가 무엇을 돌리나'와 '돌린 결과 합치기'만 맡는다.
엔진(engine.py·signals.py·exits.py·indicators.py·data_io.py·config.py)은 고치지 않는다.
지금 고정한 엔진은 v2 (11/24: v1 의 '시가 0 매수' 결함을 고침 — 회의기록.md). v1 기준 파일은 지우지 않고 남긴다.

실행 (11/24 팀장 결정 D6 — 회의기록.md)
  python -m indexbeater optimize --part 1        # 1~4: 아래 PARTS 표의 내 몫만 돌린다
  python -m indexbeater optimize --signals rsi   # 신호 이름(또는 번호 1~18)을 직접 줄 때
  python -m indexbeater optimize --merge         # 18개가 다 모이면 한 명이 합친다
  python -m indexbeater optimize --list          # 나눈 표만 보여 준다
  python -m indexbeater optimize --all --out 폴더 # 18개 전부를 한 컴퓨터에서 (재현 확인용)
  python -m indexbeater optimize --compare 폴더A 폴더B   # 두 결과가 똑같은지 표마다 비교

  같은 엔진·같은 데이터·같은 프로그램 판이면 결과는 소수점까지 똑같아야 한다 (엔진에 무작위가 없고,
  순서가 같은 값이면 종목코드 순으로 정한다). 다르면 → 지문·판부터 보고, 둘 다 같은데 다르면 코드 결함.

  --out 폴더       결과 묶음을 다른 폴더에 쓴다 (기본 results/scorecard_parts)
  --no-portfolio   포트폴리오를 건너뛴다 (점검용. 합치기에서는 거부된다)

결과 묶음: results/scorecard_parts/<신호>/ grid.csv · train.csv · selection.csv · scorecard.csv ·
           portfolio_daily.csv · oos_trades.parquet · meta.json(지문)
합친 결과: results/single_signal.csv(그리드 성적표) · results/param_selection.csv · results/scorecard.csv ·
           results/scorecard_portfolio_daily.csv · results/scorecard_oos_trades.parquet · results/scorecard_meta.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from indexbeater import config as C
from indexbeater import signals as S

FROZEN = ("engine.py", "signals.py", "exits.py", "indicators.py", "data_io.py", "config.py")
ENGINE_VERSION = "v2"
REFERENCE = C.RESULTS / f"engine_{ENGINE_VERSION}_reference.csv"   # 노트북 04 10번이 처음 한 번 만든다
SIGNAL_ORDER = list(S.SIGNALS)                 # 1~18 번호 = 계획서 부록 A 순서

# 4명 나누기 (일의 양 = 세트 수·신호 수로 맞춤. 결과와 상관없이 정함 — 회의기록 11/24 D6)
PARTS = {
    1: ("송재영", ["vol_breakout", "candle", "obv", "squeeze", "sar"]),
    2: ("김동겸", ["bollinger", "macd", "alignment", "donchian", "ichimoku"]),
    3: ("김승찬", ["rsi", "volume_breakout", "adx_di", "stochastic"]),
    4: ("김정철", ["disparity", "ma_cross", "cci", "williams_r"]),
}

PART_FILES = ("grid.csv", "train.csv", "selection.csv", "scorecard.csv", "portfolio_daily.csv",
              "oos_trades.parquet", "meta.json")
MERGED = {"grid": "single_signal.csv", "selection": "param_selection.csv", "scorecard": "scorecard.csv",
          "portfolio_daily": "scorecard_portfolio_daily.csv"}


# ---------------------------------------------------------------- 지문 확인
def reference_fingerprints(path: Path | None = None) -> dict:
    path = REFERENCE if path is None else path
    if not Path(path).exists():
        raise SystemExit(f"기준 파일이 없습니다: {path}\n→ 노트북 04 를 처음부터 끝까지 한 번 실행하면 10번 칸이 만듭니다.")
    ref = pd.read_csv(path, encoding="utf-8-sig")
    rows = ref[ref["항목"].str.startswith("지문: ")]
    return {r["항목"][len("지문: "):]: str(r["기준값"]) for _, r in rows.iterrows()}


def check_frozen(now: dict, ref: dict) -> list[str]:
    """엔진 파일 6개·데이터 지문이 고정값(REFERENCE)과 같은지. 다른 것 목록을 돌려준다."""
    bad = []
    for f in [f"indexbeater/{x}" for x in FROZEN] + ["data/krx_adjusted.parquet"]:
        if now.get(f) != ref.get(f):
            bad.append(f"{f}: 지금 {now.get(f)} / 고정 {ref.get(f)}")
    return bad


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True,
                             text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "indexbeater"], cwd=C.ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + (" (고친 파일 있음)" if dirty else "")
    except Exception:
        return "알 수 없음"


def environment() -> dict:
    """계산 결과에 영향을 줄 수 있는 프로그램 판(버전). 판이 다르면 소수점 끝자리가 달라질 수 있다."""
    import numpy
    import pyarrow
    import scipy
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "pyarrow": pyarrow.__version__}


def resolve_signals(text: str) -> list[str]:
    out = []
    for tok in [t.strip() for t in text.split(",") if t.strip()]:
        name = tok
        digits = tok.upper().lstrip("S")
        if digits.isdigit():
            k = int(digits)
            if not 1 <= k <= len(SIGNAL_ORDER):
                raise SystemExit(f"신호 번호 {tok} 는 1~{len(SIGNAL_ORDER)} 이어야 합니다")
            name = SIGNAL_ORDER[k - 1]
        if name not in S.SIGNALS:
            raise SystemExit(f"모르는 신호: {tok}  (가능: {', '.join(SIGNAL_ORDER)})")
        out.append(name)
    return list(dict.fromkeys(out))


def check_parts_table() -> None:
    names = [s for _, sigs in PARTS.values() for s in sigs]
    if sorted(names) != sorted(SIGNAL_ORDER) or len(names) != len(set(names)):
        raise RuntimeError("PARTS 표가 신호 18개를 한 번씩 덮지 않습니다")


# ---------------------------------------------------------------- 돌리기
def run(signals: list[str], out_dir: Path, portfolio: bool = True, who: str = "") -> int:
    from indexbeater import data_io
    from indexbeater import scorecard as SC

    t0 = time.time()
    print(f"1) 지문 확인 (엔진 {ENGINE_VERSION} 고정값과 같아야 돌린다)")
    now = SC.fingerprints(with_data=True)
    bad = check_frozen(now, reference_fingerprints())
    if bad:
        print(f"  ✗ 엔진 {ENGINE_VERSION} 과 다릅니다. 실행하지 않습니다:\n   - " + "\n   - ".join(bad))
        return 1
    print("  ✓ 엔진 6개·데이터 지문 같음")
    print(f"2) 데이터 불러오기 (설계 구간 ~{C.DESIGN_END.date()} 만)")
    data = data_io.load_wide()
    print(f"  {len(data['adj_close'])}일 × {data['adj_close'].shape[1]}종목 ({time.time() - t0:.0f}초)")
    print("3) 대조군 바탕 만들기 (H 5·10·20)")
    try:
        ctx = SC.Context(data)
    except SC.BadTradeError as e:
        print(f"  ✗ {e}")
        return 1
    for k, name in enumerate(signals, 1):
        print(f"4-{k}) {name} ({k}/{len(signals)})")
        t1 = time.time()
        try:
            tables = SC.run_signal(name, ctx, portfolio=portfolio)
        except SC.BadTradeError as e:
            print(f"  ✗ {e}")
            return 1
        meta = {"신호": name, "번호": SIGNAL_ORDER.index(name) + 1, "돌린사람": who,
                "포트폴리오": portfolio, "지문": now, "환경": environment(), "git": git_commit(),
                "돌린시각": time.strftime("%Y-%m-%d %H:%M:%S"), "걸린초": round(time.time() - t1, 1),
                "행수": {key: len(df) for key, df in tables.items()}}
        d = SC.write_part(name, tables, meta, out_dir)
        print(f"  저장: {d}")
    print(f"끝 ({time.time() - t0:.0f}초). 결과 묶음 폴더 {out_dir} 를 공유 드라이브에 올려 주세요.")
    return 0


# ---------------------------------------------------------------- 합치기
def check_parts(part_dir: Path, now: dict | None = None) -> tuple[list[str], dict]:
    """합치기 전 점검. (문제 목록, 신호별 meta) 를 돌려준다. now=None 이면 지금 파일과는 비교하지 않는다."""
    problems, metas = [], {}
    for name in SIGNAL_ORDER:
        d = Path(part_dir) / name
        missing = [f for f in PART_FILES if not (d / f).exists()]
        if missing:
            problems.append(f"{name}: 없음 → {', '.join(missing)}")
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        metas[name] = meta
        if meta.get("신호") != name:
            problems.append(f"{name}: meta 의 신호 이름이 {meta.get('신호')}")
        if not meta.get("포트폴리오"):
            problems.append(f"{name}: 포트폴리오 없이 돌린 묶음 (--no-portfolio)")
    if metas:
        first_name, first = next(iter(metas.items()))
        for name, meta in metas.items():
            diff = sorted(k for k in set(first["지문"]) | set(meta["지문"])
                          if first["지문"].get(k) != meta["지문"].get(k))
            if diff:
                problems.append(f"{name}: {first_name} 와 지문 다름 → {', '.join(diff)}")
        if now is not None:
            diff = sorted(k for k in set(first["지문"]) | set(now) if first["지문"].get(k) != now.get(k))
            if diff:
                problems.append(f"지금 내 컴퓨터 파일과 지문 다름 → {', '.join(diff)}")
        bad = check_frozen(first["지문"], reference_fingerprints())
        problems += [f"엔진 {ENGINE_VERSION} 고정값과 다름 → {b}" for b in bad]
    return problems, metas


def environment_notes(metas: dict) -> list[str]:
    """묶음마다 프로그램 판이 다르면 알려 준다 (거부하지는 않음 — 차이가 생기면 --compare 로 확인)."""
    seen = {}
    for name, m in metas.items():
        env = json.dumps(m.get("환경", {}), sort_keys=True, ensure_ascii=False)
        seen.setdefault(env, []).append(name)
    if len(seen) <= 1:
        return []
    return [f"프로그램 판 {env} → {', '.join(ns)}" for env, ns in seen.items()]


# ---------------------------------------------------------------- 같은지 비교 (재현 확인)
TABLES = ("grid.csv", "train.csv", "selection.csv", "scorecard.csv", "portfolio_daily.csv", "oos_trades.parquet")


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})


def compare_tables(a: pd.DataFrame, b: pd.DataFrame) -> tuple[str, float]:
    """두 표가 같은지. ('같음' | '아주 작은 차이' | '다름: 이유', 가장 큰 숫자 차이)."""
    if list(a.columns) != list(b.columns):
        return "다름: 열이 다름", float("nan")
    if len(a) != len(b):
        return f"다름: 줄 수 {len(a)} / {len(b)}", float("nan")
    worst = 0.0
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            if not (np.isnan(xv) == np.isnan(yv)).all():
                return f"다름: {col} 빈칸 위치", float("nan")
            ok = ~np.isnan(xv)
            if (np.isinf(xv[ok]) != np.isinf(yv[ok])).any():
                return f"다름: {col} 무한대 위치", float("nan")
            fin = ok & np.isfinite(xv)
            if (xv[ok & ~fin] != yv[ok & ~fin]).any():
                return f"다름: {col} 무한대 부호", float("nan")
            if fin.any():
                worst = max(worst, float(np.max(np.abs(xv[fin] - yv[fin]))))
        elif not x.astype(str).equals(y.astype(str)):
            return f"다름: {col} 값", float("nan")
    if worst == 0:
        return "같음", 0.0
    return ("아주 작은 차이" if worst <= COMPARE_TOL else "다름: 숫자 차이 큼"), worst


COMPARE_TOL = 1e-9          # 프로그램 판·컴퓨터가 달라 생기는 소수점 끝자리 차이는 이보다 훨씬 작다 (약 1e-15)


def compare(dir_a: Path, dir_b: Path) -> int:
    """두 결과 묶음 폴더(신호별 하위 폴더)를 표마다 비교한다. 둘 다 있는 신호만 비교."""
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    names = [n for n in SIGNAL_ORDER if (dir_a / n).is_dir() and (dir_b / n).is_dir()]
    if not names:
        print("  두 폴더에 같이 있는 신호가 없습니다")
        return 1
    bad = 0
    for n in names:
        ma = json.loads((dir_a / n / "meta.json").read_text(encoding="utf-8"))
        mb = json.loads((dir_b / n / "meta.json").read_text(encoding="utf-8"))
        fp = "지문 같음" if ma.get("지문") == mb.get("지문") else "지문 다름"
        env = "판 같음" if ma.get("환경") == mb.get("환경") else "판 다름"
        res = []
        for t in TABLES:
            pa, pb = dir_a / n / t, dir_b / n / t
            if not (pa.exists() and pb.exists()):
                res.append((t, "다름: 파일 없음", float("nan")))
                continue
            st, d = compare_tables(_read(pa), _read(pb))
            res.append((t, st, d))
        ok = not any(st.startswith("다름") for _, st, _ in res)
        bad += not ok
        print(f"  {'✓' if ok else '✗'} {n} ({fp}, {env})")
        for t, st, d in res:
            if st != "같음":
                print(f"      {t}: {st}" + (f" (가장 큰 차이 {d:.3g})" if d == d and d > 0 else ""))
    print(f"비교 {len(names)}개 신호: " + ("모두 같음 ✓" if not bad else f"{bad}개 다름 ✗"))
    return 1 if bad else 0


def merge(part_dir: Path, out_dir: Path = C.RESULTS) -> int:
    from indexbeater import scorecard as SC

    print(f"1) 결과 묶음 점검: {part_dir}")
    problems, metas = check_parts(part_dir, now=SC.fingerprints(with_data=True))
    if problems:
        print("  ✗ 합치지 않습니다:\n   - " + "\n   - ".join(problems))
        return 1
    print(f"  ✓ 신호 {len(metas)}개, 지문 모두 같음")
    for note in environment_notes(metas):
        print(f"  주의: {note}")
    print("2) 합치기")
    out_dir = Path(out_dir)
    for key, fname in MERGED.items():
        dfs = [pd.read_csv(Path(part_dir) / n / f"{key}.csv", encoding="utf-8-sig", dtype={"code": str})
               for n in SIGNAL_ORDER]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(out_dir / fname, index=False, encoding="utf-8-sig")
        print(f"  {out_dir / fname}: {len(df):,}줄")
    oos = pd.concat([pd.read_parquet(Path(part_dir) / n / "oos_trades.parquet") for n in SIGNAL_ORDER],
                    ignore_index=True)
    oos.to_parquet(out_dir / "scorecard_oos_trades.parquet", index=False)
    print(f"  {out_dir / 'scorecard_oos_trades.parquet'}: {len(oos):,}줄")
    first = next(iter(metas.values()))
    summary = {"합친시각": time.strftime("%Y-%m-%d %H:%M:%S"), "지문": first["지문"],
               "신호별": {n: {k: m[k] for k in ("돌린사람", "git", "돌린시각", "걸린초")} for n, m in metas.items()}}
    (out_dir / "scorecard_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print("끝. 벤치마크 초과수익 열은 벤치마크 수집 뒤 채운다 (결정 D4).")
    return 0


# ---------------------------------------------------------------- 명령줄
def main(argv=None):
    check_parts_table()
    ap = argparse.ArgumentParser(prog="python -m indexbeater optimize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--part", type=int, choices=sorted(PARTS), help="내 몫 번호 (1~4)")
    g.add_argument("--signals", help="신호 이름 또는 번호, 쉼표로 (예: rsi,bollinger 또는 7,12)")
    g.add_argument("--all", action="store_true", help="18개 전부를 한 컴퓨터에서 돌린다 (재현 확인용)")
    g.add_argument("--merge", action="store_true", help="18개 결과 묶음 합치기")
    g.add_argument("--compare", nargs=2, type=Path, metavar=("폴더A", "폴더B"),
                   help="두 결과 묶음 폴더가 같은지 비교 (예: 나눠 돌린 것 vs 한 번에 돌린 것)")
    g.add_argument("--list", action="store_true", help="나눈 표 보기")
    ap.add_argument("--out", type=Path, default=None, help="결과 묶음 폴더 (기본 results/scorecard_parts)")
    ap.add_argument("--no-portfolio", action="store_true", help="포트폴리오 건너뛰기 (점검용)")
    a = ap.parse_args(argv)

    if a.list:
        for k, (who, sigs) in PARTS.items():
            nums = [SIGNAL_ORDER.index(s) + 1 for s in sigs]
            sets = sum(len(list(S.iter_params(s))) for s in sigs)
            print(f"  {k}. {who}: " + ", ".join(f"{n}.{s}" for n, s in zip(nums, sigs)) + f"  (세트 {sets}개)")
        return 0
    if a.compare:
        return compare(*a.compare)
    from indexbeater.scorecard import PART_DIR
    part_dir = a.out or PART_DIR
    if a.merge:
        return merge(part_dir)
    if a.part:
        who, sigs = PARTS[a.part]
        print(f"{a.part}번 몫 ({who}): {', '.join(sigs)}")
    elif a.all:
        who, sigs = "", list(SIGNAL_ORDER)
    else:
        who, sigs = "", resolve_signals(a.signals)
    return run(sigs, part_dir, portfolio=not a.no_portfolio, who=who)


if __name__ == "__main__":
    sys.exit(main())
