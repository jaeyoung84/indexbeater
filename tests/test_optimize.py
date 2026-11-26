"""파라미터 고르기(5.5)·나눠 돌리기·합치기 점검(11/24 결정 D6·D7)이 규칙대로 되는지 확인한다."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from indexbeater import config as C
from indexbeater import optimize as O
from indexbeater import scorecard as SC
from indexbeater import signals as S


@pytest.fixture(autouse=True)
def _reference_before_first_notebook04(tmp_path_factory, monkeypatch):
    """엔진 v2 기준 파일은 노트북 04 10번이 만든다. 그런데 노트북 04 는 9번에서 pytest 를 먼저 돌린다.
    기준 파일이 아직 없는 그 첫 실행에서만: v1 기준 파일에 지금 engine.py 지문을 넣은 임시 파일을 쓴다.
    (그래야 첫 실행과 다음 실행의 pytest 통과 수가 같다. 기준 파일이 생긴 뒤에는 진짜 파일로 검사한다)"""
    if O.REFERENCE.exists():
        return
    import pandas as pd
    v1 = pd.read_csv(C.RESULTS / "engine_v1_reference.csv", dtype=str, encoding="utf-8-sig")
    eng = SC.fingerprints(with_data=False)["indexbeater/engine.py"]
    v1.loc[v1["항목"] == "지문: indexbeater/engine.py", "기준값"] = eng
    p = tmp_path_factory.mktemp("ref") / O.REFERENCE.name
    v1.to_csv(p, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(O, "REFERENCE", p)


# ---------------------------------------------------------------- 그리드 모양·기본값
def test_default_params_are_in_grid_for_all_18():
    assert set(SC.DEFAULT_PARAMS) == set(S.SIGNALS)
    for name in S.SIGNALS:
        sets = SC.param_sets(name)
        assert SC.DEFAULT_PARAMS[name] == sets[SC.default_index(name, sets)]


def test_neighbors_one_step_on_one_axis():
    sets = SC.param_sets("rsi")                         # period (7,14) × low (20,25,30,35)
    pos = [SC.grid_pos("rsi", p) for p in sets]
    nb = SC.neighbors(pos)
    k = sets.index({"period": 7, "low": 30})
    got = sorted((sets[q]["period"], sets[q]["low"]) for q in nb[k])
    assert got == [(7, 25), (7, 35), (14, 30)]          # 대각선(14,25) 은 이웃 아님


def test_neighbors_skip_missing_ma_cross_pairs():
    sets = SC.param_sets("ma_cross")                    # 단기 < 장기만 → (20,20) 없음
    assert len(sets) == 8
    pos = [SC.grid_pos("ma_cross", p) for p in sets]
    nb = SC.neighbors(pos)
    k = sets.index({"short": 20, "long": 60})
    got = sorted((sets[q]["short"], sets[q]["long"]) for q in nb[k])
    assert got == [(10, 60), (20, 120)]                 # (20,20) 은 없으니 건너뜀


# ---------------------------------------------------------------- 고르기 규칙
def _line_grid(n):
    pos = [(i,) for i in range(n)]
    return pos, SC.neighbors(pos)


def test_select_uses_median_of_self_and_neighbors():
    pos, nb = _line_grid(5)
    edges = np.array([0.00, 0.01, 0.10, 0.01, 0.02])    # 가운데 0.10 은 혼자 튐
    counts = np.full(5, 500)
    sel = SC.select_set(edges, counts, nb, pos, default=0)
    # 점수: 0→med(0,.01)=.005, 1→med(.01,0,.10)=.01, 2→med(.10,.01,.01)=.01, 3→.02, 4→med(.02,.01)=.015
    assert sel["set"] == 3 and sel["score"] == pytest.approx(0.02)


def test_select_excludes_small_sets_from_pick_and_neighbors():
    pos, nb = _line_grid(3)
    edges = np.array([0.01, 0.50, 0.02])
    counts = np.array([500, C.MIN_TRADES - 1, 500])     # 가운데는 299건 → 고르기·이웃 모두 제외
    sel = SC.select_set(edges, counts, nb, pos, default=0)
    assert sel["set"] == 2 and sel["eligible"] == 2


def test_select_tie_goes_to_nearest_default_then_grid_order():
    pos, nb = _line_grid(5)
    edges = np.full(5, 0.01)
    counts = np.full(5, 500)
    assert SC.select_set(edges, counts, nb, pos, default=3)["set"] == 3
    pos2 = [(0,), (2,), (1,)]                            # 기본값(칸 1)에서 칸 0·2 는 같은 거리 → 그리드 순서 앞
    sel = SC.select_set(np.array([0.02, 0.02, 0.0]), np.full(3, 500), [[], [], []], pos2, default=2)
    assert sel["set"] == 0


def test_select_all_short_falls_back_to_default():
    pos, nb = _line_grid(4)
    sel = SC.select_set(np.full(4, 0.05), np.full(4, 10), nb, pos, default=2)
    assert sel["set"] == 2 and sel["short"] and sel["used_default"] and math.isnan(sel["score"])


# ---------------------------------------------------------------- 지문·나누기·합치기 점검
def test_code_hash_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\r\ny = 2\r\n")
    b.write_bytes(b"x = 1\ny = 2\n")
    assert SC.code_hash(a) == SC.code_hash(b) and len(SC.code_hash(a)) == 12


def test_parts_cover_18_once():
    O.check_parts_table()
    names = [s for _, sigs in O.PARTS.values() for s in sigs]
    assert len(names) == 18 and set(names) == set(S.SIGNALS)
    assert len(O.PARTS) == 4


def test_resolve_signals_numbers_and_names():
    assert O.resolve_signals("7, rsi, S01") == ["rsi", "ma_cross"]
    with pytest.raises(SystemExit):
        O.resolve_signals("19")
    with pytest.raises(SystemExit):
        O.resolve_signals("없는신호")


def test_reference_fingerprints_match_frozen_files():
    ref = O.reference_fingerprints()
    now = SC.fingerprints(with_data=False)
    for f in O.FROZEN:
        assert now[f"indexbeater/{f}"] == ref[f"indexbeater/{f}"], f"{f} 가 엔진 {O.ENGINE_VERSION} 고정값과 다릅니다"


def _fake_parts(root, fp, skip=(), no_port=()):
    for name in S.SIGNALS:
        if name in skip:
            continue
        d = root / name
        d.mkdir(parents=True)
        for f in O.PART_FILES:
            (d / f).write_text("", encoding="utf-8")
        meta = {"신호": name, "포트폴리오": name not in no_port, "지문": dict(fp)}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _ref_fp():
    return dict(O.reference_fingerprints())


def test_check_parts_ok(tmp_path):
    fp = _ref_fp()
    _fake_parts(tmp_path, fp)
    problems, metas = O.check_parts(tmp_path, now=fp)
    assert problems == [] and len(metas) == 18


def test_check_parts_refuses_missing_signal(tmp_path):
    _fake_parts(tmp_path, _ref_fp(), skip=("rsi",))
    problems, _ = O.check_parts(tmp_path)
    assert any(p.startswith("rsi:") for p in problems)


def test_check_parts_refuses_different_fingerprint(tmp_path):
    fp = _ref_fp()
    _fake_parts(tmp_path, fp)
    meta_path = tmp_path / "cci" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["지문"]["indexbeater/engine.py"] = "000000000000"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    problems, _ = O.check_parts(tmp_path)
    assert any("cci" in p and "engine.py" in p for p in problems)


def test_check_parts_refuses_no_portfolio_and_local_mismatch(tmp_path):
    fp = _ref_fp()
    _fake_parts(tmp_path, fp, no_port=("obv",))
    now = dict(fp, **{"indexbeater/scorecard.py": "ffffffffffff"})
    problems, _ = O.check_parts(tmp_path, now=now)
    assert any(p.startswith("obv:") for p in problems)
    assert any("scorecard.py" in p for p in problems)


def test_check_frozen_detects_change():
    ref = _ref_fp()
    assert O.check_frozen(ref, ref) == []
    bad = dict(ref, **{"indexbeater/signals.py": "x"})
    assert len(O.check_frozen(bad, ref)) == 1


# ---------------------------------------------------------------- 같은지 비교·멈춤 장치 (11/24 재검토)
def test_compare_tables_same_tiny_and_different():
    a = pd.DataFrame({"x": [0.1, np.nan, 0.3], "s": ["a", "b", "c"]})
    assert O.compare_tables(a, a.copy()) == ("같음", 0.0)
    b = a.copy()
    b.loc[0, "x"] += 1e-15
    assert O.compare_tables(a, b)[0] == "아주 작은 차이"
    c = a.copy()
    c.loc[2, "x"] = 0.31
    assert O.compare_tables(a, c)[0] == "다름: 숫자 차이 큼"
    d = a.copy()
    d.loc[1, "x"] = 0.0
    assert O.compare_tables(a, d)[0].startswith("다름: x 빈칸")
    e = a.copy()
    e.loc[1, "s"] = "z"
    assert O.compare_tables(a, e)[0].startswith("다름: s")
    assert O.compare_tables(a, a.iloc[:2])[0].startswith("다름: 줄 수")


def test_compare_folders(tmp_path, capsys):
    for root in ("A", "B"):
        d = tmp_path / root / "rsi"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"지문": {"x": "1"}, "환경": {"numpy": "1"}}), encoding="utf-8")
        for t in O.TABLES:
            df = pd.DataFrame({"v": [1.0, 2.0 if root == "A" else 2.5]})
            if t.endswith(".parquet"):
                df.to_parquet(d / t)
            else:
                df.to_csv(d / t, index=False, encoding="utf-8-sig")
    assert O.compare(tmp_path / "A", tmp_path / "A") == 0
    assert O.compare(tmp_path / "A", tmp_path / "B") == 1
    assert "rsi" in capsys.readouterr().out


def test_environment_notes_only_when_versions_differ():
    same = {"a": {"환경": {"numpy": "2"}}, "b": {"환경": {"numpy": "2"}}}
    assert O.environment_notes(same) == []
    diff = {"a": {"환경": {"numpy": "2"}}, "b": {"환경": {"numpy": "1"}}}
    assert len(O.environment_notes(diff)) == 2


def test_check_finite_stops_on_zero_price_buy():
    SC.check_finite(np.array([0.1, np.nan, -0.2]), "정상")      # 빈칸은 괜찮음 (체결 안 된 칸)
    with pytest.raises(SC.BadTradeError):
        SC.check_finite(np.array([0.1, np.inf]), "0원 매수")
