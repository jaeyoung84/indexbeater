"""미래 정보 유출 테스트 (계획서 부록 D 마지막 항목).

원리: k일 가격을 임의로 바꿔도 k일 이전 날짜의 신호는 하나도 변하면 안 된다.
마지막 날뿐 아니라 중간 날짜 여러 곳을 바꿔 본다 (ewm·cumsum 같은 누적 계산까지 잡기 위해).

진짜 데이터 대신 '가짜 주가'(무작위로 만든 주가)를 쓴다. 이유:
- 시험 구간(2020년 이후)을 열 일이 없다
- 늦게 상장한 종목·중간에 상장폐지된 종목·거래정지일을 일부러 넣어 볼 수 있다
- 몇 초 안에 끝나서 신호를 고칠 때마다 돌릴 수 있다
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indexbeater import config as C
from indexbeater import data_io
from indexbeater.signals import SIGNALS, iter_params, raw_signal

# k일 이후를 바꿔 볼 재료 (신호가 쓸 수 있는 모든 가격·거래량 칸)
PERTURB_KEYS = ["adj_open", "adj_high", "adj_low", "adj_close", "value", "volume_s",
                "open", "close", "volume"]


def make_synthetic(n_days: int = 800, n_codes: int = 10, seed: int = 0,
                   start=C.DATA_START) -> dict:
    """가짜 주가 wide 묶음. 실제 파일과 같은 열 이름으로 만들고 data_io.to_wide 를 그대로 통과시킨다.

    - 종목 n_codes−2 번은 150일째에 늦게 상장, n_codes−1 번은 600일째에 상장폐지
    - 종목마다 거래정지일(거래량 0, 시가·고가·저가 0) 몇 개를 넣는다 (실제 파일과 같은 모양)
    """
    # default_rng(seed) 는 '같은 seed 면 매번 같은 난수'를 주는 난수 생성기다 (테스트가 매번 같게)
    rng = np.random.default_rng(seed)
    # bdate_range 는 주말을 뺀 평일 날짜 목록을 만든다
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for j in range(n_codes):
        code = f"{100000 + j * 10:06d}"           # 끝자리 0 (보통주)
        market = "KOSPI" if j % 2 == 0 else "KOSDAQ"
        first = 150 if j == n_codes - 2 else 0
        last = 600 if j == n_codes - 1 else n_days - 1
        n = last - first + 1
        # 추세가 바뀌는 무작위 걸음: 오르내림이 섞여야 교차 신호가 생긴다
        drift = np.repeat(rng.normal(0, 0.004, n // 40 + 1), 40)[:n]
        ret = drift + rng.normal(0, 0.02, n)
        close = 20000 * np.exp(np.cumsum(ret))
        opn = close * np.exp(rng.normal(0, 0.01, n))
        opn[1:] = close[:-1] * np.exp(rng.normal(0, 0.01, n - 1))
        high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
        low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
        volume = rng.lognormal(np.log(200_000), 0.5, n)
        volume *= np.where(rng.random(n) < 0.03, 8.0, 1.0)   # 가끔 거래량 폭발 (16번 신호가 켜지게)
        halt = rng.random(n) < 0.01
        halt[0] = False
        for k in np.nonzero(halt)[0]:             # 거래정지일: 종가는 전날 종가, 나머지는 0
            close[k] = close[k - 1]
        volume[halt] = 0
        for i in range(n):
            o, h, l = (0.0, 0.0, 0.0) if halt[i] else (opn[i], high[i], low[i])
            rows.append(dict(date=dates[first + i], market=market, code=code, name=f"가짜{j}",
                             adj_open=o, adj_high=h, adj_low=l, adj_close=close[i],
                             open=o, close=close[i], volume=volume[i], volume_s=volume[i],
                             value=close[i] * volume[i], mktcap=close[i] * 1e7))
    return data_io.to_wide(pd.DataFrame(rows))


def perturb(data: dict, k: int, rng: np.random.Generator) -> dict:
    """k일부터 끝까지의 가격·거래량을 0.5~1.5배로 마구 바꾼 사본."""
    out = dict(data)
    for key in PERTURB_KEYS:
        arr = data[key].to_numpy(float).copy()
        arr[k:] *= rng.uniform(0.5, 1.5, arr[k:].shape)
        out[key] = pd.DataFrame(arr, index=data[key].index, columns=data[key].columns)
    return out


def check_no_lookahead(name: str, params: dict, data: dict, ks=None, seed: int = 1) -> list[int]:
    """k일 이후를 바꿨을 때 k일 이전 신호가 달라진 k 목록 (빈 목록이면 통과)."""
    rng = np.random.default_rng(seed)
    base = raw_signal(name, data, params)
    n = len(base)
    ks = ks or [n - 1, n // 4, n // 2, (3 * n) // 4, 30]
    bad = []
    for k in ks:
        new = raw_signal(name, perturb(data, k, rng), params)
        # iloc[:k] 는 '0번째부터 k−1번째 줄까지'다. 이 부분이 한 칸도 달라지면 안 된다
        if not base.iloc[:k].equals(new.iloc[:k]):
            bad.append(k)
    return bad


def check_all(data: dict | None = None) -> pd.DataFrame:
    """18개 신호의 모든 파라미터 세트(72세트)를 검사한 결과표."""
    data = data or make_synthetic()
    rows = []
    for name in SIGNALS:
        for p in iter_params(name):
            bad = check_no_lookahead(name, p, data)
            rows.append(dict(signal=name, params=p, n_signal=int(raw_signal(name, data, p).to_numpy().sum()),
                             failed_k=bad, ok=not bad))
    return pd.DataFrame(rows)
