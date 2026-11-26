"""매도 전용 규칙 S1~S4 (계획서 부록 B).

S1 고정 손절·익절, S2 ATR 트레일링, S3 시간 청산은 ExitRule 하나로 묶어 엔진에 넘긴다.
S4 시장 필터는 '신규 매수 금지' 규칙이라 매수 신호에 곱한다.

1단계(11/21)에서는 S3 시간 청산과 S1 손절·익절만 만든다.
S2 ATR 트레일링·S4 시장 필터는 STEP 4 할 일이라 아직 만들지 않았다 (11/20 결정 7A).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from indexbeater import config as C


# @dataclass 는 '값 몇 개를 묶어 두는 상자'를 간단히 만드는 표시다.
# frozen=True 는 만든 뒤에 값을 못 바꾸게 잠근다 (실수로 규칙이 바뀌는 것을 막기 위해).
@dataclass(frozen=True)
class ExitRule:
    """매도 규칙 묶음.

    hold : S3 시간 청산 H (매수일 b 로부터 H 거래일 뒤 시가에 판다, 기준 5.3)
    stop : S1 손절 비율 (예: 0.05 = 매수가 −5%). None 이면 안 쓴다
    take : S1 익절 비율 (예: 0.10 = 매수가 +10%). None 이면 안 쓴다
    """
    hold: int
    stop: Optional[float] = None
    take: Optional[float] = None

    def __post_init__(self):
        # 보유 기간은 최대 20일을 넘으면 안 된다 (기준 4.3 퍼지 20일과 맞추기 위해)
        if not (1 <= self.hold <= C.MAX_HOLD):
            raise ValueError(f"hold={self.hold} 는 1~{C.MAX_HOLD} 사이여야 한다 (기준 4.3)")
        for name, v in (("stop", self.stop), ("take", self.take)):
            if v is not None and not (0 < v < 1):
                raise ValueError(f"{name}={v} 는 0과 1 사이 비율이어야 한다")
