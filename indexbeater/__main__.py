"""명령줄 진입점:  python -m indexbeater <명령>

  check        자가 점검 (환경·폴더·데이터·미래 정보 테스트)   STEP 0
  collect      데이터 수집                                    STEP 1
  crosscheck   네이버 금융 교차 검증                           STEP 1
  preprocess   전처리                                         STEP 2
  optimize     매수 신호 최적화 (설계 구간만)                   STEP 3
  benchmark    배당(분배금) 반영 ETF 비교 지수 (설계 구간만)     D4
  combo        신호 AND 조합 (2신호 153개·3신호, 설계 구간만)   5.6~5.7
  stage4       조합 합격 시험 1~7 (순위·SPA·랜덤·비용, 설계 구간만)   5.8·5.11·5.12
  classic      대표 일봉차트 조합 8개 별개 비교 (참고, 설계 구간만)   실험기록 30
  closeexit    다음 날 종가 매도 vs 5거래일 뒤 시가 매도 (참고, 1등 4칸만)   실험기록 32
  teamcmp      팀원 최적 전략 동일 조건 비교 (참고, 설계 구간만)   실험기록 34
  top3csv      TOP3 매수 신호 CSV (팀원 매도 ML·DL 용, 설계 구간만)   실험기록 36
  sellml       매도 ML·DL 공통 시작 코드 (점검 --check · 학습 --run, 설계 구간만)   11/26
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {"check": "check", "collect": "collect", "crosscheck": "crosscheck_naver",
            "preprocess": "preprocess", "optimize": "optimize", "benchmark": "benchmark", "combo": "combo",
            "stage4": "stage4", "classic": "classic", "closeexit": "close_exit",
            "teamcmp": "team_compare", "top3csv": "buy_export",
            "sellml": "sell_common"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"알 수 없는 명령: {cmd}\n{__doc__}")
        return 2
    module = importlib.import_module(f".{COMMANDS[cmd]}", __package__)
    return module.main(rest) or 0


if __name__ == "__main__":
    sys.exit(main())
