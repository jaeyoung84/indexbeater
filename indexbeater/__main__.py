"""명령줄 진입점:  python -m indexbeater <명령>

  check        자가 점검 (환경·폴더·데이터·미래 정보 테스트)   STEP 0
  collect      데이터 수집                                    STEP 1
  crosscheck   네이버 금융 교차 검증                           STEP 1
  preprocess   전처리                                         STEP 2
  optimize     매수 신호 최적화 (설계 구간만)                   STEP 3
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {"check": "check", "collect": "collect", "crosscheck": "crosscheck_naver",
            "preprocess": "preprocess", "optimize": "optimize"}


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
