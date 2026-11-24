"""python -m indexbeater.krx_source {collect|splits|dividends|jumps|yahoo_divs|rights|adjust} [옵션]"""
from __future__ import annotations

import importlib
import sys

CMDS = ("collect", "splits", "dividends", "jumps", "yahoo_divs", "rights", "adjust")   # 쓸 수 있는 명령


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in CMDS:     # 명령이 없거나 모르는 명령이면 사용법만 보여 준다
        print(__doc__)
        return 2
    # 입력한 이름(예: splits)의 파일을 그때 불러와서 그 파일의 main 을 실행한다.
    # 필요한 것만 불러오면 시작이 빠르고, 키가 필요 없는 명령은 키 없이도 돈다.
    importlib.import_module(f".{argv[0]}", __package__).main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
