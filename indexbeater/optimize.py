"""STEP 3 매수 신호 최적화 (1차 목적) — 매도는 S3 시간 청산으로 통일.

- 데이터는 설계 구간 끝(2019-12-31)까지만 불러온다. 2020년 이후는 열지 않는다.
- 신호마다 파라미터 그리드 × H(5·10·20)를 돌려 거래수·승률·평균·Edge·t값·카이제곱 p값을 구한다.
- Base(무작위 매수)는 같은 매도 규칙으로 아무 날이나 산 매매의 평균수익률이다.
- 연도별(2015~2019) Edge로 '고르게 좋은가'를 함께 본다.

실행: python -m indexbeater optimize [--signals S01,S07] [--holds 5,10] [--codes 300]
결과: results/single_signal.csv, results/single_signal_best.csv, figures/heatmap_*.png
"""

# TODO(STEP 3): 계획서 해당 STEP의 '할 일'에 따라 구현한다.


def main(argv=None):
    raise NotImplementedError("STEP 3 구현 전입니다 (optimize.py)")
