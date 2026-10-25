#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 설계 검증 시뮬레이션 (연구용 — 플러그인 실행에는 필요 없음, numpy 필요).

    pip install numpy
    python3 research/v2_validation.py            # 전부 (약 20~30분)
    python3 research/v2_validation.py --quick    # 시뮬 수를 줄인 빠른 확인

세 가지를 같은 판정 규칙(Jeffreys Beta 사후, 시나리오 평균 차이, ROPE ±delta, 캘리브레이션 문턱 0.99,
batch 10 / max_runs 100 / 순차 판정)으로 잰다. verdict.py 와 같은 수식이지만 속도를 위해 numpy 로
다시 짰고, 체크포인트마다 사후 표본 S개를 한 벌 뽑아 모든 집계(전체·층별)에 같이 쓴다.

  A. 층을 '수정 전 통과율'로 나누면 생기는 편향 — A/A(효과 없음)에서 중간층 판정의 거짓 양성률과 방향
  B. 효과가 없을 때(A/A) v1 규칙(전체 평균)과 v2 규칙(중간층 주 판정 + 어느 층이든 개악 우선)의 거짓 양성률
  C. 효과가 있을 때 v1·v2 의 검출률과 평균 멈춤 시점(같은 경로로 짝지어 비교)

규칙 정의
  v1        : 전체 평균의 improved/regressed 에서 멈춤
  v2        : 주 판정 = 중간층(양팔 합산 통과율 0.10~0.90) 평균, 중간층이 없으면 전체 평균.
              주 판정·천장층·바닥층·전체 평균 중 하나라도 regressed 면 regressed, 아니면 주 판정이 improved 일 때 improved
  mid_bef   : (A 전용) 중간층을 수정 전 통과율로 나눈 주 판정, 가드 없음 — v2가 채택하지 않은 방식
  mid_pool  : (A 전용) 중간층을 양팔 합산 통과율로 나눈 주 판정, 가드 없음 — 층 배정 기준만 비교하기 위한 짝
"""
import argparse
import sys

import numpy as np

THRESHOLD = 0.99
DELTA = 0.03
CEIL, FLOOR = 0.90, 0.10


def simulate(p_before, p_after, rules, sims, seed, n_max=100, batch=10, S=3000, chunk=200):
    """같은 경로 위에서 여러 규칙을 돌린다. 반환: 규칙별 (improved 수, regressed 수, 멈춤 시점 합(방향별))."""
    rng = np.random.default_rng(seed)
    K = len(p_before)
    pb_true = np.array(p_before, dtype=float)
    pa_true = np.array(p_after, dtype=float)
    tally = {r: {"improved": 0, "regressed": 0, "stop_improved": 0, "stop_regressed": 0} for r in rules}
    done = 0
    while done < sims:
        B = min(chunk, sims - done)
        kb = np.zeros((B, K))
        ka = np.zeros((B, K))
        state = {r: np.zeros(B, dtype=int) for r in rules}  # 0 진행, 1 improved, 2 regressed
        for c in range(1, n_max // batch + 1):
            kb += rng.binomial(batch, np.broadcast_to(pb_true, (B, K)))
            ka += rng.binomial(batch, np.broadcast_to(pa_true, (B, K)))
            n = c * batch
            post_b = rng.beta((kb + 0.5)[:, :, None], (n - kb + 0.5)[:, :, None], size=(B, K, S))
            post_a = rng.beta((ka + 0.5)[:, :, None], (n - ka + 0.5)[:, :, None], size=(B, K, S))
            d = post_a - post_b
            judged = ~(((kb == n) & (ka == n)) | ((kb == 0) & (ka == 0)))

            def decide(mask):
                cnt = mask.sum(axis=1)
                m = ((d * mask[:, :, None]).sum(axis=1) / np.maximum(cnt, 1)[:, None])
                ok = cnt > 0
                return ((m > DELTA).mean(axis=1) >= THRESHOLD) & ok, ((m < -DELTA).mean(axis=1) >= THRESHOLD) & ok

            rp = (kb + ka) / (2.0 * n)
            rb = kb / float(n)
            mid_p = judged & (rp > FLOOR) & (rp < CEIL)
            ov_i, ov_r = decide(judged)
            dec = {}
            if "v1" in rules:
                dec["v1"] = (ov_i, ov_r)
            if "v2" in rules:
                prim = np.where(mid_p.any(axis=1)[:, None], mid_p, judged)
                pr_i, pr_r = decide(prim)
                _, ce_r = decide(judged & (rp >= CEIL))
                _, fl_r = decide(judged & (rp <= FLOOR))
                reg = pr_r | ce_r | fl_r | ov_r
                dec["v2"] = (pr_i & ~reg, reg)
            if "mid_bef" in rules:
                mid_b = judged & (rb > FLOOR) & (rb < CEIL)
                prim_b = np.where(mid_b.any(axis=1)[:, None], mid_b, judged)
                dec["mid_bef"] = decide(prim_b)
            if "mid_pool" in rules:
                prim_p = np.where(mid_p.any(axis=1)[:, None], mid_p, judged)
                dec["mid_pool"] = decide(prim_p)
            for r in rules:
                imp, reg = dec[r]
                live = state[r] == 0
                new_i = live & imp & ~reg
                new_r = live & reg
                state[r][new_i] = 1
                state[r][new_r] = 2
                tally[r]["stop_improved"] += int(new_i.sum()) * n
                tally[r]["stop_regressed"] += int(new_r.sum()) * n
        for r in rules:
            tally[r]["improved"] += int((state[r] == 1).sum())
            tally[r]["regressed"] += int((state[r] == 2).sum())
        done += B
    return tally


def _rate(x, sims):
    p = x / float(sims)
    return p, 1.96 * (p * (1 - p) / sims) ** 0.5


def part_a(sims):
    print("\n[A] 층 배정 기준에 따른 편향 — A/A, 중간층 주 판정(가드 없이)의 거짓 양성률과 방향 (budget 0.05)")
    print("    %-30s %-30s %-30s" % ("시나리오 구성(효과 없음)", "수정 전 통과율로 층 배정", "양팔 합산 통과율로 층 배정(v2)"))
    mixes = [("층 경계(90%) 근처 6개", [0.90] * 6),
             ("층 경계(10%) 근처 6개", [0.10] * 6),
             ("넓게 섞임 8개", [0.08, 0.12, 0.3, 0.5, 0.7, 0.88, 0.92, 0.97])]
    for i, (name, p) in enumerate(mixes):
        t = simulate(p, p, ["mid_bef", "mid_pool"], sims, seed=101 + i)
        cells = []
        for r in ("mid_bef", "mid_pool"):
            f, ci = _rate(t[r]["improved"] + t[r]["regressed"], sims)
            cells.append("%.3f±%.3f (거짓개선 %d/거짓개악 %d)" % (f, ci, t[r]["improved"], t[r]["regressed"]))
        print("    %-30s %-30s %-30s" % (name, cells[0], cells[1]))
        sys.stdout.flush()


def part_b(sims):
    print("\n[B] 효과가 없을 때(A/A) 거짓 양성률 — v1(전체 평균) vs v2 (budget 0.05)")
    mixes = [("중간층만 3개 (62%)", [0.62] * 3),
             ("중간 3 + 천장 6 (93~98%)", [0.6, 0.6, 0.6, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]),
             ("층 경계(90%) 근처 6개", [0.90] * 6),
             ("넓게 섞임 8개", [0.08, 0.12, 0.3, 0.5, 0.7, 0.88, 0.92, 0.97]),
             ("중간 1 (50%) + 바닥 4", [0.5, 0.04, 0.05, 0.06, 0.08])]
    for i, (name, p) in enumerate(mixes):
        t = simulate(p, p, ["v1", "v2"], sims, seed=201 + i)
        out = []
        for r in ("v1", "v2"):
            f, ci = _rate(t[r]["improved"] + t[r]["regressed"], sims)
            out.append("%s %.3f±%.3f" % (r, f, ci))
        print("    %-28s %s" % (name, "   ".join(out)))
        sys.stdout.flush()


def part_c(sims):
    print("\n[C] 효과가 있을 때 — 올바른 방향 검출률과 검출 시 평균 멈춤(팔당, max 100)")
    cases = [("중간 3개만 60->75%, 천장 2·바닥 2 그대로",
              [0.6, 0.6, 0.6, 0.97, 0.97, 0.05, 0.05], [0.75, 0.75, 0.75, 0.97, 0.97, 0.05, 0.05], "improved"),
             ("중간 2개만 55->70%, 천장 6개 그대로",
              [0.55, 0.55] + [0.95] * 6, [0.70, 0.70] + [0.95] * 6, "improved"),
             ("전부 중간 3개 60->75% (층 섞임 없음)",
              [0.6] * 3, [0.75] * 3, "improved"),
             ("중간 2개 그대로, 천장 3개 99->89% (잘 되던 게 깨짐)",
              [0.6, 0.6, 0.99, 0.99, 0.99], [0.6, 0.6, 0.89, 0.89, 0.89], "regressed")]
    for i, (name, pb, pa, want) in enumerate(cases):
        t = simulate(pb, pa, ["v1", "v2"], sims, seed=301 + i)
        out = []
        for r in ("v1", "v2"):
            hit = t[r][want]
            wrong = t[r]["regressed" if want == "improved" else "improved"]
            mean = (t[r]["stop_" + want] / float(hit)) if hit else float("nan")
            out.append("%s 검출 %.0f%% (평균 %.0f회, 반대방향 %d)" % (r, 100.0 * hit / sims, mean, wrong))
        print("    %-44s %s" % (name, " | ".join(out)))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="v2 설계 검증 시뮬레이션 (numpy)")
    ap.add_argument("--quick", action="store_true", help="시뮬 수를 줄인 빠른 확인")
    ap.add_argument("--parts", default="ABC", help="돌릴 부분 (예: A, BC — 기본 ABC)")
    args = ap.parse_args()
    sa, sb, sc = (300, 300, 200) if args.quick else (2000, 2000, 1000)
    print("v2 설계 검증 — 문턱 %.2f, delta %.2f, batch 10 / max_runs 100, 사후 표본 3000/체크포인트" % (THRESHOLD, DELTA))
    if "A" in args.parts.upper():
        part_a(sa)
    if "B" in args.parts.upper():
        part_b(sb)
    if "C" in args.parts.upper():
        part_c(sc)


if __name__ == "__main__":
    main()
