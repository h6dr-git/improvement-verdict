#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 설계 검증 시뮬레이션 (연구용 — 플러그인 실행에는 필요 없음, numpy 필요).

    pip install numpy
    python3 research/v2_validation.py            # 전부 (2코어 기준 1시간 안팎)
    python3 research/v2_validation.py --quick    # 시뮬 수를 줄인 빠른 확인
    python3 research/v2_validation.py --parts B  # 일부만

verdict.py 와 같은 판정 규칙(Jeffreys Beta 사후, 시나리오 평균 차이, ROPE ±delta, 캘리브레이션 문턱 0.99,
batch 10 / max_runs 100 순차 판정)을 속도를 위해 numpy 로 다시 짰다. 시나리오 하나짜리 집계는 도구와 똑같이
_stats.prob_diff(결정론적 적분)로, 둘 이상은 체크포인트마다 뽑은 사후 표본 S개 한 벌로 계산한다
(도구는 체크포인트 1만 개 — 표본이 적은 만큼 경계 근처에서 아주 조금 더 자주 문턱을 넘는다, 즉 보수적인 추정).
futility(무익 중단)는 넣지 않는다 — 거짓 양성률은 실제보다 높게(상한 쪽으로) 나온다.

  A. 층 배정 기준에 따른 쏠림 — 효과 없음(A/A), 중간층 판정만(가드 없이): 수정 전 통과율 vs 양팔 합산 통과율
  B. 효과가 없을 때(A/A) 거짓 양성률 — v1(전체 평균) vs v2, 검토에서 나온 최악 구성 포함
  C. 효과가 있을 때 올바른 방향 검출률과 검출 시 평균 멈춤 — v1 vs v2 (같은 경로로 짝지어 비교)
  D. 계획 모드(verdict.py --plan) 근사 대조 — 같은 계획을 시나리오별 판정 규칙(v2)으로 돌린 검출률

규칙
  v1       : 전체 평균의 improved/regressed 에서 멈춤
  v2       : 층 = 양팔 합산 통과율 (>=0.90 천장, <=0.10 바닥, 나머지 중간).
             중간층과 천장·바닥층이 함께 있으면 주 판정 = 중간층 평균(문턱 0.99), 아니면 전체 평균(= v1).
             주 판정일 때 천장·바닥층 평균의 개악·개선을 가드 문턱 0.999로 본다.
             개악이 하나라도 있으면 개악, 아니면 개선이 하나라도 있으면 개선.
  mid_bef  : (A 전용) 중간층을 수정 전 통과율로 나눈 주 판정, 가드 없음 — v2가 채택하지 않은 방식
  mid_pool : (A 전용) 중간층을 양팔 합산 통과율로 나눈 주 판정, 가드 없음 — 층 배정 기준만 비교하기 위한 짝
"""
import argparse
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "scripts"))
import _stats  # noqa: E402  (도구의 결정론적 적분 — 시나리오 하나짜리 집계용)

THRESHOLD = 0.99
GUARD_THRESHOLD = 0.999
DELTA = 0.03
CEIL, FLOOR = 0.90, 0.10


@lru_cache(maxsize=None)
def _exact(kb, ka, n):
    p_imp, p_reg, _ = _stats.prob_diff(int(kb), int(n), int(ka), int(n), DELTA)
    return p_imp, p_reg


def simulate(p_before, p_after, rules, sims, seed, n_max=100, batch=10, S=3000, chunk=200):
    """같은 경로 위에서 여러 규칙을 돌린다. 반환: 규칙별 improved/regressed 수와 방향별 멈춤 시점 합."""
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
            cache = {}

            def probs(mask):
                key = mask.tobytes()
                if key not in cache:
                    cnt = mask.sum(axis=1)
                    m = (d * mask[:, :, None]).sum(axis=1) / np.maximum(cnt, 1)[:, None]
                    pi, pr = (m > DELTA).mean(axis=1), (m < -DELTA).mean(axis=1)
                    for row in np.nonzero(cnt == 1)[0]:  # 시나리오 하나짜리 집계는 도구처럼 정확 적분
                        j = int(np.argmax(mask[row]))
                        pi[row], pr[row] = _exact(kb[row, j], ka[row, j], n)
                    cache[key] = (pi, pr, cnt > 0)
                return cache[key]

            def decide(mask, thr=THRESHOLD):
                pi, pr, ok = probs(mask)
                return (pi >= thr) & ok, (pr >= thr) & ok

            rp = (kb + ka) / (2.0 * n)
            rb = kb / float(n)
            dec = {}
            if "v1" in rules:
                dec["v1"] = decide(judged)
            if "v2" in rules or "mid_pool" in rules:
                ceil = judged & (rp >= CEIL)
                flo = judged & (rp <= FLOOR)
                mid = judged & ~ceil & ~flo
                use_mid = mid.any(axis=1) & (ceil | flo).any(axis=1)
                prim = np.where(use_mid[:, None], mid, judged)
                pr_i, pr_r = decide(prim)
                if "mid_pool" in rules:
                    dec["mid_pool"] = (pr_i, pr_r)
                if "v2" in rules:
                    reg, imp = pr_r.copy(), pr_i.copy()
                    for side in (ceil, flo):
                        g_i, g_r = decide(side, GUARD_THRESHOLD)
                        reg |= g_r & use_mid
                        imp |= g_i & use_mid
                    dec["v2"] = (imp & ~reg, reg)
            if "mid_bef" in rules:
                ceil_b = judged & (rb >= CEIL)
                flo_b = judged & (rb <= FLOOR)
                mid_b = judged & ~ceil_b & ~flo_b
                use_b = mid_b.any(axis=1) & (ceil_b | flo_b).any(axis=1)
                dec["mid_bef"] = decide(np.where(use_b[:, None], mid_b, judged))
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
    print("\n[A] 층 배정 기준에 따른 쏠림 — A/A, 중간층 주 판정(가드 없이)의 거짓 양성률과 방향 (budget 0.05)")
    print("    %-30s %-34s %-34s" % ("시나리오 구성(효과 없음)", "수정 전 통과율로 층 배정", "양팔 합산 통과율로 층 배정(v2)"))
    mixes = [("층 경계(90%) 근처 6개", [0.90] * 6),
             ("층 경계(10%) 근처 6개", [0.10] * 6),
             ("넓게 섞임 8개", [0.08, 0.12, 0.3, 0.5, 0.7, 0.88, 0.92, 0.97])]
    for i, (name, p) in enumerate(mixes):
        t = simulate(p, p, ["mid_bef", "mid_pool"], sims, seed=101 + i)
        cells = []
        for r in ("mid_bef", "mid_pool"):
            f, ci = _rate(t[r]["improved"] + t[r]["regressed"], sims)
            cells.append("%.3f±%.3f (거짓개선 %d/거짓개악 %d)" % (f, ci, t[r]["improved"], t[r]["regressed"]))
        print("    %-30s %-34s %-34s" % (name, cells[0], cells[1]))
        sys.stdout.flush()


def part_b(sims):
    print("\n[B] 효과가 없을 때(A/A) 거짓 양성률 — v1(전체 평균) vs v2 (budget 0.05)")
    mixes = [("중간층만 3개 (62%)", [0.62] * 3),
             ("중간 3 + 천장 6 (93~98%)", [0.6, 0.6, 0.6, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]),
             ("층 경계(90%) 근처 6개", [0.90] * 6),
             ("넓게 섞임 8개", [0.08, 0.12, 0.3, 0.5, 0.7, 0.88, 0.92, 0.97]),
             ("중간 1 (50%) + 바닥 4", [0.5, 0.04, 0.05, 0.06, 0.08]),
             ("중간 1 (50%) + 92% + 8% (검토 최악)", [0.50, 0.92, 0.08]),
             ("중간 1 (50%) + 95% + 5%", [0.50, 0.95, 0.05]),
             ("중간 1 (50%) + 93% x3", [0.50, 0.93, 0.93, 0.93])]
    for i, (name, p) in enumerate(mixes):
        t = simulate(p, p, ["v1", "v2"], sims, seed=201 + i)
        out = []
        for r in ("v1", "v2"):
            f, ci = _rate(t[r]["improved"] + t[r]["regressed"], sims)
            out.append("%s %.3f±%.3f" % (r, f, ci))
        print("    %-34s %s" % (name, "   ".join(out)))
        sys.stdout.flush()


def part_c(sims, only=None):
    print("\n[C] 효과가 있을 때 — 올바른 방향 검출률과 검출 시 평균 멈춤(팔당, max 100)")
    cases = [("중간 3개만 60->75%, 천장 2·바닥 2 그대로",
              [0.6, 0.6, 0.6, 0.97, 0.97, 0.05, 0.05], [0.75, 0.75, 0.75, 0.97, 0.97, 0.05, 0.05], "improved"),
             ("중간 2개만 55->70%, 천장 6개 그대로",
              [0.55, 0.55] + [0.95] * 6, [0.70, 0.70] + [0.95] * 6, "improved"),
             ("전부 중간 3개 60->75% (층 섞임 없음)",
              [0.6] * 3, [0.75] * 3, "improved"),
             ("85->99%로 크게 고친 3개 + 그대로인 50% 1개",
              [0.85, 0.85, 0.85, 0.5], [0.99, 0.99, 0.99, 0.5], "improved"),
             ("0->20%로 살린 3개 + 그대로인 50% 1개",
              [0.0, 0.0, 0.0, 0.5], [0.2, 0.2, 0.2, 0.5], "improved"),
             ("중간 2개 그대로, 천장 3개 99->89% (잘 되던 게 깨짐)",
              [0.6, 0.6, 0.99, 0.99, 0.99], [0.6, 0.6, 0.89, 0.89, 0.89], "regressed"),
             ("중간 2개 그대로, 천장 3개 100->93% (조금씩 깨짐)",
              [0.6, 0.6, 1.0, 1.0, 1.0], [0.6, 0.6, 0.93, 0.93, 0.93], "regressed"),
             ("중간 3개 60->52% + 천장 3개 97->92% (층에 고르게 퍼진 개악)",
              [0.6, 0.6, 0.6, 0.97, 0.97, 0.97], [0.52, 0.52, 0.52, 0.92, 0.92, 0.92], "regressed"),
             ("중간 3개 60->70% + 천장 3개 92->97% (층에 고르게 퍼진 개선)",
              [0.6, 0.6, 0.6, 0.92, 0.92, 0.92], [0.70, 0.70, 0.70, 0.97, 0.97, 0.97], "improved")]
    for i, (name, pb, pa, want) in enumerate(cases):
        if only and (i + 1) not in only:
            continue
        t = simulate(pb, pa, ["v1", "v2"], sims, seed=301 + i)
        out = []
        for r in ("v1", "v2"):
            hit = t[r][want]
            wrong = t[r]["regressed" if want == "improved" else "improved"]
            mean = (t[r]["stop_" + want] / float(hit)) if hit else float("nan")
            out.append("%s 검출 %.0f%% (평균 %.0f회, 반대방향 %d)" % (r, 100.0 * hit / sims, mean, wrong))
        print("    %-46s %s" % (name, " | ".join(out)))
        sys.stdout.flush()


def part_d(sims):
    print("\n[D] 계획 모드 근사 대조 — 시나리오별 판정 규칙(v2) 검출률 (batch 10 / max 100)")
    cases = [("--base .60 --target .75 --scenarios 3 --unchanged-middle 1", [0.6] * 4, [0.75] * 3 + [0.6]),
             ("--base .85 --target .97 --scenarios 1 --unchanged-middle 2 (그대로인 쪽 85%)",
              [0.85, 0.85, 0.85], [0.97, 0.85, 0.85]),
             ("--base .85 --target .97 --scenarios 1 --unchanged-middle 2 (그대로인 쪽 60%)",
              [0.85, 0.6, 0.6], [0.97, 0.6, 0.6]),
             ("--base .02 --target .16 --scenarios 2 --unchanged-middle 2 (그대로인 쪽 50%)",
              [0.02, 0.02, 0.5, 0.5], [0.16, 0.16, 0.5, 0.5])]
    for i, (name, pb, pa) in enumerate(cases):
        t = simulate(pb, pa, ["v2"], sims, seed=901 + i)
        f, ci = _rate(t["v2"]["improved"], sims)
        print("    %-70s v2 검출 %.3f±%.3f" % (name, f, ci))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="v2 설계 검증 시뮬레이션 (numpy)")
    ap.add_argument("--quick", action="store_true", help="시뮬 수를 줄인 빠른 확인")
    ap.add_argument("--parts", default="ABCD", help="돌릴 부분 (예: A, BC — 기본 ABCD)")
    ap.add_argument("--c-cases", default=None, help="C에서 돌릴 줄 번호만 (예: 8 또는 1,8 — 기본 전부)")
    args = ap.parse_args()
    sa, sb, sc, sd = (300, 300, 200, 300) if args.quick else (2000, 4000, 1000, 2000)
    print("v2 설계 검증 — 문턱 %.2f(가드 %.3f), delta %.2f, batch 10 / max_runs 100, 사후 표본 3000/체크포인트"
          % (THRESHOLD, GUARD_THRESHOLD, DELTA))
    parts = args.parts.upper()
    if "A" in parts:
        part_a(sa)
    if "B" in parts:
        part_b(sb)
    if "C" in parts:
        part_c(sc, [int(x) for x in args.c_cases.split(",")] if args.c_cases else None)
    if "D" in parts:
        part_d(sd)


if __name__ == "__main__":
    main()
