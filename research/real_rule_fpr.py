#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실제 verdict.py 판정 함수로 재는 A/A 거짓 양성률 (표준 라이브러리만 — 연구용).

    python3 research/real_rule_fpr.py src/scripts 20000 0.50 0.95 0.05   # 시나리오별 통과율(효과 없음)

batch 10 / max_runs 100 순차 판정을 시뮬레이션하며, 층 배정·주 판정·가드는 verdict.judge_scenario /
headline_decision 을 그대로 쓴다. 시나리오 하나짜리 층은 _stats.prob_diff(정확 적분), 둘 이상인 층은
verdict.aggregate_probs(표본 1만 개). futility 는 빼서 상한 쪽 값이다. 'primary_only' 는 가드를 뺀 주 판정만의 값.
시나리오가 적은 구성에 맞춘 빠른 경로라, 층 경계를 자주 넘나드는 구성은 느리다.
"""
import random
import sys

sys.path.insert(0, sys.argv[1])
import _stats  # noqa: E402
import verdict  # noqa: E402

NS = int(sys.argv[2])
PS = [float(x) for x in sys.argv[3:]]
THR, _, _ = verdict.load_calibration(sys.argv[1] + "/calibration-default.json", 10, 100, 0.05)
BUDGET, DELTA = 0.05, 0.03


def group_probs(cells, seed):
    if len(cells) == 1:
        return _stats.prob_diff(*cells[0], DELTA)
    return verdict.aggregate_probs(cells, None, DELTA, seed, verdict.CKPT_SAMPLES)["all"]


rng = random.Random(4242)
hits = {"v2": 0, "primary_only": 0}
for s in range(NS):
    b = [[] for _ in PS]
    a = [[] for _ in PS]
    done = {k: False for k in hits}
    for c in range(1, 11):
        for i, p in enumerate(PS):
            b[i].extend(rng.random() < p for _ in range(10))
            a[i].extend(rng.random() < p for _ in range(10))
        recs = [verdict.judge_scenario("s%d" % i, b[i], a[i], DELTA, THR, BUDGET, 0.90, 0.10)[0] for i in range(len(PS))]
        judged = [r for r in recs if r["_judged"]]
        if not judged:
            continue
        agg = {}
        groups = {}
        for r in judged:
            groups.setdefault(r["stratum"], []).append(r["_eff"])
        for g, cells in groups.items():
            agg[g] = group_probs(cells, 1000 * s + c)
        if "middle" in agg and ("ceiling" in agg or "floor" in agg):
            agg["all"] = (0.0, 0.0, 1.0)  # basis is middle: the overall average is not used
        else:
            agg["all"] = group_probs([r["_eff"] for r in judged], 7000 * s + c)
        head = verdict.headline_decision(agg, THR, BUDGET, "stratified")
        if not done["v2"] and head["verdict"] in ("improved", "regressed"):
            done["v2"] = True
        pp = head["primary_probs"]
        if not done["primary_only"] and (pp[0] >= THR or pp[1] >= THR):
            done["primary_only"] = True
    for k in hits:
        hits[k] += done[k]
    if (s + 1) % 2000 == 0:
        print("progress", s + 1, {k: round(v / float(s + 1), 4) for k, v in hits.items()})
        sys.stdout.flush()
for k, v in hits.items():
    p = v / float(NS)
    print("%s FPR %.4f ±%.4f" % (k, p, 1.96 * (p * (1 - p) / NS) ** 0.5))
