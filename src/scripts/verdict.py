#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""개선·개악 판별기 CLI — before/after 통과 결과(JSONL)를 받아 improved/regressed/no_difference/undetermined 를 판정.

사용 예시:
  python3 src/scripts/synth.py --case 2 | python3 src/scripts/verdict.py - --delta 0.03 --batch-size 20 --max-runs 400 --yes
  python3 src/scripts/verdict.py runs.jsonl --delta 0.03 --online          # QA 도중: 멈춤/계속 신호(종료코드)
  python3 src/scripts/verdict.py --plan --base 0.62 --target 0.75 --delta 0.03   # 돌리기 전: 필요한 실행 수

- 판정 수식은 전부 _stats.decide()/prob_diff() (calibrate.py 와 동일 경로 — 캘리브레이션 전제)
- v2: 층(천장/중간/바닥)은 양팔 합산 통과율로 나누고, 주 판정은 중간층 기준(개악은 어느 층이든 우선).
  --primary overall 이면 v1과 같은 판정·멈춤 규칙(전체 평균)으로 돈다.
- 종료코드: 0 정상 / 1 selftest 실패 / 2 입력·설정 오류 / 3 delta 미확정 중단
  --online 일 때만: 0 계속 / 10 멈춤(개선 확정) / 11 멈춤(개악 확정) / 12 멈춤(무익) / 13 멈춤(예산 소진)
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stats

DEFAULT_SEED = 20260710
DEFAULT_BATCH = 10
DEFAULT_MAX_RUNS = 100
UNCALIBRATED_THRESHOLD = 0.99  # 캘리브레이션이 없는 설정의 기본문턱 — 그 설정의 거짓 양성률은 보장되지 않는다
FINAL_SAMPLES = 20000          # 전체·층별 집계 판정 MC 표본 수 — 시나리오당
CKPT_SAMPLES = 10000           # 순차 재생 체크포인트용 (문턱 대비 MC 오차 ~0.001)
FUT_SAMPLES = 4000             # futility 상한 검사용 (경계 비교라 정밀도 요구 낮음)
VERDICTS = ("improved", "regressed", "no_difference", "undetermined")
STRATA = ("middle", "ceiling", "floor")
# 층 묶음: 양팔 합산 통과율 구간 — 두 팔에 대칭이라 효과가 없을 때 판정 방향을 기울이지 않는다(README 'v2 검증' A).
SIDE_STRATA = ("ceiling", "floor")
# 가드: 주 판정 근거가 중간층일 때 천장·바닥 묶음의 개선·개악도 본다(크게 고친 시나리오·깨진 시나리오를 놓치지 않게).
# 판정 기회가 늘어나는 만큼 캘리브레이션 문턱(0.99)보다 엄격한 문턱으로만 확정한다(README 'v2 검증' B).
GUARD_THRESHOLD = 0.999
STRATUM_KO = {"ceiling": "천장", "floor": "바닥", "middle": "중간", "overall": "전체 평균"}
ONLINE_EXIT = {"continue": 0, "improved": 10, "regressed": 11, "futility": 12, "max_runs": 13}
SCHEMA_VERSION = 2


def _fail_input(line_no, field, msg):
    """입력 스키마 위반: 어느 줄의 어느 필드인지 정확히 지목하고 종료코드 2."""
    if field is None:
        sys.stderr.write("입력 오류: %d번째 줄: %s\n" % (line_no, msg))
    else:
        sys.stderr.write("입력 오류: %d번째 줄, 필드 '%s' %s\n" % (line_no, field, msg))
    sys.exit(2)


def parse_input(text):
    """JSONL 엄격 검증. 반환: (도착 순서 실행 목록, 시나리오 등장 순서)."""
    runs = []
    order = []
    seen = set()
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except ValueError:
            _fail_input(i, None, "JSON 파싱 실패")
        if not isinstance(obj, dict):
            _fail_input(i, None, "JSON 객체가 아님")
        for field in ("scenario_id", "arm", "passed"):
            if field not in obj:
                _fail_input(i, field, "누락")
        sid = obj["scenario_id"]
        if not isinstance(sid, str):
            _fail_input(i, "scenario_id", "문자열이어야 함 (현재 %r)" % (sid,))
        arm = obj["arm"]
        if arm not in ("before", "after"):
            _fail_input(i, "arm", "'before'|'after' 여야 함 (현재 %r)" % (arm,))
        passed = obj["passed"]
        if not isinstance(passed, bool):
            _fail_input(i, "passed", "불리언이어야 함 (현재 %r)" % (passed,))
        # 그 외 필드는 무시(전방 호환)
        runs.append((sid, arm, passed))
        if sid not in seen:
            seen.add(sid)
            order.append(sid)
    if not runs:
        sys.stderr.write("입력 오류: 데이터가 없습니다 (JSONL 한 줄 = 실행 하나)\n")
        sys.exit(2)
    return runs, order


def resolve_delta(args, scen, order):
    """delta 미지정 시: 관측 기저 통과율·현재 n으로 A/A 노이즈 바닥을 계산해 제안 + 진행 확인.

    기본값을 임의로 넣지 않는다. 결정은 사람이 한다.
    """
    if args.delta is not None:
        if not (0.0 < args.delta < 0.5):
            sys.stderr.write("설정 오류: --delta 는 (0, 0.5) 사이여야 합니다 (현재 %r)\n" % args.delta)
            sys.exit(2)
        return args.delta, "user"
    if args.online:
        # 제안값은 그때까지의 표본 수에 따라 달라진다 — 순차 판정 도중 delta가 움직이면 캘리브레이션이 무의미해진다.
        print("[delta 미지정] --online 에서는 delta를 QA 시작 전에 --delta 로 고정해야 합니다(--yes 불가)."
              " 정하기 어렵다면 --online 없이 한 번 실행해 제안값과 근거를 보세요.")
        sys.exit(3)
    tb = sum(sum(scen[s]["before"]) for s in order)
    nb = sum(len(scen[s]["before"]) for s in order)
    base = (tb + 0.5) / (nb + 1.0) if nb else 0.5
    ns = [min(len(scen[s]["before"]), len(scen[s]["after"])) for s in order]
    ns = sorted(v for v in ns if v > 0) or [10]
    n_rep = ns[len(ns) // 2]
    floor95 = _stats.aa_noise_floor(base, n_rep, sims=4000, seed=args.seed)
    sug = max(0.01, math.ceil(floor95 * 200.0) / 200.0)  # 0.005 단위 올림
    print("[delta 미지정] 제안값과 근거:")
    print("  관측 기저 통과율(before) %.3f, 대표 팔당 실행 수 %d회 기준," % (base, n_rep))
    print("  양팔이 완전히 동일해도 우연만으로 |통과율 차이|가 %.1f%%p까지 벌어집니다(95%% 분위, 시드 고정 이항 MC)." % (floor95 * 100))
    print("  => 제안 delta = %.3f (그보다 작은 delta는 노이즈를 차이로 오인할 수 있음)" % sug)
    if args.yes:
        print("  --yes: 제안값 %.3f 으로 진행합니다." % sug)
        return sug, "suggested(--yes)"
    if args.input != "-" and sys.stdin.isatty():
        try:
            ans = input("  이 delta로 진행할까요? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans in ("y", "yes"):
            return sug, "suggested(interactive)"
        print("  중단: --delta 로 직접 지정해 다시 실행하세요.")
        sys.exit(3)
    print("  비대화형 실행: 위 제안을 검토한 뒤 --delta 로 지정하거나 --yes 로 제안값을 수락하세요.")
    sys.exit(3)


def load_entries(path):
    """캘리브레이션 파일의 엔트리 목록(없거나 깨졌으면 빈 목록)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries", []) if isinstance(data, dict) else data
    except (OSError, ValueError):
        return []


def load_calibration(path, batch, maxr, budget):
    """(batch_size, max_runs, error_budget) 매칭 엔트리 로드. 없으면 보수 기본문턱 + 정직 고지."""
    for e in load_entries(path):
        cfg = e.get("config", {})
        if cfg.get("batch_size") == batch and cfg.get("max_runs") == maxr \
                and abs(cfg.get("error_budget", -1) - budget) < 1e-9:
            calib = {
                "threshold": e["threshold"],
                "sims": e.get("sims", 0),
                "achieved_fpr": e.get("achieved_fpr"),
                "fpr_ci95": e.get("fpr_ci95"),
                "calibrated_at_delta": e.get("calibrated_at_delta"),
                "source": "%s (batch=%d, max_runs=%d, budget=%g, A/A 몬테카를로)"
                          % (os.path.basename(path), batch, maxr, budget),
            }
            return e["threshold"], calib, False
    calib = {
        "threshold": UNCALIBRATED_THRESHOLD,
        "sims": 0,
        "achieved_fpr": None,
        "fpr_ci95": None,
        "source": "uncalibrated — 이 설정(batch=%d, max_runs=%d, budget=%g)의 캘리브레이션 없음. "
                  "기본문턱 %.2f 사용 — 이 설정의 거짓 양성률은 보장되지 않음(훔쳐보기가 많을수록 커짐). "
                  "calibrate.py 실행 권장." % (batch, maxr, budget, UNCALIBRATED_THRESHOLD),
    }
    return UNCALIBRATED_THRESHOLD, calib, True


def assign_stratum(kb, nbe, ka, nae, ceiling, floor):
    """층 묶음: 양팔 합산 통과율이 ceiling 이상이면 천장, floor 이하면 바닥, 나머지는 중간.

    두 팔에 대칭인 규칙이라(두 팔을 바꿔도 같은 층) 효과가 없을 때 층 배정이 판정 방향을 기울이지 않는다.
    수정 전 통과율만 보면 경계 근처에서 우연히 낮게(높게) 나온 시나리오만 중간층에 들어와 '개선'('개악')
    쪽으로 쏠린다(README 'v2 검증' A). 85%->99%처럼 천장 쪽으로 크게 고친 시나리오는 천장 묶음에 들어가지만,
    천장·바닥 묶음의 개선·개악도 가드 문턱으로 주 판정에 반영되므로 사라지지 않는다.
    """
    rate_pool = (kb + ka) / float(nbe + nae)
    if rate_pool >= ceiling:
        return "ceiling"
    if rate_pool <= floor:
        return "floor"
    return "middle"


def judge_scenario(sid, blist, alist, delta, threshold, budget, ceiling, floor):
    """시나리오 하나의 판정과 층 배정(assign_stratum — LLM 분류 없음).

    팔 불균형이면 적은 쪽 n에 맞춰 도착 순서 앞쪽만 쓰는 보수 판정(_eff)을 하고 imbalanced_arms 를 단다.
    """
    nb, na = len(blist), len(alist)
    flags = []
    warns = []
    m = min(nb, na)
    if m == 0:
        rec = {"scenario_id": sid, "stratum": "middle", "n_before": nb, "n_after": na,
               "rate_before": None, "rate_after": None,
               "verdict": "undetermined", "p_improved": 0.0, "p_regressed": 0.0, "p_rope": 0.0,
               "flags": ["imbalanced_arms", "needs_more_runs"],
               "_eff": (0, 0, 0, 0), "_judged": False}
        warns.append("시나리오 '%s': 한쪽 팔의 실행이 없어(before %d / after %d) 판정 불가 — 합산에서 제외" % (sid, nb, na))
        return rec, warns
    if max(nb, na) >= 1.5 * m:
        flags.append("imbalanced_arms")
        warns.append("시나리오 '%s': 팔 불균형(before %d / after %d) — 적은 쪽 %d회 기준으로 보수 판정" % (sid, nb, na, m))
        kb, ka, nbe, nae = sum(blist[:m]), sum(alist[:m]), m, m
    else:
        kb, ka, nbe, nae = sum(blist), sum(alist), nb, na
    rate_b, rate_a = kb / float(nbe), ka / float(nae)
    stratum = assign_stratum(kb, nbe, ka, nae, ceiling, floor)
    # 해석 플래그는 '두 팔 모두' 기준: 85%->99%는 천장 묶음이어도 '개선 여지가 작다'가 아니고,
    # 0%->20%는 바닥 묶음이어도 '태스크 수정으로 풀 문제가 아니다'가 아니다.
    if min(rate_b, rate_a) >= ceiling:
        flags.append("ceiling")
    elif max(rate_b, rate_a) <= floor:
        flags.extend(["floor", "not_task_fixable"])
        warns.append("시나리오 '%s': 바닥(수정 전 %.0f%%, 수정 후 %.0f%%) — 수정 전후 모두 거의 실패. "
                     "태스크 수정으로 풀 문제가 아닙니다" % (sid, rate_b * 100, rate_a * 100))
    no_info = (kb == nbe and ka == nae) or (kb == 0 and ka == 0)
    verdict, p_imp, p_reg, p_rope = _stats.decide(kb, nbe, ka, nae, delta, threshold, budget)
    if no_info:
        flags.append("no_information")
    rec = {"scenario_id": sid, "stratum": stratum, "n_before": nb, "n_after": na,
           "rate_before": round(rate_b, 4), "rate_after": round(rate_a, 4),
           "verdict": verdict, "p_improved": round(p_imp, 6), "p_regressed": round(p_reg, 6),
           "p_rope": round(p_rope, 6), "flags": flags,
           "_eff": (kb, nbe, ka, nae), "_judged": not no_info}
    if verdict == "undetermined":
        flags.append("needs_more_runs")
        pb = (kb + 0.5) / (nbe + 1.0)
        pa = (ka + 0.5) / (nae + 1.0)
        rec["needed_runs_per_arm"] = _stats.needed_runs(pb, pa, delta, budget)
    return rec, warns


def aggregate_probs(cells, groups, delta, seed, nsamples):
    """전체 평균과 층별 평균의 확률 3종을 **같은 사후 표본 한 벌**로 계산한다.

    cells: (kb, nb, ka, na) 목록, groups: 칸별 층 이름(None 이면 전체만).
    반환: {"all": (p_imp, p_reg, p_rope), "<층>": (...), ...}
    - 칸이 하나뿐인 집계는 결정론적 prob_diff, 둘 이상은 시드 고정 샘플링(층화 평균 = 단순 평균).
    - "all" 은 v1 overall_probs 와 같은 시드·같은 추출 순서라 수치가 비트 단위로 같다.
    - 칸이 없으면 "all" 만 사전분포끼리 비교(불확실성이 그대로 드러남).
    """
    out = {}
    if not cells:
        out["all"] = _stats.prob_diff(0, 0, 0, 0, delta)
        return out
    members = {}
    if groups is not None:
        for i, g in enumerate(groups):
            members.setdefault(g, []).append(i)
    if len(cells) == 1:
        kb, nb, ka, na = cells[0]
        out["all"] = _stats.prob_diff(kb, nb, ka, na, delta)
    for g, idx in members.items():
        if len(idx) == 1:
            kb, nb, ka, na = cells[idx[0]]
            out[g] = _stats.prob_diff(kb, nb, ka, na, delta)
    if len(cells) == 1:
        return out
    mc_groups = [g for g, idx in members.items() if len(idx) > 1]
    slot_of = {g: j for j, g in enumerate(mc_groups)}
    slots = [slot_of.get(g, -1) for g in groups] if groups is not None else [-1] * len(cells)
    inv_g = [1.0 / len(members[g]) for g in mc_groups]
    G = len(mc_groups)
    rng = random.Random(seed)
    bv = rng.betavariate
    params = [(kb + 0.5, nb - kb + 0.5, ka + 0.5, na - ka + 0.5) for (kb, nb, ka, na) in cells]
    inv = 1.0 / len(cells)
    ci = cr = 0
    gci = [0] * G
    gcr = [0] * G
    if G == 0:
        for _ in range(nsamples):
            s = 0.0
            for ab, bb, aa, ba in params:
                s += bv(aa, ba) - bv(ab, bb)
            d = s * inv
            if d > delta:
                ci += 1
            elif d < -delta:
                cr += 1
    else:
        rows = [(p[0], p[1], p[2], p[3], sl) for p, sl in zip(params, slots)]
        for _ in range(nsamples):
            s = 0.0
            gs = [0.0] * G
            for ab, bb, aa, ba, sl in rows:
                d = bv(aa, ba) - bv(ab, bb)
                s += d
                if sl >= 0:
                    gs[sl] += d
            d = s * inv
            if d > delta:
                ci += 1
            elif d < -delta:
                cr += 1
            for j in range(G):
                dg = gs[j] * inv_g[j]
                if dg > delta:
                    gci[j] += 1
                elif dg < -delta:
                    gcr[j] += 1
    n = float(nsamples)
    out["all"] = (ci / n, cr / n, max(0.0, 1.0 - ci / n - cr / n))
    for j, g in enumerate(mc_groups):
        pi, pr = gci[j] / n, gcr[j] / n
        out[g] = (pi, pr, max(0.0, 1.0 - pi - pr))
    return out


_AGG_KEY = {"middle": "middle", "overall": "all", "ceiling": "ceiling", "floor": "floor"}


def headline_basis(agg, mode):
    """주 판정 근거: 중간층과 천장·바닥층이 함께 있을 때만 중간층, 아니면 전체 평균.

    천장·바닥층이 없으면 중간층 평균 = 전체 평균이라 v1과 똑같이 동작한다.
    """
    if mode == "stratified" and "middle" in agg and ("ceiling" in agg or "floor" in agg):
        return "middle"
    return "overall"


def headline_decision(agg, threshold, budget, mode, guard_threshold=GUARD_THRESHOLD):
    """주 판정(헤드라인).

    mode='overall'   : v1과 같다 — 전체 평균 판정.
    mode='stratified': 근거는 headline_basis(). 근거가 중간층이면 천장·바닥 묶음을 가드로 본다 —
                       묶음 평균의 개악이나 개선이 guard_threshold(캘리브레이션 문턱보다 엄격)로 확정되면 반영한다.
                       개악이 하나라도 있으면 개악(가장 급한 신호), 아니면 개선이 하나라도 있으면 개선, 아니면 근거 층의 판정.
    반환: {"verdict", "basis", "probs", "primary_probs", "decided_by", "regressed_in", "improved_in", "primary_verdict"}
    """
    v_all = _stats.decide_from_probs(agg["all"][0], agg["all"][1], agg["all"][2], threshold, budget)
    if mode == "overall":
        return {"verdict": v_all, "basis": "overall", "probs": agg["all"], "primary_probs": agg["all"],
                "decided_by": "overall", "regressed_in": ["overall"] if v_all == "regressed" else [],
                "improved_in": ["overall"] if v_all == "improved" else [], "primary_verdict": v_all}
    basis = headline_basis(agg, mode)
    probs = agg[_AGG_KEY[basis]]
    v_p = _stats.decide_from_probs(probs[0], probs[1], probs[2], threshold, budget)
    regressed_in = [basis] if v_p == "regressed" else []
    improved_in = [basis] if v_p == "improved" else []
    if basis == "middle":
        for g in SIDE_STRATA:
            if g in agg:
                if agg[g][1] >= guard_threshold:
                    regressed_in.append(g)
                elif agg[g][0] >= guard_threshold:
                    improved_in.append(g)
    if regressed_in:
        verdict, decided_by = "regressed", regressed_in[0]
    elif improved_in:
        verdict, decided_by = "improved", improved_in[0]
    else:
        verdict, decided_by = v_p, basis
    return {"verdict": verdict, "basis": basis, "probs": agg[_AGG_KEY[decided_by]], "primary_probs": probs,
            "decided_by": decided_by, "regressed_in": regressed_in, "improved_in": improved_in,
            "primary_verdict": v_p}


def _complete_improve(cell, maxr):
    kb, nb, ka, na = cell
    rb, ra = max(0, maxr - nb), max(0, maxr - na)
    return (kb, nb + rb, ka + ra, na + ra)


def _complete_regress(cell, maxr):
    kb, nb, ka, na = cell
    rb, ra = max(0, maxr - nb), max(0, maxr - na)
    return (kb + rb, nb + rb, ka, na + ra)


def _complete_rope_candidates(cell, maxr):
    """no_difference 도달 가능성 검사용 후보 완성 3종 — P(|diff|<delta)의 상한을 노린다.

    Beta 사후분포는 통과율이 극단(0/1)에 가까울수록 좁아지므로, '통과 수 균등화 후
    나머지 전부 실패' 단일 완성은 P(동등)의 최댓값이 아니다(전부 통과 채움이 더 높을 수 있음).
    후보 {균등화+실패 채움, 전부 통과(양팔), 전부 실패(양팔)}의 최댓값을 상한으로 쓴다.
    """
    kb, nb, ka, na = cell
    rb, ra = max(0, maxr - nb), max(0, maxr - na)
    kb1, ka1 = kb, ka
    d0 = ka - kb
    if d0 > 0:
        kb1 += min(rb, d0)
    elif d0 < 0:
        ka1 += min(ra, -d0)
    return [
        (kb1, nb + rb, ka1, na + ra),          # 통과 수 균등화 + 나머지 전부 실패 (diff 를 0에 정렬)
        (kb + rb, nb + rb, ka + ra, na + ra),  # 전부 통과 — 극단 통과율로 사후분포를 좁힘
        (kb, nb + rb, ka, na + ra),            # 전부 실패 — 반대쪽 극단
    ]


def _futile(recs, delta, threshold, budget, maxr, seed_base, mode, guard_threshold=GUARD_THRESHOLD, side_active=False):
    """futility: 남은 예산을 가장 유리하게 다 써도 확정 판정에 닿을 수 없음이 상한 계산으로 확실한가.

    전체 평균의 improved·regressed·no_difference 상한(v1과 같은 계산·같은 시드)이 모두 닿지 않을 때만 참.
    주 판정 근거가 중간층일 때(side_active)는 추가로 현재 중간층의 세 상한과 천장·바닥층의 개선·개악 상한까지
    모두 닿지 않아야 참 — 근거가 전체 평균이면 v1과 똑같은 조건이다
    (층 소속은 현재 기준으로 고정한 근사 — futility 는 판정을 막을 뿐 거짓 양성을 만들지 않는다).
    """
    all_cells = [r["_eff"] for r in recs]
    groups = [r["stratum"] for r in recs] if mode == "stratified" else None
    agg_i = aggregate_probs([_complete_improve(c, maxr) for c in all_cells], groups, delta, seed_base + 1, FUT_SAMPLES)
    agg_r = aggregate_probs([_complete_regress(c, maxr) for c in all_cells], groups, delta, seed_base + 2, FUT_SAMPLES)
    po = {}
    cand_sets = [_complete_rope_candidates(c, maxr) for c in all_cells]
    for off, cand_cells in enumerate(zip(*cand_sets)):
        agg_o = aggregate_probs(list(cand_cells), groups, delta, seed_base + 3 + off, FUT_SAMPLES)
        for k, v in agg_o.items():
            po[k] = max(po.get(k, 0.0), v[2])
    reachable = (agg_i["all"][0] >= threshold or agg_r["all"][1] >= threshold
                 or po.get("all", 0.0) >= 1.0 - budget)
    if mode == "stratified" and side_active and not reachable:
        if "middle" in agg_i:
            reachable = (agg_i["middle"][0] >= threshold or agg_r["middle"][1] >= threshold
                         or po.get("middle", 0.0) >= 1.0 - budget)
        for g in SIDE_STRATA:
            if not reachable and g in agg_r:
                reachable = agg_r[g][1] >= guard_threshold or agg_i[g][0] >= guard_threshold
    return not reachable


def sequential_replay(runs, order, delta, threshold, budget, batch, maxr, ceiling, floor, seed, mode="stratified"):
    """입력을 도착 순서대로 재생하며 batch 단위마다 판정.

    체크포인트 = 모든 (시나리오, 팔) 칸의 최소 누적 실행 수가 batch 배수에 닿는 순간 (max_runs 까지만 —
    캘리브레이션이 본 훔쳐보기 횟수 밖은 보지 않는다).
    종료 3조건: 주 판정이 개선 확정 / 개악 확정(실패 시나리오 목록 동반) / futility.
    """
    data = {sid: {"before": [], "after": []} for sid in order}
    next_ck = batch
    ck_idx = 0
    stop = None
    for (sid, arm, passed) in runs:
        data[sid][arm].append(passed)
        if next_ck > maxr:
            continue
        cur_min = min(min(len(d["before"]), len(d["after"])) for d in data.values())
        if cur_min < next_ck:
            continue
        ck_idx += 1
        recs = []
        for s in order:
            rec, _ = judge_scenario(s, data[s]["before"], data[s]["after"], delta, threshold, budget, ceiling, floor)
            recs.append(rec)
        judged = [r for r in recs if r["_judged"]]
        cells = [r["_eff"] for r in judged]
        groups = [r["stratum"] for r in judged] if mode == "stratified" else None
        agg = aggregate_probs(cells, groups, delta, seed + 104729 * ck_idx, CKPT_SAMPLES)
        head = headline_decision(agg, threshold, budget, mode)
        v = head["verdict"]
        if v in ("improved", "regressed"):
            bad = [r["scenario_id"] for r in recs if r["verdict"] == "regressed"]
            stop = {"at": next_ck, "reason": v, "regressed_scenarios": bad, "basis": head["basis"],
                    "decided_by": head["decided_by"], "regressed_in": head["regressed_in"], "probs": head["probs"]}
            break
        if v == "undetermined" and maxr > next_ck:
            if _futile(recs, delta, threshold, budget, maxr, seed + 104729 * ck_idx, mode,
                       side_active=(head["basis"] == "middle")):
                stop = {"at": next_ck, "reason": "futility", "regressed_scenarios": [], "basis": head["basis"],
                        "decided_by": None, "regressed_in": [], "probs": head["probs"]}
                break
        next_ck += batch
    final_min = min(min(len(d["before"]), len(d["after"])) for d in data.values())
    if stop is None:
        stop = {"at": min(final_min, maxr), "reason": "max_runs", "regressed_scenarios": [], "basis": None,
                "decided_by": None, "regressed_in": [], "probs": None}
    # 절감은 '판정에 의한 조기 종료'가 있을 때만 의미가 있다. 예산 소진은 절감 0.
    saved = max(0, maxr - stop["at"]) if stop["reason"] != "max_runs" else 0
    out = {"stopped_at_per_arm": stop["at"], "stop_reason": stop["reason"],
           "runs_saved_per_arm": saved, "regressed_scenarios": stop["regressed_scenarios"],
           "stop_basis": stop["basis"], "stop_decided_by": stop["decided_by"],
           "regressed_in": stop["regressed_in"], "checkpoints": ck_idx}
    if stop["probs"] is not None:
        out["stop_probs"] = {"p_improved": round(stop["probs"][0], 6), "p_regressed": round(stop["probs"][1], 6),
                             "p_rope": round(stop["probs"][2], 6)}
    return out


def _split(runs, order):
    scen = {sid: {"before": [], "after": []} for sid in order}
    for sid, arm, passed in runs:
        scen[sid][arm].append(passed)
    return scen


def apply_expected(order, spec):
    """--expect-scenarios: 정수 N 또는 쉼표로 구분한 시나리오 ID 목록.

    반환 (order, note). ID 목록이면 아직 안 나온 시나리오를 실행 0회로 order 에 넣는다(판정이 그 칸을 기다린다).
    정수면 이름을 모르므로, 아직 N개가 안 나왔을 때 note 를 'WAIT:'로 시작해 온라인 판정을 미루게 한다.
    """
    if not spec:
        return order, None
    s = spec.strip()
    if s.isdigit():
        n = int(s)
        if len(order) < n:
            return order, "WAIT:지금까지 시나리오 %d/%d개만 나왔습니다 — 모든 시나리오가 나올 때까지 판정하지 않습니다" % (len(order), n)
        if len(order) > n:
            return order, "--expect-scenarios(%d개)보다 많은 시나리오(%d개)가 들어왔습니다 — 입력을 확인하세요" % (n, len(order))
        return order, None
    ids = []
    for x in s.split(","):
        x = x.strip()
        if x and x not in ids:
            ids.append(x)
    missing = [x for x in ids if x not in order]
    extra = [x for x in order if x not in ids]
    notes = []
    if missing:
        notes.append("아직 실행이 없는 시나리오 %d개를 0회로 잡았습니다: %s" % (len(missing), ", ".join(missing[:8])))
    if extra:
        notes.append("--expect-scenarios 에 없는 시나리오가 들어왔습니다: %s" % ", ".join(extra[:8]))
    return order + missing, (" / ".join(notes) or None)


def _probs_json(p):
    return {"p_improved": round(p[0], 6), "p_regressed": round(p[1], 6), "p_rope": round(p[2], 6)}


def _needed_for(cells, delta, budget):
    skb = sum(c[0] for c in cells); snb = sum(c[1] for c in cells)
    ska = sum(c[2] for c in cells); sna = sum(c[3] for c in cells)
    pb = (skb + 0.5) / (snb + 1.0)
    pa = (ska + 0.5) / (sna + 1.0)
    return _stats.needed_runs(pb, pa, delta, budget)


def _conflict(records, verdict):
    """시나리오별 방향과 (전체·주) 판정의 상충 — Simpson 경고용."""
    dirs = set(r["verdict"] for r in records if r["_judged"]) & {"improved", "regressed"}
    return (("improved" in dirs and "regressed" in dirs)
            or ("regressed" in dirs and verdict != "regressed")
            or ("improved" in dirs and verdict == "regressed"))


def analyze(runs, order, args, delta, delta_source):
    """전체 분석: 시나리오별 판정 + 층별·전체 판정 + 주 판정 + 순차 재생 + 예외 처리. 반환: 결과 JSON dict."""
    scen = _split(runs, order)
    threshold, calib, uncalibrated = load_calibration(args.calibration, args.batch_size,
                                                      args.max_runs, args.error_budget)
    mode = args.primary
    warnings = []
    cal_delta = calib.get("calibrated_at_delta")
    if cal_delta is not None and delta < cal_delta - 1e-9:
        warnings.append("사용자 delta(%.3f)가 캘리브레이션 기준 delta(%.3f)보다 작습니다 — "
                        "FPR<=%g 보장은 delta>=%.3f 에서만 성립하며 그 아래에서는 근사입니다. "
                        "calibrate.py --delta %.3f 재실행을 권장합니다."
                        % (delta, cal_delta, args.error_budget, cal_delta, delta))
    max_n = max(max(len(scen[s]["before"]), len(scen[s]["after"])) for s in order)
    if max_n > args.max_runs:
        warnings.append("입력(팔당 최대 %d회)이 max_runs(%d)보다 많습니다 — 순차 재생은 %d회까지만 봅니다"
                        "(캘리브레이션 밖의 훔쳐보기 방지). 최종 판정은 전체 데이터 한 번으로 냅니다."
                        % (max_n, args.max_runs, args.max_runs))
    records = []
    for sid in order:
        rec, warns = judge_scenario(sid, scen[sid]["before"], scen[sid]["after"], delta,
                                    threshold, args.error_budget, args.ceiling, args.floor)
        records.append(rec)
        warnings.extend(warns)

    judged = [r for r in records if r["_judged"]]
    judged_cells = [r["_eff"] for r in judged]
    agg = aggregate_probs(judged_cells, [r["stratum"] for r in judged], delta, args.seed, FINAL_SAMPLES)
    p_imp, p_reg, p_rope = agg["all"]
    overall_verdict = _stats.decide_from_probs(p_imp, p_reg, p_rope, threshold, args.error_budget)

    overall_flags = []
    if uncalibrated:
        overall_flags.append("uncalibrated_config")
        warnings.append("이 설정은 캘리브레이션이 없어 기본문턱 %.2f 로 판정했습니다 — 거짓 양성률이 보장되지 않습니다"
                        "(훔쳐보기가 많을수록 커짐). python3 calibrate.py --batch-size %d --max-runs %d --error-budget %g "
                        "실행을 권장합니다." % (UNCALIBRATED_THRESHOLD, args.batch_size, args.max_runs, args.error_budget))
    if len(order) == 1:
        overall_flags.append("single_scenario")
        warnings.append("시나리오가 하나뿐이라 층화가 불가능합니다 — 전체 판정만 냅니다")
    else:
        warnings.append("캘리브레이션은 단일 시나리오 A/A 기준입니다 — 여러 시나리오·층별 평균과 가드를 합친 v2 판정의 "
                        "거짓 양성률은 README 'v2 검증' B의 시뮬레이션 값을 보세요")

    total_pass = sum(sum(scen[s]["before"]) + sum(scen[s]["after"]) for s in order)
    total_runs = len(runs)
    if total_pass == total_runs or total_pass == 0:
        warnings.append("모든 실행이 %s했습니다 — 두 팔을 구분할 정보가 없어 판정 불가입니다"
                        % ("통과" if total_pass else "실패"))
    if not judged_cells:
        overall_flags.append("no_information")
        warnings.append("모든 시나리오가 정보 없음(양팔 전승/전패) 또는 판정 불가 — 전체 판정을 낼 수 없습니다")
    else:
        excluded = [r["scenario_id"] for r in records if not r["_judged"]]
        if excluded:
            warnings.append("양팔 전승/전패로 정보가 없는 시나리오를 합산에서 제외했습니다(목록에는 유지): %s"
                            % ", ".join(excluded))

    strata_set = set(r["stratum"] for r in records)
    if strata_set == {"ceiling"}:
        overall_flags.append("ceiling")
        base_all = sum(r["_eff"][0] for r in records) / max(1, sum(r["_eff"][1] for r in records))
        head_room = (1.0 - base_all) * 100
        warnings.append("모든 시나리오가 천장 층입니다 — 이미 대부분 통과해 최대 가능 개선폭이 약 %.1f%%p에 불과합니다" % head_room)
    if strata_set == {"floor"}:
        overall_flags.extend(["floor", "not_task_fixable"])

    if _conflict(records, overall_verdict):
        overall_flags.append("simpson_conflict")

    needed = None
    if overall_verdict == "undetermined" and judged_cells:
        overall_flags.append("needs_more_runs")
        needed = _needed_for(judged_cells, delta, args.error_budget)

    # ---- 층별 판정과 주 판정 ----
    strata_out = []
    for g in STRATA:
        if g in agg:
            pg = agg[g]
            strata_out.append({"stratum": g,
                               "scenarios": [r["scenario_id"] for r in judged if r["stratum"] == g],
                               "verdict": _stats.decide_from_probs(pg[0], pg[1], pg[2], threshold, args.error_budget),
                               "p_improved": round(pg[0], 6), "p_regressed": round(pg[1], 6),
                               "p_rope": round(pg[2], 6)})
    head = headline_decision(agg, threshold, args.error_budget, mode)
    head_flags = []
    basis_cells = [r["_eff"] for r in judged if head["basis"] == "overall" or r["stratum"] == head["basis"]]
    head_needed = None
    if head["verdict"] == "undetermined" and basis_cells:
        head_flags.append("needs_more_runs")
        head_needed = _needed_for(basis_cells, delta, args.error_budget)
    if not judged_cells:
        head_flags.append("no_information")
    def _ids(g):
        ids = [r["scenario_id"] for r in judged if r["stratum"] == g]
        return len(ids), ", ".join(ids[:8]) + (" 외 %d개" % (len(ids) - 8) if len(ids) > 8 else "")

    guards = [g for g in head["regressed_in"] if g != head["basis"]]
    if guards:
        head_flags.append("guard_regressed")
        for g in guards:
            n_ids, ids = _ids(g)
            warnings.append("%s(%d개: %s)의 평균이 개악으로 확정돼(가드 문턱 %.3f) 주 판정을 개악으로 올렸습니다 — "
                            "%s" % (_basis_ko(g), n_ids, ids, GUARD_THRESHOLD,
                                    "잘 되던 시나리오가 깨졌는지 확인하세요" if g == "ceiling"
                                    else "거의 안 되던 시나리오가 더 나빠졌습니다"))
    if head["verdict"] == "improved" and head["decided_by"] != head["basis"]:
        head_flags.append("guard_improved")
        n_ids, ids = _ids(head["decided_by"])
        warnings.append("%s(%d개: %s)의 평균이 개선으로 확정돼(가드 문턱 %.3f) 주 판정을 개선으로 올렸습니다 — "
                        "중간층 자체 판정은 %s입니다" % (_basis_ko(head["decided_by"]), n_ids, ids, GUARD_THRESHOLD,
                                                    head["primary_verdict"]))
    if (mode == "stratified" and head["basis"] == "middle" and head["primary_verdict"] in ("improved", "regressed")
            and overall_verdict in ("undetermined", "no_difference")):
        head_flags.append("overall_diluted")
        n_other = sum(1 for r in judged if r["stratum"] != "middle")
        warnings.append("전체 평균만 보면 %s지만 중간층은 %s입니다 — 변화가 없는 천장·바닥 쪽 시나리오 %d개가 "
                        "평균을 희석했습니다(전체 평균, 즉 v1 방식으로는 확정되지 않는 신호)"
                        % (overall_verdict, head["primary_verdict"], n_other))
    if mode == "stratified" and head["basis"] == "middle":
        for st in strata_out:
            g = st["stratum"]
            if g == "middle" or g in head["improved_in"] or g in head["regressed_in"]:
                continue
            if st["verdict"] in ("improved", "regressed"):
                head_flags.append("side_signal_below_guard")
                n_ids, ids = _ids(g)
                warnings.append("%s(%d개: %s)의 평균은 문턱 %.3f로는 %s이지만 가드 문턱 %.3f에는 못 미쳐 주 판정에 "
                                "반영하지 않았습니다 — 더 돌리면 확정될 수 있습니다"
                                % (_basis_ko(g), n_ids, ids, threshold, st["verdict"], GUARD_THRESHOLD))
    if _conflict(records, head["verdict"]):
        head_flags.append("simpson_conflict")
        bad = [r["scenario_id"] for r in records if r["_judged"] and r["verdict"] == "regressed"]
        good = [r["scenario_id"] for r in records if r["_judged"] and r["verdict"] == "improved"]
        warnings.append("주 판정(%s)과 시나리오별 판정이 어긋납니다(Simpson) — 개악: [%s]%s"
                        % (head["verdict"], ", ".join(bad) or "-",
                           (" / 개선: [%s]" % ", ".join(good)) if good else ""))
    if head["verdict"] == "undetermined" and head_needed is None and basis_cells:
        warnings.append("관측 차이가 delta 경계와 거의 같아 유한한 표본 계획이 나오지 않습니다 — "
                        "delta를 조정하거나 시나리오를 늘리는 편이 낫습니다")

    ns_list = sorted(min(len(scen[s]["before"]), len(scen[s]["after"])) for s in order)
    n_rep = ns_list[len(ns_list) // 2] if ns_list else 0
    if n_rep > 0 and judged_cells:
        base = sum(c[0] for c in judged_cells) / max(1, sum(c[1] for c in judged_cells))
        floor95 = _stats.aa_noise_floor(base, n_rep, sims=4000, seed=args.seed + 1)
        if delta < floor95 and head["verdict"] == "undetermined":
            need_d = _stats.needed_runs(base, base, delta, args.error_budget)
            warnings.append("요청한 delta(%.3f)가 현재 표본(팔당 %d회)의 노이즈 바닥(%.3f)보다 작아 판별 불가입니다"
                            " — 이 기저 통과율에서 delta %.3f 을 분해하려면 팔당 약 %s회가 필요합니다(정규근사 표본계획)"
                            % (delta, n_rep, floor95, delta, need_d if need_d else "산정 불가"))

    seq = sequential_replay(runs, order, delta, threshold, args.error_budget,
                            args.batch_size, args.max_runs, args.ceiling, args.floor, args.seed, mode)
    calls_saved = None
    if args.calls_per_run is not None:
        calls_saved = int(round(seq["runs_saved_per_arm"] * 2 * len(order) * args.calls_per_run))
    seq_out = dict(seq)
    seq_out["calls_saved"] = calls_saved
    if not seq_out["regressed_scenarios"]:
        seq_out.pop("regressed_scenarios")
    if seq["stop_reason"] == "regressed" and seq["regressed_scenarios"]:
        warnings.append("순차 재생: 개악 확정 시점의 실패 시나리오 목록: %s" % ", ".join(seq["regressed_scenarios"]))
    if seq["stop_reason"] == "futility":
        warnings.append("순차 재생: 남은 예산(max_runs %d)을 가장 유리하게 써도 delta=%.3f 판별이 불가능해 조기 중단했습니다"
                        % (args.max_runs, delta))

    scen_json = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    result = {
        "schema_version": SCHEMA_VERSION,
        "config": {"input": args.input, "delta": delta, "delta_source": delta_source, "primary": mode,
                   "ceiling": args.ceiling, "floor": args.floor, "batch_size": args.batch_size,
                   "max_runs": args.max_runs, "error_budget": args.error_budget,
                   "calls_per_run": args.calls_per_run, "seed": args.seed},
        "calibration": calib,
        "headline": dict({"verdict": head["verdict"], "basis": head["basis"], "decided_by": head["decided_by"],
                          "primary_verdict": head["primary_verdict"],
                          "primary_probs": _probs_json(head["primary_probs"]),
                          "regressed_in": head["regressed_in"], "improved_in": head["improved_in"],
                          "guard_threshold": GUARD_THRESHOLD, "flags": head_flags,
                          "needed_runs_per_arm": head_needed}, **_probs_json(head["probs"])),
        "strata": strata_out,
        "overall": {"verdict": overall_verdict, "p_improved": round(p_imp, 6),
                    "p_regressed": round(p_reg, 6), "p_rope": round(p_rope, 6),
                    "flags": overall_flags, "needed_runs_per_arm": needed},
        "scenarios": scen_json,
        "sequential": seq_out,
        "warnings": warnings,
    }
    return result


def _basis_ko(g):
    return STRATUM_KO[g] if g == "overall" else STRATUM_KO[g] + "층"


def _need_text(need):
    if need and need <= 20000:
        return "결론에는 팔당 약 %d회가 필요합니다 (관측 통과율 고정 가정의 정규근사 표본계획 — 판정에는 미사용)." % need
    if need:
        return ("정규근사 표본계획상 팔당 약 %d회 — 사실상 비현실적입니다. 관측 차이가 delta 경계 부근이라 "
                "합산으로는 판별이 어렵습니다. 시나리오별 필요 횟수와 플래그를 보세요." % need)
    return "관측 차이가 delta 경계에 붙어 있어 유한한 표본 계획이 나오지 않습니다. 시나리오별 필요 횟수와 플래그를 보세요."


def print_human(res):
    """사람이 읽는 한국어 요약 (stdout). 기계용 JSON 과 반드시 둘 다 낸다."""
    cfg = res["config"]
    cal = res["calibration"]
    ov = res["overall"]
    hd = res["headline"]
    print("=" * 72)
    print("개선·개악 판별 결과  (delta=%.3f [%s], batch=%d, max_runs=%d, error_budget=%g, seed=%d, primary=%s)"
          % (cfg["delta"], cfg["delta_source"], cfg["batch_size"], cfg["max_runs"],
             cfg["error_budget"], cfg["seed"], cfg["primary"]))
    if cal["achieved_fpr"] is not None:
        print("캘리브레이션: 문턱 %.3f | 시뮬 %d회 | 달성 FPR %.4f (CI95 %.4f~%.4f) | 출처: %s"
              % (cal["threshold"], cal["sims"], cal["achieved_fpr"],
                 cal["fpr_ci95"][0], cal["fpr_ci95"][1], cal["source"]))
    else:
        print("캘리브레이션: 문턱 %.3f | %s" % (cal["threshold"], cal["source"]))
    print("-" * 72)
    print("시나리오별 판정 (층은 양팔 합산 통과율로 결정 — 표본이 작으면 undetermined 가 정상):")
    for s in res["scenarios"]:
        nb, na = s["n_before"], s["n_after"]
        nr = s.get("needed_runs_per_arm")
        if nr and nr <= 20000:
            extra = " | 필요 팔당 약 %d회(근사)" % nr
        elif s["verdict"] == "undetermined":
            extra = " | 관측 차이가 delta 경계 부근 — 필요 횟수 비현실적"
        else:
            extra = ""
        print("  - %-16s [%s층] n=%d/%d  ->  %-13s (P개선 %.3f, P개악 %.3f, P동등 %.3f)%s%s"
              % (s["scenario_id"], STRATUM_KO[s["stratum"]],
                 nb, na, s["verdict"], s["p_improved"], s["p_regressed"], s["p_rope"],
                 (" | " + ",".join(s["flags"])) if s["flags"] else "", extra))
    if res["strata"] and len(res["scenarios"]) > 1:
        print("-" * 72)
        print("층별 판정 (같은 층 시나리오들의 평균 차이):")
        for st in res["strata"]:
            print("  - %s층 %d개: %-13s (P개선 %.3f, P개악 %.3f, P동등 %.3f)"
                  % (STRATUM_KO[st["stratum"]], len(st["scenarios"]), st["verdict"],
                     st["p_improved"], st["p_regressed"], st["p_rope"]))
    print("-" * 72)
    flag_txt = ("   플래그: " + ",".join(hd["flags"])) if hd["flags"] else ""
    if hd["decided_by"] != hd["basis"]:
        # 주 판정 근거(중간층)가 아니라 가드(다른 층)의 확정이 판정을 정했다
        pp = hd["primary_probs"]
        print("주 판정: %s   (근거: %s — 가드 문턱 %.3f로 확정)  P(개선)=%.3f  P(개악)=%.3f  P(동등)=%.3f%s"
              % (hd["verdict"].upper(), _basis_ko(hd["decided_by"]), GUARD_THRESHOLD, hd["p_improved"],
                 hd["p_regressed"], hd["p_rope"], flag_txt))
        print("  %s 자체 판정: %s  (P개선 %.3f, P개악 %.3f, P동등 %.3f)"
              % (_basis_ko(hd["basis"]), hd["primary_verdict"], pp["p_improved"], pp["p_regressed"], pp["p_rope"]))
    else:
        print("주 판정: %s   (근거: %s)  P(개선)=%.3f  P(개악)=%.3f  P(동등)=%.3f%s"
              % (hd["verdict"].upper(), _basis_ko(hd["basis"]), hd["p_improved"], hd["p_regressed"], hd["p_rope"],
                 flag_txt))
    if hd["verdict"] == "undetermined":
        if "no_information" in hd["flags"]:
            print("  -> 판정 불가 — 두 팔을 구분할 정보가 없습니다(양팔 전승/전패)."
                  " 실행을 늘리는 것이 아니라 변별력 있는 시나리오가 필요합니다.")
        else:
            print("  -> 아직 모릅니다. " + _need_text(hd["needed_runs_per_arm"]))
    if hd["basis"] != "overall" or hd["verdict"] != ov["verdict"]:
        print("전체 평균(v1 방식): %s   P(개선)=%.3f  P(개악)=%.3f  P(동등)=%.3f%s"
              % (ov["verdict"].upper(), ov["p_improved"], ov["p_regressed"], ov["p_rope"],
                 ("   플래그: " + ",".join(ov["flags"])) if ov["flags"] else ""))
    seq = res["sequential"]
    reason_ko = {"improved": "개선 확정", "regressed": "개악 확정", "futility": "무익(futility)",
                 "max_runs": "예산 소진(조기 종료 없음)"}[seq["stop_reason"]]
    if seq["stop_reason"] in ("improved", "regressed") and seq.get("stop_decided_by"):
        reason_ko += "(%s)" % _basis_ko(seq["stop_decided_by"])
    line = ("순차 재생: 팔당 %d회 시점 %s — max_runs %d회 대비 팔당 %d회 절감"
            % (seq["stopped_at_per_arm"], reason_ko, cfg["max_runs"], seq["runs_saved_per_arm"]))
    if seq["calls_saved"] is not None:
        line += " (약 %d LLM 호출 절감)" % seq["calls_saved"]
    else:
        line += " (calls_per_run 미지정: 실행 횟수 기준)"
    print(line)
    if seq["runs_saved_per_arm"] > 0:
        print("  (다 돌린 뒤의 재생 결과입니다 — QA 도중 실제로 멈추려면 배치마다 --online 으로 호출하세요)")
    if res["warnings"]:
        print("-" * 72)
        print("경고/고지:")
        for w in res["warnings"]:
            print("  ! " + w)
    print("=" * 72)


# ---------------------------- 온라인 모드 (멈춤/계속 신호) ----------------------------

def run_online(runs, order, args, delta, delta_source, expected_note=None):
    """QA 도중 배치마다 호출하는 모드. 지금까지의 실행으로 순차 판정을 재생해 멈춤/계속을 종료코드로 낸다.

    오프라인 순차 재생과 같은 함수·같은 시드를 쓰므로, 같은 데이터면 멈추는 시점이 정확히 같다 — 단 판정은
    지금까지 나온 시나리오만 안다. 시나리오를 하나씩 차례로 돌리는 파이프라인이면 --expect-scenarios 로
    전체 시나리오를 알려 줘야 한다(아직 안 나온 시나리오는 0회로 잡혀 판정이 시작되지 않는다).
    종료코드: 0 계속 / 10 개선 확정 / 11 개악 확정 / 12 무익 / 13 예산 소진 (2·3은 입력·설정 오류).
    """
    threshold, calib, uncalibrated = load_calibration(args.calibration, args.batch_size,
                                                      args.max_runs, args.error_budget)
    if expected_note and expected_note.startswith("WAIT:"):
        # --expect-scenarios N 인데 아직 N개가 다 나오지 않았다 — 판정하지 않고 계속
        note = expected_note[5:]
        print("CONTINUE — %s" % note)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"schema_version": SCHEMA_VERSION, "mode": "online", "action": "continue",
                       "reason": "waiting_for_scenarios", "exit_code": 0, "note": note,
                       "scenarios_seen": order}, f, ensure_ascii=False, indent=2)
            f.write("\n")
        sys.exit(ONLINE_EXIT["continue"])
    scen = _split(runs, order)
    counts = {(s, a): len(scen[s][a]) for s in order for a in ("before", "after")}
    cur_min = min(counts.values())
    max_count = max(counts.values())
    if max_count > args.max_runs and cur_min < args.max_runs:
        # 어떤 칸은 예산을 넘겼는데 어떤 칸은 다음 체크포인트에도 못 미친다 — 파이프라인이 그 칸을 안 돌리는 중
        stuck = sorted("%s/%s(%d회)" % (s, a, n) for (s, a), n in counts.items()
                       if n < (cur_min // args.batch_size + 1) * args.batch_size)
        sys.stderr.write("입력 오류: 다른 칸은 max_runs(%d)를 넘었는데 아래 칸은 뒤처져 있어 판정을 진행할 수 없습니다 — "
                         "파이프라인이 이 칸을 돌리는지 확인하세요: %s\n"
                         % (args.max_runs, ", ".join(stuck[:12]) + (" 외 %d칸" % (len(stuck) - 12) if len(stuck) > 12 else "")))
        sys.exit(2)
    seq = sequential_replay(runs, order, delta, threshold, args.error_budget, args.batch_size,
                            args.max_runs, args.ceiling, args.floor, args.seed, args.primary)
    reason = seq["stop_reason"]
    next_ck = (cur_min // args.batch_size + 1) * args.batch_size
    last_ck = min(cur_min, args.max_runs) // args.batch_size * args.batch_size
    if reason in ("improved", "regressed", "futility"):
        action, code = "stop", ONLINE_EXIT[reason]
    elif next_ck > args.max_runs:
        # 다음 판정 시점이 예산 밖 — 더 볼 체크포인트가 없다
        action, code = "stop", ONLINE_EXIT["max_runs"]
    else:
        action, code, reason = "continue", ONLINE_EXIT["continue"], None
    out = {"schema_version": SCHEMA_VERSION, "mode": "online", "action": action, "reason": reason,
           "exit_code": code, "runs_per_arm_min": cur_min,
           "config": {"delta": delta, "delta_source": delta_source, "primary": args.primary,
                      "batch_size": args.batch_size, "max_runs": args.max_runs,
                      "error_budget": args.error_budget, "seed": args.seed},
           "calibration": {"threshold": threshold, "source": calib["source"], "uncalibrated": uncalibrated},
           "sequential": seq}
    if action == "stop" and reason in ("improved", "regressed", "futility"):
        where = _basis_ko(seq["stop_decided_by"]) if seq.get("stop_decided_by") else ""
        msg = {"improved": "개선 확정", "regressed": "개악 확정", "futility": "무익(더 돌려도 결론 불가)"}[reason]
        pr = seq.get("stop_probs") or {}
        print("STOP — 팔당 %d회 시점 %s%s. 남은 QA는 돌리지 않아도 됩니다 (max_runs 대비 팔당 %d회 절감)"
              % (seq["stopped_at_per_arm"], msg, (" [%s]" % where) if where else "", seq["runs_saved_per_arm"]))
        if pr:
            print("  그 시점 확률: P(개선)=%.3f  P(개악)=%.3f  P(동등)=%.3f  (문턱 %.3f)"
                  % (pr["p_improved"], pr["p_regressed"], pr["p_rope"], threshold))
        if reason == "regressed" and seq.get("regressed_scenarios"):
            print("  개악 시나리오: %s" % ", ".join(seq["regressed_scenarios"]))
        print("  전체 보고서: --online 을 빼고 같은 명령으로 다시 실행하세요.")
    elif action == "stop":
        print("STOP — 마지막 판정(팔당 %d회)까지 조기 결론 없이 예산(max_runs %d)을 다 썼습니다. "
              "--online 을 빼고 실행해 최종 판정을 보세요." % (last_ck, args.max_runs))
    else:
        lag = [{"scenario_id": s, "arm": a, "runs": n, "more": next_ck - n}
               for (s, a), n in counts.items() if n < next_ck]
        out["next_checkpoint_per_arm"] = next_ck
        out["to_run"] = lag
        if seq["checkpoints"]:
            head = "아직 결론 없음(마지막 판정 팔당 %d회 시점)" % last_ck
        else:
            head = "아직 첫 판정 전(팔당 %d회부터 판정)" % args.batch_size
        print("CONTINUE — %s. 모든 시나리오·팔이 팔당 %d회가 되면 다시 부르세요." % (head, next_ck))
        if lag and len(lag) <= 8:
            print("  더 돌릴 것: " + ", ".join("%s/%s +%d" % (x["scenario_id"], x["arm"], x["more"]) for x in lag))
        elif lag:
            print("  더 돌릴 칸 %d개 (목록은 JSON의 to_run)" % len(lag))
    if expected_note:
        print("  ! " + expected_note)
        out["expected_scenarios_note"] = expected_note
    if uncalibrated:
        print("  ! 이 설정은 캘리브레이션이 없어 기본문턱 %.2f 로 판정 중입니다 — 거짓 양성률 보장 없음(calibrate.py 권장)"
              % UNCALIBRATED_THRESHOLD)
    cal_delta = calib.get("calibrated_at_delta")
    if cal_delta is not None and delta < cal_delta - 1e-9:
        print("  ! delta(%.3f)가 캘리브레이션 기준 delta(%.3f)보다 작아 거짓 양성률 보장이 근사입니다"
              " — calibrate.py --delta %.3f 재실행 권장" % (delta, cal_delta, delta))
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    sys.exit(code)


# ---------------------------- 계획 모드 ----------------------------

def run_plan(args):
    """돌리기 전에: 기저 통과율·기대 개선폭·시나리오 수로 설정별 검출률과 평균 멈춤 시점을 시뮬레이션한다.

    판정 규칙은 verdict 와 같은 prob_diff + 캘리브레이션 문턱. 주 판정은 중간층 평균이므로, 바뀔 시나리오 K개와
    그대로일 중간층 시나리오 M개를 함께 넣는다. 이들은 체크포인트마다 실행 수를 합친 단일 시나리오로 근사한다
    (기대 차이 = 개선폭 x K/(K+M) — M=0, K=1이면 판정 경로가 verdict 와 정확히 같다).
    권장은 캘리브레이션된 설정 중에서만 한다 — 캘리브레이션이 없는 설정은 거짓 양성률이 보장되지 않는다.
    """
    import calibrate  # 계획 모드에서만 필요

    errs = []
    if args.base is None or not (0.0 < args.base < 1.0):
        errs.append("--base (수정 전 예상 통과율, 0~1) 가 필요합니다")
    if args.target is None or not (0.0 < args.target < 1.0):
        errs.append("--target (수정 후 기대 통과율, 0~1) 이 필요합니다")
    if args.delta is None or not (0.0 < args.delta < 0.5):
        errs.append("--delta (무시해도 좋은 차이) 가 필요합니다 — 쓰는 쪽 도메인 값이라 기본값이 없습니다"
                    " (참고: python3 calibrate.py --noise-only)")
    if args.scenarios < 1:
        errs.append("--scenarios 는 1 이상이어야 합니다")
    if args.unchanged_middle < 0:
        errs.append("--unchanged-middle 은 0 이상이어야 합니다")
    if not (0.0 < args.power < 1.0):
        errs.append("--power 는 (0, 1) 사이여야 합니다")
    if not (0.0 < args.error_budget < 0.5):
        errs.append("--error-budget 은 (0, 0.5) 사이여야 합니다")
    if args.sims < 50:
        errs.append("--sims 는 50 이상이어야 합니다")
    if args.batch_size is not None and args.batch_size < 1:
        errs.append("--batch-size 는 1 이상이어야 합니다")
    if args.max_runs is not None and args.max_runs < 1:
        errs.append("--max-runs 는 1 이상이어야 합니다")
    if errs:
        for e in errs:
            sys.stderr.write("설정 오류: %s\n" % e)
        sys.exit(2)
    base, target, K, M, delta = args.base, args.target, args.scenarios, args.unchanged_middle, args.delta
    eff_target = base + (target - base) * K / float(K + M)
    configs = []
    for e in load_entries(args.calibration):
        c = e.get("config", {})
        if abs(c.get("error_budget", -1) - args.error_budget) < 1e-9:
            configs.append((c["batch_size"], c["max_runs"]))
    if args.batch_size is not None or args.max_runs is not None:
        mine = (args.batch_size if args.batch_size is not None else DEFAULT_BATCH,
                args.max_runs if args.max_runs is not None else DEFAULT_MAX_RUNS)
        if mine[0] > mine[1]:
            sys.stderr.write("설정 오류: batch=%d, max_runs=%d — batch 는 max_runs 이하여야 합니다\n" % mine)
            sys.exit(2)
        if mine not in configs:
            configs.append(mine)
    if not configs:
        configs.append((DEFAULT_BATCH, DEFAULT_MAX_RUNS))
    configs.sort(key=lambda bm: (bm[1], bm[0]))
    rows = []
    for i, (b, m) in enumerate(configs):
        thr, calib, unc = load_calibration(args.calibration, b, m, args.error_budget)
        sim = calibrate.plan_paths(b, m, base, eff_target, K + M, args.sims, args.seed + 7919 * (i + 1), delta, thr)
        row = {"batch_size": b, "max_runs": m, "threshold": thr, "calibrated": not unc,
               "recorded_fpr": calib["achieved_fpr"]}
        row.update(sim)
        rows.append(row)
    eligible = [r for r in rows if r["calibrated"] and r["power"] >= args.power]
    rec = min(eligible, key=lambda r: (r["expected_runs_per_arm"], r["max_runs"])) if eligible else None
    notes = []
    lo, hi = min(base, target), max(base, target)
    if lo >= args.ceiling or hi <= args.floor:
        g = "ceiling" if lo >= args.ceiling else "floor"
        notes.append("수정 전후 통과율이 둘 다 %s층 범위라, 실제 판정에서는 이 시나리오들이 중간층이 아니라 %s층으로 분류됩니다"
                     "(중간층이 따로 있으면 주 판정에 들어가지 않습니다)." % (STRATUM_KO[g], STRATUM_KO[g]))
    if abs(target - base) <= delta:
        notes.append("기대 개선폭 %.1f%%p가 delta(%.1f%%p) 이하입니다 — 이 도구는 delta 밖의 차이만 개선으로 확정합니다."
                     % (abs(target - base) * 100, delta * 100))
    elif abs(eff_target - base) <= delta:
        notes.append("그대로일 중간층 시나리오 %d개와 평균하면 기대 차이가 %.1f%%p로 delta(%.1f%%p) 이하가 됩니다 — "
                     "중간층 평균으로는 확정할 수 없습니다." % (M, abs(eff_target - base) * 100, delta * 100))
    if any(not r["calibrated"] for r in rows):
        notes.append("캘리브레이션이 없는 설정은 기본문턱 0.99로 시뮬레이션한 참고값이며 권장 대상이 아닙니다 — 그 설정의 "
                     "거짓 양성률은 보장되지 않습니다(쓰려면 calibrate.py 로 먼저 캘리브레이션).")
    approx = _stats.needed_runs(base, eff_target, delta, args.error_budget, power=args.power)
    out = {"schema_version": SCHEMA_VERSION, "mode": "plan",
           "inputs": {"base": base, "target": target, "scenarios": K, "unchanged_middle": M,
                      "effective_target": round(eff_target, 6), "delta": delta,
                      "error_budget": args.error_budget, "target_power": args.power, "sims": args.sims,
                      "seed": args.seed},
           "configs": rows,
           "recommended": ({"batch_size": rec["batch_size"], "max_runs": rec["max_runs"]} if rec else None),
           "normal_approx_runs_per_arm": (int(math.ceil(approx / float(K + M))) if approx else None),
           "notes": notes,
           "method": "바뀔 시나리오 K개와 그대로일 중간층 M개를 체크포인트마다 실행 수를 합친 단일 시나리오로 근사한"
                     " 순차 판정 시뮬레이션 — futility 미적용(평균 실행 수는 상한 쪽)"}
    print("=" * 72)
    print("계획 모드: 수정 전 %.0f%% -> 수정 후 %.0f%% 기대(%s 검출), 바뀔 시나리오 %d개 + 그대로일 중간층 %d개, "
          "delta %.3f, error_budget %g, 목표 검출률 %.0f%%"
          % (base * 100, target * 100, "개악" if target < base else "개선", K, M, delta, args.error_budget,
             args.power * 100))
    print("-" * 72)
    print("  %-22s %-8s %-10s %-16s %-14s %s" % ("설정", "검출률", "평균 멈춤", "평균 실행(팔당)", "거짓양성(기록)", "문턱"))
    for r in rows:
        print("  %-22s %-8s %-10s %-16s %-14s %.3f%s"
              % ("batch %d / max %d" % (r["batch_size"], r["max_runs"]),
                 "%.0f%%" % (r["power"] * 100),
                 ("%d회" % round(r["mean_stop_when_detected"])) if r["mean_stop_when_detected"] else "-",
                 "%.0f회 (%.0f%% 절감)" % (r["expected_runs_per_arm"],
                                         100.0 * (1 - r["expected_runs_per_arm"] / float(r["max_runs"]))),
                 ("%.3f" % r["recorded_fpr"]) if r["recorded_fpr"] is not None else "보장 없음",
                 r["threshold"], "" if r["calibrated"] else " (미캘리브레이션 — 참고용)"))
    print("-" * 72)
    if rec:
        print("권장: batch %d / max_runs %d — 캘리브레이션된 설정 중 목표 검출률 %.0f%% 이상이면서 평균 실행 수가 가장 적음"
              % (rec["batch_size"], rec["max_runs"], args.power * 100))
        print("  실행 예: python3 verdict.py runs.jsonl --delta %g --batch-size %d --max-runs %d --online"
              % (delta, rec["batch_size"], rec["max_runs"]))
    else:
        cal_rows = [r for r in rows if r["calibrated"]]
        if cal_rows:
            best = max(cal_rows, key=lambda r: r["power"])
            print("캘리브레이션된 설정 중 목표 검출률 %.0f%%에 닿는 것이 없습니다(최고 %.0f%%: batch %d / max %d)."
                  % (args.power * 100, best["power"] * 100, best["batch_size"], best["max_runs"]))
        else:
            print("캘리브레이션된 설정이 없어 권장할 수 없습니다.")
        print("  -> 바뀔 시나리오를 늘리거나, max_runs를 키운 설정을 calibrate.py로 캘리브레이션한 뒤 다시 계획하세요.")
    if out["normal_approx_runs_per_arm"]:
        print("참고: 정규근사 표본계획으로는 중간층 시나리오 %d개 기준 팔당 약 %d회 (고정 표본 근사 — 순차 판정과 다름)"
              % (K + M, out["normal_approx_runs_per_arm"]))
    for n in notes:
        print("  ! " + n)
    print("  (시뮬 %d회/설정, %s)" % (args.sims, out["method"]))
    print("=" * 72)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("기계용 JSON 저장: %s" % args.json_out)


# ---------------------------- selftest ----------------------------

def _run(cmd, stdin_text=None):
    p = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _check_dataset_case(meta, exp, res, problems):
    ov = res["overall"]
    hd = res.get("headline", {})
    allflags = set(ov["flags"]) | set(hd.get("flags", []))
    for s in res["scenarios"]:
        allflags |= set(s["flags"])
    if exp.get("verdict_in") and ov["verdict"] not in exp["verdict_in"]:
        problems.append("overall verdict %r ∉ %r" % (ov["verdict"], exp["verdict_in"]))
    if exp.get("headline_in") and hd.get("verdict") not in exp["headline_in"]:
        problems.append("headline verdict %r ∉ %r" % (hd.get("verdict"), exp["headline_in"]))
    if exp.get("headline_basis") and hd.get("basis") != exp["headline_basis"]:
        problems.append("headline basis %r 기대, 실제 %r" % (exp["headline_basis"], hd.get("basis")))
    missing = set(exp.get("flags_required", [])) - allflags
    if missing:
        problems.append("필수 플래그 누락: %s" % ",".join(sorted(missing)))
    for sid, want in (exp.get("scenario_verdicts") or {}).items():
        actual = {s["scenario_id"]: s["verdict"] for s in res["scenarios"]}.get(sid)
        if actual != want:
            problems.append("시나리오 %s: %r 기대, 실제 %r" % (sid, want, actual))
    for sid, want in (exp.get("scenario_strata") or {}).items():
        actual = {s["scenario_id"]: s["stratum"] for s in res["scenarios"]}.get(sid)
        if actual != want:
            problems.append("시나리오 %s 층: %r 기대, 실제 %r" % (sid, want, actual))
    for sid, banned in (exp.get("scenario_flags_absent") or {}).items():
        actual = {s["scenario_id"]: set(s["flags"]) for s in res["scenarios"]}.get(sid, set())
        wrong = sorted(set(banned) & actual)
        if wrong:
            problems.append("시나리오 %s 에 붙으면 안 되는 플래그: %s" % (sid, ",".join(wrong)))
    es = exp.get("early_stop")
    seq = res["sequential"]
    if es is True and (seq["stop_reason"] not in ("improved", "regressed") or seq["runs_saved_per_arm"] <= 0):
        problems.append("조기 종료 기대: 실제 stop_reason=%s, 절감 %d" % (seq["stop_reason"], seq["runs_saved_per_arm"]))
    # early_stop=False 는 '결론에 의한 조기 종료 없음'으로 해석(futility 는 허용)
    if es is False and seq["stop_reason"] in ("improved", "regressed"):
        problems.append("조기 종료 없음 기대: 실제 %s" % seq["stop_reason"])
    if exp.get("stop_reason") and seq["stop_reason"] != exp["stop_reason"]:
        problems.append("stop_reason %r 기대, 실제 %r" % (exp["stop_reason"], seq["stop_reason"]))


def _prefix_lines(lines, per_arm, n_scen):
    """라운드로빈(회차마다 시나리오별 before/after) 순서의 합성 데이터에서 팔당 per_arm 회까지의 앞부분."""
    return "\n".join(lines[:per_arm * 2 * n_scen]) + "\n"


def run_selftest(args):
    """synth.py 케이스 생성 -> 판정 -> 기대 대비 표. 하나라도 어긋나면 종료코드 1."""
    here = os.path.dirname(os.path.abspath(__file__))
    synth = os.path.join(here, "synth.py")
    calibrate = os.path.join(here, "calibrate.py")
    me = os.path.abspath(__file__)
    if not os.path.exists(synth):
        sys.stderr.write("selftest 오류: synth.py 가 없습니다(같은 폴더 필요)\n")
        sys.exit(1)
    rc, out, err = _run([sys.executable, synth, "--list"])
    if rc != 0:
        sys.stderr.write("selftest 오류: synth.py --list 실패\n" + err)
        sys.exit(1)
    metas = json.loads(out)
    tmp = tempfile.TemporaryDirectory(prefix="verdict-selftest-")
    tmpdir = tmp.name
    seed = str(args.seed)
    rows = []
    all_ok = True
    for meta in metas:
        case = meta["case"]
        exp = meta["expected"]
        kind = meta.get("kind")
        t0 = time.time()
        problems = []
        got = ""
        common = ["--delta", str(meta["delta"]), "--batch-size", str(meta["batch_size"]),
                  "--max-runs", str(meta["max_runs"]), "--error-budget", str(meta["error_budget"]),
                  "--calibration", args.calibration, "--seed", seed]
        if kind == "calibration":
            rc, out, err = _run([sys.executable, calibrate, "--verify",
                                 "--batch-size", str(meta["batch_size"]),
                                 "--max-runs", str(meta["max_runs"]),
                                 "--error-budget", str(meta["error_budget"]),
                                 "--sims", "300", "--seed", seed])
            got = "verify rc=%d" % rc
            if rc != 0:
                problems.append("캘리브레이션 검증 실패(경계가 error_budget 미준수)")
                if err.strip():
                    problems.append(err.strip().splitlines()[-1])
        elif kind == "plan":
            p = meta["plan"]
            jpath = os.path.join(tmpdir, "case%d.json" % case)
            rc, out, err = _run([sys.executable, me, "--plan", "--base", str(p["base"]), "--target", str(p["target"]),
                                 "--scenarios", str(p["scenarios"]), "--delta", str(meta["delta"]),
                                 "--error-budget", str(meta["error_budget"]), "--sims", str(p["sims"]),
                                 "--power", str(p["power"]), "--calibration", args.calibration,
                                 "--seed", seed, "--json-out", jpath])
            if rc != 0:
                problems.append("plan 종료코드 %d: %s" % (rc, err.strip()[:200]))
            else:
                with open(jpath, "r", encoding="utf-8") as f:
                    res = json.load(f)
                rec = res.get("recommended")
                got = "권장 %s" % ("%d/%d" % (rec["batch_size"], rec["max_runs"]) if rec else "없음")
                recorded = {(e["config"]["batch_size"], e["config"]["max_runs"]): e.get("power_62_75")
                            for e in load_entries(args.calibration)
                            if abs(e["config"].get("error_budget", -1) - meta["error_budget"]) < 1e-9}
                for r in res["configs"]:
                    want = recorded.get((r["batch_size"], r["max_runs"]))
                    if want is not None and abs(r["power"] - want) > exp["power_tolerance"]:
                        problems.append("batch %d/max %d 검출률 %.3f — 캘리브레이션 기록 %.3f 와 %.2f 넘게 차이"
                                        % (r["batch_size"], r["max_runs"], r["power"], want, exp["power_tolerance"]))
                want_rec = exp.get("recommended")
                if want_rec and (not rec or [rec["batch_size"], rec["max_runs"]] != want_rec):
                    problems.append("권장 설정 %r 기대, 실제 %r" % (want_rec, rec))
        else:
            rc, data, err = _run([sys.executable, synth, "--case", str(case)])
            if rc != 0:
                problems.append("synth --case %d 실패" % case)
                data = ""
            jpath = os.path.join(tmpdir, "case%d.json" % case)
            cmd = [sys.executable, me, "-"] + common + ["--yes", "--json-out", jpath]
            rc, out, err = _run(cmd, stdin_text=data)
            if exp.get("error"):
                got = "rc=%d" % rc
                if rc != 2:
                    problems.append("종료코드 2 기대, 실제 %d" % rc)
                line_no, field = str(exp.get("error_line", 7)), exp.get("error_field", "arm")
                if line_no not in err or field not in err:
                    problems.append("오류 메시지에 줄 번호(%s)·필드명(%s)이 없음: %r" % (line_no, field, err.strip()))
            elif rc != 0:
                problems.append("종료코드 0 기대, 실제 %d: %s" % (rc, err.strip()[:200]))
            else:
                with open(jpath, "r", encoding="utf-8") as f:
                    res = json.load(f)
                got = res["headline"]["verdict"]
                if res["headline"]["verdict"] != res["overall"]["verdict"]:
                    got += " (전체 %s)" % res["overall"]["verdict"]
                _check_dataset_case(meta, exp, res, problems)
                if kind == "online" and not problems:
                    # 온라인 일치 검증: 멈춘 시점 직전 앞부분은 '계속'(0), 멈춘 시점 앞부분은 같은 멈춤 코드
                    seq = res["sequential"]
                    stop_at, reason = seq["stopped_at_per_arm"], seq["stop_reason"]
                    lines = [ln for ln in data.splitlines() if ln.strip()]
                    n_scen = len(res["scenarios"])
                    want_code = ONLINE_EXIT.get(reason)
                    checks = [(stop_at - meta["batch_size"], ONLINE_EXIT["continue"]), (stop_at, want_code)]
                    codes = []
                    for per_arm, want in checks:
                        if per_arm < meta["batch_size"]:
                            continue
                        rc2, out2, err2 = _run([sys.executable, me, "-"] + common +
                                               ["--online", "--json-out", os.path.join(tmpdir, "online.json")],
                                               stdin_text=_prefix_lines(lines, per_arm, n_scen))
                        codes.append("%d회:%d" % (per_arm, rc2))
                        if rc2 != want:
                            problems.append("온라인 팔당 %d회: 종료코드 %s 기대, 실제 %d (%s)"
                                            % (per_arm, want, rc2, (err2 or out2).strip()[:120]))
                    # 시나리오를 차례로 돌리는 파이프라인: 마지막 시나리오가 아직 안 나왔으면 멈추지 말고 기다려야 한다
                    ids = [s["scenario_id"] for s in res["scenarios"]]
                    partial = "\n".join(ln for ln in lines if json.loads(ln)["scenario_id"] != ids[-1]) + "\n"
                    for spec in (str(len(ids)), ",".join(ids)):
                        rc3, out3, err3 = _run([sys.executable, me, "-"] + common +
                                               ["--online", "--expect-scenarios", spec,
                                                "--json-out", os.path.join(tmpdir, "online.json")], stdin_text=partial)
                        if rc3 != ONLINE_EXIT["continue"]:
                            problems.append("--expect-scenarios %s: 빠진 시나리오가 있으면 계속(0)이어야 하는데 %d"
                                            % ("N" if spec.isdigit() else "목록", rc3))
                    codes.append("빠진 시나리오 대기:%s" % ("ok" if not problems else "x"))
                    got += " | 온라인 " + " ".join(codes)
        ok = not problems
        all_ok &= ok
        rows.append((case, meta["name"], got, "PASS" if ok else "FAIL", "; ".join(problems),
                     time.time() - t0))
    tmp.cleanup()
    print("selftest 결과 (synth.py %d케이스, MC seed=%d — 합성 데이터는 고정):" % (len(metas), args.seed))
    print("  %-4s %-20s %-34s %-5s %s" % ("case", "name", "결과", "판정", "문제"))
    for case, name, got, ok, prob, el in rows:
        print("  %-4d %-20s %-34s %-5s %s(%.1fs)" % (case, name, got, ok, (prob + " ") if prob else "", el))
    if not all_ok:
        print("selftest 실패 — 위 문제를 확인하세요")
        sys.exit(1)
    print("selftest 전부 통과")
    sys.exit(0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="개선·개악 판별기 — before/after 통과 결과(JSONL)의 상대 비교 판정",
        epilog="입력 한 줄 예: {\"scenario_id\": \"checkout\", \"arm\": \"before\", \"passed\": true}")
    ap.add_argument("input", nargs="?", default="-", help="JSONL 파일 경로 ('-'=stdin, 기본)")
    ap.add_argument("--delta", type=float, default=None, help="무시해도 좋은 차이(ROPE 반폭). 미지정 시 데이터에서 제안")
    ap.add_argument("--primary", choices=("stratified", "overall"), default="stratified",
                    help="주 판정 기준: stratified(기본 — 천장·바닥층이 있으면 중간층 기준 + 가드) | overall(v1과 같은 전체 평균)")
    ap.add_argument("--ceiling", type=float, default=0.90,
                    help="천장 층 경계 — 수정 전후 두 팔 모두 이 통과율 이상이면 천장층 (기본 0.90)")
    ap.add_argument("--floor", type=float, default=0.10,
                    help="바닥 층 경계 — 수정 전후 두 팔 모두 이 통과율 이하면 바닥층 (기본 0.10)")
    ap.add_argument("--expect-scenarios", default=None,
                    help="전체 시나리오 수(정수) 또는 ID 목록(쉼표 구분) — 시나리오를 하나씩 차례로 돌리는 파이프라인에서 "
                         "--online 이 아직 안 나온 시나리오를 기다리게 함")
    ap.add_argument("--batch-size", type=int, default=None, help="조기 종료 판정 주기 (기본 %d)" % DEFAULT_BATCH)
    ap.add_argument("--max-runs", type=int, default=None, help="팔당 최대 실행 수 예산 (기본 %d)" % DEFAULT_MAX_RUNS)
    ap.add_argument("--error-budget", type=float, default=0.05, help="허용 거짓 양성률 (기본 0.05)")
    ap.add_argument("--calls-per-run", type=float, default=None, help="실행 1회당 평균 LLM 호출 수(절감 환산용)")
    ap.add_argument("--calibration", default=os.path.join(here, "calibration-default.json"),
                    help="캘리브레이션 파일 (기본: 스크립트 폴더의 calibration-default.json)")
    ap.add_argument("--json-out", default="verdict.json", help="기계용 JSON 출력 경로 (기본 verdict.json)")
    ap.add_argument("--yes", action="store_true", help="delta 제안값을 확인 없이 수락")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="MC 시드 (기본 20260710, 결정론 보장)")
    ap.add_argument("--online", action="store_true",
                    help="QA 도중 배치마다 호출: 멈춤/계속을 종료코드로 (0 계속, 10 개선, 11 개악, 12 무익, 13 예산 소진)")
    ap.add_argument("--plan", action="store_true", help="돌리기 전 계획: 설정별 검출률·평균 멈춤 시점 시뮬레이션")
    ap.add_argument("--base", type=float, default=None, help="[--plan] 수정 전 예상 통과율")
    ap.add_argument("--target", type=float, default=None, help="[--plan] 수정 후 기대 통과율")
    ap.add_argument("--scenarios", type=int, default=1, help="[--plan] 바뀔 것으로 보는 (중간층) 시나리오 수 (기본 1)")
    ap.add_argument("--unchanged-middle", type=int, default=0,
                    help="[--plan] 그대로일 것으로 보는 중간층 시나리오 수 — 주 판정 평균에 함께 들어간다 (기본 0)")
    ap.add_argument("--power", type=float, default=0.8, help="[--plan] 목표 검출률 (기본 0.8)")
    ap.add_argument("--sims", type=int, default=400, help="[--plan] 설정당 시뮬레이션 횟수 (기본 400)")
    ap.add_argument("--selftest", action="store_true", help="synth.py 케이스 자기검증")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args)
        return
    if args.plan:
        run_plan(args)
        return
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH
    if args.max_runs is None:
        args.max_runs = DEFAULT_MAX_RUNS
    if args.batch_size < 1 or args.max_runs < 1:
        sys.stderr.write("설정 오류: --batch-size/--max-runs 는 1 이상이어야 합니다\n")
        sys.exit(2)
    if args.batch_size > args.max_runs:
        sys.stderr.write("설정 오류: --batch-size(%d)가 --max-runs(%d)보다 크면 순차 판정 시점이 하나도 없습니다\n"
                         % (args.batch_size, args.max_runs))
        sys.exit(2)
    if not (0.0 < args.error_budget < 0.5):
        sys.stderr.write("설정 오류: --error-budget 은 (0, 0.5) 사이여야 합니다\n")
        sys.exit(2)
    if not (0.0 < args.floor < args.ceiling < 1.0):
        sys.stderr.write("설정 오류: 0 < --floor < --ceiling < 1 이어야 합니다\n")
        sys.exit(2)

    if args.input == "-":
        text = sys.stdin.read()
    else:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            sys.stderr.write("입력 오류: 파일을 열 수 없습니다: %s\n" % e)
            sys.exit(2)

    runs, order = parse_input(text)
    order, expected_note = apply_expected(order, args.expect_scenarios)
    delta, delta_source = resolve_delta(args, _split(runs, order), order)
    if args.online:
        run_online(runs, order, args, delta, delta_source, expected_note)
        return
    res = analyze(runs, order, args, delta, delta_source)
    if expected_note:
        res["warnings"].append(expected_note[5:] if expected_note.startswith("WAIT:") else expected_note)
    print_human(res)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("기계용 JSON 저장: %s" % args.json_out)


if __name__ == "__main__":
    main()
