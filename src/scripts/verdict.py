#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""개선·개악 판별기 CLI — before/after 통과 결과(JSONL)를 받아 improved/regressed/no_difference/undetermined 를 판정.

사용 예시: python3 src/scripts/synth.py --case 2 | python3 src/scripts/verdict.py - --delta 0.03 --batch-size 20 --max-runs 400 --yes

- 판정 수식은 전부 _stats.decide()/prob_diff() (calibrate.py 와 동일 경로 — SPEC §8 캘리브레이션 전제)
- 종료코드: 0 정상 / 1 selftest 실패 / 2 입력·설정 오류 / 3 delta 미확정 중단
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
UNCALIBRATED_THRESHOLD = 0.99  # 해당 설정의 캘리브레이션이 없을 때의 보수 기본문턱
FINAL_SAMPLES = 20000          # 전체(층화) 판정 MC 표본 수 — 시나리오당
CKPT_SAMPLES = 10000           # 순차 재생 체크포인트용 (문턱 대비 MC 오차 ~0.001)
FUT_SAMPLES = 4000             # futility 상한 검사용 (경계 비교라 정밀도 요구 낮음)
VERDICTS = ("improved", "regressed", "no_difference", "undetermined")


def _fail_input(line_no, field, msg):
    """입력 스키마 위반: 어느 줄의 어느 필드인지 정확히 지목하고 종료코드 2 (SPEC §9)."""
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
    """delta 미지정 시: 관측 기저 통과율·현재 n으로 A/A 노이즈 바닥을 계산해 제안 + 진행 확인 (SPEC §5·§9).

    기본값을 임의로 넣지 않는다(SPEC §11-3). 결정은 사람이 한다.
    """
    if args.delta is not None:
        if not (0.0 < args.delta < 0.5):
            sys.stderr.write("설정 오류: --delta 는 (0, 0.5) 사이여야 합니다 (현재 %r)\n" % args.delta)
            sys.exit(2)
        return args.delta, "user"
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


def load_calibration(path, batch, maxr, budget):
    """(batch_size, max_runs, error_budget) 매칭 엔트리 로드. 없으면 보수 기본문턱 + 정직 고지 (SPEC §11-5)."""
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", []) if isinstance(data, dict) else data
    except (OSError, ValueError):
        pass
    for e in entries:
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
                  "보수 기본문턱 %.2f 사용. calibrate.py 실행 권장." % (batch, maxr, budget, UNCALIBRATED_THRESHOLD),
    }
    return UNCALIBRATED_THRESHOLD, calib, True


def judge_scenario(sid, blist, alist, delta, threshold, budget, ceiling, floor):
    """시나리오 하나의 판정. 층은 before 팔 관측 통과율로만 정한다(SPEC §7 — LLM 분류 금지).

    팔 불균형이면 적은 쪽 n에 맞춰 도착 순서 앞쪽만 쓰는 보수 판정(_eff)을 하고 imbalanced_arms 를 단다.
    """
    nb, na = len(blist), len(alist)
    flags = []
    warns = []
    m = min(nb, na)
    if m == 0:
        rec = {"scenario_id": sid, "stratum": "middle", "n_before": nb, "n_after": na,
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
    rate_b = kb / nbe
    if rate_b >= ceiling:
        stratum = "ceiling"
        flags.append("ceiling")
    elif rate_b <= floor:
        stratum = "floor"
        flags.extend(["floor", "not_task_fixable"])
        warns.append("시나리오 '%s': 바닥 층(기저 %.0f%%) — 수정 전후 모두 거의 실패. 태스크 수정으로 풀 문제가 아닙니다" % (sid, rate_b * 100))
    else:
        stratum = "middle"
    no_info = (kb == nbe and ka == nae) or (kb == 0 and ka == 0)
    verdict, p_imp, p_reg, p_rope = _stats.decide(kb, nbe, ka, nae, delta, threshold, budget)
    if no_info:
        flags.append("no_information")
    rec = {"scenario_id": sid, "stratum": stratum, "n_before": nb, "n_after": na,
           "verdict": verdict, "p_improved": round(p_imp, 6), "p_regressed": round(p_reg, 6),
           "p_rope": round(p_rope, 6), "flags": flags,
           "_eff": (kb, nbe, ka, nae), "_judged": not no_info}
    if verdict == "undetermined":
        flags.append("needs_more_runs")
        pb = (kb + 0.5) / (nbe + 1.0)
        pa = (ka + 0.5) / (nae + 1.0)
        rec["needed_runs_per_arm"] = _stats.needed_runs(pb, pa, delta, budget)
    return rec, warns


def overall_probs(cells, delta, seed, nsamples):
    """층화 전체 판정의 확률 3종. 시나리오 1개면 결정론적 prob_diff, 여럿이면 사후분포 시드 고정 샘플링.

    diff 사후분포 = 시나리오별 (p_after - p_before) 표본의 단순 평균(층화 평균).
    """
    if not cells:
        cells = [(0, 0, 0, 0)]  # 정보 없음 -> 사전분포끼리의 비교(불확실성이 그대로 드러남)
    if len(cells) == 1:
        kb, nb, ka, na = cells[0]
        return _stats.prob_diff(kb, nb, ka, na, delta)
    rng = random.Random(seed)
    bv = rng.betavariate
    params = [(kb + 0.5, nb - kb + 0.5, ka + 0.5, na - ka + 0.5) for (kb, nb, ka, na) in cells]
    inv = 1.0 / len(cells)
    ci = cr = 0
    for _ in range(nsamples):
        s = 0.0
        for ab, bb, aa, ba in params:
            s += bv(aa, ba) - bv(ab, bb)
        d = s * inv
        if d > delta:
            ci += 1
        elif d < -delta:
            cr += 1
    p_imp = ci / float(nsamples)
    p_reg = cr / float(nsamples)
    return p_imp, p_reg, max(0.0, 1.0 - p_imp - p_reg)


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


def sequential_replay(runs, order, delta, threshold, budget, batch, maxr, ceiling, floor, seed):
    """입력을 도착 순서대로 재생하며 batch 단위마다 판정 (SPEC §8).

    체크포인트 = 모든 (시나리오, 팔) 칸의 최소 누적 실행 수가 batch 배수에 닿는 순간.
    종료 3조건: 개선 확정 / 개악 확정(실패 시나리오 목록 동반) / futility.
    futility = 남은 예산을 가장 유리하게 다 써도 improved·regressed·no_difference 어느 것도
    확정 불가. improved/regressed 상한은 최선 완성(한쪽 전부 통과·반대쪽 전부 실패) 하나면 되지만,
    no_difference 는 단일 완성이 상한이 아니어서 후보 완성들(_complete_rope_candidates)의
    최댓값으로 상한을 잡는다 — 상한 계산이므로 '확실할 때만' 멈춘다.
    """
    data = {sid: {"before": [], "after": []} for sid in order}
    next_ck = batch
    ck_idx = 0
    stop = None
    for (sid, arm, passed) in runs:
        data[sid][arm].append(passed)
        cur_min = min(min(len(d["before"]), len(d["after"])) for d in data.values())
        if cur_min < next_ck:
            continue
        ck_idx += 1
        recs = []
        for s in order:
            rec, _ = judge_scenario(s, data[s]["before"], data[s]["after"], delta, threshold, budget, ceiling, floor)
            recs.append(rec)
        cells = [r["_eff"] for r in recs if r["_judged"]]
        p_imp, p_reg, p_rope = overall_probs(cells, delta, seed + 104729 * ck_idx, CKPT_SAMPLES)
        v = _stats.decide_from_probs(p_imp, p_reg, p_rope, threshold, budget)
        if v in ("improved", "regressed"):
            bad = [r["scenario_id"] for r in recs if r["verdict"] == "regressed"]
            stop = {"at": next_ck, "reason": v, "regressed_scenarios": bad}
            break
        if v == "undetermined" and maxr > next_ck:
            all_cells = [r["_eff"] for r in recs]
            pi_b, _, _ = overall_probs([_complete_improve(c, maxr) for c in all_cells],
                                       delta, seed + 104729 * ck_idx + 1, FUT_SAMPLES)
            _, pr_b, _ = overall_probs([_complete_regress(c, maxr) for c in all_cells],
                                       delta, seed + 104729 * ck_idx + 2, FUT_SAMPLES)
            po_b = 0.0
            cand_sets = [_complete_rope_candidates(c, maxr) for c in all_cells]
            for off, cand_cells in enumerate(zip(*cand_sets)):
                _, _, po = overall_probs(list(cand_cells),
                                         delta, seed + 104729 * ck_idx + 3 + off, FUT_SAMPLES)
                po_b = max(po_b, po)
            if pi_b < threshold and pr_b < threshold and po_b < 1.0 - budget:
                stop = {"at": next_ck, "reason": "futility", "regressed_scenarios": []}
                break
        next_ck += batch
    final_min = min(min(len(d["before"]), len(d["after"])) for d in data.values())
    if stop is None:
        stop = {"at": final_min, "reason": "max_runs", "regressed_scenarios": []}
    # 절감은 '판정에 의한 조기 종료'가 있을 때만 의미가 있다. 예산 소진은 절감 0.
    saved = max(0, maxr - stop["at"]) if stop["reason"] != "max_runs" else 0
    return {"stopped_at_per_arm": stop["at"], "stop_reason": stop["reason"],
            "runs_saved_per_arm": saved, "regressed_scenarios": stop["regressed_scenarios"]}


def analyze(runs, order, args, delta, delta_source):
    """전체 분석: 시나리오별 판정 + 층화 전체 판정 + 순차 재생 + 예외 처리(SPEC §9). 반환: 결과 JSON dict."""
    scen = {sid: {"before": [], "after": []} for sid in order}
    for sid, arm, passed in runs:
        scen[sid][arm].append(passed)

    threshold, calib, uncalibrated = load_calibration(args.calibration, args.batch_size,
                                                      args.max_runs, args.error_budget)
    warnings = []
    cal_delta = calib.get("calibrated_at_delta")
    if cal_delta is not None and delta < cal_delta - 1e-9:
        warnings.append("사용자 delta(%.3f)가 캘리브레이션 기준 delta(%.3f)보다 작습니다 — "
                        "FPR<=%g 보장은 delta>=%.3f 에서만 성립하며 그 아래에서는 근사입니다. "
                        "calibrate.py --delta %.3f 재실행을 권장합니다."
                        % (delta, cal_delta, args.error_budget, cal_delta, delta))
    records = []
    for sid in order:
        rec, warns = judge_scenario(sid, scen[sid]["before"], scen[sid]["after"], delta,
                                    threshold, args.error_budget, args.ceiling, args.floor)
        records.append(rec)
        warnings.extend(warns)

    judged_cells = [r["_eff"] for r in records if r["_judged"]]
    p_imp, p_reg, p_rope = overall_probs(judged_cells, delta, args.seed, FINAL_SAMPLES)
    overall_verdict = _stats.decide_from_probs(p_imp, p_reg, p_rope, threshold, args.error_budget)

    overall_flags = []
    if uncalibrated:
        overall_flags.append("uncalibrated_config")
        warnings.append("이 설정의 캘리브레이션이 없어 보수 기본문턱 %.2f 로 판정했습니다. "
                        "python3 calibrate.py --batch-size %d --max-runs %d --error-budget %g 실행을 권장합니다."
                        % (UNCALIBRATED_THRESHOLD, args.batch_size, args.max_runs, args.error_budget))
    if len(order) == 1:
        overall_flags.append("single_scenario")
        warnings.append("시나리오가 하나뿐이라 층화가 불가능합니다 — 전체 판정만 냅니다")
    else:
        warnings.append("캘리브레이션은 단일 시나리오 A/A 기준입니다 — 다중 시나리오 합산 판정의 실효 FPR은 같은 문턱의 근사치입니다")

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

    strata = set(r["stratum"] for r in records)
    if strata == {"ceiling"}:
        if "ceiling" not in overall_flags:
            overall_flags.append("ceiling")
        base_all = sum(r["_eff"][0] for r in records) / max(1, sum(r["_eff"][1] for r in records))
        head = (1.0 - base_all) * 100
        warnings.append("모든 시나리오가 천장 층입니다 — 이미 대부분 통과해 최대 가능 개선폭이 약 %.1f%%p에 불과합니다" % head)
    if strata == {"floor"}:
        for f in ("floor", "not_task_fixable"):
            if f not in overall_flags:
                overall_flags.append(f)

    dirs = set(r["verdict"] for r in records if r["_judged"]) & {"improved", "regressed"}
    conflict = (("improved" in dirs and "regressed" in dirs)
                or ("regressed" in dirs and overall_verdict != "regressed")
                or ("improved" in dirs and overall_verdict == "regressed"))
    if conflict:
        overall_flags.append("simpson_conflict")
        bad = [r["scenario_id"] for r in records if r["_judged"] and r["verdict"] == "regressed"]
        good = [r["scenario_id"] for r in records if r["_judged"] and r["verdict"] == "improved"]
        warnings.append("전체 판정(%s)과 시나리오별 판정이 어긋납니다(Simpson) — 개악: [%s]%s"
                        % (overall_verdict, ", ".join(bad) or "-",
                           (" / 개선: [%s]" % ", ".join(good)) if good else ""))

    needed = None
    if overall_verdict == "undetermined" and judged_cells:
        overall_flags.append("needs_more_runs")
        pool = judged_cells
        skb = sum(c[0] for c in pool); snb = sum(c[1] for c in pool)
        ska = sum(c[2] for c in pool); sna = sum(c[3] for c in pool)
        pb = (skb + 0.5) / (snb + 1.0)
        pa = (ska + 0.5) / (sna + 1.0)
        needed = _stats.needed_runs(pb, pa, delta, args.error_budget)
        if needed is None:
            warnings.append("관측 차이가 delta 경계와 거의 같아 유한한 표본 계획이 나오지 않습니다 — "
                            "delta를 조정하거나 시나리오를 늘리는 편이 낫습니다")

    ns_list = sorted(min(len(scen[s]["before"]), len(scen[s]["after"])) for s in order)
    n_rep = ns_list[len(ns_list) // 2] if ns_list else 0
    if n_rep > 0 and judged_cells:
        base = sum(c[0] for c in judged_cells) / max(1, sum(c[1] for c in judged_cells))
        floor95 = _stats.aa_noise_floor(base, n_rep, sims=4000, seed=args.seed + 1)
        if delta < floor95 and overall_verdict == "undetermined":
            need_d = _stats.needed_runs(base, base, delta, args.error_budget)
            warnings.append("요청한 delta(%.3f)가 현재 표본(팔당 %d회)의 노이즈 바닥(%.3f)보다 작아 판별 불가입니다"
                            " — 이 기저 통과율에서 delta %.3f 을 분해하려면 팔당 약 %s회가 필요합니다(정규근사 표본계획)"
                            % (delta, n_rep, floor95, delta, need_d if need_d else "산정 불가"))

    seq = sequential_replay(runs, order, delta, threshold, args.error_budget,
                            args.batch_size, args.max_runs, args.ceiling, args.floor, args.seed)
    calls_saved = None
    if args.calls_per_run is not None:
        calls_saved = int(round(seq["runs_saved_per_arm"] * 2 * len(order) * args.calls_per_run))
    seq_out = {"stopped_at_per_arm": seq["stopped_at_per_arm"], "stop_reason": seq["stop_reason"],
               "runs_saved_per_arm": seq["runs_saved_per_arm"], "calls_saved": calls_saved}
    if seq["stop_reason"] == "regressed" and seq["regressed_scenarios"]:
        seq_out["regressed_scenarios"] = seq["regressed_scenarios"]
        warnings.append("순차 재생: 개악 확정 시점의 실패 시나리오 목록: %s" % ", ".join(seq["regressed_scenarios"]))
    if seq["stop_reason"] == "futility":
        warnings.append("순차 재생: 남은 예산(max_runs %d)을 가장 유리하게 써도 delta=%.3f 판별이 불가능해 조기 중단했습니다"
                        % (args.max_runs, delta))

    scen_json = []
    for r in records:
        c = {k: v for k, v in r.items() if not k.startswith("_")}
        scen_json.append(c)

    result = {
        "config": {"input": args.input, "delta": delta, "delta_source": delta_source,
                   "ceiling": args.ceiling, "floor": args.floor, "batch_size": args.batch_size,
                   "max_runs": args.max_runs, "error_budget": args.error_budget,
                   "calls_per_run": args.calls_per_run, "seed": args.seed},
        "calibration": calib,
        "overall": {"verdict": overall_verdict, "p_improved": round(p_imp, 6),
                    "p_regressed": round(p_reg, 6), "p_rope": round(p_rope, 6),
                    "flags": overall_flags, "needed_runs_per_arm": needed},
        "scenarios": scen_json,
        "sequential": seq_out,
        "warnings": warnings,
    }
    return result


def print_human(res):
    """사람이 읽는 한국어 요약 (stdout). 기계용 JSON 과 반드시 둘 다 낸다 (SPEC §5)."""
    cfg = res["config"]
    cal = res["calibration"]
    ov = res["overall"]
    print("=" * 72)
    print("개선·개악 판별 결과  (delta=%.3f [%s], batch=%d, max_runs=%d, error_budget=%g, seed=%d)"
          % (cfg["delta"], cfg["delta_source"], cfg["batch_size"], cfg["max_runs"],
             cfg["error_budget"], cfg["seed"]))
    if cal["achieved_fpr"] is not None:
        print("캘리브레이션: 문턱 %.3f | 시뮬 %d회 | 달성 FPR %.4f (CI95 %.4f~%.4f) | 출처: %s"
              % (cal["threshold"], cal["sims"], cal["achieved_fpr"],
                 cal["fpr_ci95"][0], cal["fpr_ci95"][1], cal["source"]))
    else:
        print("캘리브레이션: 문턱 %.3f | %s" % (cal["threshold"], cal["source"]))
    print("-" * 72)
    print("시나리오별 판정 (층은 before 팔 관측 통과율로 결정 — 표본이 작으면 undetermined 가 정상):")
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
              % (s["scenario_id"], {"ceiling": "천장", "floor": "바닥", "middle": "중간"}[s["stratum"]],
                 nb, na, s["verdict"], s["p_improved"], s["p_regressed"], s["p_rope"],
                 (" | " + ",".join(s["flags"])) if s["flags"] else "", extra))
    print("-" * 72)
    print("전체 판정(층화 합산): %s   P(개선)=%.3f  P(개악)=%.3f  P(동등)=%.3f%s"
          % (ov["verdict"].upper(), ov["p_improved"], ov["p_regressed"], ov["p_rope"],
             ("   플래그: " + ",".join(ov["flags"])) if ov["flags"] else ""))
    if ov["verdict"] == "undetermined":
        need = ov["needed_runs_per_arm"]
        if "no_information" in ov["flags"]:
            print("  -> 판정 불가 — 두 팔을 구분할 정보가 없습니다(양팔 전승/전패)."
                  " 실행을 늘리는 것이 아니라 변별력 있는 시나리오가 필요합니다.")
        elif need and need <= 20000:
            print("  -> 아직 모릅니다. 결론에는 팔당 약 %d회가 필요합니다"
                  " (관측 통과율 고정 가정의 정규근사 표본계획 — 판정에는 미사용)." % need)
        elif need:
            print("  -> 아직 모릅니다. 정규근사 표본계획상 팔당 약 %d회 — 사실상 비현실적입니다."
                  " 관측 차이가 delta 경계 부근이라 전체 합산으로는 판별이 어렵습니다."
                  " 시나리오별 필요 횟수와 플래그를 보세요." % need)
        else:
            print("  -> 아직 모릅니다. 관측 차이가 delta 경계에 붙어 있어 유한한 표본 계획이 나오지 않습니다."
                  " 시나리오별 필요 횟수와 플래그를 보세요.")
    seq = res["sequential"]
    reason_ko = {"improved": "개선 확정", "regressed": "개악 확정", "futility": "무익(futility)",
                 "max_runs": "예산 소진(조기 종료 없음)"}[seq["stop_reason"]]
    line = ("순차 재생: 팔당 %d회 시점 %s — max_runs %d회 대비 팔당 %d회 절감"
            % (seq["stopped_at_per_arm"], reason_ko, cfg["max_runs"], seq["runs_saved_per_arm"]))
    if seq["calls_saved"] is not None:
        line += " (약 %d LLM 호출 절감)" % seq["calls_saved"]
    else:
        line += " (calls_per_run 미지정: 실행 횟수 기준)"
    print(line)
    if res["warnings"]:
        print("-" * 72)
        print("경고/고지:")
        for w in res["warnings"]:
            print("  ! " + w)
    print("=" * 72)


# ---------------------------- selftest (SPEC §10 8케이스) ----------------------------

def _run(cmd, stdin_text=None):
    p = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def run_selftest(args):
    """synth.py 8케이스 생성 -> 판정 -> 기대 대비 표. 하나라도 어긋나면 종료코드 1."""
    here = os.path.dirname(os.path.abspath(__file__))
    synth = os.path.join(here, "synth.py")
    calibrate = os.path.join(here, "calibrate.py")
    me = os.path.abspath(__file__)
    if not os.path.exists(synth):
        sys.stderr.write("selftest 오류: synth.py 가 없습니다(같은 폴더 필요) — 통합 단계에서 함께 검증하세요\n")
        sys.exit(1)
    rc, out, err = _run([sys.executable, synth, "--list"])
    if rc != 0:
        sys.stderr.write("selftest 오류: synth.py --list 실패\n" + err)
        sys.exit(1)
    metas = json.loads(out)
    tmpdir = tempfile.mkdtemp(prefix="verdict-selftest-")
    rows = []
    all_ok = True
    for meta in metas:
        case = meta["case"]
        exp = meta["expected"]
        t0 = time.time()
        problems = []
        got = ""
        if meta.get("kind") == "calibration":
            rc, out, err = _run([sys.executable, calibrate, "--verify",
                                 "--batch-size", str(meta["batch_size"]),
                                 "--max-runs", str(meta["max_runs"]),
                                 "--error-budget", str(meta["error_budget"]),
                                 "--sims", "300"])
            got = "verify rc=%d" % rc
            if rc != 0:
                problems.append("캘리브레이션 검증 실패(경계가 error_budget 미준수)")
                if err.strip():
                    problems.append(err.strip().splitlines()[-1])
        else:
            rc, data, err = _run([sys.executable, synth, "--case", str(case)])
            if rc != 0:
                problems.append("synth --case %d 실패" % case)
                data = ""
            jpath = os.path.join(tmpdir, "case%d.json" % case)
            cmd = [sys.executable, me, "-", "--delta", str(meta["delta"]),
                   "--batch-size", str(meta["batch_size"]), "--max-runs", str(meta["max_runs"]),
                   "--error-budget", str(meta["error_budget"]), "--yes",
                   "--calibration", args.calibration, "--json-out", jpath]
            rc, out, err = _run(cmd, stdin_text=data)
            if exp.get("error"):
                got = "rc=%d" % rc
                if rc != 2:
                    problems.append("종료코드 2 기대, 실제 %d" % rc)
                if "7" not in err or "arm" not in err:
                    problems.append("오류 메시지에 줄 번호(7)·필드명(arm)이 없음: %r" % err.strip())
            else:
                if rc != 0:
                    problems.append("종료코드 0 기대, 실제 %d: %s" % (rc, err.strip()[:200]))
                else:
                    with open(jpath, "r", encoding="utf-8") as f:
                        res = json.load(f)
                    ov = res["overall"]
                    got = ov["verdict"]
                    allflags = set(ov["flags"])
                    for s in res["scenarios"]:
                        allflags |= set(s["flags"])
                    if exp.get("verdict_in") and ov["verdict"] not in exp["verdict_in"]:
                        problems.append("verdict %r ∉ %r" % (ov["verdict"], exp["verdict_in"]))
                    missing = set(exp.get("flags_required", [])) - allflags
                    if missing:
                        problems.append("필수 플래그 누락: %s" % ",".join(sorted(missing)))
                    for sid, want in (exp.get("scenario_verdicts") or {}).items():
                        actual = {s["scenario_id"]: s["verdict"] for s in res["scenarios"]}.get(sid)
                        if actual != want:
                            problems.append("시나리오 %s: %r 기대, 실제 %r" % (sid, want, actual))
                    es = exp.get("early_stop")
                    seq = res["sequential"]
                    if es is True and (seq["stop_reason"] not in ("improved", "regressed")
                                       or seq["runs_saved_per_arm"] <= 0):
                        problems.append("조기 종료 기대: 실제 stop_reason=%s, 절감 %d"
                                        % (seq["stop_reason"], seq["runs_saved_per_arm"]))
                    # early_stop=False 는 '결론에 의한 조기 종료 없음'으로 해석(futility 는 허용)
                    if es is False and seq["stop_reason"] in ("improved", "regressed"):
                        problems.append("조기 종료 없음 기대: 실제 %s" % seq["stop_reason"])
        ok = not problems
        all_ok &= ok
        rows.append((case, meta["name"], got, "PASS" if ok else "FAIL", "; ".join(problems),
                     time.time() - t0))
    print("selftest 결과 (synth.py 8케이스, seed=%d):" % args.seed)
    print("  %-4s %-18s %-22s %-5s %s" % ("case", "name", "결과", "판정", "문제"))
    for case, name, got, ok, prob, el in rows:
        print("  %-4d %-18s %-22s %-5s %s(%.1fs)" % (case, name, got, ok, (prob + " ") if prob else "", el))
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
    ap.add_argument("--ceiling", type=float, default=0.90, help="천장 층 경계 기저 통과율 (기본 0.90)")
    ap.add_argument("--floor", type=float, default=0.10, help="바닥 층 경계 기저 통과율 (기본 0.10)")
    ap.add_argument("--batch-size", type=int, default=10, help="조기 종료 판정 주기 (기본 10)")
    ap.add_argument("--max-runs", type=int, default=100, help="팔당 최대 실행 수 예산 (기본 100)")
    ap.add_argument("--error-budget", type=float, default=0.05, help="허용 거짓 양성률 (기본 0.05)")
    ap.add_argument("--calls-per-run", type=float, default=None, help="실행 1회당 평균 LLM 호출 수(절감 환산용)")
    ap.add_argument("--calibration", default=os.path.join(here, "calibration-default.json"),
                    help="캘리브레이션 파일 (기본: 스크립트 폴더의 calibration-default.json)")
    ap.add_argument("--json-out", default="verdict.json", help="기계용 JSON 출력 경로 (기본 verdict.json)")
    ap.add_argument("--yes", action="store_true", help="delta 제안값을 확인 없이 수락")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="MC 시드 (기본 20260710, 결정론 보장)")
    ap.add_argument("--selftest", action="store_true", help="synth.py 8케이스 자기검증")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args)
        return
    if args.batch_size < 1 or args.max_runs < 1:
        sys.stderr.write("설정 오류: --batch-size/--max-runs 는 1 이상이어야 합니다\n")
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
    scen = {sid: {"before": [], "after": []} for sid in order}
    for sid, arm, passed in runs:
        scen[sid][arm].append(passed)
    delta, delta_source = resolve_delta(args, scen, order)

    res = analyze(runs, order, args, delta, delta_source)
    print_human(res)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("기계용 JSON 저장: %s" % args.json_out)


if __name__ == "__main__":
    main()
