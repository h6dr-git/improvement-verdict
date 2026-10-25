#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""조기 종료 경계 캘리브레이션 + 노이즈 바닥 표 (SPEC §2(a), §8 — 상수 문턱 금지의 근거를 직접 생산).

사용 예시: python3 src/scripts/calibrate.py           # 기본 두 설정 캘리브레이션 + calibration-default.json 기록

- A/A(양팔 동일) 순차 재생을 몬테카를로로 돌려, improved/regressed 오판율(FPR)이
  error_budget 이하가 되는 최소 사후확률 문턱을 그리드에서 찾고, 독립 시드로 재검증한다.
- 판정 경로는 verdict.py 와 동일한 _stats.prob_diff/decide 를 쓴다 — 경로가 갈라지면 캘리브레이션이 무의미.
- 기준 delta(기본 0.03)에서 캘리브레이션한다: A/A FPR은 delta에 단조 감소하므로
  이 문턱은 '캘리브레이션 delta 이상의 모든 delta'에 대해 보수적(상한)이다.
  (delta=0 은 ROPE 가 없어 훔쳐보기 반복 시 상수 문턱으로는 FPR 통제가 원리적으로 불가능하다 —
   그 측정치가 SPEC §2(c) '경계 보정 필수'의 근거다. verdict.py 는 사용자 delta 가
   캘리브레이션 delta 보다 작으면 보장이 근사로 약해진다고 경고한다.)
- 재생에 futility 중단을 넣지 않는다: futility 는 improved/regressed 를 내지 않으므로
  이를 뺀 시뮬레이션 FPR 은 실제 FPR 의 상한이다(보수 방향).
- plan_paths(): verdict.py --plan 용 순차 판정 시뮬레이션(같은 prob_diff·같은 문턱) — 검출률·평균 멈춤 시점.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stats

DEFAULT_SEED = 20260710
THRESHOLD_GRID = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
DEFAULT_CONFIGS = [(10, 100, 0.05), (20, 400, 0.05)]  # (batch_size, max_runs, error_budget)
NOISE_BASES = [0.10, 0.30, 0.50, 0.62, 0.70, 0.90, 0.95]
NOISE_NS = [25, 50, 100, 200, 400]


def replay_paths(batch, max_runs, p_before, p_after, sims, seed, delta):
    """순차 재생 경로 sims개를 생성해 경로별 (max max(p_imp,p_reg), max p_imp) 를 반환.

    문턱 t 로 조기 종료할 때 improved/regressed 가 한 번이라도 나오는 사건은
    '체크포인트 통계의 최댓값이 t 이상' 과 동치이므로, 최댓값만 저장하면 모든 문턱을 한 번에 평가한다.
    """
    ncheck = max_runs // batch
    cdf_b = _stats.binom_cdf_table(batch, p_before)
    cdf_a = _stats.binom_cdf_table(batch, p_after)
    rng = random.Random(seed)
    draw = _stats.binom_draw
    pdiff = _stats.prob_diff
    out = []
    for _ in range(sims):
        kb = ka = n = 0
        mx = mx_imp = 0.0
        for _c in range(ncheck):
            kb += draw(rng, cdf_b)
            ka += draw(rng, cdf_a)
            n += batch
            p_imp, p_reg, _ = pdiff(kb, n, ka, n, delta)
            if p_imp > mx_imp:
                mx_imp = p_imp
            m = p_imp if p_imp > p_reg else p_reg
            if m > mx:
                mx = m
            if mx >= 0.9995 and mx_imp >= 0.9995:
                break  # 그리드 최고 문턱(0.995)을 이미 넘었다 — 남은 체크포인트는 결과 불변
        out.append((mx, mx_imp))
    return out


def plan_paths(batch, max_runs, p_before, p_after, scenarios, sims, seed, delta, threshold,
               unchanged=0, guard_threshold=None, ceiling=0.90, floor=0.10):
    """계획 모드(verdict.py --plan)용 순차 판정 시뮬레이션.

    바뀔 시나리오 K개(p_before -> p_after)와 그대로인 중간층 시나리오 M개(수정 전후 모두 p_before)를 각각
    체크포인트마다 실행 수를 합친 단일 시나리오로 근사한다(K개 평균 차이의 사후분포와 합친 단일 시나리오의
    사후분포는 분산이 같아지는 근사 — K=1, M=0 이면 판정 경로가 verdict.py 와 정확히 같다).
      - guard_threshold 가 없으면(전체 평균 판정): K+M개를 합친 평균으로 판정.
      - guard_threshold 가 있으면(층별 판정): 체크포인트마다 바뀔 쪽의 관측 합산 통과율로 층을 정한다.
        천장·바닥층이고 M>=1 이면 verdict 의 주 판정 규칙과 같이 — 바뀔 쪽은 가드(guard_threshold),
        주 판정은 그대로인 중간층(threshold), 개악이 있으면 개악, 아니면 개선. 그 밖에는 K+M개 평균(threshold).
    첫 확정(improved/regressed)에서 멈추고 futility 는 넣지 않는다(평균 실행 수는 상한 쪽).
    '검출'은 기대 방향의 확정이다 — p_after > p_before 면 improved, 작으면 regressed.
    반환: 검출률, 반대 방향 오판율, 검출 시 평균 멈춤(팔당), 전체 평균 실행 수(팔당), 가드 경로로 판정한 체크포인트 비율.
    """
    ncheck = max_runs // batch
    nk = batch * scenarios
    nm = batch * unchanged
    want = "regressed" if p_after < p_before else "improved"
    cdf_b = _stats.binom_cdf_table(nk, p_before)
    cdf_a = _stats.binom_cdf_table(nk, p_after)
    cdf_u = _stats.binom_cdf_table(nm, p_before) if nm else None
    rng = random.Random(seed)
    draw = _stats.binom_draw
    pdiff = _stats.prob_diff
    detected = wrong = 0
    stop_sum = 0
    runs_sum = 0
    looks = side_looks = 0
    for _ in range(sims):
        kb = ka = ub = ua = 0
        n_k = n_m = 0
        outcome = None
        for c in range(1, ncheck + 1):
            kb += draw(rng, cdf_b)
            ka += draw(rng, cdf_a)
            n_k += nk
            if nm:
                ub += draw(rng, cdf_u)
                ua += draw(rng, cdf_u)
                n_m += nm
            looks += 1
            side = False
            if guard_threshold is not None and nm:
                pooled = (kb + ka) / (2.0 * n_k)
                side = pooled >= ceiling or pooled <= floor
            if side:
                side_looks += 1
                g_imp, g_reg, _ = pdiff(kb, n_k, ka, n_k, delta)
                m_imp, m_reg, _ = pdiff(ub, n_m, ua, n_m, delta)
                reg = g_reg >= guard_threshold or m_reg >= threshold
                imp = (not reg) and (g_imp >= guard_threshold or m_imp >= threshold)
            else:
                p_imp, p_reg, _ = pdiff(kb + ub, n_k + n_m, ka + ua, n_k + n_m, delta)
                imp, reg = p_imp >= threshold, p_reg >= threshold
            if imp:
                outcome = ("improved", c * batch)
                break
            if reg:
                outcome = ("regressed", c * batch)
                break
        if outcome is None:
            runs_sum += ncheck * batch
        else:
            runs_sum += outcome[1]
            if outcome[0] == want:
                detected += 1
                stop_sum += outcome[1]
            else:
                wrong += 1
    return {"direction": want,
            "power": round(detected / float(sims), 4),
            "wrong_direction": round(wrong / float(sims), 4),
            "mean_stop_when_detected": (round(stop_sum / float(detected), 1) if detected else None),
            "expected_runs_per_arm": round(runs_sum / float(sims), 1),
            "guard_path_share": round(side_looks / float(looks), 4) if looks else 0.0,
            "sims": sims}


def calibrate_config(batch, max_runs, budget, base, sims, seed, delta):
    """한 설정의 문턱 탐색 -> 독립 시드 재검증 -> 검정력(0.62->0.75) 추정. 엔트리 dict 반환."""
    t0 = time.time()
    print("[캘리브레이션] batch=%d, max_runs=%d, budget=%g, base=%.2f, delta=%g, sims=%d, seed=%d"
          % (batch, max_runs, budget, base, delta, sims, seed))
    search = replay_paths(batch, max_runs, base, base, sims, seed, delta)
    print("  문턱 탐색 (A/A 오판율, 훔쳐보기 %d회 포함):" % (max_runs // batch))
    threshold = None
    for t in THRESHOLD_GRID:
        fpr = sum(1 for mx, _ in search if mx >= t) / float(sims)
        mark = ""
        if threshold is None and fpr <= budget:
            threshold = t
            mark = "  <- 선택(최소 문턱)"
        print("    문턱 %.3f -> FPR %.4f%s" % (t, fpr, mark))
    warn = None
    if threshold is None:
        threshold = THRESHOLD_GRID[-1]
        warn = ("그리드 내 어떤 문턱도 budget %g 를 만족하지 못해 최고값 %.3f 을 기록합니다"
                " — 이 설정은 훔쳐보기 횟수가 너무 많습니다" % (budget, threshold))
        print("  ! " + warn)
    # 독립 시드 재검증: 탐색 표본으로 자기 채점하면 경계에서 과적합(winner's curse)이 난다.
    # 재검증 FPR 이 budget 을 넘으면 문턱을 한 칸씩 올리고, 최종 확정치는 세 번째 시드로 측정한다.
    vseed = seed + 99991
    verify = replay_paths(batch, max_runs, base, base, sims, vseed, delta)
    gi = THRESHOLD_GRID.index(threshold)
    while True:
        fpr_v = sum(1 for mx, _ in verify if mx >= threshold) / float(sims)
        if fpr_v <= budget or gi + 1 >= len(THRESHOLD_GRID):
            break
        gi += 1
        threshold = THRESHOLD_GRID[gi]
        print("  재검증 FPR %.4f > budget — 문턱을 %.3f 로 상향" % (fpr_v, threshold))
    # 최종 확정: 상향에 쓴 표본과도 독립인 시드 + 더 큰 표본(경계 근처 MC 오차를 줄여 기록을 정직하게)
    sims_c = max(sims, 4000)
    cseed = seed + 199982
    confirm = replay_paths(batch, max_runs, base, base, sims_c, cseed, delta)
    fpr = sum(1 for mx, _ in confirm if mx >= threshold) / float(sims_c)
    se = math.sqrt(max(fpr * (1.0 - fpr), 1e-12) / sims_c)
    if fpr > budget + se and gi + 1 < len(THRESHOLD_GRID):
        gi += 1
        threshold = THRESHOLD_GRID[gi]
        print("  확정 FPR %.4f 이 budget 을 유의하게 초과 — 문턱을 %.3f 로 최종 상향 후 재확정" % (fpr, threshold))
        confirm = replay_paths(batch, max_runs, base, base, sims_c, seed + 299973, delta)
        fpr = sum(1 for mx, _ in confirm if mx >= threshold) / float(sims_c)
        se = math.sqrt(max(fpr * (1.0 - fpr), 1e-12) / sims_c)
    ci = [max(0.0, fpr - 1.96 * se), min(1.0, fpr + 1.96 * se)]
    sims_p = min(sims, 400)
    power_paths = replay_paths(batch, max_runs, 0.62, 0.75, sims_p, seed + 424243, delta)
    power = sum(1 for _, mi in power_paths if mi >= threshold) / float(sims_p)
    print("  선택 문턱 %.3f | 최종 확정 FPR %.4f (CI95 %.4f~%.4f, 독립 시드 %d회)"
          % (threshold, fpr, ci[0], ci[1], sims_c))
    print("  검정력(0.62->0.75, 팔당 %d회 내 검출): %.3f (%d회) | 소요 %.1fs"
          % (max_runs, power, sims_p, time.time() - t0))
    entry = {
        "config": {"batch_size": batch, "max_runs": max_runs, "error_budget": budget},
        "threshold": threshold,
        "sims": sims_c,
        "achieved_fpr": round(fpr, 6),
        "fpr_ci95": [round(ci[0], 6), round(ci[1], 6)],
        "power_62_75": round(power, 6),
        "seed": seed,
        "calibrated_at_delta": delta,
        "base": base,
        "note": "A/A 단일 시나리오, delta=%g 기준 캘리브레이션 — delta>=%g 인 판정에 보수적 상한. "
                "futility 미적용 FPR 상한." % (delta, delta),
    }
    if warn:
        entry["warning"] = warn
    return entry


def print_noise_table(seed, sims=10000):
    """A/A 노이즈 바닥 표 — delta 하한 제안의 근거 (SPEC §2(a) 재생산)."""
    print("[노이즈 바닥] 양팔이 완전히 동일할 때 우연만으로 벌어지는 |통과율 차이|의 95%% 분위 (%%p, 시드 고정 이항 MC %d회):" % sims)
    header = "  기저 통과율 " + "".join("  팔당 %4d회" % n for n in NOISE_NS)
    print(header)
    for base in NOISE_BASES:
        cells = []
        for n in NOISE_NS:
            f = _stats.aa_noise_floor(base, n, sims=sims, seed=seed + int(base * 1000) * 7 + n)
            cells.append("      ±%4.1f" % (f * 100))
        print("      %4.0f%%   " % (base * 100) + "".join(cells))
    print("  => delta 는 해당 칸의 노이즈 바닥 이상으로 잡아야 우연을 차이로 오인하지 않습니다.")
    print("     (기저 통과율에 따라 폭이 2배 이상 달라지므로 단일 delta를 전 시나리오에 강제하지 말 것)")


def do_verify(args, out_path):
    """캘리브레이션 파일의 문턱이 error_budget 을 실제로 지키는지 독립 시드로 자기검증 (selftest 케이스 8).

    통과 기준: 달성 FPR <= budget + 2 * MC표준오차.
    """
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", []) if isinstance(data, dict) else data
    except (OSError, ValueError) as e:
        sys.stderr.write("verify 오류: 캘리브레이션 파일을 읽을 수 없습니다(%s) — calibrate.py 기본 실행으로 먼저 생성하세요\n" % e)
        sys.exit(1)
    entry = None
    for e in entries:
        cfg = e.get("config", {})
        if cfg.get("batch_size") == args.batch_size and cfg.get("max_runs") == args.max_runs \
                and abs(cfg.get("error_budget", -1) - args.error_budget) < 1e-9:
            entry = e
            break
    if entry is None:
        sys.stderr.write("verify 오류: (batch=%d, max_runs=%d, budget=%g) 엔트리가 없습니다 — 먼저 캘리브레이션하세요\n"
                         % (args.batch_size, args.max_runs, args.error_budget))
        sys.exit(1)
    threshold = entry["threshold"]
    base = entry.get("base", args.base)
    delta = entry.get("calibrated_at_delta", args.delta)
    vseed = args.seed + 777777  # 기록 시드와 다른 독립 시드
    sims = args.sims
    print("[검증] 문턱 %.3f (batch=%d, max_runs=%d, budget=%g, delta=%g) | 독립 시드 %d, A/A %d회"
          % (threshold, args.batch_size, args.max_runs, args.error_budget, delta, vseed, sims))
    paths = replay_paths(args.batch_size, args.max_runs, base, base, sims, vseed, delta)
    hits = sum(1 for mx, _ in paths if mx >= threshold)
    fpr = hits / float(sims)
    se = math.sqrt(max(fpr * (1.0 - fpr), 1e-12) / sims)
    ok = fpr <= args.error_budget + 2.0 * se
    print("  달성 FPR %.4f | MC표준오차 %.4f | 기준 %.4f + 2*%.4f = %.4f -> %s"
          % (fpr, se, args.error_budget, se, args.error_budget + 2 * se, "통과" if ok else "실패"))
    print(json.dumps({"mode": "verify", "config": {"batch_size": args.batch_size,
                      "max_runs": args.max_runs, "error_budget": args.error_budget},
                      "threshold": threshold, "sims": sims, "achieved_fpr": round(fpr, 6),
                      "mc_se": round(se, 6), "pass": ok}, ensure_ascii=False))
    sys.exit(0 if ok else 1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="조기 종료 경계 캘리브레이션 + 노이즈 바닥 표")
    ap.add_argument("--batch-size", type=int, default=None, help="특정 설정만 캘리브레이션(미지정 시 기본 두 설정)")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--error-budget", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.03,
                    help="캘리브레이션 기준 delta (기본 0.03) — 이 값 이상의 delta 판정에 보수적 상한")
    ap.add_argument("--base", type=float, default=0.62, help="A/A 시뮬 기저 통과율 (기본 0.62)")
    ap.add_argument("--sims", type=int, default=800, help="설정당 시뮬 횟수 (기본 800)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=os.path.join(here, "calibration-default.json"),
                    help="결과 기록 파일 (기본: 스크립트 폴더의 calibration-default.json)")
    ap.add_argument("--verify", action="store_true", help="기록된 문턱이 budget을 지키는지 독립 시드로 검증")
    ap.add_argument("--noise-only", action="store_true", help="노이즈 바닥 표만 출력")
    ap.add_argument("--no-noise", action="store_true", help="노이즈 바닥 표 생략")
    args = ap.parse_args()

    if args.verify:
        if args.batch_size is None or args.max_runs is None:
            sys.stderr.write("verify 오류: --batch-size 와 --max-runs 를 지정하세요\n")
            sys.exit(1)
        do_verify(args, args.out)
        return

    t0 = time.time()
    if not args.no_noise:
        print_noise_table(args.seed)
        print()
    if args.noise_only:
        return

    if args.batch_size is not None or args.max_runs is not None:
        configs = [(args.batch_size or 10, args.max_runs or 100, args.error_budget)]
    else:
        configs = DEFAULT_CONFIGS

    new_entries = []
    for batch, maxr, budget in configs:
        new_entries.append(calibrate_config(batch, maxr, budget, args.base, args.sims,
                                            args.seed, args.delta))
        print()

    # 기존 파일과 병합: 같은 설정은 교체, 다른 설정은 유지
    entries = []
    try:
        with open(args.out, "r", encoding="utf-8") as f:
            old = json.load(f)
        entries = old.get("entries", []) if isinstance(old, dict) else old
    except (OSError, ValueError):
        pass
    new_keys = set((e["config"]["batch_size"], e["config"]["max_runs"], e["config"]["error_budget"])
                   for e in new_entries)
    entries = [e for e in entries
               if (e.get("config", {}).get("batch_size"), e.get("config", {}).get("max_runs"),
                   e.get("config", {}).get("error_budget")) not in new_keys]
    entries.extend(new_entries)
    entries.sort(key=lambda e: (e["config"]["max_runs"], e["config"]["batch_size"]))
    doc = {"entries": entries,
           "meta": {"tool": "calibrate.py", "seed": args.seed,
                    "note": "verdict.py 가 (batch_size, max_runs, error_budget) 매칭으로 문턱을 로드한다. "
                            "합성 A/A 시뮬레이션만 사용(실데이터 없음)."}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("기록 완료: %s (엔트리 %d개, 총 %.1fs)" % (args.out, len(entries), time.time() - t0))


if __name__ == "__main__":
    main()
