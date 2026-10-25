#!/usr/bin/env python3
"""
synth.py — 개선·개악 판별기(verdict.py) 검증용 합성 데이터 생성기.

SPEC §10의 8케이스(v1)와 v2 검증 5케이스(9 층 희석된 개선, 10 천장층 개악 가드,
11 온라인 멈춤 시점 일치, 12 계획 모드 검출률, 13 크게 고친 시나리오의 가드 개선)를 생성한다. 실제 상담 데이터는 일절 쓰지 않는다(전부 합성).
표준 라이브러리만 사용한다(Python 3.9+). 네트워크·LLM 호출 없음.

사용법:
    python3 src/scripts/synth.py --case N [--seed S]   # 케이스 N의 JSONL -> stdout
    python3 src/scripts/synth.py --meta N              # 케이스 N의 메타 JSON -> stdout
    python3 src/scripts/synth.py --list                # 전 케이스 메타 배열 JSON -> stdout

데이터 한 줄 = {"scenario_id": str, "arm": "before"|"after", "passed": bool, "synthetic": true}
  - "synthetic"은 최소 스키마 밖의 무해한 추가 필드다. verdict.py가 모르는 필드를
    무시하는지(전방 호환) 실전으로 검사하는 용도.

생성 원리:
  - (시나리오, 팔)별 통과 수를 round(목표율 x 실행 수)로 정확히 고정하고,
    통과/실패의 "순서"만 시드로 섞는다. 따라서 --seed를 바꿔도 각 케이스의
    기대 판정은 유지되고, 순차 재생(조기 종료) 경로만 달라진다.
    단일 표본의 우연에 기대지 않는 거짓 양성률 검증은 케이스 8(캘리브레이션)의 몫이다.
  - 줄 순서는 실제 실행 순서를 시뮬레이트한다: 매 회차마다 시나리오를 라운드로빈으로
    돌며 before/after 쌍을 순서대로 낸다. 순차 재생이 의미를 갖는 배치다.
  - 같은 시드면 출력은 완전히 결정론적이다. 시드는 문자열 "{seed}-case{N}"으로
    케이스별로 파생하므로 케이스 간 간섭이 없고 파이썬 버전과 무관하게 재현된다.
"""

import argparse
import json
import random
import sys

DEFAULT_SEED = 20260710

_ALL_VERDICTS = ["improved", "regressed", "no_difference", "undetermined"]

# ---------------------------------------------------------------------------
# 8케이스 정의 (SPEC §10 + 검정력 사전 검산 반영)
#
# scenarios: [{scenario_id, n(팔당 실행 수), before/after: {"rate": f} 또는 {"passes": k}}]
#   rate  -> 통과 수 = round(rate * n) 로 고정
#   passes-> 통과 수를 정수로 직접 고정 (극단 실현을 확정적으로 재현할 때 사용)
# ---------------------------------------------------------------------------

CASES = {
    1: {
        "name": "aa-null",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 100,
        "error_budget": 0.05,
        "scenarios": [
            {"scenario_id": "s1_aa", "n": 100, "before": {"rate": 0.62}, "after": {"rate": 0.62}},
            {"scenario_id": "s2_aa", "n": 100, "before": {"rate": 0.62}, "after": {"rate": 0.62}},
            {"scenario_id": "s3_aa", "n": 100, "before": {"rate": 0.62}, "after": {"rate": 0.62}},
        ],
        "expected": {
            "verdict_in": ["undetermined", "no_difference"],
            "flags_required": [],
            "error": False,
            "early_stop": None,
        },
        "notes": (
            "A/A 자기검증: 양팔이 완전히 동일(기저 0.62, 시나리오 3개)이므로 improved/regressed가 "
            "나오면 판정기가 노이즈를 차이로 오인한 것이다. 통과 수는 목표율의 정확한 반올림으로 "
            "고정하고 순서만 시드로 섞으므로 어떤 시드에서도 이 기대가 유지된다."
        ),
    },
    2: {
        "name": "true-improvement",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 20,
        "max_runs": 400,
        "error_budget": 0.05,
        "scenarios": [
            {"scenario_id": "s1_target", "n": 400, "before": {"rate": 0.62}, "after": {"rate": 0.75}},
        ],
        "expected": {
            "verdict_in": ["improved"],
            "flags_required": [],
            "error": False,
            "early_stop": True,
        },
        "notes": (
            "진짜 개선 62%->75%. 팔당 100회.delta 0.05로는 검출확률 ~50%(SPEC §2(d))라 데모가 "
            "반반으로 깨지므로, 팔당 400회.delta 0.03으로 사후확률 ~0.999의 안정 검출을 확보했다 "
            "(검정력 사전 검산). 400회를 다 쓰기 전에 조기 종료가 걸려 절감이 보고되어야 한다. "
            "시나리오가 하나라 single_scenario 플래그도 정상 동작이나 필수 검사 항목은 아니다."
        ),
    },
    3: {
        "name": "true-regression",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 20,
        "max_runs": 400,
        "error_budget": 0.05,
        "scenarios": [
            {"scenario_id": "s1_target", "n": 400, "before": {"rate": 0.62}, "after": {"rate": 0.50}},
        ],
        "expected": {
            "verdict_in": ["regressed"],
            "flags_required": [],
            "error": False,
            "early_stop": True,
        },
        "notes": (
            "진짜 개악 62%->50%. 케이스 2와 동일한 검정력 사전 검산에 따라 팔당 400회.delta 0.03. "
            "개악은 가장 급한 신호이므로 조기 종료(즉시 중단)가 걸려야 한다."
        ),
    },
    4: {
        "name": "simpson",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 100,
        "error_budget": 0.05,
        "scenarios": [
            {"scenario_id": "s1_good", "n": 100, "before": {"rate": 0.55}, "after": {"rate": 0.7125}},
            {"scenario_id": "s2_good", "n": 100, "before": {"rate": 0.55}, "after": {"rate": 0.7125}},
            {"scenario_id": "s3_good", "n": 100, "before": {"rate": 0.55}, "after": {"rate": 0.7125}},
            {"scenario_id": "s4_good", "n": 100, "before": {"rate": 0.55}, "after": {"rate": 0.7125}},
            {"scenario_id": "s5_bad", "n": 100, "before": {"rate": 0.75}, "after": {"rate": 0.25}},
        ],
        "expected": {
            "verdict_in": list(_ALL_VERDICTS),
            "flags_required": ["simpson_conflict"],
            "error": False,
            "early_stop": None,
            "scenario_verdicts": {"s5_bad": "regressed"},
        },
        "notes": (
            "Simpson 케이스: 4개 시나리오는 +16%p 개선, s5_bad만 -50%p 개악으로 합산 diff는 약 "
            "+2.8%p에 불과하다. 합계만 보면 개악을 놓치므로 전체 판정이 무엇이든 simpson_conflict "
            "플래그와 s5_bad=regressed 판정이 필수다. verdict_in에 네 판정을 전부 나열한 것은 "
            "전체 판정에 제약이 없다는 뜻이다."
        ),
    },
    5: {
        "name": "ceiling",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 20,
        "error_budget": 0.05,
        "scenarios": [
            {"scenario_id": "s1_allpass", "n": 20, "before": {"passes": 20}, "after": {"passes": 20}},
            {"scenario_id": "s2_near_ceiling", "n": 20, "before": {"passes": 19}, "after": {"passes": 19}},
        ],
        "expected": {
            "verdict_in": ["undetermined"],
            "flags_required": ["ceiling", "needs_more_runs"],
            "error": False,
            "early_stop": None,
        },
        "notes": (
            "극단 케이스(기저 0.97, 팔당 20회). s1_allpass는 양팔 전승(이 조건에서 ~30% 확률로 실제 "
            "발생하는 실현, SPEC §2(b))을 고정 재현해 표준오차 0에서의 계산 붕괴 내성을 검사하고, "
            "s2는 19/20으로 천장 층 분류를 검사한다. 핵심은 오류 없이 완주하는 것: "
            "undetermined + ceiling/needs_more_runs가 기대 결과다."
        ),
    },
    6: {
        "name": "floor",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 100,
        "error_budget": 0.05,
        "scenarios": [
            {"scenario_id": "s1_hard", "n": 100, "before": {"rate": 0.05}, "after": {"rate": 0.06}},
            {"scenario_id": "s2_hard", "n": 100, "before": {"rate": 0.05}, "after": {"rate": 0.06}},
        ],
        "expected": {
            "verdict_in": ["undetermined", "no_difference"],
            "flags_required": ["floor", "not_task_fixable"],
            "error": False,
            "early_stop": None,
        },
        "notes": (
            "바닥 케이스: 수정 전후 모두 거의 전패(5%->6%). 자동 개선 루프가 영원히 붙들고 도는 "
            "지점이므로 '태스크 수정으로 풀 문제가 아니다'(not_task_fixable) 신호가 나와야 한다."
        ),
    },
    7: {
        "name": "schema-error",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 10,
        "error_budget": 0.05,
        "scenarios": [
            {"scenario_id": "s1_checkout", "n": 10, "before": {"rate": 0.6}, "after": {"rate": 0.6}},
            {"scenario_id": "s2_refund", "n": 10, "before": {"rate": 0.6}, "after": {"rate": 0.6}},
        ],
        "expected": {
            "verdict_in": [],
            "flags_required": [],
            "error": True,
            "early_stop": None,
            "error_line": 7,
            "error_field": "arm",
        },
        "notes": (
            "스키마 오류 케이스: 정확히 7번째 줄에서 arm 필드를 구조적으로(시드와 무관하게) 제거했다. "
            "판정기는 즉시 중단하고 오류 메시지에 줄 번호 7과 필드명 arm을 지목하며 종료코드 2로 "
            "끝나야 한다. expected.error_line/error_field는 기계 검증용 보조 필드다(error=true이므로 "
            "verdict_in은 빈 배열 = 해당 없음)."
        ),
    },
    8: {
        "name": "calibration",
        "kind": "calibration",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 100,
        "error_budget": 0.05,
        "scenarios": [],
        "expected": {
            "verdict_in": [],
            "flags_required": [],
            "error": False,
            "early_stop": None,
        },
        "notes": (
            "데이터셋이 아닌 자기검증 케이스. calibrate.py 빠른 모드를 이 메타의 설정(delta 0.03, "
            "batch 10, max 100, error_budget 0.05)으로 실행해, 조기 종료 경계의 달성 거짓 양성률이 "
            "error_budget + 2x몬테카를로 표준오차 이하인지 확인한다(SPEC §2(c)의 훔쳐보기 팽창 "
            "18.44%가 실제로 억제되는지의 검증). --case 8은 데이터를 출력하지 않는다."
        ),
    },
    # ------------------------- v2 추가 케이스 (9~13) -------------------------
    9: {
        "name": "masked-improvement",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 100,
        "error_budget": 0.05,
        "scenarios": None,  # _masked_scenarios() 로 채움
        "expected": {
            "verdict_in": ["undetermined"],
            "headline_in": ["improved"],
            "headline_basis": "middle",
            "flags_required": ["overall_diluted"],
            "scenario_strata": {"mid_1": "middle", "ceil_1": "ceiling", "floor_1": "floor"},
            "error": False,
            "early_stop": None,
        },
        "notes": (
            "v2 층별 판정 케이스: 중간층 3개는 60%->75%로 확실히 좋아졌고, 천장(97%) 2개·바닥(5%) 2개는 그대로다. "
            "v1의 전체 평균은 변화 없는 4개에 희석돼 최종 판정이 undetermined 에 머문다(순차 재생은 도착 순서에 따라 "
            "중간에 개선으로 멈추기도 한다). v2 주 판정은 중간층 기준이라 improved 여야 하고 overall_diluted 플래그가 필수다. "
            "(최종 판정은 통과 수만으로 정해져 시드와 무관하다. 조기 종료 여부는 도착 순서에 따라 드물게 달라져 케이스 11이 본다.)"
        ),
    },
    10: {
        "name": "guard-regression",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 100,
        "error_budget": 0.05,
        "scenarios": [
            {"scenario_id": "mid_1", "n": 100, "before": {"rate": 0.60}, "after": {"rate": 0.60}},
            {"scenario_id": "mid_2", "n": 100, "before": {"rate": 0.60}, "after": {"rate": 0.60}},
            {"scenario_id": "ceil_1", "n": 100, "before": {"passes": 100}, "after": {"passes": 91}},
            {"scenario_id": "ceil_2", "n": 100, "before": {"passes": 100}, "after": {"passes": 91}},
            {"scenario_id": "ceil_3", "n": 100, "before": {"passes": 100}, "after": {"passes": 91}},
            {"scenario_id": "ceil_4", "n": 100, "before": {"passes": 100}, "after": {"passes": 91}},
        ],
        "expected": {
            "verdict_in": list(_ALL_VERDICTS),
            "headline_in": ["regressed"],
            "flags_required": ["guard_regressed"],
            "scenario_strata": {"ceil_1": "ceiling", "mid_1": "middle"},
            "error": False,
            "early_stop": True,
        },
        "notes": (
            "v2 가드 케이스: 고치려던 중간층은 그대로(60%)인데, 늘 통과하던 천장층 4개가 100%->91%로 조금씩 깨졌다. "
            "두 팔 모두 90% 이상이라 이 시나리오들은 천장층에 남아 주 판정(중간층)에 들어가지 않는다. 그래서 천장층 평균의 "
            "개악이 가드 문턱으로 확정되면 주 판정이 regressed 로 올라가고 guard_regressed 플래그와 함께 조기 종료가 "
            "걸려야 한다. (99%->89%처럼 90% 밑으로 떨어지는 고장은 중간층으로 들어와 주 판정이 직접 잡는다.)"
        ),
    },
    11: {
        "name": "online-consistency",
        "kind": "online",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 100,
        "error_budget": 0.05,
        "scenarios": None,  # _masked_scenarios() 로 채움
        "expected": {
            "verdict_in": list(_ALL_VERDICTS),
            "headline_in": ["improved"],
            "headline_basis": "middle",
            "flags_required": [],
            "error": False,
            "early_stop": True,
        },
        "notes": (
            "v2 온라인 모드 일치 검증: 케이스 9와 같은 구성(단 중간층은 55%->85%로 효과를 키워 어떤 시드에서도 max_runs 전에 "
            "멈추게 함)을 오프라인으로 판정해 멈춘 시점 T를 구한 뒤, "
            "데이터 앞부분만 --online 으로 넣는다. 팔당 T-batch 회까지는 종료코드 0(계속), 팔당 T회에서 10(개선 확정)이어야 한다 "
            "— QA 도중 배치마다 부르는 온라인 판정과 다 돌린 뒤의 재생이 같은 시점에 멈춘다는 보장."
        ),
    },
    12: {
        "name": "plan-consistency",
        "kind": "plan",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 100,
        "error_budget": 0.05,
        "scenarios": [],
        "plan": {"base": 0.62, "target": 0.75, "scenarios": 1, "sims": 400, "power": 0.8},
        "expected": {
            "verdict_in": [],
            "flags_required": [],
            "error": False,
            "early_stop": None,
            "power_tolerance": 0.10,
            "recommended": [20, 400],
        },
        "notes": (
            "v2 계획 모드 검증: 62%->75%, 시나리오 1개, delta 0.03 으로 --plan 을 돌려 설정별 검출률이 "
            "calibration-default.json 에 기록된 검정력(power_62_75: batch10/max100 0.345, batch20/max400 0.8375)과 "
            "몬테카를로 오차(±0.10) 안에서 일치하고, 목표 검출률 80%에서 batch 20 / max 400 을 권장해야 한다. "
            "데이터를 출력하지 않는다."
        ),
    },
    13: {
        "name": "big-fix-guard",
        "kind": "dataset",
        "delta": 0.03,
        "batch_size": 10,
        "max_runs": 100,
        "error_budget": 0.05,
        "scenarios": [
            {"scenario_id": "fix_1", "n": 100, "before": {"passes": 85}, "after": {"passes": 99}},
            {"scenario_id": "fix_2", "n": 100, "before": {"passes": 85}, "after": {"passes": 99}},
            {"scenario_id": "fix_3", "n": 100, "before": {"passes": 85}, "after": {"passes": 99}},
            {"scenario_id": "rescue_1", "n": 100, "before": {"passes": 0}, "after": {"passes": 20}},
            {"scenario_id": "same_1", "n": 100, "before": {"rate": 0.50}, "after": {"rate": 0.50}},
            {"scenario_id": "ceil_1", "n": 100, "before": {"passes": 97}, "after": {"passes": 97}},
            {"scenario_id": "ceil_2", "n": 100, "before": {"passes": 97}, "after": {"passes": 97}},
        ],
        "expected": {
            "verdict_in": list(_ALL_VERDICTS),
            "headline_in": ["improved"],
            "headline_basis": "middle",
            "flags_required": ["guard_improved"],
            "scenario_strata": {"fix_1": "ceiling", "rescue_1": "floor", "same_1": "middle", "ceil_1": "ceiling"},
            "scenario_flags_absent": {"fix_1": ["ceiling"], "rescue_1": ["floor", "not_task_fixable"]},
            "error": False,
            "early_stop": True,
        },
        "notes": (
            "v2 가드 개선 케이스(검토에서 찾은 결함의 회귀 방지): 85%->99%로 크게 고친 시나리오는 양팔 합산 통과율이 높아 "
            "천장층에, 0%->20%로 살린 시나리오는 바닥층에 묶인다. 주 판정 근거(중간층 = 그대로인 50% 하나)만 보면 개선을 "
            "놓치므로, 천장·바닥층 평균의 개선이 가드 문턱으로 확정되면 주 판정이 improved 로 올라가야 한다(guard_improved). "
            "또 이 시나리오들에 '개선 여지가 작다'(ceiling)·'태스크 수정으로 풀 문제가 아니다'(not_task_fixable) 같은 "
            "해석 플래그가 붙으면 안 된다 — 그 플래그는 수정 전후 두 팔이 모두 극단일 때만 붙는다."
        ),
    },
}


def _masked_scenarios(mid_before=0.60, mid_after=0.75):
    """케이스 9·11: 중간층 3개만 개선(기본 60->75%), 천장 2개·바닥 2개는 그대로."""
    out = []
    for i in range(1, 4):
        out.append({"scenario_id": "mid_%d" % i, "n": 100, "before": {"rate": mid_before}, "after": {"rate": mid_after}})
    for i in range(1, 3):
        out.append({"scenario_id": "ceil_%d" % i, "n": 100, "before": {"passes": 97}, "after": {"passes": 97}})
    for i in range(1, 3):
        out.append({"scenario_id": "floor_%d" % i, "n": 100, "before": {"passes": 5}, "after": {"passes": 5}})
    return out


CASES[9]["scenarios"] = _masked_scenarios()
# 케이스 11은 온라인·오프라인 일치를 보는 케이스라, 어떤 도착 순서(시드)에서도 max_runs 전에 멈추도록 효과를 키운다
CASES[11]["scenarios"] = _masked_scenarios(mid_before=0.55, mid_after=0.85)


# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------

def _pass_count(arm_spec, n):
    """팔 스펙에서 통과 수를 결정한다. 목표율은 정확한 반올림으로 고정한다."""
    if arm_spec.get("passes") is not None:
        return int(arm_spec["passes"])
    k = round(arm_spec["rate"] * n)
    return int(k)


def generate_case_lines(case_no, seed):
    """케이스 N의 JSONL 줄 목록을 만든다. 같은 (case, seed)면 완전히 결정론적."""
    spec = CASES[case_no]
    rng = random.Random("{0}-case{1}".format(seed, case_no))

    # (시나리오, 팔)별 통과/실패 시퀀스: 통과 수는 고정, 순서만 셔플.
    sequences = {}
    for sc in spec["scenarios"]:
        n = sc["n"]
        for arm in ("before", "after"):
            k = _pass_count(sc[arm], n)
            outcomes = [True] * k + [False] * (n - k)
            rng.shuffle(outcomes)
            sequences[(sc["scenario_id"], arm)] = outcomes

    # 실행 순서 시뮬레이트: 회차마다 시나리오 라운드로빈, before/after 쌍 순서대로.
    records = []
    if spec["scenarios"]:
        n_max = max(sc["n"] for sc in spec["scenarios"])
        for i in range(n_max):
            for sc in spec["scenarios"]:
                if i >= sc["n"]:
                    continue
                for arm in ("before", "after"):
                    records.append({
                        "scenario_id": sc["scenario_id"],
                        "arm": arm,
                        "passed": sequences[(sc["scenario_id"], arm)][i],
                        "synthetic": True,
                    })

    lines = []
    for idx, rec in enumerate(records):
        if case_no == 7 and idx == 6:  # 7번째 줄(1-기준)에서 arm 필드 제거
            rec = {k: v for (k, v) in rec.items() if k != "arm"}
        lines.append(json.dumps(rec, ensure_ascii=False))
    return lines


def build_meta(case_no):
    spec = CASES[case_no]
    meta = {
        "case": case_no,
        "name": spec["name"],
        "kind": spec["kind"],
        "delta": spec["delta"],
        "batch_size": spec["batch_size"],
        "max_runs": spec["max_runs"],
        "error_budget": spec["error_budget"],
        "expected": spec["expected"],
        "notes": spec["notes"],
    }
    if "plan" in spec:
        meta["plan"] = spec["plan"]
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="synth.py",
        description="개선·개악 판별기 검증용 합성 데이터 생성기 (v1 8케이스 + v2 4케이스, 전부 합성 데이터)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", type=int, metavar="N", help="케이스 N의 JSONL을 stdout으로")
    group.add_argument("--meta", type=int, metavar="N", help="케이스 N의 메타 JSON을 stdout으로")
    group.add_argument("--list", action="store_true", help="전 케이스 메타 배열 JSON을 stdout으로")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="S",
                        help="난수 시드 (기본 {0}). 순서만 바뀌고 통과 수는 불변".format(DEFAULT_SEED))
    args = parser.parse_args(argv)

    if args.list:
        metas = [build_meta(n) for n in sorted(CASES)]
        print(json.dumps(metas, ensure_ascii=False, indent=2))
        return 0

    case_no = args.case if args.case is not None else args.meta
    if case_no not in CASES:
        sys.stderr.write("오류: 케이스 번호는 1..{0} 이어야 합니다 (받은 값: {1})\n".format(max(CASES), case_no))
        return 2

    if args.meta is not None:
        print(json.dumps(build_meta(case_no), ensure_ascii=False, indent=2))
        return 0

    # --case
    if CASES[case_no]["kind"] == "calibration":
        sys.stderr.write(
            "케이스 8은 데이터셋이 아니라 캘리브레이션 자기검증입니다.\n"
            "calibrate.py 빠른 모드를 --meta 8 의 설정으로 실행해 조기 종료 경계가\n"
            "error_budget을 실제로 지키는지 확인하세요. (stdout에는 데이터가 없습니다)\n"
        )
        return 0
    if CASES[case_no]["kind"] == "plan":
        sys.stderr.write(
            "케이스 {0}는 데이터셋이 아니라 계획 모드 자기검증입니다.\n"
            "verdict.py --plan 을 --meta {0} 의 plan 설정으로 실행해 검출률이 캘리브레이션 기록과\n"
            "맞는지 확인하세요. (stdout에는 데이터가 없습니다)\n".format(case_no)
        )
        return 0

    out = sys.stdout
    for line in generate_case_lines(case_no, args.seed):
        out.write(line)
        out.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
