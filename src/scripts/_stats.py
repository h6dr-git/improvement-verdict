#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""공용 수치 코어 — Beta 사후분포 기반 개선/개악 판정의 원시 함수 (표준 라이브러리만, Python 3.9+).

사용 예시: python3 -c "import _stats; print(_stats.prob_diff(62, 100, 75, 100, 0.03))"

- posterior():   통과 k / 실행 n -> Jeffreys 사후분포 Beta(k+0.5, n-k+0.5)  (SPEC §6)
- betainc():     정규화 불완전 베타 I_x(a,b), Lentz 연속분수 (스팟체크·검증용)
- prob_diff():   diff = p_after - p_before 의 ROPE 확률 3종을 결정론적 심프슨 적분으로 계산
- decide():      SPEC §6 판정 규칙. verdict.py 와 calibrate.py 가 반드시 이 함수를 공유한다 —
                 판정 경로가 갈라지면 캘리브레이션이 무의미해진다.
- 판정에 정규근사(z검정)를 쓰지 않는다(SPEC §2(b), §11-1).
  정규근사는 표본계획 근사(needed_runs)에만 쓰고, 호출측이 "근사"임을 명시해 출력한다.
"""
import math
import random
import statistics
from bisect import bisect_right
from functools import lru_cache

_NEG_HUGE = -1e308  # log(0) 대용. 계수 0과 곱하면 -0.0, 양수 계수와 곱하면 -inf -> exp()=0


def log_beta(a, b):
    """log B(a,b) — math.lgamma 기반."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a, b, x):
    """불완전 베타의 연속분수(Lentz 변형). Numerical Recipes 방식."""
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            return h
    # 수렴 실패 안전장치: 발산 대신 마지막 근사값을 반환한다(호출측에서 [0,1] 클램프).
    return h


def betainc(a, b, x):
    """정규화 불완전 베타 I_x(a,b). x가 (a+1)/(a+b+2)보다 크면 대칭식으로 전환."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = a * math.log(x) + b * math.log1p(-x) - log_beta(a, b)
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        v = bt * _betacf(a, b, x) / a
    else:
        v = 1.0 - bt * _betacf(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, v))


def posterior(k, n):
    """이항 관측 (통과 k / 실행 n) -> Jeffreys 사후분포 Beta(k+0.5, n-k+0.5)."""
    if not (0 <= k <= n):
        raise ValueError("posterior: 0 <= k <= n 이어야 함 (k=%r, n=%r)" % (k, n))
    return (k + 0.5, n - k + 0.5)


@lru_cache(maxsize=8)
def _grid(nseg):
    """x = sin^2(theta) 치환 격자. Jeffreys 사후의 끝점 특이점(적분 가능)을 유계로 만든다.

    반환: (dt, x[0..N], logsin, logcos, 심프슨 가중치 w) — 모두 길이 N+1.
    """
    dt = (math.pi / 2.0) / nseg
    x, logsin, logcos, w = [], [], [], []
    for j in range(nseg + 1):
        t = dt * j
        s, c = math.sin(t), math.cos(t)
        x.append(s * s)
        logsin.append(math.log(s) if s > 0.0 else _NEG_HUGE)
        logcos.append(math.log(c) if c > 0.0 else _NEG_HUGE)
        if j == 0 or j == nseg:
            w.append(dt / 3.0)
        elif j % 2 == 1:
            w.append(4.0 * dt / 3.0)
        else:
            w.append(2.0 * dt / 3.0)
    x[0], x[-1] = 0.0, 1.0
    return dt, x, logsin, logcos, w


def _nseg_for(n_max):
    """표본이 클수록 사후분포가 좁아지므로 격자를 촘촘하게. 최소 800구간(801점, SPEC 하한)."""
    if n_max <= 1000:
        return 800
    if n_max <= 10000:
        return 1600
    return 3200


def _g_values(a, b, nseg):
    """치환 후 밀도 g(theta) = f_Beta(sin^2 t; a,b) * sin(2t) = 2 sin^(2a-1) cos^(2b-1) / B(a,b).

    a,b >= 0.5 (Jeffreys)이면 지수 계수가 0 이상이라 전 구간 유계다.
    """
    _, _, logsin, logcos, _ = _grid(nseg)
    c1, c2 = 2.0 * a - 1.0, 2.0 * b - 1.0
    lc = math.log(2.0) - log_beta(a, b)
    exp = math.exp
    out = []
    for ls, lcz in zip(logsin, logcos):
        e = c1 * ls + c2 * lcz + lc
        out.append(exp(e) if e > -700.0 else 0.0)
    return out


def _cdf_table(g, nseg):
    """g(theta)의 누적 사다리꼴 -> [0,1]로 정규화한 CDF 표 (상수 계수는 정규화로 소거)."""
    F = [0.0] * (nseg + 1)
    acc = 0.0
    prev = g[0]
    for j in range(1, nseg + 1):
        cur = g[j]
        acc += prev + cur
        prev = cur
        F[j] = acc
    tot = F[-1]
    if tot <= 0.0:
        return F
    inv = 1.0 / tot
    for j in range(nseg + 1):
        F[j] *= inv
    F[-1] = 1.0
    return F


def _cdf_shifted(x, F, shift):
    """모든 격자점 x_j 에 대해 F(x_j + shift)를 선형 보간으로 구한다(질의가 오름차순 -> 한 번의 병합 스캔)."""
    n = len(x)
    out = [0.0] * n
    i = 0
    for j in range(n):
        q = x[j] + shift
        if q <= 0.0:
            continue
        if q >= 1.0:
            out[j] = 1.0
            continue
        while i + 1 < n and x[i + 1] <= q:
            i += 1
        x0, x1 = x[i], x[i + 1]
        t = (q - x0) / (x1 - x0) if x1 > x0 else 0.0
        out[j] = F[i] + t * (F[i + 1] - F[i])
    return out


@lru_cache(maxsize=None)
def prob_diff(k_b, n_b, k_a, n_a, delta):
    """diff = p_after - p_before 에 대해 (P(diff > +delta), P(diff < -delta), P(|diff| <= delta)).

    결정론적 계산(샘플링 아님): 심프슨 적분 한 패스로
      p_improved  = ∫ f_after(x) * F_before(x - delta) dx
      p_regressed = ∫ f_after(x) * (1 - F_before(x + delta)) dx
    를 동시에 구한다. F_before 는 같은 격자의 누적 적분표(선형 보간)로 평가하며,
    이 경로는 Lentz betainc() 및 몬테카를로와 교차 검증되어 있다(오차 << 0.01).
    lru_cache 필수: 캘리브레이션이 같은 (k,n) 조합을 대량 반복 호출한다.
    """
    a_b, b_b = posterior(k_b, n_b)
    a_a, b_a = posterior(k_a, n_a)
    nseg = _nseg_for(max(n_b, n_a))
    _, x, _, _, w = _grid(nseg)
    g_b = _g_values(a_b, b_b, nseg)
    g_a = _g_values(a_a, b_a, nseg)
    F_b = _cdf_table(g_b, nseg)
    if delta == 0.0:
        Fm = F_b
        Fp = F_b
    else:
        Fm = _cdf_shifted(x, F_b, -delta)
        Fp = _cdf_shifted(x, F_b, +delta)
    s_tot = s_imp = s_reg = 0.0
    for wj, gj, fm, fp in zip(w, g_a, Fm, Fp):
        wg = wj * gj
        s_tot += wg
        s_imp += wg * fm
        s_reg += wg * (1.0 - fp)
    if s_tot <= 0.0:
        return (0.0, 0.0, 1.0)
    p_imp = min(1.0, max(0.0, s_imp / s_tot))
    p_reg = min(1.0, max(0.0, s_reg / s_tot))
    p_rope = max(0.0, 1.0 - p_imp - p_reg)
    return (p_imp, p_reg, p_rope)


def decide_from_probs(p_imp, p_reg, p_rope, threshold, error_budget):
    """SPEC §6 판정 규칙. improved/regressed 는 캘리브레이션된 문턱, no_difference 는 1-error_budget.

    no_difference(delta 안임을 확신)와 undetermined(아직 모름)는 절대 합치지 않는다(SPEC §11-2).
    """
    if p_imp >= threshold:
        return "improved"
    if p_reg >= threshold:
        return "regressed"
    if p_rope >= 1.0 - error_budget:
        return "no_difference"
    return "undetermined"


def decide(k_b, n_b, k_a, n_a, delta, threshold, error_budget):
    """관측 (k,n) 쌍 -> (verdict, p_improved, p_regressed, p_rope). verdict/calibrate 공용 판정 함수."""
    p_imp, p_reg, p_rope = prob_diff(k_b, n_b, k_a, n_a, delta)
    return decide_from_probs(p_imp, p_reg, p_rope, threshold, error_budget), p_imp, p_reg, p_rope


# ---------- 이항 표본 추출 (노이즈 바닥·캘리브레이션 공용, 시드 고정 결정론) ----------

def binom_cdf_table(n, p):
    """Binomial(n,p)의 CDF 표. 추출은 bisect 한 번 — 대량 시뮬레이션용."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("binom_cdf_table: p 범위 오류")
    lp = math.log(p) if p > 0.0 else _NEG_HUGE
    lq = math.log1p(-p) if p < 1.0 else _NEG_HUGE
    lg = math.lgamma
    cdf = []
    acc = 0.0
    for k in range(n + 1):
        e = lg(n + 1) - lg(k + 1) - lg(n - k + 1) + k * lp + (n - k) * lq
        acc += math.exp(e) if e > -700.0 else 0.0
        cdf.append(acc)
    cdf[-1] = 1.0
    return cdf


def binom_draw(rng, cdf):
    """CDF 표에서 이항 표본 하나 추출."""
    return bisect_right(cdf, rng.random())


def aa_noise_floor(base, n_per_arm, sims=4000, seed=20260710):
    """A/A(양팔 동일 base)에서 우연만으로 벌어지는 |통과율 차이|의 95% 분위 — delta 하한 제안 근거.

    시드 고정 이항 몬테카를로. 판정이 아니라 제안용이다.
    """
    n = max(1, int(n_per_arm))
    cdf = binom_cdf_table(n, base)
    rng = random.Random(seed)
    diffs = sorted(abs(binom_draw(rng, cdf) - binom_draw(rng, cdf)) / n for _ in range(sims))
    idx = min(sims - 1, max(0, int(math.ceil(0.95 * sims)) - 1))
    return diffs[idx]


def needed_runs(p_b, p_a, delta, error_budget, power=0.8):
    """결론까지 필요한 팔당 실행 수의 **정규근사** 표본계획 (판정에는 쓰지 않는다 — SPEC §11-1).

    관측 통과율 고정 가정. |diff|>delta 면 우월성(경계 delta와의 간격), 아니면 동등성(ROPE 폭과의 간격) 기준.
    반환 None = 관측 차이가 delta 경계에 너무 가까워 유한한 계획이 안 나옴.
    """
    nd = statistics.NormalDist()
    z1 = nd.inv_cdf(1.0 - error_budget)
    z2 = nd.inv_cdf(power)
    eff = abs(p_a - p_b)
    gap = (eff - delta) if eff > delta else (delta - eff)
    if gap < 1e-4:
        return None
    var = max(p_b * (1.0 - p_b) + p_a * (1.0 - p_a), 0.01)
    n = int(math.ceil(((z1 + z2) ** 2) * var / (gap * gap)))
    return max(1, n)
