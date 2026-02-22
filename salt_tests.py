"""
salt_tests.py
=============
SALT — Spectral-Arithmetic Learning Theory
Full test suite. Run with: python salt_tests.py

All tests are standalone — no pytest required, no external test runner.
Each test prints PASS / FAIL and a brief description.
A summary at the end shows total pass/fail counts.

Dependencies: numpy, scipy (same as salt_core.py)
"""

import sys
import math
import traceback
import numpy as np

# numpy 2.0 renamed np.trapz → np.trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ── Import the library under test ─────────────────────────────────────────────
try:
    from salt_core import (
        gradient_ratio, gradient_change, signal_to_noise,
        cf_convergents, best_convergent, is_farey_neighbor,
        word_step, path_entropy,
        farey_consolidation_index, phase_from_percentile,
        jordan_product, associator, renormalized_gradient,
        sturm_liouville_eigenfunctions, spectral_features, spectral_reconstruct,
        JordanLayerNumpy, CauchyMonitor, SaltAnalyzer, StepResult,
    )
except ImportError as e:
    print(f"FATAL: could not import salt_core: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

_results = []

def test(name: str):
    """Decorator: mark a function as a test case."""
    def decorator(fn):
        _results.append((name, fn))
        return fn
    return decorator

def run_all():
    passed = 0
    failed = 0
    print("\n" + "═"*70)
    print("  SALT TEST SUITE")
    print("═"*70)
    for name, fn in _results:
        try:
            fn()
            print(f"  ✓  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {name}")
            print(f"       AssertionError: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗  {name}")
            print(f"       {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print("─"*70)
    print(f"  {passed} passed   {failed} failed   {passed+failed} total")
    print("═"*70 + "\n")
    return failed == 0


def assert_close(a, b, tol=1e-6, msg=""):
    assert abs(a - b) < tol, f"{msg}  got {a}, expected {b}, diff={abs(a-b):.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1: Scalar Observables
# ─────────────────────────────────────────────────────────────────────────────

@test("gradient_ratio: basic range [0,1]")
def _():
    g1 = np.array([1.0, 0.0, 0.0])
    g2 = np.array([0.0, 1.0, 0.0])
    rho = gradient_ratio(g1, g2)
    assert 0.0 < rho < 1.0, f"rho={rho} not in (0,1)"

@test("gradient_ratio: equal norms → rho = 0.5")
def _():
    g1 = np.array([1.0, 1.0])
    g2 = np.array([1.0, 1.0])
    assert_close(gradient_ratio(g1, g2), 0.5, msg="equal norms:")

@test("gradient_ratio: zero g2 → rho near 0")
def _():
    g1 = np.array([1.0, 0.0])
    g2 = np.zeros(2)
    rho = gradient_ratio(g1, g2)
    assert rho < 1e-9, f"rho={rho}"

@test("gradient_ratio: zero g1 → rho near 1")
def _():
    g1 = np.zeros(2)
    g2 = np.array([1.0, 0.0])
    rho = gradient_ratio(g1, g2)
    assert rho > 1 - 1e-9, f"rho={rho}"

@test("gradient_ratio: invariant to uniform positive scaling of both gradients")
def _():
    g1 = np.array([1.0, 2.0, 3.0])
    g2 = np.array([4.0, 5.0, 6.0])
    rho_base = gradient_ratio(g1, g2)
    # Scaling BOTH gradients by the same positive constant preserves ratio
    rho_scaled = gradient_ratio(g1 * 100, g2 * 100)
    assert_close(rho_base, rho_scaled, tol=1e-10, msg="uniform scaling invariance:")

@test("gradient_ratio: invariant to orthogonal rotation")
def _():
    rng = np.random.default_rng(7)
    g1 = rng.standard_normal(4)
    g2 = rng.standard_normal(4)
    # Random orthogonal matrix via QR
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    rho1 = gradient_ratio(g1, g2)
    rho2 = gradient_ratio(Q @ g1, Q @ g2)
    assert_close(rho1, rho2, tol=1e-10, msg="orthogonal rotation:")

@test("gradient_change: zero when g1=g2")
def _():
    g = np.array([1.0, 2.0, 3.0])
    eps = gradient_change(g, g)
    assert_close(eps, 0.0, tol=1e-12, msg="same gradient:")

@test("gradient_change: range [0,1]")
def _():
    rng = np.random.default_rng(42)
    for _ in range(20):
        g1 = rng.standard_normal(10)
        g2 = rng.standard_normal(10)
        eps = gradient_change(g1, g2)
        assert 0.0 <= eps <= 1.0 + 1e-9, f"eps={eps} out of [0,1]"

@test("signal_to_noise: pure signal → C_α >> 1")
def _():
    # All gradients point in same direction (pure signal)
    g = np.array([1.0, 0.0, 0.0])
    grads = [g * (1 + 0.01 * i) for i in range(30)]
    c = signal_to_noise(grads)
    assert c > 2.0, f"C_α={c:.3f} not >> 1 for pure signal"

@test("signal_to_noise: pure noise → C_α < 1")
def _():
    # Gradients uniformly random → μ_g ≈ 0, signal ≈ 0
    rng = np.random.default_rng(0)
    grads = [rng.standard_normal(50) for _ in range(100)]
    c = signal_to_noise(grads)
    assert c < 0.5, f"C_α={c:.3f} not < 1 for pure noise"

@test("signal_to_noise: single gradient returns 1.0")
def _():
    result = signal_to_noise([np.array([1.0, 2.0])])
    assert_close(result, 1.0, msg="single gradient:")


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2: Continued Fractions and Farey Arithmetic
# ─────────────────────────────────────────────────────────────────────────────

@test("cf_convergents: 1/2 has convergent (1,2)")
def _():
    convs = cf_convergents(0.5, q_max=10)
    assert (1, 2) in convs, f"convergents={convs}"

@test("cf_convergents: 1/3 has convergent (1,3)")
def _():
    convs = cf_convergents(1/3, q_max=10)
    assert (1, 3) in convs, f"convergents={convs}"

@test("cf_convergents: denominators are non-decreasing")
def _():
    convs = cf_convergents(0.618, q_max=50)
    qs = [q for (_, q) in convs]
    assert qs == sorted(qs), f"denominators not sorted: {qs}"

@test("cf_convergents: denominators respect q_max")
def _():
    for q_max in [5, 10, 20, 100]:
        convs = cf_convergents(0.37, q_max=q_max)
        for (_, q) in convs:
            assert q <= q_max, f"q={q} exceeds q_max={q_max}"

@test("cf_convergents: golden ratio gives Fibonacci denominators")
def _():
    phi = (math.sqrt(5) - 1) / 2   # ≈ 0.618...
    convs = cf_convergents(phi, q_max=100)
    qs = [q for (_, q) in convs]
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    for f in fib:
        if f <= 100:
            assert f in qs, f"Fibonacci denominator {f} missing from {qs}"

@test("cf_convergents: approximation error < 1/q² (best approximant property)")
def _():
    x = 0.7182818  # near e-2
    convs = cf_convergents(x, q_max=200)
    for (p, q) in convs:
        err = abs(x - p/q)
        assert err < 1/q**2 + 1e-12, f"p/q={p}/{q}, err={err:.2e}, 1/q²={1/q**2:.2e}"

@test("best_convergent: returns single (p,q) tuple")
def _():
    result = best_convergent(0.42, epsilon=0.05)
    assert isinstance(result, tuple) and len(result) == 2

@test("best_convergent: p/q approximates rho")
def _():
    rho = 0.3571
    eps = 0.02
    p, q = best_convergent(rho, eps)
    assert abs(rho - p/q) < eps, f"approximation error {abs(rho-p/q):.4f} > epsilon {eps}"

@test("is_farey_neighbor: 1/2 and 1/3 are neighbors")
def _():
    # |2·1 - 1·3| = |2-3| = 1 ✓
    assert is_farey_neighbor(1, 2, 1, 3)

@test("is_farey_neighbor: 1/2 and 2/3 are neighbors")
def _():
    # |2·2 - 1·3| = |4-3| = 1 ✓
    assert is_farey_neighbor(1, 2, 2, 3)

@test("is_farey_neighbor: 1/3 and 1/4 are neighbors")
def _():
    assert is_farey_neighbor(1, 3, 1, 4)

@test("is_farey_neighbor: 1/2 and 1/4 are NOT neighbors")
def _():
    # |2·1 - 1·4| = 2 ≠ 1
    assert not is_farey_neighbor(1, 2, 1, 4)

@test("is_farey_neighbor: symmetric — order of args doesn't matter")
def _():
    assert is_farey_neighbor(1, 3, 2, 5) == is_farey_neighbor(2, 5, 1, 3)

@test("farey_neighbors: Stern-Brocot tree property — mediants are neighbors of both parents")
def _():
    # Parents: 1/3 and 2/5 → mediant: 3/8
    assert is_farey_neighbor(1, 3, 3, 8), "mediant not neighbor of left parent"
    assert is_farey_neighbor(3, 8, 2, 5), "mediant not neighbor of right parent"

@test("word_step: increasing rho → R, decreasing → L")
def _():
    assert word_step(0.4, 0.6) == 'R'
    assert word_step(0.6, 0.4) == 'L'

@test("path_entropy: all same letter → entropy 0")
def _():
    h = path_entropy("RRRRRRR")
    assert_close(h, 0.0, tol=1e-10, msg="entropy of constant word:")

@test("path_entropy: equal R and L → entropy 1")
def _():
    h = path_entropy("RLRLRLRL")
    assert_close(h, 1.0, tol=1e-10, msg="entropy of balanced word:")

@test("path_entropy: range [0, 1]")
def _():
    for word in ["R", "L", "RRLL", "RLLRRL", "RLRLRLRL", "RRRRRR"]:
        h = path_entropy(word)
        assert 0.0 <= h <= 1.0 + 1e-12, f"entropy {h} out of [0,1] for word '{word}'"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3: Farey Consolidation Index
# ─────────────────────────────────────────────────────────────────────────────

@test("farey_consolidation_index: structured sequence scores above random")
def _():
    # A perfectly structured Stern-Brocot path: 1/2, 1/3, 2/5, 3/8, ...
    # Each consecutive pair should be Farey neighbors
    structured = [(1,2), (1,3), (2,5), (3,8), (5,13), (8,21), (13,34)]
    f_obs, pct = farey_consolidation_index(structured, n_permutations=200, seed=0)
    assert f_obs > 0.5, f"structured sequence FCI={f_obs:.3f} unexpectedly low"

@test("farey_consolidation_index: returns values in valid range")
def _():
    rng = np.random.default_rng(1)
    seq = [(int(rng.integers(1, 5)), int(rng.integers(2, 10))) for _ in range(20)]
    f_obs, pct = farey_consolidation_index(seq)
    assert 0.0 <= f_obs <= 1.0, f"f_obs={f_obs}"
    assert 0.0 <= pct <= 100.0, f"pct={pct}"

@test("phase_from_percentile: covers all five phases")
def _():
    phases = {
        phase_from_percentile(30),
        phase_from_percentile(65),
        phase_from_percentile(87),
        phase_from_percentile(97),
        phase_from_percentile(99.5),
    }
    expected = {"MEMORIZATION", "APPROACHING", "CRITICAL", "GENERALIZING", "CONVERGED"}
    assert phases == expected, f"phases={phases}"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4: Jordan Algebra
# ─────────────────────────────────────────────────────────────────────────────

@test("jordan_product: commutative X∘W = W∘X")
def _():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((4, 4))
    W = rng.standard_normal((4, 4))
    XoW = jordan_product(X, W)
    WoX = jordan_product(W, X)
    assert np.allclose(XoW, WoX, atol=1e-12), "Jordan product not commutative"

@test("jordan_product: identity matrices → XoI = X")
def _():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((4, 4))
    I = np.eye(4)
    XoI = jordan_product(X, I)
    assert np.allclose(XoI, X, atol=1e-12), "X∘I should equal X"

@test("jordan_product: symmetric matrices → result is symmetric")
def _():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((4, 4))
    X = X + X.T                      # make symmetric
    W = rng.standard_normal((4, 4))
    W = W + W.T
    result = jordan_product(X, W)
    assert np.allclose(result, result.T, atol=1e-12), "symmetric∘symmetric should be symmetric"

@test("associator: non-zero for generic matrices (non-associativity)")
def _():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((4, 4))
    Y = rng.standard_normal((4, 4))
    Z = rng.standard_normal((4, 4))
    A = associator(X, Y, Z)
    # Should be non-zero (with probability 1 for random matrices)
    assert np.linalg.norm(A) > 1e-10, "associator is zero — algebra appears associative"

@test("associator: zero for commuting matrices (diagonal)")
def _():
    # Diagonal matrices commute → Jordan product reduces to ordinary product
    D1 = np.diag([1.0, 2.0, 3.0, 4.0])
    D2 = np.diag([5.0, 6.0, 7.0, 8.0])
    D3 = np.diag([9.0, 10.0, 11.0, 12.0])
    A = associator(D1, D2, D3)
    assert np.allclose(A, 0, atol=1e-12), f"associator non-zero for commuting matrices: {np.linalg.norm(A)}"

@test("renormalized_gradient: norm bounded by 1/alpha")
def _():
    rng = np.random.default_rng(2)
    for alpha in [0.1, 0.5, 1.0, 2.0, 10.0]:
        for _ in range(10):
            g = rng.standard_normal(20) * 100     # large gradient
            g_renorm = renormalized_gradient(g, alpha=alpha)
            norm = np.linalg.norm(g_renorm)
            assert norm <= 1/alpha + 1e-10, f"‖∇̃L‖={norm:.4f} > 1/α={1/alpha:.4f} for α={alpha}"

@test("renormalized_gradient: small gradients nearly unchanged")
def _():
    g = np.array([0.001, 0.001, 0.001])
    g_renorm = renormalized_gradient(g, alpha=1.0)
    # For small g, 1+α‖g‖ ≈ 1, so ∇̃L ≈ ∇L
    assert np.allclose(g, g_renorm, atol=0.01), "small gradient should be nearly unchanged"

@test("renormalized_gradient: direction preserved")
def _():
    rng = np.random.default_rng(4)
    g = rng.standard_normal(10)
    g_renorm = renormalized_gradient(g, alpha=1.0)
    # Cosine similarity should be 1.0 (same direction)
    cos = np.dot(g, g_renorm) / (np.linalg.norm(g) * np.linalg.norm(g_renorm))
    assert_close(cos, 1.0, tol=1e-10, msg="direction should be preserved:")


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5: Jordan Layer (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

@test("JordanLayerNumpy: output shape matches input")
def _():
    layer = JordanLayerNumpy(dim=8)
    X = np.random.standard_normal((8, 8))
    out = layer.forward(X)
    assert out.shape == (8, 8), f"shape mismatch: {out.shape}"

@test("JordanLayerNumpy: output is symmetric when input is symmetric")
def _():
    layer = JordanLayerNumpy(dim=6)
    X = np.random.standard_normal((6, 6))
    X = X + X.T
    W_sym = layer.W + layer.W.T
    layer.W = W_sym
    out = layer.forward(X)
    assert np.allclose(out, out.T, atol=1e-12), "symmetric input/weight → symmetric output"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6: Cauchy Monitor
# ─────────────────────────────────────────────────────────────────────────────

@test("CauchyMonitor: converging sequence is detected as Cauchy")
def _():
    monitor = CauchyMonitor(tol=1e-3, window=20)
    theta = np.array([10.0, 10.0])
    for i in range(1, 200):
        theta = theta * 0.95 + np.array([0.0, 0.0])
        monitor.update(theta)
    assert monitor.is_cauchy(), "converging sequence should be Cauchy"

@test("CauchyMonitor: oscillating sequence is NOT Cauchy")
def _():
    monitor = CauchyMonitor(tol=1e-3, window=20)
    theta = np.array([0.0])
    for i in range(50):
        theta = -theta + np.array([(-1)**i * 1.0])
        monitor.update(theta)
    assert not monitor.is_cauchy(), "oscillating sequence should not be Cauchy"

@test("CauchyMonitor: trend detection")
def _():
    monitor = CauchyMonitor(window=20)
    theta = np.array([100.0])
    for i in range(30):
        theta = theta * 0.9
        monitor.update(theta)
    assert monitor.trend() == "decreasing", f"expected 'decreasing', got '{monitor.trend()}'"

@test("CauchyMonitor: summary has required keys")
def _():
    monitor = CauchyMonitor()
    monitor.update(np.array([1.0]))
    monitor.update(np.array([0.9]))
    summary = monitor.summary()
    for key in ("is_cauchy", "trend", "mean_diff", "max_diff"):
        assert key in summary, f"missing key '{key}'"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7: Spectral Basis
# ─────────────────────────────────────────────────────────────────────────────

@test("sturm_liouville_eigenfunctions: returns correct shapes")
def _():
    x, psi = sturm_liouville_eigenfunctions(n_modes=5, n_points=100)
    assert x.shape == (100,), f"x.shape={x.shape}"
    assert psi.shape == (100, 5), f"psi.shape={psi.shape}"

@test("sturm_liouville_eigenfunctions: eigenfunctions are L²-normalized")
def _():
    x, psi = sturm_liouville_eigenfunctions(n_modes=8, n_points=256)
    for k in range(psi.shape[1]):
        norm_sq = _trapz(psi[:, k] ** 2, x)
        assert_close(norm_sq, 1.0, tol=1e-3, msg=f"eigenfunction {k} not normalized:")

@test("sturm_liouville_eigenfunctions: eigenfunctions are mutually orthogonal")
def _():
    x, psi = sturm_liouville_eigenfunctions(n_modes=5, n_points=256)
    for i in range(psi.shape[1]):
        for j in range(i + 1, psi.shape[1]):
            inner = _trapz(psi[:, i] * psi[:, j], x)
            assert abs(inner) < 1e-2, f"modes {i},{j} not orthogonal: inner={inner:.4f}"

@test("spectral_features: coefficients have correct length")
def _():
    x, psi = sturm_liouville_eigenfunctions(n_modes=6, n_points=100)
    data = np.sin(np.pi * x)
    coeffs = spectral_features(data, psi, x)
    assert len(coeffs) == 6, f"len(coeffs)={len(coeffs)}"

@test("spectral_reconstruct: reconstruction approximates original")
def _():
    x, psi = sturm_liouville_eigenfunctions(n_modes=20, n_points=256)
    data = np.sin(np.pi * x)                      # first eigenfunction (p=1, q=0 basis)
    coeffs = spectral_features(data, psi, x)
    recon = spectral_reconstruct(coeffs, psi)
    err = np.sqrt(_trapz((data - recon)**2, x))
    assert err < 0.05, f"reconstruction L² error = {err:.4f} too large"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 8: Main SaltAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

def _make_signal_gradients(n=80, seed=0):
    """Simulate gradients from a converging model (signal-dominated).
    Gradients consistently point in a dominant direction with small noise."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(20)
    direction /= np.linalg.norm(direction)
    grads = []
    for i in range(n):
        # Strong consistent signal + small noise
        g = direction * 2.0 + rng.standard_normal(20) * 0.1
        grads.append(g)
    return grads

def _make_noise_gradients(n=80, seed=1):
    """Simulate gradients from a model memorizing (noise-dominated)."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(20) for _ in range(n)]

@test("SaltAnalyzer: step returns StepResult")
def _():
    analyzer = SaltAnalyzer(window=20)
    grads = _make_signal_gradients(n=10)
    result = analyzer.step(grads[0], grads[1])
    assert isinstance(result, StepResult)

@test("SaltAnalyzer: StepResult fields are in valid ranges")
def _():
    analyzer = SaltAnalyzer(window=30)
    grads = _make_signal_gradients(n=50)
    for i in range(len(grads) - 1):
        r = analyzer.step(grads[i], grads[i+1])
        assert 0.0 < r.rho < 1.0,            f"rho={r.rho}"
        assert 0.0 <= r.epsilon <= 1.0,       f"eps={r.epsilon}"
        assert r.convergent[1] >= 1,          f"q={r.convergent[1]}"
        assert r.word_letter in ('R', 'L'),   f"letter={r.word_letter}"
        assert r.c_alpha >= 0,                f"c_alpha={r.c_alpha}"
        assert r.q_star >= 1,                 f"q*={r.q_star}"
        assert 0.0 <= r.f_c_pct <= 100.0,    f"pct={r.f_c_pct}"
        assert 0.0 <= r.path_entropy <= 1.0, f"H={r.path_entropy}"
        assert r.phase in {
            "MEMORIZATION", "APPROACHING", "CRITICAL",
            "GENERALIZING", "CONVERGED"
        },                                    f"phase={r.phase}"

@test("SaltAnalyzer: step count increments correctly")
def _():
    analyzer = SaltAnalyzer(window=20)
    grads = _make_signal_gradients(n=15)
    for i in range(len(grads) - 1):
        analyzer.step(grads[i], grads[i+1])
    assert analyzer.steps == len(grads) - 1, f"steps={analyzer.steps}"

@test("SaltAnalyzer: reset clears state")
def _():
    analyzer = SaltAnalyzer(window=20)
    grads = _make_signal_gradients(n=30)
    for i in range(len(grads) - 1):
        analyzer.step(grads[i], grads[i+1])
    assert analyzer.steps > 0
    analyzer.reset()
    assert analyzer.steps == 0, "steps should be 0 after reset"

@test("SaltAnalyzer: signal-dominated gradients → C_α > 1 eventually")
def _():
    analyzer = SaltAnalyzer(window=40)
    grads = _make_signal_gradients(n=80, seed=10)
    results = [analyzer.step(grads[i], grads[i+1]) for i in range(len(grads)-1)]
    final_c = results[-1].c_alpha
    assert final_c > 1.0, f"final C_α={final_c:.3f} not > 1 for signal-dominated gradients"

@test("SaltAnalyzer: noise-dominated gradients → C_α < 1")
def _():
    analyzer = SaltAnalyzer(window=40)
    grads = _make_noise_gradients(n=80, seed=20)
    results = [analyzer.step(grads[i], grads[i+1]) for i in range(len(grads)-1)]
    final_c = results[-1].c_alpha
    assert final_c < 1.0, f"final C_α={final_c:.3f} not < 1 for noise-dominated gradients"

@test("SaltAnalyzer: word grows monotonically")
def _():
    analyzer = SaltAnalyzer(window=20)
    grads = _make_signal_gradients(n=20)
    prev_len = 0
    for i in range(len(grads) - 1):
        r = analyzer.step(grads[i], grads[i+1])
        assert len(r.word) == prev_len + 1, "word should grow by 1 each step"
        prev_len = len(r.word)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 9: Mathematical Theorem Verification
# ─────────────────────────────────────────────────────────────────────────────

@test("THEOREM — Hurwitz bound: |ρ - p/q| < 1/√5·q² for golden ratio")
def _():
    """
    Hurwitz 1891: for any irrational x, infinitely many convergents p/q satisfy
    |x - p/q| < 1/(√5·q²). The constant 1/√5 is sharp (golden ratio is hardest).
    """
    phi = (math.sqrt(5) - 1) / 2   # golden ratio ≈ 0.618...
    convs = cf_convergents(phi, q_max=200)
    hurwitz_satisfied = 0
    for p, q in convs:
        err = abs(phi - p/q)
        if err < 1 / (math.sqrt(5) * q**2):
            hurwitz_satisfied += 1
    # At least half of convergents should satisfy Hurwitz
    assert hurwitz_satisfied >= len(convs) // 2, \
        f"only {hurwitz_satisfied}/{len(convs)} satisfy Hurwitz"

@test("THEOREM — Jordan commutativity: X∘W = W∘X")
def _():
    """Proved: Jordan product is commutative by definition."""
    rng = np.random.default_rng(99)
    for _ in range(10):
        X = rng.standard_normal((5, 5))
        W = rng.standard_normal((5, 5))
        assert np.allclose(jordan_product(X, W), jordan_product(W, X), atol=1e-12)

@test("THEOREM — Gradient renorm bound: ‖∇̃L‖ ≤ 1/α")
def _():
    """Theorem 9 in SALT: renormalized gradient is uniformly bounded by 1/α."""
    rng = np.random.default_rng(88)
    for alpha in np.linspace(0.1, 5.0, 20):
        g = rng.standard_normal(100) * rng.uniform(0.1, 100)
        g_r = renormalized_gradient(g, alpha=float(alpha))
        assert np.linalg.norm(g_r) <= 1/alpha + 1e-10

@test("THEOREM — Farey adjacency: mediants of neighbors are neighbors of both")
def _():
    """
    If a/b and c/d are Farey neighbors (|bc-ad|=1),
    then their mediant (a+c)/(b+d) is a neighbor of both.
    """
    test_pairs = [(1,3,1,2), (2,5,3,7), (1,4,2,7), (3,5,4,7)]
    for (a, b, c, d) in test_pairs:
        if not is_farey_neighbor(a, b, c, d):
            continue  # skip if not actually neighbors
        med_p, med_q = a+c, b+d
        assert is_farey_neighbor(a, b, med_p, med_q), \
            f"{a}/{b} and {med_p}/{med_q} not neighbors"
        assert is_farey_neighbor(med_p, med_q, c, d), \
            f"{med_p}/{med_q} and {c}/{d} not neighbors"

@test("THEOREM — SL(2,Z) generators have unit determinant")
def _():
    """L and R are in SL(2,ℤ) — verified algebraically."""
    L = np.array([[1, 0], [1, 1]])
    R = np.array([[1, 1], [0, 1]])
    assert_close(float(np.linalg.det(L)), 1.0, msg="det(L):")
    assert_close(float(np.linalg.det(R)), 1.0, msg="det(R):")

@test("THEOREM — L²-orthonormality of S-L eigenfunctions")
def _():
    """Self-adjoint S-L operators have orthonormal eigenfunctions (Theorem 6)."""
    x, psi = sturm_liouville_eigenfunctions(n_modes=6, n_points=512)
    G = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            G[i, j] = _trapz(psi[:, i] * psi[:, j], x)
    # G should be ≈ identity
    assert np.allclose(G, np.eye(6), atol=5e-3), \
        f"Gram matrix not close to identity:\n{G}"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
