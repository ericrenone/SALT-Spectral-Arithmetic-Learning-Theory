"""
salt_core.py
============
SALT — Spectral-Arithmetic Learning Theory
Core library: all observables, diagnostics, and the Jordan layer.

Dependencies: numpy, scipy
Optional:     torch (for JordanLayer as an nn.Module)

Usage:
    from salt_core import SaltAnalyzer, JordanLayerNumpy
    analyzer = SaltAnalyzer(window=50)
    result = analyzer.step(g_prev, g_curr)
    print(result["phase"], result["c_alpha"])
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import percentileofscore

# numpy 2.0 renamed np.trapz → np.trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Scalar Observables
# ─────────────────────────────────────────────────────────────────────────────

def gradient_ratio(g1: np.ndarray, g2: np.ndarray) -> float:
    """
    ρ = ‖g2‖ / (‖g1‖ + ‖g2‖)  ∈ (0, 1)

    The fundamental SALT observable. Encodes the dominant Hessian curvature
    via a Möbius transformation near a critical point.

    Returns 0.5 if both gradients are zero (degenerate case).
    """
    n1 = np.linalg.norm(g1)
    n2 = np.linalg.norm(g2)
    total = n1 + n2
    return float(n2 / total) if total > 1e-12 else 0.5


def gradient_change(g1: np.ndarray, g2: np.ndarray) -> float:
    """
    ε = ‖g2 - g1‖ / (‖g1‖ + ‖g2‖)  ∈ [0, 1]

    Sets the data-adaptive resolution Q_max = floor(1/ε).
    By the Hurwitz theorem, the CF convergent at this resolution
    has approximation error below the gradient measurement noise.
    """
    diff = np.linalg.norm(g2 - g1)
    total = np.linalg.norm(g1) + np.linalg.norm(g2)
    return float(diff / total) if total > 1e-12 else 1.0


def signal_to_noise(gradients: List[np.ndarray]) -> float:
    """
    C_α = ‖μ_g‖² / Tr(Σ_g)

    The primary generalization phase indicator.
      C_α > 1  →  signal dominates  →  generalization (λ₁ > 0)
      C_α = 1  →  critical boundary →  grokking / double-descent
      C_α < 1  →  noise dominates   →  memorization (λ₁ < 0)

    Requires at least 2 gradient vectors.
    """
    if len(gradients) < 2:
        return 1.0
    G = np.stack([g.ravel() for g in gradients])          # shape (T, d)
    mu = G.mean(axis=0)
    centered = G - mu
    noise_trace = float(np.mean(np.sum(centered ** 2, axis=1)))
    signal = float(np.dot(mu, mu))
    return signal / (noise_trace + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Continued Fractions and Farey Arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def cf_convergents(x: float, q_max: int) -> List[Tuple[int, int]]:
    """
    Compute continued-fraction convergents of x ∈ (0, 1) with denominator ≤ q_max.

    Uses the standard three-term recurrence:
        p_k = a_k * p_{k-1} + p_{k-2},  seed: p_{-1}=1, p_0=a_0
        q_k = a_k * q_{k-1} + q_{k-2},  seed: q_{-1}=0, q_0=1

    Each convergent p_k/q_k is the best rational approximant with that denominator
    (Lagrange 1770). Returns a list of (p, q) pairs in order.
    """
    convergents: List[Tuple[int, int]] = []
    p_prev, p_curr = 1, int(x)
    q_prev, q_curr = 0, 1
    xi = float(x)

    convergents.append((p_curr, q_curr))

    for _ in range(60):                      # 60 iterations more than enough
        remainder = xi - math.floor(xi)
        if remainder < 1e-13:
            break
        xi = 1.0 / remainder
        a_k = int(xi)
        p_next = a_k * p_curr + p_prev
        q_next = a_k * q_curr + q_prev
        if q_next > q_max:
            break
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next
        convergents.append((p_curr, q_curr))

    return convergents


def best_convergent(rho: float, epsilon: float) -> Tuple[int, int]:
    """
    Adaptive CF approximation of ρ at data-driven resolution Q_max = ⌊1/ε⌋.

    Hurwitz guarantee: approximation error < ε² / √5, which is below
    the gradient measurement noise level ε. No finer arithmetic structure
    is imposed than the data supports.
    """
    q_max = max(2, int(1.0 / max(epsilon, 1e-6)))
    convs = cf_convergents(rho, q_max)
    return convs[-1]


def is_farey_neighbor(p1: int, q1: int, p2: int, q2: int) -> bool:
    """
    True iff |q1·p2 - p1·q2| = 1   (Farey unimodular condition, Cauchy 1816).

    Two fractions satisfying this condition are adjacent in some Farey sequence,
    and their Ford circles are externally tangent. In SALT: consecutive training
    steps satisfying this are arithmetically "adjacent" — a signal of structure.
    """
    return abs(q1 * p2 - p1 * q2) == 1


def word_step(rho_prev: float, rho_curr: float) -> str:
    """
    Determine the SL(2,ℤ) monoid step: 'R' if gradient norm increased, 'L' otherwise.

    R = [[1,1],[0,1]] (right generator), L = [[1,0],[1,1]] (left generator).
    The sequence of R/L letters is the word in the free monoid ℳ encoding
    the arithmetic state of the training run.
    """
    return 'R' if rho_curr > rho_prev else 'L'


def path_entropy(word: str) -> float:
    """
    Shannon entropy of the L/R binary string.

    H ≈ 0  →  persistent dominance (signal or noise, learning or stuck)
    H ≈ 1  →  equal mixing (oscillation, plateau)
    """
    if len(word) < 2:
        return 1.0
    p_r = word.count('R') / len(word)
    p_l = 1.0 - p_r
    h = 0.0
    for p in (p_r, p_l):
        if p > 0:
            h -= p * math.log2(p)
    return h


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Farey Consolidation Index with Permutation Null
# ─────────────────────────────────────────────────────────────────────────────

def _unimodular_fraction(seq: List[Tuple[int, int]]) -> float:
    """Fraction of consecutive pairs that are Farey neighbors."""
    if len(seq) < 2:
        return 0.0
    hits = sum(
        is_farey_neighbor(seq[i][0], seq[i][1], seq[i+1][0], seq[i+1][1])
        for i in range(len(seq) - 1)
    )
    return hits / (len(seq) - 1)


def farey_consolidation_index(
    convergents: List[Tuple[int, int]],
    n_permutations: int = 200,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute the Farey Consolidation Index and its permutation-null percentile.

    Returns:
        (f_c_obs, percentile)
        f_c_obs    — fraction of consecutive Farey-neighbor pairs in the sequence
        percentile — rank of f_c_obs vs 200 random permutations (0–100)

    Why permutation null? During training, gradients cluster, so the background
    rate of unimodular pairs is higher than the analytic Farey-sequence rate.
    Permuting destroys temporal order while preserving the marginal distribution,
    giving the correct baseline.
    """
    f_obs = _unimodular_fraction(convergents)
    rng = np.random.default_rng(seed=seed)
    null = []
    arr = list(convergents)
    for _ in range(n_permutations):
        rng.shuffle(arr)
        null.append(_unimodular_fraction(arr))
    pct = float(percentileofscore(null, f_obs, kind='strict'))
    return f_obs, pct


def phase_from_c_alpha(c_alpha: float, f_c_pct: float) -> str:
    """
    Determine training phase from the two primary SALT signals.

    C_α (signal-to-noise) is the primary indicator:
      C_α > 1.1  →  GENERALIZING / CONVERGED (signal dominates)
      C_α ~ 1.0  →  CRITICAL (grokking boundary / double-descent)
      C_α < 0.9  →  MEMORIZATION (noise dominates)

    The FCI percentile is a corroborating secondary signal.
    """
    if c_alpha > 2.0 and f_c_pct >= 60:
        return "CONVERGED"
    elif c_alpha > 1.1:
        return "GENERALIZING"
    elif c_alpha > 0.9:
        return "CRITICAL"
    elif f_c_pct >= 80:
        return "APPROACHING"
    else:
        return "MEMORIZATION"


def phase_from_percentile(pct: float) -> str:
    """Map FCI percentile to a training phase label (secondary signal)."""
    if pct < 50:
        return "MEMORIZATION"
    elif pct < 80:
        return "APPROACHING"
    elif pct < 95:
        return "CRITICAL"
    elif pct < 99:
        return "GENERALIZING"
    else:
        return "CONVERGED"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Jordan Product Operations (NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def jordan_product(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Jordan symmetrized product: X ∘ W = ½(XW + WX)

    Properties:
      - Commutative:     X ∘ W = W ∘ X
      - Not associative: (X∘Y)∘Z ≠ X∘(Y∘Z) in general
      - Bounded if X, W are bounded operators

    In the SALT architecture this replaces standard matrix multiplication,
    enforcing symmetric bilinear interaction and reducing directional bias.
    """
    return 0.5 * (X @ W + W @ X)


def associator(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    A(X, Y, Z) = (X∘Y)∘Z − X∘(Y∘Z)

    Non-zero in general — the algebra remembers computation order.
    Two update paths reaching the same weight matrix via different orderings
    produce different associators. Standard linear layers cannot distinguish them.
    """
    return jordan_product(jordan_product(X, Y), Z) - jordan_product(X, jordan_product(Y, Z))


def renormalized_gradient(grad: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Renormalized gradient: ∇̃L = ∇L / (1 + α‖∇L‖)

    Theorem: ‖∇̃L‖ ≤ 1/α  uniformly.
    Proof:   g/(1+αg) ≤ 1/α  for all g ≥ 0.

    This is gradient clipping derived from operator theory.
    The bound 1/α is tight (achieved as ‖∇L‖ → ∞).
    """
    g_norm = np.linalg.norm(grad)
    return grad / (1.0 + alpha * g_norm)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Optional PyTorch Jordan Layer
# ─────────────────────────────────────────────────────────────────────────────

class JordanLayerNumpy:
    """
    NumPy implementation of the SALT Jordan interaction layer.

    Replaces standard Wx with the symmetrized Jordan product:
        output = ½(XW + WX)

    Parameters
    ----------
    dim : int
        Feature dimension (square weight matrix W ∈ ℝ^{dim×dim}).
    init_scale : float
        Standard deviation for weight initialization.
    """

    def __init__(self, dim: int, init_scale: float = 0.1):
        self.dim = dim
        rng = np.random.default_rng(seed=0)
        self.W = rng.normal(scale=init_scale, size=(dim, dim))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply Jordan product: X ∘ W = ½(XW + WX)."""
        return jordan_product(X, self.W)

    def update_weights(self, grad_W: np.ndarray, lr: float = 0.01) -> None:
        """Simple gradient step on weights."""
        self.W -= lr * grad_W


try:
    import torch
    import torch.nn as nn

    class JordanLayer(nn.Module):
        """
        PyTorch Jordan interaction layer for use in training pipelines.

        Replaces standard linear(x) with the symmetrized Jordan product:
            output = ½(XW + WX)

        This enforces symmetric bilinear interaction, acts as an implicit
        regularizer by symmetry constraint, and is provably bounded.

        Parameters
        ----------
        dim : int
            Square feature dimension.
        """
        def __init__(self, dim: int):
            super().__init__()
            self.W = nn.Parameter(torch.randn(dim, dim) / dim ** 0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return 0.5 * (x @ self.W + self.W @ x.transpose(-1, -2).transpose(-1, -2))

        def extra_repr(self) -> str:
            return f"dim={self.W.shape[0]}, symmetric=True"

except ImportError:
    JordanLayer = None     # type: ignore  # torch not available; use JordanLayerNumpy


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Cauchy Convergence Monitor
# ─────────────────────────────────────────────────────────────────────────────

class CauchyMonitor:
    """
    Verify that the parameter sequence {θ_t} is Cauchy.

    A sequence is Cauchy if consecutive differences → 0 and the sequence
    is eventually confined to an arbitrarily small ball. This is equivalent
    to convergence in a complete metric space (ℝ^d).

    Usage:
        monitor = CauchyMonitor(tol=1e-4)
        for theta in parameter_sequence:
            monitor.update(theta)
        print(monitor.is_cauchy(), monitor.summary())
    """

    def __init__(self, tol: float = 1e-4, window: int = 20):
        self.tol = tol
        self.window = window
        self._diffs: deque = deque(maxlen=window)
        self._prev: Optional[np.ndarray] = None

    def update(self, theta: np.ndarray) -> float:
        """Record new parameter vector; return ‖θ_t - θ_{t-1}‖."""
        if self._prev is not None:
            diff = float(np.linalg.norm(theta - self._prev))
            self._diffs.append(diff)
        else:
            diff = float("inf")
        self._prev = theta.copy()
        return diff

    def is_cauchy(self) -> bool:
        """True if recent differences are all below tolerance."""
        if len(self._diffs) < self.window // 2:
            return False
        return max(self._diffs) < self.tol

    def trend(self) -> str:
        """'decreasing', 'stable', or 'increasing'."""
        if len(self._diffs) < 4:
            return "insufficient_data"
        recent = list(self._diffs)
        mid = len(recent) // 2
        if np.mean(recent[mid:]) < np.mean(recent[:mid]) * 0.9:
            return "decreasing"
        elif np.mean(recent[mid:]) > np.mean(recent[:mid]) * 1.1:
            return "increasing"
        return "stable"

    def summary(self) -> Dict:
        diffs = list(self._diffs)
        return {
            "is_cauchy": self.is_cauchy(),
            "trend": self.trend(),
            "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
            "max_diff": float(np.max(diffs)) if diffs else float("nan"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Main Analyzer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """
    Result of a single SALT analyzer step.

    Fields
    ------
    rho           : gradient ratio ρ_t ∈ (0, 1)
    epsilon       : relative gradient change ε_t
    convergent    : best CF convergent (p, q) of ρ_t
    word_letter   : 'R' or 'L' — current monoid step
    c_alpha       : signal-to-noise C_α (primary phase indicator)
    q_star        : median CF denominator over window (slow variable)
    f_c_obs       : Farey consolidation index (observed)
    f_c_pct       : FCI percentile vs permutation null
    path_entropy  : Shannon entropy of current word
    phase         : training phase label
    word          : full L/R word accumulated so far
    """
    rho: float
    epsilon: float
    convergent: Tuple[int, int]
    word_letter: str
    c_alpha: float
    q_star: float
    f_c_obs: float
    f_c_pct: float
    path_entropy: float
    phase: str
    word: str


class SaltAnalyzer:
    """
    Online SALT analyzer — attach to any training loop.

    Computes all SALT observables from consecutive gradient vectors
    without requiring access to the Hessian, Fisher matrix, or held-out data.

    Parameters
    ----------
    window : int
        Sliding window for computing q* (slow variable) and C_α.
        Larger window → more stable estimates, slower response.
        Recommended: 20–100 steps.
    n_permutations : int
        Number of permutations for the FCI null test.
    alpha : float
        Gradient renormalization constant. ‖∇̃L‖ ≤ 1/alpha.

    Usage
    -----
        analyzer = SaltAnalyzer(window=50)
        for g_prev, g_curr in zip(gradients, gradients[1:]):
            result = analyzer.step(g_prev, g_curr)
            print(result.phase, result.c_alpha)
    """

    def __init__(
        self,
        window: int = 50,
        n_permutations: int = 200,
        alpha: float = 1.0,
        rng_seed: int = 42,
    ):
        self.window = window
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.rng_seed = rng_seed

        # Rolling buffers
        self._gradients: deque = deque(maxlen=window)
        self._rhos: deque = deque(maxlen=window)
        self._convergents: deque = deque(maxlen=window)
        self._word: List[str] = []
        self._prev_rho: Optional[float] = None
        self._step_count: int = 0

    def step(self, g_prev: np.ndarray, g_curr: np.ndarray) -> StepResult:
        """
        Process one pair of consecutive gradients; return all SALT observables.

        Parameters
        ----------
        g_prev : np.ndarray   gradient at step t
        g_curr : np.ndarray   gradient at step t+1
        """
        self._step_count += 1

        # ── Raw observables ──────────────────────────────────────────────────
        rho = gradient_ratio(g_prev, g_curr)
        eps = gradient_change(g_prev, g_curr)
        conv = best_convergent(rho, eps)
        letter = word_step(self._prev_rho if self._prev_rho is not None else 0.5, rho)

        # ── Update buffers ───────────────────────────────────────────────────
        self._gradients.append(g_curr)
        self._rhos.append(rho)
        self._convergents.append(conv)
        self._word.append(letter)
        self._prev_rho = rho

        # ── Slow variables (window-level) ────────────────────────────────────
        q_vals = [q for (_, q) in self._convergents]
        q_star = float(np.median(q_vals))
        c_alpha = signal_to_noise(list(self._gradients))

        # ── FCI with permutation null ────────────────────────────────────────
        conv_list = list(self._convergents)
        if len(conv_list) >= 4:
            f_obs, f_pct = farey_consolidation_index(
                conv_list, self.n_permutations, self.rng_seed
            )
        else:
            f_obs, f_pct = 0.0, 50.0

        phase = phase_from_c_alpha(c_alpha, f_pct)
        word = "".join(self._word)
        h = path_entropy(word[-min(len(word), self.window):])

        return StepResult(
            rho=rho,
            epsilon=eps,
            convergent=conv,
            word_letter=letter,
            c_alpha=c_alpha,
            q_star=q_star,
            f_c_obs=f_obs,
            f_c_pct=f_pct,
            path_entropy=h,
            phase=phase,
            word=word,
        )

    def reset(self) -> None:
        """Clear all state."""
        self._gradients.clear()
        self._rhos.clear()
        self._convergents.clear()
        self._word.clear()
        self._prev_rho = None
        self._step_count = 0

    @property
    def steps(self) -> int:
        return self._step_count


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Spectral Basis Utilities
# ─────────────────────────────────────────────────────────────────────────────

def sturm_liouville_eigenfunctions(
    n_modes: int,
    n_points: int = 256,
    p_func=None,
    q_func=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the first n_modes eigenfunctions of the Sturm–Liouville operator:

        Lψ = -d/dx[p(x) dψ/dx] + q(x)ψ

    on [0, 1] with Dirichlet boundary conditions (ψ(0) = ψ(1) = 0).

    Defaults to p(x) = 1, q(x) = 0, giving ordinary Fourier sine modes.
    The eigenfunctions form an orthonormal basis of L²([0,1]).

    Parameters
    ----------
    n_modes  : number of eigenfunctions to return
    n_points : spatial discretization points
    p_func   : callable x → p(x), default: constant 1
    q_func   : callable x → q(x), default: constant 0

    Returns
    -------
    x     : shape (n_points,) — spatial grid
    psi   : shape (n_points, n_modes) — eigenfunctions, L²-normalized
    """
    if p_func is None:
        p_func = lambda x: np.ones_like(x)
    if q_func is None:
        q_func = lambda x: np.zeros_like(x)

    x = np.linspace(0, 1, n_points + 2)[1:-1]   # interior points
    h = x[1] - x[0]
    p = p_func(x)
    q = q_func(x)

    # Build finite-difference matrix for -d/dx[p dψ/dx] + qψ
    # Interior only → Dirichlet BCs automatically satisfied
    N = len(x)
    diag_main = np.zeros(N)
    diag_up   = np.zeros(N - 1)
    diag_down = np.zeros(N - 1)

    for i in range(N):
        p_half_right = 0.5 * (p[i] + p[i+1]) if i < N-1 else p[i]
        p_half_left  = 0.5 * (p[i] + p[i-1]) if i > 0  else p[i]
        diag_main[i] = (p_half_right + p_half_left) / h**2 + q[i]
        if i < N - 1:
            diag_up[i]   = -0.5 * (p[i] + p[i+1]) / h**2
        if i > 0:
            diag_down[i-1] = -0.5 * (p[i] + p[i-1]) / h**2

    L_mat = (
        np.diag(diag_main)
        + np.diag(diag_up, k=1)
        + np.diag(diag_down, k=-1)
    )

    # Eigendecomposition (symmetric → real eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(L_mat)
    idx = np.argsort(eigenvalues)[:n_modes]
    psi = eigenvectors[:, idx]

    # L²-normalize (so ∫ψᵢ² dx ≈ 1)
    for k in range(psi.shape[1]):
        norm = np.sqrt(_trapz(psi[:, k] ** 2, x))
        if norm > 1e-12:
            psi[:, k] /= norm

    return x, psi


def spectral_features(
    data: np.ndarray,
    psi: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Project data onto the spectral basis {ψ_n}.

        a_n = ∫ f(x) ψ_n(x) dx   (inner product in L²)

    Parameters
    ----------
    data : shape (n_points,) — function values on the grid x
    psi  : shape (n_points, n_modes) — eigenfunctions from sturm_liouville_eigenfunctions
    x    : shape (n_points,) — spatial grid

    Returns
    -------
    coeffs : shape (n_modes,) — spectral coefficients
    """
    n_modes = psi.shape[1]
    coeffs = np.array([_trapz(data * psi[:, k], x) for k in range(n_modes)])
    return coeffs


def spectral_reconstruct(
    coeffs: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    """Reconstruct f(x) = Σ a_n ψ_n(x) from spectral coefficients."""
    return psi @ coeffs
