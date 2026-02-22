# SALT — Spectral-Arithmetic Learning Theory

> *"Does arithmetic structure precede grokking under fixed-point quantization constraints?"*


### Why software results are invalid for this question:

Every execution path — NumPy, SciPy, PyTorch, a Q.16 emulator, a CORDIC
emulator — runs on a CPU or GPU. Those processors execute in **IEEE 754 float64 arithmetic**.

```
What you think you are running:
  Q.16 fixed-point training → Q.16 CORDIC log → Q.16 gradient → T_arith

What you are actually running:
  float64 simulation of Q.16 → float64 simulation of CORDIC → float64 gradient → T_arith

These are not the same thing.
```

The SALT observables are all affected:

| Observable | Why float64 ≠ fixed-point silicon |
|---|---|
| `C_α` (power-law slope) | Computed via `log` — on hardware this is a CORDIC approximation with finite iteration error. float64 `log` is exact to 15 decimal places. The power-law slope is a *different function* on real hardware. |
| `entropy H` | Shannon entropy requires `p·log(p)`. CORDIC log error is non-uniform across the probability range — small probabilities are computed less accurately. float64 has no such structure. |
| `denominator` (FFT) | Fixed-point FFT has a quantization noise floor that creates spurious spectral peaks. float64 FFT does not. The dominant frequency — and therefore T_arith — will differ. |
| `T_grok` | Grokking onset is sensitive to the loss landscape. Q.16 weight updates truncate/saturate every step, permanently altering the trajectory. float64 updates are lossless. |
| `ΔT = T_grok − T_arith` | Both endpoints are wrong. The difference is doubly wrong. |

### The emulator trap

It is tempting to write a Q.16 emulator in Python:

```python
def q16_multiply(a, b):
    result = (int(a * 65536) * int(b * 65536)) >> 16
    return max(-32768, min(32767, result))  # saturate
```

This is still running on float64 hardware. The integer arithmetic is simulated inside a
float64 register. The saturation is a conditional branch, not an electrical clamp. The
rounding is a software rule, not a transistor behavior. The accumulator width is Python's
arbitrary-precision integer, not a 32-bit or 40-bit hardware accumulator.

A CORDIC emulator has the same problem:

```
CORDIC emulator runs on  →  CPU/GPU  →  which is float64 under the hood
Q.16 emulator runs on    →  CPU/GPU  →  which is float64 under the hood

You are simulating fixed-point behavior using floating-point arithmetic.
It is not the same thing.
```

The quantization noise in real silicon is **structural** — it emerges from the physical
bit-width of the datapath and propagates deterministically through every operation. A
software simulation adds noise as a post-hoc approximation. The two noise processes have
different statistical properties, different frequency content, and different effects on
gradient dynamics.

### What valid results require

To answer the research question with scientific validity:

| Requirement | Why |
|---|---|
| **FPGA or ASIC with Q.16 datapath** | Bit-identical arithmetic to target hardware |
| **CORDIC core** (hardware, not emulated) | Softmax, entropy, log — all need CORDIC on fixed-point substrate |
| **Fixed-point BLAS / LAGREBA** | Matrix multiply and gradient accumulation at correct accumulator width |
| **Cycle-accurate logging** | T_grok and T_arith must be measured at hardware clock resolution |
| **Bit-identical software reference** | For debugging only — not for results |

Minimum viable hardware: **Xilinx Ultrascale+ or Intel Agilex FPGA** with a custom or
HLS-generated Q.16 MAC array and a CORDIC core at 16–24 iterations.

### Recommended path

The only valid path to answer this question is a **university lab partnership**
with an ECE or computer architecture group that has:

- FPGA or ASIC infrastructure
- Fixed-point toolchains (Vivado HLS, Cadence, Synopsys)
- Experience with neuromorphic or efficient ML hardware


---

## Summary

| Question | Can this code answer it? |
|---|---|
| Does SALT hold in float32? | Yes — run `--mode full` |
| Does SALT hold on Q.16 hardware? | **No. Requires physical fixed-point silicon.** |
| Does a Q.16 emulator answer the hardware question? | **No. Emulators run on float64 CPUs/GPUs.** |
| What would valid results require? | FPGA/ASIC lab, CORDIC core, fixed-point BLAS, hardware logging |
| What is the right next step? | University ECE/architecture lab partnership |

---

## Quick Start

```bash
# Run all 7 demo scenarios (saves plots to ./salt_demo_output/)
python salt_demo.py

# Run a specific scenario
python salt_demo.py --scenario grokking
python salt_demo.py --scenario jordan
python salt_demo.py --scenario live --show

# Run the 64-test suite
python salt_tests.py
```

**Dependencies:** `numpy`, `scipy`, `matplotlib` — no GPU, no deep learning framework required.

---

## Files

| File | Description |
|------|-------------|
| `salt_core.py` | Core library — all observables, diagnostics, Jordan layer, spectral basis |
| `salt_tests.py` | 64-test suite — unit, integration, theorem verification |
| `salt_demo.py` | 7 demo scenarios with full visualizations |

---

## The Central Idea

Every gradient step encodes one of two arithmetic operations:

```
R = [[1,1],[0,1]]   (gradient norm increased)
L = [[1,0],[1,1]]   (gradient norm decreased)
```

These are elements of a free monoid ℳ inside SL(2,ℤ). The word built over training — `RLLRRLRR...` — is the model's **arithmetic state**. The continued-fraction denominators of the gradient ratio ρ_t measure the curvature of the loss basin the model currently occupies.

Five independent mathematical frameworks all reduce to a single observable, **C_α**:

```
C_α  =  ‖μ_g‖² / Tr(Σ_g)       gradient signal-to-noise ratio

C_α > 1   →   signal dominates   →   GENERALIZING
C_α ≈ 1   →   critical boundary  →   grokking / double-descent
C_α < 1   →   noise dominates    →   MEMORIZING
```

Computable from gradients alone — no Hessian, no held-out data.

---

## Five Pillars

### I — Arithmetic

The gradient ratio ρ_t = ‖g_{t+1}‖ / (‖g_t‖ + ‖g_{t+1}‖) is approximated by its best continued-fraction convergent p_t/q_t at data-adaptive precision. Each training step is L or R in the Stern–Brocot tree. Consecutive steps satisfying the **Farey unimodular condition** |q_t·p_{t+1} - p_t·q_{t+1}| = 1 are arithmetically adjacent — their Ford circles touch.

### II — Geometry

The learning trajectory is a geodesic ray in the upper half-plane ℍ. Each rational convergent p/q corresponds to a **Ford circle** of radius 1/(2q²). Large radius = flat loss basin = good generalization.

```
small q   ←→   large Ford circle   ←→   flat basin   ←→   generalization
large q   ←→   small Ford circle   ←→   sharp basin  ←→   memorization
```

The generators R and L are exponentials of nilpotent Lie algebra elements: exp(E) = R, exp(F) = L. The arithmetic skeleton lives inside the continuous Lie structure.

### III — Algebra

The **Jordan product** replaces standard matrix multiplication:

```python
X ∘ W = ½(XW + WX)
```

- **Commutative:** X∘W = W∘X
- **Not associative:** (X∘Y)∘Z ≠ X∘(Y∘Z) — the algebra retains memory of computation order
- **Bounded:** ‖X∘W‖ ≤ ‖X‖·‖W‖

This enforces symmetric bilinear feature mixing and acts as an implicit regularizer. Implementable in any linear algebra library.

### IV — Spectral Theory

On the quotient learning manifold ℬ = Θ/G, define the Jordan–Liouville operator:

```
ℒ_JL ψ = -d/dx[p(x) dψ/dx] + q(x)ψ
```

Because ℒ_JL is self-adjoint, its eigenvalues are real. The **ground eigenvalue λ₁** is the stability oracle:

```
sign(λ₁)  =  sign(C_α - 1)
```

Its eigenfunctions form an orthonormal L² basis — universal approximation guaranteed.

### V — Dynamics

Two timescales operate simultaneously:
- **Fast** (per step): individual L/R gradient steps
- **Slow** (window W): evolution of C_α and q* (median CF denominator)

The slow variable q* is a noisy, non-monotone proxy for basin curvature — use it statistically over windows, not as a pointwise oracle. C_α is the primary phase indicator.

---

## Diagnostics at a Glance

All returned by `SaltAnalyzer.step(g_prev, g_curr)`:

| Field | Description | Interpretation |
|-------|-------------|----------------|
| `rho` | Gradient ratio ρ_t ∈ (0,1) | Near 0.5 = flat basin |
| `c_alpha` | Signal-to-noise C_α | **Primary phase indicator** |
| `q_star` | Median CF denominator | Slow variable; statistical proxy |
| `f_c_pct` | Farey Consolidation Index % | Corroborating signal |
| `path_entropy` | L/R word entropy H ∈ [0,1] | 0=persistent trend, 1=plateau |
| `phase` | Training phase label | See below |
| `word` | Full L/R word | Full arithmetic state |

**Phase labels:**

| Phase | C_α | Meaning |
|-------|-----|---------|
| MEMORIZATION | < 0.9 | Noise dominates |
| APPROACHING | 0.9–1.0 | Transition building |
| CRITICAL | ≈ 1.0 | Grokking boundary / double-descent peak |
| GENERALIZING | > 1.1 | Signal dominates |
| CONVERGED | > 2.0 + FCI | Stable generalization |

---

## Usage

### Attach to any training loop

```python
from salt_core import SaltAnalyzer

analyzer = SaltAnalyzer(window=50)
prev_grad = None

for step in training_loop:
    loss.backward()
    curr_grad = get_flat_gradients(model)   # numpy array shape (d,)

    if prev_grad is not None:
        result = analyzer.step(prev_grad, curr_grad)
        if step % 50 == 0:
            print(f"step={step:4d}  phase={result.phase:<14}  "
                  f"C_α={result.c_alpha:.3f}  q*={result.q_star:.0f}")

    prev_grad = curr_grad.copy()
    optimizer.step()
```

### Jordan product layer (numpy)

```python
from salt_core import JordanLayerNumpy

layer = JordanLayerNumpy(dim=64)
output = layer.forward(X)    # X ∘ W = ½(XW + WX)
```

### Jordan product layer (PyTorch, if available)

```python
from salt_core import JordanLayer   # None if torch not installed
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    JordanLayer(64),         # symmetric bilinear interaction
    nn.ReLU(),
    nn.Linear(64, 10),
)
```

### Monitor Cauchy convergence

```python
from salt_core import CauchyMonitor

monitor = CauchyMonitor(tol=1e-4, window=20)
for theta in parameter_sequence:
    monitor.update(theta)

print(monitor.is_cauchy())   # True when ‖θ_{t+1} - θ_t‖ < tol
print(monitor.trend())       # 'decreasing', 'stable', or 'increasing'
print(monitor.summary())     # {'is_cauchy': bool, 'mean_diff': float, ...}
```

### Spectral features

```python
from salt_core import (
    sturm_liouville_eigenfunctions,
    spectral_features,
    spectral_reconstruct,
)

x, psi = sturm_liouville_eigenfunctions(n_modes=20, n_points=256)
coeffs = spectral_features(data, psi, x)    # project onto basis
recon  = spectral_reconstruct(coeffs, psi)  # reconstruct
```

### Raw operations

```python
from salt_core import (
    gradient_ratio, gradient_change, signal_to_noise,
    cf_convergents, best_convergent, is_farey_neighbor,
    jordan_product, associator, renormalized_gradient,
)

rho  = gradient_ratio(g1, g2)              # ∈ (0, 1)
eps  = gradient_change(g1, g2)             # relative change
c_a  = signal_to_noise(list_of_grads)      # C_α

convs    = cf_convergents(rho, q_max=50)   # list of (p, q)
p, q     = best_convergent(rho, eps)       # best at data resolution
adjacent = is_farey_neighbor(p1,q1,p2,q2) # |q1·p2 - p1·q2| == 1

XoW    = jordan_product(X, W)             # ½(XW + WX), commutative
A      = associator(X, Y, Z)             # non-zero = non-associative
g_safe = renormalized_gradient(grad, alpha=1.0)  # ‖g_safe‖ ≤ 1/α
```

---

## Demo Scenarios

```bash
python salt_demo.py                              # all 7, save to file
python salt_demo.py --scenario memorization      # noise vs signal
python salt_demo.py --scenario grokking          # phase transition detection
python salt_demo.py --scenario jordan            # commutativity, renorm
python salt_demo.py --scenario spectral          # eigenfunctions, Ford circles
python salt_demo.py --scenario convergence       # Cauchy monitor
python salt_demo.py --scenario live              # full pipeline dashboard
python salt_demo.py --scenario arithmetic        # CF convergents, Farey neighbors
python salt_demo.py --show                       # display instead of save
python salt_demo.py --outdir /tmp/salt_plots     # custom output path
```

| Scenario | What you see |
|----------|--------------|
| `memorization` | C_α, q*, entropy — diverge cleanly between noise and signal |
| `grokking` | SALT detecting the memorization→generalization transition |
| `jordan` | Commutativity proof, non-associativity, gradient renorm bound |
| `spectral` | S-L eigenfunctions, spectral reconstruction, Ford circle packing |
| `convergence` | Four parameter trajectories; Cauchy condition highlighted |
| `live` | Dashboard: 7 panels, step-by-step readout in terminal |
| `arithmetic` | CF approximation quality, Farey neighbor hit-rate in trajectories |

---

## Test Suite

```bash
python salt_tests.py
# 64 passed   0 failed   64 total
```

| Block | Tests | Covers |
|-------|-------|--------|
| Scalar observables | 11 | gradient_ratio, gradient_change, signal_to_noise |
| Continued fractions | 9 | cf_convergents, best_convergent |
| Farey arithmetic | 7 | is_farey_neighbor, word_step, path_entropy |
| FCI | 3 | farey_consolidation_index, phase_from_percentile |
| Jordan algebra | 7 | jordan_product, associator, renormalized_gradient |
| Jordan layer | 2 | JordanLayerNumpy |
| Cauchy monitor | 4 | CauchyMonitor |
| Spectral basis | 5 | sturm_liouville_eigenfunctions, spectral_features |
| SaltAnalyzer | 7 | full integration tests |
| Theorem verification | 6 | Hurwitz, commutativity, renorm bound, Farey, SL(2,ℤ), L²-orthonormality |

---

## Provable Bounds

| Result | Statement |
|--------|-----------|
| Renormalized gradient | ‖∇̃L‖ ≤ 1/α uniformly; tight as ‖∇L‖ → ∞ |
| Resolvent stability | ‖(L-λI)⁻¹‖ = 1/dist(λ, σ(L)) |
| Hurwitz approximation | CF error < 1/(√5·q²) — sharper than 1/q² |
| Cauchy convergence | {θ_t} Cauchy under Lipschitz loss + summable step sizes |
| L² universality | Σ aₙ ψₙ converges to any f ∈ L² as N → ∞ |
| Jordan commutativity | X∘W = W∘X (immediate from definition) |

---

## Training Phenomena in SALT

| Phenomenon | SALT interpretation |
|------------|---------------------|
| **Grokking** | C_α(t) crosses 1 after prolonged C_α < 1; Farey backtrack in q* |
| **Double descent** | Interpolation threshold = C_α = 1 = λ₁ = 0 |
| **Neural collapse** | Convergence to ground eigenfunction φ₁ of ℒ_JL |
| **Lottery tickets** | Subnetwork whose restricted ℒ_JL already has λ₁ > 0 at init |

---

## Open Problems

**Grokking validation (Conjecture C1).** Does q* decrease 50–200 steps before test accuracy rises on Power et al. (2022) modular arithmetic?

**Farey-SAM (Conjecture C4).** Under smooth loss, penalizing q* approximates penalizing λ_max(H) with no double forward pass:
```
L_Farey(θ) = L(θ) + λ·(q*)²
```

**Non-smooth extension.** Assumption of C² loss fails at ReLU kinks. Within each linear region the CF map is well-defined; across boundaries, gradient jumps correspond to mediant insertions.

**Fisher-weighted C_α.** The coordinate-invariant form C_α^F = μ_g^T F⁻¹ μ_g / Tr(F⁻¹ Σ_g) is O(d³) but low-rank Fisher approximations (K-FAC) may make it practical.

---

## References

| Work | Contribution to SALT |
|------|---------------------|
| Cauchy (1816) | Farey unimodular condition |
| Hurwitz (1891) | Best rational approximation bound |
| Ford (1938) | Ford circles and Farey tangency |
| Calkin & Wilf (2000) | Bijection ℳ → ℚ₊ |
| Hardy & Wright (1979) | Continued fractions (Ch. 10–11) |
| Albert (1934) | Exceptional Jordan algebra H₃(𝕆) |
| Lubotzky, Phillips & Sarnak (1988) | Ramanujan graphs and optimal mixing |
| Sturm (1836), Liouville (1836) | Sturm–Liouville spectral theory |
| Zettl (2005) | Modern S-L theory reference |
| Fenichel (1979) | Fast–slow dynamical systems |
| McAllester (1999) | PAC-Bayes generalization bounds |
| Foret et al. (2021) | Sharpness-Aware Minimization |
| Power et al. (2022) | Grokking benchmark |
| Papyan et al. (2020) | Neural collapse |
| Li et al. (2020) | Fourier Neural Operators |
| Vigouroux et al. (2024) | Deep Sturm–Liouville |
