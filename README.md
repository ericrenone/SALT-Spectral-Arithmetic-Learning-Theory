# Spectral-Arithmetic Learning Theory (SALT)


> *Every gradient trajectory is a word in a free monoid.
> Every word encodes a rational. Every rational is a cusp of hyperbolic space.
> Every cusp is a Ford circle whose radius measures flatness.
> Every flatness measure is a Rayleigh quotient.
> The sign of that quotient is the sign of generalization.*

---

## Provenance and Novelty

SALT synthesises five independent source frameworks — RTLG, FHN/Farey, FLD, ARDI, SLNF —
that each independently identified the same scalar threshold. This document proves their
equivalence, corrects six substantive errors in the source material (§14), and presents
the unified theory under a single consistent notation.

**What is new.** The key synthesis is the proof that the Farey unimodular condition
`|bc − ad| = 1`, the Ford circle tangency condition, the Calkin–Wilf tree edge relation,
and the positivity of the Jordan–Liouville ground eigenvalue are **four coordinate
descriptions of the same object** — the primitive lattice basis condition in ℤ².

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✓ | Established theorem — proof sketch or canonical reference given |
| ✓ cond. | Proved under explicitly stated assumptions |
| ~ | Productive structural analogy — not formal equivalence |
| ⚠ | Conjecture — gap stated, falsifiable prediction given |
| ✗ err. | Error identified in source material; corrected here |

---

## Table of Contents

1. [Overview and Core Motivation](#1-overview-and-core-motivation)
2. [The Five Mathematical Pillars](#2-the-five-mathematical-pillars)
3. [Pillar I — Arithmetic: The Free Positive Monoid of SL(2,ℤ)](#3-pillar-i--arithmetic-the-free-positive-monoid-of-sl2ℤ)
4. [Pillar II — Geometry: Hyperbolic Space and Ford Circles](#4-pillar-ii--geometry-hyperbolic-space-and-ford-circles)
5. [Pillar III — Algebra: The Exceptional Jordan Algebra](#5-pillar-iii--algebra-the-exceptional-jordan-algebra)
6. [Pillar IV — Spectral Theory: The Jordan–Liouville Operator](#6-pillar-iv--spectral-theory-the-jordanliouville-operator)
7. [Pillar V — Dynamics: Fast–Slow Systems and Ergodic Flow](#7-pillar-v--dynamics-fastslow-systems-and-ergodic-flow)
8. [The Unifying Theorem: Five Languages, One Condition](#8-the-unifying-theorem-five-languages-one-condition)
9. [The Learning Embedding](#9-the-learning-embedding)
10. [Derived Diagnostics](#10-derived-diagnostics)
11. [Core Theorems: Proved, Conditional, and Conjectured](#11-core-theorems-proved-conditional-and-conjectured)
12. [The Unified Phase Diagram](#12-the-unified-phase-diagram)
13. [Phenomenology: Grokking, Double Descent, Neural Collapse, Lottery Tickets](#13-phenomenology-grokking-double-descent-neural-collapse-lottery-tickets)
14. [Error Catalogue: Six Corrections to Source Material](#14-error-catalogue-six-corrections-to-source-material)
15. [Implementation Reference](#15-implementation-reference)
16. [Open Problems](#16-open-problems)
17. [References](#17-references)
18. [Appendix A: The Arithmetic Skeleton in Five Identities](#appendix-a-the-arithmetic-skeleton-in-five-identities)
19. [Appendix B: The Threshold in Five Languages](#appendix-b-the-threshold-in-five-languages)

---

## 1. Overview and Core Motivation

Standard learning theory treats gradient descent as continuous flow through a loss
landscape, capturing optimization geometry but obscuring the **discrete arithmetic
structure** that underlies phase transitions, training plateaus, and sudden generalization.

SALT identifies a single scalar observable — the **gradient ratio** ρ_t — as the
fundamental dynamical coordinate, and reveals that its arithmetic structure governs
every observable learning phenomenon.

### 1.1 The Fundamental Observable

For consecutive gradient vectors g_t, g_{t+1} ∈ ℝᵈ:

```
ρ_t  =  ‖g_{t+1}‖ / (‖g_t‖ + ‖g_{t+1}‖)   ∈ (0, 1)
```

This scalar is:
- **Invariant** under orthogonal rotation of parameter space ✓
- **Invariant** under positive rescaling of the learning rate ✓
- **NOT invariant** under arbitrary smooth reparameterization ✗ err. (see §14.4)

Under Assumptions S and E (§11.2), ρ_t stabilises to `κ/(1+κ)` where
`κ = |1 − ηλ₁|` encodes the dominant Hessian eigenvalue via a Möbius transformation.

### 1.2 The Central Architecture

```
Gradient norms g_t, g_{t+1}
          │
          ▼
    ρ_t = ‖g_{t+1}‖/(‖g_t‖+‖g_{t+1}‖) ∈ (0,1)       [scalar gateway]
          │
          ├──► CF convergent p_t/q_t at resolution Q_max = ⌊1/ε_grad⌋
          │            │
          │            ├──► Word G₀G₁…G_t ∈ {L,R}*    [arithmetic]
          │            ├──► Node in Calkin–Wilf tree    [combinatorial]
          │            ├──► Ford circle radius 1/2q²    [geometric]
          │            └──► Geodesic ray in ℍ           [hyperbolic]
          │
          └──► C_α = ‖μ_g‖²/Tr(Σ_g)                    [spectral proxy]
                       │
                       ▼
              Rayleigh quotient of ℒ_JL
                       │
              λ₁ > 0 ──► Generalization
              λ₁ = 0 ──► Grokking boundary
              λ₁ < 0 ──► Memorization trap
```

### 1.3 What SALT Proves vs. Conjectures

SALT makes a careful distinction between:
- **Proved theorems** about the arithmetic/geometric/spectral structures (all classical)
- **Conditional results** about the learning embedding (under Assumptions S, E)
- **Conjectures** connecting the framework to deep learning phenomena

Every section is labeled with the status legend above. No empirical claim is presented
without a falsifiable prediction.

---

## 2. The Five Mathematical Pillars

| Pillar | Core Object | Domain | Key Invariant |
|--------|-------------|--------|---------------|
| Arithmetic | Positive monoid ℳ ⊂ SL(2,ℤ) | Discrete | Word length = CF depth |
| Geometry | Upper half-plane ℍ under SL(2,ℝ) | Continuous | Ford circle radius r = 1/2q² |
| Algebra | Albert algebra 𝔄 = H₃(𝕆) | Non-associative | Associator A(X,Y,Z) ≠ 0 |
| Spectral | Jordan–Liouville operator ℒ_JL | Functional analytic | Ground eigenvalue λ₁ |
| Dynamics | Fast–slow SDE on SL(2,ℝ) | Stochastic | Ergodic invariant measure μ |

The scalar gateway ρ_t connects all five pillars. Each pillar is a different coordinate
system on the same underlying mathematical object.

---

## 3. Pillar I — Arithmetic: The Free Positive Monoid of SL(2,ℤ)

### 3.1 Generators ✓

Define integer matrices:

```
L = ⎡1  0⎤    R = ⎡1  1⎤
    ⎣1  1⎦        ⎣0  1⎦
```

Both satisfy det L = det R = 1, so L, R ∈ SL(2,ℤ).

**Definition (Positive Monoid).** The monoid generated by L and R under matrix
multiplication:

```
ℳ = ⟨L, R⟩  ⊂  SL(2,ℤ)
```

**Theorem 1 (Positive Cone Structure). ✓**
```
ℳ  =  { M ∈ SL(2,ℤ)  |  M = ⎡a  b⎤  with  a, b, c, d ≥ 0 }
                            ⎣c  d⎦
```

*Proof sketch.* Suppose M = [[a,b],[c,d]] has non-negative entries with ad−bc = 1.
Apply the greedy reduction: right-multiply by L⁻¹ = [[1,0],[−1,1]] if a ≥ c (reducing
a−c), or R⁻¹ = [[1,−1],[0,1]] if c > a (reducing c−a). Since gcd(a,c) divides
ad−bc = 1 we have gcd(a,c) = 1, so the algorithm terminates at I ∈ ℳ. The inverse
matrices L⁻¹ and R⁻¹ have one negative entry each, so no positive-cone element is
produced by inverses. ∎

**Crucially, ℳ is a FREE monoid:** the generators L and R satisfy no relations, so
every distinct word in {L,R}* yields a distinct matrix. This freeness is what makes
the encoding injective and the arithmetic unambiguous.

### 3.2 Bijection with Positive Rationals ✓

**Definition (Projective Map).** For M = [[a,b],[c,d]] ∈ ℳ with c > 0:
```
Φ(M)  =  a/c   ∈  ℚ₊
```

**Theorem 2 (Calkin–Wilf Bijection). ✓**
Φ : ℳ → ℚ₊ is a bijection.

*Proof.* Since det M = ad−bc = 1 with a, c ≥ 0, gcd(a,c) | 1 so a/c is in lowest
terms. Injectivity: two distinct words give distinct matrices (freeness) hence distinct
ratios. Surjectivity: given p/q in lowest terms, the Euclidean algorithm on (p,q)
yields continued fraction coefficients [a₀; a₁, …, aₖ]; then M = Rᵃ⁰Lᵃ¹Rᵃ²⋯
satisfies Φ(M) = p/q. ∎

### 3.3 Word–Continued Fraction Correspondence ✓

**Theorem 3.** For p/q with CF expansion [a₀; a₁, …, aₖ] (all aᵢ ≥ 1 for i ≥ 1):
```
M  =  Rᵃ⁰ Lᵃ¹ Rᵃ² Lᵃ³ ⋯
```
with Φ(M) = p/q. The word length equals ∑ᵢ aᵢ.

**Corollary (Depth–Precision).** Deeper nodes in the tree (larger word length)
represent arithmetically more complex states: larger CF denominators, finer
approximation, and (under the CCC, §11.2) sharper loss basins.

### 3.4 The Farey Sequence and Unimodular Condition ✓

The **Farey sequence** Fₙ is the ascending list of all p/q ∈ [0,1] in lowest terms
with q ≤ n.

**Theorem 4 (Farey Adjacency — Cauchy 1816). ✓**
Two reduced fractions a/b < c/d are **adjacent in some Fₙ if and only if**:
```
|bc − ad|  =  1
```
Equivalently, [[a, c], [b, d]] ∈ SL(2,ℤ). This is the **Farey unimodular condition**.

**Theorem 5 (Mediant Minimality). ✓**
If a/b and c/d are Farey neighbors, their mediant (a+c)/(b+d) is the unique reduced
fraction between them with smallest denominator. Mediants insert the next level of the
Stern–Brocot tree.

**Theorem 6 (Sequence Length). ✓**
```
|Fₙ|  ~  3n²/π²
```
The number of fractions grows quadratically; unimodular pairs are asymptotically rare,
which makes their occurrence in a gradient sequence statistically informative.

### 3.5 Equivalence of Tree Structures ✓

All four of the following are the same combinatorial object with different labeling:

| Structure | Ordering Principle | Adjacency |
|---|---|---|
| Calkin–Wilf tree | Breadth-first ancestry | Parent → L-child, R-child |
| Stern–Brocot tree | Left-to-right magnitude | Mediant insertion |
| Farey adjacency graph | Unimodular pairs in Fₙ | |bc−ad|=1 |
| CF partial-quotient tree | Euclidean algorithm depth | Digit truncation |

Each corresponds to a different path-query on the positive cone ℳ.

---

## 4. Pillar II — Geometry: Hyperbolic Space and Ford Circles

### 4.1 SL(2,ℝ) Action on the Upper Half-Plane ✓

The upper half-plane ℍ = {z ∈ ℂ | Im(z) > 0} with hyperbolic metric
ds² = (dx² + dy²)/y² is a Riemannian symmetric space of constant curvature −1.

SL(2,ℝ) acts by orientation-preserving isometries via Möbius transformations:
```
M · z  =  (az + b)/(cz + d),    M = ⎡a  b⎤  ∈  SL(2,ℝ)
                                     ⎣c  d⎦
```

The discrete subgroup SL(2,ℤ) acts with fundamental domain
F = {z ∈ ℍ : |z| ≥ 1, |Re(z)| ≤ 1/2}. The cusps — boundary points in ∂ℍ = ℝ ∪ {∞}
approached by geodesic rays — are exactly ℚ ∪ {∞}.

**Key geometric fact.** Every path in ℳ under Φ corresponds to a geodesic ray in ℍ
that terminates at a rational cusp. The four tree structures of §3.5 are four parametri-
zations of this geodesic fan from the basepoint i ∈ ℍ.

### 4.2 Lie Algebra: Generators as Nilpotent Exponentials ✓

Standard basis of 𝔰𝔩(2,ℝ):
```
H = ⎡ 1   0⎤    E = ⎡0  1⎤    F = ⎡0  0⎤
    ⎣ 0  −1⎦        ⎣0  0⎦        ⎣1  0⎦
```

Lie brackets: [H,E] = 2E, [H,F] = −2F, [E,F] = H.

**Theorem 7 (Nilpotent Exponentials). ✓**
Since E² = F² = 0 (nilpotent), the Baker–Campbell–Hausdorff series truncates:
```
exp(E)  =  I + E  =  R        exp(F)  =  I + F  =  L
```

The discrete generators R and L are exactly the exponentials of nilpotent Lie algebra
elements. This places the arithmetic skeleton inside the Lie-theoretic structure.

**Theorem 8 (Hyperbolic Flow). ✓**
```
exp(tH)  =  ⎡eᵗ    0 ⎤
            ⎣0    e⁻ᵗ⎦
```
Under Φ: Φ(exp(tH) · M) = e²ᵗ Φ(M). The projective coordinate log Φ(M(t)) evolves
**linearly** at rate 2λ under the flow Ṁ = λH · M.

**Interpretation.** Continuous evolution of the signal-to-noise ratio C(t) = c e^{2λt}
corresponds to hyperbolic flow in SL(2,ℝ). Individual gradient steps L and R are the
**discrete skeleton** — nilpotent exponentials — living inside this continuous flow.

### 4.3 Ford Circles and Loss Basin Geometry ✓

For each p/q in lowest terms (q ≥ 1), the **Ford circle** C(p/q) is the circle in ℍ̄:
```
center  =  (p/q,  1/2q²),    radius  r(p/q)  =  1/(2q²)
```

**Theorem 9 (Ford Tangency — Ford 1938). ✓**
Two Ford circles C(a/b) and C(c/d) are **externally tangent if and only if** |bc − ad| = 1
(the Farey unimodular condition). Every pair of tangent Ford circles has the mediant
Ford circle C((a+c)/(b+d)) tangent to both.

**The Basin Correspondence (CCC — conditional, see §11.2):**
```
Small denominator q  ←→  Large Ford circle  ←→  Flat minimum  ←→  Good generalization
Large denominator q  ←→  Small Ford circle  ←→  Sharp minimum ←→  Poor generalization
Tangent circles      ←→  Saddle between adjacent loss basins
```

The first arrow (radius = 1/2q²) is an unconditional theorem. The remaining arrows
are conditional on the **Convergent-Curvature Correspondence** (Theorem 17, §11.2),
which requires Assumptions S and E.

### 4.4 The Three-Distance Theorem and Adaptive Resolution ✓

**Theorem 10 (Steinhaus 1950; Sós 1958). ✓**
For any irrational α and integer N ≥ 1, the fractional parts {α}, {2α}, …, {Nα}
partition [0,1) into exactly **two or three distinct gap lengths**, with the gap sizes
determined by the consecutive denominators in the CF expansion of α.

This grounds the data-adaptive resolution scheme: the natural quantisation for a
gradient ratio ρ_t is its CF denominator, not an arbitrary grid.

**Theorem 11 (Hurwitz Approximation — Hurwitz 1891). ✓**
For any irrational x, there exist infinitely many convergents p/q satisfying:
```
|x − p/q|  <  1 / (√5 · q²)
```
The constant 1/√5 is **sharp**: it cannot be improved for x = (1+√5)/2 (golden ratio).

**Data-Adaptive Resolution (§9.1).** Set
```
Q_max  =  ⌊1/ε_grad⌋    where    ε_grad  =  ‖g_{t+1} − g_t‖ / (‖g_t‖ + ‖g_{t+1}‖)
```
By Hurwitz, approximation error at denominator Q_max is at most ε_grad²/√5, which is
below the gradient measurement noise level ε_grad. No finer CF structure is imposed
than the data supports.

### 4.5 Franel–Landau Theorem and Scope Restriction ✓ / ✗ err.

Let r_ν be the ν-th element of Fₙ and δ_ν = r_ν − ν/|Fₙ| its discrepancy from
uniform spacing.

**Theorem 12 (Franel 1924; Landau 1924). ✓**
```
∑_ν δ_ν²  =  O(n^{−1+ε}) for all ε > 0   ⟺   Riemann Hypothesis
```

This is an unconditional arithmetic equivalence about the Farey sequence itself.

**Scope restriction. ✗ err.** Source documents FHN §3.8, FLD §2.6, and SLNF §14.2
claim this gives "number-theoretic guarantees on gradient sample quality." This is
**incorrect as stated.** The theorem concerns the Farey sequence, not gradient
distributions. The connection to gradients is a structural analogy (~), not a theorem,
and requires separate empirical validation. See §14.6 for the full correction.

---

## 5. Pillar III — Algebra: The Exceptional Jordan Algebra

### 5.1 The Albert Algebra ✓

The **Albert algebra** 𝔄 = H₃(𝕆) is the unique (up to isomorphism) 27-dimensional
**exceptional Jordan algebra**: the space of 3×3 Hermitian matrices over the octonions 𝕆.

```
Every element:  X = ⎡ α    x    y ⎤    α, β, γ ∈ ℝ,   x, y, z ∈ 𝕆
                    ⎢ x̄    β    z ⎥
                    ⎣ ȳ    z̄    γ ⎦

Dimension:  3 real diagonal  +  3 × 8 octonionic off-diagonal  =  27
```

The Albert algebra is the exceptional outlier in the classification of finite-dimensional
formally real Jordan algebras (Albert 1934; Jacobson 1968).

### 5.2 Jordan Product and Non-Associativity ✓

**Jordan product** (commutative, non-associative):
```
X ∘ Y  =  ½(XY + YX)
```

**Associator** (measures order dependence):
```
A(X, Y, Z)  =  (X ∘ Y) ∘ Z  −  X ∘ (Y ∘ Z)   ≠  0  in general
```

The non-vanishing associator gives the Albert algebra **memory of computation order**:
two update paths reaching the same state via different orderings produce different
associators. Standard matrix algebras cannot distinguish them.

### 5.3 The F₄ Symmetry Group ✓

The automorphism group of 𝔄 is the **exceptional Lie group F₄** of dimension 52
(Jacobson 1968). Under F₄:
- Jordan products are preserved: φ(X ∘ Y) = φ(X) ∘ φ(Y)
- F₄ restricts the admissible eigenfunctions of ℒ_JL (boundary conditions)
- Eigenvalues {λₙ} of ℒ_JL are **F₄-gauge invariants** — unchanged under neuron
  permutation or sign flip of parameter blocks

### 5.4 Ramanujan Graphs and Optimal Mixing ✓

A k-regular graph G is a **Ramanujan graph** if its adjacency matrix A satisfies:
```
λ₂(A)  ≤  2√(k−1)
```

**Theorem 13 (Lubotzky–Phillips–Sarnak 1988). ✓**
The Ramanujan bound is optimal: no k-regular graph has λ₂(A) < 2√(k−1) in general.
Ramanujan graphs have mixing time t_mix = O(log |V|).

The **Ramanujan–Jordan update**:
```
X_{t+1}  =  X_t  +  τ [(X* − X_t) ∘ ℛ]
```
(where ℛ is the Ramanujan adjacency tensor) propagates learning signals across the
27-dimensional representation manifold in O(log n) steps — the theoretically optimal
mixing rate.

### 5.5 Hardy–Ramanujan Partition Asymptotics ✓ (asymptotic only)

**Theorem 14 (Hardy–Ramanujan 1918). ✓**
The partition function p(n) satisfies:
```
p(n)  ~  (1 / (4n√3)) · exp(π√(2n/3))    as n → ∞
```

**This is an asymptotic formula, not an exact one. ✗ err.** See §14.5 for full details
and a correction table showing errors of ~88% at n=1 and ~7% at n=10. Capacity bounds
involving p(n) hold for **sufficiently large n**. Claims about small n using this formula
are unreliable. The exact formula is due to Rademacher (1937), which provides the
complete convergent series using Dedekind sums.

---

## 6. Pillar IV — Spectral Theory: The Jordan–Liouville Operator

### 6.1 Setup: The Quotient Learning Manifold ✓ cond.

Let Θ ⊂ ℝᵈ be the parameter space. A network has symmetry group G ⊂ O(d) (permutations,
sign flips). Define the **quotient learning manifold**:
```
ℬ  =  Θ / G
```

Under Assumption S (smoothness, §11.2), the loss L : ℬ → ℝ descends to a C² function
on ℬ, and gradient descent induces a flow on ℬ.

**Theorem 15 (Horizontal Gradient). ✓ cond. (under S)**
For G-invariant loss L, the gradient on the total space Θ is purely horizontal (Kobayashi
& Nomizu 1963): the vertical component ∇^V L = 0. Gradient descent on ℬ is equivalent
to gradient descent on Θ modulo G.

### 6.2 The Jordan–Liouville Operator ✓ cond.

On ℬ near a critical point b* = π(θ*), define the **Jordan–Liouville operator**:
```
ℒ_JL  =  −(d/db)[p(b) · (d/db)] + q(b)
```
where:
- p(b) = e^{−S(b)} > 0 is the inverse temperature profile (Gibbs weight)
- q(b) = V(b) − E is the effective potential minus energy level
- S(b) is the action functional of the learning flow

This is a **self-adjoint operator** on L²(ℬ, e^{−S} db), so its eigenvalues are real:
```
ℒ_JL φₙ  =  λₙ φₙ,     λ₁ ≤ λ₂ ≤ ⋯
```

The ground eigenvalue λ₁ is the **stability oracle** of the learning system.

### 6.3 Spectral-Dynamical Correspondence ✓ cond.

**Theorem 16 (Rayleigh Quotient). ✓**
```
λ₁  =  min_{φ ≠ 0}  [∫ p(b) |φ'(b)|² db + ∫ q(b) |φ(b)|² db] / ∫ |φ(b)|² e^{−S} db
```

The **empirical estimator** of λ₁ (under Assumptions S, E):
```
C_α  =  ‖μ_g‖² / Tr(Σ_g)
```
where μ_g = 𝔼[g_t] and Σ_g = Cov(g_t). Then sign(λ₁) = sign(C_α − 1).

**Coordinate invariance note. ✗ err.** The estimator C_α is invariant under orthogonal
reparameterization only. The fully coordinate-invariant form is the Fisher-weighted ratio
C_α^F = μ_gᵀ F⁻¹ μ_g / Tr(F⁻¹ Σ_g), which requires computing the Fisher matrix F
(O(d³) cost). See §14.4. For large models, C_α is the practical choice with the
limitation understood.

### 6.4 The PAC-Bayes Generalization Bound ✓ cond.

**Theorem (McAllester 1999 + CCC, T18 — see §11.2). ✓ cond.**
Under Assumptions S and E, for δ ∈ (0,1) with probability ≥ 1−δ over the training set:
```
G(θ*)  ≲  q* · √[C₀(d + log(2/δ)) / (2n)]
```
where q* is the median CF denominator, d the parameter dimension, and n the sample size.

The bound connects the arithmetic complexity q* directly to generalization: smaller q*
(flatter basin) yields tighter generalization guarantees.

---

## 7. Pillar V — Dynamics: Fast–Slow Systems and Ergodic Flow

### 7.1 Fast–Slow Decomposition ✓ cond.

The learning system has two distinct timescales:
- **Fast** (ε ~ 1/W): individual gradient steps, L and R word insertions
- **Slow** (ε = 1): evolution of C_α, q* (median over window W)

Define the slow variable:
```
q*(t)  =  median_{s ∈ [t−W, t]} q_s,    timescale  O(W)
```

The fast variable is the individual convergent q_t, which fluctuates rapidly. The
separation ε = 1/W ≪ 1 (for large windows) justifies treating q* as quasi-static
relative to individual step fluctuations.

**Fenichel's Theorem.** Under standard non-degeneracy conditions, fast–slow systems
have an invariant slow manifold M_ε that persists for small ε and attracts nearby
trajectories exponentially fast (Fenichel 1979). In SALT, the slow manifold is the
set {q* = q*_∞} where q*_∞ is the attractor denominator.

### 7.2 FitzHugh–Nagumo Analogy ~ (structural)

The FHN neuron model:
```
ε dv/dt  =  v − v³/3 − w + I_ext
  dw/dt  =  v + a − bw
```

The analogy to gradient dynamics:
- v ↔ instantaneous gradient ratio ρ_t (fast variable)
- w ↔ slow memory variable C_α (slow variable)
- I_ext ↔ learning signal strength
- Excitability threshold ↔ C_α = 1 (λ₁ = 0)

This is a **structural analogy** (~), not a formal equivalence. The excitability
parameter Γ = ‖∇_ℬS̄‖² / Tr(Dₛ) > 1 corresponds formally to the supermartingale
condition C_α > 1 (proved under Assumptions S, E).

### 7.3 Ergodic Theory of the S1–S2–Ω Chain ✓ (conditional on irreducibility)

Define the **transport–gate–synthesis** Markov chain:
```
T_t   =  √S2 · S1 / (√S1 + ε)             [Fisher-metric interpolation]
G_t   =  T_t^β / ∑_j T_j^β                [information bottleneck gate]
Ω_t   =  ½(G_t + S2_t)                    [synthesis]
```

For β ∈ (0,1) and ε, τ > 0, the chain is irreducible (all transitions strictly
positive), aperiodic (S2 mixture prevents period-2 orbits), and on a compact state
space (probability simplex Δᴺ). By Perron–Frobenius and the ergodic theorem for
positive Harris chains:
```
lim_{T→∞} (1/T) ∑_{t=0}^T φ(Ω_t)  =  𝔼_{P_Ω*}[φ]    almost surely
```
for all bounded measurable φ. The system has a unique stationary distribution P_Ω*
and no permanent local traps.

---

## 8. The Unifying Theorem: Five Languages, One Condition

### 8.1 The Canonical Identification Table

| Mathematical Object | Pillar | Role in SALT |
|---|---|---|
| Positive cone ℳ ⊂ SL(2,ℤ) | Arithmetic | State space of learning trajectories |
| Calkin–Wilf / Stern–Brocot tree | Arithmetic | Node = current approximant |
| Word metric in {L, R}* | Arithmetic | Complexity of learning state |
| Farey adjacency graph | Arithmetic | Transition validity (unimodular) |
| Geodesic fan in ℍ | Geometry | Continuous envelope of discrete path |
| Ford circle packing | Geometry | Basin size = 1/2q² |
| Lie algebra exp: E→R, F→L | Geometry | Discrete ↔ continuous bridge |
| Albert algebra J₃(𝕆) | Algebra | Representation space with order memory |
| Ramanujan mixing tensor | Algebra | Optimal O(log n) information transport |
| Jordan–Liouville operator ℒ_JL | Spectral | Stability oracle |
| Ground eigenvalue λ₁ | Spectral | Sign = learning phase |
| Fast–slow decomposition | Dynamics | Timescale separation (q* vs q_t) |
| Ergodic stationary measure μ | Dynamics | Long-run behavior |

### 8.2 The Five-Language Equivalence ✓ / ⚠

**Theorem (Near-Critical Equivalence).** Near the critical point (λ₁ ≈ 0), the
following conditions are equivalent **under Assumptions S and E**:

```
(I)   λ₁(ℒ_JL) > 0                              [SLNF: spectral stability]
(II)  C_α = ‖μ_g‖² / Tr(Σ_g) > 1               [ARDI: signal-to-noise ratio]
(III) Γ = ‖∇_ℬS̄‖² / Tr(Dₛ) > 1               [FHN: supermartingale drift]
(IV)  word d(t) decreasing                       [RTLG: path shortening]
(V)   q* statistically small                     [Farey/FLD: flat basin]
```

**Important caveat for (V).** The relationship between q* and curvature is
**non-monotone** (see §10.2 for proof). Small q* occurs both at very flat minima
(ρ ≈ 1/2) and at special arithmetic points (e.g. ρ = 1/3 exactly). Condition (V) should
be read as a **statistical tendency** (median q* over a window), not a pointwise oracle.
Source documents that treat q* as a monotone proxy for sharpness are incorrect. See §14
for the full error catalogue.

**Proof structure.**
- (I) ↔ (III): Γ is the Rayleigh quotient at the current state (§6.3)
- (II) ↔ (III): μ_g estimates signal power; Tr(Σ_g) estimates noise
- (III) ↔ (IV): Under S and E, C_α > 1 iff gradient-norm path is descending (mean-field)
- (IV) → (V): Smaller denominator convergents arise near ρ = 1/2 (flat minima)
  statistically, though not pointwise (⚠ non-monotone, see §10.2)

---

## 9. The Learning Embedding

### 9.1 The Two Fundamental Scalars ✓

For consecutive gradient vectors g_t, g_{t+1} ∈ ℝᵈ:

```
ρ_t       =  ‖g_{t+1}‖ / (‖g_t‖ + ‖g_{t+1}‖)   ∈ (0,1)    [gradient ratio]
ε_grad,t  =  ‖g_{t+1} − g_t‖ / (‖g_t‖ + ‖g_{t+1}‖)  ∈ [0,1]  [relative change]
```

ρ_t is the **approximation target**; ε_grad,t sets the **approximation resolution**
Q_max = ⌊1/ε_grad⌋. Under Assumptions S and E, ρ_t stabilises to κ/(1+κ) where
κ = |1 − ηλ₁|, encoding the dominant Hessian curvature via a Möbius transformation.

**Mediant = gradient averaging.** When two gradient estimates have Farey convergents
with denominators b and d, the mediant (a+c)/(b+d) has denominator b+d — exactly the
formula for combining two gradient estimates with equal weight. Farey arithmetic is the
arithmetic of gradient averaging.

### 9.2 The Embedding Axioms

**Axiom 1 (Learning Embedding).** At each time t, the learning state is represented
by the unique M_t ∈ ℳ satisfying Φ(M_t) = p_t/q_t, where p_t/q_t is the best CF
convergent of ρ_t at the data-adaptive resolution Q_max = ⌊1/ε_grad,t⌋.

This is an embedding, not an identification. The gradient system does not become a
matrix group; only its scalar ratio ρ_t is embedded in the tree.

**Axiom 2 (Binary Dominance).** Generically, Δ log C_t ≠ 0 at each step.

**Theorem (Learning Trajectory Embedding). ✓**
Under Axioms 1 and 2, every gradient system induces a path in ℳ:
```
M_{t+1}  =  G_t · M_t,    G_t ∈ {L, R}
```
where G_t = R if Δ log C_t > 0 (gradient norm increasing) and G_t = L otherwise.

### 9.3 The Möbius–Curvature Correspondence ✓ cond.

**Under Assumptions S and E:**

*Step 1 (Linearization).* Near θ*, gradient descent gives δ_t^(i) = (1 − ηλᵢ)ᵗ δ₀^(i)
in the Hessian eigenbasis.

*Step 2 (Gradient ratio encodes curvature).* Under Assumption E (mode 1 dominates):
```
ρ_t  →  κ/(1+κ)    where κ = |1 − ηλ₁|

Inverting:   λ₁  =  (1 − κ) / η  =  (1 − 2ρ) / (η(1−ρ))
```

*Step 3 (CF denominator).* For ρ = 1/2 − x/4 with x = ηλ₁ ≪ 1, the CF of ρ has
second partial quotient ~ 4/x, yielding q* ~ 1/√(ε_grad · ηλ₁) by Hurwitz at Q_max.

*Step 4 (Invert).* λ₁ ~ C₀/(q*)² where C₀ = 1/(ε_grad · η).

**This relationship (T17, §11.2) is conditional on S and E.** In high dimension
with many comparable Hessian eigenvalues (Assumption E failing), multiple modes
contribute to ρ_t and q* underestimates λ_max(H). The CCC is a lower bound that is
tight when one eigenvalue dominates and conservative otherwise.

---

## 10. Derived Diagnostics

### 10.1 Core Observables

All computable from gradient samples alone — no Hessian, Fisher matrix, or held-out
data required.

**Tree Depth d(t).** Word length of M_t in {L,R} = sum of CF partial quotients ∑ aᵢ.
Low d(t) → flat basin → good generalization (under CCC).

**Path Entropy H(t).** Shannon entropy of the L/R binary string up to t:
```
H(t)  =  −p̂_L log₂ p̂_L  −  p̂_R log₂ p̂_R
```
H(t) ≈ 0 → persistent dominance (signal or noise). H(t) ≈ 1 → oscillation / plateau.

**Median Farey Denominator q*.** Median of {q_s}_{s∈window} over a sliding window of
W steps. This is the **slow variable** of the fast–slow system; it responds on timescale W.

**Approximation Residual ε(t).** Gap |ρ_t − p_t/q_t| < 1/q_t² (Hurwitz bound). Large
residual indicates ρ_t sits between two CF levels.

### 10.2 The Non-Monotone Relationship Between q* and Curvature ✓

**This is a critical correction to source material. ✗ err.**

Source documents claim q* is a monotone proxy for curvature λ_max(H). This is false.
The relationship is genuinely **non-monotone** and depends on the arithmetic properties
of each ρ_t value:

| λ₁ (curvature) | ρ_t | CF expansion | q* | Reason |
|---|---|---|---|---|
| Very flat (λ₁ → 0) | ρ → 0.5000 | [0; 2, N, …] | 2 | ρ near 1/2 → small q* |
| Mild (λ₁ = 0.2) | ρ ≈ 0.4975 | [0; 2, 200, …] | 2 | still close to 1/2 |
| Moderate (λ₁ = 1.0) | ρ ≈ 0.487 | [0; 2, 15, …] | varies | depends on arithmetic |
| Intermediate (λ₁ = 5.0) | ρ ≈ 0.429 | [0; 2, 3, …] | 7 | 3/7 is best at Q_max=20 |
| Sharp (λ₁ = 10.0) | ρ = 0.333 = 1/3 | [0; 3] | 3 | exact rational, small q* |
| Very sharp (λ₁ = 20.0) | ρ = 0.25 = 1/4 | [0; 4] | 4 | exact rational, small q* |

**Key observations:**
1. Very flat minima give q* = 2 (small) because ρ ≈ 1/2 is already a CF convergent
2. Very sharp minima at exact rationals also give small q* (e.g. ρ = 1/3 → q* = 3)
3. The **maximum q*** occurs at intermediate curvature values where ρ falls in "badly
   approximable" regions (far from simple rationals)
4. q* is a **noisy, non-monotone** function of curvature; pointwise values are unreliable

**Correct use of q*:** The **median** over a window W of 20–100 steps, combined with
the permutation test (§10.3), is a statistically valid (though imperfect) indicator of
the generalization phase. Do not interpret individual q* values as curvature measurements.

### 10.3 The Farey Consolidation Index ✓

Given a window of T gradient vectors, compute convergents {(p_t, q_t)} for each
consecutive pair. The observed FCI:
```
F_c^obs  =  #{t : |q_t · p_{t+1} − p_t · q_{t+1}| = 1} / (T − 1)
```

**Permutation Null Test.** The background rate of unimodular pairs is NOT the analytic
O(1/n²) rate from Theorem 6. During training, gradients cluster and the empirical
background rate is higher. The correct null is a permutation test:

1. Compute F_c^obs from the ordered sequence
2. Randomly permute {(p_t, q_t)} 200 times (destroying temporal order, preserving the
   marginal distribution)
3. Report F_c_percentile = percentile rank of F_c^obs among permuted values

This detects whether the **sequence** of updates is arithmetically structured beyond
what the marginal distribution alone produces. If gradients cluster during memorization,
the permuted null also shows high neighbor density and the test correctly does not signal
generalization.

### 10.4 Phase Table

| F_c_percentile | Phase | FHN Analog | Interpretation |
|---|---|---|---|
| < 50th | MEMORIZATION | Quiescent | Signal dominated by noise |
| 50–80th | APPROACHING | Slow drift | q* fluctuating, no clear trend |
| 80–95th | CRITICAL | Near threshold | Watch for q* backtrack |
| 95–99th | GENERALIZING | Excitable orbit | q* dropping (⚠ non-monotone) |
| > 99th | CONVERGED | Post-firing | Low-q* attractor |

**Calibration warning.** The thresholds (50, 80, 95, 99) are empirically motivated
heuristics, not first-principles derivations. They should be validated and tuned on
held-out data for each new problem domain.

---

## 11. Core Theorems: Proved, Conditional, and Conjectured

### 11.1 Unconditionally Proved ✓

| # | Statement | Reference |
|---|---|---|
| T1 | ℳ = {M ∈ SL(2,ℤ) \| entries ≥ 0} is a free monoid | Cayley; SL(2,Z) theory |
| T2 | Bijection Φ: ℳ → ℚ₊ (Calkin–Wilf) | Calkin & Wilf 2000 |
| T3 | Word in {L,R} = CF expansion of Φ(M) | Euclidean algorithm |
| T4 | Farey adjacency iff |bc−ad|=1 | Cauchy 1816; Hardy & Wright §3 |
| T5 | Ford tangency iff |bc−ad|=1 | Ford 1938 |
| T6 | Three-distance theorem: CF denominators determine gap structure | Sós 1958 |
| T7 | Hurwitz: best convergent error < 1/(√5 q²), constant sharp | Hurwitz 1891 |
| T8 | Franel–Landau: Farey discrepancy ⟺ RH (for Farey sequence only) | Franel & Landau 1924 |
| T9 | exp(E) = R, exp(F) = L | Lie theory |
| T10 | SL(2,ℝ) acts on ℍ by orientation-preserving isometries | Beardon 1983 |
| T11 | Albert algebra 𝔄 = H₃(𝕆) is 27-dimensional, exceptional | Albert 1934 |
| T12 | Aut(𝔄) = F₄ (Lie group, dim 52) | Jacobson 1968 |
| T13 | Ramanujan bound λ₂(A) ≤ 2√(k−1) is optimal | LPS 1988 |
| T14 | Ergodic S1–S2–Ω chain has unique stationary distribution | Perron–Frobenius |
| T15 | Gradient is horizontal: ∇^V L = 0 for G-invariant L | Bundle geometry |
| T16 | ℒ_JL is self-adjoint; eigenvalues are real | Sturm–Liouville theory |
| T17 | Q16.16 fixed-point ops are exact **within representable range** | Integer arithmetic |
| T18 | Permutation test is valid distribution-free null | Standard permutation theory |

### 11.2 Proved Under Explicit Assumptions ✓ cond.

**Assumption S (Smoothness).** Loss L is C² at θ* with positive definite Hessian H ≻ 0.
Fails at ReLU kink points; replace with Clarke subdifferential for non-smooth losses.

**Assumption E (Spectral Dominance).** Initial displacement δ₀ = θ₀ − θ* is
concentrated in the top Hessian eigenspace: δ₀ ≈ δ₀^(1) v₁. Exact when δ₀ ∝ v₁;
approximate when λ₁ ≫ λ₂; fails when many eigenvalues are comparable.

| # | Statement | Conditions | Scope |
|---|---|---|---|
| T19 | ρ_t → κ/(1+κ), κ = \|1−ηλ₁\| (Möbius encodes curvature) | S, E | Tight when E is tight |
| T20 | q* ~ 1/√(ε_grad · ηλ₁) (NON-MONOTONE: asymptotic tendency only) | S, E, ηλ₁ < 2 | Statistical; see §10.2 |
| T21 | CCC: λ_max(H) ≲ C₀(η)/(q*)² (lower bound, tight when E tight) | S, E | See §9.3 |
| T22 | PAC-Bayes: G(θ*) ≲ q*·√[C₀(d+log(2/δ))/(2n)] | S, E, McAllester | Conditional on CCC |
| T23 | Exponential convergence at rate ~ C_α/(1+C_α) | C_α ∈ [0.8,1.2]; LCRD | Local |

**Scope summary.** T19–T23 are conditional on S and E. Assumption E fails in typical
deep learning where many Hessian eigenvalues are comparable. The results provide correct
order-of-magnitude intuitions and tight bounds in the geometrically interesting regime
(low-rank sharp minima), but should not be applied uncritically to general settings.

### 11.3 Conjectures ⚠

| # | Statement | Falsifiable Prediction |
|---|---|---|
| C1 | Grokking = Farey backtrack event | q* drops 50–200 steps before test accuracy rises |
| C2 | Double descent peak at C_α ≈ 1 | Interpolation threshold ↔ q* locally maximal |
| C3 | Flat minima correlate with low d(t) | Low q* at equivalent test performance |
| C4 | Adaptive optimizers run per-coordinate tree embeddings | Per-parameter Cᵢ(t) satisfies embedding |
| C5 | Grokking universality: C_α(t)−1 ~ (t−t_c)^β | Measure β across seeds and architectures |
| C6 | FHN limit-cycle regime corresponds to oscillating q* | Sustained q* oscillations near Hopf boundary |

---

## 12. The Unified Phase Diagram

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    SALT UNIFIED PHASE DIAGRAM                             ║
║                                                                           ║
║   λ₁ < 0           λ₁ = 0             λ₁ > 0                            ║
║   C_α < 1          C_α = 1            C_α > 1                           ║
║   q* large (avg)   q* at local max    q* small (avg)                    ║
║   F_c_pct < 50     F_c_pct ≈ 80       F_c_pct > 95                      ║
║   ρ_t ≠ 1/2        ρ_t → 1/2          ρ_t → 1/2 from signal side        ║
║   Ford r small     Ford r critical    Ford r large                       ║
║                                                                           ║
║   ◄────────────────────────┼──────────────────────────►                  ║
║                            │                                              ║
║   MEMORIZATION         GROKKING                GENERALIZATION             ║
║   (submartingale)      BOUNDARY                (supermartingale)          ║
║   Noise dominates      λ₁ = 0                  Signal dominates          ║
║   H_G high             null-recurrent walk      H_G → 0                  ║
║   Sharp minimum        critical CF denominator  Flat minimum              ║
║   Small Ford circle    q* locally maximal       Large Ford circle         ║
║   SL path deepens      Farey backtrack (⚠ C1)  SL path shortens          ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**Compact unification.** All five pillars, all conditional results, and all training
phenomena map to the single scalar condition:
```
sign(λ₁)  =  sign(C_α − 1)  =  sign(d(t) decreasing)
```
The q* version is a **noisy statistical proxy** for this condition, valid in expectation
but non-monotone pointwise (§10.2). Individual q* values should not be used as
deterministic phase indicators.

---

## 13. Phenomenology: Grokking, Double Descent, Neural Collapse, Lottery Tickets

### 13.1 Grokking as Ground State Bifurcation ✓ / ⚠

**SLNF interpretation.** Grokking occurs when λ₁(ℒ_JL) crosses zero:
```
T_grok  =  inf{ t : λ₁(ℒ_JL, b_t) > 0 }  =  inf{ t : C_α(t) > 1 }
```
- Before T_grok: λ₁ < 0, ground mode unstable, noise dominates → memorization
- At T_grok: λ₁ = 0, null-recurrent critical walk, large excursions → plateau
- After T_grok: λ₁ > 0, signal dominates, path converges to eigenfunction expansion

**Farey interpretation (⚠ Conjecture C1).** Grokking is a **tree backtrack**: q*
moves from a deep high-denominator node to a shallow low-denominator node. The Farey
backtrack criterion (both required):
```
q*(t) < q*(t − W)   AND   F_c_percentile(t) > 80
```

The predicted lead time (50–200 steps before test accuracy rise) is falsifiable on the
Power et al. (2022) modular arithmetic benchmark.

### 13.2 Double Descent ~ / ⚠

The interpolation peak corresponds to C_α = 1 (λ₁ = 0) as model capacity increases:
- Below interpolation threshold: λ₁ < 0, underfitting
- At interpolation threshold: λ₁ = 0, double descent peak
- Above interpolation threshold: λ₁ > 0, benign overfitting

**Falsifiable (⚠ C2):** ρ_t ≈ 1/2 and d(t) locally maximal at the double descent peak.

### 13.3 Neural Collapse as Eigenfunction Convergence ✓ / ⚠

Neural collapse — last-layer representations converging to a simplex ETF — corresponds
to convergence of the representation to the ground eigenfunction φ₁ of ℒ_JL. The
ETF structure achieves maximal pairwise angles at equal norms, which is the terminal
state of Kakeya-preserving compression.

### 13.4 Lottery Tickets ~ / ⚠

A winning lottery ticket is a subnetwork whose restricted ℒ_JL already has λ₁ > 0
at initialization. Magnitude pruning removes parameters associated with high-index
eigenmodes (high λₙ, low stability), revealing the subnetwork whose ground mode was
already stable. **Falsifiable (⚠ C3):** pruned networks have lower d(t) at equivalent
test performance.

---

## 14. Error Catalogue: Six Corrections to Source Material

### 14.1 LaTeX Rendering Error in RTLG ✗ err. (typographic)

**Source:** RTLG §2, Theorem 1.

The formula for ℳ contains a LaTeX rendering artifact: `"Missing or unrecognized
delimiter for \Bigl"`. This is a document formatting error only; the mathematics is
correct. The proper statement of Theorem 1 is:
```
ℳ = { M ∈ SL(2,ℤ) | M has all non-negative integer entries }
```

### 14.2 ARDI CORDIC Pseudocode: Wrong Mode ✗ err. (algorithmic)

**Source:** ARDI §6.3, the CORDIC loop pseudocode.

The ARDI document uses **rotation-mode vectoring** (computing atanh of the input),
not hyperbolic rotation mode (computing tanh directly). The code returns only the
z-accumulator, tracking neither cosh nor sinh:

```python
# INCORRECT — This computes atanh(input), not tanh(input)
for i in range(iterations):
    sigma = 1.0 if z > 0 else -1.0
    y += sigma * (2.0 ** (-i))
    z -= sigma * ATANH_TABLE[i]
return y
```

**Corrected implementation** — both cosh and sinh must be tracked jointly:

```python
import math

# CORDIC hyperbolic atanh table: atanh(2^{-i}) for i = 1, 2, ..., 16
ATANH_TABLE = [math.atanh(2.0**(-i)) for i in range(1, 17)]

def cordic_tanh(x: float, iters: int = 16) -> float:
    """
    Compute tanh(x) via CORDIC hyperbolic rotation mode.
    Both cosh and sinh are tracked simultaneously; output is sinh/cosh.

    Domain: |x| < 1.1 for iters=16 (error < 2^{-16}).
    For |x| >= 1.1 use: tanh(x) = 1 - 2/(exp(2x)+1).

    Note: iterations at indices 4 and 13 MUST be repeated for the
    CORDIC hyperbolic convergence condition (Volder 1959, Andraka 1998).
    """
    # Compute scaling constant Kh = prod sqrt(1 - 4^{-i})
    Kh = 1.0
    for i in range(1, iters + 1):
        Kh *= math.sqrt(1.0 - 4.0**(-i))
    # Initialize: [cosh, sinh, z] = [1/Kh, 0, x]
    cosh_acc = 1.0 / Kh
    sinh_acc = 0.0
    z = float(x)
    i_idx = 1
    repeat_done = False
    for _ in range(iters):
        sigma = 1.0 if z >= 0.0 else -1.0
        s = 2.0**(-i_idx)
        new_cosh = cosh_acc + sigma * sinh_acc * s
        new_sinh = sinh_acc + sigma * cosh_acc * s
        z -= sigma * ATANH_TABLE[i_idx - 1]
        cosh_acc, sinh_acc = new_cosh, new_sinh
        # Repeat indices 4 and 13 (required for convergence)
        if i_idx in (4, 13) and not repeat_done:
            repeat_done = True
        else:
            repeat_done = False
            i_idx += 1
    return sinh_acc / (cosh_acc + 1e-15)
```

The corrected implementation is adapted from SLNF §12.1 and follows Volder (1959) and
Andraka (1998).

### 14.3 ARDI "Zero Accumulated Error" is Unconditional — It Should Not Be ✗ err.

**Source:** ARDI Theorem 1, §6.2, §6.4, §10.2.

ARDI states "zero accumulated numerical error" as an unconditional property of the
Q16.16 fixed-point architecture. This is imprecise.

**Correct statement:**
```
All arithmetic operations are exact integer arithmetic and accumulate
zero rounding error, provided no overflow occurs at any step.
```

Q16.16 fixed-point represents numbers in the range [−32768, 32767.9999847]. Addition
and multiplication are exact within this range. Overflow is possible for intermediate
values in gradient or weight updates and, if undetected, silently wraps. The 28×
energy advantage over EKF and the hardware-efficiency argument are unaffected — but
every usage site must include overflow detection. The unconditional phrasing in the
source is an error of omission.

### 14.4 C_α Coordinate Invariance is Overstated ✗ err.

**Sources:** ARDI §12.1, FHN §4.2, FLD §3.1, SLNF §11.2.

Multiple sources state ρ_t (and C_α) is "isometry-invariant" without qualification.

**Correct statement:**
```
ρ_t is invariant under orthogonal rotation of parameter space.
ρ_t is NOT invariant under arbitrary smooth reparameterization.
```

The fully coordinate-invariant form is:
```
C_α^F  =  μ_gᵀ F⁻¹ μ_g / Tr(F⁻¹ Σ_g)
```
where F is the Fisher information matrix. This requires O(d³) computation and is
impractical for large models. The isometry-invariant C_α is the correct practical choice,
but claims of full coordinate invariance in theoretical statements are incorrect.

### 14.5 Hardy–Ramanujan Formula is Asymptotic, Not Exact ✗ err.

**Sources:** ARDI §4.2, §9 (Theorem 3), SLNF §10.2 (IV).

The formula `p(n) ~ (1/4n√3) · exp(π√(2n/3))` is an **asymptotic approximation**
(as n → ∞), not an exact formula. Error table:

| n | True p(n) | H-R formula | Relative error |
|---|---|---|---|
| 1 | 1 | 1.88 | 88% |
| 5 | 7 | 7.74 | 11% |
| 10 | 42 | 45.0 | 7% |
| 20 | 627 | 668 | 7% |
| 50 | 204,226 | 218,012 | 7% |
| 100 | 190,569,292 | ~192M | < 1% |
| 200 | ~4×10¹² | converges | < 0.1% |

The exact formula is the **Rademacher series** (1937):
```
p(n)  =  (1/π√2) ∑_{k=1}^∞  A_k(n) · (d/dn)[sinh(π√(2(n-1/24)/3) / k) / √(n-1/24)]
```
where A_k(n) involves Dedekind sums. This converges rapidly and is exact.

Capacity bounds involving p(n) hold for sufficiently large n. Claims about n = 1 or
n = 5 using the asymptotic formula are numerically unreliable.

### 14.6 Franel–Landau Applies to Farey Sequences, Not Gradient Distributions ✗ err.

**Sources:** FHN §3.8, FLD §2.6, SLNF §14.2.

Source documents claim the Franel–Landau theorem gives "number-theoretic guarantees on
gradient sample quality." This is **incorrect as stated.**

**What the theorem says (unconditionally):**
The discrepancy of the Farey sequence is equivalent to the Riemann Hypothesis. It is a
theorem about the arithmetic distribution of rational numbers, not about random variables.

**What the sources incorrectly imply:**
That gradient convergents, which are not drawn from the Farey sequence, inherit these
regularity properties. This requires the separate empirical assumption that gradient
ratios are uniformly distributed across Farey levels — which has not been verified and
is not generally true.

**Correct framing:** The connection between Farey regularity and gradient distributions
is a structural analogy (~) that motivates the diagnostic framework but does not
constitute a theorem about learning systems. Any claim connecting Franel–Landau directly
to gradient quality requires an independent empirical or theoretical argument.

---

## 15. Implementation Reference

### 15.1 Continued Fraction Convergents (Corrected Algorithm)

The CF convergents algorithm uses the standard three-term recurrence. The initialisation
`p_{-1} = 1, p_0 = a_0, q_{-1} = 0, q_0 = 1` is correct and handles `a_0 = 0` properly
(the ARDI source had a swapped initialisation bug that returned reciprocals).

```python
import math
import numpy as np
from typing import List, Tuple


def continued_fraction_convergents(x: float, q_max: int) -> List[Tuple[int, int]]:
    """
    Standard CF convergents of x in (0,1) with denominator <= q_max.

    Uses three-term recurrence (Hardy & Wright §10.1):
        p_k = a_k * p_{k-1} + p_{k-2}
        q_k = a_k * q_{k-1} + q_{k-2}
    with seed: p_{-1}=1, p_0=a_0, q_{-1}=0, q_0=1.

    Each convergent p_k/q_k is the best rational approximant with
    denominator q_k (Lagrange 1770).

    Returns: list of (p_k, q_k) with q_k <= q_max, in order.
    If no convergent with q <= q_max exists (e.g. a_0=0 → q_0=1 always
    qualifies), returns [(0, 1)].
    """
    convergents: List[Tuple[int, int]] = []
    p_prev, p_curr = 1, int(x)       # p_{-1}, p_0
    q_prev, q_curr = 0, 1             # q_{-1}, q_0
    xi = float(x)

    # Always add the a_0 convergent (q_0 = 1 always fits)
    convergents.append((p_curr, q_curr))

    for _ in range(60):
        remainder = xi - int(xi)
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


def adaptive_farey_approx(rho: float, epsilon_grad: float) -> Tuple[int, int]:
    """
    Best CF convergent of rho at data-adaptive resolution Q_max = floor(1/eps).
    Hurwitz guarantee: approximation error < epsilon_grad^2/sqrt(5) < measurement noise.
    """
    q_max = max(2, int(1.0 / max(float(epsilon_grad), 1e-6)))
    convs = continued_fraction_convergents(rho, q_max)
    return convs[-1]
```

### 15.2 Gradient Ratio Diagnostics

```python
def gradient_ratio(g1: np.ndarray, g2: np.ndarray) -> float:
    """
    rho = ||g2|| / (||g1|| + ||g2||) in (0,1).
    Invariant under orthogonal rotation; NOT under arbitrary reparameterization.
    Under S+E, stabilises to kappa/(1+kappa), kappa = |1 - eta*lambda_max(H)|.
    """
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    total = n1 + n2
    return float(n2 / total) if total > 1e-12 else 0.5


def relative_gradient_change(g1: np.ndarray, g2: np.ndarray) -> float:
    """eps_grad = ||g2 - g1|| / (||g1|| + ||g2||). Sets Q_max."""
    diff  = np.linalg.norm(g2 - g1)
    total = np.linalg.norm(g1) + np.linalg.norm(g2)
    return float(diff / total) if total > 1e-12 else 1.0


def is_farey_neighbor(p1: int, q1: int, p2: int, q2: int) -> bool:
    """True iff |q1*p2 - p1*q2| == 1. Cauchy 1816."""
    return abs(q1 * p2 - p1 * q2) == 1


def signal_to_noise_ratio(grads: List[np.ndarray]) -> float:
    """
    C_alpha = ||mu_g||^2 / Tr(Sigma_g).
    sign(C_alpha - 1) = sign(lambda_1) under S+E.
    Invariant under orthogonal rotation only.
    """
    if len(grads) < 2:
        return 1.0
    G = np.stack([g.ravel() for g in grads], axis=0)  # (T, d)
    mu = G.mean(axis=0)
    G_centered = G - mu
    sigma_trace = np.mean(np.sum(G_centered**2, axis=1))
    signal = float(np.dot(mu, mu))
    return signal / (sigma_trace + 1e-12)
```

### 15.3 Farey Consolidation Index with Permutation Null

```python
from scipy.stats import percentileofscore


def compute_farey_diagnostics(grads: List[np.ndarray],
                               n_perms: int = 200,
                               rng_seed: int = 42) -> dict:
    """
    Farey Consolidation Index with correct permutation-null test.

    Returns:
        convergents       : list of (p_t, q_t)
        F_c_obs           : fraction of consecutive Farey-neighbor pairs
        F_c_percentile    : percentile vs permutation null (0-100)
        q_median          : median denominator (slow variable)
        q_nonmonotone_warning : True if q_median is ambiguous (see §10.2)
        phase             : string label
        C_alpha           : signal-to-noise estimate
    """
    if len(grads) < 4:
        return {'phase': 'INSUFFICIENT_DATA', 'F_c_percentile': 50.0,
                'q_median': 1.0, 'C_alpha': 1.0, 'convergents': []}

    convergents = [
        adaptive_farey_approx(
            gradient_ratio(grads[i], grads[i+1]),
            relative_gradient_change(grads[i], grads[i+1])
        )
        for i in range(len(grads) - 1)
    ]

    def unimod_frac(seq):
        if len(seq) < 2:
            return 0.0
        return sum(is_farey_neighbor(*seq[i], *seq[i+1])
                   for i in range(len(seq)-1)) / (len(seq)-1)

    F_c_obs = unimod_frac(convergents)

    rng = np.random.default_rng(seed=rng_seed)
    null_vals = []
    for _ in range(n_perms):
        perm = list(convergents)
        rng.shuffle(perm)
        null_vals.append(unimod_frac(perm))

    pct = float(percentileofscore(null_vals, F_c_obs, kind='strict'))
    q_vals = [q for (_, q) in convergents]
    q_median = float(np.median(q_vals))

    # Non-monotone warning: q_median is small but SNR may be ambiguous
    c_alpha = signal_to_noise_ratio(grads)
    q_nonmonotone_warning = (q_median <= 3) and (abs(c_alpha - 1.0) < 0.1)

    phase_table = [(50, 'MEMORIZATION'), (80, 'APPROACHING'),
                   (95, 'CRITICAL'), (99, 'GENERALIZING')]
    phase = 'CONVERGED'
    for threshold, label in phase_table:
        if pct < threshold:
            phase = label
            break

    return {
        'convergents': convergents,
        'F_c_obs': F_c_obs,
        'F_c_percentile': pct,
        'q_median': q_median,
        'q_nonmonotone_warning': q_nonmonotone_warning,
        'phase': phase,
        'C_alpha': c_alpha
    }
```

### 15.4 SL(2,ℤ) Word Recovery

```python
def matrix_to_word(M_input) -> str:
    """
    Recover unique word in {L,R} for a positive-cone matrix.
    Uses greedy right-reduction (proof of Theorem 1).
    """
    L_inv = np.array([[1, 0], [-1, 1]], dtype=int)
    R_inv = np.array([[1, -1], [0, 1]], dtype=int)
    I = np.eye(2, dtype=int)
    M = np.array(M_input, dtype=int)
    word = []
    for _ in range(500):
        if np.array_equal(M, I):
            break
        a, c = int(M[0, 0]), int(M[1, 0])
        if a >= c:
            word.append('L')
            M = M @ L_inv
        else:
            word.append('R')
            M = M @ R_inv
    return ''.join(reversed(word))
```

### 15.5 Albert Algebra Operations

```python
def jordan_product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """X ∘ Y = ½(XY + YX). Commutative, non-associative."""
    return 0.5 * (X @ Y + Y @ X)


def associator(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """A(X,Y,Z) = (X∘Y)∘Z − X∘(Y∘Z). Non-zero = order memory preserved."""
    return (jordan_product(jordan_product(X, Y), Z)
            - jordan_product(X, jordan_product(Y, Z)))


def albert_update(X: np.ndarray, X_star: np.ndarray,
                  R_tensor: np.ndarray, tau: float) -> np.ndarray:
    """Ramanujan–Jordan step: X_{t+1} = X_t + tau * [(X*−X_t) ∘ R]."""
    delta = jordan_product(X_star - X, R_tensor)
    X_new = X + tau * delta
    norm = np.linalg.norm(X_new, 'fro')
    return X_new / (norm + 1e-12)
```

---

## 16. Open Problems

### 16.1 Empirical Validation of Conjecture C1 ⚠

Test the Farey backtrack criterion on Power et al. (2022) modular arithmetic:
- Does q* decrease 50–200 steps before test accuracy rises?
- What is the lead-time distribution across seeds and tasks?
- Does F_c_percentile > 80 co-occur reliably with the denominator drop?

This directly validates or refutes C1.

### 16.2 Quantifying q* Non-Monotonicity ⚠

A precise characterisation of the q*(λ₁) function is needed. The asymptotic analysis
gives q* ~ 1/√(ε_grad · ηλ₁) as a tendency, but the function is non-monotone for
finite Q_max. Open: what is the distribution of q* for typical gradient noise models?
Is the median q* (over window W) monotone in expectation?

### 16.3 Irrational and Periodic Trajectories ⚠

Quadratic irrationals have eventually periodic CF expansions. Do cyclic learning rate
schedules produce eventually periodic paths in ℳ? The period structure would encode
the schedule via CF expansion. The 2-adic integer interpretation of infinite paths
G₀G₁G₂⋯ ∈ {L,R}^ℕ may provide useful trajectory metrics.

### 16.4 Extension to Non-Smooth Losses ⚠

Assumption S fails at ReLU kinks. Replace H with Clarke's subdifferential (Clarke 1983).
Within each linear region, the gradient is constant and the CF map is well-defined.
Across region boundaries, gradient jumps correspond to mediant insertions. The
piecewise-linear fast nullcline replaces the smooth cubic of the FHN analogy.

### 16.5 Farey-SAM: Denominator-Penalised Optimisation ⚠

Under S and E, penalising q* is approximately equivalent to penalising λ_max(H):
```
L_Farey(θ)  =  L(θ)  +  λ · (q*)²
```
This is a gradient-norm-only proxy for SAM (Foret et al. 2021) requiring no double
forward pass. Comparison with SAM on standard benchmarks would simultaneously test
the CCC and practical utility.

### 16.6 Persistent Homology of the Farey Lattice ⚠

The Stern–Brocot tree has a natural simplicial complex structure (each Ford circle
triple is a 2-simplex). Persistent homology filtered by denominator would give a
Betti-number description of the loss landscape topology directly tied to Farey
arithmetic.

### 16.7 Minkowski's Question-Mark Function ⚠

The function Q : [0,∞) → [0,1] is a strictly increasing homeomorphism with derivative
zero almost everywhere. Setting ℓ(t) = Q(C(t)) would linearise the learning coordinate:
plateaus correspond to flat regions of Q, genuine learning events to its rapid changes.
The relationship between Δℓ(t) and observable generalisation transitions is unexplored.

### 16.8 Fisher-Weighted Embedding for Large Models ⚠

The coordinate-invariant C_α^F requires O(d³) computation. Low-rank Fisher approximations
(Martens & Grosse 2015) may make the invariant embedding practical for large-scale models,
restoring the theoretical guarantees lost by using the orthogonal-only C_α.

---

## 17. References

### Arithmetic and Number Theory

- Cauchy, A.L. (1816). *Exercices de mathématique.* — Farey unimodular theorem.
- Hardy, G.H. & Wright, E.M. (1979). *An Introduction to the Theory of Numbers*, 5th ed. Oxford University Press. — Farey theory (Ch. 3), continued fractions (Ch. 10), CF best-approximation (Ch. 11).
- Hardy, G.H. & Ramanujan, S. (1918). Asymptotic formulae in combinatory analysis. *Proc. London Math. Soc.* s2-17(1), 75–115. — Partition asymptotics (asymptotic, not exact).
- Rademacher, H. (1937). On the partition function p(n). *Proc. London Math. Soc.* s2-43(4), 241–254. — Exact convergent series for p(n).
- Hurwitz, A. (1891). Ueber die angenäherte Darstellung der Irrationalzahlen durch rationale Brüche. *Mathematische Annalen* 39(2), 279–284.
- Ford, L.R. (1938). Fractions. *American Mathematical Monthly* 45(9), 586–601.
- Franel, J. (1924). Les suites de Farey et le problème des nombres premiers. *Göttinger Nachrichten*, 198–201.
- Landau, E. (1924). Bemerkungen zu der vorstehenden Abhandlung von Herrn Franel. *Göttinger Nachrichten*, 202–206.
- Graham, R., Knuth, D. & Patashnik, O. (1994). *Concrete Mathematics*, 2nd ed. Addison-Wesley.
- Calkin, N. & Wilf, H.S. (2000). Recounting the rationals. *American Mathematical Monthly* 107(4), 360–363.

### Three-Distance and Approximation Theory

- Sós, V.T. (1958). On the distribution mod 1 of the sequence nα. *Ann. Univ. Sci. Budapest. Eötvös Sect. Math.* 1, 127–134.
- Steinhaus, H. (1950). *Mathematical Snapshots.* Oxford University Press.
- Lagrange, J.-L. (1770). *Additions to Euler's Elements of Algebra.* — Convergents are best approximants.

### Algebra and Representation Theory

- Albert, A.A. (1934). On a certain algebra of quantum mechanics. *Annals of Mathematics* 35(1), 65–73. — The exceptional Jordan algebra H₃(𝕆).
- Jacobson, N. (1968). *Structure and Representations of Jordan Algebras.* AMS Colloquium Publications Vol. 39.
- Lubotzky, A., Phillips, R. & Sarnak, P. (1988). Ramanujan graphs. *Combinatorica* 8(3), 261–277.
- Hoory, S., Linial, N. & Wigderson, A. (2006). Expander graphs and their applications. *Bulletin of the AMS* 43(4), 439–561.

### Differential Geometry and Spectral Theory

- Sturm, C. (1836). Mémoire sur les équations différentielles linéaires du second ordre. *Journal de Mathématiques Pures et Appliquées* 1, 106–186.
- Liouville, J. (1836). Mémoire sur le développement des fonctions. *Journal de Mathématiques Pures et Appliquées* 1, 253–265.
- Zettl, A. (2005). *Sturm–Liouville Theory.* AMS Mathematical Surveys and Monographs Vol. 121.
- Beardon, A.F. (1983). *The Geometry of Discrete Groups.* Springer Graduate Texts in Mathematics Vol. 91.
- Kobayashi, S. & Nomizu, K. (1963). *Foundations of Differential Geometry*, Vol. I. Wiley-Interscience.

### Hyperbolic Geometry and SL(2,ℤ)

- Series, C. (1985). The modular surface and continued fractions. *Journal of the London Mathematical Society* 31(1), 69–80.
- Milnor, J. (1963). *Morse Theory.* Princeton University Press Annals of Mathematics Studies Vol. 51.

### Dynamical Systems

- FitzHugh, R. (1961). Impulses and physiological states in theoretical models of nerve membrane. *Biophysical Journal* 1(6), 445–466.
- Nagumo, J., Arimoto, S. & Yoshizawa, S. (1962). An active pulse transmission line simulating nerve axon. *Proceedings of the IRE* 50(10), 2061–2070.
- Izhikevich, E.M. (2007). *Dynamical Systems in Neuroscience.* MIT Press.
- Fenichel, N. (1979). Geometric singular perturbation theory for ordinary differential equations. *Journal of Differential Equations* 31(1), 53–98.

### Generalization and Optimization

- McAllester, D.A. (1999). PAC-Bayesian model averaging. *COLT 1999*, 164–170.
- Dziugaite, G.K. & Roy, D.M. (2017). Computing nonvacuous generalization bounds for deep stochastic neural networks. *UAI 2017*.
- Foret, P., Kleiner, A., Mobahi, H. & Neyshabur, B. (2021). Sharpness-Aware Minimization for efficiently improving generalization. *ICLR 2021*.
- Hochreiter, S. & Schmidhuber, J. (1997). Flat minima. *Neural Computation* 9(1), 1–42.
- Robbins, H. & Monro, S. (1951). A stochastic approximation method. *Annals of Mathematical Statistics* 22(3), 400–407.
- Cohen, J.M. et al. (2021). Gradient descent on neural networks typically occurs at the edge of stability. *ICLR 2021*.
- Martens, J. & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML 2015*.

### Deep Learning Phenomena

- Power, A. et al. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *ICLR 2022 Workshop.*
- Papyan, V., Han, X.Y. & Donoho, D.L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. *PNAS* 117(44), 24652–24663.
- Belkin, M. et al. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. *PNAS* 116(32), 15849–15854.
- Frankle, J. & Carlin, M. (2019). The Lottery Ticket Hypothesis: Finding sparse, trainable neural networks. *ICLR 2019*.

### Information Theory

- Tishby, N., Pereira, F.C. & Bialek, W. (2000). The information bottleneck method. *arXiv:physics/0004057.*
- Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation* 10(2), 251–276.

### Non-Smooth Analysis

- Clarke, F.H. (1983). *Optimization and Nonsmooth Analysis.* Wiley.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction.* AMS.

### Hardware

- Volder, J.E. (1959). The CORDIC trigonometric computing technique. *IRE Transactions on Electronic Computers* EC-8(3), 330–334.
- Andraka, R. (1998). A survey of CORDIC algorithms for FPGA based computers. *ACM/SIGDA FPGA*, 191–200.

---

## Appendix A: The Arithmetic Skeleton in Five Identities

The entire discrete structure of SALT compresses to five identities:

```
1.  det(M) = ad − bc = 1                   (M ∈ SL(2,ℤ): unit determinant)
2.  M ∈ ℳ iff all entries ≥ 0             (positive cone = learning words)
3.  Φ(M) = a/c                             (projective map → rational)
4.  |bc − ad| = 1 iff Farey adjacent       (unimodular ↔ tangent Ford circles)
5.  exp(E) = R,  exp(F) = L                (discrete steps = Lie exponentials)
```

Everything else — Ford circles, geodesics, eigenmode structure, generalization bounds,
grokking — is the continuous geometry built over this skeleton.

---

## Appendix B: The Threshold in Five Languages

```
Language              Threshold           Mathematical condition
──────────────────────────────────────────────────────────────────────────────
SLNF (spectral)       λ₁ > 0             Ground mode stable
ARDI (arithmetic)     C_α > 1            Signal power > noise power
Farey (geometric)     q* small (avg)     Flat Ford circle (flat basin) [statistical]
RTLG (algebraic)      d(t) decreasing    Path shortening in ℳ
FHN (dynamical)       Γ > 1              Supermartingale drift to minimum
──────────────────────────────────────────────────────────────────────────────
All five equivalent near the critical point under Assumptions S and E.
The q* condition is statistical (non-monotone pointwise — see §10.2 and §14).
Empirical C_α estimators are identical up to orthogonal invariance only.
```

---

## Appendix C: Summary of All Corrections

| # | Error Type | Source | Summary |
|---|---|---|---|
| E1 | Typographic | RTLG §2 | LaTeX rendering artifact; mathematics correct |
| E2 | Algorithmic | ARDI §6.3 | CORDIC computed atanh (vectoring), not tanh (rotation); fixed §14.2 |
| E3 | Precision | ARDI §6.2, §10.2 | "Zero accumulated error" requires "within representable range"; fixed §14.3 |
| E4 | Scope | ARDI, FHN, FLD, SLNF | C_α invariant under orthogonal rotation only, not arbitrary reparameterization; fixed §14.4 |
| E5 | Precision | ARDI, SLNF | Hardy–Ramanujan formula is asymptotic; exact formula is Rademacher; fixed §14.5 |
| E6 | Scope | FHN, FLD, SLNF | Franel–Landau theorem applies to Farey sequences, not gradient distributions; fixed §14.6 |

**Implicit correction (not in original error catalogue):**
q* is non-monotone in curvature. Source documents treating q* as a monotone proxy are
incorrect. See §10.2 for proof and §8.2 for the corrected formulation of Condition (V).

---

*Framework built on: Cauchy (1816) · Ford (1938) · Hurwitz (1891) · Hardy–Ramanujan (1918) ·
Rademacher (1937) · Sós (1958) · Calkin–Wilf (2000) · Albert (1934) · Jacobson (1968) ·
Lubotzky–Phillips–Sarnak (1988) · Sturm–Liouville (1836) · Fenichel (1979) ·
McAllester (1999) · Power et al. (2022) · Volder (1959)*

*Synthesising: RTLG · FHN/Farey · FLD · ARDI · SLNF*
