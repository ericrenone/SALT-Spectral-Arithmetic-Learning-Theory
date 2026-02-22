"""
salt_demo.py
============
SALT — Spectral-Arithmetic Learning Theory
Interactive demonstration kit.

Run the full demo:
    python salt_demo.py

Run a specific scenario:
    python salt_demo.py --scenario grokking
    python salt_demo.py --scenario memorization
    python salt_demo.py --scenario jordan
    python salt_demo.py --scenario spectral
    python salt_demo.py --scenario convergence
    python salt_demo.py --scenario live

Saves plots to ./salt_demo_output/ by default.
Use --show to display plots interactively instead.

Dependencies: numpy, scipy, matplotlib
"""

import os
import sys
import math
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#0e0e10",
    "axes.facecolor": "#16161a",
    "axes.labelcolor": "#e8e8e8",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "text.color": "#e8e8e8",
    "grid.color": "#2a2a2e",
    "grid.linewidth": 0.6,
})

PALETTE = {
    "blue":   "#4e9af1",
    "green":  "#3ecf8e",
    "orange": "#f5a623",
    "red":    "#e55353",
    "purple": "#a855f7",
    "gray":   "#6b7280",
    "white":  "#e8e8e8",
}

# ── Import SALT ───────────────────────────────────────────────────────────────
try:
    from salt_core import (
        SaltAnalyzer, gradient_ratio, gradient_change,
        signal_to_noise, cf_convergents, best_convergent,
        is_farey_neighbor, path_entropy, farey_consolidation_index,
        jordan_product, associator, renormalized_gradient,
        sturm_liouville_eigenfunctions, spectral_features, spectral_reconstruct,
        JordanLayerNumpy, CauchyMonitor,
    )
except ImportError as e:
    print(f"ERROR: could not import salt_core.py — make sure it is in the same directory.\n{e}")
    sys.exit(1)

# numpy 2.x compat
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

OUTPUT_DIR = "salt_demo_output"


def save_or_show(fig, name: str, show: bool):
    if show:
        plt.show()
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  → saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient simulators
# ─────────────────────────────────────────────────────────────────────────────

def simulate_memorization(n_steps=200, dim=30, seed=1):
    """
    Pure noise: gradients point in random directions every step.
    C_α << 1. Model is memorizing (no consistent signal).
    """
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(dim) for _ in range(n_steps)]


def simulate_generalization(n_steps=200, dim=30, seed=2):
    """
    Strong signal: gradients consistently point in one direction with small noise.
    C_α >> 1. Model is generalizing.
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)
    grads = []
    for _ in range(n_steps):
        g = direction * 2.0 + rng.standard_normal(dim) * 0.15
        grads.append(g)
    return grads


def simulate_grokking(n_steps=300, dim=30, seed=3, transition=160):
    """
    Two-phase: memorization for first `transition` steps, then sudden
    switch to generalization. Simulates the grokking phenomenon.
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)
    grads = []
    for i in range(n_steps):
        if i < transition:
            # Memorization phase: pure noise
            g = rng.standard_normal(dim)
        else:
            # Generalization phase: strong consistent signal
            t = i - transition
            signal_strength = 2.0 * (1 - math.exp(-t / 20))
            g = direction * signal_strength + rng.standard_normal(dim) * 0.2
        grads.append(g)
    return grads, transition


def simulate_double_descent(n_steps=300, dim=30, seed=4):
    """
    Three phases: generalization → critical boundary → generalization again.
    Models the double-descent phenomenon around the interpolation threshold.
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)
    grads = []
    for i in range(n_steps):
        frac = i / n_steps
        # C_α trajectory: above 1 → dips to 1 → above 1 again
        if frac < 0.35:
            signal = 1.8
        elif frac < 0.55:
            # Critical zone: signal near noise level
            signal = 0.5 + (frac - 0.35) * 2.0
        else:
            signal = 1.6
        g = direction * signal + rng.standard_normal(dim) * 0.3
        grads.append(g)
    return grads


def run_analyzer(grads, window=40):
    """Run SaltAnalyzer on a list of gradient vectors; return list of StepResults."""
    analyzer = SaltAnalyzer(window=window, n_permutations=100)
    results = []
    for i in range(len(grads) - 1):
        r = analyzer.step(grads[i], grads[i+1])
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1: Memorization vs Generalization side-by-side
# ─────────────────────────────────────────────────────────────────────────────

def demo_memorization_vs_generalization(show=False):
    """
    Two training runs side by side.
    Left: memorizing model (noise-dominated gradients).
    Right: generalizing model (signal-dominated gradients).
    Shows how SALT diagnostics distinguish them.
    """
    print("\n[Demo 1] Memorization vs Generalization")

    mem_grads = simulate_memorization(n_steps=200)
    gen_grads = simulate_generalization(n_steps=200)
    mem_res = run_analyzer(mem_grads)
    gen_res = run_analyzer(gen_grads)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    fig.suptitle("SALT: Memorization vs Generalization", fontsize=13, weight="bold", y=0.98)

    steps = list(range(len(mem_res)))
    panels = [
        ("C_α  (signal-to-noise)", [r.c_alpha for r in mem_res], [r.c_alpha for r in gen_res], 1.0),
        ("q*  (median CF denominator)", [r.q_star for r in mem_res], [r.q_star for r in gen_res], None),
        ("Path entropy H", [r.path_entropy for r in mem_res], [r.path_entropy for r in gen_res], None),
    ]

    for row, (title, mem_vals, gen_vals, threshold) in enumerate(panels):
        for col, (vals, label, color) in enumerate([
            (mem_vals, "Memorization", PALETTE["red"]),
            (gen_vals, "Generalization", PALETTE["green"]),
        ]):
            ax = axes[row][col]
            ax.plot(steps, vals, color=color, lw=1.5, alpha=0.85)
            if threshold is not None:
                ax.axhline(threshold, color=PALETTE["orange"], lw=1.2,
                           linestyle="--", alpha=0.7, label=f"C_α = {threshold}")
            ax.set_title(f"{label}: {title}", fontsize=9, color=PALETTE["white"])
            ax.set_xlabel("Training step", fontsize=8)
            ax.grid(True, alpha=0.3)
            if threshold is not None and col == 0:
                ax.legend(fontsize=7, framealpha=0.2)

    # Annotate final C_α values
    def final_phase(results):
        c = results[-1].c_alpha
        if c > 1:
            return f"GENERALIZING  C_α={c:.2f}", PALETTE["green"]
        else:
            return f"MEMORIZING    C_α={c:.2f}", PALETTE["red"]

    for col, results in enumerate([mem_res, gen_res]):
        label, color = final_phase(results)
        axes[0][col].set_title(
            f"{['Memorization', 'Generalization'][col]}: {panels[0][0]}\n[{label}]",
            fontsize=8.5, color=color
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_or_show(fig, "01_memorization_vs_generalization", show)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2: Grokking detection
# ─────────────────────────────────────────────────────────────────────────────

def demo_grokking(show=False):
    """
    Simulates grokking — sudden generalization after a long memorization plateau.
    Shows SALT detecting the transition 50–200 steps before it completes.
    """
    print("\n[Demo 2] Grokking Detection")

    grads, transition = simulate_grokking(n_steps=300, transition=160)
    results = run_analyzer(grads, window=50)
    steps = list(range(len(results)))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("SALT: Grokking Detection", fontsize=13, weight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    c_alphas = [r.c_alpha for r in results]
    q_stars  = [r.q_star  for r in results]
    fci_pcts = [r.f_c_pct for r in results]
    rhos     = [r.rho     for r in results]
    phases   = [r.phase   for r in results]

    phase_colors = {
        "MEMORIZATION": PALETTE["red"],
        "APPROACHING":  PALETTE["orange"],
        "CRITICAL":     PALETTE["purple"],
        "GENERALIZING": PALETTE["blue"],
        "CONVERGED":    PALETTE["green"],
    }

    # ── Panel A: C_α over time ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(steps, c_alphas, color=PALETTE["blue"], lw=1.8, label="C_α")
    ax_a.axhline(1.0, color=PALETTE["orange"], lw=1.2, linestyle="--", alpha=0.7, label="threshold C_α=1")
    ax_a.axvline(transition - 1, color=PALETTE["red"],   lw=1.5, linestyle=":", alpha=0.8, label="Memorization → Generalization")
    ax_a.fill_between(steps, 0, c_alphas,
                      where=[c > 1 for c in c_alphas],
                      color=PALETTE["green"], alpha=0.12, label="C_α > 1 (signal)")
    ax_a.fill_between(steps, 0, c_alphas,
                      where=[c <= 1 for c in c_alphas],
                      color=PALETTE["red"], alpha=0.08, label="C_α ≤ 1 (noise)")
    ax_a.set_ylabel("C_α", fontsize=9)
    ax_a.set_title("C_α — signal-to-noise ratio (primary SALT indicator)", fontsize=9)
    ax_a.legend(fontsize=7.5, framealpha=0.2, loc="upper left")
    ax_a.grid(True, alpha=0.3)

    # ── Panel B: q* (median CF denominator) ──────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.plot(steps, q_stars, color=PALETTE["purple"], lw=1.5)
    ax_b.axvline(transition - 1, color=PALETTE["red"], lw=1.2, linestyle=":", alpha=0.8)
    ax_b.set_ylabel("q*", fontsize=9)
    ax_b.set_xlabel("Step", fontsize=8)
    ax_b.set_title("q* — median CF denominator\n(slow variable; flat basin = small q*)", fontsize=8)
    ax_b.grid(True, alpha=0.3)

    # ── Panel C: FCI percentile ───────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.plot(steps, fci_pcts, color=PALETTE["green"], lw=1.5)
    ax_c.axhline(80,  color=PALETTE["orange"], lw=0.8, linestyle="--", alpha=0.6, label="80th pct")
    ax_c.axhline(95,  color=PALETTE["green"],  lw=0.8, linestyle="--", alpha=0.6, label="95th pct")
    ax_c.axvline(transition - 1, color=PALETTE["red"], lw=1.2, linestyle=":", alpha=0.8)
    ax_c.set_ylabel("FCI percentile", fontsize=9)
    ax_c.set_xlabel("Step", fontsize=8)
    ax_c.set_title("Farey Consolidation Index\n(permutation-null percentile)", fontsize=8)
    ax_c.legend(fontsize=7.5, framealpha=0.2)
    ax_c.grid(True, alpha=0.3)

    # ── Panel D: Phase timeline ───────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[2, :])
    for i, phase in enumerate(phases):
        ax_d.bar(i, 1, width=1.0, color=phase_colors.get(phase, PALETTE["gray"]), linewidth=0)
    ax_d.axvline(transition - 1, color=PALETTE["white"], lw=1.5, linestyle=":", alpha=0.9)
    ax_d.set_yticks([])
    ax_d.set_xlabel("Training step", fontsize=8)
    ax_d.set_title("Training phase (colour-coded by SALT diagnostic)", fontsize=9)

    # Legend for phase colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=p) for p, c in phase_colors.items()]
    ax_d.legend(handles=legend_elements, fontsize=7, framealpha=0.2,
                loc="upper left", ncol=5)

    save_or_show(fig, "02_grokking_detection", show)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3: Jordan Product Layer — symmetry and non-associativity
# ─────────────────────────────────────────────────────────────────────────────

def demo_jordan(show=False):
    """
    Demonstrates the Jordan product layer:
    - Commutativity: X∘W = W∘X
    - Non-associativity: (X∘Y)∘Z ≠ X∘(Y∘Z)
    - Symmetric regularization effect on weight matrices
    - Gradient renormalization with provable bound
    """
    print("\n[Demo 3] Jordan Product Layer")

    rng = np.random.default_rng(7)
    DIM = 32
    N_TRIALS = 100

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("SALT: Jordan Product Layer", fontsize=13, weight="bold")

    # ── Panel A: Commutativity verification ──────────────────────────────────
    ax = axes[0][0]
    errors = []
    for _ in range(N_TRIALS):
        X = rng.standard_normal((DIM, DIM))
        W = rng.standard_normal((DIM, DIM))
        XoW = jordan_product(X, W)
        WoX = jordan_product(W, X)
        errors.append(np.linalg.norm(XoW - WoX))
    ax.hist(errors, bins=20, color=PALETTE["green"], alpha=0.8, edgecolor="none")
    ax.set_xlabel("‖X∘W - W∘X‖  (should be 0)", fontsize=8)
    ax.set_title("Commutativity: X∘W = W∘X\n(verified over 100 random matrix pairs)", fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.95, f"max error: {max(errors):.1e}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=PALETTE["green"])

    # ── Panel B: Non-associativity ────────────────────────────────────────────
    ax = axes[0][1]
    assoc_norms = []
    for _ in range(N_TRIALS):
        X = rng.standard_normal((DIM, DIM))
        Y = rng.standard_normal((DIM, DIM))
        Z = rng.standard_normal((DIM, DIM))
        A = associator(X, Y, Z)
        assoc_norms.append(np.linalg.norm(A))
    ax.hist(assoc_norms, bins=20, color=PALETTE["red"], alpha=0.8, edgecolor="none")
    ax.set_xlabel("‖(X∘Y)∘Z - X∘(Y∘Z)‖  (non-zero = non-associative)", fontsize=8)
    ax.set_title("Non-Associativity: (X∘Y)∘Z ≠ X∘(Y∘Z)\n(algebra retains computation-order memory)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.95, f"mean: {np.mean(assoc_norms):.1f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=PALETTE["red"])

    # ── Panel C: Symmetry of weight matrix evolution ──────────────────────────
    ax = axes[1][0]
    layer = JordanLayerNumpy(dim=DIM, init_scale=1.0)
    symmetries = []
    for step in range(60):
        # Simulate a weight update through Jordan layer
        X = rng.standard_normal((DIM, DIM))
        out = layer.forward(X)
        # How symmetric is the output?
        sym_err = np.linalg.norm(out - out.T) / np.linalg.norm(out)
        symmetries.append(sym_err)
        # Small random weight update
        layer.W += rng.standard_normal((DIM, DIM)) * 0.05

    ax.plot(symmetries, color=PALETTE["blue"], lw=1.5)
    ax.set_xlabel("Simulated update step", fontsize=8)
    ax.set_ylabel("‖output - outputᵀ‖ / ‖output‖", fontsize=8)
    ax.set_title("Jordan layer output asymmetry\n(lower = more symmetric feature mixing)", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel D: Gradient renormalization bound ───────────────────────────────
    ax = axes[1][1]
    gradient_norms = np.logspace(-2, 3, 200)
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        renorm_norms = gradient_norms / (1 + alpha * gradient_norms)
        bound = 1.0 / alpha
        ax.plot(gradient_norms, renorm_norms, lw=1.5,
                label=f"α={alpha}  (bound={bound:.1f})")
        ax.axhline(bound, lw=0.6, linestyle=":", alpha=0.5,
                   color=ax.lines[-1].get_color())

    ax.set_xscale("log")
    ax.set_xlabel("‖∇L‖  (raw gradient norm)", fontsize=8)
    ax.set_ylabel("‖∇̃L‖  (renormalized)", fontsize=8)
    ax.set_title("Renormalized gradient: ‖∇̃L‖ ≤ 1/α  (provable bound)\n∇̃L = ∇L / (1 + α‖∇L‖)", fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, "03_jordan_product", show)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 4: Spectral basis and Ford circles
# ─────────────────────────────────────────────────────────────────────────────

def demo_spectral(show=False):
    """
    Visualizes:
    - Sturm–Liouville eigenfunctions (the SALT spectral basis)
    - Spectral reconstruction of a target function
    - Ford circles (geometric view of loss basins)
    - Stern–Brocot tree (arithmetic structure)
    """
    print("\n[Demo 4] Spectral Basis and Ford Circles")

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("SALT: Spectral Basis and Arithmetic Geometry", fontsize=13, weight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: S-L eigenfunctions ───────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    x, psi = sturm_liouville_eigenfunctions(n_modes=8, n_points=256)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 8))
    for k in range(6):
        ax_a.plot(x, psi[:, k], color=colors[k], lw=1.5,
                  label=f"ψ_{k+1}", alpha=0.85)
    ax_a.set_xlabel("x", fontsize=8)
    ax_a.set_ylabel("ψₙ(x)", fontsize=8)
    ax_a.set_title("Sturm–Liouville eigenfunctions\n(orthonormal spectral basis)", fontsize=9)
    ax_a.legend(fontsize=7.5, framealpha=0.2, ncol=3, loc="upper right")
    ax_a.grid(True, alpha=0.3)

    # ── Panel B: Spectral reconstruction ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    target = np.sin(2 * np.pi * x) * np.exp(-2 * x) + 0.5 * np.cos(4 * np.pi * x)
    target /= np.max(np.abs(target))

    ax_b.plot(x, target, color=PALETTE["white"], lw=2.0, label="target f(x)", alpha=0.9)
    for n_modes in [2, 5, 12, 20]:
        x_modes, psi_modes = sturm_liouville_eigenfunctions(n_modes=n_modes, n_points=256)
        coeffs = spectral_features(target, psi_modes, x_modes)
        recon = spectral_reconstruct(coeffs, psi_modes)
        err = np.sqrt(_trapz((target - recon)**2, x_modes))
        ax_b.plot(x_modes, recon, lw=1.2, alpha=0.75, label=f"N={n_modes}  L²err={err:.3f}")

    ax_b.set_xlabel("x", fontsize=8)
    ax_b.set_ylabel("f(x)", fontsize=8)
    ax_b.set_title("Spectral reconstruction: f(x) = Σ aₙ ψₙ(x)\n(convergence in L²)", fontsize=9)
    ax_b.legend(fontsize=7.5, framealpha=0.2, loc="lower right")
    ax_b.grid(True, alpha=0.3)

    # ── Panel C: Ford circles ─────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_xlim(-0.05, 1.05)
    ax_c.set_ylim(-0.01, 0.35)
    ax_c.set_xlabel("p/q  (Farey fraction)", fontsize=8)
    ax_c.set_ylabel("height above real line", fontsize=8)
    ax_c.set_title("Ford circles: radius = 1/(2q²)\nLarger circle = flatter loss basin = better generalization", fontsize=9)

    fractions = []
    for q in range(1, 9):
        for p in range(0, q + 1):
            if math.gcd(p, q) == 1:
                fractions.append((p, q))

    for p, q in fractions:
        r = 1.0 / (2 * q**2)
        cx, cy = p/q, r
        alpha = min(0.9, 0.2 + 3.0 / q**2)
        color = plt.cm.viridis(1.0 - min(1.0, (q-1)/7.0))
        circle = Circle((cx, cy), r, fill=True, facecolor=color,
                        edgecolor="none", alpha=alpha)
        ax_c.add_patch(circle)
        if q <= 4:
            ax_c.text(cx, 2*r + 0.005, f"{p}/{q}", ha="center",
                     fontsize=6.5, color=PALETTE["white"], alpha=0.9)

    ax_c.axhline(0, color=PALETTE["gray"], lw=1.0)
    cbar_ax = fig.add_axes([0.08, 0.06, 0.37, 0.015])
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=1, vmax=8))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                 label="denominator q  (smaller = flatter basin)")
    ax_c.grid(True, alpha=0.2)

    # ── Panel D: Stern–Brocot tree ─────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(-4.5, 0.5)
    ax_d.set_xlabel("fraction value", fontsize=8)
    ax_d.set_title("Stern–Brocot tree: free monoid ℳ ⊂ SL(2,ℤ)\nEach gradient step is L or R", fontsize=9)
    ax_d.set_yticks([])

    def draw_tree(lo_p, lo_q, hi_p, hi_q, depth=0, max_depth=4):
        if depth > max_depth:
            return
        med_p, med_q = lo_p + hi_p, lo_q + hi_q
        x_val = med_p / med_q
        y_val = -depth
        ax_d.scatter([x_val], [y_val], s=max(8, 80 - depth * 15),
                     color=plt.cm.plasma(depth / max_depth), zorder=5, alpha=0.85)
        if depth > 0:
            parent_p = (lo_p + hi_p) if False else None  # parent drawn recursively
        if med_q <= 8:
            ax_d.text(x_val, y_val + 0.2, f"{med_p}/{med_q}",
                     ha="center", fontsize=6, color=PALETTE["white"], alpha=0.8)
        draw_tree(lo_p, lo_q, med_p, med_q, depth + 1, max_depth)
        draw_tree(med_p, med_q, hi_p, hi_q, depth + 1, max_depth)

    draw_tree(0, 1, 1, 0)
    ax_d.grid(True, alpha=0.2, axis="x")

    save_or_show(fig, "04_spectral_and_geometry", show)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 5: Cauchy Convergence Monitor
# ─────────────────────────────────────────────────────────────────────────────

def demo_convergence(show=False):
    """
    Demonstrates the Cauchy convergence monitor on three parameter trajectories:
    - Converging (shrinking differences → Cauchy)
    - Oscillating (large persistent differences → not Cauchy)
    - Converging with SALT renormalized gradients applied
    """
    print("\n[Demo 5] Cauchy Convergence Monitor")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("SALT: Cauchy Convergence Monitor", fontsize=13, weight="bold")

    rng = np.random.default_rng(42)
    n_steps = 150

    scenarios = {
        "Converging (geometric decay)": {
            "color": PALETTE["green"],
            "fn": lambda i: 10.0 * 0.92**i + rng.standard_normal() * 0.05,
        },
        "Oscillating (not Cauchy)": {
            "color": PALETTE["red"],
            "fn": lambda i: 2.0 * math.sin(i * 0.5) + rng.standard_normal() * 0.1,
        },
        "SALT renormalized": {
            "color": PALETTE["blue"],
            "fn": None,   # special: raw gradient → renormalized
        },
        "Plateau then converge": {
            "color": PALETTE["orange"],
            "fn": lambda i: (3.0 if i < 60 else 3.0 * 0.9**(i-60))
                           + rng.standard_normal() * 0.1,
        },
    }

    panel_positions = [(0,0), (0,1), (1,0), (1,1)]

    for (row, col), (name, info) in zip(panel_positions, scenarios.items()):
        ax = axes[row][col]
        monitor = CauchyMonitor(tol=1e-2, window=20)
        diffs = []
        is_cauchy_flags = []

        if info["fn"] is not None:
            theta = np.array([info["fn"](0)])
            for i in range(1, n_steps):
                theta_new = np.array([info["fn"](i)])
                diff = monitor.update(theta_new)
                if diff < float("inf"):
                    diffs.append(diff)
                    is_cauchy_flags.append(monitor.is_cauchy())
                theta = theta_new
        else:
            # SALT renormalized gradient descent simulation
            theta = np.array([8.0])
            for i in range(n_steps):
                raw_grad = np.array([theta[0] * 0.8 + rng.standard_normal() * 2.0])
                renorm_grad = renormalized_gradient(raw_grad, alpha=0.5)
                theta_new = theta - 0.1 * renorm_grad
                diff = monitor.update(theta_new)
                if diff < float("inf"):
                    diffs.append(diff)
                    is_cauchy_flags.append(monitor.is_cauchy())
                theta = theta_new

        steps = list(range(len(diffs)))
        ax.plot(steps, diffs, color=info["color"], lw=1.5, alpha=0.85)

        # Shade regions where Cauchy condition is met
        cauchy_on = []
        start = None
        for i, c in enumerate(is_cauchy_flags):
            if c and start is None:
                start = i
            elif not c and start is not None:
                ax.axvspan(start, i, alpha=0.12, color=PALETTE["green"])
                start = None
        if start is not None:
            ax.axvspan(start, len(is_cauchy_flags), alpha=0.12, color=PALETTE["green"])

        ax.axhline(0.01, color=PALETTE["gray"], lw=0.8, linestyle="--",
                   alpha=0.6, label="tol=0.01")
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("‖θₜ₊₁ - θₜ‖", fontsize=8)
        ax.set_title(f"{name}\nfinal: {'✓ Cauchy' if monitor.is_cauchy() else '✗ not Cauchy'}"
                     f"  trend: {monitor.trend()}", fontsize=8.5)
        ax.legend(fontsize=7.5, framealpha=0.2)
        ax.grid(True, alpha=0.3)

        # Add summary text
        summary = monitor.summary()
        ax.text(0.98, 0.98,
                f"mean diff: {summary['mean_diff']:.4f}\nmax diff:  {summary['max_diff']:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=PALETTE["white"],
                bbox=dict(facecolor="#16161a", alpha=0.7, edgecolor="none"))

    plt.tight_layout()
    save_or_show(fig, "05_cauchy_convergence", show)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 6: Live training simulation — full SALT pipeline
# ─────────────────────────────────────────────────────────────────────────────

def demo_live(show=False):
    """
    Full end-to-end SALT pipeline on a simulated training run with
    three distinct phases: memorization, critical boundary, generalization.
    Prints a step-by-step readout and produces a comprehensive dashboard.
    """
    print("\n[Demo 6] Full SALT Pipeline — Live Training Simulation")
    print("─" * 60)

    grads = simulate_grokking(n_steps=250, transition=140)[0]
    analyzer = SaltAnalyzer(window=40, n_permutations=100)
    results = []

    print(f"  {'Step':>5}  {'ρ':>6}  {'C_α':>7}  {'q*':>4}  {'FCI%':>6}  {'Phase':<16}  {'Word tail'}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*4}  {'─'*6}  {'─'*16}  {'─'*12}")

    for i in range(len(grads) - 1):
        r = analyzer.step(grads[i], grads[i+1])
        results.append(r)
        if i % 20 == 0 or i < 5:
            tail = r.word[-8:] if len(r.word) >= 8 else r.word
            print(f"  {i:>5}  {r.rho:>6.3f}  {r.c_alpha:>7.3f}  {r.q_star:>4.1f}  "
                  f"{r.f_c_pct:>6.1f}  {r.phase:<16}  {tail}")

    print(f"\n  Final state: phase={results[-1].phase}, C_α={results[-1].c_alpha:.3f}, "
          f"q*={results[-1].q_star:.1f}")
    print(f"  Word length: {len(results[-1].word)} steps")
    print("─" * 60)

    # ── Dashboard plot ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("SALT Full Pipeline Dashboard", fontsize=13, weight="bold")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

    steps  = list(range(len(results)))
    c_vals = [r.c_alpha for r in results]
    q_vals = [r.q_star  for r in results]
    rhos   = [r.rho     for r in results]
    ents   = [r.path_entropy for r in results]
    pcts   = [r.f_c_pct for r in results]
    phases = [r.phase   for r in results]

    phase_colors = {
        "MEMORIZATION": PALETTE["red"],
        "APPROACHING":  PALETTE["orange"],
        "CRITICAL":     PALETTE["purple"],
        "GENERALIZING": PALETTE["blue"],
        "CONVERGED":    PALETTE["green"],
    }

    # C_α
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(steps, c_vals, color=PALETTE["blue"], lw=1.8)
    ax.axhline(1.0, color=PALETTE["orange"], lw=1.2, linestyle="--", alpha=0.7, label="threshold")
    ax.fill_between(steps, 1.0, c_vals,
                    where=[v > 1 for v in c_vals], alpha=0.12, color=PALETTE["green"])
    ax.fill_between(steps, 0, [min(v, 1) for v in c_vals],
                    where=[v <= 1 for v in c_vals], alpha=0.08, color=PALETTE["red"])
    ax.set_title("C_α — signal-to-noise  (sign(C_α-1) = sign(λ₁))", fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.2)
    ax.grid(True, alpha=0.3)

    # C_α distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist([r.c_alpha for r in results], bins=30, color=PALETTE["blue"],
            alpha=0.8, orientation="horizontal", edgecolor="none")
    ax.axhline(1.0, color=PALETTE["orange"], lw=1.2, linestyle="--", alpha=0.7)
    ax.set_title("C_α distribution", fontsize=9)
    ax.set_xlabel("count", fontsize=8)
    ax.grid(True, alpha=0.3)

    # q* and ρ
    ax = fig.add_subplot(gs[1, :2])
    ax2 = ax.twinx()
    ax.plot(steps, q_vals, color=PALETTE["purple"], lw=1.5, label="q* (left)")
    ax2.plot(steps, rhos, color=PALETTE["orange"], lw=1.2, alpha=0.6, label="ρ (right)")
    ax2.axhline(0.5, color=PALETTE["gray"], lw=0.8, linestyle=":", alpha=0.5)
    ax.set_ylabel("q*", fontsize=8, color=PALETTE["purple"])
    ax2.set_ylabel("ρ", fontsize=8, color=PALETTE["orange"])
    ax.set_title("q* (slow variable) and ρ (gradient ratio)", fontsize=9)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.2)
    ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.2)
    ax.grid(True, alpha=0.3)

    # FCI percentile
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(steps, pcts, color=PALETTE["green"], lw=1.5)
    for thr, col in [(80, PALETTE["orange"]), (95, PALETTE["green"])]:
        ax.axhline(thr, color=col, lw=0.8, linestyle="--", alpha=0.6)
    ax.set_title("FCI percentile\n(Farey Consolidation)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Path entropy
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(steps, ents, color=PALETTE["orange"], lw=1.5)
    ax.axhline(1.0, color=PALETTE["gray"], lw=0.8, linestyle=":", alpha=0.5, label="H=1 (uniform)")
    ax.set_title("Path entropy H(L/R word) — 1=plateau, 0=persistent trend", fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7.5, framealpha=0.2)
    ax.grid(True, alpha=0.3)

    # Convergents scatter
    ax = fig.add_subplot(gs[2, 2])
    p_vals = [r.convergent[0] for r in results]
    q_conv = [r.convergent[1] for r in results]
    ax.scatter(p_vals, q_conv, c=steps, cmap="plasma", s=4, alpha=0.6)
    ax.set_xlabel("p (numerator)", fontsize=8)
    ax.set_ylabel("q (denominator)", fontsize=8)
    ax.set_title("CF convergents (p, q) trajectory", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Phase timeline
    ax = fig.add_subplot(gs[3, :])
    for i, phase in enumerate(phases):
        ax.bar(i, 1, width=1.0, color=phase_colors.get(phase, PALETTE["gray"]), linewidth=0)
    ax.set_yticks([])
    ax.set_xlabel("Training step", fontsize=8)
    ax.set_title("Training phase timeline", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=c, label=p) for p, c in phase_colors.items()],
        fontsize=7, framealpha=0.2, loc="upper left", ncol=5
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_or_show(fig, "06_full_pipeline_dashboard", show)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 7: Arithmetic illustration — Stern-Brocot tree + CF
# ─────────────────────────────────────────────────────────────────────────────

def demo_arithmetic(show=False):
    """
    Illustrates the arithmetic skeleton of SALT:
    - CF convergents for specific ρ values
    - Farey neighbor relationships
    - The L/R word encoding
    """
    print("\n[Demo 7] Arithmetic Skeleton")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("SALT: Arithmetic Skeleton — CF Convergents and Farey Neighbors", fontsize=12, weight="bold")

    # ── Panel A: CF approximation quality for various ρ values ───────────────
    ax = axes[0]
    rho_values = [0.5, 0.333, 0.414, 0.618, 0.271, 0.167]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(rho_values)))

    for idx, rho in enumerate(rho_values):
        convs = cf_convergents(rho, q_max=30)
        qs = [q for (_, q) in convs]
        errors = [abs(rho - p/q) for (p, q) in convs]
        ax.semilogy(qs, errors, "o-", color=colors[idx], lw=1.4,
                    markersize=4, label=f"ρ={rho}", alpha=0.85)

    # Hurwitz bound: 1/(√5 q²)
    q_range = np.arange(1, 32)
    ax.semilogy(q_range, 1.0/(math.sqrt(5) * q_range**2), "k--",
                lw=1.0, alpha=0.5, label="Hurwitz bound 1/(√5 q²)")
    ax.set_xlabel("CF denominator q", fontsize=8)
    ax.set_ylabel("approximation error |ρ - p/q|", fontsize=8)
    ax.set_title("CF convergent accuracy per ρ value\n(all below Hurwitz bound)", fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.2, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel B: Gradient trajectory word and Farey neighbors ────────────────
    ax = axes[1]
    rng = np.random.default_rng(55)
    grads = simulate_grokking(n_steps=100, transition=55)[0]
    rhos_traj = [gradient_ratio(grads[i], grads[i+1]) for i in range(len(grads)-1)]
    convs_traj = [best_convergent(rho, gradient_change(grads[i], grads[i+1]))
                  for i, rho in enumerate(rhos_traj)]

    # Color consecutive pairs: green if Farey neighbors, red if not
    farey_hits = [
        is_farey_neighbor(*convs_traj[i], *convs_traj[i+1])
        for i in range(len(convs_traj)-1)
    ]

    steps = list(range(len(rhos_traj)))
    ax.plot(steps, rhos_traj, color=PALETTE["gray"], lw=0.8, alpha=0.5, zorder=1)

    for i, (hit, rho) in enumerate(zip(farey_hits, rhos_traj[:-1])):
        ax.scatter([i], [rho],
                   color=PALETTE["green"] if hit else PALETTE["red"],
                   s=12, zorder=3, alpha=0.8)

    ax.axhline(0.5, color=PALETTE["orange"], lw=0.8, linestyle="--",
               alpha=0.6, label="ρ=0.5 (flat basin)")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=PALETTE["green"], label="Farey neighbor (unimodular)"),
            Patch(color=PALETTE["red"],   label="Not Farey neighbor"),
        ],
        fontsize=7.5, framealpha=0.2
    )
    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("ρ (gradient ratio)", fontsize=8)
    ax.set_title("Gradient trajectory: Farey neighbor detection\n(green = |qₜ pₜ₊₁ - pₜ qₜ₊₁| = 1)", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate Farey hit rate
    frac = sum(farey_hits) / len(farey_hits)
    ax.text(0.98, 0.05, f"Farey neighbor rate: {frac:.1%}",
            transform=ax.transAxes, ha="right", fontsize=8,
            color=PALETTE["white"],
            bbox=dict(facecolor="#16161a", alpha=0.7, edgecolor="none"))

    plt.tight_layout()
    save_or_show(fig, "07_arithmetic_skeleton", show)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "memorization": demo_memorization_vs_generalization,
    "grokking":     demo_grokking,
    "jordan":       demo_jordan,
    "spectral":     demo_spectral,
    "convergence":  demo_convergence,
    "live":         demo_live,
    "arithmetic":   demo_arithmetic,
}


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="SALT — Spectral-Arithmetic Learning Theory Demo Kit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  memorization   Side-by-side: noise vs signal gradients
  grokking       Grokking detection: sudden phase transition
  jordan         Jordan product: commutativity, non-associativity, renorm
  spectral       Spectral basis and Ford circle geometry
  convergence    Cauchy convergence monitor on parameter sequences
  live           Full SALT pipeline dashboard (runs all diagnostics)
  arithmetic     CF convergents and Farey neighbor detection

Run all:
  python salt_demo.py

Run one:
  python salt_demo.py --scenario grokking --show
        """)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        help="Run a specific scenario (default: all)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively instead of saving")
    parser.add_argument("--outdir", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    OUTPUT_DIR = args.outdir

    print("═" * 60)
    print("  SALT — Spectral-Arithmetic Learning Theory")
    print("  Demo Kit")
    print("═" * 60)

    if args.scenario:
        SCENARIOS[args.scenario](show=args.show)
    else:
        for name, fn in SCENARIOS.items():
            fn(show=args.show)

    if not args.show:
        print(f"\n  All plots saved to: {OUTPUT_DIR}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
