"""Figure A: the DART proposal family across the MRW-MALA-Newton-independence
spectrum, and the classical proposal each regime approaches.

Top row is Pi_x ~ exp(-V_x) at four (theta, gamma) settings, tempering
increasing and regularisation decreasing left to right. Bottom row is the
matching classical proposal. All measures are drawn as nested 25%, 50% and
75% HPD bands, computed by quadrature over 'localisationModel'.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

import localisationModel as lm
from styne.parameter.vector import Vector
from styne.utility.densityarithmetic import LogScalingWrapper

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "axes.linewidth": 0.7,
})

TARGET_COLOR = "#4d4d4d"
SURROGATE_COLOR = "#D55E00"
PROPOSAL_COLOR = "#0072B2"
SURROGATE_DASH = (0, (4, 2))

# Sits left of and above the target's bulk, far enough out that the two rows
# differ visibly. Lowering it much further makes grad^2 g indefinite, which
# the Newton column cannot invert.
CURRENT_STATE = np.array([-3.0, 4.85])
SETTINGS = [(0.10, 10.0), (0.50, 5.0), (0.90, 0.25), (1.00, 0.025)]
REGIME_LABELS = ["MRW regime", "MALA regime", "Newton regime",
                 "independence regime"]

XLIM = (-5.5, 3.0)
YLIM = (-5.5, 6.0)
QUADRATURE_BOUNDS = (-20.0, 20.0, -20.0, 20.0)
QUADRATURE_RESOLUTION = 420
RENDER_RESOLUTION = 260

# 8.4 in at 180 dpi is 1512 px, comfortably retina-sharp against the ~830 px
# GitHub renders a README at. 220 buys no visible detail for 85 KB more, and
# git keeps every copy of that forever.
README_DPI = 180

HPD_LEVELS = (0.75, 0.50, 0.25)
BAND_ALPHA = 0.13
PANEL_TEXT_SIZE = 7.6


def chi_squared_radius(level):
    """Mahalanobis radius enclosing `level` probability mass, 2 dof."""
    return np.sqrt(-2.0 * np.log(1.0 - level))


def draw_ellipse(ax, mean, covariance, level, **kwargs):
    """Mahalanobis-radius contour of a 2D Gaussian.

    'eigh' returns ascending eigenvalues, so width takes the last eigenpair:
    pairing the smallest eigenvalue with the largest eigenvector's angle
    draws the ellipse rotated by a right angle.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    radius = chi_squared_radius(level)
    width = 2.0 * radius * np.sqrt(eigenvalues[-1])
    height = 2.0 * radius * np.sqrt(eigenvalues[0])
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    ax.add_patch(Ellipse(mean, width, height, angle=angle, **kwargs))


def draw_gaussian_measure(ax, mean, covariance):
    """Nested HPD bands plus outlines, in the proposal colour."""
    for level in HPD_LEVELS:
        draw_ellipse(ax, mean, covariance, level, facecolor=PROPOSAL_COLOR,
                     alpha=BAND_ALPHA, edgecolor="none", zorder=2)
        draw_ellipse(ax, mean, covariance, level, facecolor="none",
                     edgecolor=PROPOSAL_COLOR, linewidth=1.0, zorder=3)


def to_grid(values, xAxis, yAxis):
    return values.reshape(len(xAxis), len(yAxis))


def draw_quadrature_measure(ax, density, quadraturePoints, quadratureCellArea,
                             renderXAxis, renderYAxis, renderPoints,
                             color, fill, dashed=False):
    """Contour a measure at the shared HPD levels, nesting the fills so the
    core reads more opaque than the tails."""
    quadratureLog = np.asarray(density.evaluate_log(Vector(quadraturePoints)))
    levels = [lm.hpd_log_density_level(quadratureLog, quadratureCellArea, level)
              for level in HPD_LEVELS]
    renderLog = np.asarray(density.evaluate_log(Vector(renderPoints)))
    field = to_grid(renderLog, renderXAxis, renderYAxis)
    renderX, renderY = np.meshgrid(renderXAxis, renderYAxis, indexing="ij")

    if fill:
        for level in levels:
            ax.contourf(renderX, renderY, field, levels=[level, field.max()],
                        colors=[color], alpha=BAND_ALPHA, zorder=2)
    # Stated explicitly: matplotlib dashes negative-valued levels by default.
    linestyle = [SURROGATE_DASH] * len(levels) if dashed else ["solid"] * len(levels)
    ax.contour(renderX, renderY, field, levels=levels, colors=color,
               linewidths=1.0, linestyles=linestyle, zorder=3)


def draw_current_state(ax):
    ax.plot(*CURRENT_STATE, marker="x", color="black", markersize=7.0,
            markeredgewidth=1.7, linestyle="none", zorder=6)


def panel_text(ax, text):
    """Caption placed low-left, where no measure reaches."""
    ax.text(0.04, 0.035, text, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=PANEL_TEXT_SIZE, color="#333333", zorder=7)


def style_panel(ax, boxAspect):
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(boxAspect)


def legend_handles():
    return [
        Line2D([], [], color=TARGET_COLOR, lw=1.0,
               label=r"target $\pi \propto e^{-f}$"),
        Line2D([], [], color=SURROGATE_COLOR, lw=1.0, ls=SURROGATE_DASH,
               label=r"surrogate $\propto e^{-g}$"),
        Line2D([], [], color=PROPOSAL_COLOR, lw=1.0,
               label="proposal"),
        Line2D([], [], color="black", lw=0, marker="x", mew=1.7,
               label=r"current state $x$"),
    ]


def build_figure():
    quadraturePoints, quadratureCellArea = lm.build_grid(
        QUADRATURE_BOUNDS, QUADRATURE_RESOLUTION)
    renderXAxis, renderYAxis, renderPoints, _ = lm.grid_mesh(
        (XLIM[0], XLIM[1], YLIM[0], YLIM[1]), RENDER_RESOLUTION)

    gradientAtState = lm.target_gradient(CURRENT_STATE[None, :])[0]

    # Local inverse Hessian at the state, not a fixed one at the mode.
    newtonHessian = lm.surrogate_hessian(CURRENT_STATE[None, :])[0]
    newtonEigenvalues = np.linalg.eigvalsh(newtonHessian)
    if newtonEigenvalues[0] <= 0.0:
        raise ValueError(
            "Hessian of g is not positive definite at the current state; "
            f"eigenvalues {newtonEigenvalues}. Stochastic Newton needs a "
            "positive-definite metric there.")
    newtonMetric = np.linalg.inv(newtonHessian)

    boxAspect = (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
    figure, axes = plt.subplots(2, 4, figsize=(8.4, 8.4 * boxAspect / 2.0 + 0.5))

    for column, (theta, gamma) in enumerate(SETTINGS):
        topAxis = axes[0, column]
        bottomAxis = axes[1, column]

        for ax in (topAxis, bottomAxis):
            style_panel(ax, boxAspect)
            draw_quadrature_measure(
                ax, lm.target_density, quadraturePoints, quadratureCellArea,
                renderXAxis, renderYAxis, renderPoints, TARGET_COLOR,
                fill=True)
            draw_current_state(ax)

        # Untempered: a small theta flattens it until its HPD bands swamp
        # the panel.
        draw_quadrature_measure(
            topAxis, lm.surrogate_density, quadraturePoints, quadratureCellArea,
            renderXAxis, renderYAxis, renderPoints, SURROGATE_COLOR,
            fill=False, dashed=True)

        proposal = lm.localised_proposal(theta, gamma, CURRENT_STATE)
        draw_quadrature_measure(
            topAxis, proposal, quadraturePoints, quadratureCellArea,
            renderXAxis, renderYAxis, renderPoints, PROPOSAL_COLOR, fill=True)

        hSquared = 2.0 * theta / gamma
        if column == 0:
            draw_gaussian_measure(
                bottomAxis, CURRENT_STATE, (1.0 / gamma) * np.eye(2))
            bottomText = "MRW\n" rf"$\sigma^2 \approx \gamma^{{-1}} = {1.0 / gamma:.3g}$"
        elif column == 1:
            mean = CURRENT_STATE - 0.5 * hSquared * gradientAtState
            draw_gaussian_measure(bottomAxis, mean, hSquared * np.eye(2))
            bottomText = ("MALA\n"
                          rf"$h^2 = 2\theta/\gamma = {hSquared:.3g}$")
        elif column == 2:
            # Stein h^2 puts an ellipse several times the proposal's size on
            # the panel, so match its isotropic sd instead and report by how
            # much that overrides the rule.
            _, proposalCovariance, _, _ = lm.quadrature_moments(
                proposal, quadraturePoints, quadratureCellArea)
            proposalSd = lm.isotropic_sd(proposalCovariance)
            metricSd = lm.isotropic_sd(newtonMetric)
            overrideSquared = (proposalSd / metricSd) ** 2
            print(f"Newton step override: h^2={overrideSquared:.4f} vs "
                  f"Stein h^2={hSquared:.4f} (factor "
                  f"{overrideSquared / hSquared:.4f})")
            mean = CURRENT_STATE - 0.5 * overrideSquared * (
                newtonMetric @ gradientAtState)
            draw_gaussian_measure(bottomAxis, mean, overrideSquared * newtonMetric)
            bottomText = ("stochastic Newton\n"
                          rf"$h^2 = {overrideSquared:.3g}$")
        else:
            tempered = LogScalingWrapper(lm.surrogate_density, theta)
            draw_quadrature_measure(
                bottomAxis, tempered, quadraturePoints, quadratureCellArea,
                renderXAxis, renderYAxis, renderPoints, PROPOSAL_COLOR,
                fill=True)
            draw_quadrature_measure(
                bottomAxis, lm.surrogate_density, quadraturePoints,
                quadratureCellArea, renderXAxis, renderYAxis, renderPoints,
                SURROGATE_COLOR, fill=False, dashed=True)
            bottomText = "independence sampler\n" r"$\propto e^{-g}$"

        panel_text(topAxis,
                   f"{REGIME_LABELS[column]}\n"
                   rf"$\theta = {theta:g},\ \gamma = {gamma:g}$")
        panel_text(bottomAxis, bottomText)

    axes[0, 0].set_ylabel("DART proposal", fontsize=9)
    axes[1, 0].set_ylabel("classical proposal", fontsize=9)

    figure.tight_layout(h_pad=0.3, w_pad=0.3, rect=(0, 0.055, 1, 1))
    figure.legend(handles=legend_handles(), loc="lower center", ncol=4,
                 frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    return figure


if __name__ == "__main__":
    fig = build_figure()
    # Absolute paths, not the working directory: running from the repo root
    # would otherwise write there and leave these copies stale.
    here = Path(__file__).resolve()
    # The PNG is the tracked README asset. SVG is not an option here: the
    # filled quadrature bands vectorise to 883 KB against 285 KB rasterised.
    # White is explicit rather than inherited from rcParams, so a local style
    # cannot ship a transparent background that GitHub's dark theme renders
    # the axis labels unreadably against.
    readmeAsset = here.parents[2] / "docs" / "assets" / "dart-proposal-family.png"
    fig.savefig(readmeAsset, dpi=README_DPI, facecolor="white")
    # The PDF stays beside this file for the paper, and stays gitignored.
    fig.savefig(here.with_suffix(".pdf"), dpi=220)
    print(f"written to {readmeAsset}")
