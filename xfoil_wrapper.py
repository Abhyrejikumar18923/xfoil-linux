#!/usr/bin/env python3
"""
XFOIL Wrapper — headless XFOIL with matplotlib rendering.

Runs XFOIL as a subprocess (no X11 needed), parses its text output, and
generates publication-quality plots via matplotlib.

Usage examples:
    python3 xfoil_wrapper.py --naca 2412 --alpha -5:15:0.5 --re 1e6
    python3 xfoil_wrapper.py --naca 0012 --alpha 5 --re 5e5
    python3 xfoil_wrapper.py --airfoil airfoil.dat --alpha -2:12:0.25 --re 3e6
    python3 xfoil_wrapper.py --naca 4415 --alpha -5:15:0.5 --re 1e6 --format svg --csv

Outputs:
    - Polar plots: CL vs alpha, CL vs CD, CD vs alpha, CM vs alpha
    - Cp distribution + airfoil shape (at representative alphas)
    - Airfoil geometry plot
    - Transition location plot
    - CSV polar data (optional)
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# XFOIL binary discovery
# ---------------------------------------------------------------------------

def find_xfoil(user_path=None):
    """Return an absolute path to a working xfoil binary."""
    candidates = []
    if user_path:
        candidates.append(user_path)
    env_path = os.environ.get("XFOIL_BIN")
    if env_path:
        candidates.append(env_path)
    # Relative to this script
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, "bin", "xfoil"),
        os.path.join(here, "xfoil"),
    ]
    # System paths
    candidates += ["/usr/local/bin/xfoil", "/usr/bin/xfoil"]
    for p in candidates:
        p = os.path.expanduser(p)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.abspath(p)
    # Last resort: rely on PATH
    try:
        r = subprocess.run(["which", "xfoil"], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    return None


# ---------------------------------------------------------------------------
# XFOIL runner
# ---------------------------------------------------------------------------

def run_xfoil(xfoil_bin, commands, workdir=None, timeout=300):
    """Feed *commands* to XFOIL via stdin; return (stdout, stderr)."""
    cmd_str = "\n".join(commands) + "\n"
    env = os.environ.copy()
    env["DISPLAY"] = ""  # prevent any X11 attempt

    result = subprocess.run(
        [xfoil_bin],
        input=cmd_str,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=workdir or os.getcwd(),
        env=env,
    )
    return result.stdout, result.stderr


# ---------------------------------------------------------------------------
# XFOIL command builders
# ---------------------------------------------------------------------------

def _load_cmds(naca=None, airfoil_file=None):
    """Commands to load an airfoil."""
    if naca:
        return [f"NACA {naca}"]
    if airfoil_file:
        return [f"LOAD {os.path.abspath(airfoil_file)}"]
    raise ValueError("Specify --naca or --airfoil")


def _visc_cmds(re, mach=0.0, ncrit=9.0, max_iter=100):
    """Commands to set up viscous mode."""
    cmds = [f"VISC {re:.0f}"]
    if mach > 0:
        cmds.append(f"MACH {mach}")
    cmds.append(f"ITER {max_iter}")
    if ncrit != 9.0:
        cmds += ["VPAR", f"N {ncrit}", ""]
    return cmds


def build_polar_commands(naca, airfoil_file, re, mach, ncrit, max_iter,
                         alpha_start, alpha_end, alpha_step, polar_file):
    """Build the full command list for a polar sweep."""
    cmds = ["PLOP", "G F", ""]
    cmds += _load_cmds(naca, airfoil_file)
    cmds.append("OPER")
    cmds += _visc_cmds(re, mach, ncrit, max_iter)
    cmds += ["PACC", polar_file, ""]  # polar save, no dump

    # Split into segments if range is large for better convergence
    total = abs(alpha_end - alpha_start)
    if total > 15 and abs(alpha_step) < 1:
        mid = (alpha_start + alpha_end) / 2
        cmds.append(f"ASEQ {alpha_start} {mid} {alpha_step}")
        cmds.append(f"ASEQ {mid + alpha_step} {alpha_end} {alpha_step}")
    else:
        cmds.append(f"ASEQ {alpha_start} {alpha_end} {alpha_step}")

    cmds += ["PACC", "", "QUIT"]
    return cmds


def build_cp_commands(naca, airfoil_file, re, mach, ncrit, max_iter,
                      alpha, cp_file, bl_file, coord_file):
    """Build commands for single-alpha Cp + BL dump + coordinate save."""
    cmds = ["PLOP", "G F", ""]
    cmds += _load_cmds(naca, airfoil_file)
    cmds += [f"SAVE {coord_file}"]
    cmds.append("OPER")
    cmds += _visc_cmds(re, mach, ncrit, max_iter)
    cmds += [
        f"ALFA {alpha}",
        f"CPWR {cp_file}",
        f"DUMP {bl_file}",
        "",
        "QUIT",
    ]
    return cmds


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_polar_file(path):
    """Parse an XFOIL polar save file → dict of numpy arrays."""
    cols = {"alpha": [], "cl": [], "cd": [], "cdp": [], "cm": [],
            "top_xtr": [], "bot_xtr": []}
    if not os.path.isfile(path):
        return None

    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                vals = [float(p) for p in parts[:7]]
            except ValueError:
                continue
            for key, val in zip(cols, vals):
                cols[key].append(val)

    if not cols["alpha"]:
        return None
    return {k: np.array(v) for k, v in cols.items()}


def parse_cp_file(path):
    """Parse CPWR output → (x_array, cp_array)."""
    xs, cps = [], []
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0]))
                    cps.append(float(parts[1]))
                except ValueError:
                    continue
    if not xs:
        return None
    return np.array(xs), np.array(cps)


def parse_bl_file(path):
    """Parse DUMP output → dict of numpy arrays."""
    names = ["s", "x", "y", "Ue_Vinf", "Dstar", "Theta", "Cf", "H"]
    cols = {n: [] for n in names}
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                vals = [float(p) for p in parts[:8]]
            except ValueError:
                continue
            for name, val in zip(names, vals):
                cols[name].append(val)
    if not cols["s"]:
        return None
    return {k: np.array(v) for k, v in cols.items()}


def parse_coordinate_file(path):
    """Parse XFOIL SAVE coordinate file → (x, y) arrays."""
    xs, ys = [], []
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue  # skip name header
            parts = line.split()
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    continue
    if not xs:
        return None
    return np.array(xs), np.array(ys)


def parse_aero_from_stdout(stdout):
    """Extract last converged CL/CD/CM from XFOIL screen output."""
    aero = {"alpha": None, "cl": None, "cd": None, "cm": None}
    for line in reversed(stdout.splitlines()):
        if "CL =" in line and aero["cl"] is None:
            m = re.search(r"CL\s*=\s*([-+]?\d*\.?\d+)", line)
            if m:
                aero["cl"] = float(m.group(1))
        if "CD =" in line and aero["cd"] is None:
            m = re.search(r"CD\s*=\s*([-+]?\d*\.?\d+)", line)
            if m:
                aero["cd"] = float(m.group(1))
        if "Cm =" in line and aero["cm"] is None:
            m = re.search(r"Cm\s*=\s*([-+]?\d*\.?\d+)", line)
            if m:
                aero["cm"] = float(m.group(1))
        if "a =" in line and aero["alpha"] is None:
            m = re.search(r"a\s*=\s*([-+]?\d*\.?\d+)", line)
            if m:
                aero["alpha"] = float(m.group(1))
    return aero


# ---------------------------------------------------------------------------
# NACA 4-digit geometry (for standalone geometry plot)
# ---------------------------------------------------------------------------

def naca_4digit(code, n=200):
    """Generate NACA 4-digit airfoil coordinates (upper + lower)."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:4]) / 100.0

    beta = np.linspace(0, np.pi, n)
    xc = 0.5 * (1 - np.cos(beta))  # cosine spacing

    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc
                  - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    if p > 0:
        yc = np.where(xc <= p,
                      m / p**2 * (2 * p * xc - xc**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xc - xc**2))
        dyc = np.where(xc <= p,
                       2 * m / p**2 * (p - xc),
                       2 * m / (1 - p)**2 * (p - xc))
    else:
        yc = np.zeros_like(xc)
        dyc = np.zeros_like(xc)

    theta = np.arctan(dyc)
    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_STYLE_DARK = {
    "bg": "#0c0c0c", "fg": "white",
    "upper": "#00ccff", "lower": "#ffcc00", "airfoil": "#00ccff",
    "colors": ["#00ccff", "#ff5555", "#55ff55", "#ff55ff"],
    "grid_alpha": 0.15, "box_bg": "#1a1a1a", "box_edge": "#444",
}
_STYLE_LIGHT = {
    "bg": "white", "fg": "black",
    "upper": "blue", "lower": "red", "airfoil": "black",
    "colors": ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"],
    "grid_alpha": 0.25, "box_bg": "wheat", "box_edge": "gray",
}


def _apply_style(dark):
    sty = _STYLE_DARK if dark else _STYLE_LIGHT
    plt.style.use("dark_background" if dark else "default")
    return sty


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_cp(x, cp, airfoil_name, alpha, re, mach, aero, coord_xy,
            output, dark=True):
    """Cp distribution + airfoil shape."""
    sty = _apply_style(dark)

    # Split upper / lower at the leading-edge (minimum x)
    le = np.argmin(x)
    x_up, cp_up = x[: le + 1], cp[: le + 1]
    x_lo, cp_lo = x[le:], cp[le:]

    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(3, 1, height_ratios=[3, 0.05, 1], hspace=0.08, figure=fig)
    ax_cp = fig.add_subplot(gs[0])
    ax_af = fig.add_subplot(gs[2])

    # -- Cp --
    ax_cp.plot(x_up, cp_up, color=sty["upper"], lw=1.3, label="Upper")
    ax_cp.plot(x_lo, cp_lo, color=sty["lower"], lw=1.3, label="Lower")
    ax_cp.invert_yaxis()
    ax_cp.set_ylabel("$C_p$", fontsize=13)
    ax_cp.set_xlim(-0.02, 1.02)
    ax_cp.axhline(0, color=sty["fg"], lw=0.3, ls="--")
    ax_cp.grid(True, alpha=sty["grid_alpha"])
    ax_cp.legend(fontsize=10, loc="lower left")
    ax_cp.tick_params(labelbottom=False)
    ax_cp.set_title(f"{airfoil_name} — $C_p$ at α = {alpha}°", fontsize=13,
                    fontweight="bold", pad=10)

    # Info box
    lines = [f"XFOIL 6.97", "", f"Re = {re:.3e}"]
    if mach > 0:
        lines.append(f"M  = {mach:.3f}")
    lines.append(f"α  = {alpha}°")
    if aero["cl"] is not None:
        lines.append(f"CL = {aero['cl']:.4f}")
    if aero["cd"] is not None:
        lines.append(f"CD = {aero['cd']:.5f}")
    if aero["cm"] is not None:
        lines.append(f"CM = {aero['cm']:.4f}")
    if aero["cl"] and aero["cd"] and aero["cd"] > 0:
        lines.append(f"L/D = {aero['cl'] / aero['cd']:.1f}")
    ax_cp.text(0.82, 0.97, "\n".join(lines), transform=ax_cp.transAxes,
               fontsize=9.5, va="top", fontfamily="monospace", color=sty["fg"],
               bbox=dict(boxstyle="round,pad=0.5", facecolor=sty["box_bg"],
                         edgecolor=sty["box_edge"], alpha=0.9))

    # -- Airfoil shape --
    if coord_xy is not None:
        cx, cy = coord_xy
    else:
        # Fallback: use Cp x-coordinates
        cx = np.concatenate([x_up[::-1], x_lo])
        cy = np.zeros_like(cx)
    ax_af.plot(cx, cy, color=sty["airfoil"], lw=1.5)
    ax_af.fill(cx, cy, alpha=0.06, color=sty["upper"])
    ax_af.set_xlabel("x/c", fontsize=13)
    ax_af.set_ylabel("y/c", fontsize=13)
    ax_af.set_xlim(-0.02, 1.02)
    ax_af.set_aspect("equal")
    ax_af.grid(True, alpha=sty["grid_alpha"])

    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=sty["bg"])
    plt.close(fig)
    print(f"    Cp plot → {output}")


def plot_polars(polar, airfoil_name, re, mach, output, dark=True):
    """2×2 polar plots: CL-α, CL-CD, CD-α, CM-α."""
    sty = _apply_style(dark)
    a, cl, cd, cm = polar["alpha"], polar["cl"], polar["cd"], polar["cm"]
    c = sty["colors"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    re_s = f"{re:.2e}".replace("+0", "").replace("+", "")
    title = f"{airfoil_name} — Re = {re_s}"
    if mach > 0:
        title += f", M = {mach}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # CL vs alpha
    ax = axes[0, 0]
    ax.plot(a, cl, "-o", color=c[0], ms=3, lw=1.5)
    ax.set_xlabel("α (°)")
    ax.set_ylabel("$C_L$")
    ax.set_title("Lift Curve")
    ax.grid(True, alpha=sty["grid_alpha"])
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    cl_max_i = np.argmax(cl)
    ax.annotate(f"$C_{{L,max}}$ = {cl[cl_max_i]:.3f}\nα = {a[cl_max_i]:.1f}°",
                xy=(a[cl_max_i], cl[cl_max_i]),
                xytext=(a[cl_max_i] - 5, cl[cl_max_i] - 0.25),
                arrowprops=dict(arrowstyle="->", color=c[0]),
                fontsize=9, color=c[0])

    # Drag polar
    ax = axes[0, 1]
    ax.plot(cd, cl, "-o", color=c[1], ms=3, lw=1.5)
    ax.set_xlabel("$C_D$")
    ax.set_ylabel("$C_L$")
    ax.set_title("Drag Polar")
    ax.grid(True, alpha=sty["grid_alpha"])
    ld = cl / np.where(cd > 0, cd, np.inf)
    ld_i = np.argmax(ld)
    ax.plot([0, cd[ld_i] * 2], [0, cl[ld_i] * 2], "--", color="gray",
            lw=0.8, alpha=0.5)
    ax.annotate(f"L/D$_{{max}}$ = {ld[ld_i]:.1f}\nα = {a[ld_i]:.1f}°",
                xy=(cd[ld_i], cl[ld_i]),
                xytext=(cd[ld_i] + 0.005, cl[ld_i] - 0.3),
                arrowprops=dict(arrowstyle="->", color=c[1]),
                fontsize=9, color=c[1])

    # CD vs alpha
    ax = axes[1, 0]
    ax.plot(a, cd, "-o", color=c[2], ms=3, lw=1.5, label="$C_D$ total")
    if "cdp" in polar:
        ax.plot(a, polar["cdp"], "--", color=c[2], lw=1, alpha=0.6,
                label="$C_{Dp}$ pressure")
    ax.set_xlabel("α (°)")
    ax.set_ylabel("$C_D$")
    ax.set_title("Drag Curve")
    ax.grid(True, alpha=sty["grid_alpha"])
    ax.legend(fontsize=9)

    # CM vs alpha
    ax = axes[1, 1]
    ax.plot(a, cm, "-o", color=c[3], ms=3, lw=1.5)
    ax.set_xlabel("α (°)")
    ax.set_ylabel("$C_M$")
    ax.set_title("Pitching Moment")
    ax.grid(True, alpha=sty["grid_alpha"])
    ax.axhline(0, color="gray", lw=0.4)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=sty["bg"])
    plt.close(fig)
    print(f"    Polar plots → {output}")


def plot_transition(polar, airfoil_name, re, output, dark=True):
    """Transition location vs alpha."""
    sty = _apply_style(dark)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(polar["alpha"], polar["top_xtr"], "-o", color=sty["upper"],
            ms=3, lw=1.5, label="Upper surface")
    ax.plot(polar["alpha"], polar["bot_xtr"], "-o", color=sty["lower"],
            ms=3, lw=1.5, label="Lower surface")
    ax.set_xlabel("α (°)", fontsize=12)
    ax.set_ylabel("$x_{tr}/c$", fontsize=12)
    ax.set_title(f"{airfoil_name} — Transition Location (Re = {re:.2e})",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=sty["grid_alpha"])
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=sty["bg"])
    plt.close(fig)
    print(f"    Transition plot → {output}")


def plot_geometry(coord_xy, airfoil_name, output, dark=True):
    """Standalone airfoil geometry plot."""
    sty = _apply_style(dark)
    cx, cy = coord_xy

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cx, cy, color=sty["airfoil"], lw=1.8)
    ax.fill(cx, cy, alpha=0.08, color=sty["upper"])
    ax.set_xlabel("x/c", fontsize=13)
    ax.set_ylabel("y/c", fontsize=13)
    ax.set_title(f"{airfoil_name} — Airfoil Geometry", fontsize=13,
                 fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=sty["grid_alpha"])

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=sty["bg"])
    plt.close(fig)
    print(f"    Geometry plot → {output}")


def plot_bl(bl, airfoil_name, alpha, output, dark=True):
    """Boundary-layer parameters: Cf and H vs x/c."""
    sty = _apply_style(dark)
    c = sty["colors"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{airfoil_name} — BL Parameters at α = {alpha}°",
                 fontsize=13, fontweight="bold")

    ax1.plot(bl["x"], bl["Cf"], "-", color=c[0], lw=1.2)
    ax1.set_ylabel("$C_f$", fontsize=12)
    ax1.axhline(0, color="gray", lw=0.4, ls="--")
    ax1.grid(True, alpha=sty["grid_alpha"])

    ax2.plot(bl["x"], bl["H"], "-", color=c[1], lw=1.2)
    ax2.set_ylabel("H (shape factor)", fontsize=12)
    ax2.set_xlabel("x/c", fontsize=12)
    ax2.set_ylim(0, min(bl["H"].max() * 1.2, 50))
    ax2.grid(True, alpha=sty["grid_alpha"])

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=sty["bg"])
    plt.close(fig)
    print(f"    BL plot → {output}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_polar_csv(polar, airfoil_name, re, mach, path):
    """Write polar data to CSV."""
    with open(path, "w", newline="") as fh:
        fh.write(f"# {airfoil_name}  Re={re:.0f}  M={mach}\n")
        w = csv.writer(fh)
        w.writerow(["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"])
        for i in range(len(polar["alpha"])):
            w.writerow([f"{polar['alpha'][i]:.3f}",
                        f"{polar['cl'][i]:.6f}",
                        f"{polar['cd'][i]:.6f}",
                        f"{polar['cdp'][i]:.6f}",
                        f"{polar['cm'][i]:.6f}",
                        f"{polar['top_xtr'][i]:.4f}",
                        f"{polar['bot_xtr'][i]:.4f}"])
    print(f"    Polar CSV → {path}")


# ---------------------------------------------------------------------------
# Alpha range parser
# ---------------------------------------------------------------------------

def parse_alpha(spec):
    """Parse alpha specification.

    Accepts:
        "5"            → single alpha  (5.0, None)
        "-5:15:0.5"    → sweep         (None, (-5, 15, 0.5))
    """
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "Alpha range must be START:END:STEP, e.g. -5:15:0.5")
        return None, tuple(float(p) for p in parts)
    return float(spec), None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fixup_argv(argv):
    """Fix argparse issue with negative-looking alpha values like -5:15:0.5.

    Converts ``--alpha -5:15:0.5`` to ``--alpha=-5:15:0.5`` so argparse
    doesn't treat the value as a flag.
    """
    out = list(argv)
    for i, arg in enumerate(out):
        if arg == "--alpha" and i + 1 < len(out):
            nxt = out[i + 1]
            if nxt.startswith("-") and ":" in nxt:
                out[i] = f"--alpha={nxt}"
                del out[i + 1]
                break
    return out


def main():
    # Pre-process argv to handle --alpha -5:15:0.5
    sys.argv = _fixup_argv(sys.argv)

    parser = argparse.ArgumentParser(
        description="XFOIL Wrapper — headless XFOIL with matplotlib plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --naca 2412 --alpha -5:15:0.5 --re 1e6
  %(prog)s --naca 0012 --alpha 5 --re 5e5
  %(prog)s --airfoil e387.dat --alpha -2:10:0.25 --re 1e5
  %(prog)s --naca 4415 --alpha -5:15:0.5 --re 1e6 --format svg --csv
  %(prog)s --naca 2412 --alpha 8 --re 1e6 --cp-alphas 0,4,8
""")

    foil = parser.add_mutually_exclusive_group(required=True)
    foil.add_argument("--naca", type=str, help="NACA 4-digit code (e.g. 2412)")
    foil.add_argument("--airfoil", type=str,
                      help="Path to airfoil coordinate .dat file")

    parser.add_argument("--alpha", type=str, required=True,
                        help="Single value (e.g. 5) or sweep START:END:STEP "
                             "(e.g. -5:15:0.5)")
    parser.add_argument("--re", type=float, required=True,
                        help="Reynolds number (e.g. 1e6)")
    parser.add_argument("--mach", type=float, default=0.0,
                        help="Mach number (default: 0)")
    parser.add_argument("--ncrit", type=float, default=9.0,
                        help="Ncrit for e^n transition (default: 9)")
    parser.add_argument("--iter", type=int, default=100,
                        help="Max XFOIL iterations per point (default: 100)")
    parser.add_argument("--cp-alphas", type=str, default=None,
                        help="Comma-separated alphas for Cp plots "
                             "(default: auto-select 3 representative)")
    parser.add_argument("--xfoil", type=str, default=None,
                        help="Path to xfoil binary "
                             "(also settable via XFOIL_BIN env var)")
    parser.add_argument("--output-dir", "-o", type=str, default=".",
                        help="Output directory (default: .)")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "svg", "both"],
                        help="Image format (default: png)")
    parser.add_argument("--csv", action="store_true",
                        help="Also export polar data as CSV")
    parser.add_argument("--light", action="store_true",
                        help="Light colour theme (default: dark)")

    args = parser.parse_args()

    # ── Locate XFOIL ──
    xfoil_bin = find_xfoil(args.xfoil)
    if xfoil_bin is None:
        print("ERROR: Could not find xfoil binary.")
        print("  Build it first (cd Xfoil && bash build.sh), then either:")
        print("  • run from the Xfoil/ directory (auto-detected at bin/xfoil)")
        print("  • pass --xfoil /path/to/xfoil")
        print("  • set XFOIL_BIN=/path/to/xfoil in your environment")
        sys.exit(1)
    print(f"Using XFOIL: {xfoil_bin}")

    # ── Parse alpha ──
    single_alpha, sweep = parse_alpha(args.alpha)
    dark = not args.light
    os.makedirs(args.output_dir, exist_ok=True)

    airfoil_name = f"NACA {args.naca}" if args.naca else os.path.basename(
        args.airfoil)
    safe = airfoil_name.replace(" ", "_").lower()

    exts = []
    if args.format in ("png", "both"):
        exts.append("png")
    if args.format in ("svg", "both"):
        exts.append("svg")

    def out(name):
        """Return list of output paths for each requested format."""
        return [os.path.join(args.output_dir, f"{safe}_{name}.{e}")
                for e in exts]

    print(f"\n{'=' * 62}")
    print(f"  XFOIL Wrapper")
    print(f"  Airfoil : {airfoil_name}")
    print(f"  Re      : {args.re:.3e}")
    if args.mach > 0:
        print(f"  Mach    : {args.mach}")
    if single_alpha is not None and sweep is None:
        print(f"  Alpha   : {single_alpha}°")
    elif sweep is not None:
        print(f"  Alpha   : {sweep[0]}° → {sweep[1]}° (Δ{sweep[2]}°)")
    print(f"  Ncrit   : {args.ncrit}")
    print(f"{'=' * 62}\n")

    coord_xy = None  # airfoil coordinates (x, y)

    # ──────────────────────────────────────────────────────────────────────
    # 1. POLAR SWEEP
    # ──────────────────────────────────────────────────────────────────────
    polar = None
    if sweep is not None:
        a0, a1, da = sweep
        print(f"[1/3] Running polar sweep α = {a0}° → {a1}° ...")
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = os.path.join(tmpdir, "polar.dat")
            cmds = build_polar_commands(
                args.naca, args.airfoil, args.re, args.mach, args.ncrit,
                args.iter, a0, a1, da, pf)
            stdout, stderr = run_xfoil(xfoil_bin, cmds, workdir=tmpdir)
            polar = parse_polar_file(pf)

        if polar is None:
            print("  WARNING: polar sweep produced no data. XFOIL output:")
            print(stdout[-2000:] if len(stdout) > 2000 else stdout)
        else:
            n = len(polar["alpha"])
            cl_max = polar["cl"].max()
            ld = polar["cl"] / np.where(polar["cd"] > 0, polar["cd"], np.inf)
            ld_max = ld.max()
            print(f"  {n} converged points,  CL_max = {cl_max:.4f}, "
                  f" L/D_max = {ld_max:.1f}")

            for p in out("polar"):
                plot_polars(polar, airfoil_name, args.re, args.mach, p, dark)
            for p in out("transition"):
                plot_transition(polar, airfoil_name, args.re, p, dark)
            if args.csv:
                write_polar_csv(polar, airfoil_name, args.re, args.mach,
                                os.path.join(args.output_dir,
                                             f"{safe}_polar.csv"))
    else:
        print("[1/3] No polar sweep (single alpha only).")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Cp / BL AT REPRESENTATIVE ALPHAS
    # ──────────────────────────────────────────────────────────────────────
    # Decide which alphas to compute Cp for
    cp_alphas = []
    if args.cp_alphas:
        cp_alphas = [float(a.strip()) for a in args.cp_alphas.split(",")]
    elif single_alpha is not None and sweep is None:
        cp_alphas = [single_alpha]
    elif sweep is not None and polar is not None:
        # Auto-pick: low, mid, near CL_max
        a_arr = polar["alpha"]
        picks = set()
        picks.add(a_arr[len(a_arr) // 4])       # low
        picks.add(a_arr[len(a_arr) // 2])        # mid
        picks.add(a_arr[np.argmax(polar["cl"])])  # CL_max
        cp_alphas = sorted(picks)

    if cp_alphas:
        print(f"\n[2/3] Computing Cp at α = "
              f"{', '.join(f'{a}°' for a in cp_alphas)} ...")
        for alpha_i in cp_alphas:
            with tempfile.TemporaryDirectory() as tmpdir:
                cpf = os.path.join(tmpdir, "cp.dat")
                blf = os.path.join(tmpdir, "bl.dat")
                cof = os.path.join(tmpdir, "coords.dat")
                cmds = build_cp_commands(
                    args.naca, args.airfoil, args.re, args.mach,
                    args.ncrit, args.iter, alpha_i, cpf, blf, cof)
                stdout, _ = run_xfoil(xfoil_bin, cmds, workdir=tmpdir)
                aero = parse_aero_from_stdout(stdout)
                cp_data = parse_cp_file(cpf)
                bl_data = parse_bl_file(blf)
                if coord_xy is None:
                    coord_xy = parse_coordinate_file(cof)

            if cp_data is not None:
                a_tag = f"a{alpha_i:+.1f}".replace("+", "")
                for p in out(f"cp_{a_tag}"):
                    plot_cp(cp_data[0], cp_data[1], airfoil_name, alpha_i,
                            args.re, args.mach, aero, coord_xy, p, dark)
            else:
                print(f"    WARNING: No Cp data at α = {alpha_i}°")

            if bl_data is not None:
                a_tag = f"a{alpha_i:+.1f}".replace("+", "")
                for p in out(f"bl_{a_tag}"):
                    plot_bl(bl_data, airfoil_name, alpha_i, p, dark)
    else:
        print("[2/3] No Cp analysis requested.")

    # ──────────────────────────────────────────────────────────────────────
    # 3. GEOMETRY
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[3/3] Airfoil geometry ...")
    if coord_xy is None and args.naca and len(args.naca) == 4:
        xu, yu, xl, yl = naca_4digit(args.naca)
        coord_xy = (np.concatenate([xu, xl[::-1]]),
                    np.concatenate([yu, yl[::-1]]))
    if coord_xy is not None:
        for p in out("geometry"):
            plot_geometry(coord_xy, airfoil_name, p, dark)
    else:
        print("    No geometry data available.")

    # ──────────────────────────────────────────────────────────────────────
    print(f"\nDone.  Outputs in {os.path.abspath(args.output_dir)}/\n")


if __name__ == "__main__":
    main()
