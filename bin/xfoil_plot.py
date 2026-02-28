#!/usr/bin/env python3
"""
XFOIL Plot Wrapper — headless XFOIL with matplotlib rendering.

Usage:
    python3 xfoil_plot.py --airfoil "NACA 2412" --alpha 8 --re 1e6
    python3 xfoil_plot.py --airfoil "NACA 0012" --alpha-range -5 15 0.5 --re 5e5
    python3 xfoil_plot.py --airfoil-file my_airfoil.dat --alpha 5 --re 2e6
    python3 xfoil_plot.py --airfoil "NACA 4415" --alpha 6 --re 1e6 --mach 0.3

Generates:
    - Cp distribution + airfoil shape (single alpha)
    - Polar plots: CL vs alpha, CL vs CD, CD vs alpha, CM vs alpha (alpha sweep)
    - Both if you specify --alpha AND --alpha-range
"""

import subprocess
import tempfile
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─── XFOIL Interface ───────────────────────────────────────────────────────────

XFOIL_PATH = None  # Auto-detected or set manually

def find_xfoil():
    """Find xfoil binary."""
    global XFOIL_PATH
    if XFOIL_PATH:
        return XFOIL_PATH
    
    candidates = [
        './xfoil',
        os.path.expanduser('~/Downloads/xfoil6.97/Xfoil/bin/xfoil'),
        '/usr/local/bin/xfoil',
        '/usr/bin/xfoil',
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            XFOIL_PATH = os.path.abspath(path)
            return XFOIL_PATH
    
    # Try which
    try:
        result = subprocess.run(['which', 'xfoil'], capture_output=True, text=True)
        if result.returncode == 0:
            XFOIL_PATH = result.stdout.strip()
            return XFOIL_PATH
    except:
        pass
    
    print("ERROR: Cannot find xfoil binary.")
    print("Either run this script from the xfoil bin directory,")
    print("or set XFOIL_PATH at the top of this script.")
    sys.exit(1)


def run_xfoil(commands, workdir=None):
    """Run XFOIL with a list of commands, return stdout."""
    xfoil = find_xfoil()
    input_str = '\n'.join(commands) + '\n'
    
    env = os.environ.copy()
    env['DISPLAY'] = ''  # Force no X11
    
    result = subprocess.run(
        [xfoil],
        input=input_str,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=workdir or os.path.dirname(xfoil),
        env=env,
    )
    return result.stdout, result.stderr


def run_cp_analysis(airfoil, alpha, re, mach=0.0, airfoil_file=None, ncrit=9.0):
    """Run single-alpha analysis, return Cp data and aero coefficients."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_file = os.path.join(tmpdir, 'cp.dat')
        
        cmds = ['PLOP', 'G F', '']  # Disable graphics
        
        if airfoil_file:
            cmds.append(f'LOAD {airfoil_file}')
        elif airfoil.upper().startswith('NACA'):
            cmds.append(f'NACA {airfoil.split()[-1]}')
        else:
            print(f"ERROR: Don't know how to load airfoil '{airfoil}'")
            sys.exit(1)
        
        cmds.append('OPER')
        cmds.append(f'VISC {re:.0f}')
        if mach > 0:
            cmds.append(f'MACH {mach}')
        if ncrit != 9.0:
            cmds.append(f'VPAR')
            cmds.append(f'N {ncrit}')
            cmds.append('')
        cmds.append(f'ALFA {alpha}')
        cmds.append(f'CPWR {cp_file}')
        cmds.append('')
        cmds.append('QUIT')
        
        stdout, stderr = run_xfoil(cmds, workdir=tmpdir)
        
        # Parse aero coefficients from stdout
        aero = parse_aero_coefficients(stdout, alpha)
        
        # Parse Cp file
        x, cp = [], []
        if os.path.exists(cp_file):
            with open(cp_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x.append(float(parts[0]))
                            cp.append(float(parts[1]))
                        except ValueError:
                            continue
        
        if not x:
            print("WARNING: No Cp data generated. XFOIL may have failed to converge.")
            print("XFOIL output:")
            print(stdout[-1000:] if len(stdout) > 1000 else stdout)
            return None, aero
        
        return (np.array(x), np.array(cp)), aero


def run_polar_sweep(airfoil, alpha_start, alpha_end, alpha_step, re, mach=0.0, 
                    airfoil_file=None, ncrit=9.0):
    """Run alpha sweep, return polar data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        polar_file = os.path.join(tmpdir, 'polar.dat')
        
        cmds = ['PLOP', 'G F', '']
        
        if airfoil_file:
            cmds.append(f'LOAD {airfoil_file}')
        elif airfoil.upper().startswith('NACA'):
            cmds.append(f'NACA {airfoil.split()[-1]}')
        
        cmds.append('OPER')
        cmds.append(f'VISC {re:.0f}')
        if mach > 0:
            cmds.append(f'MACH {mach}')
        if ncrit != 9.0:
            cmds.append('VPAR')
            cmds.append(f'N {ncrit}')
            cmds.append('')
        cmds.append('PACC')
        cmds.append(polar_file)
        cmds.append('')  # No dump file
        cmds.append(f'ASEQ {alpha_start} {alpha_end} {alpha_step}')
        cmds.append('')
        cmds.append('QUIT')
        
        stdout, stderr = run_xfoil(cmds, workdir=tmpdir)
        
        # Parse polar file
        alpha, cl, cd, cdp, cm, top_xtr, bot_xtr = [], [], [], [], [], [], []
        if os.path.exists(polar_file):
            with open(polar_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            alpha.append(float(parts[0]))
                            cl.append(float(parts[1]))
                            cd.append(float(parts[2]))
                            cdp.append(float(parts[3]))
                            cm.append(float(parts[4]))
                            top_xtr.append(float(parts[5]))
                            bot_xtr.append(float(parts[6]))
                        except ValueError:
                            continue
        
        if not alpha:
            print("WARNING: No polar data generated.")
            print("XFOIL output:")
            print(stdout[-1000:] if len(stdout) > 1000 else stdout)
            return None
        
        # Safety: trim all arrays to same length
        min_len = min(len(alpha), len(cl), len(cd), len(cdp), len(cm), len(top_xtr), len(bot_xtr))
        
        return {
            'alpha': np.array(alpha[:min_len]),
            'cl': np.array(cl[:min_len]),
            'cd': np.array(cd[:min_len]),
            'cdp': np.array(cdp[:min_len]),
            'cm': np.array(cm[:min_len]),
            'top_xtr': np.array(top_xtr[:min_len]),
            'bot_xtr': np.array(bot_xtr[:min_len]),
        }


def parse_aero_coefficients(stdout, target_alpha):
    """Extract converged aero coefficients from XFOIL stdout."""
    aero = {'cl': None, 'cd': None, 'cdf': None, 'cdp': None, 'cm': None, 'converged': False}
    
    lines = stdout.split('\n')
    for i, line in enumerate(lines):
        # Look for the last converged iteration
        alpha_str1 = 'a = {:.3f}'.format(target_alpha)
        alpha_str2 = 'a ={:.3f}'.format(target_alpha)
        if alpha_str1 in line or alpha_str2 in line:
            parts = line.split()
            for j, p in enumerate(parts):
                if p == 'CL' and j+2 < len(parts):
                    try: aero['cl'] = float(parts[j+2])
                    except: pass
            # Next line has Cm, CD
            if i+1 < len(lines):
                next_line = lines[i+1]
                parts = next_line.split()
                for j, p in enumerate(parts):
                    if p == 'Cm' and j+2 < len(parts):
                        try: aero['cm'] = float(parts[j+2])
                        except: pass
                    if p == 'CD' and j+2 < len(parts):
                        try: aero['cd'] = float(parts[j+2])
                        except: pass
                    if p == 'CDf' and j+2 < len(parts):
                        try: aero['cdf'] = float(parts[j+2])
                        except: pass
                    if p == 'CDp' and j+2 < len(parts):
                        try: aero['cdp'] = float(parts[j+2])
                        except: pass
    
    # Simpler parse — just find last occurrence
    for line in reversed(lines):
        if 'CL =' in line and aero['cl'] is None:
            try:
                idx = line.index('CL =')
                aero['cl'] = float(line[idx+4:].split()[0])
            except: pass
        if 'CD =' in line and aero['cd'] is None:
            try:
                idx = line.index('CD =')
                aero['cd'] = float(line[idx+4:].split()[0])
            except: pass
        if 'Cm =' in line and aero['cm'] is None:
            try:
                idx = line.index('Cm =')
                aero['cm'] = float(line[idx+4:].split()[0])
            except: pass
    
    if aero['cl'] is not None:
        aero['converged'] = True
    
    return aero


# ─── NACA Airfoil Geometry ─────────────────────────────────────────────────────

def naca_4digit(code, num_points=200):
    """Generate NACA 4-digit airfoil coordinates."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:4]) / 100.0
    
    beta = np.linspace(0, np.pi, num_points)
    xc = 0.5 * (1 - np.cos(beta))
    
    yt = 5*t*(0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
    
    if p > 0:
        yc = np.where(xc <= p, m/p**2 * (2*p*xc - xc**2), m/(1-p)**2 * ((1-2*p) + 2*p*xc - xc**2))
        dyc = np.where(xc <= p, 2*m/p**2 * (p - xc), 2*m/(1-p)**2 * (p - xc))
    else:
        yc = np.zeros_like(xc)
        dyc = np.zeros_like(xc)
    
    theta = np.arctan(dyc)
    xu = xc - yt*np.sin(theta); yu = yc + yt*np.cos(theta)
    xl = xc + yt*np.sin(theta); yl = yc - yt*np.cos(theta)
    return xu, yu, xl, yl


# ─── Plotting ──────────────────────────────────────────────────────────────────

def plot_cp(x, cp, airfoil, alpha, re, mach, aero, output_file='cp_plot.png', dark=True):
    """Plot Cp distribution with airfoil shape, PltLib style."""
    
    # Split upper and lower surfaces
    le_idx = np.argmin(x)
    x_upper, cp_upper = x[:le_idx+1], cp[:le_idx+1]
    x_lower, cp_lower = x[le_idx:], cp[le_idx:]
    
    # Airfoil shape
    code = airfoil.split()[-1]
    if len(code) == 4 and code.isdigit():
        xu, yu, xl, yl = naca_4digit(code)
    else:
        # Fallback: use Cp x-coords as proxy
        xu, yu = x_upper, np.zeros_like(x_upper)
        xl, yl = x_lower, np.zeros_like(x_lower)
    
    if dark:
        plt.style.use('dark_background')
        bg_color = '#0a0a0a'
        fg_color = 'white'
        upper_color = '#00ffff'
        lower_color = '#ffff00'
        airfoil_color = '#00ffff'
        box_bg = '#1a1a1a'
        box_edge = '#444444'
    else:
        plt.style.use('default')
        bg_color = 'white'
        fg_color = 'black'
        upper_color = 'blue'
        lower_color = 'red'
        airfoil_color = 'black'
        box_bg = 'wheat'
        box_edge = 'gray'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), 
                                     height_ratios=[3, 1], 
                                     gridspec_kw={'hspace': 0.08})
    
    # Cp plot
    ax1.plot(x_upper, cp_upper, color=upper_color, linewidth=1.2, label='Upper surface')
    ax1.plot(x_lower, cp_lower, color=lower_color, linewidth=1.2, label='Lower surface')
    ax1.set_ylabel('$C_p$', fontsize=14)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.15, color='gray')
    ax1.legend(fontsize=10, loc='lower left')
    ax1.set_xlim(-0.01, 1.01)
    ax1.axhline(y=0, color=fg_color, lw=0.3, ls='--')
    ax1.tick_params(labelbottom=False)
    
    # Title
    title = f'{airfoil} — Pressure Distribution'
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    # Aero data
    ld = aero['cl'] / aero['cd'] if aero['cd'] and aero['cd'] > 0 else 0
    lines = [
        f"XFOIL  v6.97",
        f"",
        f"{airfoil}",
        f"",
        f"Re  = {re:.3e}",
        f"α   = {alpha:.1f}°",
    ]
    if mach > 0:
        lines.append(f"M   = {mach:.3f}")
    if aero['cl'] is not None:
        lines.append(f"CL  = {aero['cl']:.4f}")
    if aero['cm'] is not None:
        lines.append(f"CM  = {aero['cm']:.4f}")
    if aero['cd'] is not None:
        lines.append(f"CD  = {aero['cd']:.5f}")
    if ld != 0:
        lines.append(f"L/D = {ld:.1f}")
    
    textstr = '\n'.join(lines)
    ax1.text(0.82, 0.97, textstr, transform=ax1.transAxes, fontsize=9.5,
             verticalalignment='top', fontfamily='monospace', color=fg_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=box_bg, 
                      edgecolor=box_edge, alpha=0.9))
    
    # Airfoil shape
    ax2.plot(xu, yu, color=airfoil_color, linewidth=1.5)
    ax2.plot(xl, yl, color=airfoil_color, linewidth=1.5)
    ax2.plot([xu[-1], xl[-1]], [yu[-1], yl[-1]], color=airfoil_color, linewidth=1.5)
    ax2.plot([xu[0], xl[0]], [yu[0], yl[0]], color=airfoil_color, linewidth=1.5)
    
    if len(xu) == len(xl):
        ax2.fill_between(xu, yu, np.interp(xu, xl, yl), alpha=0.08, color=upper_color)
    
    ax2.set_xlabel('x/c', fontsize=14)
    ax2.set_ylabel('y/c', fontsize=14)
    ax2.set_xlim(-0.01, 1.01)
    ax2.set_ylim(-0.15, 0.15)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.15, color='gray')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"  Cp plot saved: {output_file}")


def plot_polars(polar, airfoil, re, mach, output_file='polar_plot.png', dark=True):
    """Plot polar curves."""
    
    if dark:
        plt.style.use('dark_background')
        bg_color = '#0a0a0a'
        colors = ['#00ffff', '#ff6666', '#66ff66', '#ff66ff']
    else:
        plt.style.use('default')
        bg_color = 'white'
        colors = ['blue', 'red', 'green', 'magenta']
    
    alpha = polar['alpha']
    cl = polar['cl']
    cd = polar['cd']
    cm = polar['cm']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    re_str = f"{re:.2e}".replace("+0", "").replace("+", "")
    title = f'{airfoil} — Re = {re_str}'
    if mach > 0:
        title += f', M = {mach}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # CL vs Alpha
    ax = axes[0, 0]
    ax.plot(alpha, cl, f'-o', color=colors[0], markersize=3, linewidth=1.5)
    ax.set_xlabel('α (°)'); ax.set_ylabel('$C_L$'); ax.set_title('Lift Curve')
    ax.grid(True, alpha=0.2); ax.axhline(y=0, color='gray', lw=0.5); ax.axvline(x=0, color='gray', lw=0.5)
    
    # Find and annotate CL_max
    cl_max_idx = np.argmax(cl)
    ax.annotate(f'$C_{{L,max}}$ = {cl[cl_max_idx]:.3f}\nα = {alpha[cl_max_idx]:.1f}°',
                xy=(alpha[cl_max_idx], cl[cl_max_idx]),
                xytext=(alpha[cl_max_idx]-5, cl[cl_max_idx]-0.2),
                arrowprops=dict(arrowstyle='->', color=colors[0]),
                fontsize=9, color=colors[0])
    
    # Drag Polar
    ax = axes[0, 1]
    ax.plot(cd, cl, f'-o', color=colors[1], markersize=3, linewidth=1.5)
    ax.set_xlabel('$C_D$'); ax.set_ylabel('$C_L$'); ax.set_title('Drag Polar')
    ax.grid(True, alpha=0.2)
    
    # L/D max line
    ld = cl / cd
    ld_max_idx = np.argmax(ld)
    ax.plot([0, cd[ld_max_idx]*2], [0, cl[ld_max_idx]*2], '--', color='gray', lw=0.8, alpha=0.5)
    ax.annotate(f'L/D$_{{max}}$ = {ld[ld_max_idx]:.1f}\nα = {alpha[ld_max_idx]:.1f}°',
                xy=(cd[ld_max_idx], cl[ld_max_idx]),
                xytext=(cd[ld_max_idx]+0.005, cl[ld_max_idx]-0.3),
                arrowprops=dict(arrowstyle='->', color=colors[1]),
                fontsize=9, color=colors[1])
    
    # CD vs Alpha
    ax = axes[1, 0]
    ax.plot(alpha, cd, f'-o', color=colors[2], markersize=3, linewidth=1.5, label='$C_D$ (total)')
    if 'cdp' in polar:
        ax.plot(alpha, polar['cdp'], f'--', color=colors[2], markersize=2, linewidth=1, alpha=0.6, label='$C_{Dp}$ (pressure)')
    ax.set_xlabel('α (°)'); ax.set_ylabel('$C_D$'); ax.set_title('Drag Curve')
    ax.grid(True, alpha=0.2); ax.legend(fontsize=9)
    
    # CM vs Alpha
    ax = axes[1, 1]
    ax.plot(alpha, cm, f'-o', color=colors[3], markersize=3, linewidth=1.5)
    ax.set_xlabel('α (°)'); ax.set_ylabel('$C_M$'); ax.set_title('Pitching Moment')
    ax.grid(True, alpha=0.2); ax.axhline(y=0, color='gray', lw=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"  Polar plot saved: {output_file}")


def plot_transition(polar, airfoil, re, output_file='transition_plot.png', dark=True):
    """Plot boundary layer transition locations."""
    
    if dark:
        plt.style.use('dark_background')
        bg_color = '#0a0a0a'
    else:
        plt.style.use('default')
        bg_color = 'white'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(polar['alpha'], polar['top_xtr'], 'c-o', markersize=3, linewidth=1.5, label='Upper surface')
    ax.plot(polar['alpha'], polar['bot_xtr'], 'y-o', markersize=3, linewidth=1.5, label='Lower surface')
    ax.set_xlabel('α (°)', fontsize=12)
    ax.set_ylabel('$x_{tr}/c$', fontsize=12)
    ax.set_title(f'{airfoil} — Boundary Layer Transition Location (Re = {re:.2e})', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"  Transition plot saved: {output_file}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='XFOIL Plot Wrapper — headless XFOIL with matplotlib rendering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --airfoil "NACA 2412" --alpha 8 --re 1e6
  %(prog)s --airfoil "NACA 0012" --alpha-range -5 15 0.5 --re 5e5
  %(prog)s --airfoil "NACA 4415" --alpha 6 --alpha-range -5 15 1 --re 1e6
  %(prog)s --airfoil-file selig1223.dat --alpha 5 --re 2e6
  %(prog)s --airfoil "NACA 2412" --alpha 8 --re 1e6 --light
        """)
    
    airfoil_group = parser.add_mutually_exclusive_group(required=True)
    airfoil_group.add_argument('--airfoil', type=str, help='NACA designation (e.g. "NACA 2412")')
    airfoil_group.add_argument('--airfoil-file', type=str, help='Path to airfoil coordinate file')
    
    parser.add_argument('--alpha', type=float, help='Single angle of attack for Cp plot')
    parser.add_argument('--alpha-range', nargs=3, type=float, metavar=('START', 'END', 'STEP'),
                       help='Alpha sweep range for polar plots')
    parser.add_argument('--re', type=float, required=True, help='Reynolds number')
    parser.add_argument('--mach', type=float, default=0.0, help='Mach number (default: 0)')
    parser.add_argument('--ncrit', type=float, default=9.0, help='Ncrit for transition (default: 9)')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for plots')
    parser.add_argument('--light', action='store_true', help='Use light theme instead of dark')
    parser.add_argument('--xfoil-path', type=str, help='Path to xfoil binary')
    
    args = parser.parse_args()
    
    if not args.alpha and not args.alpha_range:
        parser.error("Specify --alpha and/or --alpha-range")
    
    if args.xfoil_path:
        global XFOIL_PATH
        XFOIL_PATH = args.xfoil_path
    
    airfoil = args.airfoil or os.path.basename(args.airfoil_file)
    dark = not args.light
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Sanitize name for filenames
    safe_name = airfoil.replace(' ', '_').replace('/', '_').lower()
    
    print(f"\n{'='*60}")
    print(f"  XFOIL Plot Wrapper")
    print(f"  Airfoil: {airfoil}")
    print(f"  Re = {args.re:.3e}, M = {args.mach}")
    print(f"{'='*60}\n")
    
    # Single alpha — Cp plot
    if args.alpha is not None:
        print(f"Running Cp analysis at α = {args.alpha}°...")
        cp_data, aero = run_cp_analysis(
            airfoil, args.alpha, args.re, args.mach, 
            args.airfoil_file, args.ncrit
        )
        
        if aero.get('converged'):
            ld = aero['cl']/aero['cd'] if aero['cd'] else 0
            print(f"  Converged: CL = {aero['cl']:.4f}, CD = {aero['cd']:.5f}, "
                  f"CM = {aero['cm']:.4f}, L/D = {ld:.1f}")
        
        if cp_data is not None:
            cp_file = os.path.join(args.output_dir, f'{safe_name}_cp_a{args.alpha:.1f}.png')
            plot_cp(cp_data[0], cp_data[1], airfoil, args.alpha, args.re, args.mach, aero, cp_file, dark)
        else:
            print("  FAILED: No Cp data. Check if XFOIL converged.")
    
    # Alpha sweep — polar plots
    if args.alpha_range:
        a_start, a_end, a_step = args.alpha_range
        print(f"\nRunning polar sweep α = {a_start}° to {a_end}° (Δα = {a_step}°)...")
        
        polar = run_polar_sweep(
            airfoil, a_start, a_end, a_step, args.re, args.mach,
            args.airfoil_file, args.ncrit
        )
        
        if polar is not None:
            n = len(polar['alpha'])
            cl_max = polar['cl'].max()
            ld_max = (polar['cl'] / polar['cd']).max()
            print(f"  {n} converged points")
            print(f"  CL_max = {cl_max:.4f}, L/D_max = {ld_max:.1f}")
            
            polar_file = os.path.join(args.output_dir, f'{safe_name}_polar.png')
            plot_polars(polar, airfoil, args.re, args.mach, polar_file, dark)
            
            trans_file = os.path.join(args.output_dir, f'{safe_name}_transition.png')
            plot_transition(polar, airfoil, args.re, trans_file, dark)
        else:
            print("  FAILED: No polar data.")
    
    print(f"\nDone! Check {args.output_dir}/ for plots.\n")


if __name__ == '__main__':
    main()
