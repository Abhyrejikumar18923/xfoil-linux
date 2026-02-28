import matplotlib.pyplot as plt
import numpy as np

alpha, cl, cd, cdp, cm = [], [], [], [], []
with open('polar.dat', 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-') or not line[0].lstrip('-').replace('.','').isdigit():
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                alpha.append(float(parts[0]))
                cl.append(float(parts[1]))
                cd.append(float(parts[2]))
                cdp.append(float(parts[3]))
                cm.append(float(parts[4]))
            except ValueError:
                continue

alpha, cl, cd, cdp, cm = np.array(alpha), np.array(cl), np.array(cd), np.array(cdp), np.array(cm)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('NACA 2412 — Re = 1e6, M = 0.0', fontsize=14, fontweight='bold')

ax = axes[0, 0]
ax.plot(alpha, cl, 'b-o', markersize=3, linewidth=1.5)
ax.set_xlabel('α (°)'); ax.set_ylabel('CL'); ax.set_title('Lift Curve')
ax.grid(True, alpha=0.3); ax.axhline(y=0, color='k', lw=0.5); ax.axvline(x=0, color='k', lw=0.5)

ax = axes[0, 1]
ax.plot(cd, cl, 'r-o', markersize=3, linewidth=1.5)
ax.set_xlabel('CD'); ax.set_ylabel('CL'); ax.set_title('Drag Polar')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(alpha, cd, 'g-o', markersize=3, linewidth=1.5)
ax.set_xlabel('α (°)'); ax.set_ylabel('CD'); ax.set_title('Drag Curve')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(alpha, cm, 'm-o', markersize=3, linewidth=1.5)
ax.set_xlabel('α (°)'); ax.set_ylabel('CM'); ax.set_title('Pitching Moment')
ax.grid(True, alpha=0.3); ax.axhline(y=0, color='k', lw=0.5)

plt.tight_layout()
plt.savefig('naca2412_polar.png', dpi=150, bbox_inches='tight')
plt.show()
