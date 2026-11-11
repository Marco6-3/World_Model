"""Publication-style (CoRL / RSS / ICRA / RA-L) figure for Stage C.2 bag packing.

Reads exactly the same predeclared paired case and traces as ``bag_report.figure``
(no outcome-based selection) and writes to ``outputs/stage_c_bag/paper/`` so the
audited ``bag_packing_demo.*`` files are left untouched.

Usage:  python paper_fig_stage_c.py
"""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from bag_common import COLLAR_Z, SIZE, THRESHOLD

OUT = Path('outputs/stage_c_bag')
DST = OUT / 'paper'

# ---------------------------------------------------------------- style
TEXT_W = 7.16  # IEEE double-column width (inch); RA-L/ICRA/IROS. CoRL ~ 6.75
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType (IEEE PDF eXpress)
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})

# method colours: baselines muted, ours saturated accent
METHOD_STYLE = {
    'BC H4':             dict(label='BC (H=4)',           color='#8C9BAB'),
    'Joint-FM':          dict(label='Joint-FM',           color='#A9B4C2'),
    'GPC-RANK':          dict(label='GPC-Rank',           color='#7C9CC9'),
    'GPC-OPT':           dict(label='GPC-Opt (M=20)',     color='#3E6FB0'),
    'Ours Short-MLP H4': dict(label='Ours w/ Short-MLP',  color='#F2B49B'),
    'Ours 4D-WM':        dict(label='Ours w/ 4D-WM',      color='#EE9A7A'),
    'Ours tau=0.7':      dict(label='Ours (τmax=0.7)',       color='#E8775A'),
    'Ours tau=0.9':      dict(label='Ours (τmax=0.9)',       color='#D1432B'),
}
# scene entity colours (independent of method colours)
ENT = dict(left='#7A7A7A', right='#2B2B2B', A='#E9A23B', B='#2A9D8F',
           bag='#B8C2CC', slot='#9AA3AC')
OK, BAD = '#2E8B57', '#C0392B'


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def quat_rot(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def cube(ax, centre, half, R, color):
    v = (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half) @ R.T + centre
    faces = [[v[j] for j in ids] for ids in ((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor='#333333', lw=.3, alpha=.85))


def style3d(ax, lo, hi):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((.975, .978, .982, 1.))
        a.pane.set_edgecolor('#D5D9DE'); a.pane.set_linewidth(.5)
        a._axinfo['grid'].update(color='#E4E7EB', linewidth=.35)
        a._axinfo['tick'].update(inward_factor=0., outward_factor=.15)
        a.line.set_linewidth(.5); a.line.set_color('#9AA0A6')
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set_xticks([-.1, 0, .1]); ax.set_yticks([-.2, 0, .2]); ax.set_zticks([0, .1, .2])
    ax.tick_params(labelsize=5, pad=-3)
    # axis titles omitted to reduce clutter; state "all axes in metres" in the caption
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.view_init(elev=24, azim=-58); ax.set_box_aspect(hi - lo, zoom=1.12)


def draw_bag(ax):
    x0, x1, y0, y1, z0 = -.098, .098, -.087, .087, .014
    walls = [[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
             [(x0,y0,z0),(x1,y0,z0),(x1,y0,COLLAR_Z),(x0,y0,COLLAR_Z)],
             [(x0,y1,z0),(x1,y1,z0),(x1,y1,COLLAR_Z),(x0,y1,COLLAR_Z)],
             [(x0,y0,z0),(x0,y1,z0),(x0,y1,COLLAR_Z),(x0,y0,COLLAR_Z)],
             [(x1,y0,z0),(x1,y1,z0),(x1,y1,COLLAR_Z),(x1,y0,COLLAR_Z)]]
    ax.add_collection3d(Poly3DCollection(walls, facecolor=ENT['bag'], edgecolor='none', alpha=.12))
    rim_x = [x0, x1, x1, x0, x0]; rim_y = [y0, y0, y1, y1, y0]
    ax.plot(rim_x, rim_y, np.full(5, COLLAR_Z), color='#6B7884', lw=1.0)   # elastic collar
    ax.plot(rim_x, rim_y, np.full(5, z0), color='#AAB3BC', lw=.5)
    for xx, yy in zip(rim_x[:4], rim_y[:4]):
        ax.plot([xx, xx], [yy, yy], [z0, COLLAR_Z], color='#C3CAD1', lw=.4)
    for tx in (-.048, .048):  # target slots
        s = .022; ax.plot([tx-s, tx+s, tx+s, tx-s, tx-s], [-s, -s, s, s, -s], np.full(5, z0),
                          color=ENT['slot'], lw=.6, ls=(0, (2, 1.5)))


def badge(ax, y, text, good):
    ax.text2D(.5, y, text, transform=ax.transAxes, ha='center', va='center', fontsize=5.6,
              family='DejaVu Sans', color=OK if good else BAD,
              bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.6', fc='white',
                        ec=OK if good else BAD, lw=.5))


def main(out=OUT):
    global OUT,DST
    OUT=Path(out);DST=OUT/'paper'
    data = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    if data.get('protocol_version')!='stage_c2':raise ValueError('Audited C.2 results required')
    case = json.loads((OUT / 'protocol.json').read_text())['figure_case']
    dt = data['control_dt']
    rows, traces = [], []
    for m in case['methods']:
        r = next(e for e in data['episodes'] if e['method'] == m and e['mode'] == case['mode']
                 and e['order'] == case['order'] and e['seed'] == case['seed'] and e['perturbed'] == case['perturbed'])
        rows.append(r)
        with np.load(OUT / r['trace_file']) as raw:
            traces.append({k: raw[k] for k in ('frames', 'learn', 'poses')})

    pts = np.concatenate([t['frames'][t['learn'], 2:14].reshape(-1, 3) for t in traces])
    lo = np.minimum(np.array([-.11, -.23, 0.]), pts.min(0) - .02)
    hi = np.maximum(np.array([.11, .26, .28]), pts.max(0) + .02)

    fig = plt.figure(figsize=(TEXT_W, 4.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=.22, top=.92,bottom=.17)
    top = outer[0].subgridspec(1, 3, wspace=-.05)
    bot = outer[1].subgridspec(1, 3, wspace=.62, width_ratios=[1, 1, 1.0])

    # ------------------------------------------------------ (a) 3D rollouts
    timeline = []
    for col, (r, t) in enumerate(zip(rows, traces)):
        f = t['frames']; ax = fig.add_subplot(top[col], projection='3d')
        style3d(ax, lo, hi); draw_bag(ax)
        vis = np.where(t['learn'][:, None], f[:, 2:14], np.nan)
        ax.plot(*vis[:, 0:3].T, color=ENT['left'], lw=.8, ls=(0, (3, 1.5)))
        ax.plot(*vis[:, 3:6].T, color=ENT['right'], lw=.8)
        for i, key in enumerate('AB'):
            p = vis[:, 6+3*i:9+3*i]
            ax.plot(*p.T, color=ENT[key], lw=1.6, solid_capstyle='round')
            w, qx, qy, qz = t['poses'][-1, i, 3:7]
            cube(ax, f[-1, 8+3*i:11+3*i], SIZE[i]/2, quat_rot(w, qx, qy, qz), ENT[key])
        st = METHOD_STYLE[r['method']]
        ax.text2D(.5, 1.0, st['label'], transform=ax.transAxes, ha='center', va='bottom',
                  color=st['color'] if 'Ours' in r['method'] else '#333333',
                  fontweight='bold', fontsize=7.5)
        badge(ax, .92, ('✓ packed' if r['success'] else '✗ not packed') + '   ' +
              ('✓ collision-free' if not r['collision'] else '✗ collision'),
              r['success'] and not r['collision'])
        if col == 0:
            ax.text2D(.02, 1.0, 'a', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
        slips = [e['t'] for e in r['events'] if e['event'] == 'collar_slip']
        origin = slips[0] if slips else r['learn_ranges'][0][0]
        timeline.append((r, f, f[:, 0] - origin * dt, origin))

    # scene legend (entities)
    handles = [Line2D([], [], color=ENT['left'], lw=.8, ls=(0, (3, 1.5)), label='Left gripper'),
               Line2D([], [], color=ENT['right'], lw=.8, label='Right gripper'),
               Line2D([], [], color=ENT['A'], lw=1.6, label='Item A path'),
               Line2D([], [], color=ENT['B'], lw=1.6, label='Item B path'),
               Line2D([], [], color='#6B7884', lw=1.0, label='Elastic collar'),
               Patch(fc='none', ec=ENT['slot'], ls='--', lw=.6, label='Target slot')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=6,
               frameon=False, handlelength=1.8, columnspacing=1.4)

    # ------------------------------------------------------ (b) aperture, (c) clearance
    axb, axc, axd = (fig.add_subplot(bot[i]) for i in range(3))
    for ax in (axb, axc):
        ax.axvspan(-1, 0, color='#F3F4F6', lw=0, zorder=0)
        ax.axvline(0, color='#555555', lw=.6, ls=(0, (2, 1.5)), zorder=1)
        ax.set_xlim(-1, 8); ax.set_xticks(range(0, 9, 2)); ax.set_xlabel('Time from collar slip (s)')
        ax.grid(axis='y', color='#ECEEF1', lw=.5); ax.set_axisbelow(True)
    axb.text(.05, 160, 'slip', fontsize=5.5, color='#555555', ha='left', va='center')
    axc.axhspan(-30, 0, color=BAD, alpha=.07, lw=0, zorder=0)
    axc.text(7.9, -28.5, 'penetration', color=BAD, fontsize=5.5, ha='right', va='bottom')
    axc.axhline(28, color='#777777', lw=.6, ls=(0, (1, 1.2)))
    axc.text(7.9, 30, '28 mm margin', color='#666666', fontsize=5.5, ha='right', va='bottom')
    axb.axhline(70, color='#777777', lw=.6, ls=(0, (1, 1.2)))

    order = sorted(range(len(timeline)), key=lambda i: 'Ours' in timeline[i][0]['method'])  # ours on top
    for k, i in enumerate(order):
        r, f, tt, origin = timeline[i]; st = METHOD_STYLE[r['method']]
        w = (tt >= -1) & (tt <= 8); ours = 'Ours' in r['method']; z = 5 if ours else 3
        lw = 1.4 if ours else 1.0
        axb.plot(tt[w], f[w, 1]*1000, color=st['color'], lw=lw, zorder=z, label=st['label'])
        occ = f[:, 15] > .5
        axc.plot(tt[w], np.where(occ, f[:, 14]*1000, np.nan)[w], color=st['color'], lw=lw, zorder=z)
        hit = (f[:, 17] >= THRESHOLD) & w
        axc.scatter(tt[hit], np.full(hit.sum(), -27 + 2.2*i), marker='|', s=14, lw=.8,
                    color=st['color'], zorder=z)
        meas = r.get('recovery_measurement', {})
        for key, mk in (('withdraw_tick', 'v'), ('reopen_tick', '^')):
            tick = meas.get(key)
            if tick is None: continue
            tv = (tick - origin) * dt; j = int(np.argmin(np.abs(tt - tv)))
            if -1 <= tv <= 8:
                axb.scatter(tv, f[j, 1]*1000, marker=mk, s=26, color=st['color'], edgecolor='#222222',
                            lw=.5, zorder=8)
    axb.set(ylim=(-8, 172), ylabel='Collar aperture (mm)')
    axc.set(ylim=(-30, 66), ylabel='Throat clearance (mm)')
    mh = [Line2D([], [], color=METHOD_STYLE[m]['color'], lw=1.4 if 'Ours' in m else 1.0,
                 label=METHOD_STYLE[m]['label']) for m in case['methods']]
    mk_h = [Line2D([], [], marker='v', ls='', mfc='white', mec='#222222', ms=3.5, label='withdraw'),
            Line2D([], [], marker='^', ls='', mfc='white', mec='#222222', ms=3.5, label='reopen')]
    axb.legend(handles=mk_h, loc='lower right', frameon=False, handlelength=1.0, handletextpad=.3,
               borderaxespad=.1, labelspacing=.25, bbox_to_anchor=(1.0, .22), fontsize=5.5)
    axc.legend(handles=mh, loc='lower right', frameon=False, handlelength=1.4,
               borderaxespad=.1, labelspacing=.25, bbox_to_anchor=(1.0, .33), fontsize=5.5)
    axc.text(-.5, -24.5, 'contact', fontsize=4.8, color='#777777', ha='center', va='center')

    # ------------------------------------------------------ (d) full-matrix safe recovery
    names = [m for m in METHOD_STYLE if any(e['method'] == m for e in data['episodes'])]
    stats = [];timeouts={}
    for m in names:
        pop = [e for e in data['episodes'] if e['method'] == m and e['perturbed']]
        k = sum(bool(e['safe_recovery']) for e in pop); stats.append((m, k, len(pop)))
        timeouts[m]=sum(e['safe_recovery'] and any(e['insertion_budget_exhausted']) for e in pop)
    y = np.arange(len(stats))[::-1]
    for yi, (m, k, n) in zip(y, stats):
        st = METHOD_STYLE[m]; p = k / n; l, h = wilson(k, n)
        axd.barh(yi, p*100, height=.68, color=st['color'], lw=0)
        axd.errorbar(p*100, yi, xerr=[[100*(p-l)], [100*(h-p)]], fmt='none', ecolor='#333333',
                     elinewidth=.6, capsize=1.5, capthick=.6)
        axd.text(min(h*100, 100) + 2, yi, f'{k}/{n}'+('*' if timeouts[m] else ''), va='center', fontsize=5.5,
                 fontweight='bold' if m == 'Ours tau=0.9' else 'normal')
    axd.set_yticks(y, [METHOD_STYLE[m]['label'] for m, _, _ in stats])
    for lab, (m, _, _) in zip(axd.get_yticklabels(), stats):
        if m == 'Ours tau=0.9': lab.set_fontweight('bold'); lab.set_color(METHOD_STYLE[m]['color'])
    axd.set_xlim(0, 118); axd.set_xticks([0, 25, 50, 75, 100])
    axd.set_xlabel('Safe recovery under slip (%)')
    axd.tick_params(axis='y', length=0)
    axd.grid(axis='x', color='#ECEEF1', lw=.5); axd.set_axisbelow(True)

    for ax, tag in ((axb, 'b'), (axc, 'c'), (axd, 'd')):
        ax.text(-.30 if tag != 'd' else -.80, 1.06, tag, transform=ax.transAxes, fontsize=9,
                fontweight='bold', va='bottom')

    DST.mkdir(parents=True, exist_ok=True)
    fig.text(.5,.015,'Stage C.2 | GPC-Opt did not trigger a slip in panels a-c; its time origin is insertion start.\n'
             'Wilson intervals are descriptive (3 seed clusters). * Includes insertion timeout; strict counts in report.',
             ha='center',va='bottom',fontsize=5.5,color='#555555')
    for ext in ('pdf', 'png', 'svg'):
        exported=DST/f'fig_stage_c_bag.{ext}'
        fig.savefig(exported)
        if ext=='svg':
            exported.write_text('\n'.join(line.rstrip() for line in exported.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    (DST / 'fig_stage_c_bag_manifest.json').write_text(json.dumps(dict(
        protocol_version=data['protocol_version'],
        selection='predeclared paired case from protocol.json, identical to bag_report.figure',
        case=case, traces={r['method']: dict(file=r['trace_file'],
                           sha256=sha256((OUT / r['trace_file']).read_bytes()).hexdigest()) for r in rows},
        panel_d=[dict(method=m, safe_recovery=k, n=n,safe_recoveries_with_timeout=timeouts[m]) for m, k, n in stats],
        ci='Descriptive episode-wise Wilson 95% intervals; seed clustering ignored. See paired cluster bootstrap in report.'), indent=2), encoding='utf-8')
    print('written to', DST.resolve())


if __name__ == '__main__':
    main()
