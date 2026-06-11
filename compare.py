"""
Compare response and resolution curves of two (or more) L1DeepMETv2 trainings
on the same plots.

Usage:
    python compare.py <ckpts_1> <ckpts_2> [<ckpts_3> ...] [-o out_dir] [--file best.resolutions]

Each <ckpts_*> is a checkpoints folder (the one passed as --ckpts to train.py),
holding a `best.resolutions` file written by evaluate.py. Saves
response_comparison.png and resolution_comparison.png into out_dir
(default: current directory).
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import load

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except Exception:
    pass


def _xy(hist_tuple):
    """A saved resolution entry is a np.histogram tuple (values, bin_edges).
    Return (bin_centers, values) for plotting."""
    values = np.asarray(hist_tuple[0])
    edges = np.asarray(hist_tuple[1])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, values


def _load(folder, fname):
    path = folder if os.path.isabs(folder) else os.path.join(os.getcwd(), folder)
    f = os.path.join(path, fname)
    if not os.path.exists(f):
        raise FileNotFoundError(f"No {fname} found in {path}")
    return load(f)


def plot_response(folders, out=None, puppi=True, show=False, labels=None, fname='best.resolutions'):
    """Overlay the MET response curve (R = -<u_par>/<q_T>) for each run folder.

    out: if given, save response_comparison.png into this dir.
    show: if True, leave the figure open (for inline display in notebooks).
    labels: optional list of legend labels (defaults to folder names).
    """
    if labels is None:
        labels = [os.path.basename(os.path.normpath(f)) for f in folders]
    fig = plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    puppi_ref = None
    for i, (folder, label) in enumerate(zip(folders, labels)):
        d = _load(folder, fname)
        x, y = _xy(d['MET']['R'])
        plt.plot(x, y, color=colors[i % len(colors)], lw=2, marker='o', ms=3, label=label)
        if puppi_ref is None:
            puppi_ref = d
    if puppi and puppi_ref is not None:
        x, y = _xy(puppi_ref['puppiMET']['R'])
        plt.plot(x, y, color='magenta', lw=2, marker='s', ms=3, label='PUPPI')
    plt.axhline(y=1.0, color='black', linestyle='-.')
    plt.axis([0, 400, 0, 1.2])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Response $-\frac{<u_{\parallel}>}{<q_{T}>}$')
    plt.title('Response comparison')
    plt.legend(fontsize=10)
    if out is not None:
        os.makedirs(out, exist_ok=True)
        resp_out = os.path.join(out, 'response_comparison.png')
        fig.savefig(resp_out, dpi=150, bbox_inches='tight')
        print(f"Saved {resp_out}")
    if not show:
        plt.close(fig)
    return fig


def plot_resolution(folders, out=None, puppi=True, show=False, labels=None, fname='best.resolutions'):
    """Overlay the scaled u_perp / u_par MET resolution curves for each run folder.

    out: if given, save resolution_comparison.png into this dir.
    show: if True, leave the figure open (for inline display in notebooks).
    labels: optional list of legend labels (defaults to folder names).
    """
    if labels is None:
        labels = [os.path.basename(os.path.normpath(f)) for f in folders]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    puppi_ref = None
    for i, (folder, label) in enumerate(zip(folders, labels)):
        d = _load(folder, fname)
        x, y = _xy(d['MET']['u_perp_scaled_resolution'])
        axes[0].plot(x, y, color=colors[i % len(colors)], lw=2, marker='o', ms=3, label=label)
        x, y = _xy(d['MET']['u_par_scaled_resolution'])
        axes[1].plot(x, y, color=colors[i % len(colors)], lw=2, marker='o', ms=3, label=label)
        if puppi_ref is None:
            puppi_ref = d
    if puppi and puppi_ref is not None:
        x, y = _xy(puppi_ref['puppiMET']['u_perp_scaled_resolution'])
        axes[0].plot(x, y, color='magenta', lw=2, marker='s', ms=3, label='PUPPI')
        x, y = _xy(puppi_ref['puppiMET']['u_par_scaled_resolution'])
        axes[1].plot(x, y, color='magenta', lw=2, marker='s', ms=3, label='PUPPI')
    axes[0].set_title(r'Scaled $\sigma(u_{\perp})$')
    axes[1].set_title(r'Scaled $\sigma(u_{\parallel})$')
    for ax in axes:
        ax.set_xlabel(r'$q_{T}$ [GeV]')
        ax.set_xlim(0, 400)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Resolution [GeV]')
    fig.tight_layout()
    if out is not None:
        os.makedirs(out, exist_ok=True)
        res_out = os.path.join(out, 'resolution_comparison.png')
        fig.savefig(res_out, dpi=150, bbox_inches='tight')
        print(f"Saved {res_out}")
    if not show:
        plt.close(fig)
    return fig


def main():
    ap = argparse.ArgumentParser(description="Compare L1DeepMETv2 trainings' response/resolution.")
    ap.add_argument('folders', nargs='+', help='checkpoints folders (each holding a best.resolutions)')
    ap.add_argument('-o', '--out', default='.', help='output directory')
    ap.add_argument('--file', default='best.resolutions', help='resolutions file name to read in each folder')
    ap.add_argument('--no-puppi', dest='puppi', action='store_false', help="don't draw PUPPI baseline")
    args = ap.parse_args()

    plot_response(args.folders, out=args.out, puppi=args.puppi, fname=args.file)
    plot_resolution(args.folders, out=args.out, puppi=args.puppi, fname=args.file)


if __name__ == '__main__':
    main()
