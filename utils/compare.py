"""
Compare response and resolution curves of two (or more) trainings on the same
plots. Folders may be L1DeepMETv2 or TransforMET runs, mixed freely -- the format
is auto-detected and normalized, so the curves overlay directly (both use the same
binning and the same response/resolution definitions).

Usage:
    python compare.py <folder_1> <folder_2> [<folder_3> ...] [-o out_dir]

Each folder is a run/checkpoints folder holding either an L1DeepMETv2
`best.resolutions` (cloudpickle) or a TransforMET `*.resolutions.npz`. Saves
response_comparison.png and resolution_comparison.png into out_dir
(default: current directory).
"""
import os
import argparse
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import load

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


def _load(folder, fname='best.resolutions'):
    """Load a run's resolution histograms as an L1-style nested dict
    {'MET': {...}, 'puppiMET': {...}}, each entry a (values, bin_edges) tuple.

    Auto-detects the framework: TransforMET writes flat arrays in a
    `*.resolutions.npz`; L1DeepMETv2 writes the nested dict directly via
    cloudpickle. Both use identical binning and identical response/resolution
    definitions (TransforMET's met_response == L1's R), so the normalized dicts
    overlay bin-for-bin.
    """
    path = folder if os.path.isabs(folder) else os.path.join(os.getcwd(), folder)
    npz = glob(os.path.join(path, '*.resolutions.npz'))
    if npz:  # TransforMET
        d = np.load(npz[0])
        e = d['bin_edges']
        return {
            'MET': {
                'R': (d['met_response'], e),
                'u_perp_resolution': (d['met_u_perp'], e),
                'u_perp_scaled_resolution': (d['met_u_perp_scaled'], e),
                'u_par_resolution': (d['met_u_par'], e),
                'u_par_scaled_resolution': (d['met_u_par_scaled'], e),
            },
            'puppiMET': {
                'R': (d['puppi_response'], e),
                'u_perp_resolution': (d['puppi_u_perp'], e),
                'u_perp_scaled_resolution': (d['puppi_u_perp_scaled'], e),
                'u_par_resolution': (d['puppi_u_par'], e),
                'u_par_scaled_resolution': (d['puppi_u_par_scaled'], e),
            },
        }
    # L1DeepMETv2 (cloudpickle, lz4-compressed) -- already in the target structure
    f = os.path.join(path, fname)
    if not os.path.exists(f):
        raise FileNotFoundError(f"No *.resolutions.npz or {fname} found in {path}")
    return load(f)


def plot_response(folders, out=None, puppi=True, show=False, labels=None, fname='best.resolutions'):
    """Overlay the MET response curve (R = -<u_par>/<q_T>) for each run folder.

    out: if given, save response_comparison.png into this dir.
    show: if True, leave the figure open (for inline display in notebooks).
    labels: optional list of legend labels (defaults to folder names).
    """
    if isinstance(folders, str):
        folders = [folders]
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
    if isinstance(folders, str):
        folders = [folders]
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
