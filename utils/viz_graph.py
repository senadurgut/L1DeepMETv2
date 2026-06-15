"""
Visualize the particle graphs used by L1DeepMETv2 -- the eta-phi graph built by
radius_graph (same as training), and the ParticleNeXt-style pairwise edge features
[ln(dR), ln(kT), ln(z)] living on each edge.

Notebook usage:
    import sys; sys.path.insert(0, "/home/export/sdurgut/scratch/L1DeepMETv2")
    import viz_graph as vg
    ds = vg.load_dataset("data_ttbar")
    evt = ds[0]                                   # one event (a PyG Data)

    vg.plot_event(evt, color_by="puppi")          # graph, no edge features
    vg.plot_event(evt, edge_color="ln_kt")        # graph, edges colored by ln(kT)
    vg.plot_event_comparison(evt, edge_color="ln_kt")   # side-by-side
    vg.plot_edge_feature_distributions(ds, n_events=300)
    vg.plot_graph_stats(ds, n_events=500)

Node x = [pt, px, py, eta, phi, puppiWeight, pdgId, charge] (indices 0..7).
"""
import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from torch_cluster import radius_graph

from model.data_loader import METDataset
from model.graph_met_network_edge_features import DR_FLOOR

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except Exception:
    pass

# index of each feature in data.x
PT, PX, PY, ETA, PHI, PUPPI, PDGID, CHARGE = range(8)
PDG_LABELS = {1: "d", 2: "u", 11: "e", 13: r"$\mu$", 22: r"$\gamma$", 130: r"$K_L$", 211: r"$\pi$"}
EDGE_LABELS = {"ln_dr": r"$\ln \Delta R$", "ln_kt": r"$\ln k_T$", "ln_z": r"$\ln z$"}


def _repo_root():
    # walk up from this file until we find the dir containing the model/ package
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(d, "model")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))


def load_dataset(data_dir="data_ttbar"):
    """Return the indexable METDataset (ds[i] -> one event).

    A relative data_dir is resolved against the repo root (the folder holding
    model/), so this works regardless of the notebook's working directory.
    """
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_repo_root(), data_dir)
    ds = METDataset(data_dir)
    if len(ds) == 0:
        raise FileNotFoundError(
            f"No processed events under {data_dir}/processed. "
            f"Pass an absolute path to your data folder.")
    return ds


def build_edge_index(data, deltaR=0.4):
    """Rebuild the training graph for one event: radius_graph on (eta, phi)."""
    etaphi = torch.stack([data.x[:, ETA], data.x[:, PHI]], dim=1).float()
    return radius_graph(etaphi, r=deltaR, loop=False, max_num_neighbors=255)


def edge_features(data, edge_index, eps=1e-8):
    """Raw (physical-unit) pairwise edge features for each edge, as dict of np arrays.
    Mirrors GraphMETNetworkEdgeFeatures.compute_edge_features (same DR_FLOOR / clamps)."""
    x = data.x.float()
    s, d = edge_index[0], edge_index[1]
    pt_i, pt_j = x[s, PT], x[d, PT]
    eta_i, eta_j = x[s, ETA], x[d, ETA]
    phi_i, phi_j = x[s, PHI], x[d, PHI]
    dphi = (phi_i - phi_j + math.pi) % (2 * math.pi) - math.pi
    dr = torch.sqrt((eta_i - eta_j) ** 2 + dphi ** 2).clamp(min=DR_FLOOR)
    ptmin = torch.minimum(pt_i, pt_j)
    return {
        "ln_dr": torch.log(dr.clamp(min=eps)).numpy(),
        "ln_kt": torch.log((ptmin * dr).clamp(min=eps)).numpy(),
        "ln_z": torch.log((ptmin / (pt_i + pt_j).clamp(min=eps)).clamp(min=eps)).numpy(),
    }


def _node_color(data, color_by):
    x = data.x
    if color_by == "puppi":
        return x[:, PUPPI].numpy(), "PUPPI weight", (0.0, 1.0), "viridis"
    if color_by == "pt":
        return x[:, PT].numpy(), r"$p_T$ [GeV]", None, "plasma"
    if color_by == "charge":
        return x[:, CHARGE].numpy(), "charge", (-1.5, 1.5), "coolwarm"
    if color_by == "pdgid":
        return np.abs(x[:, PDGID].numpy()), "|pdgId|", None, "tab10"
    raise ValueError(f"color_by must be puppi|pt|charge|pdgid, got {color_by}")


def _edge_segments(data, edge_index):
    eta = data.x[:, ETA].numpy()
    phi = data.x[:, PHI].numpy()
    s, d = edge_index[0].numpy(), edge_index[1].numpy()
    return np.stack([np.column_stack([eta[s], phi[s]]),
                     np.column_stack([eta[d], phi[d]])], axis=1)  # (E, 2, 2)


def plot_event(data, deltaR=0.4, color_by="puppi", edge_color=None, draw_edges=True,
               ax=None, title=None, pt_size=12.0):
    """Plot one event's particles in eta-phi, with graph edges.

    edge_color: None -> plain gray edges (graph WITHOUT edge features);
                'ln_dr'|'ln_kt'|'ln_z' -> edges colored by that feature (WITH edge features).
    Node size scales with pT; node color set by `color_by`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ei = build_edge_index(data, deltaR)

    if draw_edges and ei.shape[1] > 0:
        segs = _edge_segments(data, ei)
        if edge_color is None:
            lc = LineCollection(segs, colors="0.7", linewidths=0.6, alpha=0.6, zorder=1)
            ax.add_collection(lc)
        else:
            vals = edge_features(data, ei)[edge_color]
            lc = LineCollection(segs, array=vals, cmap="magma", linewidths=1.3, alpha=0.85, zorder=1)
            ax.add_collection(lc)
            cb = ax.figure.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(EDGE_LABELS.get(edge_color, edge_color))

    c, clabel, clim, cmap = _node_color(data, color_by)
    sizes = pt_size + pt_size * data.x[:, PT].numpy()
    sc = ax.scatter(data.x[:, ETA].numpy(), data.x[:, PHI].numpy(), c=c, s=sizes,
                    cmap=cmap, edgecolors="k", linewidths=0.4, zorder=2)
    if clim is not None:
        sc.set_clim(*clim)
    cb2 = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.10)
    cb2.set_label(clabel)

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_ylim(-math.pi - 0.2, math.pi + 0.2)
    n_e = ei.shape[1] // 2  # radius_graph yields both directions
    ax.set_title(title or f"{data.x.shape[0]} particles, {n_e} edges (dR<{deltaR})")
    return ax


def plot_event_comparison(data, deltaR=0.4, color_by="puppi", edge_color="ln_kt"):
    """Side-by-side: plain graph (no edge features) vs edges colored by `edge_color`."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_event(data, deltaR, color_by, edge_color=None, ax=axes[0], title="Graph (no edge features)")
    plot_event(data, deltaR, color_by, edge_color=edge_color, ax=axes[1],
               title=f"Graph + {EDGE_LABELS.get(edge_color, edge_color)}")
    fig.tight_layout()
    return fig


def _collect_edge_features(dataset, n_events, deltaR):
    idx = range(min(n_events, len(dataset)))
    acc = {"ln_dr": [], "ln_kt": [], "ln_z": []}
    for i in idx:
        data = dataset[i]
        ei = build_edge_index(data, deltaR)
        if ei.shape[1] == 0:
            continue
        f = edge_features(data, ei)
        for k in acc:
            acc[k].append(f[k])
    return {k: np.concatenate(v) if v else np.array([]) for k, v in acc.items()}


def plot_edge_feature_distributions(dataset, n_events=200, deltaR=0.4, bins=60):
    """Histogram ln(dR), ln(kT), ln(z) over the edges of the first `n_events` events."""
    feats = _collect_edge_features(dataset, n_events, deltaR)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key in zip(axes, ("ln_dr", "ln_kt", "ln_z")):
        ax.hist(feats[key], bins=bins, histtype="stepfilled", alpha=0.7)
        ax.set_xlabel(EDGE_LABELS[key])
        ax.set_ylabel("edges")
        ax.set_title(f"{EDGE_LABELS[key]}  (mean {feats[key].mean():.2f}, std {feats[key].std():.2f})")
    fig.suptitle(f"Edge features over {min(n_events, len(dataset))} events "
                 f"({sum(len(feats[k]) for k in feats)//3} edges)")
    fig.tight_layout()
    return fig


def plot_graph_stats(dataset, n_events=500, deltaR=0.4, bins=40):
    """Distributions of nodes/event, edges/event, and node degree."""
    n_nodes, n_edges, degrees = [], [], []
    for i in range(min(n_events, len(dataset))):
        data = dataset[i]
        ei = build_edge_index(data, deltaR)
        n_nodes.append(data.x.shape[0])
        n_edges.append(ei.shape[1] // 2)
        if ei.shape[1] > 0:
            deg = torch.bincount(ei[0], minlength=data.x.shape[0]).numpy()
            degrees.append(deg)
    degrees = np.concatenate(degrees) if degrees else np.array([])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(n_nodes, bins=bins); axes[0].set_xlabel("particles / event"); axes[0].set_ylabel("events")
    axes[1].hist(n_edges, bins=bins); axes[1].set_xlabel(f"edges / event (dR<{deltaR})"); axes[1].set_ylabel("events")
    axes[2].hist(degrees, bins=range(0, int(degrees.max()) + 2) if degrees.size else bins)
    axes[2].set_xlabel("node degree"); axes[2].set_ylabel("particles")
    fig.suptitle(f"Graph statistics over {min(n_events, len(dataset))} events")
    fig.tight_layout()
    return fig
