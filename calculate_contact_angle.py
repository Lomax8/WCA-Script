#!/usr/bin/env python
"""
Water Contact Angle (WCA) Calculator
Based on the method by Le Nhan Pham: https://github.com/lenhanpham/water-contact-angle
Calculates droplet contact angle from 2D number-density maps (e.g., from GROMACS MD simulations).
"""

import os
import argparse
import numpy as np
import sympy as sp
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def density_func(x, a, b, c):
    """Tanh density profile along the droplet symmetry axis."""
    return a * (1 - np.tanh((x - b) / c))


def circle_func(x, a, b, c):
    """Circular profile for droplet fitting."""
    return b + np.sqrt(c ** 2 - (x - a) ** 2)


def calculate_wca(
    dat_path: str,
    output_dir: str = None,
    cell_x: float = 26.26790,
    shift: float = 3.0,
    substrate_h: float = 1.7,
    save_plots: bool = True,
):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(dat_path))

    os.makedirs(output_dir, exist_ok=True)
    filename_base = os.path.splitext(os.path.basename(dat_path))[0]

    print(f"Loading density map from: {dat_path}")
    densmap = np.loadtxt(dat_path)

    # Convert number density to mass density (kg/m^3 for water)
    to_kg_m3 = ((18.01528) * (10 ** -3)) / ((6.022 * 10 ** 23) * (10 ** -27))
    densmass = densmap * to_kg_m3

    # Height and width coordinates
    z = densmap[0, 1:-1]
    x = densmap[1:-1, 0] + shift

    dens = densmass[1:-1, 1:-1]
    trans_dens = np.transpose(dens)

    # Center point with maximum density
    maxdens_idx = np.unravel_index(np.argmax(trans_dens, axis=None), trans_dens.shape)
    center_line = trans_dens[:, maxdens_idx[1]]

    # Center droplet in the box
    xshift = np.empty(x.size)
    for i in range(x.size):
        if x[i] > cell_x:
            xshift[i] = x[i] - cell_x - x[maxdens_idx[1]]
        else:
            xshift[i] = x[i] - x[maxdens_idx[1]]

    # Density profile fitting along symmetric axis
    density_line = np.transpose(center_line).flatten()

    # Look for fitting window
    idx_start = np.argmin(np.abs(z - 2.2))
    idx_stop = np.argmin(np.abs(z - 8.0))
    if idx_start >= idx_stop:
        idx_start, idx_stop = 0, len(z)

    params, params_cov = optimize.curve_fit(
        f=density_func,
        xdata=z[idx_start:idx_stop],
        ydata=density_line[idx_start:idx_stop],
        p0=[613, 4, 1],
        maxfev=10000,
    )
    half_density = params[0]

    # Find the contour level corresponding to the half-density / fitted density boundary
    plt.figure()
    masscontour = plt.contour(xshift, z, trans_dens, levels=[half_density])
    if hasattr(masscontour, "get_paths"):
        paths = masscontour.get_paths()
    elif hasattr(masscontour, "collections") and len(masscontour.collections) > 0:
        paths = masscontour.collections[0].get_paths()
    else:
        paths = []

    if not paths:
        raise RuntimeError("No contour found at the fitted density boundary.")
    circle_pts = paths[0].vertices
    plt.close()

    # Filter contour points above the surface to avoid near-substrate boundary artifacts
    threshold_z = substrate_h + 0.6  # typically ~2.3 - 2.4 nm
    valid_mask = circle_pts[:, 1] >= threshold_z
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        valid_pts = circle_pts
    else:
        valid_pts = circle_pts[valid_indices[0]:valid_indices[-1]]

    # Fit circle to contour
    pars, pars_cov = optimize.curve_fit(
        f=circle_func,
        xdata=valid_pts[:, 0],
        ydata=valid_pts[:, 1],
        p0=[0, -2, 5],
        maxfev=10000,
    )

    a, b, c = pars[0], pars[1], pars[2]
    h = substrate_h

    # Contact angle analytical calculation
    xs = sp.Symbol("x")
    xvalues = sp.solve(sp.sqrt(c ** 2 - (xs - a) ** 2) + b - h)
    
    if len(xvalues) >= 2:
        base = float(sp.Abs(xvalues[1] - xvalues[0])) / 2.0
    else:
        base = float(np.sqrt(c ** 2 - (h - b) ** 2))

    height = float(b + c - h)
    theta = np.arcsin((2 * height * base) / (base ** 2 + height ** 2))
    theta_deg = float(np.degrees(theta))

    print("=" * 45)
    print("RESULTS:")
    print(f"Droplet Center (x0, z0): ({a:.4f}, {b:.4f}) nm")
    print(f"Fitted Droplet Radius:    {c:.4f} nm")
    print(f"Droplet Height (above h): {height:.4f} nm")
    print(f"Contact Base Radius:     {base:.4f} nm")
    print(f"Contact Angle:           {theta_deg:.2f} deg")
    print("=" * 45)

    # Save summary report
    report_file = os.path.join(output_dir, f"{filename_base}_contact_angle.txt")
    with open(report_file, "w") as rf:
        rf.write(f"Source file: {dat_path}\n")
        rf.write(f"Droplet Center: ({a:.4f}, {b:.4f}) nm\n")
        rf.write(f"Fitted Droplet Radius: {c:.4f} nm\n")
        rf.write(f"Droplet Height: {height:.4f} nm\n")
        rf.write(f"Base Radius: {base:.4f} nm\n")
        rf.write(f"Contact Angle: {theta_deg:.2f} deg\n")
    print(f"Report saved to: {report_file}")

    # Generate plots
    if save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Droplet mass density contour
        im = axes[0].imshow(
            trans_dens,
            extent=[xshift.min(), xshift.max(), z.min(), z.max()],
            origin="lower",
            aspect="auto",
            cmap="jet",
        )
        axes[0].set_title("2D Mass Density Map")
        axes[0].set_xlabel("X (nm)")
        axes[0].set_ylabel("Z (nm)")
        plt.colorbar(im, ax=axes[0], label="Mass Density (kg/m³)")

        # Circle fit
        axes[1].plot(valid_pts[:, 0], valid_pts[:, 1], "r.", label="Contour Points")
        circle_patch = plt.Circle((a, b), c, color="b", fill=False, linewidth=2, label="Fitted Circle")
        axes[1].add_artist(circle_patch)
        axes[1].axhline(h, color="black", linestyle="--", label=f"Substrate (h={h}nm)")
        axes[1].set_xlim([valid_pts[:, 0].min() - 2, valid_pts[:, 0].max() + 2])
        axes[1].set_ylim([h - 0.5, b + c + 1])
        axes[1].set_aspect("equal")
        axes[1].set_title(f"Circle Fit (θ = {theta_deg:.1f}°)")
        axes[1].set_xlabel("X (nm)")
        axes[1].set_ylabel("Z (nm)")
        axes[1].legend()

        plot_path = os.path.join(output_dir, f"{filename_base}_fit_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Plot saved to: {plot_path}")

    return {
        "height": height,
        "base": base,
        "contact_angle_deg": theta_deg,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Water Contact Angle from 2D density map.")
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(os.path.dirname(__file__), "densmap-3000.dat"),
        help="Path to 2D density map .dat file (default: densmap-3000.dat)"
    )
    parser.add_argument("--substrate-h", type=float, default=1.7, help="Substrate height in nm (default: 1.7)")
    parser.add_argument("--cell-x", type=float, default=26.26790, help="Box length in X (default: 26.26790)")
    parser.add_argument("--shift", type=float, default=3.0, help="Shift in X (default: 3.0)")
    args = parser.parse_args()

    calculate_wca(
        dat_path=args.input,
        substrate_h=args.substrate_h,
        cell_x=args.cell_x,
        shift=args.shift,
    )
