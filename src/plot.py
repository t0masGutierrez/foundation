from typing import Annotated
import jax
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path


def plot_init(
        x: Annotated[jax.Array, "(nx,)"],
        u: Annotated[jax.Array, "(n, nx)"]
        ) -> Figure:
    """
    plot initial solution

    parameters
    ----------
    x
        spatial coordinates
    u
        initial solution

    returns
    -------
    fig
        figure
    """
    path = str(Path(__file__).resolve().parents[1] / "plots" / "initial.png")
    fig, ax = plt.subplots()
    ax.plot(x, u)
    fig.supxlabel("x")
    fig.supylabel("u(0, x)")
    fig.suptitle("initial condition")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_step(
        x: Annotated[jax.Array, "(nx,)"],
        u: Annotated[jax.Array, "(nx,)"],
        u_tau: Annotated[jax.Array, "(nx,)"]
        ) -> Figure:
    """
    plot time-stepped solution

    parameters
    ----------
    u
        solution
    u_tau
        solution at next time step
    returns
    -------
    fig
        figure
    """
    path = str(Path(__file__).resolve().parents[1] / "plots" / "step.png")
    fig, ax = plt.subplots()
    ax.plot(x, u, label="initial solution: u(t, x)")
    ax.plot(x, u_tau, label="time-stepped solution: u(t + tau, x)")
    fig.legend()
    fig.supxlabel("x")
    fig.supylabel("u(t, x)")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig

def plot_traj(
        x: Annotated[jax.Array, "(nx,)"],
        t: Annotated[jax.Array, "(nt,)"],
        trajs: Annotated[jax.Array, "(nt, nx)"]
        ) -> Figure:
    """
    plot trajectories

    parameters
    ----------
    x
        spatial coordinates
    t
        temporal coordinates
    trajs
        solution trajectories

    returns
    -------
    fig
        figure
    """
    path = str(Path(__file__).resolve().parents[1] / "plots" / "traj.png")
    # plot image
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im = ax[0].imshow(
        trajs,
        extent=[x[0], x[-1], t[0], t[-1]],
        origin="lower",
        aspect="auto",
        cmap="plasma_r"
        )
    ax[0].set_ylabel("time (t)")
    fig.colorbar(mappable=im, ax=ax[0])

    # plot curve
    norm = Normalize(vmin=min(t), vmax=max(t))
    cmap = plt.get_cmap("viridis_r")
    step = max(1, len(trajs) // 10)
    for time, sol in zip(t[step::len(trajs) // 10], trajs[step:: step]):
        ax[1].plot(x, sol, color=cmap(norm(time)))
    ax[1].set_ylabel("u(t, x)")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable=sm, ax=ax[1], label="time (t)")

    # configure plot
    fig.supxlabel("space (x)")
    fig.suptitle("trajectories")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig

def main() -> None:
    pass


if __name__ == "__main__":
    main()