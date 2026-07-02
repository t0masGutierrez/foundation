from typing import Annotated
import jax
from matplotlib import pyplot as plt
from generate import rk4_step


def plot_init(
        x: Annotated[jax.Array, "(nx,)"],
        u_0: Annotated[jax.Array, "(n, nx)"]
        ) -> None:
    """
    plot initial solution

    parameters
    ----------
    x
        spatial coordinates
    u_0
        initial solution

    returns
    -------
    None
    """
    for i in range(len(u_0)):
        plt.plot(x, u_0[i])
        plt.xlabel("x")
        plt.ylabel("u(0, x)")
        plt.title("initial conditions")
        plt.show()
    return None


def plot_step(
        x: Annotated[jax.Array, "(nx,)"],
        u: Annotated[jax.Array, "(nx,)"],
        u_dt: Annotated[jax.Array, "(nx,)"]
        ) -> None:
    """
    plot time-stepped solution

    parameters
    ----------
    u
        solution
    u_dt
        solution differential between time step
    returns
    -------
    None
    """
    plt.plot(x, u, label="initial solution: u(t, x)")
    plt.plot(x, u + u_dt, label="time-stepped solution: u(t + dt, x)")
    plt.xlabel("x")
    plt.ylabel("u(t, x)")
    plt.legend()
    plt.show()
    return None

def plot_traj(
        x: Annotated[jax.Array, "(nx,)"],
        t: Annotated[jax.Array, "(nt,)"],
        trajs: Annotated[jax.Array, "(nt, nx)"]
        ) -> None:
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
    None
    """
    # plot image
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].imshow(trajs,
            extent=[x[0], x[-1], t[0], t[-1]],
            origin="lower",
            aspect="auto")
    ax[0].set_ylabel("time (t)")

    # plot curve
    for i in range(0, len(trajs), len(trajs) // 10):
        ax[1].plot(x, trajs[i])
    ax[1].set_ylabel("u(t, x)")

    # configure plot
    fig.supxlabel("space (x)")
    fig.suptitle("trajectories")
    plt.show()
    return None

def main() -> None:
    pass


if __name__ == "__main__":
    main()