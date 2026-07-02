from __future__ import annotations
from pathlib import Path
from typing import Annotated, Any, Sequence
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr


def load_config(
        path: str | Path
        ) -> dict[str, Any]:
    """
    load YAML configuration file into Python configuration object

    parameters
    ----------
    path
        path to the YAML configuration file

    returns
    -------
    config
        nested configuration object containing experiment, space, time, pde, initial_condition, boundary_condition, solver, dataset, model, training, evaluation, and logging settings
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def build_domain(
        config: dict[str, Any]
        ) -> tuple[
            Annotated[jax.Array, "(nx,)"],
            Annotated[jax.Array, "(nt,)"]
            ]:
    """
    configure spatiotemporal domain

    parameters
    ----------
    config
        nested configuration object containing experiment, space, time, pde, initial_condition, boundary_condition, solver, dataset, model, training, evaluation, and logging settings
    
    returns
    -------
    x
        spatial coordinates
    t
        temporal coordinates
    """
    # spatial domain
    # TODO: convert from 1d x to nd R
    # dim = config["space"]["dimension"]
    # axis = list(config["space"]["axis"].keys())
    # R = dict()
    # for i in range(dim):
    #     ax = axis[i]
    #     min = config["space"]["axis"][ax]["min"]
    #     max = config["space"]["axis"][ax]["max"]
    #     n = config["space"]["axis"][ax]["n"]
    #     periodic = config["space"]["axis"][ax]["periodic"]
    #     R[ax] = jnp.linspace(min, max, n, endpoint=periodic)
    min = config["space"]["axis"]["x"]["min"]
    max = config["space"]["axis"]["x"]["max"]
    n = config["space"]["axis"]["x"]["n"]
    periodic = config["space"]["axis"]["x"]["periodic"]
    x = jnp.linspace(min, max, n, endpoint=not periodic)

    # temporal domain
    min = config["time"]["min"]
    max = config["time"]["max"]
    n = config["time"]["n"]
    t = jnp.linspace(min, max, n)
    return x, t


def covariance(
    x: Annotated[jax.Array, "(nx,)"],
    *,
    sigma: float = 1.0,
    ell: float = 1.0,
    eps: float = 1e-5,
) -> Annotated[jax.Array, "(nx, nx)"]:
    """
    compute the covariance between u(x_i) and u(x_j)
    symmetric: K = K^T
    positive semidefinite: x^T * K * x >= 0

    parameters
    ----------
    x
        spatial coordinate array
    sigma
        standard deviation
    ell
        speed of correlation decay
    eps
        safety factor for numerical stability

    returns
    -------
    L
        lower triangular covariance matrix
    """
    distance = x[:, None] - x[None, :]

    # periodic gaussian kernel
    K = sigma**2 * jnp.exp(-1 * (1 - jnp.cos(2 * jnp.pi * distance)) / ell ** 2)

    # symmetrize covariance matrix
    K = 0.5 * (K + K.T)

    # regularize covariance matrix
    I = jnp.eye(K.shape[0])
    K += eps * I

    # factorize covariance matrix
    L = jnp.linalg.cholesky(K)
    return L


def initialize(
    x: Annotated[jax.Array, "(nx,)"],
    L: Annotated[jax.Array, "(nx, nx)"],
    key: Annotated[jax.Array, "() | (2,)"],
    *,
    n: int,
) -> Annotated[jax.Array, "(n, nx)"]:
    """
    generate initial conditions

    parameters
    ----------
    x
        spatial coordinate array
    L
        lower triangular covariance matrix
    key
        random number generator
    n
        number of initial conditions
    
    returns
    -------
    u_0
        initial conditions
    """
    nx = len(x)
    u_0 = np.empty((n, nx))
    for i in range(n):
        key, subkey = jr.split(key)
        z = jr.normal(subkey, (nx,))
        u_0[i] = jnp.matmul(L, z)
    u_0 = jnp.array(u_0)
    return u_0


def sample_coeff(
    domain: Annotated[Sequence[float], "(2,)"],
    key: Annotated[jax.Array, "() | (2,)"],
    *,
    n: int,
) -> Annotated[jax.Array, "(n,)"]:
    """
    sample coefficients from uniform probability distribution

    parameters
    ----------
    domain
        minimum and maximum value of coefficients
    key
        random number generator
    n
        number of coefficients
    
    returns
    -------
    coeffs
        coefficients
    """
    return jr.uniform(key, (n,), minval=domain[0], maxval=domain[1])


def flux(
    u: Annotated[jax.Array, "() | (nx,)"],
    coeffs: Annotated[jax.Array, "(3,)"],
) -> Annotated[jax.Array, "() | (nx,)"]:
    """
    compute flux

    parameters
    ----------
    u
        solution
    coeffs
        coefficients

    returns
    -------
    f
        flux
    """
    return coeffs[0] * u ** 3 + coeffs[1] * u ** 2 + coeffs[2] * u


def speed(
    u: Annotated[jax.Array, "() | (nx,)"],
    coeffs: Annotated[jax.Array, "(3,)"],
) -> Annotated[jax.Array, "() | (nx,)"]:
    """
    compute speed

    parameters
    ----------
    u
        solution
    coeffs
        coefficients
    
    returns
    ------
    speed
        speed
    """
    J = jax.jacrev(lambda value: flux(value, coeffs))(u)
    return jnp.abs(J)


def max_speed(
    speed: Annotated[jax.Array, "() | (nx,)"],
    *,
    eps: float = 1e-8,
) -> Annotated[jax.Array, "()"]:
    """
    compute max speed

    parameters
    ----------
    speed
        rate of change of position
    eps
        safety factor for numerical stability
    
    returns
    ------
    max_speed
        maximum rate of change of position
    """
    max_speed = jnp.max(speed)
    return jnp.maximum(max_speed, eps)


def time_step(
    x: Annotated[jax.Array, "(nx,)"],
    dx: float,
    max_speed: Annotated[jax.Array, "()"],
    *,
    cfl: float = 0.5,
) -> Annotated[jax.Array, "()"]:
    """
    compute time step

    parameters
    ----------
    max_speed
        maximum rate of change of position
    dx
        grid spacing
    cfl
        safety factor for limiting the size of time step (courant-friedrachs-lewy)
    
    returns
    ------
    dt
        time step
    """
    return cfl * dx / max_speed


def neighbor(
    u: Annotated[jax.Array, "(nx,)"],
    *,
    i: int,
    n: int,
) -> tuple[
    Annotated[jax.Array, "()"],
    Annotated[jax.Array, "()"],
]:
    """
    compute solution at neighboring spatial coordinate

    parameters
    ----------
    u
        solution
    i
        index of center solution
    n
        number of neighbors
    
    returns
    ------
    before_sol
        solution before center solution
    after_sol
        solution after center solution
    """
    before_index = (i - n) % len(u)
    after_index = (i + n) % len(u)
    before_sol = u[before_index]
    after_sol = u[after_index]
    return before_sol, after_sol


def numerical_flux(
    u: Annotated[jax.Array, "(nx,)"],
    coeffs: Annotated[jax.Array, "(3,)"],
) -> Annotated[jax.Array, "(nx,)"]:
    """
    compute numerical flux

    parameters
    ----------
    u
        solution
    coeffs
        coefficients

    returns
    -------
    num_flux
        numerical flux
    """
    ORDER = 5  # 5th order WENO
    CENTER = 2  # index of central solution

    num_flux = np.empty(u.shape)
    for i in range(len(u)):

        def create_stencil(
            u: Annotated[jax.Array, "(nx,)"],
            *,
            i: int,
            side: str,
        ) -> tuple[
            Annotated[jax.Array, "(3,)"],
            Annotated[jax.Array, "(3,)"],
            Annotated[jax.Array, "(3,)"],
        ]:
            """
            choose spatial stencil for approximating solution at cell interface \\
            left = {i-2, i-1, i, i+1, i+2} \\
            right = {i+3, i+2, i+1, i, i-1}
            """
            if side == "right":
                i += 1
            stencil = np.empty((ORDER,))
            for n in range(ORDER - CENTER):
                before, after = neighbor(u, i=i, n=n)
                stencil[CENTER - n] = before
                stencil[CENTER + n] = after
            if side == "right":
                stencil = jnp.flip(stencil)
            s0 = jnp.array([stencil[0], stencil[1], stencil[2]])
            s1 = jnp.array([stencil[1], stencil[2], stencil[3]])
            s2 = jnp.array([stencil[2], stencil[3], stencil[4]])
            return s0, s1, s2

        left_s0, left_s1, left_s2 = create_stencil(u, i=i, side="left")
        right_s0, right_s1, right_s2 = create_stencil(u, i=i, side="right")

        def smooth_stencil(
            s0: Annotated[jax.Array, "(3,)"],
            s1: Annotated[jax.Array, "(3,)"],
            s2: Annotated[jax.Array, "(3,)"],
        ) -> tuple[
            Annotated[jax.Array, "()"],
            Annotated[jax.Array, "()"],
            Annotated[jax.Array, "()"],
        ]:
            """
            compute smoothness of stencil
            """
            beta0 = 13 / 12 * (s0[0] - 2 * s0[1] + s0[2]) ** 2 + 1 / 4 * (
                s0[0] - 4 * s0[1] + 3 * s0[2]
            ) ** 2
            beta1 = 13 / 12 * (s1[0] - 2 * s1[1] + s1[2]) ** 2 + 1 / 4 * (
                s1[0] - s1[2]
            ) ** 2
            beta2 = 13 / 12 * (s2[0] - 2 * s2[1] + s2[2]) ** 2 + 1 / 4 * (
                3 * s2[0] - 4 * s2[1] + s2[2]
            ) ** 2
            return beta0, beta1, beta2

        left_beta0, left_beta1, left_beta2 = smooth_stencil(left_s0, left_s1, left_s2)
        right_beta0, right_beta1, right_beta2 = smooth_stencil(right_s0, right_s1, right_s2)

        def weight_stencil(
            beta0: Annotated[jax.Array, "()"],
            beta1: Annotated[jax.Array, "()"],
            beta2: Annotated[jax.Array, "()"],
            *,
            eps: float = 1e-6,
        ) -> tuple[
            Annotated[jax.Array, "()"],
            Annotated[jax.Array, "()"],
            Annotated[jax.Array, "()"],
        ]:
            """
            if smooth stencil then large weight \\
            if rough stencil then small weight
            """
            alpha0 = 1 / 10 / (eps + beta0) ** 2
            alpha1 = 6 / 10 / (eps + beta1) ** 2
            alpha2 = 3 / 10 / (eps + beta2) ** 2
            return alpha0, alpha1, alpha2

        left_alpha0, left_alpha1, left_alpha2 = weight_stencil(left_beta0, left_beta1, left_beta2)
        right_alpha0, right_alpha1, right_alpha2 = weight_stencil(right_beta0, right_beta1, right_beta2)

        def norm_weight(
            alpha0: Annotated[jax.Array, "()"],
            alpha1: Annotated[jax.Array, "()"],
            alpha2: Annotated[jax.Array, "()"],
        ) -> tuple[
            Annotated[jax.Array, "()"],
            Annotated[jax.Array, "()"],
            Annotated[jax.Array, "()"],
        ]:
            """
            normalize stencil weight
            """
            omega0 = alpha0 / (alpha0 + alpha1 + alpha2)
            omega1 = alpha1 / (alpha0 + alpha1 + alpha2)
            omega2 = alpha2 / (alpha0 + alpha1 + alpha2)
            return omega0, omega1, omega2

        left_omega0, left_omega1, left_omega2 = norm_weight(left_alpha0, left_alpha1, left_alpha2)
        right_omega0, right_omega1, right_omega2 = norm_weight(
            right_alpha0, right_alpha1, right_alpha2
        )

        def approx_sol(
            s0: Annotated[jax.Array, "(3,)"],
            s1: Annotated[jax.Array, "(3,)"],
            s2: Annotated[jax.Array, "(3,)"],
        ) -> tuple[
            Annotated[jax.Array, "()"],
            Annotated[jax.Array, "()"],
            Annotated[jax.Array, "()"],
        ]:
            """
            approximate solution at cell interface using Lagrange interpolation \\
            q \\approx u(x_{i+0.5})
            """
            q0 = 1 / 3 * s0[0] - 7 / 6 * s0[1] + 11 / 6 * s0[2]
            q1 = -1 / 6 * s1[0] + 5 / 6 * s1[1] + 1 / 3 * s1[2]
            q2 = 1 / 3 * s2[0] + 5 / 6 * s2[1] - 1 / 6 * s2[2]
            return q0, q1, q2

        left_q0, left_q1, left_q2 = approx_sol(left_s0, left_s1, left_s2)
        right_q0, right_q1, right_q2 = approx_sol(right_s0, right_s1, right_s2)

        # reconstruct solution
        """
        WENO5 (Weighted Essentially Non-Oscillatory fifth-order) reconstructs high-order estimates of the solution at cell interfaces by adaptively combining several lower-order stencils, assigning larger weights to smooth stencils and smaller weights to stencils containing discontinuities. This produces fifth-order accuracy in smooth parts of the solution while preventing the spurious oscillations that standard high-order methods generate near shocks and other sharp gradients.
        """
        left_sol = left_q0 * left_omega0 + left_q1 * left_omega1 + left_q2 * left_omega2
        right_sol = right_q0 * right_omega0 + right_q1 * right_omega1 + right_q2 * right_omega2

        # compute local lax-friedrichs / rusanov flux
        left_flux = flux(left_sol, coeffs)
        right_flux = flux(right_sol, coeffs)
        left_speed = speed(left_sol, coeffs)
        right_speed = speed(right_sol, coeffs)
        left_max_speed = max_speed(left_speed)
        right_max_speed = max_speed(right_speed)
        bi_max_speed = jnp.maximum(left_max_speed, right_max_speed)
        num_flux[i] = 0.5 * (
            left_flux + right_flux
            ) - 0.5 * bi_max_speed * (
            right_sol - left_sol
            )
    return jnp.array(num_flux)


def rhs(
    num_flux: Annotated[jax.Array, "(nx,)"],
    dx: float,
) -> Annotated[jax.Array, "(nx,)"]:
    """
    compute right hand side of equation \\
    du/dt = -df(u)/dx

    parameters
    ----------
    num_flux
        numerical flux
    dx
        grid spacing
    
    returns
    -------
    rhs
        rhs of 1d conservation law
    """
    dfdx = np.empty(num_flux.shape)
    for i in range(len(num_flux)):
        dfdx[i] = -(num_flux[i] - num_flux[i - 1]) / dx
    return jnp.array(dfdx)


def rk4_step(
    u: Annotated[jax.Array, "(nx,)"],
    dx: float,
    dt: float,
) -> Annotated[jax.Array, "(nx,)"]:
    """
    approximate solution at next time step using 4th order runge-kutta

    parameters
    ----------
    u
        solution
    dx
        grid spacing
    dt
        time step

    returns
    -------
    u_dt
        solution differential between time step
    """
    k1 = rhs(u, dx)
    k2 = rhs(u + k1 * dt / 2, dx)
    k3 = rhs(u + k2 * dt / 2, dx)
    k4 = rhs(u + k3 * dt, dx)
    u_dt = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_dt


def rk4_evolve(
    u: Annotated[jax.Array, "(nx,)"],
    t: Annotated[jax.Array, "(nt,)"],
    dx: float,
    dt: float,
) -> Annotated[jax.Array, "(nt, nx)"]:
    """
    evolve approximate solution through time using repeated RK4 steps

    parameters
    ----------
    u
        solution
    t
        temporal coordinates
    dx
        grid spacing
    dt
        time step

    returns
    -------
    trajs
        solution trajectories
    """
    t = t[0]
    max_time = t[-1]
    trajs = list()
    while t < max_time:
        trajs.append(u)
        u_dt = rk4_step(u, dx, dt)
        u += u_dt
        t += dt
    trajs = jnp.array(trajs)
    return trajs


def demonstrate(
    trajs: Annotated[jax.Array, "(nt, nx)"],
    *,
    n: int,
) -> Annotated[jax.Array, "(num_examples, 2, nx)"]:
    """
    create (input, target) examples

    parameters
    ----------
    trajs
        solution trajectories
    n
        number of time steps
    
    returns
    -------
    examples
        input-target examples
    """
    num_exs = len(trajs) - n
    examples = list()
    for i in range(num_exs):
        u_t = trajs[i]
        u_tau = trajs[i + n]
        ex = (u_t, u_tau)
        examples.append(ex)
    examples = jnp.array(examples)
    return examples


def main() -> None:
    pass


if __name__ == "__main__":
    main()
