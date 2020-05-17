import numpy as np
from jax.experimental import loops
from memory_profiler import profile
import jax.numpy as jnp
from jax import jit


def initialize(nx, ny):

    xgrid = np.linspace(-32, 32, nx)
    ygrid = np.linspace(-16, 16, ny)

    initial_density = get_initial_density(xgrid, ygrid)
    initial_velocity = get_initial_velocity(xgrid, ygrid)

    return initial_density, initial_velocity


def get_initial_density(xgrid, ygrid):
    xm, ym = np.meshgrid(xgrid, ygrid, indexing="ij")
    x0 = np.random.uniform(0.5 * xgrid.min(), 0.0)
    y0 = np.random.uniform(0.5 * ygrid.min(), 0.5 * ygrid.max())
    xs = 1e-1 * np.random.uniform()
    ys = 1e-2 * np.random.uniform()
    initial_density = np.exp(-xs * (xm - x0) ** 2.0 - ys * (ym - y0) ** 2.0)
    # plt.figure()

    # plt.contourf(xgrid, ygrid, initial_density.T)
    # plt.xlabel("x (Normalized Units)", fontsize=16)
    # plt.ylabel("y (Normalized Units)", fontsize=16)
    # plt.title("Initial Fluid Density", fontsize=18)
    # plt.colorbar()

    return initial_density


def get_initial_velocity(xgrid, ygrid):
    xm, ym = np.meshgrid(xgrid, ygrid, indexing="ij")
    ys = 1e-1 * np.random.uniform()
    initial_velocity = np.exp(-ys * ym ** 2.0)
    # plt.figure()
    # plt.contourf(xgrid, ygrid, initial_velocity.T)
    # plt.xlabel("x (Normalized Units)", fontsize=16)
    # plt.ylabel("y (Normalized Units)", fontsize=16)
    # plt.title("Initial Fluid Velocity", fontsize=18)
    # plt.colorbar()

    return initial_velocity


dt = 0.1


def get_dndt(velocity, density):

    return


@profile
def run_simulation_numpy(initial_density, initial_velocity, dt):
    density_prev = initial_density.copy()

    for i in range(10):
        grad_n = np.gradient(density_prev, axis=0)
        dndt = -initial_velocity * grad_n
        density_next = density_prev + dt * dndt
        density_prev = density_next

    return density_next

@jit
@profile
def run_simulation_jax(initial_density, initial_velocity, dt):
    density_prev = jnp.array(initial_density)

    for i in range(100):
        grad_n = jnp.gradient(density_prev, axis=0)
        dndt = -initial_velocity * grad_n
        density_next = density_prev + dt * dndt
        density_prev = density_next

    return density_next

@jit
@profile
def run_simulation_jax_loops(initial_density, initial_velocity, dt):
    with loops.Scope() as s:
        s.density_prev = jnp.array(initial_density)

        for i in range(10):
            s.grad_n = jnp.gradient(s.density_prev, axis=0)
            s.dndt = -initial_velocity * s.grad_n

            s.density_next = s.density_prev + dt * s.dndt
            s.density_prev = s.density_next

    return s.density_next