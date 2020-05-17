import numpy as np
from jax.experimental import loops
from memory_profiler import profile
import jax.numpy as jnp
from jax import jit


def initialize(nx, ny):
    """
    Initialize the density and velocity profiles in 2D

    :param nx:
    :param ny:
    :return:
    """

    xgrid = np.linspace(-32, 32, nx)
    ygrid = np.linspace(-16, 16, ny)

    initial_density = get_initial_density(xgrid, ygrid)
    initial_velocity = get_initial_velocity(xgrid, ygrid)

    return initial_density, initial_velocity


def get_initial_density(xgrid, ygrid):
    """
    Initialize a "relatively" random density profile. Please see notebook for visualization

    :param xgrid:
    :param ygrid:
    :return:
    """
    xm, ym = np.meshgrid(xgrid, ygrid, indexing="ij")
    x0 = np.random.uniform(0.5 * xgrid.min(), 0.0)
    y0 = np.random.uniform(0.5 * ygrid.min(), 0.5 * ygrid.max())
    xs = 1e-1 * np.random.uniform()
    ys = 1e-2 * np.random.uniform()
    initial_density = np.exp(-xs * (xm - x0) ** 2.0 - ys * (ym - y0) ** 2.0)
    return initial_density


def get_initial_velocity(xgrid, ygrid):
    """
    Initialize a "relatively" random velocity profile. Please see notebook for visualization

    :param xgrid:
    :param ygrid:
    :return:
    """
    xm, ym = np.meshgrid(xgrid, ygrid, indexing="ij")
    ys = 1e-1 * np.random.uniform()
    initial_velocity = np.exp(-ys * ym ** 2.0)

    return initial_velocity

@profile
def run_simulation_numpy(initial_density, initial_velocity, dt):
    """
    Run a simulation for 10 steps using NumPy as a baseline

    :param initial_density:
    :param initial_velocity:
    :param dt:
    :return:
    """
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
    """
    Run a simulation for 10 steps using JAX but with regular for loops.

    The eventual learning here is that JAX unrolls the loops, and that can take a long time.

    It probably also costs more in memory but the `@profile` decorator doesn't see it unfortunately


    :param initial_density:
    :param initial_velocity:
    :param dt:
    :return:
    """
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
    """
    The eventual learning here is that JAX unrolls the loops, and that can take a long time.
    If you use the scoping like below, though, JAX doesn't unroll the loop and just runs through!

    Saves a lot of time, and probably memory

    TODO
    Measure the memory savings

    :param initial_density:
    :param initial_velocity:
    :param dt:
    :return:
    """
    with loops.Scope() as s:
        s.density_prev = jnp.array(initial_density)

        for i in range(10):
            s.grad_n = jnp.gradient(s.density_prev, axis=0)
            s.dndt = -initial_velocity * s.grad_n

            s.density_next = s.density_prev + dt * s.dndt
            s.density_prev = s.density_next

    return s.density_next