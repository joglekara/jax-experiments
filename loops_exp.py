import jax_for_loops_experiment
import time

if __name__ == "__main__":
    dt = 0.1
    init_density, init_velocity = jax_for_loops_experiment.initialize(512, 256)

    t1 = time.perf_counter()
    numpy_out = jax_for_loops_experiment.run_simulation_jax_loops(
        init_density, init_velocity, dt
    )
    print("This took " + str(time.perf_counter() - t1) + " s.")
