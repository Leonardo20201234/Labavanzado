import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import matplotlib.pyplot as plt
import time

# ODE: oscilador armónico  dy/dt = f(t, y)
def f(t, y):
    return jnp.array([y[1], -y[0]])

@jit
def rk4_step(t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt * k1/2)
    k3 = f(t + dt/2, y + dt * k2/2)
    k4 = f(t + dt,   y + dt * k3)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

@partial(jit, static_argnums=(3,))
def rk4_integrate(y0, t0, dt, n_steps):
    def step(carry, _):
        t, y = carry
        y_new = rk4_step(t, y, dt)
        return (t + dt, y_new), y_new
    (_, _), ys = jax.lax.scan(step, (t0, y0), None, length=n_steps)
    return ys

# --- Parámetros ---
y0      = jnp.array([1.0, 0.0])
t0      = 0.0
dt      = 0.1
n_steps = 30

# --- Warm-up ---
print("Compilando JIT (warm-up)...")
_ = rk4_integrate(y0, t0, dt, n_steps).block_until_ready()

# --- Ejecución real ---
start = time.perf_counter()
ys = rk4_integrate(y0, t0, dt, n_steps).block_until_ready()
end = time.perf_counter()

print("=== JAX (JIT + lax.scan) ===")
print(f"Pasos         : {n_steps}")
print(f"Tiempo        : {end - start:.4f} s")
print(f"Primeras pos  : {ys[:5, 0]}")
print(f"Estado final  : {ys[-1]}")

# --- Gráfica ---
t_arr = jnp.linspace(t0 + dt, t0 + dt * n_steps, n_steps)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Oscilador Armónico — RK4 con JAX", fontsize=13)

# Posición vs tiempo
axes[0].plot(t_arr, ys[:, 0], color='royalblue', lw=0.8)
axes[0].set_title("Posición x(t)")
axes[0].set_xlabel("t")
axes[0].set_ylabel("x")
axes[0].grid(True, alpha=0.3)

# Velocidad vs tiempo
axes[1].plot(t_arr, ys[:, 1], color='tomato', lw=0.8)
axes[1].set_title("Velocidad v(t)")
axes[1].set_xlabel("t")
axes[1].set_ylabel("v")
axes[1].grid(True, alpha=0.3)

# Espacio de fases x vs v
axes[2].plot(ys[:, 0], ys[:, 1], color='seagreen', lw=0.5)
axes[2].set_title("Espacio de fases")
axes[2].set_xlabel("x")
axes[2].set_ylabel("v")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("oscilador_jax.png", dpi=150, bbox_inches='tight')
plt.show()