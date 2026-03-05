import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from christoffel import Gamma, metric
from faraday import F2

# ── Coordenadas y velocidades simbólicas ──────────────────────────────────────
q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3')
u0, u1, u2, u3 = sp.symbols('u0 u1 u2 u3')

coords    = [q0, q1, q2, q3]
velocidad = [u0, u1, u2, u3]

# ── Parámetros del sistema ────────────────────────────────────────────────────
m   = 1     # masa (Schwarzschild, frutos, cuad_frutos)
a   = 0.5   # parámetro de frutos / cuad_frutos
mu  = 2.0   # parámetro de cuad_dipolo / cuad_frutos
q   = 0.1   # parámetro de frutos / cuad_frutos
qe  = 5     # carga eléctrica
qm  = qe/m  # cociente carga/masa para geodésicas no nulas
r_min = 2*m # para métricas distintas a mink

# Símbolos para sustitución paramétrica
m_sym, a_sym, mu_sym, q_sym, qe_sym = sp.symbols('m a mu q qe', real=True)

sym_params = {m_sym: m, a_sym: a, mu_sym: mu, q_sym: q, qe_sym: qe}

Gamma  = [Gi.subs(sym_params) for Gi in Gamma]
metric = metric.subs(sym_params)
F2     = F2.subs(sym_params)

# ── Ecuaciones de movimiento (geodésica + fuerza electromagnética) ────────────
aceleracion = sp.zeros(4, 1)

for i in range(4):
    for j in range(4):
        aceleracion[i] += qm * F2[i, j] * velocidad[j]
        for k in range(4):
            aceleracion[i] -= Gamma[i][j, k] * velocidad[j] * velocidad[k]

args_sym        = (*coords, *velocidad)
aceleracion_num = sp.lambdify(args_sym, aceleracion, 'numpy')
metrica_num     = sp.lambdify(coords,   metric,       'numpy')

# ── Funciones de integración ──────────────────────────────────────────────────
def sistema(_, Y):
    pos, vel = Y[:4], Y[4:]
    acc = np.array(aceleracion_num(*pos, *vel), dtype=float).flatten()
    return np.concatenate((vel, acc))


def normalizar_vt(r_ini, theta_ini, vr0, vth0, vphi0, phi_ini, masa=1.0):
    """Calcula u^t para que g_{μν} u^μ u^ν = -masa²."""
    g = np.array(metrica_num(0.0, r_ini, theta_ini, phi_ini), dtype=float)
    espacial = g[1,1]*vr0**2 + g[2,2]*vth0**2 + g[3,3]*vphi0**2
    cociente = (-masa**2 - espacial) / g[0,0]
    return np.sqrt(cociente) if cociente >= 0 else 1.0


def rk4(Y0, N, h, r_min, r_max):
    """Integrador Runge-Kutta de orden 4 con condiciones de parada."""
    Sol = np.zeros((N, 8))
    Sol[0] = Y0
    for i in range(N - 1):
        r = Sol[i, 1]
        if r <= r_min + 0.1 or r > r_max or not np.isfinite(r):
            Sol[i+1:] = Sol[i]
            break
        k1 = sistema(i*h,       Sol[i])
        k2 = sistema(i*h + h/2, Sol[i] + h*k1/2)
        k3 = sistema(i*h + h/2, Sol[i] + h*k2/2)
        k4 = sistema(i*h + h,   Sol[i] + h*k3)
        Sol[i+1] = Sol[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return Sol


def a_cartesianas(Sol):
    """Convierte coordenadas esféricas (r, θ, φ) a cartesianas."""
    r, theta, phi = Sol[:, 1], Sol[:, 2], Sol[:, 3]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# ── Parámetros de integración ─────────────────────────────────────────────────
theta0  = np.pi / 2
vtheta0 = 0.0
N       = 3000
tfinal  = 200
h       = tfinal / N
r0      = 60.0
n_rayos = 100
y_rango = (-25, 25)
y_min   = 0.5

# ── Configuración de la cámara ────────────────────────────────────────────────
ELEVACION = 90
AZIMUT    = 270

# ── Trazado de rayos ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 9))
ax  = fig.add_subplot(111, projection='3d')
ax.view_init(elev=ELEVACION, azim=AZIMUT)

for y0 in np.linspace(y_rango[0], y_rango[1], n_rayos):
    if abs(y0) < y_min:
        continue

    r_ini   = np.sqrt(r0**2 + y0**2)
    phi_ini = np.pi - np.arctan2(y0, r0)
    vr0     =  np.cos(phi_ini)
    vphi0   = -np.sin(phi_ini) / r_ini

    vt0 = normalizar_vt(r_ini, theta0, vr0, vtheta0, vphi0, phi_ini)
    Y0  = np.array([0.0, r_ini, theta0, phi_ini, vt0, vr0, vtheta0, vphi0])

    Sol     = rk4(Y0, N, h, r_min, 3*r0)
    mascara = (Sol[:, 1] > r_min + 0.1) & np.isfinite(Sol[:, 1])
    if mascara.sum() < 2:
        continue

    x, y, z = a_cartesianas(Sol[mascara])
    ax.plot(x, y, z, color='lime', lw=0.7, alpha=0.8)

# ── Horizonte y fotósfera (solo si r_min está definido) ───────────────────────
if r_min > 0.1:
    u_s = np.linspace(0, 2*np.pi, 60)
    v_s = np.linspace(0, np.pi,   60)
    sin_v = np.sin(v_s)
    cos_v = np.cos(v_s)

    for radio, color, alpha in [(r_min, 'black', 1.0), (1.5*r_min, 'orange', 0.08)]:
        xs = radio * np.outer(np.cos(u_s), sin_v)
        ys = radio * np.outer(np.sin(u_s), sin_v)
        zs = radio * np.outer(np.ones_like(u_s), cos_v)
        ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, zorder=5)

# ── Estética ──────────────────────────────────────────────────────────────────
lim = r0 * 0.5
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim),
       xlabel="x", ylabel="y", zlabel="z")
ax.set_title(
    f"Ray tracing geodésico + EM  |  Partícula masiva cargada\n"
    f"elev={ELEVACION}°  azim={AZIMUT}°  θ₀={np.degrees(theta0):.1f}°  vθ₀={vtheta0}",
    color='white'
)

fig.patch.set_facecolor('#0a0a0a')
ax.set_facecolor('#0a0a0a')
ax.tick_params(colors='white')
for label in [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]:
    label.set_color('white')
ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.3)

plt.tight_layout()
plt.savefig("raytracing.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()