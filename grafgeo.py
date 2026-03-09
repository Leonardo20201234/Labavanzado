import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
from christoffel import christoffel, metric
from faraday import F2_tensor

# ── Parámetros del sistema ────────────────────────────────────────────────────
m   = 1     # masa
a   = 1   # parámetro de frutos / cuad_frutos
mu  = 1     # parámetro de cuad_dipolo / cuad_frutos
q   = 0   # parámetro de frutos / cuad_frutos
qe  = 1     # carga eléctrica
qm  = qe/m  # cociente carga/masa
r_min = 2*m   # para métricas distintas a mink

# ── Helpers para obtener G y F2 como arrays numpy ────────────────────────────
def get_G_F2(q0, q1, q2, q3):
    """Devuelve G (4,4,4) y F2 (4,4) como arrays numpy contiguos."""
    params = (q0, q1, q2, q3, mu, m, a, q, qe)
    G_list = christoffel(*params)
    G  = np.array([np.array(Gi, dtype=np.float64) for Gi in G_list])
    F2 = np.array(F2_tensor(*params), dtype=np.float64)
    return G, F2

def get_metric(q0, q1, q2, q3):
    return np.array(metric(q0, q1, q2, q3, mu, m, a, q, qe), dtype=np.float64)

# ── Núcleos compilados con Numba ──────────────────────────────────────────────
@njit(cache=True)
def _aceleracion(vel, G, F2, qm):
    acc = np.zeros(4)
    for i in range(4):
        for j in range(4):
            acc[i] += qm * F2[i, j] * vel[j]
            for k in range(4):
                acc[i] -= G[i, j, k] * vel[j] * vel[k]
    return acc


@njit(cache=True)
def _rk4_step(pos, vel, G, F2, qm, h):
    def deriv(p, v):
        acc = _aceleracion(v, G, F2, qm)
        return v, acc

    vk1, ak1 = deriv(pos,           vel)
    vk2, ak2 = deriv(pos + h/2*vk1, vel + h/2*ak1)
    vk3, ak3 = deriv(pos + h/2*vk2, vel + h/2*ak2)
    vk4, ak4 = deriv(pos + h*vk3,   vel + h*ak3)

    pos_new = pos + (h/6) * (vk1 + 2*vk2 + 2*vk3 + vk4)
    vel_new = vel + (h/6) * (ak1 + 2*ak2 + 2*ak3 + ak4)
    return pos_new, vel_new


@njit(cache=True)
def a_cartesianas_njit(Sol):
    n = Sol.shape[0]
    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)
    for i in range(n):
        r     = Sol[i, 1]
        theta = Sol[i, 2]
        phi   = Sol[i, 3]
        x[i] = r * np.sin(theta) * np.cos(phi)
        y[i] = r * np.sin(theta) * np.sin(phi)
        z[i] = r * np.cos(theta)
    return x, y, z

# ── Integrador RK4 con G y F2 actualizados en cada paso ──────────────────────
def rk4(Y0, N, h, r_min, r_max):
    Sol = np.zeros((N, 8))
    Sol[0] = Y0
    pos = Y0[:4].copy()
    vel = Y0[4:].copy()

    for i in range(N - 1):
        r = pos[1]
        if r <= r_min + 0.1 or r > r_max or not np.isfinite(r):
            Sol[i+1:] = Sol[i]
            break

        G, F2 = get_G_F2(*pos)
        pos, vel = _rk4_step(pos, vel, G, F2, qm, h)
        Sol[i+1, :4] = pos
        Sol[i+1, 4:] = vel

    return Sol


def normalizar_vt(r_ini, theta_ini, vr0, vth0, vphi0, phi_ini, masa=1.0):
    """Calcula u^t para que g_{μν} u^μ u^ν = -masa²."""
    g = get_metric(0.0, r_ini, theta_ini, phi_ini)
    espacial = g[1,1]*vr0**2 + g[2,2]*vth0**2 + g[3,3]*vphi0**2
    cociente = (-masa**2 - espacial) / g[0,0]
    return np.sqrt(cociente) if cociente >= 0 else 1.0

# ── Warm-up de Numba ──────────────────────────────────────────────────────────
print("Compilando kernels Numba...")
_dG  = np.zeros((4, 4, 4))
_dF2 = np.zeros((4, 4))
_dv  = np.zeros(4)
_aceleracion(_dv, _dG, _dF2, 1.0)
_rk4_step(np.zeros(4), np.zeros(4), _dG, _dF2, 1.0, 0.1)
a_cartesianas_njit(np.zeros((2, 8)))
print("Listo.\n")

# ── Parámetros de integración ─────────────────────────────────────────────────
theta0  = np.pi / 2
vtheta0 = 0.0
N       = 3000
tfinal  = 100
h       = tfinal / N
r0      = 60.0
n_rayos = 80
y_rango = (-10, 10)
y_min   = 0.5

# ── Configuración de la cámara ────────────────────────────────────────────────
ELEVACION = 30
AZIMUT    = 60

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

    x, y, z = a_cartesianas_njit(Sol[mascara])
    ax.plot(x, y, z, color='lime', lw=0.7, alpha=0.8)

# ── Horizonte y fotósfera ─────────────────────────────────────────────────────
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