import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from christoffel import G, metric
from faraday import F2

# ── Símbolos genéricos ────────────────────────────────────────────────────────
q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3')
u0, u1, u2, u3 = sp.symbols('u0 u1 u2 u3')
X  = sp.Matrix([q0, q1, q2, q3])
VX = sp.Matrix([u0, u1, u2, u3])

u = [u0, u1, u2, u3]

# ── Parámetros ────────────────────────────────────────────────────────────────
m     = 2.5 #Param de metricas: frutos, cuad_frutos, swchwarzschild
a     = 0.5 #Param de frutos, cuad_frutos
mu    = 2.0 #Param de cuad_dipolo, cuad_frutos
q     = 0.1 #Param de frutos, cuad_frutos
qe    = 0.6 #Carga Electrica, 
            #Param cuad_frutos, 
qm = qe/m   #Para las geodesicas no nulas
r_min = 2*m

m_sym  = sp.Symbol('m',  real=True)
a_sym  = sp.Symbol('a',  real=True)
mu_sym = sp.Symbol('mu', real=True)
q_sym  = sp.Symbol('q',  real=True)
qe_sym  = sp.Symbol('qe',  real=True)

sym_params = {m_sym:  m,
              a_sym:  a,
              mu_sym: mu,
              q_sym:  q,
              qe_sym:  qe}

G      = [Gi.subs(sym_params) for Gi in G]
metric = metric.subs(sym_params)
F2     = F2.subs(sym_params)

# ── Fuerza total (geodésica + EM) ─────────────────────────────────────────────
Fgeo = sp.zeros(4, 1)

for i in range(4):
    for j in range(4):
        Fgeo[i] += qm * F2[i, j] * u[j]
        for k in range(4):
            Fgeo[i] += -G[i][j, k] * u[j] * u[k]

args     = (q0, q1, q2, q3, u0, u1, u2, u3)
Fgeo_num = sp.lambdify(args, Fgeo, 'numpy')
g_num    = sp.lambdify((q0, q1, q2, q3), metric, 'numpy')

def fuerza_total(Xval, VXval):
    return np.array(Fgeo_num(*Xval, *VXval), dtype=float).flatten()

def sistema(tprop, Y):
    return np.concatenate((Y[4:], fuerza_total(Y[:4], Y[4:])))

# ── Normalización de la 4-velocidad (partícula masiva) ───────────────────────
def vt_masiva(r_ini, theta_ini, vr0, vth0, vphi0, phi_ini, masa=1.0):
    """g_{μν} u^μ u^ν = -masa²"""
    g_eval = np.array(g_num(0.0, r_ini, theta_ini, phi_ini), dtype=float)
    gtt    = g_eval[0, 0]
    grr    = g_eval[1, 1]
    gthth  = g_eval[2, 2]
    gpp    = g_eval[3, 3]
    espacial = grr*vr0**2 + gthth*vth0**2 + gpp*vphi0**2
    rhs = -masa**2 - espacial
    if rhs / gtt < 0:
        return 1.0
    return np.sqrt(rhs / gtt)

def rk4(Y0, N, h, r_min, r_max):
    Sol = np.zeros((N, 8))
    Sol[0] = Y0
    for i in range(N - 1):
        r_actual = Sol[i, 1]
        if r_actual <= r_min + 0.1 or r_actual <= 0 \
                or np.isnan(r_actual) or np.isinf(r_actual) or r_actual > r_max:
            Sol[i+1:] = Sol[i]; break
        k1 = sistema(i*h,       Sol[i])
        k2 = sistema(i*h + h/2, Sol[i] + h*k1/2)
        k3 = sistema(i*h + h/2, Sol[i] + h*k2/2)
        k4 = sistema(i*h + h,   Sol[i] + h*k3)
        Sol[i+1] = Sol[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return Sol

def conversion_cartesianas(Sol):
    r     = Sol[:, 1]
    theta = Sol[:, 2]
    phi   = Sol[:, 3]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# ── Parámetros de integración ─────────────────────────────────────────────────
theta0   = np.pi / 2
vtheta0  = 0.0
N        = 3000
tfinal   = 200
h        = tfinal / N
r0       = 60.0
n_rayos  = 100
y_rango  = (-25, 25)
y_minimo = 0.5

# ── Cámara ────────────────────────────────────────────────────────────────────
ELEVACION = 90
AZIMUT    = 270

# ── Gráfica 3D ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 9))
ax  = fig.add_subplot(111, projection='3d')
ax.view_init(elev=ELEVACION, azim=AZIMUT)

ys = np.linspace(y_rango[0], y_rango[1], n_rayos)

for y0 in ys:
    if abs(y0) < y_minimo:
        continue

    r_ini   = np.sqrt(r0**2 + y0**2)
    phi_ini = np.pi - np.arctan2(y0, r0)
    vr0     =  np.cos(phi_ini)
    vphi0   = -np.sin(phi_ini) / r_ini

    vt0 = vt_masiva(r_ini, theta0, vr0, vtheta0, vphi0, phi_ini)

    Y0  = np.array([0.0, r_ini, theta0, phi_ini, vt0, vr0, vtheta0, vphi0])
    Sol = rk4(Y0, N, h, r_min, 3*r0)

    mascara  = (Sol[:, 1] > r_min + 0.1) & np.isfinite(Sol[:, 1])
    Sol_plot = Sol[mascara]
    if len(Sol_plot) < 2:
        continue

    x_arr, y_arr, z_arr = conversion_cartesianas(Sol_plot)
    ax.plot(x_arr, y_arr, z_arr, color='lime', lw=0.7, alpha=0.8)

# ── Horizonte y fotósfera ─────────────────────────────────────────────────────
u_sph = np.linspace(0, 2*np.pi, 60)
v_sph = np.linspace(0, np.pi,   40)
if r_min > 0.1:
    xs  = r_min * np.outer(np.cos(u_sph), np.sin(v_sph))
    ys_ = r_min * np.outer(np.sin(u_sph), np.sin(v_sph))
    zs  = r_min * np.outer(np.ones_like(u_sph), np.cos(v_sph))
    ax.plot_surface(xs, ys_, zs, color='black', alpha=1.0, zorder=5)

    rf = 1.5 * r_min
    xf = rf * np.outer(np.cos(u_sph), np.sin(v_sph))
    yf = rf * np.outer(np.sin(u_sph), np.sin(v_sph))
    zf = rf * np.outer(np.ones_like(u_sph), np.cos(v_sph))
    ax.plot_surface(xf, yf, zf, color='orange', alpha=0.08, zorder=4)

# ── Estética ──────────────────────────────────────────────────────────────────
lim = r0 * 0.5
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_facecolor('#0a0a0a')
fig.patch.set_facecolor('#0a0a0a')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title(f"Ray tracing geodésico + EM  |  Partícula masiva cargada\n"
             f"elev={ELEVACION}°  azim={AZIMUT}°  θ₀={np.degrees(theta0):.1f}°  vθ₀={vtheta0}",
             color='white')
ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.3)
plt.tight_layout()
plt.savefig("raytracing.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()