import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import plotly.graph_objects as go
from christoffel import christoffel
from christoffel import metric
from faraday import F2_tensor
import matplotlib.cm as cm

# ── Parámetros físicos ────────────────────────────────────────────
m   = 1.0
a   = 0.7
mu  = 0.21
q   = -0.9
qe  = 7.0e-4
r_min = 2 * m

es_foton = False

if es_foton:
    m1 = 0.0
    qm = 0.0
else:
    m1 = 1e-6
    qm = qe / m1

# ── Tensores ──────────────────────────────────────────────────────

@jit
def christoffel_jax(pos):
    raw = christoffel(pos[0], pos[1], pos[2], pos[3], mu, m, a, q, qe)
    return jnp.stack(raw)

@jit
def F2_jax(pos):
    return F2_tensor(pos[0], pos[1], pos[2], pos[3], mu, m, a, q, qe)

# ── Normalización ─────────────────────────────────────────────────

@jit
def norm_vt(pos, vel_espacial, m_particula):
    g = metric(pos[0], pos[1], pos[2], pos[3], mu, m, a, q, qe)

    gtt  = g[0, 0]
    grr  = g[1, 1]
    gth  = g[2, 2]
    gph  = g[3, 3]
    gtph = g[0, 3]

    vr, vth, vphi = vel_espacial

    A = gtt
    B = 2.0 * gtph * vphi
    C = grr * vr**2 + gth * vth**2 + gph * vphi**2 + m_particula**2

    discriminante = B**2 - 4.0 * A * C
    vt = (-B - jnp.sqrt(jnp.maximum(discriminante, 0.0))) / (2.0 * A)
    return vt

# ── Dinámica ──────────────────────────────────────────────────────

@jit
def aceleracion(vel, G, F2):
    lorentz  = jnp.einsum('ij,j->i', F2, vel) * qm
    geodesic = jnp.einsum('ijk,j,k->i', G, vel, vel)
    return lorentz - geodesic

@jit
def rk4_step(pos, vel, h):
    def deriv(p, v):
        return v, aceleracion(v, christoffel_jax(p), F2_jax(p))

    vk1, ak1 = deriv(pos,           vel)
    vk2, ak2 = deriv(pos + h/2*vk1, vel + h/2*ak1)
    vk3, ak3 = deriv(pos + h/2*vk2, vel + h/2*ak2)
    vk4, ak4 = deriv(pos + h*vk3,   vel + h*ak3)

    pos_new = pos + (h/6) * (vk1 + 2*vk2 + 2*vk3 + vk4)
    vel_new = vel + (h/6) * (ak1 + 2*ak2 + 2*ak3 + ak4)
    return pos_new, vel_new

# ── Integrador ────────────────────────────────────────────────────

def make_integrador(N, h):
    @jit
    def integrar_rayo(Y0):
        def paso(carry, _):
            pos, vel, activo = carry
            pos_new, vel_new = rk4_step(pos, vel, h)

            sigue_vivo = (
                activo &
                (pos_new[1] > r_min + 0.1) &
                jnp.isfinite(pos_new[1])
            )

            pos_f = jnp.where(sigue_vivo, pos_new, pos)
            vel_f = jnp.where(sigue_vivo, vel_new, vel)

            return (pos_f, vel_f, sigue_vivo), jnp.concatenate([pos_f, vel_f])

        _, traj = jax.lax.scan(
            paso, (Y0[:4], Y0[4:], jnp.array(True)), None, length=N
        )
        return traj

    return jit(vmap(integrar_rayo))

# ── Configuración de rayos ────────────────────────────────────────
#
#  Todos los rayos parten del mismo plano x = -x0
#  distribuidos en una grilla (y, z).
#  La dirección de propagación es +x (vx=1, vy=vz=0 en cartesianas).
#
N      = 12000
tfinal = 800
h      = tfinal / N
x0     = 40.0          # plano de inicio (x = -x0)

n_y    = 14            # puntos en y
n_z    = 14            # puntos en z
rango  = 20.0          # mitad del lado de la grilla

y_vals = np.linspace(-rango, rango, n_y)
z_vals = np.linspace(-rango, rango, n_z)

Y0_list      = []   # condiciones iniciales para el integrador
origen_list  = []   # (y0, z0) de cada rayo, para colorear

for y0 in y_vals:
    for z0 in z_vals:

        # ── Posición cartesiana inicial ──────────────────────────
        xi = -x0
        yi =  y0
        zi =  z0

        # ── Coordenadas esféricas ────────────────────────────────
        r_ini     = np.sqrt(xi**2 + yi**2 + zi**2)
        theta_ini = np.arccos(np.clip(zi / r_ini, -1.0, 1.0))
        phi_ini   = np.arctan2(yi, xi)

        # ── Velocidad cartesiana del haz: dirección +x ───────────
        vx, vy, vz = 1.0, 0.0, 0.0

        # ── Jacobiano cartesiano → esférico ─────────────────────
        sin_th = np.sin(theta_ini)
        cos_th = np.cos(theta_ini)
        sin_ph = np.sin(phi_ini)
        cos_ph = np.cos(phi_ini)

        vr   = sin_th*cos_ph*vx + sin_th*sin_ph*vy + cos_th*vz
        vth  = (cos_th*cos_ph*vx + cos_th*sin_ph*vy - sin_th*vz) / r_ini
        vphi = (-sin_ph*vx + cos_ph*vy) / (r_ini * np.abs(sin_th) + 1e-10)

        pos_ini = jnp.array([0.0, r_ini, theta_ini, phi_ini])
        vel_esp = jnp.array([vr, vth, vphi])

        vt0 = norm_vt(pos_ini, vel_esp, m1)

        Y0_list.append([0.0, r_ini, theta_ini, phi_ini, float(vt0), vr, vth, vphi])
        origen_list.append((y0, z0))

Y0_batch = jnp.array(Y0_list)

# ── Integración ───────────────────────────────────────────────────

integrar_todos = make_integrador(N, h)
print(f"Integrando {len(Y0_list)} rayos...")
trayectorias = np.array(integrar_todos(Y0_batch))
print(f"Trayectorias: {trayectorias.shape}")

# ── Plot ──────────────────────────────────────────────────────────

ELEVACION = 25
AZIMUT    = 40

fig = go.Figure()

# Colormap HSV por ángulo de origen en el plano (y,z)
for i, traj in enumerate(trayectorias):
    r     = traj[:, 1]
    theta = traj[:, 2]
    phi   = traj[:, 3]

    mascara = (r > r_min + 0.3) & np.isfinite(r) & (r < 3 * x0)
    if mascara.sum() < 2:
        continue

    x = r[mascara] * np.sin(theta[mascara]) * np.cos(phi[mascara])
    y = r[mascara] * np.sin(theta[mascara]) * np.sin(phi[mascara])
    z = r[mascara] * np.cos(theta[mascara])

    # Color por ángulo azimutal en el plano (y0,z0) de origen
    y0, z0 = origen_list[i]
    ang = np.arctan2(z0, y0)                    # -π a π
    t   = (ang + np.pi) / (2 * np.pi)           # 0 a 1
    rc, gc, bc = cm.hsv(t)[:3]
    color = f'rgb({int(255*rc)},{int(255*gc)},{int(255*bc)})'

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=color, width=1),
        opacity=0.75,
        showlegend=False
    ))

# ── Horizonte de eventos y fotósfera ─────────────────────────────

u_s = np.linspace(0, 2*np.pi, 60)
v_s = np.linspace(0, np.pi,   60)

for radio, color, opac, nombre in [
    (r_min,        'black',  1.0,  'Horizonte de eventos'),
    (1.5 * r_min,  'orange', 0.12, 'Fotósfera'),
]:
    xs = radio * np.outer(np.cos(u_s), np.sin(v_s))
    ys = radio * np.outer(np.sin(u_s), np.sin(v_s))
    zs = radio * np.outer(np.ones_like(u_s), np.cos(v_s))

    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0, color], [1, color]],
        opacity=opac,
        showscale=False,
        name=nombre,
        showlegend=True
    ))

# ── Plano de inicio de los rayos (referencia visual) ─────────────

yy, zz = np.meshgrid(
    np.linspace(-rango, rango, 2),
    np.linspace(-rango, rango, 2)
)
xx = np.full_like(yy, -x0)

fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    colorscale=[[0, 'white'], [1, 'white']],
    opacity=0.05,
    showscale=False,
    name='Plano inicial',
    showlegend=True
))

# ── Cámara y layout ───────────────────────────────────────────────

eye_x = np.cos(np.deg2rad(AZIMUT)) * np.cos(np.deg2rad(ELEVACION))
eye_y = np.sin(np.deg2rad(AZIMUT)) * np.cos(np.deg2rad(ELEVACION))
eye_z = np.sin(np.deg2rad(ELEVACION))

lim = 1.2 * x0

fig.update_layout(
    title=dict(
        text=(
            f"Ray tracing 3D  |  {'Fotón' if es_foton else 'Partícula masiva cargada'}<br>"
            f"Haz paralelo en x={-x0:.0f}  |  grilla {n_y}×{n_z}  |  "
            f"m={m}  a={a}  μ={mu}  q={q}  qe={qe}"
        ),
        font=dict(color='white', size=13),
        x=0.5
    ),
    scene=dict(
        xaxis=dict(range=[-lim, lim], title='x', color='white',
                   gridcolor='rgba(255,255,255,0.1)',
                   zerolinecolor='rgba(255,255,255,0.25)'),
        yaxis=dict(range=[-lim, lim], title='y', color='white',
                   gridcolor='rgba(255,255,255,0.1)',
                   zerolinecolor='rgba(255,255,255,0.25)'),
        zaxis=dict(range=[-lim, lim], title='z', color='white',
                   gridcolor='rgba(255,255,255,0.1)',
                   zerolinecolor='rgba(255,255,255,0.25)'),
        bgcolor='#050508',
        aspectmode='cube',
        camera=dict(eye=dict(x=eye_x, y=eye_y, z=eye_z))
    ),
    paper_bgcolor='#050508',
    font=dict(color='white'),
    legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)'),
    margin=dict(l=0, r=0, t=70, b=0)
)

# ── Exportar ──────────────────────────────────────────────────────

fig.write_html("raytracing_3d.html")
print("Guardado: raytracing_3d.html")

try:
    fig.write_image("raytracing_3d.png", width=1200, height=1000, scale=2)
    print("Guardado: raytracing_3d.png")
except Exception:
    print("Para PNG: pip install kaleido")

fig.show()