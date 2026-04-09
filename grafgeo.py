import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import plotly.graph_objects as go
from christoffel import christoffel
from christoffel import metric
from faraday import F2_tensor

# ── Parámetros ────────────────────────────────────────────────────
m   = 1.0
a   = 0.7
mu  = 1.0e-1
q   = 1.0e-2
r_min = 2*m

foton = False #True para el caso de fotones, False para partículas masivas

if foton:
    m1 = 0.0
    qm = 0.0
    qe = 0.0
else:
    m1 = 1e-6
    qe  = 5.0e-4
    qm = qe / m1

@jit
def christoffel_jax(pos):
    raw = christoffel(pos[0], pos[1], pos[2], pos[3], mu, m, a, q, qe)
    return jnp.stack(raw)

@jit
def F2_jax(pos):
    return F2_tensor(pos[0], pos[1], pos[2], pos[3], mu, m, a, q, qe)

# ── Normalización ──────────────────────────────────────────────────

@jit
def norm_vt(pos, vel_espacial, m_particula):
    r, th, phi = pos[1], pos[2], pos[3]
    vr, vth, vphi = vel_espacial

    g = metric(pos[0], r, th, phi, mu, m, a, q, qe)

    gtt  = g[0, 0]
    grr  = g[1, 1]
    gth  = g[2, 2]
    gph  = g[3, 3]
    gtph = g[0, 3]

    A = gtt
    B = 2.0 * gtph * vphi
    C = grr * vr**2 + gth * vth**2 + gph * vphi**2 + m_particula**2

    discriminante = B**2 - 4.0 * A * C
    vt = (-B - jnp.sqrt(discriminante)) / (2.0 * A)

    return vt

# ── Aceleración y paso RK4 ───────────────────────────────────────────────────

@jit
def aceleracion(vel, G, F2):
    lorentz  = jnp.einsum('ij,j->i', F2, vel) * qm
    geodesic = jnp.einsum('ijk,j,k->i', G, vel, vel)
    return lorentz - geodesic

@jit
def rk4_step(pos, vel, h):
    def deriv(p, v):
        return v, aceleracion(v, christoffel_jax(p), F2_jax(p))

    vk1, ak1 = deriv(pos,            vel)
    vk2, ak2 = deriv(pos + h/2*vk1,  vel + h/2*ak1)
    vk3, ak3 = deriv(pos + h/2*vk2,  vel + h/2*ak2)
    vk4, ak4 = deriv(pos + h*vk3,    vel + h*ak3)

    pos_new = pos + (h/6) * (vk1 + 2*vk2 + 2*vk3 + vk4)
    vel_new = vel + (h/6) * (ak1 + 2*ak2 + 2*ak3 + ak4)
    return pos_new, vel_new

# ── Integrador ──────────────────────────────────────────────────────

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

            pos_final = jnp.where(sigue_vivo, pos_new, pos)
            vel_final = jnp.where(sigue_vivo, vel_new, vel)

            return (pos_final, vel_final, sigue_vivo), jnp.concatenate([pos_final, vel_final])

        activo_ini = jnp.array(True)
        _, traj = jax.lax.scan(paso, (Y0[:4], Y0[4:], activo_ini), None, length=N)
        return traj

    return jit(vmap(integrar_rayo))

# ── Configuración de Rayos ───────────────────────────────────────────────────

theta0  = np.deg2rad(90)
N       = 12000
tfinal  = 800
h       = tfinal / N
r0      = 40.0
z0      = 0.0
n_rayos = 100
y_rango = (-20, 20)


Y0_list  = []
y_values = np.linspace(y_rango[0], y_rango[1], n_rayos)

for y0 in y_values:
    r_ini     = np.sqrt(r0**2 + y0**2 + z0**2)  
    theta_ini = np.arccos(np.clip(z0/ r_ini, -1.0, 1.0))  
    phi_ini   = np.pi - np.arctan2(y0, r0)

    sin_th = np.sin(theta_ini)
    cos_th = np.cos(theta_ini)
    sin_ph = np.sin(phi_ini)
    cos_ph = np.cos(phi_ini)

    vr0   =  sin_th*cos_ph
    vth0  = 0  
    vphi0 = -sin_ph / (r_ini * np.abs(sin_th) + 1e-10)

    pos_ini = jnp.array([0.0, r_ini, theta_ini, phi_ini])
    vel_esp = jnp.array([vr0, vth0, vphi0])

    vt0 = norm_vt(pos_ini, vel_esp, m1)
    Y0_list.append([0.0, r_ini, theta_ini, phi_ini, float(vt0), vr0, vth0, vphi0])

Y0_batch = jnp.array(Y0_list)

# ── Ejecución ────────────────────────────────────────────────────────────────

integrar_todos = make_integrador(N, h)
print("Calculando trayectorias...")
trayectorias = np.array(integrar_todos(Y0_batch))
print(f"Trayectorias calculadas: {trayectorias.shape}")

# ── Plot con Plotly ───────────────────────────────────────────────────────────

ELEVACION = 10
AZIMUT    = 30

fig = go.Figure()

# Rayos de luz
for traj in trayectorias:
    r     = traj[:, 1]
    theta = traj[:, 2]
    phi   = traj[:, 3]

    mascara = (r > 1e-1) & np.isfinite(r)
    if mascara.sum() < 2:
        continue

    x = r[mascara] * np.sin(theta[mascara]) * np.cos(phi[mascara])
    y = r[mascara] * np.sin(theta[mascara]) * np.sin(phi[mascara])
    z = r[mascara] * np.cos(theta[mascara])

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='lime', width=1),
        opacity=0.8,
        showlegend=False
    ))

# Esferas: horizonte de eventos y fotósfera
if r_min > 0.1:
    u_s = np.linspace(0, 2*np.pi, 60)
    v_s = np.linspace(0, np.pi,   60)

    for radio, color, opacity, nombre in [
        (r_min,       'black',  1.0,  'Horizonte de eventos'),
        (1.5 * r_min, 'orange', 0.15, 'Fotósfera'),
    ]:
        xs = radio * np.outer(np.cos(u_s), np.sin(v_s))
        ys = radio * np.outer(np.sin(u_s), np.sin(v_s))
        zs = radio * np.outer(np.ones_like(u_s), np.cos(v_s))

        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            colorscale=[[0, color], [1, color]],
            opacity=opacity,
            showscale=False,
            name=nombre,
            showlegend=True
        ))

#Para visualización
eye_x = np.cos(np.deg2rad(AZIMUT)) * np.cos(np.deg2rad(ELEVACION))
eye_y = np.sin(np.deg2rad(AZIMUT)) * np.cos(np.deg2rad(ELEVACION))
eye_z = np.sin(np.deg2rad(ELEVACION))

lim = r0

fig.update_layout(
    title=dict(
        text=(
            f"Ray tracing 3D  |  {'Fotón' if foton else 'Partícula masiva cargada'}<br>"
            f"Haz paralelo en x={r0:.0f}  |  {n_rayos} rayos  |  z={z0}  |  "
            f"m={m} mpart={m1}  a={a}  μ={mu}  q={q}  qe={qe}"
        ),
        font=dict(color='white', size=14),
        x=0.5
    ),
    scene=dict(
        xaxis=dict(range=[-lim, lim], title='x', color='white',
                   gridcolor='rgba(255,255,255,0.15)', zerolinecolor='rgba(255,255,255,0.3)'),
        yaxis=dict(range=[-lim, lim], title='y', color='white',
                   gridcolor='rgba(255,255,255,0.15)', zerolinecolor='rgba(255,255,255,0.3)'),
        zaxis=dict(range=[-lim, lim], title='z', color='white',
                   gridcolor='rgba(255,255,255,0.15)', zerolinecolor='rgba(255,255,255,0.3)'),
        bgcolor='#0a0a0a',
        camera=dict(eye=dict(x=eye_x, y=eye_y, z=eye_z))
    ),
    paper_bgcolor='#0a0a0a',
    font=dict(color='white'),
    legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)'),
    margin=dict(l=0, r=0, t=60, b=0)
)

# ── Html interactivo ──────────────────────────────────────────────────────────────────

fig.write_html("raytracing.html")
print("Guardado: raytracing.html")

try:
    fig.write_image("raytracing.png", width=900, height=900, scale=2)
    print("Guardado: raytracing.png")
except Exception:
    print("Para exportar PNG instalar kaleido")

fig.show()