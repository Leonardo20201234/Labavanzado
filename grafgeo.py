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
a   = 0.6
mu  = 0.0
q   = 0.0
r_min = 2*m

foton = False #True para el caso de fotones, False para partículas masivas

if foton:
    m1 = 0.0
    qm = 0.0
    qe = 0.0
else:
    m1 = 1e-6
    qe  = 7.0e-4
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
z0      = 0.5
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
    vth0  = 0.00023
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

ELEVACION = 0
AZIMUT    = 0

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
        line=dict(color='lime', width=1.5),   # línea un poco más gruesa
        opacity=0.9,
        showlegend=False
    ))

# Esferas: horizonte de eventos y fotósfera
if r_min > 0.1:
    u_s = np.linspace(0, 2*np.pi, 120)   # más puntos = esferas más suaves
    v_s = np.linspace(0, np.pi,   120)

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
            showlegend=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4, roughness=0.5),
        ))

eye_x = np.cos(np.deg2rad(AZIMUT)) * np.cos(np.deg2rad(ELEVACION))
eye_y = np.sin(np.deg2rad(AZIMUT)) * np.cos(np.deg2rad(ELEVACION))
eye_z = np.sin(np.deg2rad(ELEVACION))

lim = r0

# ── Configuración de ejes (reutilizable) ─────────────────────────────────────

def eje(titulo):
    return dict(
        title=dict(text=titulo, font=dict(size=14, color='white')),
        range=[-lim, lim],
        tickfont=dict(size=11, color='white'),
        tickcolor='white',
        ticklen=6,
        showgrid=True,
        gridcolor='rgba(255,255,255,0.20)',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.45)',
        zerolinewidth=1.5,
        showline=True,               # línea del eje siempre visible
        linecolor='white',
        linewidth=2,
        showbackground=True,         # plano de fondo del eje
        backgroundcolor='rgba(255,255,255,0.04)',
        showticklabels=True,
        nticks=9,
        mirror=True,
    )

fig.update_layout(
    title=dict(
        text=(
            f"Ray tracing 3D  |  {'Fotón' if foton else 'Partícula masiva cargada'}<br>"
            f"vtheta0={vth0:.5f}  |  {n_rayos} rayos  |  z={z0}  |  "
            f"m={m}  mpart={m1}  a={a}  μ={mu}  q={q}  qe={qe}"
        ),
        font=dict(color='white', size=15),
        x=0.5,
        y=0.97,
    ),
    scene=dict(
        xaxis=eje('x'),
        yaxis=eje('y'),
        zaxis=eje('z'),
        bgcolor='#0a0a0a',
        camera=dict(
            eye=dict(x=eye_x, y=eye_y, z=eye_z),
            projection=dict(type='perspective'),  # perspectiva más natural
        ),
        aspectmode='cube',       # ejes con la misma escala visual siempre
        dragmode='orbit',
    ),
    paper_bgcolor='#0a0a0a',
    font=dict(color='white', family='Arial'),
    legend=dict(
        font=dict(color='white', size=12),
        bgcolor='rgba(20,20,20,0.75)',
        bordercolor='rgba(255,255,255,0.3)',
        borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=70, b=0),
    width=1200,    # resolución del canvas HTML
    height=900,
)

# ── Html interactivo ──────────────────────────────────────────────────────────

fig.write_html("raytracing.html")
print("Guardado: raytracing.html")

try:
    fig.write_image("raytracing.png", width=9000, height=7000, scale=5)  
    print("Guardado: raytracing.png")
except Exception:
    print("Para exportar PNG instalar kaleido")

fig.show()