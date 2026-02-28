import sympy as sp
from christoffel import G, metric
from faraday import F2

# ── Parámetros ────────────────────────────────────────────────────────────────
m     = 1.0
a     = 0.5
mu    = 1
q     = 0.1
qe    = 0.1
r_min = 2*m

m_sym  = sp.Symbol('m',  real=True)
a_sym  = sp.Symbol('a',  real=True)
mu_sym = sp.Symbol('mu', real=True)
q_sym  = sp.Symbol('q',  real=True)

sym_params = {m_sym:  m,
              a_sym:  a,
              mu_sym: mu,
              q_sym:  q}

G      = [Gi.subs(sym_params) for Gi in G]
metric = metric.subs(sym_params)
F2     = F2.subs(sym_params)

print(F2)