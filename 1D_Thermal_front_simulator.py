import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


st.set_page_config(page_title="Thermal Front Simulator", layout="wide")

st.title("1D Thermal Front Simulator")
st.markdown("Interactive convection–conduction thermal front visualization with stability analysis.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Reservoir Parameters")

Tinj = st.sidebar.number_input("Injection Temperature [°C]", value=15.0)
Tinit = st.sidebar.number_input("Initial Reservoir Temperature [°C]", value=100.0)

modelExtend = st.sidebar.number_input("Reservoir Length [m]", value=100.0)
phi = st.sidebar.slider("Porosity", 0.01, 0.5, 0.25)

k = st.sidebar.number_input("Permeability [m²]", value=1e-13, format="%.2e")
mu = st.sidebar.number_input("Viscosity [Pa.s]", value=1e-3, format="%.2e")

deltap = st.sidebar.number_input("Pressure Difference [Pa]", value=1e6, format="%.2e")

# Rock properties
Cr = st.sidebar.number_input("Rock Heat Capacity [J/kg.K]", value=920.0)
rhoR = st.sidebar.number_input("Rock Density [kg/m³]", value=2700.0)
lambdaR = st.sidebar.number_input("Rock Thermal Conductivity [W/m.K]", value=2.7)

# Fluid properties
Cf = st.sidebar.number_input("Fluid Heat Capacity [J/kg.K]", value=4187.0)
rhoF = st.sidebar.number_input("Fluid Density [kg/m³]", value=1000.0)
lambdaF = st.sidebar.number_input("Fluid Thermal Conductivity [W/m.K]", value=0.68)

# Numerical controls
st.sidebar.header("Numerical Controls")

nx = st.sidebar.number_input("Number of Cells", value=100)
total_days = st.sidebar.number_input("Simulation Time [days]", value=100.0)
dt_days = st.sidebar.number_input("Time Step [days]", value=0.01)

physics = st.sidebar.radio(
    "Physics",
    [
        "Convection",
        "Convection + Conduction"
    ]
)

method = st.sidebar.selectbox(
    "Method",
    [
        "Analytical",
        "Explicit",
        "Implicit"
    ]
)

show_analytical = st.sidebar.checkbox("Compare with Analytical Solution", value=True)

# -----------------------------
# Derived Quantities
# -----------------------------
day = 24 * 60 * 60

t_total = total_days * day
dt = dt_days * day

dx = modelExtend / nx

x = np.linspace(0, modelExtend, nx + 1)

c_rho = phi * Cf * rhoF + (1 - phi) * Cr * rhoR

q = (k / mu) * (deltap / modelExtend)
v = q / phi

lambdaA = phi * lambdaF + (1 - phi) * lambdaR

alpha = lambdaA / c_rho

conv = (rhoF * Cf / c_rho) * v * dt / dx
D = alpha * dt / (dx ** 2)


# -----------------------------
# Stability
# -----------------------------
st.subheader("Stability Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Courant Number (C)", f"{conv:.4f}")

with col2:
    st.metric("Diffusion Number (D)", f"{D:.4f}")

criterion = conv + 2 * D

with col3:
    st.metric("C + 2D", f"{criterion:.4f}")

# -----------------------------
# Explicit Stability
# -----------------------------
if method == "Explicit":

    # Pure convection
    if physics == "Convection":

        st.markdown("### Explicit Convection Criterion")

        if conv <= 1:
            st.success("Stable Explicit Convection Scheme")
        else:
            st.error("Unstable Explicit Convection Scheme")

    # Convection + conduction
    else:

        st.markdown("### Explicit Convection–Conduction Criterion")

        if criterion <= 1:
            st.success("Stable Explicit Convection–Conduction Scheme")
        else:
            st.error("Unstable Explicit Convection–Conduction Scheme")

# -----------------------------
# Implicit Stability
# -----------------------------
else:

    st.markdown("### Implicit Scheme")

    st.info(
        "Implicit schemes are generally unconditionally stable."
    )


# -----------------------------
# Analytical Solutions
# -----------------------------
def analytical_convection(x, t):
    res = (Cf * rhoF / c_rho) * v * t
    T = np.ones_like(x) * Tinj
    T[x > res] = Tinit
    return T


def analytical_conv_diff(x, t):

    res = (Cf * rhoF / c_rho) * v * t

    efc1 = (x - res) / np.sqrt(alpha * t) 
    efc2 = (x + res) / np.sqrt(alpha * t) 

    T_x = 1 - 0.5 * (
        erfc(efc1)
        + erfc(efc2)
        * np.exp((Cf * rhoF * v * x) / lambdaA)
    )

    return (Tinit - Tinj) * T_x + Tinj
# -----------------------------
# Numerical Solvers
# -----------------------------
def explicit_solver():
    nt = int(t_total / dt)

    T = np.ones(nx + 1) * Tinit
    T[0] = Tinj

    history = []

    for n in range(nt):

        T_new = T.copy()

        for i in range(1, nx):

            conduction = D * (T[i + 1] - 2 * T[i] + T[i - 1])
            convection_term = conv * (T[i] - T[i - 1])

            if physics == "Convection":
                T_new[i] = T[i] - convection_term

            elif physics == "Conduction":
                T_new[i] = T[i] + conduction

            else:
                T_new[i] = T[i] + conduction - convection_term

        T_new[0] = Tinj
        T_new[-1] = T_new[-2]

        T = T_new.copy()

        history.append(T.copy())

    return np.array(history)

def implicit_solver():
    nt = int(t_total / dt)

    T = np.ones(nx + 1) * Tinit
    T[0] = Tinj

    history = []

    A = np.zeros((nx + 1, nx + 1))

    for i in range(1, nx):

        if physics == "Convection":
            A[i, i - 1] = -conv
            A[i, i] = 1 + conv

        else:
            A[i, i - 1] = -(D + conv)
            A[i, i] = 1 + 2 * D + conv
            A[i, i + 1] = -D

    A[0, 0] = 1
    A[-1, -1] = 1

    b = T.copy()

    for _ in range(nt):
        b[0] = Tinj
        b[-1] = b[-2]

        T = np.linalg.solve(A, b)

        history.append(T.copy())

        b = T.copy()

    return np.array(history)

# -----------------------------
# Solve
# -----------------------------
if method == "Explicit":
    history = explicit_solver()

elif method == "Implicit":
    history = implicit_solver()

else:
    nt = int(t_total / dt)

    history = []

    for n in range(nt):

        current_t = (n + 1) * dt

        if physics == "Convection":
            T = analytical_convection(x, current_t)

        elif physics == "Convection + Conduction":
            T = analytical_conv_diff(x, current_t)

        history.append(T)

    history = np.array(history)

# -----------------------------
# Time Selection
# -----------------------------
st.subheader("Thermal Front Visualization")

frame = st.slider(
    "Time Step",
    0,
    len(history) - 1,
    len(history) - 1
)

current_time_days = (frame + 1) * dt / day

st.write(f"Simulation Time = {current_time_days:.2f} days")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, history[frame], linewidth=2, label=f"{method}")

if show_analytical and method != "Analytical":

    if physics == "Convection":
        analytical = analytical_convection(x, current_time_days * day)

    else:
        analytical = analytical_conv_diff(x, current_time_days * day)

    ax.plot(
        x,
        analytical,
        "--",
        linewidth=2,
        label="Analytical"
    )

ax.set_xlabel("Distance [m]")
ax.set_ylabel("Temperature [°C]")
ax.set_title("Thermal Front Propagation")
ax.grid(True)
ax.legend()

st.pyplot(fig)

