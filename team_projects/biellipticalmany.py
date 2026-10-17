
import numpy as np
import matplotlib.pyplot as plt



mu = 398600            # Earth's gravitational parameter (km^3/s^2)
earth_radius = 6371    # km

starting_alt = 400     # km
final_alt = 180000     # km
r_b = 300000           # Intermediate apogee radius (km)

r1 = earth_radius + starting_alt
r2 = earth_radius + final_alt

if not (r_b > r2 > r1 > earth_radius):
    raise ValueError(
        "For this outward transfer, require r_b > r2 > r1 > Earth radius."
    )



def vis_viva(r, a):
    return np.sqrt(mu * (2 / r - 1 / a))




a1 = (r1 + r_b) / 2
a2 = (r2 + r_b) / 2

v_initial = np.sqrt(mu / r1)
v_final = np.sqrt(mu / r2)

# Burn 1: Initial circular orbit to first ellipse
v_transfer1 = vis_viva(r1, a1)
dv1 = abs(v_transfer1 - v_initial)

# Burn 2: First ellipse to second ellipse at apogee
v_apogee1 = vis_viva(r_b, a1)
v_apogee2 = vis_viva(r_b, a2)
dv2 = abs(v_apogee2 - v_apogee1)

# Burn 3: Second ellipse to final circular orbit
v_transfer2 = vis_viva(r2, a2)
dv3 = abs(v_final - v_transfer2)

total_bielliptical_dv = dv1 + dv2 + dv3

# Time spent traveling along both half-ellipses
time1 = np.pi * np.sqrt(a1**3 / mu)
time2 = np.pi * np.sqrt(a2**3 / mu)

total_bielliptical_time = (time1 + time2) / 3600


# ==========================================
# HOHMANN TRANSFER
# ==========================================

a_h = (r1 + r2) / 2

v_h1 = vis_viva(r1, a_h)
v_h2 = vis_viva(r2, a_h)

dv_h1 = abs(v_h1 - v_initial)
dv_h2 = abs(v_final - v_h2)

total_hohmann_dv = dv_h1 + dv_h2

hohmann_time = (
    np.pi * np.sqrt(a_h**3 / mu)
) / 3600




print("\n========== BIELLIPTICAL TRANSFER ==========")

print(f"Burn 1: {dv1:.4f} km/s")
print(f"Burn 2: {dv2:.4f} km/s")
print(f"Burn 3: {dv3:.4f} km/s")

print(f"Total Delta-V: {total_bielliptical_dv:.4f} km/s")
print(f"Transfer Time: {total_bielliptical_time:.2f} hours")
print(f"Transfer Time: {total_bielliptical_time / 24:.2f} days")

print("\n========== HOHMANN TRANSFER ==========")

print(f"Burn 1: {dv_h1:.4f} km/s")
print(f"Burn 2: {dv_h2:.4f} km/s")

print(f"Total Delta-V: {total_hohmann_dv:.4f} km/s")
print(f"Transfer Time: {hohmann_time:.2f} hours")

print("\n========== COMPARISON ==========")

if total_bielliptical_dv < total_hohmann_dv:
    print("Bielliptical transfer requires less delta-v.")
elif total_hohmann_dv < total_bielliptical_dv:
    print("Hohmann transfer requires less delta-v.")
else:
    print("Both transfers require the same delta-v.")


# ==========================================
# GRAPH 1: DELTA-V COMPARISON
# ==========================================

transfer_types = ["Bielliptical", "Hohmann"]

delta_v_values = [
    total_bielliptical_dv,
    total_hohmann_dv
]

fig1, ax1 = plt.subplots(figsize=(8, 5))

bars = ax1.bar(
    transfer_types,
    delta_v_values,
    color=["royalblue", "orange"]
)

ax1.set_ylabel("Total Delta-V (km/s)")
ax1.set_title("Bielliptical vs Hohmann Transfer")

for bar in bars:
    height = bar.get_height()

    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.4f}",
        ha="center",
        va="bottom"
    )

ax1.set_ylim(0, max(delta_v_values) * 1.15)
ax1.grid(axis="y", alpha=0.25)
fig1.tight_layout()


# ==========================================
# GRAPH 2: ORBITAL VISUALIZATION
# ==========================================

# The first burn is at +X.
# The second burn is at -X (apogee).
# The third burn is at +X.

theta = np.linspace(0, 2 * np.pi, 1000)

# Initial circular orbit
x_initial = r1 * np.cos(theta)
y_initial = r1 * np.sin(theta)

# Final circular orbit
x_final = r2 * np.cos(theta)
y_final = r2 * np.sin(theta)

# Earth
x_earth = earth_radius * np.cos(theta)
y_earth = earth_radius * np.sin(theta)


# Function to calculate points on an elliptical orbit
# with its closest point at +X and apogee at -X.

def ellipse_points(r_perigee, r_apogee, angles):
    a = (r_perigee + r_apogee) / 2
    e = (r_apogee - r_perigee) / (r_apogee + r_perigee)

    r = a * (1 - e**2) / (1 + e * np.cos(angles))

    x = r * np.cos(angles)
    y = r * np.sin(angles)

    return x, y


# First transfer ellipse
theta_first = np.linspace(0, np.pi, 500)

x_ellipse1, y_ellipse1 = ellipse_points(
    r1, r_b, theta_first
)

# Second transfer ellipse
theta_second = np.linspace(np.pi, 2 * np.pi, 500)

x_ellipse2, y_ellipse2 = ellipse_points(
    r2, r_b, theta_second
)


fig2, ax2 = plt.subplots(figsize=(11, 8))

# Earth
ax2.fill(
    x_earth,
    y_earth,
    color="deepskyblue",
    label="Earth",
    zorder=5
)

# Initial orbit
ax2.plot(
    x_initial,
    y_initial,
    color="green",
    linestyle="--",
    label="Initial Circular Orbit"
)

# Final orbit
ax2.plot(
    x_final,
    y_final,
    color="purple",
    linestyle="--",
    label="Final Circular Orbit"
)

# First transfer ellipse
ax2.plot(
    x_ellipse1,
    y_ellipse1,
    color="red",
    linewidth=2.5,
    label="First Transfer Ellipse"
)

# Second transfer ellipse
ax2.plot(
    x_ellipse2,
    y_ellipse2,
    color="orange",
    linewidth=2.5,
    label="Second Transfer Ellipse"
)

# Burn locations
burn1 = (r1, 0)
burn2 = (-r_b, 0)
burn3 = (r2, 0)

ax2.scatter(
    [burn1[0], burn2[0], burn3[0]],
    [burn1[1], burn2[1], burn3[1]],
    color="black",
    s=65,
    zorder=10
)

ax2.annotate(
    "Burn 1",
    burn1,
    xytext=(10, 15),
    textcoords="offset points",
    fontsize=10
)

ax2.annotate(
    "Burn 2 (Apogee)",
    burn2,
    xytext=(-5, 15),
    textcoords="offset points",
    fontsize=10,
    ha="right"
)

ax2.annotate(
    "Burn 3",
    burn3,
    xytext=(10, 15),
    textcoords="offset points",
    fontsize=10
)

ax2.set_title("Bielliptical Orbital Transfer Around Earth")
ax2.set_xlabel("X Position (km)")
ax2.set_ylabel("Y Position (km)")

ax2.set_aspect("equal", adjustable="box")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="best")

fig2.tight_layout()

# Display both figures
plt.show()
