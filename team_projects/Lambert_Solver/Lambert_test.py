import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from Lambert import solve_lambert


# --------------------------------------------------
# 1. Define the transfer
# --------------------------------------------------

mu = 398600.4418  # Earth gravitational parameter [km^3/s^2]

R1 = np.array([5000.0, 10000.0, 2100.0])    # Initial position [km]
R2 = np.array([-14600.0, 2500.0, 7000.0])   # Final position [km]

tof = 3600.0  # Time of flight [s]

# Solve Lambert's problem
V1, V2 = solve_lambert(R1, R2, tof, mu)

print("Initial velocity:", V1, "km/s")
print("Final velocity:  ", V2, "km/s")


# --------------------------------------------------
# 2. Propagate the spacecraft using two-body dynamics
# --------------------------------------------------

def two_body_dynamics(t, state):
    r = state[:3]
    v = state[3:]

    radius = np.linalg.norm(r)
    acceleration = -mu * r / radius**3

    return np.concatenate((v, acceleration))


n_frames = 300
times = np.linspace(0.0, tof, n_frames)

initial_state = np.concatenate((R1, V1))

solution = solve_ivp(
    two_body_dynamics,
    (0.0, tof),
    initial_state,
    t_eval=times,
    rtol=1e-10,
    atol=1e-11
)

if not solution.success:
    raise RuntimeError(solution.message)

positions = solution.y[:3].T  # Shape: (n_frames, 3)

# Check how closely the propagated spacecraft reaches R2
arrival_error = np.linalg.norm(positions[-1] - R2)
print(f"Arrival position error: {arrival_error:.6f} km")


# --------------------------------------------------
# 3. Set up the 3D figure
# --------------------------------------------------

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

fig.suptitle("Lambert Transfer Orbit", fontsize=15)

ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")

# Draw Earth as a sphere
earth_radius = 6378.137

u = np.linspace(0, 2 * np.pi, 48)
v = np.linspace(0, np.pi, 24)

earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
earth_z = earth_radius * np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(
    earth_x, earth_y, earth_z,
    color="royalblue",
    alpha=0.65,
    linewidth=0
)

# Draw the complete transfer path as a faint reference
ax.plot(
    positions[:, 0],
    positions[:, 1],
    positions[:, 2],
    "--",
    color="gray",
    alpha=0.45,
    label="Full transfer trajectory"
)

# Initial and final positions
ax.scatter(*R1, color="limegreen", s=70, label="Departure")
ax.scatter(*R2, color="red", s=70, label="Arrival")

# Initial and final position vectors
ax.plot(
    [0, R1[0]], [0, R1[1]], [0, R1[2]],
    color="limegreen", alpha=0.7
)
ax.plot(
    [0, R2[0]], [0, R2[1]], [0, R2[2]],
    color="red", alpha=0.7
)

# Animated spacecraft trail
trail, = ax.plot([], [], [], color="orange", linewidth=2.5)

# Animated spacecraft marker
spacecraft, = ax.plot(
    [], [], [],
    marker="o",
    color="orange",
    markersize=8,
    linestyle="None",
    label="Spacecraft"
)

# Time display
time_text = ax.text2D(
    0.03, 0.95, "",
    transform=ax.transAxes
)

# Equal axis scaling
all_points = np.vstack((positions, R1, R2))
limit = 1.05 * np.max(np.abs(all_points))

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)
ax.set_box_aspect((1, 1, 1))

ax.legend(loc="upper right")


# --------------------------------------------------
# 4. Animate the transfer
# --------------------------------------------------

def update(frame):
    # Draw the trajectory covered so far
    trail.set_data(
        positions[:frame + 1, 0],
        positions[:frame + 1, 1]
    )
    trail.set_3d_properties(
        positions[:frame + 1, 2]
    )

    # Move the spacecraft
    spacecraft.set_data(
        [positions[frame, 0]],
        [positions[frame, 1]]
    )
    spacecraft.set_3d_properties(
        [positions[frame, 2]]
    )

    time_text.set_text(
        f"Time: {times[frame]:.1f} / {tof:.1f} s"
    )

    return trail, spacecraft, time_text


ani = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=30,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()

# Optional: save the animation as a GIF.
# Install Pillow first: pip install pillow
# ani.save("lambert_transfer.gif", writer="pillow", fps=30)

# Optional: save as MP4 (requires FFmpeg).
# ani.save("lambert_transfer.mp4", writer="ffmpeg", fps=30)