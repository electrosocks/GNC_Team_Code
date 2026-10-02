from math import *
import numpy as np
import matplotlib.pyplot as plt

mass_e = 5.9722 * 10 ** 24 #kg
mass_s = 549000 #kg (falcon nine mass)
r_e = 6371000 #m
r_a = r_e + 120000
G = 6.6743 * 10 ** -11 #m^3 / kg * s^2
theta_initial = radians(90)

def F_g (r):
    
    F_g = (G * mass_e * mass_s) / r ** 2
    return F_g

altitude_initial = 1000 * int(input("Enter initial altitude in km: "))
v_mag_initial = int(input("Enter initial velocity in m/s: "))
dtheta_initial = radians(float(input("Enter initial direction (tangent = 0 degrees, normal = 90 degrees): ")))
print()
t_manuever_1 = int(input("Enter time until first manuever in minutes: ")) * 60
dv_manuever_1 = int(input("Enter manuever velocity in m/s: "))   # FIX: true delta-v, no longer replaces velocity
dtheta_manuever_1 = radians(float(input("Enter manuever direction (tangent = 0 degrees, normal = 90 degrees): ")))
print()
# FIX: input is time AFTER the first maneuver, so add t_manuever_1 to make it absolute
t_manuever_2 = t_manuever_1 + int(input("Enter time until second manuever in minutes: ")) * 60
dv_manuever_2 = int(input("Enter manuever velocity in m/s: "))   # FIX: true delta-v, no longer replaces velocity
dtheta_manuever_2 = radians(float(input("Enter manuever direction (tangent = 0 degrees, normal = 90 degrees): ")))
print()


r = r_e + altitude_initial
theta = theta_initial

u1 = np.array([-cos(theta),-sin(theta)])
u2 = np.array([-sin(theta),cos(theta)])

dir_v = np.array([
    cos(dtheta_initial)*u2[0] + sin(dtheta_initial)*u1[0],
    cos(dtheta_initial)*u2[1] + sin(dtheta_initial)*u1[1]
])


xi = r * cos(theta)
yi = r * sin(theta)

fig, ax = plt.subplots()

# ax.quiver(xs,ys,u1[0],u1[1],color = "grey")
# ax.quiver(xs,ys,u2[0],u2[1],color = "green")
# ax.quiver(xs,ys,dir_v[0],dir_v[1],color = "magenta")

r_vec = r * np.array([cos(theta), sin(theta)])
v_vec = v_mag_initial * dir_v

x_history = []
y_history = []

dt = 1
t_list = np.arange(0,t_manuever_1,dt)

atm = 0

orbit_check_0 = 0
orbit_check_1 = 0
orbit_check_2 = 0

for t in t_list:
    r_mag = np.linalg.norm(r_vec)
    
    if (r_mag <= r_a) and (atm == 0):
        print(f"Re-entry at t = {t} s")
        atm = 1
        
    if r_mag <= r_e:
        print(f"Impact at t = {t} s")
        break
    
    a_vec = -G * mass_e / r_mag**3 * r_vec
    v_vec = v_vec + (a_vec * dt)
    r_vec = r_vec + (v_vec * dt)
    
    if (isclose(r_vec[0], xi, abs_tol=10000) and
    isclose(r_vec[1], yi, abs_tol=10000) and
    orbit_check_0 == 0 and
    t > 250):
        
        print("Time of first orbit = ", t / 60,"minutes.")
        orbit_check_0 = 1
        
    x_history.append(r_vec[0])
    y_history.append(r_vec[1])



# --- maneuver burn (at t = t_manuever) ---
theta_m_1 = atan2(r_vec[1], r_vec[0])   # current position angle, NOT initial theta

u1_m_1 = np.array([-cos(theta_m_1), -sin(theta_m_1)])   # points toward Earth
u2_m_1 = np.array([-sin(theta_m_1), cos(theta_m_1)])    # tangent (prograde)

dir_v_m_1 = np.array([
    cos(dtheta_manuever_1)*u2_m_1[0] + sin(dtheta_manuever_1)*u1_m_1[0],
    cos(dtheta_manuever_1)*u2_m_1[1] + sin(dtheta_manuever_1)*u1_m_1[1]
])

v_vec = v_vec + dv_manuever_1 * dir_v_m_1   # FIX: adds delta-v to current velocity
# ------------------------------------------


t_m_list_1 = np.arange(t_manuever_1, t_manuever_2, dt)

# FIX: was "if v_mag_manuever_1 > v_mag_initial", which skipped the segment for
# zero/retrograde burns. Now runs whenever the segment has time steps.
if len(t_m_list_1) > 0:

    for t in t_m_list_1:
        r_mag = np.linalg.norm(r_vec)
        
        if (r_mag <= r_a) and (atm == 0):
            print(f"Re-entry at t = {t} s")
            atm = 1
            
        if r_mag <= r_e:
            print(f"Impact at t = {t} s")
            break
        
        a_vec = -G * mass_e / r_mag**3 * r_vec
        v_vec = v_vec + (a_vec * dt)
        r_vec = r_vec + (v_vec * dt)
        
        if (isclose(r_vec[0], x_history[t_manuever_1 - 1], abs_tol=10000) and
        isclose(r_vec[1], y_history[t_manuever_1 - 1], abs_tol=10000) and
        orbit_check_1 == 0 and
        t > (t_manuever_1 + 250)):
            
            print("Time of second orbit = ", (t - t_manuever_1) / 60,"minutes.")
            orbit_check_1 = 1
            
        x_history.append(r_vec[0])
        y_history.append(r_vec[1])



# --- maneuver burn (at t = t_manuever) ---
theta_m_2 = atan2(r_vec[1], r_vec[0])   # current position angle, NOT initial theta

u1_m_2 = np.array([-cos(theta_m_2), -sin(theta_m_2)])   # points toward Earth
u2_m_2 = np.array([-sin(theta_m_2), cos(theta_m_2)])    # tangent (prograde)

dir_v_m_2= np.array([
    cos(dtheta_manuever_2)*u2_m_2[0] + sin(dtheta_manuever_2)*u1_m_2[0],
    cos(dtheta_manuever_2)*u2_m_2[1] + sin(dtheta_manuever_2)*u1_m_2[1]
])

v_vec = v_vec + dv_manuever_2 * dir_v_m_2   # FIX: adds delta-v to current velocity
# ------------------------------------------


t_m_list_2 = np.arange(t_manuever_2, t_manuever_2 + 250000, dt)

# FIX: same guard problem as above
if len(t_m_list_2) > 0:

    for t in t_m_list_2:
        r_mag = np.linalg.norm(r_vec)
        
        if (r_mag <= r_a) and (atm == 0):
            print(f"Re-entry at t = {t} s")
            atm = 1
            
        if r_mag <= r_e:
            print(f"Impact at t = {t} s")
            break
        
        a_vec = -G * mass_e / r_mag**3 * r_vec
        v_vec = v_vec + (a_vec * dt)
        r_vec = r_vec + (v_vec * dt)
        
        # FIX: second-orbit check keeps running after the second burn, in case the
        # orbit wasn't completed before t_manuever_2
        if (isclose(r_vec[0], x_history[t_manuever_1 - 1], abs_tol=10000) and
        isclose(r_vec[1], y_history[t_manuever_1 - 1], abs_tol=10000) and
        orbit_check_1 == 0 and
        t > (t_manuever_1 + 250)):
            
            print("Time of second orbit = ", (t - t_manuever_1) / 60,"minutes.")
            orbit_check_1 = 1
        
        if (isclose(r_vec[0], x_history[t_manuever_2 - 1], abs_tol=10000) and
        isclose(r_vec[1], y_history[t_manuever_2 - 1], abs_tol=10000) and
        orbit_check_2 == 0 and
        t >(t_manuever_2 + 250)):
            
            print("Time of third orbit = ", (t - t_manuever_2) / 60,"minutes.")
            orbit_check_2 = 1
            
        x_history.append(r_vec[0])
        y_history.append(r_vec[1])



#VISUALIZATION

theta_points_earth = np.linspace(0, 2*np.pi, 500)
xe = r_e * np.cos(theta_points_earth)
ye = r_e * np.sin(theta_points_earth)

xa = r_a * np.cos(theta_points_earth)
ya = r_a * np.sin(theta_points_earth)

#ax.plot(0, 0, color="black", marker="o")
ax.plot(xe, ye, color="blue")
ax.plot(xa, ya, color="red")

if orbit_check_0 == 1:
    ax.plot(xi, yi, color="green", marker=".")
if orbit_check_1 == 1:    
    ax.plot(x_history[t_manuever_1], y_history[t_manuever_1], color="green", marker=".")
if orbit_check_2 == 1:
    ax.plot(x_history[t_manuever_2], y_history[t_manuever_2], color="green", marker=".")
ax.plot(x_history, y_history, color="black")

ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('2 Body Propogator')
plt.show()
