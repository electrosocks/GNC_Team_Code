import numpy as np

radius = 6371 # kilometers
starting_alt = 400 # kilometers 
final_alt = 180000 # kilometers 
r_eliptical = 300000 #kilometers(rb)
gravity = 398600 #km^3/s^2 (mu)
R_1 = radius+starting_alt 
R_2 = radius+final_alt
radius_ratio = R_2/R_1

if radius_ratio >= 11.94:
    #print("use Bieliptical transfer")
    orbit_1_speed =  np.sqrt(gravity/R_1)
    orbit_2_speed = np.sqrt(gravity/R_2)
    a1 = (R_1 + r_eliptical)/2 # semi-major axis for inital orbit
    a2 = (R_2 +r_eliptical)/2  # semi-major acis for final orbit
#BURN 1
    v_1a = np.sqrt(gravity*((2/R_1)-(1/a1))) #periapsis speed on transfer ellipse 1
    del_v1 = v_1a - orbit_1_speed #first burn delta-V
#BURN 2 (at r_eliptical)
    v_1b = np.sqrt(gravity*((2/r_eliptical)-(1/a1))) #apoapsis speed on transfer ellipse 1
    v_2b = np.sqrt(gravity*((2/r_eliptical)-(1/a2))) #apoapsis speed on transfer ellipse 2
    del_v2 = np.abs(v_2b-v_1b)
#BURN 3 
    v_2c = np.sqrt(gravity*((2/R_2)-(1/a2))) #periapsis speed on transfer ellipse 2
    del_v3 = np.abs(orbit_2_speed - v_2c)
#total delta V
    v_total = del_v1 + del_v2 + del_v3
#time 
    time_hr = ((np.pi*np.sqrt((a1**3/gravity)))+(np.pi*np.sqrt((a2**3/gravity))))/3600 # time in hours
    time_day = ((np.pi*np.sqrt((a1**3/gravity)))+(np.pi*np.sqrt((a2**3/gravity))))/86400 # time in days

    print(f"Burn 1 (at R_1): {del_v1:.4f} km/s")
    print(f"Burn 2 (at R_b): {del_v2:.4f} km/s")
    print(f"Burn 3 (at R_2): {del_v3:.4f} km/s")
    print(f"Total Delta-V: {v_total:.4f} km/s")
    print(f"Total time: {time_hr:.2f} hours or {time_day:.2f} days")
else:
    print("Use Hohmann transfer")