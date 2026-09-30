# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:26:24 2026

@author: antha
"""
import numpy as np



radius = 6371 # kilometers
starting_alt = 400 # kilometers
final_alt = 500 # kilometers
gravity = 398600 #km^3/s^2
inner_radius = radius+starting_alt 
final_radius = radius+final_alt

#velocity needed to stay at intial alt
velo_i= np.sqrt(gravity/inner_radius)
print(f'velocity needed to stay at intial alt of {starting_alt} km is {velo_i:.4f} km/s^2')

#velocity needed to stay at intial alt
velo_f= np.sqrt(gravity/final_radius)
print(f'velocity needed to stay at final alt of {final_alt} km is {velo_f:.4f} km/s^2')

#first burn at inital alt
first_b = velo_i * np.sqrt((2*final_radius)/(inner_radius+final_radius))
delta_V1 = first_b - velo_i
print(f'the velocity need at first burn is {delta_V1:.4f} km/s')
#second/ final burn at final alt
final_b = velo_f * np.sqrt((2*inner_radius)/(inner_radius+final_radius))
delta_V2 = velo_f-final_b
print(f'the velocity need at second burn is {delta_V2:.4f} km/s')

# transfer time
a = (inner_radius+final_radius)/2 #tranfer trajectory
time = (np.pi * np.sqrt(a**3/gravity))/60
print(f'The time required to reach final altitude is {time:.3f} minutes.')
