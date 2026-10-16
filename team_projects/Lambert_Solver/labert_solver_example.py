import numpy as np

def izzo_lambert_solver(r1, r2, tf, mu=398600.4418, direction='prograde'):
    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    
    gamma = np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0)
    cross_prod = np.cross(r1, r2)
    
    d_theta = np.arccos(gamma)
    if direction == 'prograde' and cross_prod[2] < 0:
        d_theta = 2 * np.pi - d_theta
    elif direction == 'retrograde' and cross_prod[2] >= 0:
        d_theta = 2 * np.pi - d_theta
            
    c = np.sqrt(r1_norm**2 + r2_norm**2 - 2 * r1_norm * r2_norm * np.cos(d_theta))
    s = (r1_norm + r2_norm + c) / 2.0
    
    lambda_param = np.sqrt(1.0 - c / s)
    if d_theta > np.pi:
        lambda_param = -lambda_param
        
    T = np.sqrt(2 * mu / (s**3)) * tf
    
    def tof_and_derivatives(x):
        y = np.sqrt(1.0 - lambda_param**2 * (1.0 - x**2))
        if x < 1.0:
            psi = np.arccos(np.clip(x * y + lambda_param * (1.0 - x**2), -1.0, 1.0))
            alpha = 2.0 * psi
            beta = 2.0 * np.arcsin(np.clip(lambda_param * np.sqrt((1.0 - x**2) / (1.0 - lambda_param**2 * (1.0 - x**2))), -1.0, 1.0))
            t_val = (alpha - np.sin(alpha) - (beta - np.sin(beta))) / 2.0
        else:
            alpha = 2.0 * np.arccosh(x)
            beta = 2.0 * np.arcsinh(lambda_param * np.sqrt((x**2 - 1.0) / (1.0 - lambda_param**2 * (1.0 - x**2))))
            t_val = (np.sinh(alpha) - alpha - (np.sinh(beta) - beta)) / 2.0
            
        dt_dx = (3.0 * x * t_val - 2.0 + 2.0 * lambda_param**3 * x / y) / (1.0 - x**2)
        d2t_dx2 = (3.0 * t_val + 5.0 * x * dt_dx + 2.0 * (1.0 - lambda_param**5) / (y**3)) / (1.0 - x**2)
        return t_val, dt_dx, d2t_dx2

    T_0 = (2.0 / 3.0) * (1.0 - lambda_param**3)
    x = (T_0 / T)**(2.0 / 3.0) - 1.0 if T >= T_0 else (T_0 / T)**(1.0 / 3.0)
    
    for _ in range(50):
        t_val, f, f_prime = tof_and_derivatives(x)
        delta_t = t_val - T
        if abs(delta_t) < 1e-11:
            break
        step = delta_t / f * (1.0 + (delta_t * f_prime) / (2.0 * f**2))
        x -= step

    y = np.sqrt(1.0 - lambda_param**2 * (1.0 - x**2))
    gamma_val = np.sqrt(mu * s / 2.0)
    rho = (r1_norm - r2_norm) / c
    sigma = np.sqrt(1.0 - rho**2)
    
    VR1 = gamma_val * ((lambda_param * y - x) - rho * (lambda_param * y + x)) / r1_norm
    VR2 = -gamma_val * ((lambda_param * y - x) + rho * (lambda_param * y + x)) / r2_norm
    VT1 = gamma_val * sigma * (y + lambda_param * x) / r1_norm
    VT2 = gamma_val * sigma * (y + lambda_param * x) / r2_norm
    
    i_r1, i_r2 = r1 / r1_norm, r2 / r2_norm
    i_h = np.cross(r1, r2)
    i_h /= np.linalg.norm(i_h)
    if direction == 'retrograde':
        i_h = -i_h
        
    i_t1, i_t2 = np.cross(i_h, i_r1), np.cross(i_h, i_r2)
    
    v1 = VR1 * i_r1 + VT1 * i_t1
    v2 = VR2 * i_r2 + VT2 * i_t2
    return v1, v2


def get_vector_input(prompt):
    raw = input(prompt)
    return [float(val.strip()) for val in raw.split(",")]


if __name__ == "__main__":
    print("=== Interactive Lambert Solver (Izzo's Algorithm) ===")
    
    # User prompts
    r1 = get_vector_input("Enter initial position r1 (x, y, z in km): ")
    r2 = get_vector_input("Enter final position r2 (x, y, z in km): ")
    tf = float(input("Enter transfer time (in seconds): "))
    
    mu_input = input("Enter gravitational parameter mu [default Earth = 398600.44]: ").strip()
    mu = float(mu_input) if mu_input else 398600.4418
    
    v1, v2 = izzo_lambert_solver(r1, r2, tf, mu=mu)
    
    print("\n--- RESULTS ---")
    print(f"v1 (Departure Velocity) : {np.round(v1, 4)} km/s  (Magnitude: {np.linalg.norm(v1):.4f} km/s)")
    print(f"v2 (Arrival Velocity)   : {np.round(v2, 4)} km/s  (Magnitude: {np.linalg.norm(v2):.4f} km/s)")