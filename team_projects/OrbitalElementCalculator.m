% =========================================================================
% SPACECRAFT ORBIT STATE ANALYSIS
%
% Authors: Alex Jenz, Manuel Perez
% =========================================================================
clear; clc; close all;

% Constants
R_E = 6378.137;           % Earth radius in km
mu = 398600.4418;         % Earth's gravitational parameter (km^3/s^2)

% User Choice Menu
option = input(sprintf('Choose input mode:\n  (1) State Vectors (r, v)\n  (2) Orbital Elements (a, e, i, RAAN, omega, nu)\nEnter choice (1 or 2): '));

if option == 1
    r_str = input('Enter x, y, z of position vector (space-separated, km): ', 's');
    v_str = input('Enter x, y, z of velocity vector (space-separated, km/s): ', 's');
    r = str2num(r_str)';
    v = str2num(v_str)';
    
    [a, e, i, RAAN, omega, nu] = getOrbitalElements(r, v, mu);
else
    a = str2double(input('Enter semi-major axis (km): ', 's'));
    e = str2double(input('Enter eccentricity: ', 's'));
    i = deg2rad(str2double(input('Enter inclination (deg): ', 's')));
    RAAN = deg2rad(str2double(input('Enter RAAN (deg): ', 's')));
    omega = deg2rad(str2double(input('Enter argument of periapsis (deg): ', 's')));
    nu = deg2rad(str2double(input('Enter true anomaly (deg): ', 's')));
    
    [r, v] = getPosVel(a, e, i, RAAN, omega, nu, mu);
end

%% COMPREHENSIVE ORBITAL ANALYSIS
r_mag = norm(r);
speed = norm(v);
height = r_mag - R_E;

% Orbit distances and heights
perigeeRadius = a * (1 - e);
apogeeRadius = a * (1 + e);
perigeeHeight = perigeeRadius - R_E;
apogeeHeight = apogeeRadius - R_E;

% Orbital period and speeds
periodSeconds = 2 * pi * sqrt(a^3 / mu);
periodMinutes = periodSeconds / 60;
perigeeSpeed = sqrt(mu * (2 / perigeeRadius - 1 / a));
apogeeSpeed = sqrt(mu * (2 / apogeeRadius - 1 / a));

% Determine Orbit Type
if e < 0.01
    orbitType = 'Nearly Circular';
elseif e < 1
    orbitType = 'Elliptical';
elseif abs(e - 1) < 0.01
    orbitType = 'Approximately Parabolic';
else
    orbitType = 'Hyperbolic';
end

%% PRINT DETAILED REPORT
fprintf('\n');
fprintf('========================================\n');
fprintf('        SPACECRAFT ORBIT ANALYSIS       \n');
fprintf('========================================\n');
fprintf('\nSPACECRAFT STATE\n');
fprintf('----------------------------------------\n');
fprintf('Position X:             %.2f km\n', r(1));
fprintf('Position Y:             %.2f km\n', r(2));
fprintf('Position Z:             %.2f km\n', r(3));
fprintf('Current Speed:          %.3f km/s\n', speed);
fprintf('Current Height:         %.2f km\n', height);
fprintf('\nORBITAL ELEMENTS\n');
fprintf('----------------------------------------\n');
fprintf('Semi-Major Axis:        %.2f km\n', a);
fprintf('Eccentricity:           %.5f\n', e);
fprintf('Inclination:            %.2f degrees\n', rad2deg(i));
fprintf('RAAN:                   %.2f degrees\n', rad2deg(RAAN));
fprintf('Argument of Periapsis:  %.2f degrees\n', rad2deg(omega));
fprintf('True Anomaly:           %.2f degrees\n', rad2deg(nu));
fprintf('\nORBIT CHARACTERISTICS\n');
fprintf('----------------------------------------\n');
fprintf('Orbit Type:             %s\n', orbitType);
fprintf('Perigee Height:         %.2f km\n', perigeeHeight);
fprintf('Apogee Height:          %.2f km\n', apogeeHeight);
fprintf('Orbital Period:         %.2f minutes\n', periodMinutes);
fprintf('Perigee Speed:          %.3f km/s\n', perigeeSpeed);
fprintf('Apogee Speed:           %.3f km/s\n', apogeeSpeed);
fprintf('========================================\n');

%% VISUALIZE ORBIT
displayOrbit(a, e, i, RAAN, omega, r, perigeeRadius, apogeeRadius, R_E);


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [a, e, i, RAAN, omega, nu] = getOrbitalElements(r, v, mu)
    r_mag = norm(r);
    v_mag = norm(v);
    K = [0; 0; 1];
    
    energy = v_mag^2 / 2 - mu / r_mag;
    h = cross(r, v);
    h_mag = norm(h);
    
    % Semi-major Axis
    a = -mu / (2 * energy);
    
    % Eccentricity Vector & Magnitude
    e_vec = ((v_mag^2 - mu/r_mag) * r - dot(r, v) * v) / mu;
    e = norm(e_vec);
    
    % Inclination
    i = acos(h(3) / h_mag);
    
    % RAAN
    n = cross(K, h);
    n_mag = norm(n);
    if n_mag ~= 0
        RAAN = acos(n(1) / n_mag);
        if n(2) < 0
            RAAN = 2*pi - RAAN;
        end
    else
        RAAN = 0;
    end
    
    % Argument of Periapsis
    if n_mag ~= 0 && e > 1e-8
        omega = acos(dot(n, e_vec) / (n_mag * e));
        if e_vec(3) < 0
            omega = 2*pi - omega;
        end
    else
        omega = 0;
    end
    
    % True Anomaly
    if e > 1e-8
        nu = acos(dot(e_vec, r) / (e * r_mag));
        if r(3) < 0
            nu = 2*pi - nu;
        end
    else
        nu = acos(dot(n, r) / (n_mag * r_mag));
        if r(3) < 0
            nu = 2*pi - nu;
        end
    end
end

function [r, v] = getPosVel(a, e, i, RAAN, omega, nu, mu)
    p = a * (1.0 - e^2);
    r_mag = p / (1.0 + e * cos(nu));
    r_pqf = [r_mag * cos(nu); r_mag * sin(nu); 0.0];
    
    v_mag = sqrt(mu / p);
    v_pqf = v_mag * [-sin(nu); e + cos(nu); 0.0];
    
    R3_RAAN = [cos(RAAN) -sin(RAAN) 0; sin(RAAN) cos(RAAN) 0; 0 0 1];
    R1_i    = [1 0 0; 0 cos(i) -sin(i); 0 sin(i) cos(i)];
    R3_om   = [cos(omega) -sin(omega) 0; sin(omega) cos(omega) 0; 0 0 1];
    
    Q = R3_RAAN * R1_i * R3_om;
    
    r = Q * r_pqf;
    v = Q * v_pqf;
end

function displayOrbit(a, e, i, RAAN, omega, r_current, r_perigee, r_apogee, R_E)
    theta = linspace(0, 2*pi, 600);
    orbitXYZ = zeros(3, length(theta));
    
    R3_RAAN = [cos(RAAN) -sin(RAAN) 0; sin(RAAN) cos(RAAN) 0; 0 0 1];
    R1_i    = [1 0 0; 0 cos(i) -sin(i); 0 sin(i) cos(i)];
    R3_om   = [cos(omega) -sin(omega) 0; sin(omega) cos(omega) 0; 0 0 1];
    Q = R3_RAAN * R1_i * R3_om;
    
    for k = 1:length(theta)
        nu_k = theta(k);
        r_mag_k = (a * (1 - e^2)) / (1 + e * cos(nu_k));
        r_pqf = [r_mag_k * cos(nu_k); r_mag_k * sin(nu_k); 0];
        orbitXYZ(:, k) = Q * r_pqf;
    end
    
    perigeePoint = Q * [r_perigee; 0; 0];
    apogeePoint = Q * [-r_apogee; 0; 0];
    
    figure;
    [X_earth, Y_earth, Z_earth] = sphere(80);
    surf(X_earth * R_E, Y_earth * R_E, Z_earth * R_E, ...
        'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    hold on;
    
    plot3(orbitXYZ(1,:), orbitXYZ(2,:), orbitXYZ(3,:), 'r', 'LineWidth', 2);
    plot3(r_current(1), r_current(2), r_current(3), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 8);
    plot3(perigeePoint(1), perigeePoint(2), perigeePoint(3), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
    plot3(apogeePoint(1), apogeePoint(2), apogeePoint(3), 'mo', 'MarkerFaceColor', 'm', 'MarkerSize', 8);
    
    axis equal;
    grid on;
    xlabel('X Position (km)');
    ylabel('Y Position (km)');
    zlabel('Z Position (km)');
    title('Spacecraft Orbit Analysis & 3D Visualization');
    legend('Earth', 'Orbit Path', 'Current Position', 'Perigee', 'Apogee', 'Location', 'best');
    view(3);
end