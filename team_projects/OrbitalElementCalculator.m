% ############################################################### %
%
% Spacecraft Orbital Elements Calculator
%
%   Choose between inputting either (1) a spacecraft's position and
%   velocity vectors or (2) the orbital elements of a spacecraft
%   and recieve the position and velocity vectors.
%
%   If opton (1) is selected, the positon and velocity will be
%   inputted in component form with the units of kilometers and
%   kilometers per second, respectfully.
%   
%   If option (2) is chosen, the semi-major axis (a), eccentricity (e),
%   inclination (i), right ascension of ascending node (RAAN), argument
%   of perigee (omega), and true anomoly (nu) will be inputted.
%   The units are kilometers for a and degrees for i, RAAN, omega, 
%   and nu.
%
% ############################################################### %


% Displays a 3D model of the orbit
function displayOrbit(a, e, i, RAAN, omega, R_E)
    nu_span = linspace(0, 2*pi, 300);
    r_orbit = zeros(3, length(nu_span));
    
    for k = 1:length(nu_span)
        nu_k = nu_span(k);
        r_mag_k = (a * (1 - e^2)) / (1 + e * cos(nu_k));
        r_pqf = [r_mag_k * cos(nu_k); r_mag_k * sin(nu_k); 0];
    
        R3_RAAN = [cos(RAAN) -sin(RAAN) 0; sin(RAAN) cos(RAAN) 0; 0 0 1];
        R1_i    = [1 0 0; 0 cos(i) -sin(i); 0 sin(i) cos(i)];
        R3_om   = [cos(omega) -sin(omega) 0; sin(omega) cos(omega) 0; 0 0 1];
        
        Q = R3_RAAN * R1_i * R3_om;
        r_orbit(:, k) = Q * r_pqf;
    end
    
    figure;
    plot3(r_orbit(1,:), r_orbit(2,:), r_orbit(3,:), 'LineWidth', 2, 'Color', 'r');
    hold on;
    grid on;
    axis equal;
    xlabel('X (km)');
    ylabel('Y (km)');
    zlabel('Z (km)');
    title('3D Orbit');
    
    [X_earth, Y_earth, Z_earth] = sphere(50);
    surf(X_earth * R_E, Y_earth * R_E, Z_earth * R_E, ...
        'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    
    legend('Orbit Path', 'Earth', 'Location', 'best');
    view(3);
end

function [a, e, i, RAAN, omega, nu] = getOrbitalElements(r, v, M)
    G = 6.6743e-20; % km^3/(kg*s^2)
    mu = G * M;
    r_mag = norm(r);
    v_mag = norm(v);

    K = [0; 0; 1];
    
    energy = v_mag^2 / 2 - mu / r_mag;  % Mechanical Energy
    h = cross(r, v);    % Angular Momentum
    h_mag = norm(h);

    % Semi-major Axis Calculation
    a = -mu / (2 * energy);

    % Eccentricity Calculation
    e_vec = ((v_mag^2 - mu/r_mag) * r - dot(r, v) * v) / mu;
    e = norm(e_vec);

    % Inclination Calculation
    i = acos(h(3) / h_mag);

    % RAAN Cacluation
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
    
    % Argument of Perigee (omega) Calculation
    if n_mag ~= 0 && e > 1e-8
        omega = acos(dot(n, e_vec) / (n_mag * e));
        if e_vec(3) < 0
            omega = 2*pi - omega;
        end
    else
        omega = 0;
    end

    % True Anomoly Calculation
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

function [r, v] = getPosVel(a, e, i, RAAN, omega, nu, M)
    G = 6.6743e-20; % km^3/(kg*s^2)
    mu = G * M;
    
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


% Main Execution Script
R_E = 6371;     % Earth radius in km
M_E = 5.972e24; % Earth mass in kg

option = input("Do you want to get orbital elements from state vectors (1) " ...
    + "or state vectors from orbital elements (2)?\nEnter choice: ");

if option == 1
    r_str = input("Enter x, y, z of position vector (space-separated): ", 's');
    v_str = input("Enter x, y, z of velocity vector (space-separated): ", 's');
    
    r = str2num(r_str)';
    v = str2num(v_str)';
    
    [a, e, i, RAAN, omega, nu] = getOrbitalElements(r, v, M_E);
    fprintf('a: %.3f km\ne: %.3f\ni: %.3f deg\nRAAN: %.3f deg\nomega: %.3f deg\nnu: %.3f deg\n', ...
        a, e, rad2deg(i), rad2deg(RAAN), rad2deg(omega), rad2deg(nu));
    displayOrbit(a, e, i, RAAN, omega, R_E);
else
    a = str2double(input("Enter semi-major axis (km): ", 's'));
    e = str2double(input("Enter eccentricity: ", 's'));
    i = deg2rad(str2double(input("Enter inclination (deg): ", 's')));
    RAAN = deg2rad(str2double(input("Enter RAAN (deg): ", 's')));
    omega = deg2rad(str2double(input("Enter argument of periapsis degrad): ", 's')));
    nu = deg2rad(str2double(input("Enter true anomaly (deg): ", 's')));
    
    [r, v] = getPosVel(a, e, i, RAAN, omega, nu, M_E);
    disp('Position Vector r (km):');
    disp(r);
    disp('Velocity Vector v (km/s):');
    disp(v);
    displayOrbit(a, e, i, RAAN, omega, R_E);
end
