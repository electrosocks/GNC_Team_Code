function[rvec, vvec, iter] = twoBodyPropagate(r0vec,v0vec,dt,mu,tol,maxiter)
%[text] $\\begin{array}{l}\n\\mathrm{P}\\mathrm{r}\\mathrm{o}\\mathrm{p}\\mathrm{a}\\mathrm{g}\\mathrm{a}\\mathrm{t}\\mathrm{e}\\mathrm{s}\\;\\mathrm{t}\\mathrm{h}\\mathrm{e}\\;\\mathrm{s}\\mathrm{t}\\mathrm{a}\\mathrm{t}\\mathrm{e}\\;\\mathrm{b}\\mathrm{y}\\;\\mathrm{t}\\mathrm{i}\\mathrm{m}\\mathrm{e}\\;\\mathrm{d}\\mathrm{t}\\\\\n\\\\\n\\mathrm{I}\\mathrm{n}\\mathrm{p}\\mathrm{u}\\mathrm{t}\\mathrm{s}:\\\\\n\\mathrm{r}0\\mathrm{v}\\mathrm{e}\\mathrm{c}=\\mathrm{i}\\mathrm{n}\\mathrm{i}\\mathrm{t}\\mathrm{i}\\mathrm{a}\\mathrm{l}\\;\\mathrm{p}\\mathrm{o}\\mathrm{s}\\mathrm{i}\\mathrm{t}\\mathrm{i}\\mathrm{o}\\mathrm{n}\\;\\mathrm{v}\\mathrm{e}\\mathrm{c}\\mathrm{t}\\mathrm{o}\\mathrm{r}\\\\\n\\mathrm{v}0\\mathrm{v}\\mathrm{e}\\mathrm{c}=\\mathrm{i}\\mathrm{n}\\mathrm{i}\\mathrm{t}\\mathrm{i}\\mathrm{a}\\mathrm{l}\\;\\mathrm{v}\\mathrm{e}\\mathrm{l}\\mathrm{o}\\mathrm{c}\\mathrm{i}\\mathrm{t}\\mathrm{y}\\;\\mathrm{v}\\mathrm{e}\\mathrm{c}\\mathrm{t}\\mathrm{o}\\mathrm{r}\\\\\n\\mathrm{d}\\mathrm{t}=\\mathrm{t}\\mathrm{i}\\mathrm{m}\\mathrm{e}\\;\\mathrm{o}\\mathrm{f}\\;\\mathrm{f}\\mathrm{l}\\mathrm{i}\\mathrm{g}\\mathrm{h}\\mathrm{t}\\\\\n\\mathrm{m}\\mathrm{u}=\\mathrm{g}\\mathrm{r}\\mathrm{a}\\mathrm{v}\\mathrm{a}\\mathrm{t}\\mathrm{a}\\mathrm{t}\\mathrm{i}\\mathrm{o}\\mathrm{n}\\mathrm{a}\\mathrm{l}\\;\\mathrm{c}\\mathrm{o}\\mathrm{n}\\mathrm{s}\\mathrm{t}\\mathrm{a}\\mathrm{n}\\mathrm{t}\\\\\n\\mathrm{t}\\mathrm{o}\\mathrm{l}=\\mathrm{t}\\mathrm{o}\\mathrm{l}\\mathrm{e}\\mathrm{r}\\mathrm{a}\\mathrm{n}\\mathrm{c}\\mathrm{e}\\;\\mathrm{o}\\mathrm{f}\\;\\mathrm{u}\\mathrm{p}\\mathrm{d}\\mathrm{a}\\mathrm{t}\\mathrm{e}\\\\\n\\mathrm{m}\\mathrm{a}\\mathrm{x}\\mathrm{i}\\mathrm{t}\\mathrm{e}\\mathrm{r}=\\mathrm{m}\\mathrm{a}\\mathrm{x}\\mathrm{i}\\mathrm{m}\\mathrm{u}\\mathrm{m}\\;\\mathrm{i}\\mathrm{t}\\mathrm{e}\\mathrm{r}\\mathrm{a}\\mathrm{t}\\mathrm{i}\\mathrm{o}\\mathrm{n}\\mathrm{s}\\\\\n\\\\\n\\mathrm{O}\\mathrm{u}\\mathrm{t}\\mathrm{p}\\mathrm{u}\\mathrm{t}\\mathrm{s}:\\\\\n\\mathrm{r}\\mathrm{v}\\mathrm{e}\\mathrm{c}=\\mathrm{p}\\mathrm{r}\\mathrm{o}\\mathrm{p}\\mathrm{a}\\mathrm{g}\\mathrm{a}\\mathrm{t}\\mathrm{e}\\mathrm{d}\\;\\mathrm{p}\\mathrm{o}\\mathrm{s}\\mathrm{i}\\mathrm{t}\\mathrm{i}\\mathrm{o}\\mathrm{n}\\;\\mathrm{v}\\mathrm{e}\\mathrm{c}\\mathrm{t}\\mathrm{o}\\mathrm{r}\\\\\n\\mathrm{v}\\mathrm{v}\\mathrm{e}\\mathrm{c}=\\mathrm{p}\\mathrm{r}\\mathrm{o}\\mathrm{p}\\mathrm{a}\\mathrm{g}\\mathrm{a}\\mathrm{t}\\mathrm{e}\\mathrm{d}\\;\\mathrm{v}\\mathrm{e}\\mathrm{l}\\mathrm{o}\\mathrm{c}\\mathrm{i}\\mathrm{t}\\mathrm{y}\\;\\mathrm{v}\\mathrm{e}\\mathrm{c}\\mathrm{t}\\mathrm{o}\\mathrm{r}\\\\\n\\mathrm{i}\\mathrm{t}\\mathrm{e}\\mathrm{r}=\\mathrm{i}\\mathrm{t}\\mathrm{e}\\mathrm{r}\\mathrm{a}\\mathrm{t}\\mathrm{i}\\mathrm{o}\\mathrm{n}\\mathrm{s}\\;\\mathrm{u}\\mathrm{s}\\mathrm{e}\\mathrm{d}\n\\end{array}${"editStyle":"visual"}
%%
%[text] ## Setup
%[text] $\\mathrm{Set}\\;\\mathrm{default}\\;\\mathrm{tolerance}\\;\\mathrm{and}\\;\\mathrm{iterations}${"editStyle":"visual"}
if nargin <5 || isempty(tol), tol = 1e-8; end
if nargin <6 || isempty(maxiter) maxiter = 100; end
%[text] $\\mathrm{Force}\\;\\mathrm{column}\\;\\mathrm{vectors}\\;\\mathrm{and}\\;\\mathrm{normalize}\\;\\mathrm{vectors}${"editStyle":"visual"}
%%rowInput = isrow(r0vec);

r0vec = r0vec(:); 
v0vec = v0vec(:);

r0 = norm(r0vec); %initial position
v0 = norm(v0vec); %initial velocity
vr0 = dot(r0vec, v0vec) / r0;

alpha = 2/r0 - v0^2/mu; % 1/[semimajor axis] or (1/a)
%[text] $\\\[\n\\alpha = \\frac{2}{r\_0} - \\frac{v\_0^2}{\\mu}\n\\\]$
%%
%[text] ## Main Loop
if alpha> 1e-6 %ellipse
    chi =sqrt(mu) * dt * alpha;
elseif alpha < -1e-6 %hyperbola
a = 1/alpha;
chi = sign(dt) * sqrt(-a) * log((-2*mu*alpha*dt) / ...
    (dot(r0vec,v0vec) + sign(dt)*sqrt(-mu*a)*(1-r0*alpha)));
else
    hvec = cros(r0vec, v0vec); %angular momentum
    h = norm(hvec);
    p = h^2/mu; %semi latus rectum
    s = 0.5 * atan(1/(3*sqrt(mu/p^3)*dt)); %intermediate angle
    w = atan(tan(s)^(1/3)); %intermediate angle
    chi = sqrt(p) * 2/ tan(2*w);
end 
%[text] $\\\[\n\\begin{cases}\n\n\\displaystyle\n\\chi = \\sqrt{\\mu}\\,\\Delta t\\,\\alpha,\n& \\alpha \> 10^{-6}\n\\qquad \\text{(ellipse)}\n\\\\\[12pt\]\n\n\\displaystyle\na = \\frac{1}{\\alpha},\n\\\\\[6pt\]\n\n\\displaystyle\n\\chi =\n\\mathrm{sign}(\\Delta t)\\sqrt{-a}\n\\ln\\left(\n\\frac{-2\\mu\\alpha\\Delta t}\n{\\mathbf{r}\_0\\cdot\\mathbf{v}\_0\n+\\mathrm{sign}(\\Delta t)\\sqrt{-\\mu a}\n(1-r\_0\\alpha)}\n\\right),\n& \\alpha \< -10^{-6}\n\\qquad \\text{(hyperbola)}\n\\\\\[12pt\]\n\n\\displaystyle\n\\mathbf{h} = \\mathbf{r}\_0\\times\\mathbf{v}\_0,\n\\qquad\nh = \\|\\mathbf{h}\\|,\n\\qquad\np = \\frac{h^2}{\\mu},\n\\\\\[6pt\]\n\n\\displaystyle\ns =\n\\frac{1}{2}\n\\tan^{-1}\\left(\n\\frac{1}\n{3\\sqrt{\\mu/p^3}\\,\\Delta t}\n\\right),\n\\\\\[6pt\]\n\n\\displaystyle\nw =\n\\tan^{-1}\\left(\\tan(s)^{1/3}\\right),\n\\\\\[6pt\]\n\n\\displaystyle\n\\chi =\n\\sqrt{p}\\,\n\\frac{2}{\\tan(2w)},\n& |\\alpha| \\leq 10^{-6}\n\\qquad \\text{(parabola)}\n\n\\end{cases}\n\\\]$
%%
%[text] ## Kepler Equation
%Newton-Raphson solution
ratio = 1;
iter=0;

while abs(ratio) > tol && iter < maxiter
    z = alpha* chi^2; %universal variable argument
    C= stumpffC(z); %C(z) at chi
    S= stumpffS(z); %S(z) at chi
%[text] Kepler equation residual
    F = (r0*vr0/sqrt(mu))*chi^2*C + (1-alpha*r0)*chi^3*S + r0*chi - sqrt(mu)*dt;
    Fp= (r0*vr0/sqrt(mu))*chi*(1-alpha*chi^2*S) + (1-alpha*r0)*chi^2*C + r0; 
    
%[text] $\\\[\nF =\n\\frac{r\_0 v\_{r0}}{\\sqrt{\\mu}}\\chi^2 C\n+\n(1-\\alpha r\_0)\\chi^3 S\n+\nr\_0\\chi\n-\n\\sqrt{\\mu}\\Delta t\n\\\]$
%[text] $\\\[\nF' =\n\\frac{r\_0 v\_{r0}}{\\sqrt{\\mu}}\n\\chi\n\\left(\n1-\\alpha\\chi^2 S\n\\right)\n+\n(1-\\alpha r\_0)\\chi^2 C\n+\nr\_0\n\\\]$
    ratio = F/ Fp; %newton-raphson update
    chi = chi - ratio;
    iter = iter +1; %counter
end

if iter == maxiter && abs(ratio) > tol
    warning('twoBodyPropagate:noConverge', ...
        ['Universal anomaly did not converge to tolerance %.1e ', ...
        'within %d iterations (last update %.3e).'], tol, maxiter, ratio);
end
%[text] 
z = alpha * chi^2; % re-evaluate at converged chi
%Run at converged chi
C = stumpffC(z); 
S = stumpffS(z);

%[text] $\\mathrm{Lagrange}\\;\\mathrm{coefficients}${"editStyle":"visual"}
f = 1- (chi^2/r0)*C ;
g = dt - (chi^3/sqrt(mu))*S;

%new position and velocity vectors
rvec = f*r0vec +g*v0vec;
r = norm(rvec);

fdot = (sqrt(mu)/(r*r0)) * (alpha * chi^3 *S - chi);
gdot = 1- (chi^2/r)*C;

vvec = fdot*r0vec + gdot*v0vec; % new velocity vector
%[text] $\\\[\nf = 1-\\frac{\\chi^2}{r\_0}C(z)\n\\\]\n\n\\\[\ng = \\Delta t-\\frac{\\chi^3}{\\sqrt{\\mu}}S(z)\n\\\]\n\n\n\\\[\n\\mathbf{r}\n=\nf\\mathbf{r}\_0+g\\mathbf{v}\_0\n\\\]\n\n\n\n\\\[\nr=\\|\\mathbf{r}\\|.\n\\\]\n\n\n\\\[\n\\dot{f}\n=\n\\frac{\\sqrt{\\mu}}{rr\_0}\n\\left\[\n\\alpha\\chi^3S(z)-\\chi\n\\right\]\n\\\]\n\n\\\[\n\\dot{g}\n=\n1-\\frac{\\chi^2}{r}C(z).\n\\\]\n\n\n\n\\\[\n\\mathbf{v}\n=\n\\dot{f}\\mathbf{r}\_0\n+\n\\dot{g}\\mathbf{v}\_0.\n\\\]$
%[text] 
%[text] The function is called as
%[text] $\n\n\\\[\n\[\\mathbf{r},\\mathbf{v},\\mathrm{iter}\]\n=\n\\mathrm{twoBodyPropagate}\n(\\mathbf{r}\_0,\\mathbf{v}\_0,\\Delta t,\\mu,\\mathrm{tol},\\mathrm{maxiter}),\n\\\]\n\nwith default tolerance \\(\\mathrm{tol}=10^{-8}\\) and maximum iterations\n\\(\\mathrm{maxiter}=100\\). The input vectors are reshaped to columns, and\nthe initial scalars are\n\n\\\[\nr\_0 = \\|\\mathbf{r}\_0\\|,\n\\qquad\nv\_0 = \\|\\mathbf{v}\_0\\|,\n\\qquad\nv\_{r0} = \\frac{\\mathbf{r}\_0\\cdot\\mathbf{v}\_0}{r\_0}.\n\\\]\n\nThe reciprocal of the semimajor axis is\n\n\\\[\n\\alpha = \\frac{2}{r\_0} - \\frac{v\_0^2}{\\mu}.\n\\\]\n\nFor an elliptical orbit,\n\n\\\[\n\\alpha \> 10^{-6},\n\\\]\n\nthe initial universal anomaly is\n\n\\\[\n\\chi = \\sqrt{\\mu}\\,\\Delta t\\,\\alpha.\n\\\]\n\nFor a hyperbolic orbit,\n\n\\\[\n\\alpha \< -10^{-6},\n\\\]\n\nthe semimajor axis is\n\n\\\[\na = \\frac{1}{\\alpha},\n\\\]\n\nand the universal anomaly is\n\n\\\[\n\\chi =\n\\mathrm{sign}(\\Delta t)\\sqrt{-a}\n\\ln\\left(\n\\frac{-2\\mu\\alpha\\Delta t}\n{\\mathbf{r}\_0\\cdot\\mathbf{v}\_0\n+\\mathrm{sign}(\\Delta t)\\sqrt{-\\mu a}(1-r\_0\\alpha)}\n\\right).\n\\\]\n\nFor the parabolic case,\n\n\\\[\n|\\alpha| \\leq 10^{-6},\n\\\]\n\nthe angular momentum vector and its magnitude are\n\n\\\[\n\\mathbf{h} = \\mathbf{r}\_0 \\times \\mathbf{v}\_0,\n\\qquad\nh = \\|\\mathbf{h}\\|,\n\\\]\n\nthe semilatus rectum is\n\n\\\[\np = \\frac{h^2}{\\mu},\n\\\]\n\nand the intermediate angles are\n\n\\\[\ns = \\frac{1}{2}\\tan^{-1}\\!\\left(\\frac{1}{3\\sqrt{\\mu/p^3}\\,\\Delta t}\\right),\n\\qquad\nw = \\tan^{-1}\\!\\left(\\tan(s)^{1/3}\\right).\n\\\]\n\nThe universal anomaly is then\n\n\\\[\n\\chi = \\sqrt{p}\\,\\frac{2}{\\tan(2w)}.\n\\\]\n\nThe universal anomaly is refined by Newton--Raphson iteration. At each\nstep, with\n\n\\\[\nz = \\alpha\\chi^2,\n\\\]\n\nthe Stumpff functions \\(C(z)\\) and \\(S(z)\\) are evaluated, and the\nfunction and its derivative are\n\n\\\[\nF = \\frac{r\_0 v\_{r0}}{\\sqrt{\\mu}}\\chi^2 C\n+ (1-\\alpha r\_0)\\chi^3 S\n+ r\_0\\chi\n- \\sqrt{\\mu}\\,\\Delta t,\n\\\]\n\n\\\[\nF' = \\frac{r\_0 v\_{r0}}{\\sqrt{\\mu}}\\chi(1-\\alpha\\chi^2 S)\n+ (1-\\alpha r\_0)\\chi^2 C\n+ r\_0.\n\\\]\n\nThe update is\n\n\\\[\n\\chi \\leftarrow \\chi - \\frac{F}{F'},\n\\\]\n\nand iteration continues until\n\n\\\[\n\\left|\\frac{F}{F'}\\right| \\leq \\mathrm{tol}\n\\quad\\text{or}\\quad\n\\mathrm{iter} \\geq \\mathrm{maxiter}.\n\\\]\n\nIf the iteration limit is reached without convergence, a warning is\nissued.\n\nAfter convergence, \\(z\\), \\(C(z)\\), and \\(S(z)\\) are recomputed, and the\nLagrange coefficients are\n\n\\\[\nf = 1 - \\frac{\\chi^2}{r\_0}C,\n\\qquad\ng = \\Delta t - \\frac{\\chi^3}{\\sqrt{\\mu}}S.\n\\\]\n\nThe propagated position vector and its magnitude are\n\n\\\[\n\\mathbf{r} = f\\mathbf{r}\_0 + g\\mathbf{v}\_0,\n\\qquad\nr = \\|\\mathbf{r}\\|.\n\\\]\n\nThe time derivatives of the Lagrange coefficients are\n\n\\\[\n\\dot{f} = \\frac{\\sqrt{\\mu}}{rr\_0}\n\\left(\\alpha\\chi^3 S - \\chi\\right),\n\\qquad\n\\dot{g} = 1 - \\frac{\\chi^2}{r}C.\n\\\]\n\nFinally, the propagated velocity vector is\n\n\\\[\n\\mathbf{v} = \\dot{f}\\mathbf{r}\_0 + \\dot{g}\\mathbf{v}\_0.\n\\\]\n\nThe function returns the propagated state and the iteration count\n\n\\\[\n\\mathbf{r}, \\qquad \\mathbf{v}, \\qquad \\mathrm{iter}.\n\\\]$
%[text] 
%[text] 
%[text] 
%[text] 

%[appendix]{"version":"1.0"}
%---
