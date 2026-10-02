function S = stumpffS(z)
% handles elliptic z>0 and hyperbolic z<0

if z> 1e-6
%[text] $\\\[\n\\text{Elliptical case: } z = \\alpha\\chi^2 \> 0.\n\\\]$
sz = sqrt(z);
S = (sz - sin(sz)) / sz^3;
%[text] $\\\[\nS(z) =\n\\frac{\\sqrt{z} - \\sin\\left(\\sqrt{z}\\right)}\n{\\left(\\sqrt{z}\\right)^3}.\n\\\]$
elseif z< -1e-6
%[text] $\\mathrm{hyperbolic}\\;\\mathrm{case}\\;z\<0${"editStyle":"visual"}
    sz = sqrt(-z);
%[text] $z\\;\\mathrm{is}\\;\\mathrm{negative}\\;\\mathrm{to}\\;\\mathrm{avoid}\\;\\mathrm{sqrt}\\;\\mathrm{of}\\;\\mathrm{negative}\\;\\mathrm{number}${"editStyle":"visual"}
    S = (sinh(sz)-sz)/ sz^3;
%[text] $\\\[\nS(z) =\n\\frac{\\sinh\\left(\\sqrt{-z}\\right) - \\sqrt{-z}}\n{\\left(\\sqrt{-z}\\right)^3}\n\\\]$
else
%[text] $\\mathrm{Near}\\;\\mathrm{parabolic}\\;\\mathrm{case}\\;|z|\\le 1e-6\\;${"editStyle":"visual"}
S = 1/6 - z/120 + z^2/5040 - z^3/362880;
%[text] $\\\[\nS(z) = \\frac{1}{6} - \\frac{z}{120} + \\frac{z^2}{5040} - \\frac{z^3}{362880}\n\\\]$
end
end
%[text] $&dollar&;&dollar&;\n\\text{Stumpff function } S(z)\n&dollar&;&dollar&;\n\nThe Stumpff function \\(S(z)\\) is used in the universal-variable formulation of the two-body problem. It handles the elliptic, hyperbolic, and parabolic cases.\n\n&dollar&;&dollar&;\nS(z) =\n\\begin{cases}\n\\displaystyle\n\\frac{\\sqrt{z}-\\sin\\left(\\sqrt{z}\\right)}\n{\\left(\\sqrt{z}\\right)^3},\n& z \> 10^{-6}\n\\\\\[12pt\]\n\\displaystyle\n\\frac{\\sinh\\left(\\sqrt{-z}\\right)-\\sqrt{-z}}\n{\\left(\\sqrt{-z}\\right)^3},\n& z \< -10^{-6}\n\\\\\[12pt\]\n\\displaystyle\n\\frac{1}{6}\n-\\frac{z}{120}\n+\\frac{z^2}{5040}\n-\\frac{z^3}{362880},\n& |z| \\leq 10^{-6}\n\\end{cases}\n&dollar&;&dollar&;\n\nFor \\(z\>10^{-6}\\), the orbit is treated as elliptical. For \\(z\<-10^{-6}\\), the orbit is treated as hyperbolic. When \\(z\\) is close to zero, the Taylor series is used to avoid the numerical \\(0/0\\) singularity of the closed-form equations.\n$

%[appendix]{"version":"1.0"}
%---
