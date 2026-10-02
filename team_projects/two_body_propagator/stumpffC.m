function C = stumpffC(z)
% handles elliptic and hyperbolic cases. z<0 and z>0 and z=0
if z > 1e-6
%[text] $\\\[\nz = \\alpha\\chi^2 \> 0\n\\qquad\n\\text{corresponds to an elliptical orbit},\n\\\]\n\nwhere\n\n\\\[\n\\alpha = \\frac{1}{a} \> 0\n\\\]\n\nfor a bound orbit. The closed-form definition is\n\n\\\[\nC(z) = \\frac{1-\\cos\\left(\\sqrt{z}\\right)}{z}.\n\\\]$
    C = (1-cos(sqrt(z)))/z; 
elseif z< -1e-6
%[text] $\\\[\nz \< 0\n\\qquad\n\\text{corresponds to a hyperbolic orbit }(\\alpha \< 0,\\ \\text{unbound}).\n\\\]\n\n hyperbolic cosine in place of cosine, since \\(\\sqrt{z}\\) would be imaginary:\n\n\\\[\nC(z) = \\frac{\\cosh\\left(\\sqrt{-z}\\right)-1}{-z}.\n\\\]$
    C = (cosh(sqrt(-z))-1) / (-z);
else
%[text] $\\\[\n|z| \< 10^{-6}\n\\\]\n\nmeans that \\(z\\) is very close to zero, which is the parabolic boundary.\nThe equations for \\(C(z)\\) become \\(0/0\\) at \\(z=0\\),\nso they are unreliable when \\(z\\) is very small.\n\n\\\[\\\]\nInstead, we use the Taylor series of \\(C(z)\\) around \\(z=0\\):\n\n\n\\\[\nC(z) \\approx \\frac{1}{2} - \\frac{z}{24} + \\frac{z^2}{720}.\n\\\]\n\nFor \\(|z| \< 10^{-6}\\), the higher-order terms are extremely small,\nso this approximation is sufficiently accurate.$
    C = 1/2 -z/24 + z^2/72- - z^3/40320;
end
end 
%[text] $&dollar&;&dollar&;\n\\text{Stumpff function } C(z)\n&dollar&;&dollar&;\n\nThe Stumpff function \\(C(z)\\) is used in the formulation of the two-body problem. It handles the elliptic, hyperbolic, and parabolic cases.\n\n&dollar&;&dollar&;\nC(z) =\n\\begin{cases}\n\\displaystyle\n\\frac{1-\\cos\\left(\\sqrt{z}\\right)}\n{z},\n& z \> 10^{-6}\n\\\\\[12pt\]\n\\displaystyle\n\\frac{\\cosh\\left(\\sqrt{-z}\\right)-1}\n{-z},\n& z \< -10^{-6}\n\\\\\[12pt\]\n\\displaystyle\n\\frac{1}{2}\n-\\frac{z}{24}\n+\\frac{z^2}{720}\n-\\frac{z^3}{40320},\n& |z| \\leq 10^{-6}\n\\end{cases}\n&dollar&;&dollar&;\n\nFor \\(z\>10^{-6}\\), the orbit is treated as elliptical. For \\(z\<-10^{-6}\\), the orbit is treated as hyperbolic.\nWhen \\(z\\) is close to zero, the Taylor series is used to avoid the numerical \\(0/0\\) singularity.\n$

%[appendix]{"version":"1.0"}
%---
