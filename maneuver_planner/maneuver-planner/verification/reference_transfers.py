"""Speed-ratio and helper-based reference equations; never used by production tools."""

import numpy as np


def hohmann_reference(r1, r2, mu):
    vi, vf = np.sqrt(mu/r1), np.sqrt(mu/r2)
    return (vi*np.sqrt(2*r2/(r1+r2))-vi,
            vf-vf*np.sqrt(2*r1/(r1+r2)))


def bielliptic_reference(r1, r2, rb, mu):
    def vis_viva(radius, axis):
        return np.sqrt(mu*(2/radius-1/axis))
    a1, a2 = (r1+rb)/2, (r2+rb)/2
    dv = (abs(vis_viva(r1, a1)-np.sqrt(mu/r1)),
          abs(vis_viva(rb, a2)-vis_viva(rb, a1)),
          abs(np.sqrt(mu/r2)-vis_viva(r2, a2)))
    return sum(dv), np.pi*(np.sqrt(a1**3/mu)+np.sqrt(a2**3/mu))
