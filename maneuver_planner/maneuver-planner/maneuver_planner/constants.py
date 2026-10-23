"""Named environment presets. Algorithms must accept a body instead of importing EARTH."""

from .models import CentralBody

# WGS 84 constants: https://earth-info.nga.mil/index.php?action=wgs84&dir=wgs84
# The equatorial radius is used as a spherical altitude reference, not an ellipsoid.
EARTH = CentralBody(name="Earth", mu_m3_s2=3.986004418e14, radius_m=6_378_137.0)
