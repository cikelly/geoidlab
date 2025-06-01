import pytest
import numpy as np
from geoidlab import constants

def test_earth_constants() -> None:
    earth = constants.earth()
    assert isinstance(earth, dict)
    assert 'W0' in earth
    assert 'radius' in earth
    assert 'G' in earth
    assert 'rho' in earth
    assert earth['radius'] == 6_371_000  # mean Earth radius in meters

def test_wgs84_constants() -> None:
    wgs84 = constants.wgs84()
    assert isinstance(wgs84, dict)
    assert 'semi_major' in wgs84  # semi-major axis
    assert 'f' in wgs84  # flattening
    assert 'GM' in wgs84  # geocentric gravitational constant
    assert 'w' in wgs84  # angular velocity
    assert wgs84['semi_major'] == 6_378_137  # Check actual value
    assert abs(wgs84['e2'] - 6.69437999014e-3) < 1e-12  # Check eccentricity squared

def test_grs80_constants() -> None:
    grs80 = constants.grs80()
    assert isinstance(grs80, dict)
    assert 'semi_major' in grs80  # semi-major axis
    assert 'f' in grs80  # flattening
    assert 'GM' in grs80  # geocentric gravitational constant
    assert 'w' in grs80  # angular velocity
    assert grs80['semi_major'] == 6_378_137  # Check actual value
    assert abs(grs80['e2'] - 0.00669438002290) < 1e-12  # Check eccentricity squared
