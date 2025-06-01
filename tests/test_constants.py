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
    assert 'a' in wgs84  # semi-major axis
    assert 'f' in wgs84  # flattening
    assert 'GM' in wgs84  # geocentric gravitational constant
    assert 'omega' in wgs84  # angular velocity

def test_grs80_constants() -> None:
    grs80 = constants.grs80()
    assert isinstance(grs80, dict)
    assert 'a' in grs80
    assert 'f' in grs80
    assert 'GM' in grs80
    assert 'omega' in grs80
