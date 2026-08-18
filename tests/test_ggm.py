import numpy as np
import pytest

from geoidlab import constants, gravity
from geoidlab.ggm import GlobalGeopotentialModel


def _zero_degree_model(c00: float = 1.0) -> GlobalGeopotentialModel:
    model = GlobalGeopotentialModel.__new__(GlobalGeopotentialModel)
    model.ellipsoid = 'grs80'
    model.lat = np.array([0.0, 45.0])
    model.h = np.zeros(2)
    model.r = np.full(2, constants.grs80()['semi_major'])
    model.shc = {
        'GM': constants.grs80()['GM'] + 100.0,
        'Cnm': np.array([[c00]]),
    }
    return model


def test_zero_degree_modes_use_c00_and_keep_physical_w0_default() -> None:
    model = _zero_degree_model(c00=0.5)
    geoid = np.zeros(2)
    ellipsoid = constants.grs80()
    gamma = gravity.normal_gravity_somigliana(model.lat, ellipsoid='grs80')
    gm_model = model.shc['GM'] * 0.5

    expected_icgem = (gm_model - ellipsoid['GM']) / ellipsoid['semi_major'] / gamma
    expected_physical = expected_icgem - (
        constants.earth()['W0'] - ellipsoid['U0']
    ) / gamma

    assert np.allclose(model.zero_degree_term(geoid), expected_physical)
    assert np.allclose(
        model.zero_degree_term(geoid, zero_degree_mode='physical_w0'),
        expected_physical,
    )
    assert np.allclose(
        model.zero_degree_term(geoid, zero_degree_mode='icgem'), expected_icgem
    )


def test_zero_degree_none_returns_input_unchanged() -> None:
    model = _zero_degree_model()
    geoid = np.array([1.25, -3.5])

    result = model.zero_degree_term(geoid, zero_degree_mode='none')

    assert result is geoid


def test_zero_degree_mode_and_invalid_c00_fallback() -> None:
    model = _zero_degree_model()

    with pytest.raises(ValueError, match='zero_degree_mode'):
        model.zero_degree_term(np.zeros(2), zero_degree_mode='invalid')

    model.shc['Cnm'][0, 0] = np.nan
    expected = _zero_degree_model().zero_degree_term(
        np.zeros(2), zero_degree_mode='icgem'
    )
    assert np.allclose(
        model.zero_degree_term(np.zeros(2), zero_degree_mode='icgem'), expected
    )

    model.shc['Cnm'] = np.empty((0, 0))
    assert np.allclose(
        model.zero_degree_term(np.zeros(2), zero_degree_mode='icgem'), expected
    )
