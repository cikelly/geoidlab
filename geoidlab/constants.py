############################################################
# Constants for gravity field modelling                    #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################
import json
import math


ELLIPSOID_KEYS = [
    'semi_major',
    'semi_minor',
    'GM',
    'J2',
    'w',
    'E',
    'c',
    'e',
    'e2',
    'ep',
    'ep2',
    'f',
    '1/f',
    'U0',
    'J4',
    'J6',
    'J8',
    'm',
    'gamma_a',
    'gamma_b',
    'mean_gamma',
    'C20',
    'C40',
    'C60',
    'C80',
    'C100',
]


def _nan() -> float:
    return float('nan')


def ellipsoid_template() -> dict:
    '''
    Template for custom reference ellipsoids.

    Returns
    -------
    dict
        All known ellipsoid keys initialized to NaN.
    '''
    return {key: _nan() for key in ELLIPSOID_KEYS}


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def require_ellipsoid_params(ellipsoid: dict, required: list[str], context: str = 'this computation') -> None:
    '''
    Validate that the required ellipsoid parameters are present and non-NaN.
    '''
    missing = [k for k in required if k not in ellipsoid or _is_missing(ellipsoid[k])]
    if missing:
        raise ValueError(
            f"Missing required ellipsoid parameter(s) for {context}: {missing}. "
            "Provide these values in your custom ellipsoid definition."
        )


def grs80() -> dict:
    '''
    GRS 1980 reference ellipsoid parameters

    Returns
    -------
    dict
    '''
    grs80 = {
        'semi_major': 6_378_137,
        'semi_minor': 6_356_752.3141,
        'GM'        : 3_986_005e8,
        'J2'        : 108_263e-8,
        'w'         : 7_292_115e-11,
        'E'         : 521_854.0097,
        'c'         : 6_399_593.6259,
        'e2'        : 0.00669438002290,
        'ep2'       : 0.00673949677548,
        'f'         : 0.003352810681,
        '1/f'       : 298.257222101,
        'U0'        : 62_636_860.850,
        'J4'        : -0.00000237091222,
        'J6'        : 0.00000000608347,
        'J8'        : -0.00000000001427,
        'm'         : 0.00344978600308,
        'gamma_a'   : 9.7803267715,
        'gamma_b'   : 9.8321863685,
        'C20'       : -0.484166854903603e-03,
        'C40'       : 0.790304072916597e-06,
        'C60'       : -0.168725117581045e-08,
        'C80'       : 0.346053239866698e-11,
        'C100'      : -0.265006218130312e-14,
    }
    return grs80



def wgs84() -> dict:
    '''
    WGS 1984 reference ellipsoid parameters

    Returns
    -------
    dict
    '''
    wgs84 = {
        'semi_major': 6_378_137,
        'semi_minor': 6_356_752.3142,
        'GM'        : 3_986_004.418e8,
        'w'         : 7_292_115e-11,
        'E'         : 5.2185400842339e5,
        'c'         : 6_399_593.6258,
        'e'         : 8.1819190842622e-2,
        'e2'        : 6.69437999014e-3,
        'ep'        : 8.2094437949696e-2,
        'ep2'       : 6.73949674228e-3,
        'f'         : 1/298.257223563,
        'U0'        : 62_636_851.7146,
        'm'         : 0.00344978650684,
        'gamma_a'   : 9.7803253359,
        'gamma_b'   : 9.8321849378,
        'mean_gamma': 9.7976432222,
        'C20'       : -0.484166774985e-03,
        'C40'       : 0.790303733511e-06,
        'C60'       : -0.168724961151e-08,
        'C80'       : 0.346052468394e-11,
        'C100'      : -0.265002225747e-14,
    }
    return wgs84


def custom_ellipsoid(**kwargs) -> dict:
    '''
    Create a custom ellipsoid dictionary with all known parameters.

    Any unspecified parameter remains NaN and will trigger a clear error only
    when a downstream computation requires it.
    '''
    ellipsoid = ellipsoid_template()
    ellipsoid.update(kwargs)
    return ellipsoid


def resolve_ellipsoid(ellipsoid='wgs84') -> dict:
    '''
    Resolve an ellipsoid specification to a dictionary with all known keys.

    Supported inputs
    ----------------
    - 'wgs84' or 'grs80'
    - dict of parameters (partial allowed)
    - JSON object string with parameters (partial allowed)
    '''
    if isinstance(ellipsoid, dict):
        resolved = custom_ellipsoid(**ellipsoid)
    elif isinstance(ellipsoid, str):
        raw = ellipsoid.strip()
        key = raw.lower()
        if key == 'wgs84':
            resolved = wgs84()
        elif key == 'grs80':
            resolved = grs80()
        elif raw.startswith('{') and raw.endswith('}'):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid ellipsoid JSON string: {exc}') from exc
            if not isinstance(decoded, dict):
                raise TypeError('Custom ellipsoid JSON must decode to an object/dict')
            resolved = custom_ellipsoid(**decoded)
        else:
            raise ValueError(
                f"Unsupported ellipsoid '{ellipsoid}'. "
                "Use 'wgs84', 'grs80', a dict, or a JSON object string."
            )
    else:
        raise TypeError('ellipsoid must be a str or dict')

    return resolved


def earth() -> dict:
    '''
    Constants for Earth/Geoid
    '''
    earth = {
        'W0'        : 62_636_853.40,
        'radius'    : 6_371_000,
        'G'         : 6.67259e-11,
        'rho'       : 2670,
    }
    return earth
