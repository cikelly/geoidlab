############################################################
# Utilities for Curtin Earth2014 SHC and potential models  #
# Copyright (c) 2026, Caleb Kelly                          #
# Author: Caleb Kelly  (2026)                              #
############################################################

from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


CURTIN_EARTH2014_BASE_URL = "http://ddfe.curtin.edu.au/models/Earth2014/"
CURTIN_EARTH2014_POTENTIAL_URL = CURTIN_EARTH2014_BASE_URL + "potential_model/"
CURTIN_EARTH2014_SHCS_2160_URL = CURTIN_EARTH2014_BASE_URL + "data_5min/shcs_to2160/"
CURTIN_EARTH2014_SHCS_10800_URL = CURTIN_EARTH2014_BASE_URL + "data_1min/shcs_to10800/"
EARTH2014_SHC_MODELS = {
    'sur': 'SUR2014',
    'bed': 'BED2014',
    'tbi': 'TBI2014',
    'ret': 'RET2014',
    'ice': 'ICE2014',
}


def _list_bshc_files(base_url: str, timeout: int = 30) -> list[str]:
    response = requests.get(base_url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    files: list[str] = []
    for link in soup.find_all('a', href=True):
        href = link['href'].strip()
        if href.lower().endswith('.bshc'):
            files.append(href.split('/')[-1])

    return sorted(set(files))


def _download_bshc_file(
    filename: str,
    output_dir: str | Path,
    base_url: str,
    overwrite: bool = False,
    timeout: int = 30,
    local_name: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / (local_name or filename)
    if out_file.exists() and not overwrite:
        return out_file

    url_suffix = filename + '.bshc' if not filename.endswith('.bshc') else filename
    file_url = base_url.rstrip('/') + '/' + url_suffix
    response = requests.get(file_url, stream=True, timeout=timeout)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024

    with tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename) as pbar:
        with open(out_file, 'wb') as f:
            for chunk in response.iter_content(block_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    return out_file


def earth2014_shc_url(resolution: str = '5min') -> str:
    '''
    Return the Earth2014 SHC directory URL for supported grid-equivalent resolutions.
    '''
    resolution = resolution.lower()
    if resolution == '5min':
        return CURTIN_EARTH2014_SHCS_2160_URL
    if resolution == '1min':
        return CURTIN_EARTH2014_SHCS_10800_URL
    raise ValueError("resolution must be '5min' or '1min'")


def earth2014_shc_filename(model: str = 'bed', resolution: str = '5min') -> str:
    '''
    Build a canonical Earth2014 SHC filename for relief synthesis.

    Parameters
    ----------
    model : {'sur', 'bed', 'tbi', 'ret', 'ice'}
        Earth2014 relief/shape model selector:
        `sur` = surface topography,
        `bed` = bedrock topography,
        `tbi` = topography-bathymetry with ice,
        `ret` = rock-equivalent topography,
        `ice` = ice thickness.
        The default `bed` is often the most useful reference-topography surface for geodetic
        terrain applications.
    resolution : {'5min', '1min'}
        Earth2014 SHC family. `5min` maps to degree 2160 and `1min` maps to degree 10800.
    '''
    model_key = model.lower()
    if model_key not in EARTH2014_SHC_MODELS:
        raise ValueError(
            f"Unsupported Earth2014 SHC model '{model}'. "
            f"Choose from {sorted(EARTH2014_SHC_MODELS)}."
        )

    resolution = resolution.lower()
    if resolution == '5min':
        degree = 2160
    elif resolution == '1min':
        degree = 10800
    else:
        raise ValueError("resolution must be '5min' or '1min'")

    return f"Earth2014.{EARTH2014_SHC_MODELS[model_key]}.degree{degree}.bshc"


def list_shc_models(resolution: str = '5min', timeout: int = 30) -> list[str]:
    '''
    List available Earth2014 surface/shape SHC files for relief synthesis.

    Notes
    -----
    These are the SHC products shown in Curtin's `access_Earth2014_shcs2160.m`
    and `access_Earth2014_shcs10800.m` examples, e.g. SUR/BED/ICE/RET/TBI.
    '''
    return _list_bshc_files(earth2014_shc_url(resolution=resolution), timeout=timeout)


def download_shc_model(
    filename: str | None = None,
    output_dir: str | Path = 'downloads',
    model: str = 'bed',
    resolution: str = '5min',
    overwrite: bool = False,
    timeout: int = 30,
) -> Path:
    '''
    Download a user-specified Earth2014 surface/shape SHC model from Curtin.

    Parameters
    ----------
    filename : str, optional
        Explicit Earth2014 `.bshc` filename. If omitted, a canonical filename is built from
        `model` and `resolution`.
    model : {'sur', 'bed', 'tbi', 'ret', 'ice'}
        Earth2014 relief/shape model selector used when `filename` is omitted:
        `sur` = surface topography,
        `bed` = bedrock topography,
        `tbi` = topography-bathymetry with ice,
        `ret` = rock-equivalent topography,
        `ice` = ice thickness.
        The default is `bed`.
    resolution : {'5min', '1min'}
        Earth2014 SHC family. `5min` maps to degree 2160 and `1min` maps to degree 10800.
    '''
    if filename is None:
        filename = earth2014_shc_filename(model=model, resolution=resolution)
    else:
        filename = filename if filename.endswith('.bshc') else f'{filename}.bshc'
        filename = filename.replace('.degree2160.bshc', '.bshc').replace('.degree10800.bshc', '.bshc')
        if resolution == '5min':
            filename = filename.replace('.bshc', '.degree2160.bshc')
        elif resolution == '1min':
            filename = filename.replace('.bshc', '.degree10800.bshc')
        else:
            raise ValueError("resolution must be '5min' or '1min'")

    return _download_bshc_file(
        filename=filename,
        output_dir=output_dir,
        base_url=earth2014_shc_url(resolution=resolution),
        overwrite=overwrite,
        timeout=timeout,
    )


def is_potential_model_filename(filename: str | Path) -> bool:
    '''
    Return True when the filename matches Curtin's `dV_*` potential-model naming.
    '''
    return Path(filename).name.lower().startswith('dv_')


def list_potential_models(base_url: str = CURTIN_EARTH2014_POTENTIAL_URL, timeout: int = 30) -> list[str]:
    '''
    List available Earth2014 potential model files from the Curtin server.
    '''
    return _list_bshc_files(base_url=base_url, timeout=timeout)


def download_potential_model(
    filename: str,
    output_dir: str | Path = 'downloads',
    base_url: str = CURTIN_EARTH2014_POTENTIAL_URL,
    overwrite: bool = False,
    timeout: int = 30,
) -> Path:
    '''
    Download a user-specified Earth2014 potential model from Curtin.
    '''
    return _download_bshc_file(
        filename=filename,
        output_dir=output_dir,
        base_url=base_url,
        overwrite=overwrite,
        timeout=timeout,
    )


def read_bshc_coefficients(filename: str | Path, nmax: int | None = None) -> dict[str, np.ndarray | int]:
    '''
    Read Curtin `.bshc` coefficients into GeoidLab-style HCnm/HSnm triangle arrays.

    Notes
    -----
    Based on the Curtin/SHTOOLS `bshc` format:
    - data are IEEE float64 (`double`) in little-endian byte order
    - record layout:
      n_min, n_max, C(n,m) block (ascending n,m), S(n,m) block (ascending n,m)
    '''
    filename = Path(filename)
    raw = None
    nmin = None
    nmax_file = None
    for dtype in ('<f8', '>f8'):
        candidate = np.fromfile(filename, dtype=dtype)
        if candidate.size < 2:
            continue
        nmin_candidate = int(candidate[0])
        nmax_candidate = int(candidate[1])
        if 0 <= nmin_candidate <= nmax_candidate <= 30000:
            raw = candidate
            nmin = nmin_candidate
            nmax_file = nmax_candidate
            break

    if raw is None or nmin is None or nmax_file is None:
        raise ValueError(
            f'Invalid .bshc file header for {filename}. '
            'Expected little-endian or big-endian float64 header with 0 <= nmin <= nmax.'
        )

    if nmax is None:
        nmax_use = nmax_file
    else:
        nmax_use = min(int(nmax), nmax_file)

    if nmax_use < nmin:
        raise ValueError(
            f'Requested nmax={nmax_use} is below file nmin={nmin} for {filename}'
        )

    # Number of terms from nmin..nmax_file in triangular storage.
    n_down = ((nmin - 1) + 1) * ((nmin - 1) + 2) // 2 if nmin > 0 else 0
    n_up = (nmax_file + 1) * (nmax_file + 2) // 2
    n_rows = n_up - n_down
    offset = 2

    c_block = raw[offset:offset + n_rows]
    s_block = raw[offset + n_rows:offset + 2 * n_rows]
    if c_block.size != n_rows or s_block.size != n_rows:
        raise ValueError(f'Unexpected .bshc size for {filename}')

    hcnm = np.zeros((nmax_use + 1, nmax_use + 1))
    hsnm = np.zeros((nmax_use + 1, nmax_use + 1))

    k = 0
    for n in range(nmin, nmax_use + 1):
        for m in range(n + 1):
            hcnm[n, m] = float(c_block[k])
            hsnm[n, m] = float(s_block[k])
            k += 1

    if not np.isfinite(hcnm).all() or not np.isfinite(hsnm).all():
        raise ValueError(
            f'Non-finite coefficients detected while reading {filename}. '
            'The file may be corrupt or encoded in an unexpected format.'
        )

    return {
        'HCnm': hcnm,
        'HSnm': hsnm,
        'nmin': nmin,
        'nmax_file': nmax_file,
        'nmax': nmax_use,
    }
