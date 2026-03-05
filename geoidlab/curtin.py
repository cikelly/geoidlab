############################################################
# Utilities for Curtin Earth2014 potential models          #
# Copyright (c) 2026, Caleb Kelly                          #
# Author: Caleb Kelly  (2026)                              #
############################################################

from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


CURTIN_EARTH2014_POTENTIAL_URL = "http://ddfe.curtin.edu.au/models/Earth2014/potential_model/"


def list_potential_models(base_url: str = CURTIN_EARTH2014_POTENTIAL_URL, timeout: int = 30) -> list[str]:
    '''
    List available Earth2014 potential model files from the Curtin server.
    '''
    response = requests.get(base_url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    files: list[str] = []
    for link in soup.find_all('a', href=True):
        href = link['href'].strip()
        if href.lower().endswith('.bshc'):
            files.append(href.split('/')[-1])

    return sorted(set(files))


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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / filename
    if out_file.exists() and not overwrite:
        return out_file

    file_url = base_url.rstrip('/') + '/' + filename
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


def read_bshc_coefficients(filename: str | Path, nmax: int | None = None) -> dict[str, np.ndarray | int]:
    '''
    Read Curtin `.bshc` coefficients into GeoidLab-style HCnm/HSnm triangle arrays.

    Notes
    -----
    Based on Curtin/TUM MATLAB reference:
    - data are IEEE float64 (`double`) in big-endian byte order
    - record layout:
      n_min, n_max, C(n,m) block (ascending n,m), S(n,m) block (ascending n,m)
    '''
    filename = Path(filename)
    raw = np.fromfile(filename, dtype='>f8')
    if raw.size < 2:
        raise ValueError(f'Invalid .bshc file: {filename}')

    nmin = int(raw[0])
    nmax_file = int(raw[1])

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

    return {
        'HCnm': hcnm,
        'HSnm': hsnm,
        'nmin': nmin,
        'nmax_file': nmax_file,
        'nmax': nmax_use,
    }
