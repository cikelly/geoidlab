############################################################
# Utilities for topographic density models                 #
# Copyright (c) 2026, Caleb Kelly                          #
# Author: Caleb Kelly  (2026)                              #
############################################################


import xarray as xr
import numpy as np
import zipfile
import pandas as pd
from pathlib import Path
import tempfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import os


MODELS = {
    '1d': '1d_UNB_TopoDens_2v02.zip',
    '5m': '5m_UNB_TopoDens_2v02.zip',
    '30s': '30s_UNB_TopoDens_2v02.zip'
}


def get_processed_density_path(
    download_dir: str | Path,
    model: str | None = None,
    resolution: int | None = None,
    resolution_unit: str | None = None,
) -> Path:
    '''
    Return the canonical processed density NetCDF path for a given model selection.
    '''
    download_dir = Path(download_dir)
    filename = _resolve_model_filename(
        model=model,
        resolution=resolution,
        resolution_unit=resolution_unit,
    )
    return download_dir / f"density_{Path(filename).stem}.nc"


def _resolve_model_filename(
    model: str | None = None,
    resolution: int | None = None,
    resolution_unit: str | None = None,
) -> str:
    '''
    Resolve the UNB model filename from either explicit model name or resolution hint.
    '''
    if model is None:
        if resolution is None:
            model_key = '30s'
        else:
            unit = (resolution_unit or 's').lower()
            model_key = f"{resolution}{unit}"
            if model_key not in MODELS:
                print(f"{model_key} not in MODELS; falling back on default 30s model.")
                model_key = '30s'
    else:
        model_key = model

    return MODELS[model_key] if model_key in MODELS else model_key


def _extract_lon_lat_coords(target) -> tuple[np.ndarray, np.ndarray]:
    '''
    Extract 1D lon/lat coordinate arrays from an xarray object.
    Accepts either x/y or lon/lat coordinate names.
    '''
    if isinstance(target, (xr.Dataset, xr.DataArray)):
        coords = target.coords
        if 'x' in coords and 'y' in coords:
            lon = target['x'].values
            lat = target['y'].values
            return np.asarray(lon), np.asarray(lat)
        if 'lon' in coords and 'lat' in coords:
            lon = target['lon'].values
            lat = target['lat'].values
            return np.asarray(lon), np.asarray(lat)
    raise ValueError("target_grid must provide 1D coordinates named ('x','y') or ('lon','lat').")


def _normalize_bbox(
    bbox: list | tuple,
    bbox_offset: float = 0.0,
) -> tuple[float, float, float, float]:
    '''
    Validate and normalize bbox as (W, E, S, N), applying optional offset.
    '''
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must be provided as [W, E, S, N].")
    w, e, s, n = [float(v) for v in bbox]
    if w > e or s > n:
        raise ValueError("Invalid bbox. Expected [W, E, S, N] with W<=E and S<=N.")
    off = float(bbox_offset or 0.0)
    w = max(-180.0, w - off)
    e = min(180.0, e + off)
    s = max(-90.0, s - off)
    n = min(90.0, n + off)
    return w, e, s, n


def load_unb_topo_density(file_path) -> xr.Dataset:
    '''
    Load UNB topographic density data from a text file and return as an xarray Dataset.
    '''
    def _tabular_to_dataset(data: pd.DataFrame) -> xr.Dataset:
        data = data.copy()
        data['lon'] = ((data.lon + 180) % 360) - 180
        grid = (data.pivot(index='lat', columns='lon', values='density')
                    .sort_index(ascending=True)
                    .sort_index(axis=1))
        da = xr.DataArray(
            grid.values,
            coords={'lat': grid.index, 'lon': grid.columns},
            dims=['lat', 'lon']
        )
        return da.to_dataset(name='density')

    def _normalize_grd_dataset(ds: xr.Dataset, source_name: str) -> xr.Dataset:
        ds = ds.copy()
        rename_map = {}
        if 'x' in ds.coords:
            rename_map['x'] = 'lon'
        if 'y' in ds.coords:
            rename_map['y'] = 'lat'
        if 'z' in ds.data_vars:
            rename_map['z'] = 'std' if 'STD' in source_name.upper() else 'density'
        if rename_map:
            ds = ds.rename(rename_map)

        if 'lon' not in ds.coords or 'lat' not in ds.coords:
            raise ValueError(
                f"Could not identify lon/lat coordinates in GRD dataset: {source_name}"
            )
        if 'density' not in ds.data_vars and 'std' not in ds.data_vars:
            raise ValueError(
                f"Could not identify density variable in GRD dataset: {source_name}"
            )

        lon = ((ds.lon + 180) % 360) - 180
        lon_sorted = np.sort(lon)
        ds = ds.assign_coords(lon=lon).reindex(lon=lon_sorted)
        return ds

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == '.zip':
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # List all files in the zip
                file_list = zf.namelist()
                print(f"Files in zip: {file_list}")
                
                # Check for .txt file first
                txt_files = [f for f in file_list if f.endswith('.txt')]
                xyz_files = [f for f in file_list if f.endswith('.xyz')]
                grd_files = [f for f in file_list if f.endswith('.grd')]
                
                data = None
                
                if txt_files:
                    try:
                        txt_file = txt_files[0]
                        print(f"Attempting to read .txt file: {txt_file}")
                        with zf.open(txt_file) as f:
                            data = pd.read_csv(f, sep=r'\s+', names=['lat', 'lon', 'density'])
                        print(f"Successfully read {txt_file}")
                    except Exception as e:
                        print(f"Failed to read .txt file: {e}")
                elif xyz_files:
                    try:
                        xyz_file = xyz_files[0]
                        print(f"Attempting to read .xyz file: {xyz_file}")
                        with zf.open(xyz_file) as f:
                            data = pd.read_csv(f, sep=r'\s+', names=['lat', 'lon', 'density'])
                        print(f"Successfully read {xyz_file}")
                    except Exception as e:
                        print(f"Failed to read .xyz file: {e}")
                elif grd_files:
                    ds_grd = []
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_dir = Path(tmp_dir)
                        for grd_file in grd_files:
                            try:
                                extracted = Path(zf.extract(grd_file, path=tmp_dir))
                                with xr.open_dataset(extracted) as ds_in:
                                    ds = _normalize_grd_dataset(ds_in.load(), grd_file)
                                ds_grd.append(ds)
                            except Exception as e:
                                print(f"Failed to read .grd file {grd_file}: {e}")
                    if ds_grd:
                        # Merge all .grd datasets if there are multiple
                        if len(ds_grd) > 1:
                            ds = xr.merge(ds_grd)
                        else:
                            ds = ds_grd[0]
                    return ds
                else:
                    raise Exception("No .txt, .xyz, or .grd files found in archive")
                
            if data is None:
                raise Exception("Failed to load data from the archive")
        
        except zipfile.BadZipFile:
            raise Exception("The file is not a valid zip archive")

        return _tabular_to_dataset(data)

    if suffix in {'.txt', '.xyz'}:
        data = pd.read_csv(file_path, sep=r'\s+', names=['lat', 'lon', 'density'])
        return _tabular_to_dataset(data)

    if suffix in {'.grd', '.nc'}:
        with xr.open_dataset(file_path) as ds_in:
            return _normalize_grd_dataset(ds_in.load(), str(file_path))

    raise ValueError(
        f"Unsupported density model format: {file_path.suffix}. "
        "Expected one of .zip, .txt, .xyz, .grd, .nc"
    )


def download_density_model(
    base_url: str = 'https://gge.ext.unb.ca/Resources/TopographicalDensity/', 
    model: str | None = None, 
    resolution: int | None = None, 
    resolution_unit: str | None = None,
    download_dir: Path = '.'
) -> Path:
    '''
    Download one of the UNB topographic‑density models.
    
    If the caller does *not* supply a `model` string, the function will
    default to the 30‑second product (`MODELS['30s']`).  A user may also
    supply a `resolution`/`resolution_unit` pair, but that is treated as an
    advisory hint only; the model name itself is ultimately determined by
    the `MODELS` dictionary.  Passing an explicit `model` overrides all
    defaults.
    
    Parameters
    ----------
    base_url : str
        Base URL for the UNB topographic density data.
    model : str or None
        Exact file name (including ".zip").  If provided, this string is
        concatenated with ``base_url`` and downloaded verbatim.  If ``None``
        the function chooses a model based on ``resolution``/``resolution_unit``
        or defaults to the 30‑second file.
    resolution : int or None
        Numeric resolution of the desired model (e.g. ``30``).  Ignored if
        ``model`` is given; if the combination
        ``f"{resolution}{resolution_unit}"`` does not match a key in
        ``MODELS`` a ``ValueError`` is raised.
    resolution_unit : str or None
        Unit designator for the resolution: ``'d'`` degrees, ``'m'`` minutes,
        ``'s'`` seconds.  Defaults to ``'s'`` if omitted when ``resolution`` is
        supplied.
    download_dir : Path
        Directory to download the model to.
    '''
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    filename = _resolve_model_filename(
        model=model,
        resolution=resolution,
        resolution_unit=resolution_unit,
    )
    
    # defer existence/validation checks to the robust downloader below
    
    url = f"{base_url}{filename}"
    
    # perform the actual download using requests with retries, resume support, and a progress bar
    dest = download_dir / filename
    # configure a session with retries
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    # ask the server for file size and resume capability
    head = session.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    total = int(head.headers.get('content-length') or 0)
    accept_ranges = head.headers.get('Accept-Ranges', '').lower() == 'bytes'

    def validate_file(path: os.PathLike) -> bool:
        try:
            if total and path.stat().st_size != total:
                return False
            # simple validation for expected zip archive
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    # If file exists, check size and validity
    if dest.exists():
        local_size = dest.stat().st_size
        if total and local_size == total and validate_file(dest):
            print(f'{filename} already exists and appears complete. Skipping download.')
            return dest
        # try to resume if server supports it and partial file is smaller
        if accept_ranges and local_size < total:
            print(f'Resuming download for {filename} from {local_size} bytes...')
            headers = {'Range': f'bytes={local_size}-'}
            with session.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dest, 'ab') as f, tqdm(total=total, initial=local_size, unit='B', unit_scale=True, desc=filename) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            # validate after resume
            if validate_file(dest):
                print('Download complete (resumed and validated).')
                return dest
            else:
                print('Resumed file failed validation; removing and re-downloading.')
                try:
                    dest.unlink()
                except Exception:
                    pass
        else:
            # cannot resume or file larger than expected; remove and re-download
            print(f'Existing file {filename} is incomplete or invalid. Re-downloading.')
            try:
                dest.unlink()
            except Exception:
                pass

    # fresh download
    with session.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        # content-length may be missing
        total_now = int(r.headers.get('content-length') or total or 0)
        with open(dest, 'wb') as f, tqdm(total=total_now, unit='B', unit_scale=True, desc=filename) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    # final validation
    if validate_file(dest):
        print('Download complete and validated.')
        return dest
    else:
        # try one more time: remove and download without resume
        print('Downloaded file failed validation; retrying once.')
        try:
            dest.unlink()
        except Exception:
            pass
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_now = int(r.headers.get('content-length') or total or 0)
            with open(dest, 'wb') as f, tqdm(total=total_now, unit='B', unit_scale=True, desc=filename) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        if validate_file(dest):
            print('Download complete and validated.')
            return dest
        raise Exception('Failed to download a valid file after retrying.')


def ingest_unb_density(
    base_url: str = 'https://gge.ext.unb.ca/Resources/TopographicalDensity/', 
    model: str | None = None, 
    resolution: int | None = None, 
    resolution_unit: str | None = None,
    download_dir: Path | None = None,
    bbox: list | tuple | None = None,
    bbox_offset: float = 0.0,
    target_grid: xr.Dataset | xr.DataArray | None = None,
    align: bool = True,
    interp_method: str = 'nearest',
    keep_dataset: bool = False,
    unit: str = 'kg/m3',
    save: bool = False,
) -> xr.Dataset:
    '''
    Download/load UNB topographic density and prepare it for GeoidLab ingestion.
    
    Parameters
    ----------
    base_url : str
        Base URL for the UNB topographic density data.
    model : str or None
        Exact file name (including ".zip").  If provided, this string is
        concatenated with ``base_url`` and downloaded verbatim.  If ``None``
        the function chooses a model based on ``resolution``/``resolution_unit``
        or defaults to the 30-second file.
    resolution : int or None
        Numeric resolution of the desired model (e.g. ``30``).  Ignored if
        ``model`` is given; if the combination
        ``f"{resolution}{resolution_unit}"`` does not match a key in
        ``MODELS`` a ``ValueError`` is raised.
    resolution_unit : str or None
        Unit designator for the resolution: ``'d'`` degrees, ``'m'`` minutes,
        ``'s'`` seconds.  Defaults to ``'s'`` if omitted when ``resolution`` is
        supplied.
    download_dir : Path
        Directory to download the model to. If None, defaults to cwd/downloads.
    bbox : list or None
        Bounding box (W, E, S, N) for subsetting.
    bbox_offset : float
        Optional offset (degrees) applied to all bbox edges.
    target_grid : xr.Dataset | xr.DataArray | None
        Target grid whose coordinates are used for alignment. Must provide
        either x/y or lon/lat coordinates.
    align : bool
        If True, interpolate to target_grid coordinates.
    interp_method : str
        Interpolation method for xarray.interp.
    keep_dataset : bool
        If True return full Dataset. If False return Dataset with only
        density variable.
    unit : str
        Desired output unit for density. Supported values: 'kg/m3', 'g/cm3'.
    save : bool
        If True, save the processed dataset to a NetCDF file in the download_dir.

    Returns
    -------
    xr.Dataset
        Topographic density model, optionally subset and aligned to target grid.
    '''
    if download_dir is None:
        download_dir = Path.cwd() / 'downloads'
    else:
        download_dir = Path(download_dir)

    # Ensure single source of truth for model selection and on-disk location.
    file_path = download_density_model(
        base_url=base_url,
        model=model,
        resolution=resolution,
        resolution_unit=resolution_unit,
        download_dir=download_dir,
    )
    ds = load_unb_topo_density(file_path)

    # Determine AOI from explicit bbox or target grid.
    if bbox is None and target_grid is not None:
        lon_t, lat_t = _extract_lon_lat_coords(target_grid)
        bbox = [np.nanmin(lon_t), np.nanmax(lon_t), np.nanmin(lat_t), np.nanmax(lat_t)]

    if bbox is not None:
        w, e, s, n = _normalize_bbox(bbox=bbox, bbox_offset=bbox_offset)
        ds = ds.sel(lon=slice(w, e), lat=slice(s, n))

    if align:
        if target_grid is None:
            raise ValueError("target_grid must be provided when align=True.")
        lon_t, lat_t = _extract_lon_lat_coords(target_grid)
        ds = ds.interp(
            lon=xr.DataArray(lon_t, dims=('lon',)),
            lat=xr.DataArray(lat_t, dims=('lat',)),
            method=interp_method,
        )

    if unit == 'kg/m3':
        # Convert from g/cm³ to kg/m³
        ds['density'] = ds['density'] * 1000.0
        if 'std' in ds.data_vars:
            ds['std'] = ds['std'] * 1000.0
    
    ds.attrs['source'] = f"UNB Topographic Density Model ({file_path.name})"
    ds.attrs['units'] = unit if unit in {'kg/m3', 'g/cm3'} else 'g/cm3'
    
    ds_out = ds[['density']] if (not keep_dataset and 'density' in ds.data_vars) else ds

    if save:
        output_path = get_processed_density_path(
            download_dir=download_dir,
            model=model,
            resolution=resolution,
            resolution_unit=resolution_unit,
        )
        ds_out.to_netcdf(output_path)
        print(f"Processed density saved to {output_path}")

    return ds_out
