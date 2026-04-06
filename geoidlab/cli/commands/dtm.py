############################################################
# Digital terrain model synthesis CLI interface            #
# Copyright (c) 2026, Caleb Kelly                          #
# Author: Caleb Kelly  (2026)                              #
############################################################

import pandas as pd
import numpy as np

from pathlib import Path

from geoidlab import curtin
from geoidlab import constants
from geoidlab.dtm import DigitalTerrainModel
from geoidlab.cli.commands.utils.common import directory_setup, get_grid_lon_lat
from geoidlab.utils.io import save_to_netcdf


class DTMSynthesis:
    '''
    CLI workflow for synthesizing terrain heights from spherical harmonic DTM models.
    '''
    def __init__(
        self,
        dtm_model: str | Path = None,
        model_format: str | None = None,
        earth2014_model: str | None = None,
        earth2014_resolution: str = '5min',
        input_file: str | Path = None,
        model_dir: str | Path = None,
        output_dir: str | Path = None,
        max_deg: int = 2190,
        ellipsoid: str = 'wgs84',
        ellipsoid_name: str | None = None,
        bbox: list[float] | None = None,
        grid_size: float | None = None,
        grid_unit: str = 'degrees',
        chunk_size: int = 800,
        chunk_memory_gb: float | None = None,
        n_workers: int | None = None,
        leg_progress: bool = False,
        threaded_legendre: bool = False,
        legendre_method: str = 'standard',
        proj_name: str = 'GeoidProject',
    ) -> None:
        self.dtm_model = Path(dtm_model) if dtm_model is not None else None
        self.model_format = model_format
        self.earth2014_model = earth2014_model
        self.earth2014_resolution = earth2014_resolution
        self.input_file = Path(input_file) if input_file is not None else None
        self.model_dir = Path(model_dir) if model_dir else Path(proj_name) / 'downloads'
        self.output_dir = Path(output_dir) if output_dir else Path(proj_name) / 'results'
        self.max_deg = max_deg
        self.ellipsoid = ellipsoid
        self.ellipsoid_name = ellipsoid_name
        self.bbox = bbox
        self.grid_size = grid_size
        self.grid_unit = grid_unit
        self.chunk_size = chunk_size
        self.chunk_memory_gb = chunk_memory_gb
        self.n_workers = n_workers
        self.leg_progress = leg_progress
        self.threaded_legendre = threaded_legendre
        self.legendre_method = legendre_method
        self.proj_name = proj_name

        directory_setup(proj_name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._validate_params()

    def _validate_params(self) -> None:
        constants.resolve_ellipsoid(self.ellipsoid)

        if self.max_deg <= 0:
            raise ValueError('max_deg must be greater than 0')
        if self.chunk_size <= 0:
            raise ValueError('chunk_size must be greater than 0')
        if self.chunk_memory_gb is not None and self.chunk_memory_gb <= 0:
            raise ValueError('chunk_memory_gb must be greater than 0')
        if self.grid_unit not in {'degrees', 'minutes', 'seconds'}:
            raise ValueError('grid_unit must be one of: degrees, minutes, seconds')
        if self.model_format not in {None, 'dtm2006_text', 'bshc'}:
            raise ValueError("model_format must be one of: dtm2006_text, bshc")
        if self.dtm_model is not None and self.earth2014_model is not None:
            raise ValueError('Specify at most one terrain SHC source: --dtm-model or --earth2014-model.')

        if self.input_file is None:
            if self.bbox is None or len(self.bbox) != 4 or any(v is None for v in self.bbox):
                raise ValueError('Provide either --input-file or a complete --bbox [W E S N].')
            if self.grid_size is None or self.grid_size <= 0:
                raise ValueError('grid_size must be provided and positive when synthesizing a grid.')

    def _resolve_model_path(self) -> Path | None:
        if self.earth2014_model is not None:
            return curtin.download_shc_model(
                model=self.earth2014_model,
                resolution=self.earth2014_resolution,
                output_dir=self.model_dir,
            )
        return self.dtm_model

    @staticmethod
    def _read_input_file(input_file: Path) -> pd.DataFrame:
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() in {'.xlsx', '.xls'}:
            df = pd.read_excel(input_file)
        elif input_file.suffix.lower() == '.txt':
            df = pd.read_csv(input_file, delimiter='\t')
        else:
            raise ValueError('Unsupported input format. Use csv, txt, xls, or xlsx.')

        normalized = {col.lower(): col for col in df.columns}
        lon_col = next((normalized[c] for c in normalized if c in {'lon', 'long', 'longitude', 'x'}), None)
        lat_col = next((normalized[c] for c in normalized if c in {'lat', 'lati', 'latitude', 'y'}), None)
        h_col = next((normalized[c] for c in normalized if c in {'h', 'height', 'elev', 'elevation', 'z'}), None)

        if lon_col is None or lat_col is None:
            raise ValueError('Input file must contain longitude and latitude columns.')

        out = pd.DataFrame({
            'lon': df[lon_col].values,
            'lat': df[lat_col].values,
        })
        out['height'] = df[h_col].values if h_col is not None else np.zeros(len(out))
        return out

    def _prepare_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...] | None]:
        if self.input_file is not None:
            points = self._read_input_file(self.input_file)
            return (
                points['lon'].to_numpy(),
                points['lat'].to_numpy(),
                points['height'].to_numpy(),
                None,
            )

        lon_grid, lat_grid = get_grid_lon_lat(self.bbox, self.grid_size, self.grid_unit)
        height = np.zeros_like(lon_grid)
        return lon_grid, lat_grid, height, lon_grid.shape

    def synthesize(self) -> dict:
        model_path = self._resolve_model_path()
        if model_path is None:
            model_label = 'bundled DTM2006.0'
        else:
            model_label = model_path.name

        print(
            f'Synthesizing terrain heights from {model_label} '
            f'with nmax={self.max_deg}, ellipsoid={self.ellipsoid}'
        )

        lon, lat, height, grid_shape = self._prepare_coordinates()
        dtm = DigitalTerrainModel(
            model_name=model_path,
            nmax=self.max_deg,
            ellipsoid=self.ellipsoid,
            model_format=self.model_format,
            threaded_legendre=self.threaded_legendre,
            legendre_method=self.legendre_method,
            chunk_memory_gb=self.chunk_memory_gb,
        )

        H = dtm.synthesize_height(
            lon=lon,
            lat=lat,
            height=height,
            chunk_size=self.chunk_size,
            leg_progress=self.leg_progress,
            n_workers=self.n_workers,
            save=False,
        )

        if grid_shape is None:
            output_file = self.output_dir / 'H_dtm.csv'
            df = pd.DataFrame({
                'lon': np.asarray(lon).ravel(),
                'lat': np.asarray(lat).ravel(),
                'H_dtm': np.asarray(H).ravel(),
            })
            df.to_csv(output_file, index=False)
        else:
            output_file = self.output_dir / 'H_dtm.nc'
            save_to_netcdf(
                data=np.asarray(H).reshape(grid_shape),
                lon=np.asarray(lon),
                lat=np.asarray(lat),
                dataset_key='H_dtm',
                filepath=output_file,
                ellipsoid=self.ellipsoid,
                ellipsoid_name=self.ellipsoid_name,
            )

        print(f'Synthesized terrain heights written to {output_file}')
        return {'status': 'success', 'output_file': str(output_file)}


def add_dtm_arguments(parser) -> None:
    parser.add_argument('--dtm-model', type=str, default=None,
                        help='Path to a terrain spherical harmonic model file. If omitted, the bundled DTM2006.0 model is used.')
    parser.add_argument('--model-format', type=str, default=None, choices=['dtm2006_text', 'bshc'],
                        help='Optional explicit model format.')
    parser.add_argument('--earth2014-model', type=str, default=None,
                        choices=['SUR2014', 'BED2014', 'RET2014', 'TBI2014', 'ICE2014'],
                        help='Curtin Earth2014 SHC model to download and synthesize.')
    parser.add_argument('--earth2014-resolution', type=str, default='5min',
                        help='Resolution of the Earth2014 SHC model to download.')
    parser.add_argument('-i', '--input-file', type=str, default=None,
                        help='Input file with lon/lat[/height]. If omitted, a grid is synthesized from --bbox and --grid-size.')
    parser.add_argument('-md', '--model-dir', type=str, default=None,
                        help='Directory for downloaded terrain SHC models.')
    parser.add_argument('-n', '--max-deg', type=int, default=2190,
                        help='Maximum degree of truncation.')
    parser.add_argument('-gs', '--grid-size', type=float, default=None,
                        help='Grid size in degrees, minutes, or seconds for gridded synthesis.')
    parser.add_argument('-gu', '--grid-unit', type=str, default='degrees',
                        choices=['degrees', 'minutes', 'seconds'],
                        help='Unit of grid size.')
    parser.add_argument('-b', '--bbox', type=float, nargs=4, default=[None, None, None, None],
                        help='Bounding box [W,E,S,N] in degrees for gridded synthesis.')
    parser.add_argument('-cs', '--chunk-size', type=int, default=800,
                        help='Chunk size for synthesis.')
    parser.add_argument('--chunk-memory-gb', type=float, default=None,
                        help='Maximum memory budget in GiB for one DTM chunk. If set, this budget is honored directly.')
    parser.add_argument('-w', '--workers', type=int, default=None,
                        help='Number of worker processes for chunked DTM synthesis.')
    parser.add_argument('--leg-progress', action='store_true', default=False,
                        help='Show Legendre progress for non-multiprocessing synthesis.')
    parser.add_argument('--threaded-legendre', action='store_true', default=False,
                        help='Use threaded Legendre generation for supported DTM kernels.')
    parser.add_argument('--legendre-method', type=str, default='standard', choices=['standard', 'holmes'],
                        help='ALF backend for DTM synthesis. Use holmes for improved high-degree numerical stability, especially for nmax >= 2040.')
    parser.add_argument('-ell', '--ellipsoid', type=str, default='wgs84',
                        help='Reference ellipsoid: wgs84, grs80, or JSON object string.')
    parser.add_argument('--ellipsoid-name', type=str, default=None,
                        help='Optional ellipsoid name to store in output metadata.')
    parser.add_argument('-pn', '--proj-name', type=str, default='GeoidProject',
                        help='Project directory.')


def main(args) -> dict:
    workflow = DTMSynthesis(
        dtm_model=args.dtm_model,
        model_format=args.model_format,
        earth2014_model=args.earth2014_model,
        earth2014_resolution=args.earth2014_resolution,
        input_file=args.input_file,
        model_dir=args.model_dir,
        output_dir=Path(args.proj_name) / 'results',
        max_deg=args.max_deg,
        ellipsoid=args.ellipsoid,
        ellipsoid_name=args.ellipsoid_name,
        bbox=args.bbox,
        grid_size=args.grid_size,
        grid_unit=args.grid_unit,
        chunk_size=args.chunk_size,
        chunk_memory_gb=args.chunk_memory_gb,
        n_workers=args.workers,
        leg_progress=args.leg_progress,
        threaded_legendre=args.threaded_legendre,
        legendre_method=args.legendre_method,
        proj_name=args.proj_name,
    )
    return workflow.synthesize()
