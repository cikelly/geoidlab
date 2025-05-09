############################################################
# Geoid workflow CLI interface                             #
# Copyright (c) 2024, Caleb Kelly                          #
# Author: Caleb Kelly  (2024)                              #
############################################################

import typer

from typing import List, Optional
from typing_extensions import Annotated
from dataclasses import dataclass
from pathlib import Path

from geoidlab.ggm import GlobalGeopotentialModel
from geoidlab.icgem import get_ggm_tide_system
from geoidlab.tide import GravityTideSystemConverter




typer.rich_utils.STYLE_HELPTEXT = ""
app = typer.Typer(rich_markup_mode='rich', no_args_is_help=True) # print help message if ran without arguments

VALID_TIDE_SYSTEMS = ['zero_tide', 'mean_tide', 'tide_free']
VALID_ELLIPSOIDS = ['wgs84', 'grs80']

@dataclass
class WorkflowContext:
    '''Stores workflow state for geoid computation.'''
    ellipsoid: str
    ggm: Optional[GlobalGeopotentialModel] = None
    ggm_tide_system: Optional[str] = None
    tide_target: str = 'tide_free'
    tide_gravity: Optional[str] = None # if unknown, probably mean_tide
    data: dict = None # Stores file paths or lightweight data (e.g., tide system)
    project_dir: Path = None


ALL_STEPS = [
    'free-air',         # Compute the Free-air anomalies (Dg)
    'terrain',          # Terrain correction (tc)
    'helmert',          # Interpolate tc and add to Dg
    'append-marine',    # Append marine gravity anomalies (if applicable)
    'ggm',              # Compute GGM gravity anomalies (Dg_ggm)
    'residuals',        # Compute residual anomalies
    'grid',             # Grid residual anomalies
    'compute',          # Compute residual geoid (N_res)
    'indirect',         # Compute indirect effect (N_ind)
    'reference',        # Compute reference geoid from GGM (N_ggm)
    'restore',          # Add all components to get geoid (N = N_res + N_ggm + N_ind)
]


# Step function registry
STEP_FUNCTIONS = {
    'free-air'     : lambda ellipsoid: _free_air(ellipsoid),
    'terrain'      : lambda ellipsoid: _terrain_correction(ellipsoid),
    'helmert'      : lambda ellipsoid: _helmert_anomalies(ellipsoid),
    'append-marine': lambda ellipsoid: _append_marine(ellipsoid),
    'ggm'          : lambda ellipsoid: _ggm_anomalies(ellipsoid),
    'residuals'    : lambda ellipsoid: _residuals(ellipsoid),
    'grid'         : lambda ellipsoid: _grid_anomalies(ellipsoid),
    'compute'      : lambda ellipsoid: _compute_geoid(ellipsoid),
    'indirect'     : lambda ellipsoid: _indirect_effect(ellipsoid),
    'reference'    : lambda ellipsoid: _reference_geoid(ellipsoid),
    'restore'      : lambda ellipsoid: _restore_geoid(ellipsoid)
}


@app.command(
    help=(
        "Geoid Computation Workflow - Remove-Compute-Restore Method\n\n"
        "Processing Stages:\n\n"
        "----------------------------------------\n\n"
        "1. DATA PREPARATION\n\n"
        "   - free-air                          : Compute Free-air anomalies\n"
        "   - terrain                           : Calculate terrain corrections\n"
        "   - helmert                           : Create Helmert condensation anomalies\n"
        "   - append-marine                     : Merge marine gravity data\n\n"
        "2. RESIDUAL PROCESSING\n"
        "   - ggm                               : Compute GGM theoretical gravity\n"
        "   - residuals                         : Calculate residual anomalies\n"
        "   - grid                              : Interpolate to regular grid\n\n"
        "3. GEOID COMPUTATION\n"
        "   - compute                           : Calculate residual geoid\n"
        "   - indirect                          : Compute indirect effect\n"
        "   - reference                         : Reference geoid from GGM\n\n"
        "4. FINAL OUTPUT\n"
        "   - restore                           : Combine components for final geoid\n\n"
        "Usage Notes:\n"
        "- Steps must be executed in order\n"
        "- Use --start/--end for processing sequences\n"
        "- Default ellipsoid: wgs84 (options: wgs84/grs8)"
    ),
    epilog=(
        "Examples:\n"
        "----------------------------------------\n\n"
        "  # Single step processing\n\n"
        "  geoidApp --do free-air\n\n"
        "  # Process terrain corrections through gridding\n\n"
        "  geoidApp --start terrain --end grid\n\n"
        "  # Full workflow with GRS80 ellipsoid\n\n"
        "  geoidApp --ellipsoid grs80\n\n"
        "  # Complete processing (all steps)\n"
        "  geoidApp\n\n"
    )
)

# Define main program
def run(
    ctx           : typer.Context,
    do            : Optional[str] = typer.Option(None, help=f'Execute a single processing step. Available steps:\n [{", ".join(ALL_STEPS)}]'),
    start         : Optional[str] = typer.Option(None, help='Start step in processing sequence (inclusive).'),
    end           : Optional[str] = typer.Option(None, help='End step in processing sequence (inclusive).'),
    tide_target   : Optional[str] = typer.Option('tide_free', help=f'Target tide system for the geoid model. Options: {VALID_TIDE_SYSTEMS}'),
    tide_gravity  : Optional[str] = typer.Option(None, help=f'Tide system of the gravity data. If unknown, assume mean_tide. Options: {VALID_TIDE_SYSTEMS}'),
    ggm           : Optional[str] = typer.Option(None, help='Path to Global Geopotential Model (GGM) or name of GGM'),
    ellipsoid     : Annotated[str, typer.Option(help=f'Reference ellipsoid for calculations. Options: {VALID_ELLIPSOIDS}')] = 'wgs84',
    gravity_data  : Optional[str] = typer.Option(None, help='Path to gravity data file (CSV, TXT, XLSX, XLS)'),
    output        : Optional[str] = typer.Option(None, help='Output file for final geoid (default: <project-dir>/results/geoid.nc)'),
    project_dir   : Optional[str] = typer.Option(None, help='Project directory containing downloads and results subdirectories (default: current directory')
) -> None:
    # Print help message if user executes geoidApp without arguments
    if not any([do, start, end]):
        typer.echo(ctx.get_help())
        raise typer.Exit()
    
    # Validate CLI call with 
    if do and (start or end):
        typer.echo('Use either --do or --start/--end, not both.')
        raise typer.Exit(code=1)
    
    if ellipsoid not in VALID_ELLIPSOIDS:
        typer.echo(f'Invalid ellipsoid: {ellipsoid}')
        raise typer.Exit(code=1)

    if tide_target not in VALID_TIDE_SYSTEMS:
        typer.echo(f'Invalid tide target: {tide_target}')
        raise typer.Exit(code=1)

    if tide_gravity and tide_gravity not in VALID_TIDE_SYSTEMS:
        typer.echo(f'Invalid tide gravity: {tide_gravity}')
        raise typer.Exit(code=1)
    
    if gravity_data and not Path(gravity_data).exists():
        typer.echo(f'Gravity data file not found: {gravity_data}')
        raise typer.Exit(code=1)
    
    # Set up project directory
    project_dir = Path(project_dir).resolve() if project_dir else Path.cwd()
    downloads_dir = project_dir / 'downloads'
    results_dir = project_dir / 'results'
    downloads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Initialize worflow context
    context = WorkflowContext(
        ellipsoid=ellipsoid,
        tide_target=tide_target,
        tide_gravity=tide_gravity,
        data={}, # Initialize empty data store
        project_dir=project_dir
    )
    
    
    
    
    # Load gravity data if provided
    if gravity_data:
        try:
            converter = GravityTideSystemConverter(path_to_data=gravity_data)
            gravity_file = results_dir / 'gravity.csv'
            converter.data.to_csv(gravity_file, index=False)
            context.data['gravity'] = str(gravity_file)
        except Exception as e:
            typer.echo(f'Failed to load gravity data: {e}')
            raise typer.Exit(code=1)
    
    # Load GGM and fetch tide system if provided
    if ggm:
        context.ggm, context.ggm_tide_system = _load_ggm(ggm)
        typer.echo(f'Loaded GGM: {ggm}, Tide System: {context.ggm_tide_system}')
    
    # Single step mode
    if do:
        if do not in ALL_STEPS:
            typer.echo(f'Invalid step: {do}. Supported:\n[{", ".join(ALL_STEPS)}')
            raise typer.Exit(code=1)
        _run_step(do, ellipsoid=ellipsoid)
        return

    # Multi-step mode
    start_idx = ALL_STEPS.index(start) if start else 0
    end_idx = ALL_STEPS.index(end) if end else len(ALL_STEPS) - 1
    
    if start_idx > end_idx:
        typer.echo('Start step comes after end step.')
        raise typer.Exit(code=1)
    
    steps_copy = ALL_STEPS[start_idx:end_idx + 1].copy()
    typer.echo(f'Will run steps: [{", ".join(steps_copy)}].')
    for step in ALL_STEPS[start_idx:end_idx + 1]:
        _run_step(step, ellipsoid=ellipsoid)
        steps_copy.remove(step)  # Remove the processed step
        remaining = ', '.join(steps_copy) if steps_copy else 'None'
        typer.echo(f'Completed step: {step}.\nRemaining steps: [{remaining}].')


def _load_ggm(ggm: str) -> tuple[GlobalGeopotentialModel, str]:
    '''Load GGM and fetch its tide system.'''
    pass

def _run_step(step: str, context: WorkflowContext) -> None:
    '''Execute a single processing step with context.'''
    func = STEP_FUNCTIONS.get(step)
    if not func:
        typer.echo(f'Invalid step: {step}')
        raise typer.Exit(code=1)
    
    typer.echo(f'Running step: {step}...')
    func(context)
    
# === Stub functions for each step ===
def _free_air(ellipsoid: str) -> None:
    typer.echo('--> Computing Free-Air anomalies')


def _terrain_correction(ellipsoid: str) -> None:
    typer.echo('--> Computing terrain correction')
    

def _helmert_anomalies(ellipsoid: str) -> None:
    typer.echo('--> Interpolating terrain correction and forming Helmert anomalies')


def _append_marine(ellipsoid: str) -> None:
    typer.echo('--> Appending marine gravity anomalies')
    

def _ggm_anomalies(ellipsoid: str) -> None:
    typer.echo('--> Loading GGM and computing anomalies')
    

def _residuals(ellipsoid: str) -> None:
    typer.echo('--> Calculating residual anomalies')
    

def _grid_anomalies(ellipsoid: str) -> None:
    typer.echo('--> Gridding anomalies')
    

def _compute_geoid(ellipsoid: str) -> None:
    typer.echo('--> Computing residual geoid')
    

def _indirect_effect(ellipsoid: str) -> None:
    typer.echo('--> Computing indirect effect')

def _reference_geoid(ellipsoid: str) -> None:
    typer.echo('--> Computing reference geoid')

def _restore_geoid(ellipsoid: str) -> None:
    typer.echo('--> Restoring omitted components')

# if __name__ == '__main__':
#     app()

if __name__ == '__main__':
    # if len(sys.argv) == 1:
    #     # Explicitly show help when no arguments are provided
    #     sys.argv.append("--help")
    app()