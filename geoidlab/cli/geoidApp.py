import sys
import typer
from typing import List, Optional
from typing_extensions import Annotated

from geoidlab.ggm import GlobalGeopotentialModel

typer.rich_utils.STYLE_HELPTEXT = ""
app = typer.Typer(rich_markup_mode='rich', no_args_is_help=True) # print help message if ran without arguments

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

VALID_TIDE_SYSTEMS = ['zero', 'mean', 'tide-free']

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
    tide_target   : Optional[str] = typer.Option('tide-free', help=f'Target tide system for the geoid model. Options: {VALID_TIDE_SYSTEMS}'),
    tide_gravity  : Optional[str] = typer.Option(None, help=f'Tide system of the gravity data (if known). Options: {VALID_TIDE_SYSTEMS}'),
    ellipsoid     : Annotated[str, typer.Option(help='Reference ellipsoid for calculations. Options: [wgs84, grs80]')] = 'wgs84',
) -> None:
    # Print help message if user executes geoidApp without arguments
    if not any([do, start, end]):
        typer.echo(ctx.get_help())
        raise typer.Exit()
    
    # Validate CLI call with 
    if do and (start or end):
        typer.echo('Use either --do or --start/--end, not both.')
        raise typer.Exit(code=1)
    
    if do:
        if do not in ALL_STEPS:
            typer.echo(f'Invalid step: {do}')
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
    
def _run_step(step: str, ellipsoid: str) -> None:
    func = STEP_FUNCTIONS.get(step)
    
    if not func:
        typer.echo(f'Invalid step: {step}')
        raise typer.Exit(code=1)
    
    typer.echo(f'Running step: {step}...')
    func(ellipsoid=ellipsoid)
    
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