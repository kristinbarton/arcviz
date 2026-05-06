import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_track_comparison(track, grids, point, tends=None, output_path=None, title="Grid to Track Comparison", colors=None, ax=None, ylabel=None):
    """
    Plots a time series comparing TrackData observations to one or more grid data objects
    Must have 
    """
    logger.info("Generating time series plot...")

    if track.synced_da is None:
        raise ValueError(f"[{track.name}] Does not contain synced data")
    else:
        track_sync = track.synced_da

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    track_sync.plot(
        ax=ax,
        label=f'{track.name}',
        color='black',
        linestyle='-',
        marker='o',
        markersize=4,
        alpha=0.7,
    )

    if point.da is not None:
        point.da.isel(lat=0, lon=0).plot.line(
            x='time',
            ax=ax,
            color='#005f73',#'#ffA500',
            linestyle='-',
            marker='o',
            markersize=4,
            alpha=0.7,
            label=f'{point.name}',
        )


#    colors = ['#005f73', '#9b2226', '#ee9b00', '#0a9396']
    if colors is None:
        colors = ['#005f73', '#ee9b00', '#9b2226', '#0a9396', '#7b2cbf', '#bb3e03']

    for i, grid in enumerate(grids):
        if grid.track is not None:
            gtrk = grid.track
        else:
            logger.warning(f"Skipping {grid.name}: No track data found")
            continue

        if 'track_point' in gtrk.dims:
            gtrk.where((gtrk != 0).compute(), drop=True).plot.line(
                x='time',
                ax=ax,
                color=colors[i],
                linestyle='--',
                marker='.',
                markersize=4,
                alpha=0.7,
                add_legend=False
            )
            ax.plot([], [], color=colors[i], linestyle='--', label=f'{gtrk.name}')
        else:
            gtrk.where((gtrk != 0).compute(), drop=True).plot(
                ax=ax,
                label=f'{gtrk.name}',
                color=colors[i],
                linestyle='-',
                marker='o',
                markersize=4,
                alpha=1.0,
            )

    ax.set_title(title, fontsize=14)

    units = grids[0].track.attrs.get('units','m')
    ax.set_ylabel(f"{track.da.name} ({units})", fontsize=12)
    ax.set_xlabel(f"Time", fontsize=12)

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    if output_path:
        fig.autofmt_xdate()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved as: {output_path}")
    elif created_fig:
        logger.debug(f"[{title}] Showing from within routine")
        fig.autofmt_xdate()
        plt.show()

    return fig, ax

def plot_point_obs():
    pass
