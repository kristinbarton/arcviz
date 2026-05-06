import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_track_comparison(track_da, grid_das=None, point_da=None, output_path=None, title="Grid to Track Comparison", colors=None, ax=None):
    """
    Plots a time series comparing a main track DataArray to one or more grid/point DataArrays.
    Inputs are expected to be xarray.DataArray objects.
    """
    logger.info("Generating time series plot...")
    grid_das = grid_das or []

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False
    
    # Plot observation track data
    track_da.plot.line(
        x='time',
        ax=ax,
        label=track_da.name,
        color='black',
        linestyle='-',
        marker='o',
        markersize=4,
        alpha=0.7,
    )

    # Plot point data if provided 
    if point_da is not None:
        point_1d = point_da.isel(lat=0, lon=0) if 'lat' in point_da.dims else point_da
        point_1d.plot.line(
            x='time',
            ax=ax,
            color='#005f73',
            linestyle='-',
            marker='o',
            markersize=4,
            alpha=0.7,
            label=point_da.name
        )


    if colors is None:
        colors = ['#005f73', '#ee9b00', '#9b2226', '#0a9396', '#7b2cbf', '#bb3e03']

    for i, gtrk_da in enumerate(grid_das):
        c = colors[i % len(colors)]

        if 'track_point' in gtrk_da.dims:
            gtrk_da.where((gtrk_da != 0).compute(), drop=True).plot.line(
                x='time',
                ax=ax,
                color=c,
                linestyle='--',
                marker='.',
                markersize=4,
                alpha=0.7,
                add_legend=False
            )
            ax.plot([], [], color=c, linestyle='--', label=gtrk_da.name)
        else:
            gtrk_da.where((gtrk_da != 0).compute(), drop=True).plot(
                ax=ax,
                label=gtrk_da.name,
                color=c,
                linestyle='-',
                marker='o',
                markersize=4,
                alpha=1.0,
            )

    ax.set_title(title, fontsize=14)
    units = track_da.attrs.get('units', 'Unknown Units')

    base_name = track_da.attrs.get('standard_name', track_da.name or 'Value')
    ax.set_ylabel(f"{base_name} ({units})", fontsize=12)
    ax.set_xlabel(f"Time", fontsize=12)

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    if output_path:
        fig.autofmt_xdate()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved as: {output_path}")
    elif created_fig:
        fig.autofmt_xdate()
        plt.show()

    return fig, ax
