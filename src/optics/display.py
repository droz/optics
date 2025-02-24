from optics.surfaces import Surface
from chart_studio import plotly as plotly
from plotly.offline import plot
import plotly.graph_objs as graphs
import numpy as np

def surfaceGraphData(surface: Surface):
    """ Generate the data to display a surface
    Args:
      surface: a Surface object
    Returns:
      a list of graph objects that can be displayed with plotly"""
    # Get the mesh from the surface object
    x, y, z = surface.mesh()
    color = 'blue'
    # Generate a triangulation for the mesh
    vertices_indices = np.arange(0, x.shape[0] * x.shape[1], 1).reshape(x.shape)
    i0 = vertices_indices[0:-1, 0:-1].flatten()
    j0 = vertices_indices[1:, 0:-1].flatten()
    k0 = vertices_indices[1:, 1:].flatten()
    i1 = vertices_indices[0:-1, 0:-1].flatten()
    j1 = vertices_indices[1:, 1:].flatten()
    k1 = vertices_indices[0:-1, 1:].flatten()
    surface_data = graphs.Mesh3d(x=x.flatten(), y=y.flatten(), z=z.flatten(),
                                 i=np.concatenate((i0,i1)), j=np.concatenate((j0,j1)), k=np.concatenate((k0,k1)),
                                  colorscale=[[0, color], [1, color]],
                                  showscale=False,
                                  flatshading=False,
                                  lighting=dict(ambient=0.8, diffuse=0.5, specular=0.5, roughness=0.5))
    # We also add lines to highlight the shape of the surface
    line_format = dict(color='black', width=2)
    # X direction
    range_x = list(range(0, x.shape[0], 5))
    x_lines_data = []
    if x.shape[0] - 1 not in range_x:
      range_x.append(x.shape[0] - 1)
    for i in range_x:
      x_lines_data.append(graphs.Scatter3d(x=x[i, :], y=y[i, :], z=z[i, :], mode='lines', line=line_format, showlegend=False))
    # Y direction
    y_lines_data = []
    range_y = list(range(0, y.shape[1], 5))
    if y.shape[1] - 1 not in range_y:
      range_y.append(y.shape[1] - 1)
    for i in range_y:
      y_lines_data.append(graphs.Scatter3d(x=x[:, i], y=y[:, i], z=z[:, i], mode='lines', line=line_format, showlegend=False))

    return [surface_data] + x_lines_data + y_lines_data



def display(object: object):
    """ Display an element of the optics system
    Args:
      object: an object that is part of the library (system, surface, aperture, ray_bundle)"""
    # Do different things depending on the type of object
    if isinstance(object, Surface):
        surface = object
        # Get the graph data
        graph_data = surfaceGraphData(surface)
        layout = graphs.Layout(
            title='Surface sag',
            autosize=False,
            scene=dict(
                xaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    showbackground=False,
                ),
                yaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=False,
                ),
                zaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=False,
                ),
                aspectmode='data',
            )
        )
        fig = graphs.Figure(data=graph_data, layout=layout)
        plot(fig)
    else:
        raise NotImplementedError("display() is not implemented for this object")
