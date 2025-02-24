from optics.surfaces import Surface
from chart_studio import plotly as plotly
from plotly.offline import plot
import plotly.graph_objs as graphs
import numpy as np

def surfaceSceneData(surface: Surface,
                     show_surface: bool,
                     show_wireframe: bool,
                     show_normals: bool):
    """ Generate the data to display a surface
    Args:
        surface: a Surface object
        show_surface: a boolean indicating if the surface should be displayed
        show_wireframe: a boolean indicating if the wireframe should be displayed
        show_normals: a boolean indicating if the normals should be displayed
    Returns:
        a list of graph objects that can be displayed with plotly"""
    # Get the mesh from the surface object
    x, y, z = surface.mesh()
    # Generate sub-sampled indexes
    subrange_x = list(range(0, x.shape[0], 5))
    if x.shape[0] - 1 not in subrange_x:
        subrange_x.append(x.shape[0] - 1)
    subrange_y = list(range(0, y.shape[1], 5))
    if y.shape[1] - 1 not in subrange_y:
        subrange_y.append(y.shape[1] - 1)
    subranges = np.meshgrid(subrange_x, subrange_y)

    scene_data = []
    if show_surface:
        color = 'blue'
        # Generate a triangulation for the mesh
        vertices_indices = np.arange(0, x.shape[0] * x.shape[1], 1).reshape(x.shape)
        i0 = vertices_indices[0:-1, 0:-1].flatten()
        j0 = vertices_indices[1:, 0:-1].flatten()
        k0 = vertices_indices[1:, 1:].flatten()
        i1 = vertices_indices[0:-1, 0:-1].flatten()
        j1 = vertices_indices[1:, 1:].flatten()
        k1 = vertices_indices[0:-1, 1:].flatten()
        scene_data.append(graphs.Mesh3d(x=x.flatten(), y=y.flatten(), z=z.flatten(),
                                        i=np.concatenate((i0,i1)), j=np.concatenate((j0,j1)), k=np.concatenate((k0,k1)),
                                        colorscale=[[0, color], [1, color]],
                                        showscale=False,
                                        flatshading=False,
                                        lighting=dict(ambient=0.8, diffuse=0.5, specular=0.5, roughness=0.5)))

    if show_wireframe:
        # We also add lines to highlight the shape of the surface
        line_format = dict(color='black', width=2)
        # An array full of NaNs to separate the different wires
        nans0 = np.full((len(subrange_x), 1), np.nan)
        wire0_x = np.hstack((x[subrange_x, :], nans0)).flatten()
        wire0_y = np.hstack((y[subrange_x, :], nans0)).flatten()
        wire0_z = np.hstack((z[subrange_x, :], nans0)).flatten()
        nans1 = np.full((1, len(subrange_y)), np.nan)
        wire1_x = np.vstack((x[:, subrange_y], nans1)).T.flatten()
        wire1_y = np.vstack((y[:, subrange_y], nans1)).T.flatten()
        wire1_z = np.vstack((z[:, subrange_y], nans1)).T.flatten()
        wire_x = np.concatenate((wire0_x, wire1_x))
        wire_y = np.concatenate((wire0_y, wire1_y))
        wire_z = np.concatenate((wire0_z, wire1_z))
        scene_data.append(graphs.Scatter3d(x=wire_x, y=wire_y, z=wire_z, mode='lines', line=line_format, showlegend=False))

    if show_normals:
        # Compute the normals at the mesh points
        points = np.array([x[subranges].flatten(),
                           y[subranges].flatten(),
                           z[subranges].flatten()]).T
        normals = surface.normals(points[:, 0:2])
        # Scale the normals based on the total size of the surface
        range_x = x.max() - x.min()
        range_y = y.max() - y.min()
        range_z = z.max() - z.min()
        scale = np.sqrt(range_x**2 + range_y**2 + range_z**2) / 10
        normals *= scale
        # Display the normals
        nans = np.full((1, points.shape[0]), np.nan)
        endpoints = points + normals
        nx = np.vstack((points[:, 0], endpoints[:, 0], nans)).T.flatten()
        ny = np.vstack((points[:, 1], endpoints[:, 1], nans)).T.flatten()
        nz = np.vstack((points[:, 2], endpoints[:, 2], nans)).T.flatten()
        scene_data.append(graphs.Scatter3d(x=nx, y=ny, z=nz,
                                           mode='lines', line=dict(color='red', width=2), showlegend=False
                                           ))
        scene_data.append(graphs.Scatter3d(x=endpoints[:, 0], y=endpoints[:, 1], z=endpoints[:, 2],
                                           mode='markers', marker=dict(color='red', size=2), showlegend=False
                                           ))

    return scene_data



def display(object: object, show_surface: bool = True, show_wireframe: bool = True, show_normals: bool = False):
    """ Display an element of the optics system
    Args:
        object: an object that is part of the library (system, surface, aperture, ray_bundle)
        show_surface: a boolean indicating if the surface should be displayed
        show_wireframe: a boolean indicating if the wireframe should be displayed
        show_normals: a boolean indicating if the normals should be displayed"""
    # Do different things depending on the type of object
    if isinstance(object, Surface):
        surface = object
        # Get the graph data
        graph_data = surfaceSceneData(surface,
                                      show_surface=show_surface,
                                      show_wireframe=show_wireframe,
                                      show_normals=show_normals)
        # Create the layout
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
