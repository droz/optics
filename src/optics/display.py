from optics.surfaces import Surface
from optics.rays import RayBundle
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

def rayBundleSceneData(rays: RayBundle, show_display_rays: bool, show_all_rays: bool):
    """ Generate the data to display a ray bundle
    Args:
        rays: a RayBundle object
        show_display_rays: a boolean indicating if the display rays should be displayed
        show_all_rays: a boolean indicating if all the rays should be displayed
    Returns:
        a list of graph objects that can be displayed with plotly"""
    scene_data = []
    def line_data(rays, idx, color):
        """ Generate the data for a set of indexes"""
        # Get the origins of the rays
        origins = rays.origins[idx]
        lengths = rays.lengths[idx]
        directions = rays.directions[idx]

        # Start with bound (non-zero length) rays    
        idx_bound = np.where(lengths[idx] != 0)[0]
        # Get the endpoints of the rays
        origins_bound = origins[idx_bound]
        endpoints_bound = origins[idx_bound] + directions[idx_bound] * lengths[idx_bound][:, np.newaxis]
        nans_bound = np.full(len(idx_bound), np.nan)
        x_bound = np.vstack((origins_bound[:, 0], endpoints_bound[:, 0], nans_bound)).T.flatten()
        y_bound = np.vstack((origins_bound[:, 1], endpoints_bound[:, 1], nans_bound)).T.flatten()
        z_bound = np.vstack((origins_bound[:, 2], endpoints_bound[:, 2], nans_bound)).T.flatten()

        # Then deal with unbound rays
        idx_unbound = np.where(lengths[idx] == 0)[0]
        # We need to decide on a length
        # If there are any bound rays in the bundle, we use the average of their length
        idx_bound_all = np.where(rays.lengths != 0)[0]
        if len(idx_bound_all) > 0:
            length = np.mean(lengths[idx_bound_all])
        else:
            # Otherwise we use the size of the origins bounding box
            length = np.linalg.norm(rays.origins.max(axis=0) - rays.origins.min(axis=0))
        origins_unbound = origins[idx_unbound]
        nans_unbound = np.full(len(idx_unbound), np.nan)
        endpoints_unbound = origins_unbound + directions[idx_unbound] * length
        x_unbound = np.vstack((origins_unbound[:, 0], endpoints_unbound[:, 0], nans_unbound)).T.flatten()
        y_unbound = np.vstack((origins_unbound[:, 1], endpoints_unbound[:, 1], nans_unbound)).T.flatten()
        z_unbound = np.vstack((origins_unbound[:, 2], endpoints_unbound[:, 2], nans_unbound)).T.flatten()
        # Add dashes at the end of the rays
        dash_start = endpoints_unbound
        num_dash = 10
        for i in range(num_dash):
            dash_length = length * 0.05 * (num_dash - i) / num_dash
            dash_end = dash_start + directions[idx_unbound] * dash_length * 0.5
            dash_start = dash_start + directions[idx_unbound] * dash_length
            x_dash = np.vstack((dash_start[:, 0], dash_end[:, 0], nans_unbound)).T.flatten()
            y_dash = np.vstack((dash_start[:, 1], dash_end[:, 1], nans_unbound)).T.flatten()
            z_dash = np.vstack((dash_start[:, 2], dash_end[:, 2], nans_unbound)).T.flatten()
            x_unbound = np.concatenate((x_unbound, x_dash))
            y_unbound = np.concatenate((y_unbound, y_dash))
            z_unbound = np.concatenate((z_unbound, z_dash))

        # Now we can build the scene data
        x = np.concatenate((x_bound, x_unbound))
        y = np.concatenate((y_bound, y_unbound))
        z = np.concatenate((z_bound, z_unbound))
        return graphs.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=color, width=2), showlegend=False)


    if show_display_rays:
        scene_data.append(line_data(rays, rays.display_rays, 'blue'))
    if show_all_rays:
        indexes = list(set(range(rays.origins.shape[0])) - set(rays.display_rays))
        scene_data.append(line_data(rays, indexes, 'black'))

    return scene_data

    #    # For all the rays that are unbound, use the maximum size of the source as the length
    #    scale = np.linalg.norm(rays.origins.max(axis=0) - rays.origins.min(axis=0))
    #    lengths = np.where(rays.lengths == 0, scale, rays.lengths)
    #    endpoints = rays.origin + rays.directions * lengths[:, np.newaxis]
    #    nans = np.full((1, points.shape[0]), np.nan)
    #    nx = np.vstack((points[:, 0], endpoints[:, 0], nans)).T.flatten()
    #    ny = np.vstack((points[:, 1], endpoints[:, 1], nans)).T.flatten()
    #    nz = np.vstack((points[:, 2], endpoints[:, 2], nans)).T.flatten()
    #    scene_data.append(graphs.Scatter3d(x=nx, y=ny, z=nz,
    #                                       mode='lines', line=dict(color='red', width=2), showlegend=False
    #                                       ))
        

def display(object: object,
            show_surface: bool = True,
            show_wireframe: bool = True,
            show_normals: bool = False,
            show_display_rays: bool = True,
            show_all_rays: bool = False):
    """ Display an element of the optics system
    Args:
        object: an object that is part of the library (system, surface, aperture, ray_bundle)
        show_surface: a boolean indicating if the surface should be displayed
        show_wireframe: a boolean indicating if the wireframe should be displayed
        show_normals: a boolean indicating if the normals should be displayed
        show_display_rays: a boolean indicating if the display rays should be displayed
        show_all_rays: a boolean indicating if all the rays should be displayed"""
    # Do different things depending on the type of object
    if isinstance(object, Surface):
        surface = object
        # Get the scene data
        scene_data = surfaceSceneData(surface,
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
        fig = graphs.Figure(data=scene_data, layout=layout)
        plot(fig)
    if isinstance(object, RayBundle):
        rays = object
        # Get the scene data
        scene_data = rayBundleSceneData(rays,
                                        show_display_rays=show_display_rays,
                                        show_all_rays=show_all_rays)
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
        fig = graphs.Figure(data=scene_data, layout=layout)
        plot(fig)
    else:
        raise NotImplementedError("display() is not implemented for this object")
