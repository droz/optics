from .surfaces import Surface
from .rays import RayBundle
from .screen import Screen
from .system import System
from chart_studio import plotly as plotly
from plotly.offline import plot
import plotly.graph_objs as graphs
import numpy as np

def meshContour(x, y, z):
    """ A utility function to find the indexes of the mesh points that form the contour of the mesh.
        It will know how to handle meshes that close on themselves
    Args:
        x: a 2D numpy array representing the x coordinates of the mesh points
        y: a 2D numpy array representing the y coordinates of the mesh points
    Returns:
        starts_i: the first indices of the start points of the contours
        starts_j: the second indices of the start points of the contours
        ends_i: the first indices of the end points of the contours
        end_j: the second indices of the end points of the contours
        """
    # In the simplest case, we just need to use the segments on the edges of the mesh
    nr = x.shape[0]
    nc = x.shape[1]
    n = nr * nc
    starts_i = []
    starts_j = []
    ends_i = []
    ends_j = []
    # Vertical side 1
    starts_i += range(0, nr - 1)
    starts_j += [0] * (nr - 1)
    ends_i   += range(1, nr)
    ends_j   += [0] * (nr - 1)
    # Vertical side 2
    starts_i += range(0, nr - 1)
    starts_j += [nc-1] * (nr - 1)
    ends_i   += range(1, nr)
    ends_j   += [nc-1] * (nr - 1)
    # Horizontal side 1
    starts_i += [0] * (nc - 1)
    starts_j += range(0, nc - 1)
    ends_i   += [0] * (nc - 1)
    ends_j   += range(1, nc)
    # Horizontal side 2
    starts_i += [nr - 1] * (nc - 1)
    starts_j += range(0, nc - 1)
    ends_i   += [nr - 1] * (nc - 1)
    ends_j   += range(1, nc)

    # If two segments overlap, do not display them
    good_idx = []
    for i in range(len(starts_i)):
        found_overlap = False
        for j in range(len(starts_i)):
            if i == j:
                continue
            dist1 = max(abs(x[starts_i[i], starts_j[i]] - x[starts_i[j], starts_j[j]]),
                        abs(y[starts_i[i], starts_j[i]] - y[starts_i[j], starts_j[j]]),
                        abs(z[starts_i[i], starts_j[i]] - z[starts_i[j], starts_j[j]]),
                        abs(x[ends_i[i], ends_j[i]] - x[ends_i[j], ends_j[j]]),
                        abs(y[ends_i[i], ends_j[i]] - y[ends_i[j], ends_j[j]]),
                        abs(z[ends_i[i], ends_j[i]] - z[ends_i[j], ends_j[j]]))
            dist2 = max(abs(x[starts_i[i], starts_j[i]] - x[ends_i[j], ends_j[j]]),
                        abs(y[starts_i[i], starts_j[i]] - y[ends_i[j], ends_j[j]]),
                        abs(z[starts_i[i], starts_j[i]] - z[ends_i[j], ends_j[j]]),
                        abs(x[ends_i[i], ends_j[i]] - x[starts_i[j], starts_j[j]]),
                        abs(y[ends_i[i], ends_j[i]] - y[starts_i[j], starts_j[j]]),
                        abs(z[ends_i[i], ends_j[i]] - z[starts_i[j], starts_j[j]]))
            if dist1 < 1e-6 or dist2 < 1e-6:
                found_overlap = True
                break
        if not found_overlap:
            good_idx.append(i)
    starts_i = [starts_i[i] for i in good_idx]
    starts_j = [starts_j[i] for i in good_idx]
    ends_i = [ends_i[i] for i in good_idx]
    ends_j = [ends_j[i] for i in good_idx]

    return starts_i, starts_j, ends_i, ends_j


def surfaceSceneData(surface: Surface,
                     
                     show_surface: bool,
                     show_contours: bool,
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
        line_format = dict(color='gray', width=2)
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

    if show_contours:
        # Lens contour if requested
        line_format = dict(color='black', width=5)
        si, sj, ei, ej = meshContour(x, y, z)
        # An array full of NaNs to separate the different wires
        nans = np.full(len(si), np.nan)
        contour_x = np.vstack((x[si, sj], x[ei, ej], nans)).T.flatten()
        contour_y = np.vstack((y[si, sj], y[ei, ej], nans)).T.flatten()
        contour_z = np.vstack((z[si, sj], z[ei, ej], nans)).T.flatten()
        scene_data.append(graphs.Scatter3d(x=contour_x, y=contour_y, z=contour_z, mode='lines', line=line_format, showlegend=False))

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
    def line_data(rays, idx, color, opacity):
        """ Generate the data for a set of indexes"""
        scene_data = []
        # Get the origins of the rays
        origins = rays.origins[idx]
        lengths = rays.lengths[idx]
        directions = rays.directions[idx]

        # Start with bound (non-zero length) rays    
        idx_bound = np.where(lengths != 0)[0]
        # Get the endpoints of the rays
        origins_bound = origins[idx_bound]
        endpoints_bound = origins[idx_bound] + directions[idx_bound] * lengths[idx_bound][:, np.newaxis]
        nans_bound = np.full(len(idx_bound), np.nan)
        x_bound = np.vstack((origins_bound[:, 0], endpoints_bound[:, 0], nans_bound)).T.flatten()
        y_bound = np.vstack((origins_bound[:, 1], endpoints_bound[:, 1], nans_bound)).T.flatten()
        z_bound = np.vstack((origins_bound[:, 2], endpoints_bound[:, 2], nans_bound)).T.flatten()
        scene_data.append(graphs.Scatter3d(x=x_bound, y=y_bound, z=z_bound, mode='lines', line=dict(color=color, width=2), opacity=opacity, showlegend=False))

        # Then deal with unbound rays
        idx_unbound = np.where(lengths == 0)[0]
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
        scene_data.append(graphs.Scatter3d(x=x_unbound, y=y_unbound, z=z_unbound, mode='lines', line=dict(color=color, width=2), opacity=opacity, showlegend=False))

        # Add dashed lines at the end of the unbound rays
        dash_end = endpoints_unbound + directions[idx_unbound] * length * 0.1
        x_dash = np.vstack((endpoints_unbound[:, 0], dash_end[:, 0], nans_unbound)).T.flatten()
        y_dash = np.vstack((endpoints_unbound[:, 1], dash_end[:, 1], nans_unbound)).T.flatten()
        z_dash = np.vstack((endpoints_unbound[:, 2], dash_end[:, 2], nans_unbound)).T.flatten()
        scene_data.append(graphs.Scatter3d(x=x_dash, y=y_dash, z=z_dash, mode='lines', line=dict(color=color, width=2, dash='dash'), opacity=opacity, showlegend=False))

        # Now we can build the scene data
        x = np.concatenate((x_bound, x_unbound))
        y = np.concatenate((y_bound, y_unbound))
        z = np.concatenate((z_bound, z_unbound))
        return scene_data


    if show_display_rays:
        scene_data += line_data(rays, rays.display_rays, 'blue', 1)
    if show_all_rays:
        indexes = list(set(range(rays.origins.shape[0])) - set(rays.display_rays))
        # If there are too many rays, we need to sub-sample
        if len(indexes) > 10000:
            indexes = np.random.choice(indexes, 10000, replace=False)
        opacity = min(300 / len(indexes), 1.0)
        scene_data += line_data(rays, indexes, 'black', opacity)

    return scene_data

def screenSceneData(screen: Screen,
                    show_contours: bool,
                    show_surface: bool,
                    show_screen_intersections: bool):
    """ Generate the data to display a screen
    Args:
        screen: a Screen object
        show_surface: a boolean indicating if the screen surface should be displayed
        show_contours: a boolean indicating if the contours of the screen should be displayed
        show_screen_intersections: a boolean indicating if the screen/rays intersections should be displayed
    Returns:
        a list of graph objects that can be displayed with plotly"""
    # Get the mesh from the screen object
    x, y, z = screen.mesh()

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

    if show_contours:
        # Lens contour if requested
        line_format = dict(color='black', width=5)
        si, sj, ei, ej = meshContour(x, y, z)
        # An array full of NaNs to separate the different wires
        nans = np.full(len(si), np.nan)
        contour_x = np.vstack((x[si, sj], x[ei, ej], nans)).T.flatten()
        contour_y = np.vstack((y[si, sj], y[ei, ej], nans)).T.flatten()
        contour_z = np.vstack((z[si, sj], z[ei, ej], nans)).T.flatten()
        scene_data.append(graphs.Scatter3d(x=contour_x, y=contour_y, z=contour_z, mode='lines', line=line_format, showlegend=False))

    return scene_data

def display(object: object,
            show_surface: bool = True,
            show_contours: bool = True,
            show_wireframe: bool = True,
            show_normals: bool = False,
            show_display_rays: bool = True,
            show_all_rays: bool = False,
            show_screen_intersections: bool = False):
    """ Display an element of the optics system
    Args:
        object: an object that is part of the library (system, surface, aperture, ray_bundle)
        show_surface: a boolean indicating if the surface should be displayed
        show_contours: a boolean indicating if the elements contours should be displayed
        show_wireframe: a boolean indicating if the wireframe should be displayed
        show_normals: a boolean indicating if the normals should be displayed
        show_display_rays: a boolean indicating if the display rays should be displayed
        show_all_rays: a boolean indicating if all the rays should be displayed
        show_screen_intersections: a boolean indicating if the screen/rays intersections should be displayed"""
    # Do different things depending on the type of object
    if isinstance(object, Surface):
        surface = object
        # Get the scene data
        scene_data = surfaceSceneData(surface,
                                      show_surface=show_surface,
                                      show_contours=show_contours,
                                      show_wireframe=show_wireframe,
                                      show_normals=show_normals)
    if isinstance(object, RayBundle):
        rays = object
        # Get the scene data
        scene_data = rayBundleSceneData(rays,
                                        show_display_rays=show_display_rays,
                                        show_all_rays=show_all_rays)
        
    if isinstance(object, Screen):
        screen = object
        scene_data = screenSceneData(screen,
                                     show_surface=show_surface,
                                     show_contours=show_contours,
                                     show_screen_intersections=show_screen_intersections)

    if isinstance(object, System):
        system = object
        scene_data = []
        # Display the surfaces
        for surface in system.surfaces:
            scene_data += surfaceSceneData(surface,
                                           show_surface=show_surface,
                                           show_contours=show_contours,
                                           show_wireframe=show_wireframe,
                                           show_normals=show_normals)
        # Display the screens
        for screen in system.screens:
            scene_data += screenSceneData(screen,
                                          show_surface=show_surface,
                                          show_contours=show_contours,
                                          show_screen_intersections=show_screen_intersections)
        # Display the rays
        for rays in system.rays:
            scene_data += rayBundleSceneData(rays,
                                             show_display_rays=show_display_rays,
                                             show_all_rays=show_all_rays)

    else:
        raise NotImplementedError("display() is not implemented for this object")
    # Create the layout
    layout = graphs.Layout(
        title='Surface sag',
        autosize=True,
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
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=1, y=0, z=0)
            ),
        )
    )
    fig = graphs.Figure(data=scene_data, layout=layout)
    plot(fig)
