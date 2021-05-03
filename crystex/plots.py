import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from plotly import tools
import plotly.graph_objs as go
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar

from crystex import lattice
from crystex import projections
from crystex import coordgeometry
from crystex import rotations
from crystex import numutils


def plot_pole_fig(proj_poles, poles, crys=None,  lattice_sys=None, axes='xyz',
                  grid=False, clrs=None, contour=False, bins=50, title=None,
                  marker_size=0.005, clr_map=None, proj_poles_theory=None,
                  axes_lbl=True, pole_lbl=None, cols_th=None):
    """
    Return a figure object for a pole figure. For a single crystal, plots a single 
    pole figure for all `poles`. For a poly crystal, plots n pole figures, one 
    for each pole in `poles`. 

    Parameters
    ----------
    proj_poles : tuple of ndarrays of shape (n,)
        Arrays of `n` polar angles and radii as projections of poles.
    crys : string
        Specifies type of crystal: 'single' or 'poly'. For 'single' crystal, 
        `proj_poles` expects the projections of pole(s) in a single crystal 
        aligned with the sample coordinate system. For a 'poly' crystal, 
        `proj_poles` expects the projections of a given pole in a set of crystals.
    poles : ndarray of shape (3, n)
        Specify poles to be plotted as column vectors.
    lattice_sys : string, optional
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal, 
        orthorhombic, monoclinic, triclinic.
    axes  : string
        Set alignment of sample axes with projection sphere axes. Options:
        'xyz' (default); 'yzx'; 'zxy'; 'yxz'; 'zyx'; 'xzy'.
    grid : bool
        Turn grid lines on plot on or off (default).
    clrs : list of string
        A list of colours to plot `poles` in a single crystal.
    contour : bool
        Plot a contour plot. False by dafualt. Only available if `crys` = 'poly'.
    bins : int
        If `contour` = True, number of bins.
    title: string
        Plot title
    proj_poles_theory : tuple of ndarrays of shape (n,)
        Arrays of `n` polar angles and radii as projections of the theoretical poles.
    Returns
    -------
    f : matplotlib figure

    TODO:
    - Sort out plot labelling: based on lattice system for single crystal, and
        based on plotted pole for polycrystal.
    - Think about whether to have more than one pole figure for a single crystal 
    as well.
    - Add check the lenght of proj_poles = number of poles given.
    - Add label with plot details: phase, projection type, upper hemisphere, 
    data plotted.
    - Think about link between number of bins and angles.

    """
    all_crys = ['single', 'poly']
    if crys not in all_crys:
        raise ValueError('"{}" is not a valid crystal type. '
                         '`crys` must be one of: {}.'.format(
                             crys, all_crys))

    # Check valid entry for axes alignment
    all_axes = ['xyz', 'yzx', 'zxy', 'yxz', 'zyx', 'xzy']
    if axes not in all_axes:
        raise ValueError('"{}" is not a valid axes option. '
                         '`axes` must be one of: {}.'.format(
                             axes, all_axes))

    # Plot for a single crystal
    if crys == 'single':
        f, ax = plt.subplots(1, 1, subplot_kw={'polar': True}, figsize=(5, 5))

        if clrs and len(clrs) == proj_poles[0][0].shape[0]:
            for i in range(len(clrs)):
                ax.scatter(proj_poles[0][0][i], proj_poles[0][1][i], c=clrs[i])
        elif not clrs:
            ax.scatter(proj_poles[0][0], proj_poles[0][1])
        else:
            raise ValueError('Length of {} and columns of {} do not match. '
                             'Please specify colours for each pole.'.format(clrs, poles))
        if title:
            ax.set_title(title, va='bottom')
        ax.set_rmax(1)
        if lattice_sys != 'hexagonal':
            ax.set_xticklabels(['', '', '[010]', '', '', '', '', ''])
            ax.annotate('[001]', xy=(0, 0), xytext=(10.3, 0.15))
        else:
            ax.set_xticklabels([])
            ax.annotate('[0001]', xy=(0, 0), xytext=(10.3, 0.15))

        ax.set_yticklabels([])
        if not grid:
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

    # Plot for a poly crystal
    elif crys == 'poly':
        n_figs = len(proj_poles)
        width_ratios = np.append([1] * n_figs, 0.1)
        # get poles labels
        poles_lbl = []

        for i in range(n_figs):
            poles_lbl.append(''.join([str(x) for x in poles[:, i]]))

        if not contour:          
            f_width = 3 * n_figs
            f_height = 10
            f = plt.figure(1, figsize=(f_width, f_height))
            gs0 = gridspec.GridSpec(1, 1)
            gs0.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
            gs00 = gridspec.GridSpecFromSubplotSpec(1, n_figs+1, subplot_spec=gs0[0], width_ratios=width_ratios)

        else:
            # Compute histogram for projected poles data
            xgrid, ygrid = np.mgrid[-1:1:bins * 1j, -1:1:bins * 1j]
            Hs = projections.bin_proj_poles(proj_poles, bins=bins)[0]

            f_width = 4 * n_figs
            f_height = 5
            f = plt.figure(1, figsize=(f_width, f_height))
            gs0 = gridspec.GridSpec(2, 1)
            gs0.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.2)
            gs00 = gridspec.GridSpecFromSubplotSpec(1, n_figs+1, subplot_spec=gs0[0], width_ratios=width_ratios)
            gs01 = gridspec.GridSpecFromSubplotSpec(1, n_figs+1, subplot_spec=gs0[1], width_ratios=width_ratios)
            

            pts_circ = coordgeometry.pts_on_circle(1)
            n_lev = int(bins)
            v_max = np.max(Hs)
            levels = np.linspace(0.0, v_max, n_lev)
            # Plot contour pole figures
            for n in range(n_figs):
                ax = f.add_subplot(gs01[n])
                cax = ax.contourf(xgrid, ygrid, Hs[n], n_lev, antialiased=True, vmin=0.0, vmax=v_max, levels=levels)
                plot_mask_shape(pts_circ)
                ax.set_aspect(1)
                ax.axis('Off')
                ax.set_xticklabels(
                    [axes[0].upper(), '', axes[1].upper(), '', '', '', '', ''])
                ax.set_yticklabels([])
                ax.set_title("{" + poles_lbl[n] + "}", va='bottom')
            cbax = f.add_subplot(gs01[-1])
            cb = Colorbar(ax = cbax, mappable = cax, orientation = 'vertical' )
            cbax.tick_params(labelsize=12)
                

        # Plot scatter pole figures
        for n in range(n_figs):
            ax = f.add_subplot(gs00[n], projection='polar')
            if clr_map is not None:
                cax = ax.scatter(proj_poles[n][0],
                             proj_poles[n][1], edgecolors=clr_map, s=marker_size)
            else:
                cax = ax.scatter(proj_poles[n][0],
                                proj_poles[n][1], s=marker_size)
            
            if proj_poles_theory is not None and cols_th is not None:
                for i in range(12):
                    # print(2*i)
                    cax = ax.scatter(proj_poles_theory[n][0][2*i:2*(i+1)],
                                proj_poles_theory[n][1][2*i:2*(i+1)], s=10, c=cols_th[i])
            elif proj_poles_theory is not None: 
                cax = ax.scatter(proj_poles_theory[n][0],
                            proj_poles_theory[n][1], s=20)
            
            ax.set_rmax(1)
            ax.set_yticklabels([])
            if axes_lbl:
                ax.set_xticklabels(
                    [axes[0].upper(), '', axes[1].upper(), '', '', '', '', ''])
                ax.set_title("{" + poles_lbl[n] + "}", loc='left')
            else:
                ax.set_xticklabels(['', '', '', '', '', '', '', ''])
                ax.set_title(pole_lbl[n], fontsize=12)
            
            if not grid:
                ax.yaxis.grid(False)
                ax.xaxis.grid(False)

    return f


def plot_lattice(lattice_sys, align='cz'):
    """
    Plot lattice in a Cartesian reference frame.

    Parameters
    ----------
    lattice_sys : string, optional
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal, 
        orthorhombic, monoclinic, triclinic.
    align : string
        Alignment option between crystal and orthonormal reference frames. 
        Three options implemented (as described in [1]): 
        - 'ax': a-axis || x-axis and c*-axis || z*-axis
        - 'by': b-axis || y-axis and a*-axis || x*-axis
        - 'cz': c-axis || z-axis and a*-axis || x*-axis [Default]
        where * corresponds to reciprocal lattice vectors. 
    Returns
    -------


    References
    ----------
    [1] Giacovazzo et al.(2002) Fundamentals of Crystallography. Oxf Univ Press. p. 75-76.

    """

    M = lattice.crystal2ortho(lattice_sys, normed=True, degrees=True)

    cell_e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T

    # Crystal vectors in orthonormal basis as column vectors
    cell_ortho = np.dot(M.T, cell_e)
    box = coordgeometry.get_box_xyz(cell_ortho).T

    # Crystal unit vectors in Cartesian coordinates
    pole_vecs = np.stack((np.zeros((cell_ortho.shape)), cell_ortho), axis=0)
    # Labels
    poles_lbl = []
    for i in range(cell_e.shape[1]):
        poles_lbl.append(
            '[' + ''.join([str(int(x)) for x in cell_e[:, i]]) + ']')

    legend_name = lattice_sys + ' cell'
    fig_size = [400, 400]
    clrs = ['red', 'blue', 'orange']

    trace1 = go.Scatter3d(
        x=box[:, 0].ravel(),
        y=box[:, 1].ravel(),
        z=box[:, 2].ravel(),
        mode='lines',
        marker=dict(color='darkgrey'),
        legendgroup=legend_name, name=legend_name)

    traces = []
    data = [trace1]
    for i in range(pole_vecs.shape[2]):
        data.append(
            go.Scatter3d(
                x=pole_vecs[:, 0, i],
                y=pole_vecs[:, 1, i],
                z=pole_vecs[:, 2, i],
                mode='lines',
                marker=dict(color=clrs[i]),
                legendgroup=poles_lbl[i], name=poles_lbl[i])
        )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-0.1, y=-2.5, z=-0.1)
    )

    layout = go.Layout(autosize=False,
                       width=fig_size[0],
                       height=fig_size[1],

                       legend=dict(traceorder='grouped', x=10, bordercolor='#FFFFFF',
                                   borderwidth=10, xanchor='right', yanchor='top'),
                       margin=go.Margin(l=20, r=20, b=20, t=20, pad=20),
                       scene=dict(
                           camera=camera,
                           xaxis=dict(showgrid=False, zeroline=False,
                                      showline=True,
                                      ticks='',
                                      showticklabels=False),
                           yaxis=dict(showgrid=False,
                                      zeroline=True,
                                      showline=False,
                                      ticks='',
                                      showticklabels=False),
                           zaxis=dict(showgrid=False,
                                      zeroline=True,
                                      showline=False,
                                      ticks='',
                                      showticklabels=False)
                       ),
                       )

    f = go.Figure(data=data, layout=layout)

    return f


def plot_crystal_poles(poles, lattice_sys, pole_type, clrs):
    """
    Plot a single crystal and specified poles.

    TODO:
    - 
    """

    M = lattice.crystal2ortho(lattice_sys, normed=True, degrees=True)

    cell_e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T

    # Crystal vectors in orthonormal basis as column vectors
    cell_ortho = np.dot(M.T, cell_e)
    box = coordgeometry.get_box_xyz(cell_ortho).T

    pole_types = ['direction', 'plane-normal']
    proj_poles = []

    if pole_type == 'plane-normal':
        # Convert Miller-Bravais to Miller indices
        if lattice_sys == 'hexagonal' and poles.shape[0] == 4:
            poles = lattice.miller_brav2miller(poles, idx_type='plane')

        # Reciprocal lattice vectors in orthonormal basis
        cell_rec = lattice.reciprocal_lattice_vecs(cell_ortho)
        M_rec = cell_rec        # Column vectors of reciprocal a,b,c
        # Reciprocal lattice vectors for poles (column vectors)
        g_poles = np.dot(cell_rec, poles)

        raise ImplementationError('Plotting of planes and plane-normals is not '
                                  'supported yet.')

    elif pole_type == 'direction':
        # Convert Miller-Bravais to Miller indices
        if lattice_sys == 'hexagonal' and poles.shape[0] == 4:
            poles = lattice.miller_brav2miller(poles, idx_type='direction')

        d_poles = np.dot(M.T, poles)
        pole_vecs = np.stack((np.zeros((d_poles.shape)), d_poles), axis=0)

        legend_name = lattice_sys + ' cell'
        fig_size = [400, 400]

        poles_lbl = []
        for i in range(poles.shape[1]):
            poles_lbl.append(
                '{' + ''.join([str(int(x)) for x in poles[:, i]]) + '}')

        trace1 = go.Scatter3d(
            x=box[:, 0].ravel(),
            y=box[:, 1].ravel(),
            z=box[:, 2].ravel(),
            mode='lines',
            marker=dict(color='darkgrey'),
            legendgroup=legend_name, name=legend_name)

        traces = []
        data = [trace1]
        for i in range(pole_vecs.shape[2]):
            data.append(
                go.Scatter3d(
                    x=pole_vecs[:, 0, i],
                    y=pole_vecs[:, 1, i],
                    z=pole_vecs[:, 2, i],
                    mode='lines',
                    marker=dict(color=clrs[i]),
                    legendgroup=poles_lbl[i], name=poles_lbl[i])
            )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-0.1, y=-2.5, z=-0.1)
        )

        layout = go.Layout(autosize=False,
                           width=fig_size[0],
                           height=fig_size[1],

                           legend=dict(traceorder='grouped', x=10, bordercolor='#FFFFFF',
                                       borderwidth=10, xanchor='right', yanchor='top'),
                           margin=go.Margin(l=20, r=20, b=20, t=20, pad=20),
                           scene=dict(
                               camera=camera,
                               xaxis=dict(showgrid=False, zeroline=False,
                                          showline=False,
                                          ticks='',
                                          showticklabels=False),
                               yaxis=dict(showgrid=False,
                                          zeroline=False,
                                          showline=False,
                                          ticks='',
                                          showticklabels=False),
                               zaxis=dict(showgrid=False,
                                          zeroline=False,
                                          showline=False,
                                          ticks='',
                                          showticklabels=False)
                           ),
                           )

        f = go.Figure(data=data, layout=layout)

    return f


def plot_mask_shape(sh_verts, out=True, ax=None):
    """
    Plot a mask out(in)side a shape defined by `sh_verts`. 

    Parameters
    ----------
    sh_verts : list of tuples
        The coordinates of the shape vertices specified in counter-clockwise 
        direction.
    out : bool
        Specify whether to mask the outside (default) or the inside of the shape.
    ax : matplotlib Axes instance
        The axis where the mask will be plotted. Default is the current axis.

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()

    # Current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Verticies of the plot boundaries in clockwise direction
    plot_verts = [(xlim[0], ylim[0]), (xlim[0], ylim[1]),
                  (xlim[1], ylim[1]), (xlim[1], ylim[0]),
                  (xlim[0], ylim[0])]

    # Specify vertex types
    plot_verts_types = [mpath.Path.MOVETO] + \
        (len(plot_verts) - 1) * [mpath.Path.LINETO]
    sh_verts_types = [mpath.Path.MOVETO] + \
        (len(sh_verts) - 1) * [mpath.Path.LINETO]

    # Create a path and a white patch
    path = mpath.Path(plot_verts + sh_verts, plot_verts_types + sh_verts_types)
    patch = mpatches.PathPatch(path, facecolor='white', edgecolor='none')

    addpatch = ax.add_patch(patch)

    # Reset to original plot limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return addpatch


def get_3d_arrow_plotly(dir, origin, length, head_length=None,
                        head_radius=None, stem_args=None, n_points=100,
                        head_colour=None, opacity=None):
    """
    Get a list of Plotly traces which together represent a 3D arrow.

    Parameters
    ----------
    dir : ndarray of shape (3, )
        Direction vector along which the arrow should point.
    origin : ndarray of shape (3, )
        Origin for the base of the stem of the arrow.
    length : float or int
        Total length of the arrow from base of the stem to the tip of the arrow
        head.
    head_length : float or int, optional
        Length of the arrow head from the tip to the arrow head base. Default
        is None, in which case it will be set to 0.1 * `length`
    head_radius : float or int, optional
        Radius of the base of the arrow head. Default is None, in which case it
        will be set to 0.05 * `length`.
    stem_args : dict, optional
        Specifies the properties of the Plotly line trace used to represent the
        stem of the arrow. Use this to set e.g. `width` and `color`.
    n_points : int, optional
        Number of points to approximate the circular base of the arrow head.
        Default is 100.

    Returns
    -------
    list of Plotly traces

    """

    if head_length is None:
        head_length = length * 0.1

    if head_radius is None:
        head_radius = length * 0.05

    if stem_args is None:
        stem_args = {}

    if stem_args.get('width') is None:
        stem_args['width'] = head_radius * 10

    if stem_args.get('color') is None:
        stem_args['color'] = 'blue'

    if head_colour is None:
        head_colour = 'blue'

    sp = (2 * np.pi) / n_points
    θ = np.linspace(0, (2 * np.pi) - sp, n_points)
    θ_deg = np.rad2deg(θ)

    if opacity is None:
        opacity = 0.5

    # First construct arrow head as pointing in the z-direction
    # with its base on (0,0,0)
    x = head_radius * np.cos(θ)
    y = head_radius * np.sin(θ)

    # Arrow head base:
    x1 = np.hstack(([0], x))
    y1 = np.hstack(([0], y))
    z1 = np.zeros(x.shape[0] + 1)
    ah_base = np.vstack([x1, y1, z1])

    # Arrow head cone:
    x2 = np.copy(x1)
    y2 = np.copy(y1)
    z2 = np.copy(z1)
    z2[0] = head_length
    ah_cone = np.vstack([x2, y2, z2])

    # Rotate arrow head so that it points in `dir`
    dir_unit = dir / np.linalg.norm(dir)
    z_unit = np.array([0, 0, 1])

    # print('dir_unit: \n{}\n'.format(dir_unit))
    # print('z_unit: \n{}\n'.format(z_unit))
    # print('origin: \n{}\n'.format(origin))

    if np.allclose(z_unit, dir_unit):
        rot_ax = np.array([1, 0, 0])
        rot_an = 0

    elif np.allclose(-z_unit, dir_unit):
        rot_ax = np.array([1, 0, 0])
        rot_an = np.pi

    else:

        rot_ax = np.cross(z_unit, dir_unit)
        rot_an = numutils.col_wise_angles(
            dir_unit[:, np.newaxis], z_unit[:, np.newaxis])[0]

    rot_an_deg = np.rad2deg(rot_an)
    rot_mat = rotations.ax_ang2rot_mat(rot_ax, rot_an)[0]

    # Reorient arrow head and translate
    stick_length = length - head_length
    ah_translate = (origin + (stick_length * dir_unit))
    ah_base_dir = np.dot(rot_mat, ah_base) + ah_translate[:, np.newaxis]
    ah_cone_dir = np.dot(rot_mat, ah_cone) + ah_translate[:, np.newaxis]

    i = np.zeros(x1.shape[0] - 1, dtype=int)
    j = np.arange(1, x1.shape[0])
    k = np.roll(np.arange(1, x1.shape[0]), 1)

    data = [
        {
            'type': 'mesh3d',
            'x': ah_base_dir[0],
            'y': ah_base_dir[1],
            'z': ah_base_dir[2],
            'i': i,
            'j': j,
            'k': k,
            'hoverinfo': 'none',
            'color': head_colour,
            'opacity': opacity,
        },
        {
            'type': 'mesh3d',
            'x': ah_cone_dir[0],
            'y': ah_cone_dir[1],
            'z': ah_cone_dir[2],
            'i': i,
            'j': j,
            'k': k,
            'hoverinfo': 'none',
            'color': head_colour,
            'opacity': opacity,
        },
        {
            'type': 'scatter3d',
            'x': [origin[0], ah_translate[0]],
            'y': [origin[1], ah_translate[1]],
            'z': [origin[2], ah_translate[2]],
            'hoverinfo': 'none',
            'mode': 'lines',
            'line': stem_args,
            'projection': {
                'x': {
                    'show': False
                }
            }
        },
    ]

    return data
