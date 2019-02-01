"""Module for interacting with data output from Dream.3D in HDF format."""

import numpy as np
import yaml
import h5py

from plotly.offline import plot, iplot, init_notebook_mode
from plotly import graph_objs as go

from crystex import rotations
from crystex import numutils
from crystex import coordgeometry
from crystex import plots
from crystex.numutils import get_from_dict, index_lst


class Mesh(object):
    """Class to represent a surface/interface mesh"""

    def __init__(self, vert, mesh_idx, cent, region=None):

        self.vert = vert
        self.mesh_idx = mesh_idx
        self.cent = cent
        self.region = region

        # These are assigned if `fit_plane` is invoked.
        self.mean = None
        self.plane_equation = None
        self.normal = None
        self.fit_to = None

    def fit_plane(self, fit_to='centroids'):

        if fit_to == 'centroids':
            points = self.cent
        elif fit_to == 'vertices':
            points = self.vert

        (a, b, c), mean = numutils.fit_plane(points)

        self.mean = mean
        self.plane_equation = np.array([a, b, c])
        self.normal = np.array([[a, b, -1]]).T
        self.fit_to = points

    def get_plane_points(self, x, y):

        if self.plane_equation is None:
            raise ValueError('There is no fitted plane. Invoke `fit_plane` and'
                             ' try again.')

        z = numutils.plane(x, y, *self.plane_equation)
        return z

    def visualise(self, do_iplot=False, plane_args=None, invert_normal=False,
                  normal_args=None, mesh_args=None, plane_lims_pad=None,
                  show=None, add_arrows=None):

        allowed_show = [
            'plane',
            'mesh',
            'normal',
            'grid',
            'region',
        ]

        if show is None:
            show = []
        if add_arrows is None:
            add_arrows = []
        if plane_args is None:
            plane_args = {}
        if mesh_args is None:
            mesh_args = {}
        if normal_args is None:
            normal_args = {}

        plane_args_def = {
            'opacity': 0.8,
        }
        mesh_args_deg = {
            'opacity': 0.8,
        }
        normal_args_def = {
            'length': 1,
            'head_length': 2,
            'head_radius': 2,
            'stem_args': {
                'width': 5,
            }
        }

        plane_args = {**plane_args_def, **plane_args}
        normal_args = {**normal_args_def, **normal_args}

        for show_obj in show:
            if show_obj not in allowed_show:
                raise ValueError('"{}" is not an allowed `show` string, must '
                                 'be one of: {}'.format(show_obj, allowed_show))

        traces = [
            {
                'type': 'scatter3d',
                'x': self.vert[0],
                'y': self.vert[1],
                'z': self.vert[2],
                'mode': 'markers',
                'name': 'Vertices',
                'marker': {
                    'size': 3,
                },
            },
            {
                'type': 'scatter3d',
                'x': self.cent[0],
                'y': self.cent[1],
                'z': self.cent[2],
                'mode': 'markers',
                'name': 'Centroids',
                'marker': {
                    'size': 3,
                },
            },
        ]

        if 'mesh' in show:
            traces.append({
                'type': 'mesh3d',
                'x': self.vert[0],
                'y': self.vert[1],
                'z': self.vert[2],
                'i': self.mesh_idx[0],
                'j': self.mesh_idx[1],
                'k': self.mesh_idx[2],
                **mesh_args,
            })

        if 'plane' in show or 'normal' in show:

            if self.plane_equation is None:
                raise ValueError('There is no plane/normal to show. Invoke '
                                 '`fit_plane` and try again.')

            if 'plane' in show:

                # Get coordinates for fitted plane mesh
                plane_xy = coordgeometry.get_xy_bounding_trace(
                    self.fit_to[0] - self.mean[0, 0],
                    self.fit_to[1] - self.mean[1, 0])

                if plane_lims_pad is not None:
                    plane_xy[0, [0, 3, 4]] -= plane_lims_pad[0]
                    plane_xy[0, [1, 2]] += plane_lims_pad[0]

                    plane_xy[1, [0, 1, 0]] -= plane_lims_pad[1]
                    plane_xy[1, [2, 3]] += plane_lims_pad[1]

                plane_z = self.get_plane_points(*plane_xy)

                plane_trace = {
                    'type': 'mesh3d',
                    'x': plane_xy[0] + self.mean[0, 0],
                    'y': plane_xy[1] + self.mean[1, 0],
                    'z': plane_z + self.mean[2],
                    **plane_args,
                }
                traces.append(plane_trace)

            if 'normal' in show:

                scale_norm = -1 if invert_normal else 1
                norm_dir = scale_norm * self.normal[:, 0]

                norm_arrow_args = {
                    'dir': norm_dir,
                    'origin': self.mean[:, 0],
                    **normal_args,
                }

                norm_arrow_traces = plots.get_3d_arrow_plotly(
                    **norm_arrow_args)

                traces.extend(norm_arrow_traces)

        if 'region' in show:

            if self.region is None:
                raise ValueError('No region has been assigned for this Mesh.')

            region_xyz = coordgeometry.get_box_xyz(
                self.region['edges'], origin=self.region['origin'])[0]

            region_trace = {
                'type': 'scatter3d',
                'mode': 'lines',
                'x': region_xyz[0],
                'y': region_xyz[1],
                'z': region_xyz[2],
                'line': {
                    'color': 'black',
                },
                'name': 'Region',
            }
            traces.append(region_trace)

        layout = {
            'scene': {
                'aspectmode': 'data'
            },
            'height': 800,
            'width': 800,
            'showlegend': True,
        }

        for arrow_spec in add_arrows:

            add_arrow_args = {
                **arrow_spec,
                'origin': self.mean[:, 0],
            }

            add_arrow_traces = plots.get_3d_arrow_plotly(
                **add_arrow_args)

            traces.extend(add_arrow_traces)

        if 'grid' not in show:

            hide_ax_props = dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                title='',
            )

            layout['scene'].update({
                'aspectmode': 'data',
                'xaxis': {
                    **hide_ax_props,
                },
                'yaxis': {
                    **hide_ax_props,
                },
                'zaxis': {
                    **hide_ax_props,
                },
            })

        fig = go.Figure(data=traces, layout=layout)

        if do_iplot:
            init_notebook_mode()
            iplot(fig)

        return fig


class Dream3d(object):
    """Class to represent the output from a Dream.3d pipeline."""

    def __init__(self, opt, tri_data=False):

        d3d = h5py.File(opt['filepath'], 'r')

        img_data_path = [opt['all_data'], opt['img_data']]       
        cell_feat_data_path = img_data_path + [opt['cell_feat_data']]
        
        self.all_neighbours = None
        self.grain_ids = None
        self.neighbours = None

        num_neigh_path = cell_feat_data_path + [opt['neigh_num']]
        eulers_path = cell_feat_data_path + [opt['avg_euler']]
        num_grn_elms_path = cell_feat_data_path + [opt['num_elements']]
        grn_phases_path = cell_feat_data_path + [opt['gr_phases']]
        elm_grn_ids_path = img_data_path + \
            [opt['cell_data'], opt['cell_feat_id']]

        self.element_grain_ids = np.array(get_from_dict(d3d, elm_grn_ids_path))
        self.num_neighbours = np.array(get_from_dict(d3d, num_neigh_path))
        self.grain_phases =  np.array(get_from_dict(d3d, grn_phases_path))

        if opt.get('neigh_list') is not None:
            all_neigh_path = cell_feat_data_path + [opt['neigh_list']]
            self.all_neighbours = np.array(get_from_dict(d3d, all_neigh_path))
            self.set_grain_neighbours()
        self.eulers = np.array(get_from_dict(d3d, eulers_path))

        if tri_data:
            tri_data_path = [opt['all_data'], opt['tri_data']]
            face_data_path = tri_data_path + [opt['face_data']]
            tri_geom_path = tri_data_path + [opt['tri_geom']]
            
            face_labs_path = face_data_path + [opt['face_labels']]
            face_areas_path = face_data_path + [opt['face_areas']]
            face_cent_path = face_data_path + [opt['face_centroids']]
            shared_tri_path = tri_geom_path + [opt['tri_list']]
            shared_vert_path = tri_geom_path + [opt['vert_list']]

            self.face_labels = np.array(get_from_dict(d3d, face_labs_path))
            self.face_areas = np.array(get_from_dict(d3d, face_areas_path))
            self.face_centroids = np.array(get_from_dict(d3d, face_cent_path))
            self.shared_tri = np.array(get_from_dict(d3d, shared_tri_path))
            self.shared_vert = np.array(get_from_dict(d3d, shared_vert_path))

            self.num_grain_elements = np.array(
                get_from_dict(d3d, num_grn_elms_path)[:, 0])

    @classmethod
    def from_file_options(cls, opt_file_path):

        with open(opt_file_path, 'r') as f:
            opt = yaml.safe_load(f)

        return cls(opt)

    def set_grain_neighbours(self):

        if all([i is not None for i in [self.neighbours,  self.grain_ids]]):
            raise ValueError('Grain neighbours are already set.')

        # Identify unique grain IDs
        self.grain_ids = np.array(list(
            set(self.element_grain_ids.flatten())
        ))

        # Identify grain neighours
        neighbours = []
        for i in self._get_grain_neighbours_idx(self.num_neighbours):
            neighbours.append(np.array(index_lst(self.all_neighbours, i)))

        self.neighbours = neighbours

    def _get_grain_neighbours_idx(self, num_neighbours):
        """
        Get the grain IDs which neighbour a given grain ID.

        Parameters
        ----------
        num_neighbours : list of int
            The number of neighbours for each grain ID.

        Returns
        -------
        list of list of int
            The grain IDs which neighbour a given grain.

        """

        neigh_idx = []
        cumtot = 0
        for nn in num_neighbours:

            start_idx = cumtot
            cumtot += nn[0]
            end_idx = cumtot
            neigh_idx.append(np.arange(start_idx, end_idx))

        return neigh_idx

    def get_popular_grains(self, n):
        """
        Return the grain IDs of those grains with more than `n` neighbours.

        Parameters
        ----------
        n : int
            Number of neighbours to check for.

        Returns
        -------
        list of int
            Grain IDs of those grains with more than `n` neighbours.

        """

        out = []
        for i_idx, i in enumerate(self.neighbours):
            if len(i) > n:
                out.append(i_idx)

        return out

    def get_grain_mesh(self, grain_id):
        """
        Get the vertices and the vertex indices which form a mesh around a 
        grain.

        Parameters
        ----------
        grain_id : int

        Returns
        -------
        vert : ndarray of float of shape (3, N)
        mesh : ndarray of int of shape (3, M)
        cent : ndarray of float of shape (3, P)

        """

        face_labels_i = self.get_grain_face_labels(grain_id)
        face_tri_i = self.shared_tri[face_labels_i]
        face_centroids_i = self.face_centroids[face_labels_i]

        uniq_tri, uniq_tri_inv = np.unique(face_tri_i, return_inverse=True)

        mesh = uniq_tri_inv.reshape(face_tri_i.shape)
        vert = self.shared_vert[uniq_tri]

        return vert.T, mesh.T, face_centroids_i.T

    def get_grain_vertices(self, grain_id):

        face_labels_i = self.get_grain_face_labels(grain_id)
        face_tri_i = self.shared_tri[face_labels_i]
        face_vertex_i = self.shared_vert[np.unique(face_tri_i)]

        return face_vertex_i

    def get_grain_face_labels(self, grain_id):

        face_labels = self.face_labels
        face_labels_i = np.where(np.any(face_labels == grain_id, axis=1))[0]
        return face_labels_i

    def get_gb_mesh(self, grain_ids):

        face_labels_i = self.get_gb_face_labels(grain_ids)
        face_centroids_i = self.face_centroids[face_labels_i]

        ijk = self.shared_tri[face_labels_i]
        ui, uir = np.unique(ijk, return_inverse=True)
        ijk_new = uir.reshape(ijk.shape)
        xyz_new = self.shared_vert[ui]

        return xyz_new, ijk_new, face_centroids_i

    def get_gb_face_labels(self, grain_ids):

        return np.where(np.all(self.face_labels == grain_ids, axis=1))[0]

    def get_grains(self, min_neighbours=None, min_elements=None):
        """
        Get the grain IDs of grains with particular properties.

        Parameters
        ----------
        num_neighbours : int, optional
            Return grains with a minimum number of neighbours.abs
        num_elements : int, optional
            Return grains with a minimum number of elements (pixels/voxels).        

        Returns
        -------
        list of int
            Grain IDs.

        """
        raise NotImplementedError('Too lazy.')

    def isolate_mesh_region(self, grain_id, region):
        """
        Isolate the mesh of a given grain according to a region.

        Parameters
        ----------
        grain_id : int
            The grain ID of the mesh to isolate
        region : dict
            Parameterises a region in space (as a parallelepiped) over which to
            isolate the mesh. A dict with keys:
                edges : ndarray of shape (3, 3)
                    Array of column vectors representing the edge vectors of
                    the parallelopiped.
                origin : ndarray of shape (3, 1)
                    Column vector representing the origin of the 
                    parallelepiped.
                rotations : list of tuple
                    List of (rotation axis index, angle in degrees), where the
                    rotation axis index is 0, 1 or 2 and represents the axis
                    of the parallelepiped about which to (intrinsically) 
                    rotate the parallelepiped by the given angle. The reason
                    a list is used is to allow fine tuning of the region.
        """

        def rotate_box(box, ax_idx, ang):
            """Rotate the box about one of its vectors"""

            r = rotations.ax_ang2rot_mat(box[:, ax_idx], ang, degrees=True)[0]
            box = r @ box

            return box

        def in_box(points, box, box_origin):

            points = points - box_origin
            points_frac = np.linalg.inv(box) @ points
            in_idx_all = np.logical_and(points_frac <= 1, points_frac > 0)
            in_idx = np.where(np.all(in_idx_all, axis=0))[0]

            return in_idx

        def get_points_in_box(points, box, box_origin):

            in_idx = in_box(points, box, box_origin)
            out_idx = np.setdiff1d(np.arange(points.shape[1]), in_idx)
            points_in = points[:, in_idx]
            points_out = points[:, out_idx]

            return points_in, in_idx, points_out, out_idx

        vert, mesh, cent = self.get_grain_mesh(grain_id)

        box = region['edges']
        box_org = region['origin']
        for box_rots in region['rotations']:
            box = rotate_box(box, *box_rots)

        region['edges'] = box

        vin, vin_idx, vout, vout_idx = get_points_in_box(vert, box, box_org)
        cin, cin_idx, cout, cout_idx = get_points_in_box(cent, box, box_org)

        # Reduce the mesh to only that with vertices within the box
        mesh_flat = mesh.reshape(-1)

        # Get mesh (triangles) which index vertices outside the box
        mesh_is_out = np.in1d(mesh_flat, vout_idx).reshape(mesh.shape)
        mesh_out_idx = np.where(np.any(mesh_is_out, axis=0))[0]
        mesh_in_idx = np.setdiff1d(np.arange(mesh.shape[1]), mesh_out_idx)
        mesh_in = mesh[:, mesh_in_idx]

        mesh_uniq, mesh_uniq_inv = np.unique(mesh_in, return_inverse=True)
        mesh_in_tri = mesh_uniq_inv.reshape(mesh_in.shape)
        vert_in_tri = vert[:, mesh_uniq]

        ret = Mesh(vert_in_tri, mesh_in_tri, cin, region=region)
        return ret

    def visualise_grains(self, grain_ids, cols=None, opacities=None,
                         do_iplot=False, show_grid=True):
        """Visualise grains using Plotly."""

        traces = []
        for gid_idx, gid in enumerate(grain_ids):

            # Get vertices associated with this grain
            vert, mesh, cen = self.get_grain_mesh(gid)

            mesh_trace = {
                'type': 'mesh3d',
                'x': vert[0],
                'y': vert[1],
                'z': vert[2],
                'i': mesh[0],
                'j': mesh[1],
                'k': mesh[2],
                'opacity': 1,
                'name': 'Grain ID: {}'.format(gid)
            }

            if cols is not None:
                mesh_trace.update({
                    'color': cols[gid_idx]
                })

            if opacities is not None:
                mesh_trace.update({
                    'opacity': opacities[gid_idx]
                })

            traces.append(mesh_trace)

        layout = {
            'scene': {
                'aspectmode': 'data'
            },
            'height': 1100,
            'width': 1100,
            'showlegend': True,
        }

        if not show_grid:

            hide_ax_props = dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                title='',
            )

            layout['scene'].update({
                'aspectmode': 'data',
                'xaxis': {
                    **hide_ax_props,
                },
                'yaxis': {
                    **hide_ax_props,
                },
                'zaxis': {
                    **hide_ax_props,
                },
            })

        fig = go.Figure(data=traces, layout=layout)

        if do_iplot:
            init_notebook_mode()
            iplot(fig)

        return fig
