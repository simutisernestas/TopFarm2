import numpy as np
from numpy import newaxis as na
from scipy.spatial import ConvexHull
from topfarm.constraint_components import Constraint, ConstraintComponent
from topfarm.utils import smooth_max, smooth_max_gradient, is_number
import topfarm
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.pyplot import Circle


class XYBoundaryConstraint(Constraint):

    def __init__(self, boundary, boundary_type='convex_hull', units=None, relaxation=False, turbine_diameter=None, **kwargs):
        """Initialize XYBoundaryConstraint

        Parameters
        ----------
        boundary : array_like (n,2) or list of tuples (array_like (n,2), boolean)
            boundary coordinates. If boundary is array_like (n,2) it indicates a single boundary and can be used with
            boundary types: 'convex_hull', 'polygon', 'rectangle','square'. If boundary is list of tuples (array_like (n,2), boolean),
            it is multiple boundaries where the boolean is 1 for inclusion zones and 0 for exclusion zones and can be used with the
            boundary type: 'multi_polygon'.
        boundary_type : 'convex_hull', 'polygon', 'rectangle','square'
            - 'convex_hull' (default): Convex hul around boundary points\n
            - 'polygon': Polygon boundary (may be non convex). Less suitable for gradient-based optimization\n
            - 'rectangle': Smallest axis-aligned rectangle covering the boundary points\n
            - 'square': Smallest axis-aligned square covering the boundary points
            - 'multi_polygon': Mulitple polygon boundaries incl. exclusion zones (may be non convex).\n
            - 'turbine_specific': Set of multiple polygon boundaries that depend on the wind turbine type. \n
        turbine_diameter : float, optional
            Turbine diameter, used for scaling the boundary coordinates.
            If None, no scaling is applied. Default is None.

        """
        self.turbine_diameter = turbine_diameter
        if boundary_type == 'multi_polygon':
            self.zones_original = boundary # Keep original zones if needed for other logic
            self.zones = []
            for z_orig in boundary: # Assuming boundary is a list of Zone objects
                scaled_b = np.asarray(z_orig.boundary)
                if self.turbine_diameter is not None and self.turbine_diameter > 0:
                    scaled_b = scaled_b / self.turbine_diameter

                if isinstance(z_orig, InclusionZone):
                    new_zone = InclusionZone(scaled_b, z_orig.dist2wt, z_orig.geometry_type, z_orig.name)
                elif isinstance(z_orig, ExclusionZone):
                    new_zone = ExclusionZone(scaled_b, z_orig.dist2wt, z_orig.geometry_type, z_orig.name)
                else:
                    warnings.warn(f"Scaling for zone type {type(z_orig)} not fully implemented for multi_polygon. Boundary scaled, other attributes preserved.")
                    new_zone = Zone(scaled_b, z_orig.dist2wt, z_orig.geometry_type, z_orig.incl, z_orig.name)
                self.zones.append(new_zone)
            self.boundary = np.asarray(self.zones[0].boundary) if self.zones else np.array([])

        elif boundary_type == 'turbine_specific':
            self.zones_original = boundary
            self.zones = []
            for z_orig in boundary:
                scaled_b = np.asarray(z_orig.boundary)
                if self.turbine_diameter is not None and self.turbine_diameter > 0:
                    scaled_b = scaled_b / self.turbine_diameter

                if isinstance(z_orig, InclusionZone):
                    new_zone = InclusionZone(scaled_b, z_orig.dist2wt, z_orig.geometry_type, z_orig.name)
                elif isinstance(z_orig, ExclusionZone):
                    new_zone = ExclusionZone(scaled_b, z_orig.dist2wt, z_orig.geometry_type, z_orig.name)
                else:
                    warnings.warn(f"Scaling for zone type {type(z_orig)} not fully implemented for turbine_specific. Boundary scaled, other attributes preserved.")
                    new_zone = Zone(scaled_b, z_orig.dist2wt, z_orig.geometry_type, z_orig.incl, z_orig.name)
                self.zones.append(new_zone)
            self.boundary = np.asarray(self.zones[0].boundary) if self.zones else np.array([])
            assert 'turbines' in list(kwargs)
            self.turbines = kwargs['turbines']
        else:
            self.boundary = np.asarray(boundary)
            if self.turbine_diameter is not None and self.turbine_diameter > 0:
                self.boundary = self.boundary / self.turbine_diameter

        self.boundary_type = boundary_type
        self.const_id = 'xyboundary_comp_{}'.format(boundary_type)
        self.units = units
        self.relaxation = relaxation

    def get_comp(self, n_wt, turbine_diameter_to_use=None): # Added turbine_diameter_to_use
        # If turbine_diameter_to_use is provided, it takes precedence. Otherwise, use self.turbine_diameter (from __init__).
        td = turbine_diameter_to_use if turbine_diameter_to_use is not None else self.turbine_diameter

        if not hasattr(self, 'boundary_comp') or (turbine_diameter_to_use is not None and td != self.boundary_comp.turbine_diameter):
            # Re-initialize if component does not exist or if a new turbine_diameter is provided that differs
            if self.boundary_type == 'polygon':
                self.boundary_comp = PolygonBoundaryComp(
                    n_wt, self.boundary, self.const_id, self.units, self.relaxation, turbine_diameter=td)
            elif self.boundary_type == 'multi_polygon':
                self.boundary_comp = MultiPolygonBoundaryComp(n_wt, self.zones, const_id=self.const_id, units=self.units, relaxation=self.relaxation, turbine_diameter=td)
            elif self.boundary_type == 'turbine_specific':
                self.boundary_comp = TurbineSpecificBoundaryComp(n_wt, self.turbines, self.zones, const_id=self.const_id, units=self.units, relaxation=self.relaxation, turbine_diameter=td)
            else: # convex_hull, rectangle, square
                self.boundary_comp = ConvexBoundaryComp(n_wt, self.boundary, self.boundary_type, self.const_id, self.units, turbine_diameter=td)
        return self.boundary_comp

    @property
    def constraintComponent(self):
        assert hasattr(
            self, "boundary_comp"
        ), "Boundary component not initialized, call setup first"
        return self.boundary_comp

    def set_design_var_limits(self, design_vars):
        if self.boundary_type in ['multi_polygon', 'turbine_specific']:
            bound_min = np.vstack([(bound).min(0) for bound, _ in self.boundary_comp.boundaries]).min(0)
            bound_max = np.vstack([(bound).max(0) for bound, _ in self.boundary_comp.boundaries]).max(0)
        else:
            bound_min = self.boundary_comp.xy_boundary.min(0)
            bound_max = self.boundary_comp.xy_boundary.max(0)
        for k, l, u in zip([topfarm.x_key, topfarm.y_key], bound_min, bound_max):
            if k in design_vars:
                if len(design_vars[k]) == 4:
                    design_vars[k] = (design_vars[k][0], np.maximum(design_vars[k][1], l),
                                      np.minimum(design_vars[k][2], u), design_vars[k][-1])
                else:
                    design_vars[k] = (design_vars[k][0], l, u, design_vars[k][-1])

    def _setup(self, problem, group='constraint_group', **kwargs):
        n_wt = problem.n_wt
        # Get turbine_diameter from kwargs if provided by TopFarmProblem
        td_from_problem = kwargs.get('turbine_diameter')

        # Pass the turbine_diameter to get_comp.
        # self.turbine_diameter (from __init__) is used if td_from_problem is None by get_comp's logic.
        self.boundary_comp = self.get_comp(n_wt, turbine_diameter_to_use=td_from_problem)
        self.boundary_comp.problem = problem
        self.set_design_var_limits(problem.design_vars)

        # Ensure xy_boundary exists and is not empty before adding to indeps
        if hasattr(self.boundary_comp, 'xy_boundary') and self.boundary_comp.xy_boundary is not None and self.boundary_comp.xy_boundary.size > 0:
            problem.indeps.add_output('xy_boundary', self.boundary_comp.xy_boundary)
        else:
            # Fallback or warning if no boundary to output, common for circle if not explicitly made polygonal by comp
            if self.boundary_type == 'circle': # CircleBoundaryComp might not set a plottable xy_boundary in its base comp
                 if hasattr(self.boundary_comp,'center') and hasattr(self.boundary_comp,'radius'): # CircleBoundaryComp specific
                      t = np.linspace(0,2*np.pi, 100)
                      auto_circle_boundary = self.boundary_comp.center + \
                                             np.array([np.cos(t), np.sin(t)]).T * self.boundary_comp.radius
                      if auto_circle_boundary.size > 0 :
                           problem.indeps.add_output('xy_boundary', auto_circle_boundary, shape_by_conn=True, desc="Auto-generated boundary for circle plotting")
            else:
                 warnings.warn(f"XYBoundaryConstraint ({self.const_id}): xy_boundary is None or empty, not adding to indeps.")

        getattr(problem.model, group).add_subsystem(self.const_id, self.boundary_comp, promotes=['*'])


    def setup_as_constraint(self, problem, group='constraint_group', **kwargs): # Added **kwargs
        self._setup(problem, group=group, **kwargs) # Pass **kwargs
        if problem.n_wt == 1 and not isinstance(self.boundary_comp, MultiConvexBoundaryComp): # MultiConvex has per-edge constraints
            lower = 0
        else:
            lower = self.boundary_comp.zeros
        problem.model.add_constraint('boundaryDistances', lower=lower)

    def setup_as_penalty(self, problem, group='constraint_group', **kwargs): # Added **kwargs
        self._setup(problem, group=group, **kwargs) # Pass **kwargs


class CircleBoundaryConstraint(XYBoundaryConstraint):
    def __init__(self, center, radius, turbine_diameter=None):
        """Initialize CircleBoundaryConstraint

        Parameters
        ----------
        center : (float, float)
            center position (x,y)
        radius : int or float
            circle radius
        turbine_diameter : float, optional
            Turbine diameter for scaling. Default is None.
        """
        self.turbine_diameter = turbine_diameter # Store it
        self.center_original = np.array(center)
        self.radius_original = radius

        if self.turbine_diameter is not None and self.turbine_diameter > 0:
            self.center = self.center_original / self.turbine_diameter
            self.radius = self.radius_original / self.turbine_diameter
        else:
            self.center = self.center_original
            self.radius = self.radius_original

        # Essential attributes for XYBoundaryConstraint compatibility if its methods are called (e.g. _setup)
        # However, CircleBoundaryConstraint overrides get_comp and set_design_var_limits.
        # We still need const_id for the component.
        # No direct call to XYBoundaryConstraint.__init__ to avoid its boundary processing logic.
        self.boundary_type = 'circle' # Implicit
        self.units = None # Or inherit if XYBoundaryConstraint had it passed via kwargs
        self.relaxation = False # Or inherit

        # Create a const_id using potentially scaled center and radius for uniqueness if desired,
        # but ensure it's a valid OpenMDAO component name (no dots, etc.).
        # Using original values might be more stable if scaling is dynamic.
        # For now, use scaled, but ensure clean names.
        center_str = '_'.join([f"{c:.2f}".replace('.', 'p') for c in self.center])
        radius_str = f"{self.radius:.2f}".replace('.', 'p')
        self.const_id = f'circle_boundary_comp_{center_str}_{radius_str}'
        # Further clean if still needed: self.const_id = self.const_id.replace('-', 'm')


    def get_comp(self, n_wt, turbine_diameter_to_use=None): # Added turbine_diameter_to_use
        # If turbine_diameter_to_use is provided, it implies a potential change or override.
        # CircleBoundaryConstraint's __init__ already scales self.center and self.radius based on self.turbine_diameter.
        # If turbine_diameter_to_use is different, we might need to re-scale or re-initialize.
        # For now, assume self.center and self.radius are correctly scaled based on self.turbine_diameter (from __init__).
        # The turbine_diameter_to_use is passed for consistency to the component.

        td = turbine_diameter_to_use if turbine_diameter_to_use is not None else self.turbine_diameter

        # Re-initialize if component doesn't exist or if turbine_diameter context changes.
        if not hasattr(self, 'boundary_comp') or (turbine_diameter_to_use is not None and td != self.boundary_comp.turbine_diameter):
            # Pass the (already scaled by __init__) center and radius.
            self.boundary_comp = CircleBoundaryComp(n_wt, self.center, self.radius, self.const_id, units=self.units, turbine_diameter=td)
        return self.boundary_comp

    def set_design_var_limits(self, design_vars):
        for k, l, u in zip([topfarm.x_key, topfarm.y_key],
                           self.center - self.radius,
                           self.center + self.radius):
            if len(design_vars[k]) == 4:
                design_vars[k] = (design_vars[k][0], np.maximum(design_vars[k][1], l),
                                  np.minimum(design_vars[k][2], u), design_vars[k][-1])
            else:
                design_vars[k] = (design_vars[k][0], l, u, design_vars[k][-1])


class BoundaryBaseComp(ConstraintComponent):
    def __init__(self, n_wt, xy_boundary=None, const_id=None, units=None, relaxation=False, turbine_diameter=None, **kwargs):
        super().__init__(**kwargs)
        self.n_wt = n_wt
        self.turbine_diameter = turbine_diameter # Store for potential use or consistency
        self.xy_boundary = np.array(xy_boundary)
        # Scaling of xy_boundary is assumed to be done by the calling Constraint class (e.g. XYBoundaryConstraint)
        # before this component is initialized.
        self.const_id = const_id
        self.units = units
        self.relaxation = relaxation
        if xy_boundary is not None and self.xy_boundary.size > 0 and np.any(self.xy_boundary[0] != self.xy_boundary[-1]):
            self.xy_boundary = np.r_[self.xy_boundary, self.xy_boundary[:1]]

    def setup(self):
        # Explicitly size input arrays
        self.add_input(topfarm.x_key, np.zeros(self.n_wt),
                       desc='x coordinates of turbines in global ref. frame', units=self.units)
        self.add_input(topfarm.y_key, np.zeros(self.n_wt),
                       desc='y coordinates of turbines in global ref. frame', units=self.units)
        if self.relaxation:
            self.add_input('time', 0)
        if hasattr(self, 'types'):
            self.add_input('type', np.zeros(self.n_wt))
        # self.add_output('constraint_violation_' + self.const_id, val=0.0)
        # Explicitly size output array
        # (vector with positive elements if turbines outside of hull)
        self.add_output('boundaryDistances', self.zeros,
                        desc="signed perpendicular distances from each turbine to each face CCW; + is inside")
        self.declare_partials('boundaryDistances', [topfarm.x_key, topfarm.y_key])
        if self.relaxation:
            self.declare_partials('boundaryDistances', 'time')

        # self.declare_partials('boundaryDistances', ['boundaryVertices', 'boundaryNormals'], method='fd')

    def compute(self, inputs, outputs):
        # calculate distances from each point to each face
        args = {x: inputs[x] for x in [topfarm.x_key, topfarm.y_key, topfarm.type_key] if x in inputs}
        boundaryDistances = self.distances(**args)
        outputs['boundaryDistances'] = boundaryDistances
        # outputs['constraint_violation_' + self.const_id] = np.sum(np.minimum(boundaryDistances, 0) ** 2)

    def compute_partials(self, inputs, partials):
        # return Jacobian dict
        if not self.relaxation:
            dx, dy = self.gradients(**{xy: inputs[k] for xy, k in zip('xy', [topfarm.x_key, topfarm.y_key])})
        else:
            dx, dy, dt = self.gradients(**{xy: inputs[k] for xy, k in zip('xy', [topfarm.x_key, topfarm.y_key])})

        partials['boundaryDistances', topfarm.x_key] = dx
        partials['boundaryDistances', topfarm.y_key] = dy
        if self.relaxation:
            partials['boundaryDistances', 'time'] = dt

    def plot(self, ax):
        """Plot boundary"""
        ax.plot(
            self.xy_boundary[:, 0].tolist() + [self.xy_boundary[0, 0]],
            self.xy_boundary[:, 1].tolist() + [self.xy_boundary[0, 1]],
            "k",
            linewidth=1,
        )


class ConvexBoundaryComp(BoundaryBaseComp):
    def __init__(self, n_wt, xy_boundary=None, boundary_type='convex_hull', const_id=None, units=None, turbine_diameter=None):
        self.boundary_type = boundary_type
        # xy_boundary is assumed to be already scaled if turbine_diameter was provided to XYBoundaryConstraint.
        # No further scaling needed here based on turbine_diameter argument itself.
        self.calculate_boundary_and_normals(xy_boundary) # operates on (potentially scaled) xy_boundary
        super().__init__(n_wt, self.xy_boundary, const_id, units, relaxation=False, turbine_diameter=turbine_diameter) # Added relaxation=False for consistency if not passed by XYBC for this comp type
        self.calculate_gradients() # operates on self.xy_boundary and self.unit_normals (derived from scaled)
        self.zeros = np.zeros([self.n_wt, self.nVertices])

    def calculate_boundary_and_normals(self, xy_boundary):
        xy_boundary = np.asarray(xy_boundary)
        if self.boundary_type == 'convex_hull':
            # find the points that actually comprise a convex hull
            hull = ConvexHull(list(xy_boundary))

            # keep only xy_vertices that actually comprise a convex hull and arrange in CCW order
            self.xy_boundary = xy_boundary[hull.vertices]
        elif self.boundary_type == 'square':
            min_ = xy_boundary.min(0)
            max_ = xy_boundary.max(0)
            range_ = (max_ - min_)
            x_c, y_c = min_ + range_ / 2
            r = range_.max() / 2
            self.xy_boundary = np.array([(x_c - r, y_c - r), (x_c + r, y_c - r),
                                         (x_c + r, y_c + r), (x_c - r, y_c + r)])
        elif self.boundary_type == 'rectangle':
            min_ = xy_boundary.min(0)
            max_ = xy_boundary.max(0)
            range_ = (max_ - min_)
            x_c, y_c = min_ + range_ / 2
            r = range_ / 2
            self.xy_boundary = np.array([(x_c - r[0], y_c - r[1]), (x_c + r[0], y_c - r[1]),
                                         (x_c + r[0], y_c + r[1]), (x_c - r[0], y_c + r[1])])
        else:
            raise NotImplementedError("Boundary type '%s' is not implemented" % self.boundary_type)

        # get the real number of xy_vertices
        self.nVertices = self.xy_boundary.shape[0]

        # initialize normals array
        unit_normals = np.zeros([self.nVertices, 2])

        # determine if point is inside or outside of each face, and distances from each face
        for j in range(0, self.nVertices):

            # calculate the unit normal vector of the current face (taking points CCW)
            if j < self.nVertices - 1:  # all but the set of point that close the shape
                normal = np.array([self.xy_boundary[j + 1, 1] - self.xy_boundary[j, 1],
                                   -(self.xy_boundary[j + 1, 0] - self.xy_boundary[j, 0])])
                unit_normals[j] = normal / np.linalg.norm(normal)
            else:   # the set of points that close the shape
                normal = np.array([self.xy_boundary[0, 1] - self.xy_boundary[j, 1],
                                   -(self.xy_boundary[0, 0] - self.xy_boundary[j, 0])])
                unit_normals[j] = normal / np.linalg.norm(normal)

        self.unit_normals = unit_normals

    def calculate_gradients(self):
        unit_normals = self.unit_normals

        # initialize array to hold distances from each point to each face
        dfaceDistance_dx = np.zeros([self.n_wt * self.nVertices, self.n_wt])
        dfaceDistance_dy = np.zeros([self.n_wt * self.nVertices, self.n_wt])

        for i in range(0, self.n_wt):
            # determine if point is inside or outside of each face, and distances from each face
            for j in range(0, self.nVertices):

                # define the derivative vectors from the point of interest to the first point of the face
                dpa_dx = np.array([-1.0, 0.0])
                dpa_dy = np.array([0.0, -1.0])

                # find perpendicular distances derivatives from point to current surface (vector projection)
                ddistanceVec_dx = np.vdot(dpa_dx, unit_normals[j]) * unit_normals[j]
                ddistanceVec_dy = np.vdot(dpa_dy, unit_normals[j]) * unit_normals[j]

                # calculate derivatives for the sign of perpendicular distances from point to current face
                dfaceDistance_dx[i * self.nVertices + j, i] = np.vdot(ddistanceVec_dx, unit_normals[j])
                dfaceDistance_dy[i * self.nVertices + j, i] = np.vdot(ddistanceVec_dy, unit_normals[j])

        # return Jacobian dict
        self.dfaceDistance_dx = dfaceDistance_dx
        self.dfaceDistance_dy = dfaceDistance_dy

    def calculate_distance_to_boundary(self, points):
        """
        :param points: points that you want to calculate the distances from to the faces of the convex hull
        :return face_distace: signed perpendicular distances from each point to each face; + is inside
        """

        nPoints = np.array(points).shape[0]
        xy_boundary = self.xy_boundary[:-1]
        nVertices = xy_boundary.shape[0]
        vertices = xy_boundary
        unit_normals = self.unit_normals
        # initialize array to hold distances from each point to each face
        face_distance = np.zeros([nPoints, nVertices])
        from numpy import newaxis as na

        # define the vector from the point of interest to the first point of the face
        PA = (vertices[:, na] - points[na])

        # find perpendicular distances from point to current surface (vector projection)
        dist = np.sum(PA * unit_normals[:, na], 2)
        # calculate the sign of perpendicular distances from point to current face (+ is inside, - is outside)
        d_vec = dist[:, :, na] * unit_normals[:, na]
        face_distance = np.sum(d_vec * unit_normals[:, na], 2)
        return face_distance.T

    def distances(self, x, y):
        return self.calculate_distance_to_boundary(np.array([x, y]).T)

    def gradients(self, x, y):
        return self.dfaceDistance_dx, self.dfaceDistance_dy

    def satisfy(self, state, pad=1.1):
        x, y = [np.asarray(state[xyz], dtype=float) for xyz in [topfarm.x_key, topfarm.y_key]]
        dist = self.distances(x, y)
        dist = np.where(dist < 0, np.minimum(dist, -.01), dist)
        dx, dy = self.gradients(x, y)  # independent of position
        dx = dx[:self.nVertices, 0]
        dy = dy[:self.nVertices, 0]
        for i in np.where(dist.min(1) < 0)[0]:  # loop over turbines that violate edges
            # find smallest movement that where the constraints are satisfied
            d = dist[i]
            v = np.linspace(-np.abs(d.min()), np.abs(d.min()), 100)
            X, Y = np.meshgrid(v, v)
            m = np.ones_like(X)
            for dx_, dy_, d in zip(dx, dy, dist.T):
                m = np.logical_and(m, X * dx_ + Y * dy_ >= -d[i])
            index = np.argmin(X[m]**2 + Y[m]**2)
            x[i] += X[m][index]
            y[i] += Y[m][index]
        state[topfarm.x_key] = x
        state[topfarm.y_key] = y
        return state


class PolygonBoundaryComp(BoundaryBaseComp):
    def __init__(self, n_wt, xy_boundary, const_id=None, units=None, relaxation=False, turbine_diameter=None):

        self.nTurbines = n_wt # Used by some methods, equivalent to n_wt
        # self.const_id = const_id # Set by super
        # self.units = units # Set by super
        # self.turbine_diameter = turbine_diameter # Set by super

        # xy_boundary is assumed to be already scaled if turbine_diameter was provided to XYBoundaryConstraint.
        # No further scaling based on turbine_diameter argument here.
        # get_boundary_properties operates on this (potentially scaled) xy_boundary.
        self.boundary_properties = self.get_boundary_properties(xy_boundary if xy_boundary is not None and xy_boundary.size > 0 else np.array([[0,0],[0,1],[1,0]])) # Pass dummy for safety if empty

        # BoundaryBaseComp.__init__ expects the main xy_boundary for plotting, etc.
        # self.boundary_properties[0] is the processed vertex list (e.g., closed loop).
        super().__init__(n_wt, xy_boundary=self.boundary_properties[0], const_id=const_id,
                                  units=units, relaxation=relaxation, turbine_diameter=turbine_diameter)

        self.zeros = np.zeros(self.nTurbines) # Distances are per-turbine for simple polygon
        self._cache_input = None
        self._cache_output = None
        # self.relaxation = relaxation # Set by super

    def get_boundary_properties(self, xy_boundary, inclusion_zone=True):
        vertices = np.array(xy_boundary)

        def get_edges(vertices, counter_clockwise):
            if np.any(vertices[0] != vertices[-1]):
                vertices = np.r_[vertices, vertices[:1]]
            x1, y1 = A = vertices[:-1].T
            x2, y2 = B = vertices[1:].T
            double_area = np.sum((x1 - x2) * (y1 + y2))  # 2 x Area (+: counterclockwise
            assert double_area != 0, "Area must be non-zero"
            if (counter_clockwise and double_area < 0) or (not counter_clockwise and double_area > 0):  #
                return get_edges(vertices[::-1], counter_clockwise)
            else:
                return vertices[:-1], A, B

        # inclusion zones are defined counter clockwise (unit-normal vector pointing in) while
        # exclusion zones are defined clockwise (unit-normal vector pointing out)
        xy_boundary, A, B = get_edges(vertices, inclusion_zone)

        dx, dy = AB = B - A
        AB_len = np.linalg.norm(AB, axis=0)
        edge_unit_normal = (np.array([-dy, dx]) / AB_len)

        # A_normal and B_normal are the normal vectors at the nodes A,B (the mean of the adjacent edge normal vectors
        A_normal = (edge_unit_normal + np.roll(edge_unit_normal, 1, 1)) / 2
        B_normal = np.roll(A_normal, -1, 1)

        # for (x, y), (dx, dy), (unx, uny) in zip(A.T, AB.T, edge_unit_normal.T):
        #     plt.arrow(x, y, dx, dy, color='k', head_width=.2)
        #     plt.arrow(x, y, unx, uny, color='r', head_width=.2)
        # for (x, y), (nx, ny) in zip(A.T, A_normal.T):
        #     plt.arrow(x, y, nx, ny, color='b', head_width=.2)
        # for (x, y), (nx, ny) in zip(B.T, B_normal.T):
        #     plt.arrow(x, y, nx / 2, ny / 2, color='g', head_width=.2)

        return (xy_boundary, A, B, AB, AB_len, edge_unit_normal, A_normal, B_normal)

    def _calc_distance_and_gradients(self, x, y, boundary_properties=None):
        """
        distances point, P=(x,y) to edge(A->B)
        +/-: inside/outside
        """
        def vec_len(vec):
            return np.linalg.norm(vec, axis=0)

        boundary_properties = boundary_properties or self.boundary_properties[1:]
        A, B, AB, AB_len, edge_unit_normal, A_normal, B_normal = boundary_properties
        """
        A: edge start point
        B: edge end point
        edge_unit_normal: unit vector perpendicular to edge pointing to the good side
        (i.e. inside for inclusion zones and outside for exclusion zones)
        AB: Vector from A to B (edge)
        AB_len: length of AB (edge)
        A_normal: mean of edge unit normal vectors adjacent to A
        B_normal: mean of edge unit normal vectors adjacent to B
        """

        # Add dim to match (2, #P, #Edges), where the first dimension is (x,y)
        P = np.array([x, y])[:, :, na]
        A, B, AB = A[:, na], B[:, na], AB[:, na]
        edge_unit_normal, A_normal, B_normal = edge_unit_normal[:, na], A_normal[:, na], B_normal[:, na]
        AB_len = AB_len[na]

        # ===============================================================================================================
        # Determine if P is closer to A, B or the edge (between A and B)
        # ===============================================================================================================
        AP = P - A  # vector from edge start to point
        BP = P - B  # vector from edge end to point

        # signed component of AP on the edge vector
        a_tilde = np.sum(AP * AB, axis=0) / AB_len

        # a_tilde < 0: closer to A
        # a_tilde > |AB|: closer to B
        # else: closer to edge (between A and B)
        use_A = 0 > a_tilde
        use_B = a_tilde > AB_len

        # ===============================================================================================================
        # Calculate distance from P to closer point on edge
        # ===============================================================================================================

        # Perpendicular distances to edge (AP dot edge_unit_normal product).
        # This is the distance to the edge if not use_A or use_B
        distance = np.sum((AP) * edge_unit_normal, 0)

        # Update distance for points closer to A
        good_side_of_A = (np.sum((AP * A_normal)[:, use_A], 0) > 0)
        sign_use_A = np.where(good_side_of_A, 1, -1)
        distance[use_A] = (vec_len(AP[:, use_A]) * sign_use_A)

        # Update distance for points closer to B
        good_side_of_B = np.sum((BP * B_normal)[:, use_B], 0) > 0
        sign_use_B = np.where(good_side_of_B, 1, -1)
        distance[use_B] = (vec_len(BP[:, use_B]) * sign_use_B)

        # ===============================================================================================================
        # Calculate gradient of distance from P to closer point on edge wrt. x and y
        # ===============================================================================================================

        # Gradient of perpendicular distances to edge.
        # This is the gradient if not use_A or use_B
        ddist_dxy = np.tile(edge_unit_normal, (1, len(x), 1))

        # Update gradient for points closer to A or B
        eps = 1e-7  # avoid division by zero
        ddist_dxy[:, use_A] = sign_use_A * (AP[:, use_A] / (vec_len(AP[:, use_A]) + eps))
        ddist_dxy[:, use_B] = sign_use_B * (BP[:, use_B] / (vec_len(BP[:, use_B]) + eps))
        ddist_dX, ddist_dY = ddist_dxy

        return distance, ddist_dX, ddist_dY

    def calc_distance_and_gradients(self, x, y):
        if not np.shape([x, y]) == np.shape(self._cache_input):
            pass
        elif np.all(np.array([x, y]) == self._cache_input):
            return self._cache_output
        distance, ddist_dX, ddist_dY = self._calc_distance_and_gradients(x, y)
        closest_edge_index = np.argmin(np.abs(distance), 1)
        self._cache_input = np.array([x, y])
        self._cache_output = [  # pick only closest edge
            v[np.arange(len(closest_edge_index)), closest_edge_index] for v in [distance, ddist_dX, ddist_dY]
        ]
        return self._cache_output

    def distances(self, x, y):
        return self.calc_distance_and_gradients(x, y)[0]

    def gradients(self, x, y):
        _, dx, dy = self.calc_distance_and_gradients(x, y)
        return np.diagflat(dx), np.diagflat(dy)

    def satisfy(self, state, pad=1.1):
        x, y = [np.asarray(state[xy], dtype=float) for xy in [topfarm.x_key, topfarm.y_key]]
        dist = self.distances(x, y)
        dx, dy = map(np.diag, self.gradients(x, y))
        m = dist < 0
        x[m] -= dx[m] * dist[m] * pad
        y[m] -= dy[m] * dist[m] * pad
        state[topfarm.x_key] = x
        state[topfarm.y_key] = y
        return state


class CircleBoundaryComp(PolygonBoundaryComp):
    def __init__(self, n_wt, center, radius, const_id=None, units=None, turbine_diameter=None):
        # center and radius are assumed to be already scaled by CircleBoundaryConstraint.
        self.center = center
        self.radius = radius
        # self.turbine_diameter will be set by PolygonBoundaryComp's __init__ via super()

        # Generate xy_boundary from scaled center and radius for PolygonBoundaryComp's base methods (e.g. plotting).
        # This is used by PolygonBoundaryComp's __init__ and potentially by BoundaryBaseComp's plot method.
        t = np.linspace(0, 2 * np.pi, 100)
        xy_boundary_for_plot = self.center + np.array([np.cos(t), np.sin(t)]).T * self.radius

        # Call PolygonBoundaryComp's __init__
        # It expects: n_wt, xy_boundary, const_id, units, relaxation, turbine_diameter
        super().__init__(n_wt,
                         xy_boundary=xy_boundary_for_plot,
                         const_id=const_id,
                         units=units,
                         relaxation=False, # CircleBoundaryComp typically doesn't use relaxation itself
                         turbine_diameter=turbine_diameter)

        # PolygonBoundaryComp's __init__ correctly sets self.zeros = np.zeros(self.nTurbines)
        # which is appropriate for CircleBoundaryComp as its distances are per turbine (n_wt).

    def plot(self, ax=None):
        ax = ax or plt.gca()
        circle = Circle(self.center, self.radius, color='k', fill=False)
        ax.add_artist(circle)

    def distances(self, x, y):
        return self.radius - np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)

    def gradients(self, x, y):
        theta = np.arctan2(y - self.center[1], x - self.center[0])
        dx = -1 * np.ones_like(x)
        dy = -1 * np.ones_like(x)
        dist = self.radius - np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        not_center = dist != self.radius
        dx[not_center], dy[not_center] = -np.cos(theta[not_center]), -np.sin(theta[not_center])
        return np.diagflat(dx), np.diagflat(dy)


class Zone(object):
    def __init__(self, boundary, dist2wt, geometry_type, incl, name):
        self.name = name
        self.boundary = boundary
        self.dist2wt = dist2wt
        self.geometry_type = geometry_type
        self.incl = incl


class InclusionZone(Zone):
    def __init__(self, boundary, dist2wt=None, geometry_type='polygon', name=''):
        super().__init__(np.asarray(boundary), dist2wt, geometry_type, incl=1, name=name)


class ExclusionZone(Zone):
    def __init__(self, boundary, dist2wt=None, geometry_type='polygon', name=''):
        super().__init__(np.asarray(boundary), dist2wt, geometry_type, incl=0, name=name)


class MultiPolygonBoundaryComp(PolygonBoundaryComp):
    def __init__(self, n_wt, zones, const_id=None, units=None, relaxation=False, method='nearest',
                 simplify_geometry=False, turbine_diameter=None):
        '''
        Parameters
        ----------
        n_wt : TYPE
            DESCRIPTION.
        zones : list
            list of InclusionZone and ExclusionZone objects. Boundaries within zones are assumed
            to be already scaled by XYBoundaryConstraint if turbine_diameter was provided.
        const_id : TYPE, optional
            DESCRIPTION. The default is None.
        units : TYPE, optional
            DESCRIPTION. The default is None.
        method : {'nearest' or 'smooth_min'}, optional
            'nearest' calculate the distance to the nearest edge or point'smooth_min'
            calculates the weighted minimum distance to all edges/points. The default is 'nearest'.
        simplify : float or dict
            if float, simplification tolerance. if dict, shapely.simplify keyword arguments
        turbine_diameter : float, optional
            Passed to super and stored in BoundaryBaseComp. Scaling of zones' boundaries
            is assumed to be handled by XYBoundaryConstraint.
        Returns
        -------
        None.

        '''
        self.zones = zones # zones' boundaries are already scaled by XYBoundaryConstraint
        # self.turbine_diameter will be set by PolygonBoundaryComp's __init__ via super()

        # get_xy_boundaries uses self.zones (which have scaled boundaries).
        # The static _zone2poly in MultiPolygonBoundaryComp is simple and does not use turbine_diameter
        # for buffer scaling (it uses hardcoded D,H which implies physical scale, but buffers are applied to
        # potentially scaled coordinates from z.boundary).
        # This specific _zone2poly is static and does not interact with turbine_diameter.
        self.bounds_poly, xy_boundaries = self.get_xy_boundaries()

        initial_xy_for_super = xy_boundaries[0] if xy_boundaries and len(xy_boundaries) > 0 and xy_boundaries[0].size > 0 else np.array([[0,0],[0,1],[1,0]])


        super().__init__(n_wt,
                         xy_boundary=initial_xy_for_super,
                         const_id=const_id,
                         units=units,
                         relaxation=relaxation,
                         turbine_diameter=turbine_diameter)

        self.incl_excls = [x.incl for x in zones]
        if self.bounds_poly and all(isinstance(p, (Polygon, MultiPolygon)) for p in self.bounds_poly) and any(p.area > 1e-3 for p in self.bounds_poly):
             self._setup_boundaries(self.bounds_poly, self.incl_excls)
        else:
             self.boundaries = []
             self.boundary_properties_list_all = [np.array([]) for _ in range(7)]
             warnings.warn("MultiPolygonBoundaryComp: No valid boundaries were set up after processing zones.")


        self.method = method
        if simplify_geometry:
            self.simplify(simplify_geometry)

    def simplify(self, simplify_geometry):
        bounds = [bi[0] for bi in self.boundaries]
        self.incl_excls = [bi[1] for bi in self.boundaries]
        polygons = [Polygon(b) for b in bounds]
        if isinstance(simplify_geometry, dict):
            self.bounds_poly = [rp.simplify(**simplify_geometry) for rp in polygons]
        else:
            self.bounds_poly = [rp.simplify(simplify_geometry) for rp in polygons]
        self._setup_boundaries(self.bounds_poly, self.incl_excls)

    # def line_to_xy_boundary(self, line, buffer):
    #     return np.asarray(Polygon(LineString(line).buffer(buffer, join_style=2).exterior).exterior.coords)

    def get_xy_boundaries(self):
        polygons = []
        bounds = []
        for z in self.zones:
            # hardcoded values passed to this function... it does not impact the
            # result because this only happends due to turbine specific component
            # calling into parent constructor;
            poly = self._zone2poly(z, D=-1, H=-1)
            polygons.append(poly)
            bounds.append(np.asarray(poly.exterior.coords))
        return polygons, bounds

    def _setup_boundaries(self, bounds_poly, incl_excl):
        self.res_poly = self._calc_resulting_polygons(bounds_poly, incl_excl)
        self.boundaries = self._poly_to_bound(self.res_poly)

        boundary_properties_list_all = list(zip(*[self.get_boundary_properties(bound, incl_excl)[1:]
                                                  for bound, incl_excl in self.boundaries]))

        self.boundary_properties_list_all = [np.concatenate(v, -1)
                                             for v in boundary_properties_list_all]

    def _poly_to_bound(self, polygons):
        boundaries = []
        for bound in polygons:
            x, y = bound.exterior.xy
            boundaries.append((np.asarray([x, y]).T[:-1, :], 1))
            for interior in bound.interiors:
                x, y = interior.xy
                boundaries.append((np.asarray([x, y]).T[:-1, :], 0))
        return boundaries

    @staticmethod
    def _calc_resulting_polygons(boundary_polygons, incl_excls):
        """
        Parameters
        ----------
        boundary_polygons : list[shapely.Polygon]
            list of shapely polygons as specifed or inferred from user input
        incl_excls : list[bool]
            list of boolean values specifying whether the polygon is an inclusion or exclusion

        Returns
        -------
            list of merged shapely polygons. Resolves issues arrising if any are overlapping, touching or contained in each other
        """
        included_polygons = [
            boundary_polygons[i] for i, x in enumerate(incl_excls) if x == 1
        ]
        excluded_polygons = [
            boundary_polygons[i] for i, x in enumerate(incl_excls) if x == 0
        ]

        included_polygons = unary_union(included_polygons)
        excluded_polygons = unary_union(excluded_polygons)
        remain_polygons = included_polygons.difference(excluded_polygons)

        if isinstance(remain_polygons, Polygon):
            return [remain_polygons] if remain_polygons.area > 1e-3 else []

        if isinstance(remain_polygons, MultiPolygon):
            return [
                poly for poly in remain_polygons.geoms if poly.area > 1e-3
            ]

        return []

    def sign(self, Dist_ij):
        return np.sign(Dist_ij[np.arange(Dist_ij.shape[0]), np.argmin(abs(Dist_ij), axis=1)])

    def calc_distance_and_gradients(self, x, y):
        '''
        Parameters
        ----------
        x : 1d array
            Array of x-positions.
        y : 1d array
            Array of y-positions.

        Returns
        -------
        D_ij : 2d array
            Array of point-edge distances. index 'i' is points and index 'j' is total number of edges.
        sign_i : 1d array
            Array of signs of the governing distance.
        dDdk_jk : 2d array
            Jacobian of the distance matrix D_ij with respect to x and y.

        '''
        if not np.shape([x, y]) == np.shape(self._cache_input):
            pass
        elif np.all(np.array([x, y]) == self._cache_input) & (not self.relaxation):
            return self._cache_output

        Dist_ij, ddist_dX, ddist_dY = self._calc_distance_and_gradients(x, y, self.boundary_properties_list_all)

        dDdk_ijk = np.moveaxis([ddist_dX, ddist_dY], 0, -1)
        sign_i = self.sign(Dist_ij)
        self._cache_input = np.array([x, y])
        self._cache_output = [Dist_ij, dDdk_ijk, sign_i]
        return self._cache_output

    def calc_relaxation(self, iteration_no=None):
        '''
        The tupple relaxation contains a first term for the penalty constant
        and a second term for the n first iterations to apply relaxation.
        '''
        if iteration_no is None:
            iteration_no = self.problem.cost_comp.n_grad_eval + 1
        return max(0, self.relaxation[0] * (self.relaxation[1] - iteration_no))

    def distances(self, x, y):
        Dist_ij, _, sign_i = self.calc_distance_and_gradients(x, y)
        if self.method == 'smooth_min':
            Dist_i = smooth_max(np.abs(Dist_ij), -np.abs(Dist_ij).max(), axis=1) * sign_i
        elif self.method == 'nearest':
            Dist_i = Dist_ij[np.arange(x.size), np.argmin(np.abs(Dist_ij), axis=1)]
        else:
            warning = f'method: {self.method} is not implemented. Available options are smooth_min and nearest.'
            warnings.warn(warning)
        if self.relaxation:
            Dist_i += self.calc_relaxation()
        return Dist_i

    def gradients(self, x, y):
        '''
        The derivate of the smooth maximum with respect to x and y is calculated with the chain rule:
            dS/dk = dS/dD * dD/dk
            where S is smooth maximum, D is distance to edge and k is the spacial dimension
        '''
        Dist_ij, dDdk_ijk, _ = self.calc_distance_and_gradients(x, y)
        if self.relaxation:
            Dist_ij += self.calc_relaxation()
            # dDdt = -self.relaxation[1]
        if self.method == 'smooth_min':
            dSdDist_ij = smooth_max_gradient(np.abs(Dist_ij), -np.abs(Dist_ij).max(), axis=1)
            dSdkx_i, dSdky_i = (dSdDist_ij[:, :, na] * dDdk_ijk).sum(axis=1).T
        elif self.method == 'nearest':
            dSdkx_i, dSdky_i = dDdk_ijk[np.arange(x.size), np.argmin(np.abs(Dist_ij), axis=1), :].T

        if self.relaxation:
            # as relaxed distance is relaxation + distance, the gradient with respect to x and y is unchanged
            gradients = np.diagflat(dSdkx_i), np.diagflat(dSdky_i), np.ones(self.n_wt) * self.relaxation[1]
        else:
            gradients = np.diagflat(dSdkx_i), np.diagflat(dSdky_i)
        return gradients

    def relaxed_polygons(self, iteration_no=None):
        poly = [Polygon(x.boundary) for x in self.zones]
        booleans = [x.incl for x in self.zones]
        relaxed_poly = []
        for i, p in enumerate(poly):
            if booleans[i] == 0:
                pb = p.buffer(-self.calc_relaxation(iteration_no), join_style=2)
                relaxed_poly.append(pb)
            else:
                pb = p.buffer(self.calc_relaxation(iteration_no), join_style=2)
                relaxed_poly.append(pb)
        merged_poly = self._calc_resulting_polygons(relaxed_poly, booleans)
        return self._poly_to_bound(merged_poly)

    @staticmethod
    def _zone2poly(z: Zone, **kwargs):
        buffer = 0
        if hasattr(z.dist2wt, "__code__"):
            buffer = z.dist2wt(
                **{k: kwargs[k] for k in z.dist2wt.__code__.co_varnames}
            )
            if not is_number(buffer):
                raise ValueError(f"dist2wt must return a float, not {type(buffer)}")
        elif is_number(z.dist2wt):
            buffer = z.dist2wt
        elif z.dist2wt is not None:
            warnings.warn(
                f"dist2wt is not a function or a float and is ignored. Zone buffer is set to 0."
            )
        buf_direction = -1 if z.incl else 1
        buffer = abs(buffer) * buf_direction

        if z.geometry_type == "line":
            poly = Polygon(LineString(z.boundary).buffer(buffer, join_style=2).exterior)
        elif z.geometry_type == "polygon":
            poly = Polygon(z.boundary).buffer(buffer, join_style=2)
        else:
            raise NotImplementedError(
                f"Geometry type '{z.geometry_type}' is not implemented"
            )
        return poly

    def plot(self, ax):
        """Plot original and buffered boundaries"""
        legend_mask = [0, 0]
        for i, (zone, buffered_poly) in enumerate(zip(self.zones, self.bounds_poly)):
            original_coords = zone.boundary
            if zone.geometry_type == "line":
                ax.plot(
                    original_coords[:, 0],
                    original_coords[:, 1],
                    "k--",
                    linewidth=1,
                    alpha=0.5,
                    label="Original line" if i == 0 else "",
                )
            else:
                original_poly = np.vstack([original_coords, original_coords[0]])
                ax.plot(
                    original_poly[:, 0],
                    original_poly[:, 1],
                    "k--",
                    linewidth=1,
                    alpha=0.5,
                    label="Original boundary" if i == 0 else "",
                )

            x, y = buffered_poly.exterior.xy
            color = "green" if zone.incl else "red"
            label = "Inclusion" if zone.incl else "Exclusion"
            if legend_mask[zone.incl]:
                label = ""
            legend_mask[zone.incl] = 1
            ax.plot(
                x,
                y,
                color=color,
                linewidth=1,
                linestyle="-",
                label=label,
            )
            ax.fill(x, y, color=color, alpha=0.1)
        ax.legend()


class TurbineSpecificBoundaryComp(MultiPolygonBoundaryComp):
    def __init__(self, n_wt, wind_turbines, zones, const_id=None,
                 units=None, relaxation=False, method='nearest', simplify_geometry=False, turbine_diameter=None):

        self.wind_turbines = wind_turbines
        self.types = wind_turbines.types()
        self.n_wt = n_wt
        # zones' base boundaries (coordinates) are assumed to be ALREADY SCALED by XYBoundaryConstraint
        # if a global turbine_diameter was provided to it.
        self.zones = zones
        self.turbine_diameter_global_scale = turbine_diameter # This is the global scaler, used by _zone2poly for buffers.

        # get_ts_boundaries calls self._zone2poly.
        # _zone2poly (now an instance method) uses self.turbine_diameter_global_scale
        # to scale physical buffer distances calculated from D, H parameters.
        self.ts_polygon_boundaries, ts_xy_boundaries = self.get_ts_boundaries()

        # Initialize MultiPolygonBoundaryComp.
        # Pass self.zones (which have scaled coordinates from XYBoundaryConstraint).
        # Pass the global turbine_diameter for consistency and storage in BoundaryBaseComp.
        super().__init__(n_wt=n_wt, zones=self.zones, const_id=const_id, units=units,
                         relaxation=relaxation, method=method, simplify_geometry=simplify_geometry,
                         turbine_diameter=turbine_diameter)

        # Post MultiPolygonBoundaryComp initialization, TurbineSpecific specifics are set up.
        # self.ts_polygon_boundaries are now correctly scaled (base coords by XYBC, buffers by our _zone2poly).
        self.ts_merged_polygon_boundaries = self.merge_boundaries()
        self.ts_merged_xy_boundaries = self.get_ts_xy_boundaries()

        if self.ts_merged_xy_boundaries and any(b_list for b_list in self.ts_merged_xy_boundaries if any(b for b in b_list)):
            self.ts_boundary_properties = self.get_ts_boundary_properties()
            self.ts_item_indices = self.get_ts_item_indices()
        else:
            self.ts_boundary_properties = []
            self.ts_item_indices = []
            warnings.warn("TurbineSpecificBoundaryComp resulted in no effective boundaries for one or more turbine types.")


    def get_ts_boundaries(self):
        polygons = []
        bounds = []
        for t in set(self.types):
            temp1 = []
            temp2 = []
            dist2wt_input = dict(
                D=self.wind_turbines.diameter(t),
                H=self.wind_turbines.hub_height(t),
            )
            for z in self.zones:
                poly = self._zone2poly(z, **dist2wt_input)
                bound = np.asarray(poly)
                temp1.append(poly)
                temp2.append(bound)
            polygons.append(temp1)
            bounds.append(temp2)
        return polygons, bounds
        # for wt
        # temp = []
        # for n, (b, t, ie) in enumerate(zip(boundaries, geometry_types, incl_excls)):
        #     if t == 'line':
        #         bound = np.asarray(Polygon(LineString(b).buffer(default_ref, join_style=2).exterior).exterior.coords)
        #         self.dependencies[n]['ref'] = default_ref
        #     elif t == 'polygon':
        #         bound = b
        #     else:
        #         raise NotImplementedError("Geometry type '%s' is not implemented" % b)
        #     temp.append((bound, ie))

    # def get_ts_polygon_boundaries(self, types):
    #     temp = []
    #     for t in set(types):
    #         d = self.wind_turbines.diameter(t)
    #         h = self.wind_turbines.hub_height(t)
    #         temp.append(self.get_ts_polygon_boundary(d, h))
    #     return temp

    def get_ts_xy_boundaries(self):
        return [self._poly_to_bound(b) for b in self.ts_merged_polygon_boundaries]

    # def get_ts_polygon_boundary(self, d=None, h=None):
    #     temp = []
    #     for bound, dep in zip(self.polygon_boundaries, self.dependencies):
    #         ref = dep['ref'] or 0
    #         if dep['type'] == 'D':
    #             ts_polygon_boundary = bound.buffer(dep['multiplier']*d-ref, join_style=2)
    #         elif dep['type'] == 'H':
    #             ts_polygon_boundary = bound.buffer(dep['multiplier']*h-ref, join_style=2)
    #         else:
    #             ts_polygon_boundary = bound
    #         temp.append(ts_polygon_boundary)
    #     return temp

    def merge_boundaries(self):
        return [self._calc_resulting_polygons(bounds, self.incl_excls) for bounds in self.ts_polygon_boundaries]

    def get_ts_boundary_properties(self,):
        return [[self.get_boundary_properties(bound) for bound, _ in bounds] for bounds in self.ts_merged_xy_boundaries]

    def get_ts_item_indices(self):
        temp = []
        for bounds in self.ts_merged_xy_boundaries:
            n_edges = np.asarray([len(bound) for bound, _ in bounds])
            n_edges_tot = np.sum(n_edges)
            start_at = np.cumsum(n_edges) - n_edges
            end_at = start_at + n_edges
            item_indices = [n_edges_tot, start_at, end_at]
            temp.append(item_indices)
        return temp

    def calc_distance_and_gradients(self, x, y, types=None):
        if self._cache_input is None:
            pass
        elif not np.shape([x, y]) == np.shape(self._cache_input[0]) or not np.shape(types) == np.shape(self._cache_input[1]):
            pass
        elif np.all(np.array([x, y]) == self._cache_input[0]) & (not self.relaxation) & np.all(np.asarray([types]) == self._cache_input[1]):
            return self._cache_output
        if types is None:
            types = np.zeros(self.n_wt)
        Dist_i = np.zeros(self.n_wt)
        sign_i = np.zeros(self.n_wt)
        dDdx_i = np.zeros(self.n_wt)
        dDdy_i = np.zeros(self.n_wt)
        for t in set(types):
            t = int(t)
            idx = (types == t)
            n_edges_tot, start_at, end_at = self.ts_item_indices[t]
            Dist_ij = np.zeros((sum(idx), n_edges_tot))
            dDdk_ijk = np.zeros((sum(idx), n_edges_tot, 2))
            for n, (bound, bound_type) in enumerate(self.ts_merged_xy_boundaries[t]):
                sa = start_at[n]
                ea = end_at[n]
                distance, ddist_dX, ddist_dY = self._calc_distance_and_gradients(x[idx], y[idx], self.ts_boundary_properties[t][n][1:])
                if bound_type == 0:
                    distance *= -1
                    ddist_dX *= -1
                    ddist_dY *= -1
                Dist_ij[:, sa:ea] = distance
                dDdk_ijk[:, sa:ea, 0] = ddist_dX
                dDdk_ijk[:, sa:ea, 1] = ddist_dY

            sign_i[idx] = self.sign(Dist_ij)
            Dist_i[idx] = Dist_ij[np.arange(sum(idx)), np.argmin(np.abs(Dist_ij), axis=1)]
            dDdx_i[idx], dDdy_i[idx] = dDdk_ijk[np.arange(sum(idx)), np.argmin(np.abs(Dist_ij), axis=1), :].T
        self._cache_input = (np.array([x, y]), np.asarray(types))
        self._cache_output = [Dist_i, dDdx_i, dDdy_i, sign_i]
        return self._cache_output

    def distances(self, x, y, type=None):
        Dist_i, _, _, _ = self.calc_distance_and_gradients(x, y, types=type)
        if self.relaxation:
            Dist_i += self.calc_relaxation()
        return Dist_i

    def gradients(self, x, y, type=None):
        Dist_i, dDdx_i, dDdy_i, _ = self.calc_distance_and_gradients(x, y, types=type)
        if self.relaxation:
            Dist_i += self.calc_relaxation()
            dDdt = -self.relaxation[0]
        if self.relaxation:
            gradients = np.diagflat(dDdx_i), np.diagflat(dDdy_i), np.ones(self.n_wt) * dDdt
        else:
            gradients = np.diagflat(dDdx_i), np.diagflat(dDdy_i)
        return gradients

    def plot(self, ax):
        linestyles = ["--", "-"]
        colors = ["b", "r", "m", "c", "g", "y", "orange", "indigo", "grey"]
        for n, t in enumerate(self.types):
            _ = ax.plot(
                *self.ts_merged_xy_boundaries[n][0][0][0, :],
                color=colors[t % len(colors)],
                linewidth=1,
                label=f"{self.wind_turbines._names[n]} boundary",
            )
            for bound, io in self.ts_merged_xy_boundaries[n]:
                ax.plot(
                    np.asarray(bound)[:, 0].tolist() + [np.asarray(bound)[0, 0]],
                    np.asarray(bound)[:, 1].tolist() + [np.asarray(bound)[0, 1]],
                    color=colors[t % len(colors)],
                    linewidth=1,
                    linestyle=linestyles[io],
                )

        for i, zone in enumerate(self.zones):
            original_coords = zone.boundary
            if zone.geometry_type == "line":
                ax.plot(
                    original_coords[:, 0],
                    original_coords[:, 1],
                    "k-.",
                    linewidth=0.8,
                    alpha=0.7,
                    label=f"Original" if (i == 0) else "",
                )
            else:
                original_poly = np.vstack([original_coords, original_coords[0]])
                ax.plot(
                    original_poly[:, 0],
                    original_poly[:, 1],
                    "k-.",
                    linewidth=0.8,
                    alpha=0.7,
                    label=f"Original" if (i == 0) else "",
                )
        ax.legend()


class MultiCircleBoundaryComp(PolygonBoundaryComp):

    def __init__(self, n_wt, geometry, wt_groups, const_id=None, units=None, turbine_diameter=None):
        self.__validate_input(geometry, wt_groups) # Validates original geometry structure
        # self.turbine_diameter will be set by super() via PolygonBoundaryComp -> BoundaryBaseComp

        self.center_original = [np.array(g[0]) for g in geometry]
        self.radius_original = [g[1] for g in geometry]

        if turbine_diameter is not None and turbine_diameter > 0:
            self.center = [c_orig / turbine_diameter for c_orig in self.center_original]
            self.radius = [r_orig / turbine_diameter for r_orig in self.radius_original]
        else:
            self.center = self.center_original
            self.radius = self.radius_original

        self.masks = [np.isin(np.arange(n_wt), g) for g in wt_groups]

        representative_xy_boundary = None
        if self.center: # Create a representative boundary for plotting by base class
            t_plot = np.linspace(0, 2 * np.pi, 30)
            representative_xy_boundary = self.center[0] + np.array([np.cos(t_plot), np.sin(t_plot)]).T * self.radius[0]

        # Call PolygonBoundaryComp's __init__
        super().__init__(n_wt,
                         xy_boundary=representative_xy_boundary,
                         const_id=const_id,
                         units=units,
                         relaxation=False,
                         turbine_diameter=turbine_diameter)
        # PolygonBoundaryComp's __init__ sets self.zeros, which is fine for MultiCircle.

    def __validate_input(self, geometry, wt_groups):
        does_len_match = len(geometry) == len(wt_groups)
        centers_and_radii_given = all(len(g) == 2 for g in geometry)

        from topfarm.utils import _np2scalar  # fmt:skip
        are_radii_scalar = True
        try:
            _ = [_np2scalar(g[1]) for g in geometry]
        except BaseException:
            are_radii_scalar = False
        are_centers_all_2x_coords = all(
            hasattr(g[0], "__iter__") and len(g[0]) == 2 for g in geometry
        )

        assert all(
            [
                does_len_match,
                centers_and_radii_given,
                are_radii_scalar,
                are_centers_all_2x_coords,
            ]
        ), (
            "Invalid input for Circle: Ensure the number of geometries matches the number of wt_groups, "
            "each geometry is a tuple of center and radius, and the center is a tuple of x and y "
            "with the radius being a scalar. For instance, geometry = [((0, 0), 100), ...] and "
            "wt_groups = [[0, 1], ...]; Here, the first geometry is a circle with center at (0, 0) "
            "and radius 100, and turbines 0 and 1 are assigned for the circle constraint 0."
        )

    def plot(self, ax=None):
        ax = ax or plt.gca()
        for center, radius in zip(self.center, self.radius):
            circle = Circle(center, radius, color="k", fill=False)
            ax.add_artist(circle)

    def distances(self, x, y):
        assert (
            x.shape == y.shape == self.masks[0].shape
        ), f"{x.shape} != {y.shape} != {self.masks[0].shape}"
        distances = np.zeros_like(x)
        for center, radius, mask in zip(self.center, self.radius, self.masks):
            distances += mask * (
                radius - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            )
        return distances

    def gradients(self, x, y):
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        for center, radius, mask in zip(self.center, self.radius, self.masks):
            theta = np.arctan2(y - center[1], x - center[0])
            dx_tmp = -1 * np.ones_like(x)
            dy_tmp = -1 * np.ones_like(x)
            dist = radius - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            not_center = dist != radius
            dx_tmp[not_center], dy_tmp[not_center] = mask[not_center] * -np.cos(
                theta[not_center]
            ), mask[not_center] * -np.sin(theta[not_center])
            dx += dx_tmp
            dy += dy_tmp
        return np.diagflat(dx), np.diagflat(dy)


@dataclass
class Boundary(object):
    _vertices: np.ndarray
    design_var_mask: np.ndarray
    normals: np.ndarray = None

    @property
    def n_turbines(self):
        return self.design_var_mask.sum()

    @property
    def n_vertices(self):
        if np.all(self.vertices[0] == self.vertices[-1]):
            return self.vertices.shape[0] - 1
        return self.vertices.shape[0]

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, v):
        self._vertices = v

    def __post_init__(self):
        self.__validate()

    def __validate(self):
        self.vertices = np.asarray(self.vertices)
        assert self.vertices.ndim == 2, "Boundary must be a 2D array"
        assert any(
            [x for x in self.vertices.shape if x == 2]
        ), "Boundary must have shape (n, 2) or (2, n)"
        self.vertices = self.vertices.reshape(-1, 2)
        assert self.vertices.shape[0] > 2, "Boundary must have at least 3 vertices"
        assert self.design_var_mask.ndim == 1, "design_var_mask must 1 dimensional"
        self.design_var_mask = np.asarray(self.design_var_mask, dtype=bool)


class MultiConvexBoundaryComp(BoundaryBaseComp):

    def __init__(
        self,
        n_wt,
        geometry, # Assumed to be unscaled (original physical coordinates) when passed here
        wt_groups,
        const_id=None,
        units=None,
        turbine_diameter=None, # Added for scaling
    ):
        # self.turbine_diameter will be set by super()

        scaled_geometry = geometry
        if turbine_diameter is not None and turbine_diameter > 0:
            scaled_geometry = []
            for g_coords in geometry: # g_coords is a numpy array of vertices
                scaled_g_coords = np.asarray(g_coords) / turbine_diameter
                scaled_geometry.append(scaled_g_coords)

        # Create Boundary objects with potentially scaled geometry
        xy_boundaries = [
            Boundary(sg, np.isin(np.arange(n_wt), mask)) # sg is scaled geometry
            for sg, mask in zip(scaled_geometry, wt_groups)
        ]

        self.xy_boundaries = self.sort_boundaries(xy_boundaries)
        # calculate_boundary_and_normals also operates on the (potentially scaled) vertices within Boundary objects
        self.xy_boundaries = self.calculate_boundary_and_normals(self.xy_boundaries)

        # Call BoundaryBaseComp's __init__
        super().__init__(n_wt, None, const_id, units, relaxation=False, turbine_diameter=turbine_diameter)

        self.turbine_vertice_prod = 0
        total_n_active_turbines = 0
        for b in self.xy_boundaries:
            if np.any(b.vertices[0] != b.vertices[-1]):
                b.vertices = np.r_[b.vertices, b.vertices[:1]]
            self.turbine_vertice_prod += b.n_turbines * b.n_vertices
            total_n_active_turbines += b.n_turbines
        assert (
            total_n_active_turbines == n_wt
        ), "Number of active turbines in boundaries must match number of total turbines; Check that masks sum up to n_wt i.e. np.concatenate(all_masks).sum() == n_wt."
        self.calculate_gradients()
        self.zeros = np.zeros(self.turbine_vertice_prod)

    def sort_boundaries(self, boundaries):
        def centroid(boundary):  # fmt: skip
            return tuple(np.mean(boundary.vertices, axis=0))
        return sorted(boundaries, key=lambda b: centroid(b))

    def calculate_boundary_and_normals(
        self, xy_boundaries: list[Boundary]
    ) -> list[Boundary]:
        def __compute_normal(boundary_pts, ii, jj):
            """Calculate the unit normal vector of the current face (taking points CCW)"""
            normal = np.array(
                [
                    boundary_pts[ii, 1] - boundary_pts[jj, 1],
                    -(boundary_pts[ii, 0] - boundary_pts[jj, 0]),
                ]
            )
            return normal / np.linalg.norm(normal)

        for boundary in xy_boundaries:
            hull = ConvexHull(list(boundary.vertices))
            # keep only vertices that actually comprise a convex hull and arrange in CCW order
            boundary.vertices = boundary.vertices[hull.vertices]
            # initialize normals array
            unit_normals = np.zeros([boundary.n_vertices, 2])
            # determine if point is inside or outside and distances from each face
            nvtm1 = boundary.n_vertices - 1
            for j in range(0, nvtm1):
                # all but the points that close the shape
                unit_normals[j] = __compute_normal(boundary.vertices, j + 1, j)
            # points that close the shape
            unit_normals[nvtm1] = __compute_normal(boundary.vertices, 0, nvtm1)
            boundary.normals = unit_normals

        return xy_boundaries

    def calculate_gradients(self):
        # this is flawed if the order of boundaries is switched;
        # test with arbitrary number of vertices and arbitrary number
        # of turbines in a boundary; For now it's sorted at the top..
        final_dx = np.zeros([self.turbine_vertice_prod, self.n_wt])
        final_dy = np.zeros([self.turbine_vertice_prod, self.n_wt])
        for bi, boundary in enumerate(self.xy_boundaries):
            assert (
                boundary.design_var_mask is not None
            ), "Design variable mask must be provided"
            assert self.n_wt == len(
                boundary.design_var_mask
            ), "Design variable mask must must be the same length as the number of turbines"

            unit_normals = boundary.normals
            n_vertices = boundary.n_vertices
            n_turbines = boundary.n_turbines
            dfaceDistance_dx = np.zeros([n_turbines * n_vertices, n_turbines])
            dfaceDistance_dy = np.zeros([n_turbines * n_vertices, n_turbines])
            for i in range(0, n_turbines):
                # determine if point is inside or outside of each face, and distances from each face
                for j in range(0, n_vertices):
                    # define the derivative vectors from the point of interest to the first point of the face
                    dpa_dx = np.array([-1.0, 0.0])
                    dpa_dy = np.array([0.0, -1.0])
                    # find perpendicular distances derivatives from point to current surface (vector projection)
                    ddistanceVec_dx = np.vdot(dpa_dx, unit_normals[j]) * unit_normals[j]
                    ddistanceVec_dy = np.vdot(dpa_dy, unit_normals[j]) * unit_normals[j]
                    # calculate derivatives for the sign of perpendicular distances from point to current face
                    dfaceDistance_dx[i * n_vertices + j, i] = np.vdot(
                        ddistanceVec_dx, unit_normals[j]
                    )
                    dfaceDistance_dy[i * n_vertices + j, i] = np.vdot(
                        ddistanceVec_dy, unit_normals[j]
                    )
            seek = sum([(b.n_vertices * b.n_turbines) for b in self.xy_boundaries[:bi]])
            final_dx[
                seek: seek + (n_turbines * n_vertices), boundary.design_var_mask
            ] = dfaceDistance_dx
            final_dy[
                seek: seek + (n_turbines * n_vertices), boundary.design_var_mask
            ] = dfaceDistance_dy
        dfaceDistance_dx = final_dx
        dfaceDistance_dy = final_dy

        def __wrap_sparse(m):  # fmt: skip
            if m.size < 1e4:
                return m
            from scipy.sparse import csr_matrix  # fmt: skip
            return csr_matrix(m)
        # store Jacobians
        self.dfaceDistance_dx = __wrap_sparse(dfaceDistance_dx)
        self.dfaceDistance_dy = __wrap_sparse(dfaceDistance_dy)

    def distances(self, x, y):
        """
        :param points: points that you want to calculate the distances from to the faces of the convex hull
        :return face_distace: signed perpendicular distances from each point to each face; + is inside
        """
        points = np.array([x, y]).T
        face_distances = np.zeros(self.turbine_vertice_prod)
        for bi, boundary in enumerate(self.xy_boundaries):
            mask = boundary.design_var_mask
            vertices = boundary.vertices[:-1]
            n_vertices = boundary.n_vertices
            PA = vertices[:, na] - points[mask][na]
            dist = np.sum(PA * boundary.normals[:, na], axis=2)
            d_vec = dist[:, :, na] * boundary.normals[:, na]
            seek = sum([(b.n_vertices * b.n_turbines) for b in self.xy_boundaries[:bi]])
            face_distances[seek: seek + (boundary.n_turbines * n_vertices)] = np.sum(
                d_vec * boundary.normals[:, na], axis=2
            ).T.reshape(-1)
        return face_distances

    def gradients(self, x, y):
        return self.dfaceDistance_dx, self.dfaceDistance_dy

    def satisfy(self, state):
        raise NotImplementedError("Not implemented for MultiConvexBoundaryComp")

    def plot(self, ax):
        for b in self.xy_boundaries:
            ax.plot(
                b.vertices[:, 0].tolist(),
                b.vertices[:, 1].tolist(),
                "k",
                linewidth=1,
            )


class MultiWFPolygonBoundaryComp(PolygonBoundaryComp):

    def __init__(
        self,
        n_wt: int,
        geometry, # Assumed to be unscaled (original physical coordinates)
        wt_groups,
        # **kwargs may include const_id, units, relaxation, and now turbine_diameter
        **kwargs,
    ):
        turbine_diameter_from_kwargs = kwargs.get('turbine_diameter')
        # self.turbine_diameter will be set by super() via PolygonBoundaryComp -> BoundaryBaseComp

        raw_boundaries_map = {i: geom for i, geom in enumerate(geometry)}
        turbine_groups_map = {i: group for i, group in enumerate(wt_groups)}

        self.boundaries = {}  # group_id: potentially scaled boundary_coords
        for group_id, original_boundary_coords in raw_boundaries_map.items():
            # Validate original coordinates structure first
            processed_coords = self.__validate_boundary_coords(original_boundary_coords)

            if turbine_diameter_from_kwargs is not None and turbine_diameter_from_kwargs > 0:
                processed_coords = processed_coords / turbine_diameter_from_kwargs

            # Close the boundary if needed
            if not np.all(processed_coords[0] == processed_coords[-1]):
                processed_coords = np.vstack([processed_coords, processed_coords[0]])
            self.boundaries[group_id] = processed_coords

        self.__validate_group_assignments(turbine_groups_map, n_wt)
        self.turbine_groups = {i: -1 for i in range(n_wt)}
        for group_id, indices in turbine_groups_map.items():
            if group_id not in self.boundaries: # Should not happen if geometry and wt_groups have same keys
                raise ValueError(f"No boundary defined for group {group_id}")
            for idx in indices:
                self.turbine_groups[idx] = group_id
        if -1 in self.turbine_groups.values(): # Ensure all turbines covered
            unassigned_turbines = [idx for idx, grp_id in self.turbine_groups.items() if grp_id == -1]
            raise ValueError(f"All turbines must be assigned to a group. Unassigned turbines: {unassigned_turbines}")

        # For PolygonBoundaryComp's __init__, provide one of the (scaled) boundaries.
        initial_boundary_for_super = np.array(list(self.boundaries.values())[0]).reshape(-1, 2) if self.boundaries else np.array([[0,0],[0,1],[1,0]]) # Dummy if no boundaries

        # Pass all relevant kwargs (const_id, units, relaxation, turbine_diameter) to PolygonBoundaryComp's init
        super().__init__(
            n_wt, initial_boundary_for_super,
            **kwargs # This will pass const_id, units, relaxation, turbine_diameter
        )

    def __validate_boundary_coords(self, boundary_coords: np.ndarray) -> None:
        if not isinstance(boundary_coords, (np.ndarray, list)):
            raise TypeError(
                "Boundary coordinates must be a numpy array or a list of lists"
            )
        try:
            boundary_coords = np.array(boundary_coords).reshape(-1, 2)
        except BaseException:
            raise ValueError(
                "Boundary coordinates must be a 2D array with shape (n,2)"
            )
        if boundary_coords.ndim != 2 or boundary_coords.shape[1] != 2:
            raise ValueError("Boundary coordinates must be a 2D array with shape (n,2)")
        if len(boundary_coords) < 3:
            raise ValueError("Boundary must have at least 3 points")
        return boundary_coords

    def __validate_group_assignments(
        self, groups: Dict[int, List[int]], num_turbines: int
    ) -> None:
        if not isinstance(groups, dict):
            raise TypeError("Groups must be a dictionary")
        valid_types = (int, np.integer)
        for group_id, turbine_indices in groups.items():
            if not isinstance(group_id, valid_types) or group_id < 0:
                raise ValueError(
                    f"Invalid group ID: {group_id}; Should be an integer >= 0"
                )
            if not all(
                isinstance(idx, valid_types) and 0 <= idx < num_turbines
                for idx in turbine_indices
            ):
                raise ValueError(
                    f"Invalid turbine indices in group {group_id}; Should be integers in range [0, {num_turbines})"
                )

    def __dist_grad_wrapper(self, x, y, boundary_prop):
        if not np.shape([x, y]) == np.shape(self._cache_input):
            pass
        elif np.all(np.array([x, y]) == self._cache_input):
            return self._cache_output
        distance, ddist_dX, ddist_dY = self._calc_distance_and_gradients(
            x, y, boundary_prop
        )
        closest_edge_index = np.argmin(np.abs(distance), 1)
        self._cache_input = np.array([x, y])
        self._cache_output = [  # pick only closest edge
            v[np.arange(len(closest_edge_index)), closest_edge_index] for v in [distance, ddist_dX, ddist_dY]
        ]
        return self._cache_output

    def __calculate_group_distances_and_gradients(self, x, y):
        """Helper method to calculate distances and gradients for all groups."""
        n = len(x)
        ds = np.zeros(n)
        dx = np.zeros(n)
        dy = np.zeros(n)

        for group_id, boundary in self.boundaries.items():
            group_turbines = [
                i for i in range(n) if self.turbine_groups.get(i) == group_id
            ]
            if not group_turbines:
                continue

            x_group = x[group_turbines]
            y_group = y[group_turbines]

            boundary_properties = self.get_boundary_properties(boundary)[1:]
            distances, dx_group, dy_group = self.__dist_grad_wrapper(
                x_group, y_group, boundary_properties
            )

            ds[group_turbines] = distances
            dx[group_turbines] = dx_group
            dy[group_turbines] = dy_group

        return ds, dx, dy

    def distances(self, x, y):
        ds, _, _ = self.__calculate_group_distances_and_gradients(x, y)
        return ds

    def gradients(self, x, y):
        _, dx, dy = self.__calculate_group_distances_and_gradients(x, y)
        return np.diagflat(dx), np.diagflat(dy)

    def plot(self, ax=None):
        colors = plt.cm.viridis(np.linspace(0, .8, len(self.boundaries)))
        for ii, (group_id, boundary) in enumerate(self.boundaries.items()):
            ax.plot(*boundary.T, c=colors[ii], label=f"Group {group_id}", linewidth=1)
        ax.legend()


class BoundaryType(Enum):
    CIRCLE = "circle"
    CONVEX_HULL = "convex_hull"
    POLYGON = "polygon"


class MultiWFBoundaryConstraint(XYBoundaryConstraint):

    def __init__(self, geometry, wt_groups, boundtype, units=None, turbine_diameter=None):
        """Entry point for creating boundary constraints for joint multi-wind-farm optimization.

        Parameters
        ----------
        geometry : Iterable
            Geometry input (e.g., list of coords, list of (center, radius)) for the boundaries.
            These are assumed to be in original, unscaled units.
        wt_groups : Iterable
            List of lists defining turbine indices for each geometry.
        boundtype : BoundaryType
            Specifies the type of boundary constraint.
        units : str, optional
            Units for the boundary. Default is None.
        turbine_diameter : float, optional
            Turbine diameter used for scaling. If provided, geometry processed by
            components will be in terms of diameters. Default is None.
        """
        # Note: We don't call super().__init__ of XYBoundaryConstraint because its logic
        # for handling 'boundary' and 'boundary_type' is different from MultiWF.
        # We set necessary attributes directly or rely on component initializers.
        self.geometry = geometry # Store original, unscaled geometry
        self.wt_groups = wt_groups
        self.boundtype = boundtype
        self.units = units
        self.turbine_diameter = turbine_diameter # Store for passing to components and for internal scaling logic

        # Create a somewhat more deterministic const_id
        geom_repr_list = []
        if self.geometry:
            first_geom_item = self.geometry[0]
            if isinstance(first_geom_item, (list, np.ndarray)) and len(first_geom_item) > 0: # Convex or Polygon
                 geom_repr_list.append(str(np.asarray(first_geom_item)[0].tolist()))
            elif isinstance(first_geom_item, tuple) and len(first_geom_item) == 2: # Circle (center, radius)
                 geom_repr_list.append(str(np.asarray(first_geom_item[0]).tolist()) + f"_r{first_geom_item[1]}")
        geom_repr = "".join(geom_repr_list).replace('[','').replace(']','').replace(',','_').replace(' ','')

        self.const_id = f"wf_boundary_comp_{self.boundtype.value}_{geom_repr}_{id(self)}"[0:100] # Limit length for safety
        self.const_id = "".join(c if c.isalnum() or c in ['_'] else '_' for c in self.const_id) # Sanitize


        # Attributes typically set by XYBoundaryConstraint's __init__ that might be needed by its methods if called by us.
        # For MultiWF, these are less critical as we override key methods like get_comp, set_design_var_limits, _setup.
        self.relaxation = False # Default, not typically used at this top level for MultiWF
        self.boundary_type = self.boundtype.value # For consistency if any XYBC methods were to be used

        assert len(geometry) == len(
            wt_groups
        ), "Number of geometries and groups must match"

    BD2COMP = {
        BoundaryType.CIRCLE: MultiCircleBoundaryComp,
        BoundaryType.CONVEX_HULL: MultiConvexBoundaryComp,
        BoundaryType.POLYGON: MultiWFPolygonBoundaryComp,
    }

    def get_comp(self, n_wt):
        assert n_wt > 0, "Number of turbines must be greater than 0"

        max_idx_in_groups = -1
        if self.wt_groups and any(self.wt_groups):
            all_indices = [idx for group in self.wt_groups for idx in group]
            if all_indices:
                max_idx_in_groups = max(all_indices)

        assert n_wt > max_idx_in_groups, \
            f"n_wt ({n_wt}) must be greater than the max turbine index ({max_idx_in_groups}) in wt_groups."


        if hasattr(self, "boundary_comp_instance"): # Use a distinct name
            return self.boundary_comp_instance

        CompClass = self.BD2COMP.get(self.boundtype)
        if CompClass is None:
            raise NotImplementedError(f"Invalid boundary type {self.boundtype}")

        # Pass turbine_diameter to the component's constructor.
        # The component is responsible for its own internal scaling of geometry.
        self.boundary_comp_instance = CompClass(
            n_wt=n_wt,
            geometry=self.geometry, # Pass original, unscaled geometry
            wt_groups=self.wt_groups,
            const_id=self.const_id,
            units=self.units,
            turbine_diameter=self.turbine_diameter # Pass the scaler here
        )
        return self.boundary_comp_instance

    def set_design_var_limits(self, design_vars):
        # This method sets limits on design variables (turbine positions)
        # based on the overall extent of all defined boundaries.
        # If turbine_diameter is used, turbine positions are in the scaled domain,
        # so boundary extents must also be scaled before setting limits.

        if not self.geometry: # No geometry to set limits from
            return

        min_bounds_list = []
        max_bounds_list = []

        current_geometry = self.geometry # Original, unscaled

        if self.boundtype == BoundaryType.CIRCLE:
            for center_orig, radius_orig in current_geometry:
                center = np.asarray(center_orig)
                radius = radius_orig
                if self.turbine_diameter is not None and self.turbine_diameter > 0:
                    center = center / self.turbine_diameter
                    radius = radius / self.turbine_diameter
                min_bounds_list.append(center - radius)
                max_bounds_list.append(center + radius)

        elif self.boundtype in [BoundaryType.CONVEX_HULL, BoundaryType.POLYGON]:
            for bound_coords_orig in current_geometry:
                vertices = np.asarray(bound_coords_orig)
                if self.turbine_diameter is not None and self.turbine_diameter > 0:
                    vertices = vertices / self.turbine_diameter
                if vertices.size > 0:
                    min_bounds_list.append(vertices.min(axis=0))
                    max_bounds_list.append(vertices.max(axis=0))
        else:
            # Fallback or error for unsupported type for design var limits
            warnings.warn(f"set_design_var_limits not fully implemented for MultiWFBoundaryConstraint type '{self.boundtype}'.")
            return

        if not min_bounds_list or not max_bounds_list:
            return # No valid bounds derived

        # Aggregate overall min/max from all geometries
        overall_min_bounds = np.vstack(min_bounds_list).min(axis=0)
        overall_max_bounds = np.vstack(max_bounds_list).max(axis=0)

        for k, l, u in zip([topfarm.x_key, topfarm.y_key], overall_min_bounds, overall_max_bounds):
            if k in design_vars:
                current_val = design_vars[k][0]
                current_lower = design_vars[k][1] if len(design_vars[k]) > 1 else -np.inf
                current_upper = design_vars[k][2] if len(design_vars[k]) > 2 else np.inf

                new_lower = np.maximum(current_lower if current_lower is not None else -np.inf, l)
                new_upper = np.minimum(current_upper if current_upper is not None else np.inf, u)

                if len(design_vars[k]) == 4: # (val, lower, upper, scaler)
                    design_vars[k] = (current_val, new_lower, new_upper, design_vars[k][3])
                elif len(design_vars[k]) == 3: # (val, lower, upper)
                     design_vars[k] = (current_val, new_lower, new_upper)
                else:
                    warnings.warn(f"Design var '{k}' in set_design_var_limits has an unexpected format. Limits applied assuming [val, lower, upper, ...]")
                    try:
                        design_vars_list = list(design_vars[k])
                        design_vars_list[1] = new_lower
                        design_vars_list[2] = new_upper
                        design_vars[k] = tuple(design_vars_list)
                    except Exception as e:
                        warnings.warn(f"Could not update design_vars for {k}: {e}")

    # Override _setup from XYBoundaryConstraint as MultiWF needs different handling
    def _setup(self, problem, group='constraint_group'):
        n_wt = problem.n_wt
        # self.boundary_comp_instance should be created by get_comp
        if not hasattr(self, 'boundary_comp_instance') or self.boundary_comp_instance is None:
            self.boundary_comp_instance = self.get_comp(n_wt) # Calls the overridden get_comp

        self.boundary_comp_instance.problem = problem
        self.set_design_var_limits(problem.design_vars) # Uses the overridden set_design_var_limits

        representative_boundary_for_problem_output = None
        # Try to get a representative boundary from the component (which should be scaled if TD is used)
        if hasattr(self.boundary_comp_instance, 'xy_boundary') and \
           self.boundary_comp_instance.xy_boundary is not None and \
           self.boundary_comp_instance.xy_boundary.size > 0:
            representative_boundary_for_problem_output = self.boundary_comp_instance.xy_boundary
        elif self.boundtype == BoundaryType.CIRCLE and self.geometry:
            center1_orig, radius1_orig = self.geometry[0]
            center1_scaled = np.array(center1_orig)
            radius1_scaled = radius1_orig
            if self.turbine_diameter is not None and self.turbine_diameter > 0:
                center1_scaled = center1_scaled / self.turbine_diameter
                radius1_scaled = radius1_scaled / self.turbine_diameter
            t_plot = np.linspace(0, 2 * np.pi, 30)
            representative_boundary_for_problem_output = center1_scaled + np.array([np.cos(t_plot), np.sin(t_plot)]).T * radius1_scaled
        elif hasattr(self.boundary_comp_instance, 'xy_boundaries') and self.boundary_comp_instance.xy_boundaries: # MultiConvex
             rep_bound_obj = self.boundary_comp_instance.xy_boundaries[0]
             if hasattr(rep_bound_obj, 'vertices') and rep_bound_obj.vertices.size > 0:
                representative_boundary_for_problem_output = rep_bound_obj.vertices # Already scaled
        elif hasattr(self.boundary_comp_instance, 'boundaries') and self.boundary_comp_instance.boundaries: # MultiWFPolygon
             first_boundary_key = list(self.boundary_comp_instance.boundaries.keys())[0]
             rep_bound_coords = self.boundary_comp_instance.boundaries[first_boundary_key]
             if rep_bound_coords.size > 0:
                representative_boundary_for_problem_output = rep_bound_coords # Already scaled

        if representative_boundary_for_problem_output is not None and np.asarray(representative_boundary_for_problem_output).size >0:
            problem.indeps.add_output('xy_boundary', representative_boundary_for_problem_output)
        else:
            warnings.warn(f"Could not determine/use a single representative xy_boundary for problem setup in MultiWFBoundaryConstraint (type: {self.boundtype.value}).")

        comp_subsystem_name = self.const_id # Use the generated const_id for subsystem name
        getattr(problem.model, group).add_subsystem(comp_subsystem_name, self.boundary_comp_instance, promotes=['*'])

    def setup_as_constraint(self, problem, group='constraint_group'):
        self._setup(problem, group=group)

        lower_bound = 0
        if self.boundtype == BoundaryType.CONVEX_HULL:
            if hasattr(self.boundary_comp_instance, 'zeros'):
                 lower_bound = self.boundary_comp_instance.zeros
            else:
                 warnings.warn("MultiConvexBoundaryComp instance expected to have a 'zeros' attribute for constraint lower bound.")
        elif problem.n_wt == 1 and self.boundtype != BoundaryType.CONVEX_HULL :
             lower_bound = 0

        problem.model.add_constraint('boundaryDistances', lower=lower_bound)

    def setup_as_penalty(self, problem, group='constraint_group'):
        self._setup(problem, group=group)


def main():
    if __name__ == "__main__":
        from py_wake.wind_turbines import WindTurbines
        from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt

        plt.close('all')
        i1 = np.array([[2, 17], [6, 23], [16, 23], [26, 15], [19, 0], [14, 4], [4, 4]])
        e1 = np.array([[0, 10], [20, 21], [22, 12], [10, 12], [9, 6], [2, 7]])
        i2 = np.array([[12, 13], [14, 17], [18, 15], [17, 10], [15, 11]])
        e2 = np.array([[5, 17], [5, 18], [8, 19], [8, 18]])
        i3 = np.array([[5, 0], [5, 1], [10, 3], [10, 0]])
        e3 = np.array([[6, -1], [6, 18], [7, 18], [7, -1]])
        e4 = np.array([[15, 9], [15, 11], [20, 11], [20, 9]])
        e5 = np.array([[10, 25], [20, 0]])
        zones = [
            InclusionZone(i1, name='i1'),
            InclusionZone(i2, name='i2'),
            InclusionZone(i3, name='i3'),
            ExclusionZone(e1, name='e1'),
            ExclusionZone(e2, name='e2'),
            ExclusionZone(e3, name='e3'),
            ExclusionZone(e4, name='e4'),
            ExclusionZone(e5, name='e5', dist2wt=lambda: 1, geometry_type='line'),
        ]

        N_points = 50
        xs = np.linspace(-1, 30, N_points)
        ys = np.linspace(-1, 30, N_points)
        y_grid, x_grid = np.meshgrid(xs, ys)
        x = x_grid.ravel()
        y = y_grid.ravel()
        n_wt = len(x)
        MPBC = MultiPolygonBoundaryComp(n_wt, zones, method='nearest')
        distances = MPBC.distances(x, y)
        delta = 1e-9
        distances2 = MPBC.distances(x + delta, y)
        dx_fd = (distances2 - distances) / delta
        dx = np.diag(MPBC.gradients(x + delta / 2, y)[0])

        plt.figure()
        plt.plot(dx_fd, dx, '.')

        plt.figure()
        for n, bound in enumerate(MPBC.boundaries):
            x_bound, y_bound = bound[0].T
            x_bound = np.append(x_bound, x_bound[0])
            y_bound = np.append(y_bound, y_bound[0])
            line, = plt.plot(x_bound, y_bound, label=f'{n}')
            plt.plot(x_bound[0], y_bound[0], color=line.get_color(), marker='o')

        plt.legend()
        plt.grid()
        plt.axis('square')
        plt.contourf(x_grid, y_grid, distances.reshape(N_points, N_points), np.linspace(-10, 10, 100), cmap='seismic')
        plt.colorbar()

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(
            x.reshape(
                N_points, N_points), y.reshape(
                N_points, N_points), distances.reshape(
                N_points, N_points), np.linspace(-10, 10, 100), cmap='seismic')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if 0:
            for smpl in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                MPBC = MultiPolygonBoundaryComp(n_wt, zones, simplify_geometry=smpl)
                plt.figure()
                ax = plt.gca()
                MPBC.plot(ax)

        wind_turbines = WindTurbines(names=['tb1', 'tb2'],
                                     diameters=[80, 120],
                                     hub_heights=[70, 110],
                                     powerCtFunctions=[
            CubePowerSimpleCt(ws_cutin=3, ws_cutout=25, ws_rated=12,
                              power_rated=2000, power_unit='kW',
                              ct=8 / 9, additional_models=[]),
            CubePowerSimpleCt(ws_cutin=3, ws_cutout=25, ws_rated=12,
                              power_rated=3000, power_unit='kW',
                              ct=8 / 9, additional_models=[])])

        x1 = [0, 3000, 3000, 0]
        y1 = [0, 0, 3000, 3000]
        b1 = np.transpose((x1, y1))

        # Buildings
        x2 = [600, 1400, 1400, 600]
        y2 = [1700, 1700, 2500, 2500]
        b2 = np.transpose((x2, y2))

        # River
        x3 = np.linspace(520, 2420, 16)
        y3 = [0, 133, 266, 400, 500, 600, 700, 733, 866, 1300, 1633,
              2100, 2400, 2533, 2700, 3000]
        b3 = np.transpose((x3, y3))

        # Roads
        x4 = np.linspace(0, 3000, 16)
        y4 = [1095, 1038, 1110, 1006, 1028, 992, 977, 1052, 1076, 1064, 1073,
              1027, 964, 981, 1015, 1058]
        b4 = np.transpose((x4, y4))

        zones = [
            InclusionZone(b1, name='i1'),
            ExclusionZone(b2, dist2wt=lambda H: 4 * H - 360, name='building'),
            ExclusionZone(b3, geometry_type='line', dist2wt=lambda D: 3 * D, name='river'),
            ExclusionZone(b4, geometry_type='line', dist2wt=lambda D, H: max(D * 2, H * 3), name='road'),
        ]
        N_points = 50
        xs = np.linspace(0, 3000, N_points)
        ys = np.linspace(0, 3000, N_points)
        y_grid, x_grid = np.meshgrid(xs, ys)
        x = x_grid.ravel()
        y = y_grid.ravel()
        n_wt = len(x)
        types = np.zeros(n_wt)
        TSBC = TurbineSpecificBoundaryComp(n_wt, wind_turbines, zones)
        distances = TSBC.distances(x, y, type=types)
        delta = 1e-9
        distances2 = TSBC.distances(x + delta, y, type=types)
        dx_fd = (distances2 - distances) / delta
        dx = np.diag(TSBC.gradients(x + delta / 2, y, type=types)[0])

        plt.figure()
        plt.plot(dx_fd, dx, '.')

        plt.figure()
        for ll, t in enumerate(TSBC.types):
            line, = plt.plot(*TSBC.ts_merged_xy_boundaries[ll][0][0][0, :], label=f'type {ll}')
            for n, bound in enumerate(TSBC.ts_merged_xy_boundaries[ll]):
                x_bound, y_bound = bound[0].T
                x_bound = np.append(x_bound, x_bound[0])
                y_bound = np.append(y_bound, y_bound[0])
                plt.plot(x_bound, y_bound, color=line.get_color())

        plt.legend()
        plt.grid()
        plt.axis('square')

        for ll, t in enumerate(TSBC.types):
            plt.figure()
            for n, bound in enumerate(TSBC.ts_merged_xy_boundaries[ll]):
                x_bound, y_bound = bound[0].T
                x_bound = np.append(x_bound, x_bound[0])
                y_bound = np.append(y_bound, y_bound[0])
                plt.plot(x_bound, y_bound, 'b')
            plt.grid()
            plt.title(f'type {ll}')
            plt.axis('square')
            plt.contourf(x_grid, y_grid, TSBC.distances(x, y, type=t * np.ones(n_wt)).reshape(N_points, N_points), 50)
            plt.colorbar()


main()
