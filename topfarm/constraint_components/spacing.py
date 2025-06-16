import numpy as np
from numpy import newaxis as na
from topfarm.constraint_components import Constraint, ConstraintComponent
import topfarm


class SpacingConstraint(Constraint):
    def __init__(self, min_spacing, units=None, aggregation_function=None, full_aggregation=False, name='spacing_comp', turbine_diameter=None):
        """Initialize SpacingConstraint

        Parameters
        ----------
        min_spacing : int or float
            Minimum spacing between turbines [m or m/D if turbine_diameter is provided]
        units : str, optional
            Units for spacing. Defaults to None.
        aggregation_function : topfarm.utils.AggregationFunction or None
            if None: compute returns all wt-wt spacings (n_wt *(n_wt-1))/2
            if AggregationFunction: compute returns an aggregated (minimum) spacing
        name : str, optional
            Name of the component. Default is 'spacing_comp'.
        turbine_diameter : float, optional
            Turbine diameter [m]. If provided, min_spacing is assumed to be in diameters
            and will be scaled to meters, or if min_spacing is in meters, it will be
            converted to diameters for internal calculations. The constraint will be on
            (min_spacing / turbine_diameter). Default is None.
        """
        self.turbine_diameter = turbine_diameter
        self.min_spacing_original = min_spacing # Store original before potential scaling

        if self.turbine_diameter is not None and self.turbine_diameter > 0:
            # Assuming min_spacing is a physical distance (e.g. meters) that needs to be scaled to diameters
            # Or, if problem is posed in diameters, min_spacing is already in D.
            # The problem states: "scale the min_spacing by dividing it by turbine_diameter"
            # This implies min_spacing input is physical, and we convert it to a scaled value (in diameters)
            self.min_spacing = min_spacing / self.turbine_diameter
            # If units were 'm', they are now effectively 'D' (diameters) for self.min_spacing
            # We might need to adjust self.units or how units are handled in components if they expect physical units.
            # For now, assume downstream components will work with this scaled min_spacing.
        else:
            self.min_spacing = min_spacing # Use as is if no scaling

        self.aggregation_function = aggregation_function
        self.full_aggregation = full_aggregation
        self.const_id = name
        self.units = units # This 'units' param might refer to turbine coordinates, not min_spacing itself.

    @property
    def constraintComponent(self):
        return self.spacing_comp

    def _setup(self, problem, **kwargs): # Added **kwargs
        self.n_wt = problem.n_wt
        td_from_problem = kwargs.get('turbine_diameter')

        # self.min_spacing is already scaled if turbine_diameter was provided to __init__.
        # td_from_problem is passed to SpacingComp for its own reference / potential use.
        # If td_from_problem is different from self.turbine_diameter (from __init__),
        # there might be an inconsistency unless SpacingComp re-scales min_spacing.
        # Current SpacingComp.__init__ takes min_spacing_scaled, so self.min_spacing should be correct.
        # The turbine_diameter passed to SpacingComp is mostly for its own record-keeping or specific features.

        # Use td_from_problem if available, otherwise fallback to self.turbine_diameter (from __init__)
        # for the component's reference diameter.
        effective_td_for_comp = td_from_problem if td_from_problem is not None else self.turbine_diameter

        self.spacing_comp = SpacingComp(self.n_wt, self.min_spacing, self.const_id, self.units,
                                        aggregation_function=self.aggregation_function,
                                        full_aggregation=self.full_aggregation,
                                        turbine_diameter=effective_td_for_comp)
        problem.model.constraint_group.add_subsystem(self.const_id, self.spacing_comp,
                                                     promotes=[topfarm.x_key, topfarm.y_key, 'wtSeparationSquared'])

    def setup_as_constraint(self, problem, **kwargs): # Added **kwargs
        self._setup(problem, **kwargs) # Pass **kwargs
        # The lower bound uses self.min_spacing which is scaled in __init__
        problem.model.add_constraint('wtSeparationSquared', lower=self.min_spacing**2)


    def setup_as_penalty(self, problem, **kwargs): # Added **kwargs
        self._setup(problem, **kwargs) # Pass **kwargs


class SpacingComp(ConstraintComponent):
    """
    Calculates inter-turbine spacing for all turbine pairs.

    """

    def __init__(self, n_wt, min_spacing_scaled, const_id=None, units=None, aggregation_function=None, full_aggregation=False, turbine_diameter=None):
        super().__init__()
        self.n_wt = n_wt
        # min_spacing_scaled is the value that has potentially been divided by turbine_diameter
        self.min_spacing_scaled = min_spacing_scaled
        self.turbine_diameter = turbine_diameter # Store for reference, though not directly used for scaling here
        self.const_id = const_id
        if aggregation_function:
            if full_aggregation:
                self.veclen = 1
            else:
                self.veclen = n_wt
        else:
            self.veclen = int((n_wt - 1.) * n_wt / 2.)
        self.units = units
        self.aggregation_function = aggregation_function
        self.full_aggregation = full_aggregation
        self.constraint_key = 'wtSeparationSquared'

    def setup(self):
        # Explicitly size input arrays
        self.add_input(topfarm.x_key, val=np.zeros(self.n_wt),
                       desc='x coordinates of turbines in wind dir. ref. frame', units=self.units)
        self.add_input(topfarm.y_key, val=np.zeros(self.n_wt),
                       desc='y coordinates of turbines in wind dir. ref. frame', units=self.units)
        # self.add_output('constraint_violation_' + self.const_id, val=0.0)
        # Explicitly size output array
        self.add_output(self.constraint_key, val=np.zeros(self.veclen),
                        desc='spacing of all turbines in the wind farm')

        col_pairs = np.array([(i, j) for i in range(self.n_wt - 1) for j in range(i + 1, self.n_wt)])
        if self.aggregation_function:
            self.declare_partials(self.constraint_key, [topfarm.x_key, topfarm.y_key])

            self.partial_indices = np.array([np.r_[np.where(col_pairs[:, 1] == i)[0], np.where(col_pairs[:, 0] == i)[0]]
                                             for i in range(self.n_wt)]).T
            self.partial_sign = (np.ones((self.n_wt, self.n_wt)) - 2 * np.triu(np.ones((self.n_wt, self.n_wt)), 1))[:-1]
            self.col_pairs = col_pairs
        else:
            # Sparse partial declaration
            cols = np.asarray(col_pairs.flatten(), dtype=np.int32)
            rows = np.asarray(np.repeat(np.arange(self.veclen), 2), dtype=np.int32)

            self.declare_partials(self.constraint_key,
                                  [topfarm.x_key, topfarm.y_key],
                                  rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        self.x = inputs[topfarm.x_key]
        self.y = inputs[topfarm.y_key]
        separation_squared = self._compute(self.x, self.y)
        if self.aggregation_function:
            if self.full_aggregation:
                outputs[self.constraint_key] = self.aggregation_function(separation_squared)
            else:
                outputs[self.constraint_key] = self.aggregation_function(
                    separation_squared[self.partial_indices], 0)
                # print(outputs[self.constraint_key])
        else:
            outputs[self.constraint_key] = separation_squared
        # outputs['constraint_violation_' + self.const_id] = -np.minimum(separation_squared - self.min_spacing_scaled**2, 0).sum()

    def _compute(self, x, y):
        n_wt = self.n_wt

        x_eff, y_eff = x, y
        if self.turbine_diameter is not None and self.turbine_diameter > 0:
            x_eff = x / self.turbine_diameter
            y_eff = y / self.turbine_diameter

        # compute distance matrixes
        dX, dY = [np.subtract(*np.meshgrid(xy, xy, indexing='ij')).T
                  for xy in [x_eff, y_eff]]
        dXY2 = dX**2 + dY**2
        # return upper triangle (above diagonal)
        return dXY2[np.triu_indices(n_wt, 1)]

    def compute_partials(self, inputs, J):
        # obtain necessary inputs
        x = inputs[topfarm.x_key]
        y = inputs[topfarm.y_key]

        # gradient of spacing [(i,j) for i=0..n_wt-1 and j=i+1..n_wt] wrt (wt_x[i], wt_x[j]) and (wt_y[i], wt_y[j])
        # Note: _compute_partials also needs to be aware of scaling
        dS_dxij, dS_dyij = self._compute_partials(x, y)

        if self.aggregation_function:
            # self._compute(x,y) returns scaled separations if turbine_diameter is set
            separation_squared_eff = self._compute(x, y)
            if self.full_aggregation:
                # gradient of aggregated (minimum) spacing wrt. spacing(i,j)
                dSagg_dS = self.aggregation_function.gradient(separation_squared_eff).flatten()
                # partial_indices extracts the spacing elements that each wt contributes to
                # partial sign gives it the right sign
                # and finally we sum the contributions of each wt
                J[self.constraint_key, topfarm.x_key] = (
                    (dS_dxij[:, 0] * dSagg_dS)[self.partial_indices] * self.partial_sign).sum(0).T
                J[self.constraint_key, topfarm.y_key] = (
                    (dS_dyij[:, 0] * dSagg_dS)[self.partial_indices] * self.partial_sign).sum(0).T
            else:
                # gradient of aggregated (minimum) spacing wrt. spacing(i,j)
                dSdwtx = (dS_dxij[:, 0])[self.partial_indices] * self.partial_sign
                dSdwty = (dS_dyij[:, 0])[self.partial_indices] * self.partial_sign
                S = self._compute(x, y)[self.partial_indices]
                dSagg_dS = self.aggregation_function.gradient(S, 0)

                # partial_indices extracts the spacing elements that each wt contributes to
                # partial sign gives it the right sign
                # and finally we sum the contributions of each wt

                dSagg_dwtx = dSdwtx * dSagg_dS * self.partial_sign
                dSagg_dwty = dSdwty * dSagg_dS * self.partial_sign

                dSagg_dx = np.zeros((self.n_wt, self.n_wt))  # np.diag((dSagg_dwtx).sum(0))
                dSagg_dy = np.zeros((self.n_wt, self.n_wt))  # np.diag((dSagg_dwty).sum(0))
                i = range(self.n_wt)
                for j in range(dSagg_dwtx.shape[0]):
                    ai, bi = self.col_pairs[self.partial_indices][j, i, :].T
                    dSagg_dx[ai, i] += dSagg_dwtx[j, i]
                    dSagg_dx[bi, i] -= dSagg_dwtx[j, i]
                    dSagg_dy[ai, i] += dSagg_dwty[j, i]
                    dSagg_dy[bi, i] -= dSagg_dwty[j, i]

                J[self.constraint_key, topfarm.x_key] = dSagg_dx.T
                J[self.constraint_key, topfarm.y_key] = dSagg_dy.T
        else:
            # populate Jacobian dict
            J[self.constraint_key, topfarm.x_key] = dS_dxij.flatten()
            J[self.constraint_key, topfarm.y_key] = dS_dyij.flatten()

    def _compute_partials(self, x, y):
        # get number of turbines
        n_wt = self.n_wt

        x_eff, y_eff = x, y
        scaling_factor = 1.0
        if self.turbine_diameter is not None and self.turbine_diameter > 0:
            x_eff = x / self.turbine_diameter
            y_eff = y / self.turbine_diameter
            scaling_factor = 1.0 / self.turbine_diameter

        # S_scaled = ((xi-xj)/D)^2 + ((yi-yj)/D)^2
        # dS_scaled/dx_phys_i = (2 * (xi-xj)/D^2) * (1) = 2 * dx_scaled / D
        # dS_scaled/dx_phys_j = (2 * (xi-xj)/D^2) * (-1) = -2 * dx_scaled / D

        # compute distance matrixes (on scaled coordinates)
        dX_eff, dY_eff = [np.subtract(*np.meshgrid(xy, xy, indexing='ij')).T
                          for xy in [x_eff, y_eff]]

        # upper triangle -> 1 row per WT pair
        dx_eff, dy_eff = dX_eff[np.triu_indices(n_wt, 1)], dY_eff[np.triu_indices(n_wt, 1)]

        # Partials of scaled separation squared wrt physical coordinates
        # Need to apply chain rule: d(S_scaled)/dx_phys = d(S_scaled)/dx_scaled * dx_scaled/dx_phys
        # dx_scaled/dx_phys is 1/D
        dSdx_phys = np.array([-2 * dx_eff * scaling_factor, 2 * dx_eff * scaling_factor]).T
        dSdy_phys = np.array([-2 * dy_eff * scaling_factor, 2 * dy_eff * scaling_factor]).T

        return dSdx_phys, dSdy_phys


    def plot(self, ax=None):
        from matplotlib.pyplot import Circle
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()

        def get_xy(xy):
            if not hasattr(self, xy):
                setattr(self, xy, dict(self.list_inputs(out_stream=None))[f'constraint_group.{self.name}.{xy}']['value'])
            xy = getattr(self, xy)
            return xy if not isinstance(xy, tuple) else xy[0]

        for x, y in zip(get_xy('x'), get_xy('y')):
            # Plotting should use the scaled radius if coordinates are scaled
            plot_radius = self.min_spacing_scaled / 2
            circle = Circle((x, y), plot_radius, color='k', ls='--', fill=False)
            ax.add_artist(circle)

    def satisfy(self, state, n_iter=100, step_size=0.1):
        x, y = [state[xy].astype(float) for xy in [topfarm.x_key, topfarm.y_key]]
        pair_i, pair_j = np.triu_indices(len(x), 1)
        for _ in range(n_iter):
            dist_sq = self._compute(x, y) # This is squared distance (potentially scaled)
            dx_sq, dy_sq = self._compute_partials(x, y) # Partials of squared distance
            index = np.argmin(dist_sq)

            # Compare with squared scaled min_spacing
            if dist_sq[index] < self.min_spacing_scaled**2:
                i, j = pair_i[index], pair_j[index]
                # dx_sq, dy_sq are partials of d^2, not d.
                # The original satisfy logic might need review if it assumes partials of d.
                # However, using them as gradient direction for d^2 is okay.
                x[i] += dx_sq[index, 0] * step_size
                x[j] += dx_sq[index, 1] * step_size
                y[i] += dy[index, 0] * step_size
                y[j] += dy[index, 1] * step_size
            else:
                break
        state.update({topfarm.x_key: x, topfarm.y_key: y})
        return state


class SpacingTypeConstraint(SpacingConstraint):
    def __init__(self, min_spacing, units=None, aggregation_function=None, full_aggregation=False, name='spacing_type_comp', turbine_diameter=None):
        """Initialize SpacingTypeConstraint

        Parameters
        ----------
        min_spacing : array_like
            Minimum spacing for each turbine type [m or m/D if turbine_diameter provided].
            If turbine_diameter is provided, these are scaled to diameters.
        aggregation_function : topfarm.utils.AggregationFunction or None
            (description as in parent)
        turbine_diameter : float, optional
            Turbine diameter for scaling. Default is None.
        """
        # Pass turbine_diameter to SpacingConstraint's __init__
        # SpacingConstraint.__init__ will handle scaling of min_spacing (which can be an array here)
        super().__init__(min_spacing=np.asarray(min_spacing), units=units,
                         aggregation_function=aggregation_function,
                         full_aggregation=full_aggregation, name=name,
                         turbine_diameter=turbine_diameter)
        # After super().__init__, self.min_spacing is now potentially scaled (element-wise if array)
        # self.turbine_diameter is also set on the instance by super.

    def _setup(self, problem, **kwargs): # Added **kwargs
        self.n_wt = problem.n_wt
        td_from_problem = kwargs.get('turbine_diameter')

        # self.min_spacing is already scaled by SpacingConstraint's __init__.
        # td_from_problem is passed to SpacingTypeComp for its own reference.
        # Use td_from_problem if available, otherwise fallback to self.turbine_diameter (from __init__)
        effective_td_for_comp = td_from_problem if td_from_problem is not None else self.turbine_diameter

        self.spacing_comp = SpacingTypeComp(self.n_wt, self.min_spacing, self.const_id, self.units,
                                            aggregation_function=self.aggregation_function,
                                            full_aggregation=self.full_aggregation,
                                            turbine_diameter=effective_td_for_comp, # Pass effective diameter
                                            types=getattr(problem.design_vars, topfarm.type_key, None) # Pass initial types if available
                                            )
        problem.model.constraint_group.add_subsystem(self.const_id, self.spacing_comp,
                                                     promotes=[topfarm.x_key, topfarm.y_key, topfarm.type_key, 'wtRelativeSeparationSquared'])

    # setup_as_constraint is inherited from SpacingConstraint but needs to call the overridden _setup.
    # The Constraint ABC was changed so setup_as_constraint takes problem, **kwargs.
    # SpacingConstraint.setup_as_constraint already passes **kwargs to its _setup.
    # This means SpacingTypeConstraint.setup_as_constraint will correctly pass kwargs.
    # However, the constraint name is different.

    def setup_as_constraint(self, problem, **kwargs): # Added **kwargs
        self._setup(problem, **kwargs) # Pass **kwargs
        # For SpacingTypeConstraint, the output is 'wtRelativeSeparationSquared' and lower bound is 0
        problem.model.add_constraint('wtRelativeSeparationSquared', lower=0)
        # Note: setup_as_penalty will be inherited and should work correctly.


class SpacingTypeComp(SpacingComp):
    """
    Calculates inter-turbine spacing for all turbine pairs.

    """

    def __init__(self, n_wt, min_spacings_scaled, const_id=None, units=None, aggregation_function=None, full_aggregation=False, types=None, turbine_diameter=None):
        # min_spacings_scaled is the array of min_spacing values, potentially scaled by turbine_diameter
        # by SpacingTypeConstraint (via SpacingConstraint's __init__)
        super().__init__(n_wt=n_wt, min_spacing_scaled=min_spacings_scaled, # Pass to SpacingComp's constructor
                         const_id=const_id, units=units,
                         aggregation_function=aggregation_function, full_aggregation=full_aggregation,
                         turbine_diameter=turbine_diameter) # Pass for consistency
        self.constraint_key = 'wtRelativeSeparationSquared'
        self.types = types # Initial types, can be updated via input
        # self.min_spacing_scaled from super() is now the array of scaled minimum spacings for each type

    def setup(self):
        super().setup()
        self.add_input(topfarm.type_key, val=self.types or np.zeros(self.n_wt),
                       desc='turbine type number')

    def compute(self, inputs, outputs):
        self.x = inputs[topfarm.x_key]
        self.y = inputs[topfarm.y_key]
        self.type = inputs[topfarm.type_key]
        relative_separation_squared = self._compute(self.x, self.y, self.type)
        if self.aggregation_function:
            if self.full_aggregation:
                outputs[self.constraint_key] = self.aggregation_function(relative_separation_squared)
            else:
                outputs[self.constraint_key] = self.aggregation_function(
                    relative_separation_squared[self.partial_indices], 0)
                # print(outputs[self.constraint_key])
        else:
            outputs[self.constraint_key] = relative_separation_squared
        # outputs['constraint_violation_' + self.const_id] = -np.minimum(relative_separation_squared, 0).sum()

    def get_min_eff_spacing(self, t):
        # self.min_spacing_scaled is the array of per-type scaled minimum spacings
        # t contains the type index for each turbine
        types_int = np.atleast_1d(t).astype(int)
        # Effective spacing for a pair (type_i, type_j) is (min_spacing_scaled_i + min_spacing_scaled_j) / 2
        # This calculation results in a matrix of effective scaled spacings for all type pairs.
        return (self.min_spacing_scaled[types_int][:, na] + self.min_spacing_scaled[types_int][na, :]) / 2

    def _compute(self, x, y, t):
        n_wt = self.n_wt
        # x, y are turbine coordinates (potentially scaled by D)
        # Compute squared physical distances (or squared scaled distances if x,y are scaled)
        dX, dY = [np.subtract(*np.meshgrid(xy, xy, indexing='ij')).T
                  for xy in [x, y]]
        dist_sq_matrix = dX**2 + dY**2

        # Get effective minimum spacing (already scaled if D was used) for each pair of turbine types
        min_eff_spacing_matrix_scaled = self.get_min_eff_spacing(t)

        # The constraint is (dist_ij / D)^2 >= ( (min_spacing_type_i/D + min_spacing_type_j/D) / 2 )^2
        # Or, if we work with relative separation: dist_sq_scaled - min_eff_spacing_scaled^2 >= 0
        # wtRelativeSeparationSquared = dist_sq_matrix - min_eff_spacing_matrix_scaled**2
        # The output 'wtRelativeSeparationSquared' should be >= 0 for constraint satisfaction.
        # So, it should represent: actual_sep_sq_scaled - required_sep_sq_scaled

        # Taking the upper triangle for unique pairs
        # dXY2 = dX**2 + dY**2 - self.get_min_eff_spacing(t)**2 # Original line
        # This seems correct: dXY2 is already (actual_dist_sq - required_dist_sq) in scaled units.
        # If x,y are scaled by D, then dX^2, dY^2 are (dist/D)^2.
        # get_min_eff_spacing(t) returns (min_spacing_for_pair / D). So squared is (min_spacing_for_pair/D)^2.
        # Thus the result is (actual_dist/D)^2 - (required_dist_for_pair/D)^2. This should be >= 0.

        relative_separation_sq_matrix = dist_sq_matrix - min_eff_spacing_matrix_scaled**2
        return relative_separation_sq_matrix[np.triu_indices(n_wt, 1)]


    def plot(self, ax=None):
        from matplotlib.pyplot import Circle
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()

        def get_xy(xy):
            if not hasattr(self, xy):
                setattr(self, xy, dict(self.list_inputs(out_stream=None))[f'constraint_group.{self.name}.{xy}']['value'])
            xy = getattr(self, xy)
            return xy if not isinstance(xy, tuple) else xy[0]

        for x_i, y_i, t_i in zip(get_xy('x'), get_xy('y'), get_xy('type')):
            # For plotting, we need the radius for this specific turbine's type.
            # get_min_eff_spacing(t_i) would give a row/col if t_i is scalar.
            # We need individual min_spacing for type t_i, which is self.min_spacing_scaled[int(t_i)].
            # The effective radius for plotting for a single turbine is half its own min_spacing (scaled).
            plot_radius = self.min_spacing_scaled[int(t_i)] / 2.0
            circle = Circle((x_i, y_i), plot_radius, color='k', ls='--', fill=False)
            ax.add_artist(circle)

    def satisfy(self, state, n_iter=100, step_size=0.1):
        x, y, t_types = [state[xy].astype(float) for xy in [topfarm.x_key, topfarm.y_key, topfarm.type_key]]
        pair_i, pair_j = np.triu_indices(len(x), 1) # Indices of upper triangle for pairs

        for _ in range(n_iter):
            # _compute returns relative_separation_squared for all pairs (dist_sq_scaled - required_eff_dist_sq_scaled)
            relative_dist_sq_all_pairs = self._compute(x, y, t_types)

            # We need partials of actual squared distance, not relative squared distance for moving points.
            # SpacingComp._compute_partials calculates d(dist_sq)/dx, d(dist_sq)/dy
            dx_sq_partials, dy_sq_partials = super()._compute_partials(x, y) # Call parent's method

            index_of_min_relative_dist = np.argmin(relative_dist_sq_all_pairs)

            # If min relative distance is negative, constraint is violated for this pair
            if relative_dist_sq_all_pairs[index_of_min_relative_dist] < 0:
                i, j = pair_i[index_of_min_relative_dist], pair_j[index_of_min_relative_dist]

                # Use partials of actual squared distance to move points
                # dx_sq_partials[index_of_min_relative_dist, 0] is d(dist_ij^2)/dx_i
                # dx_sq_partials[index_of_min_relative_dist, 1] is d(dist_ij^2)/dx_j
                x[i] += dx_sq_partials[index_of_min_relative_dist, 0] * step_size
                x[j] += dx_sq_partials[index_of_min_relative_dist, 1] * step_size
                y[i] += dy[index, 0] * step_size
                y[j] += dy[index, 1] * step_size
            else:
                break
        state.update({topfarm.x_key: x, topfarm.y_key: y})
        return state
