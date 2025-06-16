from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
import topfarm
import numpy as np
from py_wake.flow_map import Points
from py_wake.utils.gradients import autograd


class PyWakeAEPCostModelComponent(AEPCostModelComponent):
    """TOPFARM wrapper for PyWake AEP calculator"""

    def __init__(self, windFarmModel, n_wt, wd=None, ws=None, max_eval=None, grad_method=autograd, n_cpu=1, farm_capacity_scaling=True, **kwargs):
        """Initialize wrapper for PyWake AEP calculator

        Parameters
        ----------
        windFarmModel : DeficitModel
            Wake deficit model used
        n_wt : int
            Number of wind turbines
        wd : array_like
            Wind directions to study
        ws : array_like
            Wind speeds to study
        max_eval : int
            Maximum number of function evaluations
        grad_method : function handle
            Selected method to calculate gradients, default is autograd
        n_cpu : int
            Number of CPUs to use for AEP calculation
        farm_capacity_scaling : bool, optional
            If True, scale AEP by total farm capacity (default is True).
            AEP is returned in hours if scaled (GWh/GW), otherwise GWh.
        """
        self.windFarmModel = windFarmModel
        self.n_cpu = n_cpu
        self.n_wt_for_capacity = n_wt # Used for capacity calculation, may be overridden by subclass
        self.farm_capacity_scaling = farm_capacity_scaling

        def aep(**aep_kwargs):
            try:
                raw_aep_gwh = self.windFarmModel.aep(x=aep_kwargs[topfarm.x_key],
                                                     y=aep_kwargs[topfarm.y_key],
                                                     h=aep_kwargs.get(topfarm.z_key, None),
                                                     type=aep_kwargs.get(topfarm.type_key, 0),
                                                     wd=wd, ws=ws,
                                                     n_cpu=n_cpu) # This is usually in GWh

                if self.farm_capacity_scaling:
                    current_types = aep_kwargs.get(topfarm.type_key, 0)
                    if isinstance(current_types, (int, float, np.integer)): # Homogeneous or type_key not provided
                        current_types = np.full(self.n_wt_for_capacity, current_types)
                    elif hasattr(current_types, '__len__') and len(current_types) > self.n_wt_for_capacity:
                        # Ensure we only consider types for the n_wt_for_capacity turbines
                        current_types = current_types[:self.n_wt_for_capacity]

                    total_capacity_kw = 0
                    # Ensure current_types corresponds to the self.n_wt_for_capacity turbines
                    # If type_key has fewer elements than n_wt_for_capacity, this might be an issue.
                    # Assuming type_key, if array, matches length of x/y design vars.

                    # Iterate up to self.n_wt_for_capacity, using type array
                    for i in range(self.n_wt_for_capacity):
                        turbine_type_idx = int(current_types[i] if hasattr(current_types, '__len__') and i < len(current_types) else current_types)
                        total_capacity_kw += self.windFarmModel.windTurbines.power(type=turbine_type_idx) # Power in kW

                    if total_capacity_kw > 0:
                        # Convert AEP GWh to Wh (1e9), Capacity kW to W (1e3) -> scaled AEP in hours
                        # Scaled AEP = (Raw_AEP_GWh * 1e9 Wh/GWh) / (Total_Capacity_kW * 1e3 W/kW)
                        # This results in hours. If output_unit remains 'GWh', this is effectively GWh_hours/GW_capacity_equivalent_hours
                        # To keep output_unit as 'GWh' but have it represent GWh/GW:
                        # Scaled AEP = Raw_AEP_GWh / (Total_Capacity_kW * 1e-6 GW/kW)
                        total_capacity_gw = total_capacity_kw * 1.0e-6
                        return raw_aep_gwh / total_capacity_gw
                    else:
                        return 0 # Avoid division by zero, though capacity should be > 0
                else:
                    return raw_aep_gwh
            except ValueError as e:
                if 'are at the same position' in str(e):
                    return 0
                raise ValueError(
                    f"{str(e)}\n\n ^^^^^ PyWake model call failed with an error  ^^^^^"
                ) from e

        if grad_method:
            if hasattr(self.windFarmModel, 'dAEPdxy'):
                # for backward compatibility
                dAEPdxy = self.windFarmModel.dAEPdxy(grad_method)
            else:
                def dAEPdxy_func(**aep_kwargs): # Renamed to avoid conflict
                    return self.windFarmModel.aep_gradients(
                        gradient_method=grad_method, wrt_arg=['x', 'y'], n_cpu=n_cpu, **aep_kwargs)

            def daep(**aep_kwargs):
                raw_daep = dAEPdxy_func(x=aep_kwargs[topfarm.x_key],
                                        y=aep_kwargs[topfarm.y_key],
                                        h=aep_kwargs.get(topfarm.z_key, None),
                                        type=aep_kwargs.get(topfarm.type_key, 0),
                                        wd=wd, ws=ws)
                if self.farm_capacity_scaling:
                    current_types = aep_kwargs.get(topfarm.type_key, 0)
                    if isinstance(current_types, (int, float, np.integer)):
                        current_types = np.full(self.n_wt_for_capacity, current_types)
                    elif hasattr(current_types, '__len__') and len(current_types) > self.n_wt_for_capacity:
                         current_types = current_types[:self.n_wt_for_capacity]

                    total_capacity_kw = 0
                    for i in range(self.n_wt_for_capacity):
                        turbine_type_idx = int(current_types[i] if hasattr(current_types, '__len__') and i < len(current_types) else current_types)
                        total_capacity_kw += self.windFarmModel.windTurbines.power(type=turbine_type_idx)

                    if total_capacity_kw > 0:
                        total_capacity_gw = total_capacity_kw * 1.0e-6
                        return raw_daep / total_capacity_gw
                    else:
                        # Return zero gradients or handle error appropriately
                        return np.zeros_like(raw_daep) if raw_daep is not None else None
                else:
                    return raw_daep
        else:
            daep = None

        # output_unit remains 'GWh', but if scaled, it's effectively GWh/GW (i.e., hours * 1000)
        # Or, more precisely, if AEP is GWh and capacity is GW, then scaled AEP is GWh/GW.
        # If user wants truly dimensionless hours, AEP should be Wh and capacity W.
        # Current scaling: AEP_GWh / Capacity_GW. Units are "GWh/GW".
        # If we want "hours", it would be (AEP_GWh * 1e9) / (Capacity_kW * 1e3) = AEP_GWh / Capacity_GW * 1000.
        # For now, output_unit='GWh' means the value is in GWh if not scaled, or GWh/GW if scaled.
        AEPCostModelComponent.__init__(self,
                                       input_keys=[topfarm.x_key, topfarm.y_key],
                                       n_wt=n_wt, # This n_wt is for the AEPCostModelComponent's internal array sizing
                                       cost_function=aep,
                                       cost_gradient_function=daep,
                                       output_unit='GWh', # Or change to 'GWh/GW' or 'h' if conversion is fixed
                                       max_eval=max_eval, **kwargs)

    def get_aep4smart_start(self, ws=[6, 8, 10, 12, 14], wd=np.arange(360), type=0, **kwargs):
        """Compute AEP with a smart start approach"""
        def aep4smart_start(X, Y, wt_x, wt_y, T=0, wt_t=0):
            H = np.full(X.shape, self.windFarmModel.windTurbines.hub_height())
            if type == 0:
                sim_res = self.windFarmModel(wt_x, wt_y, type=wt_t, wd=wd, ws=ws, n_cpu=self.n_cpu, **kwargs)
                next_type = T
            else:
                type_ = np.atleast_1d(type)
                t = np.zeros_like(wt_x) + type_[:len(wt_x)]
                sim_res = self.windFarmModel(wt_x, wt_y, type=t, wd=wd, ws=ws, n_cpu=self.n_cpu, **kwargs)
                H = np.full(X.shape, self.windFarmModel.windTurbines.hub_height())
                next_type = type_[min(len(type_) - 1, len(wt_x) + 1)]
            return sim_res.aep_map(Points(X, Y, H), type=next_type, n_cpu=self.n_cpu).values
        return aep4smart_start


class PyWakeAEPCostModelComponentAdditionalTurbines(PyWakeAEPCostModelComponent):
    '''PyWake AEP component that allows for including additional turbine positions that are not
    considered design variables but still considered for wake effect. Note that this functionality
    can be limited by your wind farm models ability to predict long distance wakes.'''

    def __init__(self, windFarmModel, n_wt, add_wt_x, add_wt_y, add_wt_type=0, add_wt_h=None,
                 wd=None, ws=None, max_eval=None, grad_method=autograd, n_cpu=1, farm_capacity_scaling=True, **kwargs):
        # n_wt here is the number of primary turbines (design variables)
        self.x2, self.y2 = add_wt_x, add_wt_y # Coordinates of additional, fixed turbines
        # self.windFarmModel, self.n_cpu will be set by super()
        # self.farm_capacity_scaling and self.n_wt_for_capacity will be set by super() as well.
        # n_wt_for_capacity will correctly be n_wt (primary turbines for this class).

        # The aep and daep functions are defined here and will be passed to the grandparent AEPCostModelComponent.
        # They will use self.farm_capacity_scaling and self.n_wt_for_capacity set by the parent's __init__.

        def aep(**aep_kwargs):
            x_primary = aep_kwargs[topfarm.x_key]
            y_primary = aep_kwargs[topfarm.y_key]
            x_all = np.concatenate([x_primary, self.x2])
            y_all = np.concatenate([y_primary, self.y2])

            h_all = None
            # Default hub height from windFarmModel if not specified
            default_h = self.windFarmModel.windTurbines.hub_height()
            h_primary_val = aep_kwargs.get(topfarm.z_key, default_h)
            if not isinstance(h_primary_val, (np.ndarray, list)): h_primary_val = np.full_like(x_primary, h_primary_val)

            h_secondary_val = add_wt_h if add_wt_h is not None else default_h
            if not isinstance(h_secondary_val, (np.ndarray, list)): h_secondary_val = np.full_like(self.x2, h_secondary_val)
            h_all = np.concatenate((h_primary_val[:len(x_primary)], h_secondary_val[:len(self.x2)]))

            type_primary_val = aep_kwargs.get(topfarm.type_key, 0)
            if not isinstance(type_primary_val, (np.ndarray, list)): type_primary_val = np.full_like(x_primary, type_primary_val)

            type_secondary_val = add_wt_type
            if not isinstance(type_secondary_val, (np.ndarray, list)): type_secondary_val = np.full_like(self.x2, type_secondary_val)
            type_all = np.concatenate([type_primary_val[:len(x_primary)], type_secondary_val[:len(self.x2)]])

            try:
                raw_aep_gwh_total_primary = self.windFarmModel(x=x_all,
                                                               y=y_all,
                                                               h=h_all,
                                                               type=type_all,
                                                               wd=wd, ws=ws,
                                                               n_cpu=self.n_cpu).aep().sum(['wd', 'ws']).values[:self.n_wt_for_capacity].sum()

                if self.farm_capacity_scaling:
                    # Capacity calculation for primary turbines
                    types_for_capacity_calc = type_primary_val[:self.n_wt_for_capacity]
                    # Ensure it's an array for consistent processing
                    if isinstance(types_for_capacity_calc, (int, float, np.integer)):
                        types_for_capacity_calc = np.full(self.n_wt_for_capacity, types_for_capacity_calc)

                    total_capacity_kw_primary = 0
                    for i in range(self.n_wt_for_capacity): # Use n_wt_for_capacity (which is n_wt for this class)
                        turbine_type_idx = int(types_for_capacity_calc[i])
                        total_capacity_kw_primary += self.windFarmModel.windTurbines.power(type=turbine_type_idx)

                    if total_capacity_kw_primary > 0:
                        total_capacity_gw_primary = total_capacity_kw_primary * 1.0e-6
                        return raw_aep_gwh_total_primary / total_capacity_gw_primary
                    else:
                        return 0.0
                else:
                    return raw_aep_gwh_total_primary
            except ValueError as e:
                if 'are at the same position' in str(e):
                    return 0.0
                raise ValueError(
                    f"{str(e)}\n\n ^^^^^ PyWake model call failed with an error  ^^^^^"
                ) from e

        if grad_method:
            if hasattr(self.windFarmModel, 'dAEPdxy'):
                # for backward compatibility
                dAEPdxy = self.windFarmModel.dAEPdxy(grad_method)
            else:
                def dAEPdxy_func(**aep_kwargs_grad):
                    return self.windFarmModel.aep_gradients(
                        gradient_method=grad_method, wrt_arg=['x', 'y'], n_cpu=self.n_cpu, **aep_kwargs_grad)

            def daep(**aep_kwargs):
                x_primary = aep_kwargs[topfarm.x_key]
                y_primary = aep_kwargs[topfarm.y_key]
                x_all = np.concatenate([x_primary, self.x2])
                y_all = np.concatenate([y_primary, self.y2])

                default_h = self.windFarmModel.windTurbines.hub_height()
                h_primary_val = aep_kwargs.get(topfarm.z_key, default_h)
                if not isinstance(h_primary_val, (np.ndarray, list)): h_primary_val = np.full_like(x_primary, h_primary_val)
                h_secondary_val = add_wt_h if add_wt_h is not None else default_h
                if not isinstance(h_secondary_val, (np.ndarray, list)): h_secondary_val = np.full_like(self.x2, h_secondary_val)
                h_all = np.concatenate((h_primary_val[:len(x_primary)], h_secondary_val[:len(self.x2)]))

                type_primary_val = aep_kwargs.get(topfarm.type_key, 0)
                if not isinstance(type_primary_val, (np.ndarray, list)): type_primary_val = np.full_like(x_primary, type_primary_val)
                type_secondary_val = add_wt_type
                if not isinstance(type_secondary_val, (np.ndarray, list)): type_secondary_val = np.full_like(self.x2, type_secondary_val)
                type_all = np.concatenate([type_primary_val[:len(x_primary)], type_secondary_val[:len(self.x2)]])

                raw_grad = dAEPdxy_func(x=x_all,
                                        y=y_all,
                                        h=h_all,
                                        type=type_all,
                                        wd=wd, ws=ws)[:, :self.n_wt_for_capacity] # Slice for primary turbines (n_wt_for_capacity is n_wt)

                if self.farm_capacity_scaling:
                    types_for_capacity_calc = type_primary_val[:self.n_wt_for_capacity]
                    if isinstance(types_for_capacity_calc, (int, float, np.integer)):
                        types_for_capacity_calc = np.full(self.n_wt_for_capacity, types_for_capacity_calc)

                    total_capacity_kw_primary = 0
                    for i in range(self.n_wt_for_capacity):
                        turbine_type_idx = int(types_for_capacity_calc[i])
                        total_capacity_kw_primary += self.windFarmModel.windTurbines.power(type=turbine_type_idx)

                    if total_capacity_kw_primary > 0:
                        total_capacity_gw_primary = total_capacity_kw_primary * 1.0e-6
                        return raw_grad / total_capacity_gw_primary
                    else:
                        return np.zeros_like(raw_grad) if raw_grad is not None else None
                else:
                    return raw_grad
        else:
            daep = None

        # Call PyWakeAEPCostModelComponent's __init__ to set up shared attributes
        # like self.farm_capacity_scaling and self.n_wt_for_capacity.
        # This will internally call AEPCostModelComponent's __init__ but with its *own* aep/daep definitions.
        super().__init__(windFarmModel=windFarmModel, n_wt=n_wt, wd=wd, ws=ws,
                         max_eval=max_eval, grad_method=grad_method, n_cpu=n_cpu,
                         farm_capacity_scaling=farm_capacity_scaling, **kwargs)

        # Explicitly override the cost_function and cost_gradient_function on the instance
        # to use the versions defined locally in this __init__ which handle additional turbines.
        # AEPCostModelComponent (the grandparent) will then use these.
        self.cost_function = aep
        self.cost_gradient_function = daep
        # Ensure self.n_wt_for_capacity is correctly set for this subclass context if super() didn't quite align.
        # (n_wt parameter to this class's __init__ is the number of primary turbines)
        self.n_wt_for_capacity = n_wt


def main():
    if __name__ == '__main__':
        n_wt = 16
        site = IEA37Site(n_wt)
        windTurbines = IEA37_WindTurbines()
        windFarmModel = IEA37SimpleBastankhahGaussian(site, windTurbines)
        tf = TopFarmProblem(
            design_vars=dict(zip('xy', site.initial_position.T)),
            cost_comp=PyWakeAEPCostModelComponent(windFarmModel, n_wt),
            driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(), max_iter=5),
            constraints=[CircleBoundaryConstraint([0, 0], 1300.1)],
            plot_comp=XYPlotComp())
        tf.optimize()
        tf.plot_comp.show()


main()
