import numpy as np
import pytest

from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import (
    PyWakeAEPCostModelComponentAdditionalTurbines,
    PyWakeAEPCostModelComponent,
)
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import NoPlot
from topfarm.tests import npt

from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import wt_x, wt_y, LillgrundSite, SWT2p3_93_65


def test_PyWakeAEPCostModelComponentAdditionalTurbines():
    x2 = np.array([363089.20620581, 362841.19815026])
    y2 = np.array([6154000, 6153854.5244973])
    wind_turbines = SWT2p3_93_65()
    x = wt_x[:4]
    y = wt_y[:4]
    n_wt = len(x)
    site = LillgrundSite()
    wf_model = BastankhahGaussian(site, wind_turbines)
    constraint_comp = XYBoundaryConstraint(np.asarray([x, y]).T)
    cost_comp = PyWakeAEPCostModelComponentAdditionalTurbines(
        windFarmModel=wf_model,
        n_wt=n_wt,
        add_wt_x=x2,
        add_wt_y=y2,
        grad_method=autograd,
    )
    plot_comp = NoPlot()
    problem = TopFarmProblem(
        design_vars={"x": x, "y": y},
        constraints=[
            constraint_comp,
            SpacingConstraint(min_spacing=wind_turbines.diameter() * 2),
        ],
        cost_comp=cost_comp,
        driver=EasyScipyOptimizeDriver(optimizer="SLSQP", maxiter=5),
        plot_comp=plot_comp,
    )

    cost, _, _ = problem.optimize(disp=True)
    npt.assert_almost_equal(cost, -3682.710308568642)


design_vars = {
    "x": np.array([0, 1]),
    "y": np.array([0, 1]),
    "z": np.array([10, 10]),
    "type": np.array([0, 0]),
}
UNHANDLED_PYWAKE_ERROR_MSG = "Unhandeled PyWake error"


class DummyWindTurbines:
    def hub_height(self):
        return 10


class DummyWindFarmModel:
    def __init__(self, mode="normal"):
        # mode: "normal", "raise_same", "raise_other"
        self.mode = mode
        self.windTurbines = DummyWindTurbines()

    def aep(self, x, y, h, type, wd, ws, n_cpu):
        if self.mode == "raise_same":
            raise ValueError("Error: turbines are at the same position")
        elif self.mode == "raise_other":
            raise ValueError(UNHANDLED_PYWAKE_ERROR_MSG)
        else:
            return 100

    # Minimal dummy implementations for gradients, not used in these tests
    def aep_gradients(self, gradient_method, wrt_arg, n_cpu, **kwargs):
        return np.array([[1, 1]])

    def __call__(self, *args, **kwargs):
        return self.aep(*args, **kwargs)


def test_aep_returns_correct_value():
    dummy_model = DummyWindFarmModel(mode="normal")
    comp = PyWakeAEPCostModelComponent(
        windFarmModel=dummy_model, n_wt=2, wd=[0], ws=[10]
    )
    # Call the cost_function defined internally
    result = comp.cost_function(**design_vars)
    assert result == 100


def test_aep_handles_same_position_error():
    dummy_model = DummyWindFarmModel(mode="raise_same")
    comp = PyWakeAEPCostModelComponent(
        windFarmModel=dummy_model, n_wt=2, wd=[0], ws=[10]
    )
    result = comp.cost_function(**design_vars)
    assert result == 0


def test_py_wake_wrapper_brings_up_failed_pywake_aep_call():
    dummy_model = DummyWindFarmModel(mode="raise_other")
    comp = PyWakeAEPCostModelComponent(
        windFarmModel=dummy_model, n_wt=2, wd=[0], ws=[10]
    )
    with pytest.raises(ValueError) as excinfo:
        comp.cost_function(**design_vars)
    # Verify that the exception message is augmented with the specific error string
    assert UNHANDLED_PYWAKE_ERROR_MSG in str(excinfo.value)


def test_additional_turbines_aep_handles_same_position_error():
    dummy_model = DummyWindFarmModel(mode="raise_same")
    comp = PyWakeAEPCostModelComponentAdditionalTurbines(
        windFarmModel=dummy_model,
        n_wt=2,
        add_wt_x=[0],
        add_wt_y=[0],
        grad_method=autograd,
    )
    result = comp.cost_function(**design_vars)
    assert result == 0


def test_additional_turbines_py_wake_wrapper_brings_up_failed_pywake_aep_call():
    dummy_model = DummyWindFarmModel(mode="raise_other")
    comp = PyWakeAEPCostModelComponentAdditionalTurbines(
        windFarmModel=dummy_model,
        n_wt=2,
        add_wt_x=[0],
        add_wt_y=[0],
        grad_method=autograd,
    )
    with pytest.raises(ValueError) as excinfo:
        comp.cost_function(**design_vars)
    assert UNHANDLED_PYWAKE_ERROR_MSG in str(excinfo.value)


class TestPyWakeAEPScaling(unittest.TestCase):
    def setUp(self):
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
        from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian

        self.windTurbines = IEA37_WindTurbines()
        self.site = IEA37Site(n_wt=2) # Site for 2 turbines
        self.windFarmModel = IEA37SimpleBastankhahGaussian(self.site, self.windTurbines)
        self.n_wt = 2
        self.initial_positions = np.array([[0,0], [self.windTurbines.diameter(0) * 4, 0]]) # 2 turbines, 4D spacing

        # Define simple uniform wind conditions
        self.wd = np.array([270.0])
        self.ws = np.array([10.0])

    def test_aep_scaling_on_off(self):
        # Case 1: farm_capacity_scaling = False (Raw AEP)
        cost_comp_no_scaling = PyWakeAEPCostModelComponent(
            windFarmModel=self.windFarmModel,
            n_wt=self.n_wt,
            wd=self.wd,
            ws=self.ws,
            farm_capacity_scaling=False
        )
        problem_no_scaling = TopFarmProblem(
            design_vars={'x': self.initial_positions[:,0], 'y': self.initial_positions[:,1]},
            cost_comp=cost_comp_no_scaling,
            driver=EasyScipyOptimizeDriver(maxiter=0) # No optimization, just eval
        )
        raw_aep_gwh, _ = problem_no_scaling.evaluate()
        # PyWakeAEPCostModelComponent has cost_factor=-1 by default for maximization
        raw_aep_gwh = -raw_aep_gwh

        # Case 2: farm_capacity_scaling = True (Scaled AEP)
        cost_comp_scaling = PyWakeAEPCostModelComponent(
            windFarmModel=self.windFarmModel,
            n_wt=self.n_wt,
            wd=self.wd,
            ws=self.ws,
            farm_capacity_scaling=True # Default, but explicit
        )
        problem_scaling = TopFarmProblem(
            design_vars={'x': self.initial_positions[:,0], 'y': self.initial_positions[:,1]},
            cost_comp=cost_comp_scaling,
            driver=EasyScipyOptimizeDriver(maxiter=0)
        )
        scaled_aep_val, _ = problem_scaling.evaluate()
        scaled_aep_val = -scaled_aep_val # Due to cost_factor

        # Calculate expected total capacity
        # Assuming homogeneous farm with type 0 turbines
        rated_power_kw_per_turbine = self.windFarmModel.windTurbines.power(type=0) # in kW
        total_capacity_kw = self.n_wt * rated_power_kw_per_turbine
        total_capacity_gw = total_capacity_kw * 1e-6

        self.assertTrue(total_capacity_gw > 0, "Total capacity must be positive for meaningful scaling.")

        expected_scaled_aep = raw_aep_gwh / total_capacity_gw

        npt.assert_allclose(scaled_aep_val, expected_scaled_aep, rtol=1e-5,
                             err_msg=f"Scaled AEP mismatch. Raw: {raw_aep_gwh} GWh, Scaled: {scaled_aep_val}, Expected Scaled: {expected_scaled_aep}, Capacity: {total_capacity_gw} GW")

    def test_aep_scaling_with_additional_turbines(self):
        # Test for PyWakeAEPCostModelComponentAdditionalTurbines
        add_wt_x = np.array([self.windTurbines.diameter(0) * 8]) # One additional turbine
        add_wt_y = np.array([0])
        add_wt_type = 0

        # Case 1: No scaling
        cost_comp_add_no_scaling = PyWakeAEPCostModelComponentAdditionalTurbines(
            windFarmModel=self.windFarmModel,
            n_wt=self.n_wt, # Primary turbines
            add_wt_x=add_wt_x, add_wt_y=add_wt_y, add_wt_type=add_wt_type,
            wd=self.wd, ws=self.ws,
            farm_capacity_scaling=False
        )
        problem_add_no_scaling = TopFarmProblem(
            design_vars={'x': self.initial_positions[:,0], 'y': self.initial_positions[:,1]},
            cost_comp=cost_comp_add_no_scaling
        )
        raw_aep_add_gwh, _ = problem_add_no_scaling.evaluate()
        raw_aep_add_gwh = -raw_aep_add_gwh

        # Case 2: With scaling
        cost_comp_add_scaling = PyWakeAEPCostModelComponentAdditionalTurbines(
            windFarmModel=self.windFarmModel,
            n_wt=self.n_wt, # Primary turbines
            add_wt_x=add_wt_x, add_wt_y=add_wt_y, add_wt_type=add_wt_type,
            wd=self.wd, ws=self.ws,
            farm_capacity_scaling=True
        )
        problem_add_scaling = TopFarmProblem(
            design_vars={'x': self.initial_positions[:,0], 'y': self.initial_positions[:,1]},
            cost_comp=cost_comp_add_scaling
        )
        scaled_aep_add_val, _ = problem_add_scaling.evaluate()
        scaled_aep_add_val = -scaled_aep_add_val

        # Capacity should ONLY be for the primary n_wt turbines
        rated_power_kw_per_turbine = self.windFarmModel.windTurbines.power(type=0)
        total_capacity_kw_primary = self.n_wt * rated_power_kw_per_turbine
        total_capacity_gw_primary = total_capacity_kw_primary * 1e-6

        self.assertTrue(total_capacity_gw_primary > 0)
        expected_scaled_aep_add = raw_aep_add_gwh / total_capacity_gw_primary

        npt.assert_allclose(scaled_aep_add_val, expected_scaled_aep_add, rtol=1e-5,
                             err_msg=f"Scaled AEP (add turbines) mismatch. Raw: {raw_aep_add_gwh} GWh, Scaled: {scaled_aep_add_val}, Expected: {expected_scaled_aep_add}, Capacity (primary): {total_capacity_gw_primary} GW")

# Need to import unittest if it's not already at the top of the file
import unittest
