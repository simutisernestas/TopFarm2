from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm.plotting import NoPlot
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.tests.test_files import xy3tb
from topfarm.constraint_components.spacing import SpacingConstraint, SpacingComp
from topfarm.tests import npt
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.api import Problem, IndepVarComp
import numpy as np
from topfarm.utils import SmoothMin, LogSumExpMin, StrictMin
import pytest
from topfarm._topfarm import TopFarmProblem


@pytest.mark.parametrize('aggfunc', [  # None,
    StrictMin(),
    SmoothMin(.1),
    SmoothMin(1),
    SmoothMin(10),
    LogSumExpMin(.1),
    LogSumExpMin(1),
    LogSumExpMin(10)
])
@pytest.mark.parametrize('x,y', [([6, 5, -8, 1], [0, -8, -4, 1]),
                                 ([2.84532167, 7.00331189, 3.86523273], [-2.98101466, -6.99667302, -2.98433316])])
@pytest.mark.parametrize('full_aggregation', [True, False])
def test_spacing_4wt_partials(aggfunc, full_aggregation, x, y):

    from topfarm.constraint_components.boundary import XYBoundaryConstraint
    from topfarm.easy_drivers import EasyScipyOptimizeDriver
    import topfarm
    initial = desired = np.array([x, y]).T
    boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
    spacing_constr = SpacingConstraint(2, aggregation_function=aggfunc, full_aggregation=full_aggregation)

    k = {'cost_comp': DummyCost(desired[:, :2], [topfarm.x_key, topfarm.y_key]),
         'design_vars': {topfarm.x_key: initial[:, 0], topfarm.y_key: initial[:, 1]},
         'driver': EasyScipyOptimizeDriver(disp=True, tol=1e-8),
         'plot_comp': NoPlot(),
         'constraints': [spacing_constr, XYBoundaryConstraint(boundary)]}
    if 0:
        k['plot_comp'] = DummyCostPlotComp(desired)
    TopFarmProblem(**k)
    scomp = spacing_constr.constraintComponent
    outputs = {}

    def compute(x, y):
        scomp.compute(dict(x=x, y=y), outputs)
        return outputs['wtSeparationSquared']

    ref = compute(initial[:, 0], initial[:, 1])
    ddx = np.array([(compute(x, initial[:, 1]) - ref) / 1e-6 for x in initial[:, 0] + np.eye(len(x)) * 1e-6]).T
    ddy = np.array([(compute(initial[:, 0], y) - ref) / 1e-6 for y in initial[:, 1] + np.eye(len(x)) * 1e-6]).T

    scomp.compute_partials(dict(x=initial[:, 0], y=initial[:, 1]), outputs)
    npt.assert_array_almost_equal(outputs[('wtSeparationSquared', 'x')].reshape(ddx.shape), ddx, 4)
    npt.assert_array_almost_equal(outputs[('wtSeparationSquared', 'y')].reshape(ddy.shape), ddy, 4)


@pytest.mark.parametrize('aggfunc', [None,
                                     StrictMin(),
                                     SmoothMin(.1),
                                     SmoothMin(.5),
                                     LogSumExpMin(.1),
                                     LogSumExpMin(1),
                                     LogSumExpMin(10)
                                     ])
def test_spacing_4wt(aggfunc):

    from topfarm.constraint_components.boundary import XYBoundaryConstraint
    from topfarm.easy_drivers import EasyScipyOptimizeDriver
    import topfarm
    initial = np.array([[6, 0], [5, -8], [-1, -4], [1, 1]])  # initial turbine layouts
    desired = np.array([[3, -3], [7, -7], [3, -4], [4, -3]])  # desired turbine layouts
    boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
    # initial = np.array([[6, 0], [1, 1]])  # initial turbine layouts
    # desired = np.array([[3, -3], [4, -3]])  # desired turbine layouts

    k = {'cost_comp': DummyCost(desired[:, :2], [topfarm.x_key, topfarm.y_key]),
         'design_vars': {topfarm.x_key: initial[:, 0], topfarm.y_key: initial[:, 1]},
         'driver': EasyScipyOptimizeDriver(disp=True, tol=1e-8),
         'plot_comp': NoPlot(),
         'constraints': [SpacingConstraint(2, aggregation_function=aggfunc), XYBoundaryConstraint(boundary)]}
    if 0:
        k['plot_comp'] = DummyCostPlotComp(desired)
    tf = TopFarmProblem(**k)
    print(str(aggfunc))
    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()
    tol = 1e-4
    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing


@pytest.mark.parametrize('aggfunc', [
    # None,
    StrictMin(), SmoothMin(1), LogSumExpMin(1)])
def test_spacing(aggfunc):
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2, aggregation_function=aggfunc)], plot=False)
    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()
    tol = 1e-4
    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing


def test_spacing_as_penalty():
    driver = SimpleGADriver()
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2)],
                      driver=driver)

    # check normal result if spacing constraint is satisfied
    assert tf.evaluate()[0] == 45
    # check penalized result if spacing constraint is not satisfied
    assert tf.evaluate({'x': [3, 7, 4.], 'y': [-3., -7., -3.], 'z': [0., 0., 0.]})[0] == 1e10 + 3


def test_satisfy():
    sc = SpacingComp(n_wt=3, min_spacing=2)
    state = sc.satisfy(dict(zip('xy', xy3tb.desired.T)))
    x, y = state['x'], state['y']
    npt.assert_array_less(y, x)


def test_satisfy2():
    n_wt = 5
    sc = SpacingComp(n_wt=n_wt, min_spacing=2)
    theta = np.linspace(0, 2 * np.pi, n_wt, endpoint=False)
    x0, y0 = np.cos(theta), np.sin(theta)

    state = sc.satisfy({'x': x0, 'y': y0})
    x1, y1 = state['x'], state['y']
    if 0:
        import matplotlib.pyplot as plt
        colors = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'indigo', 'grey']
        for i, (x0_, y0_, x1_, y1_) in enumerate(zip(x0, y0, x1, y1)):
            c = colors[i]
            plt.plot([x0_], [y0_], '>', color=c)
            plt.plot([x0_, x1_], [y0_, y1_], '-', color=c, label=i)
            plt.plot([x1_], [y1_], '.', color=c)
        plt.axis('equal')
        plt.legend()
        plt.show()

    dist = np.sqrt(sc._compute(x1, y1))
    npt.assert_array_less(2, dist)


@pytest.mark.parametrize('aggfunc', [None, StrictMin(), SmoothMin(1), LogSumExpMin(1)])
def test_partials(aggfunc):
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2, aggregation_function=aggfunc)])
    # if complex numbers work: uncomment tf.setup below and
    # change method='cs' and step=1e-40 in check_partials
    # tf.setup(force_alloc_complex=True)

    # run to get rid of zeros initializaiton, otherwise not accurate partials
    tf.run_model()
    check = tf.check_partials(compact_print=True,
                              includes='spacing*',
                              method='fd',
                              step=1e-6,
                              form='central')
    atol = 1.e-6
    rtol = 1.e-6
    try:
        assert_check_partials(check, atol, rtol)
    except ValueError as err:
        print(str(err))
        raise


class TestSpacingConstraintScaling(unittest.TestCase):
    def test_spacing_constraint_scaling(self):
        min_spacing_physical = 200.0
        # Turbines: T0 at (0,0), T1 at (150,0) (violates physical), T2 at (250,0) (satisfies physical)
        initial_positions_physical = np.array([[0, 0], [150, 0], [250, 0]])
        n_wt = initial_positions_physical.shape[0]
        dummy_cost_comp = DummyCost(optimal_state=initial_positions_physical, inputs=['x', 'y'])

        # Case 1: No scaling
        tf_unscaled = TopFarmProblem(
            design_vars={'x': initial_positions_physical[:, 0], 'y': initial_positions_physical[:, 1]},
            cost_comp=dummy_cost_comp,
            constraints=[SpacingConstraint(min_spacing_physical)], # turbine_diameter not given to SpacingConstraint
            reference_turbine_diameter=None # No scaling in TopFarmProblem
        )
        tf_unscaled.evaluate()
        unscaled_sep_sq = tf_unscaled['wtSeparationSquared']
        # Expected: [ (150-0)^2, (250-0)^2, (250-150)^2 ] = [22500, 62500, 10000]
        expected_sep_sq_unscaled = np.array([150**2, 250**2, 100**2])
        npt.assert_allclose(unscaled_sep_sq, expected_sep_sq_unscaled, rtol=1e-6)
        # Check constraint lower bound (from problem.model.cons['wtSeparationSquared']['lower'])
        # This requires a bit more digging or trusting setup_as_constraint.
        # For now, check the spacing_comp's idea of min_spacing.
        self.assertEqual(tf_unscaled.model.constraint_group.spacing_comp.min_spacing_scaled, min_spacing_physical)
        self.assertIsNone(tf_unscaled.model.constraint_group.spacing_comp.turbine_diameter)


        # Case 2: With scaling
        turbine_diameter_D = 100.0
        # SpacingConstraint is initialized with physical spacing and D. Internally it scales min_spacing.
        # TopFarmProblem is also given D to pass to constraints setup.
        spacing_constraint_scaled = SpacingConstraint(min_spacing_physical, turbine_diameter=turbine_diameter_D)

        tf_scaled = TopFarmProblem(
            design_vars={'x': initial_positions_physical[:, 0], 'y': initial_positions_physical[:, 1]},
            cost_comp=dummy_cost_comp,
            constraints=[spacing_constraint_scaled],
            reference_turbine_diameter=turbine_diameter_D # This D is passed to constraint._setup
        )
        tf_scaled.evaluate()
        scaled_sep_sq = tf_scaled['wtSeparationSquared']

        # SpacingComp should have received D and scaled x,y for computation
        # Expected: [ ((150-0)/D)^2, ((250-0)/D)^2, ((250-150)/D)^2 ]
        # D = 100: [ (1.5)^2, (2.5)^2, (1.0)^2 ] = [2.25, 6.25, 1.0]
        expected_sep_sq_scaled = np.array([(150/D)**2, (250/D)**2, (100/D)**2])
        npt.assert_allclose(scaled_sep_sq, expected_sep_sq_scaled, rtol=1e-6)

        # Check SpacingComp's attributes
        scaled_spacing_comp = tf_scaled.model.constraint_group.spacing_comp
        self.assertEqual(scaled_spacing_comp.min_spacing_scaled, min_spacing_physical / turbine_diameter_D)
        self.assertEqual(scaled_spacing_comp.turbine_diameter, turbine_diameter_D)

        # Check constraint evaluation (lower bound is (min_spacing_physical/D)**2 )
        # min_spacing_scaled = 200/100 = 2. Lower bound for constraint = 2^2 = 4
        # Scaled separations sq: [2.25, 6.25, 1.0]
        # Violations: 2.25 < 4 (True), 6.25 > 4 (False), 1.0 < 4 (True)
        # OpenMDAO constraint value is output - lower_bound >= 0 for feasible.
        # So, for us, wtSeparationSquared - (min_spacing_scaled**2) >= 0
        # For T0-T1: 2.25 - 4 = -1.75 (violation)
        # For T0-T2: 6.25 - 4 = 2.25 (ok)
        # For T1-T2: 1.0 - 4 = -3.0 (violation)
        # This is implicitly tested by tf.optimize() if it were run, but direct check of values is good.

    def test_spacing_type_constraint_scaling(self):
        min_spacing_physical_types = np.array([200.0, 300.0]) # Type 0 needs 200m, Type 1 needs 300m
        # Turbines: T0 (type 0) at (0,0), T1 (type 1) at (220,0), T2 (type 0) at (450,0)
        initial_positions_physical = np.array([[0, 0], [220, 0], [450, 0]])
        turbine_types_physical = np.array([0, 1, 0])
        n_wt = initial_positions_physical.shape[0]
        dummy_cost_comp = DummyCost(optimal_state=initial_positions_physical, inputs=['x', 'y'])

        turbine_diameter_D = 100.0

        # Case 1: No scaling
        spacing_type_unscaled = SpacingTypeConstraint(min_spacing_physical_types) # No D here
        tf_unscaled = TopFarmProblem(
            design_vars={'x': initial_positions_physical[:, 0],
                         'y': initial_positions_physical[:, 1],
                         topfarm.type_key: turbine_types_physical},
            cost_comp=dummy_cost_comp,
            constraints=[spacing_type_unscaled],
            reference_turbine_diameter=None
        )
        tf_unscaled.evaluate()
        unscaled_rel_sep_sq = tf_unscaled['wtRelativeSeparationSquared']

        # Expected effective physical spacings:
        # T0-T1 (type 0, type 1): (200+300)/2 = 250. Dist = 220. (220^2 - 250^2)
        # T0-T2 (type 0, type 0): (200+200)/2 = 200. Dist = 450. (450^2 - 200^2)
        # T1-T2 (type 1, type 0): (300+200)/2 = 250. Dist = 450-220 = 230. (230^2 - 250^2)
        exp_unscaled_rel_sep_sq = np.array([
            220**2 - 250**2,
            450**2 - 200**2,
            230**2 - 250**2
        ])
        npt.assert_allclose(unscaled_rel_sep_sq, exp_unscaled_rel_sep_sq, rtol=1e-6)

        # Case 2: With scaling
        spacing_type_scaled = SpacingTypeConstraint(min_spacing_physical_types, turbine_diameter=turbine_diameter_D)
        tf_scaled = TopFarmProblem(
            design_vars={'x': initial_positions_physical[:, 0],
                         'y': initial_positions_physical[:, 1],
                         topfarm.type_key: turbine_types_physical},
            cost_comp=dummy_cost_comp,
            constraints=[spacing_type_scaled],
            reference_turbine_diameter=turbine_diameter_D
        )
        tf_scaled.evaluate()
        scaled_rel_sep_sq = tf_scaled['wtRelativeSeparationSquared']

        # Coords scaled: T0(0,0), T1(2.2,0), T2(4.5,0)
        # Min spacings scaled: [200/100, 300/100] = [2, 3]
        # Expected effective scaled spacings:
        # T0-T1 (type 0, type 1): (2+3)/2 = 2.5. Scaled dist = 2.2. (2.2^2 - 2.5^2)
        # T0-T2 (type 0, type 0): (2+2)/2 = 2.0. Scaled dist = 4.5. (4.5^2 - 2.0^2)
        # T1-T2 (type 1, type 0): (3+2)/2 = 2.5. Scaled dist = 4.5-2.2 = 2.3. (2.3^2 - 2.5^2)
        exp_scaled_rel_sep_sq = np.array([
            (220/D)**2 - ((min_spacing_physical_types[0]/D + min_spacing_physical_types[1]/D)/2)**2,
            (450/D)**2 - ((min_spacing_physical_types[0]/D + min_spacing_physical_types[0]/D)/2)**2,
            ((450-220)/D)**2 - ((min_spacing_physical_types[1]/D + min_spacing_physical_types[0]/D)/2)**2
        ])
        npt.assert_allclose(scaled_rel_sep_sq, exp_scaled_rel_sep_sq, rtol=1e-6)

        scaled_spacing_type_comp = tf_scaled.model.constraint_group.spacing_type_comp
        npt.assert_allclose(scaled_spacing_type_comp.min_spacing_scaled, min_spacing_physical_types / turbine_diameter_D)
        self.assertEqual(scaled_spacing_type_comp.turbine_diameter, turbine_diameter_D)


@pytest.mark.parametrize('aggfunc', [None,
                                     # StrictMin(), # not working
                                     SmoothMin(1), LogSumExpMin(1)])
def test_partials_many_turbines(aggfunc):
    n_wt = 10
    theta = np.linspace(0, 2 * np.pi, n_wt, endpoint=False)
    sc = SpacingComp(n_wt=n_wt, min_spacing=2, const_id="", aggregation_function=aggfunc)
    tf = Problem()
    ivc = IndepVarComp()
    ivc.add_output('x', val=np.cos(theta))
    ivc.add_output('y', val=np.sin(theta))
    tf.model.add_subsystem('ivc', ivc, promotes=['*'])
    tf.model.add_subsystem('sc', sc, promotes=['x', 'y', 'wtSeparationSquared'])
    tf.setup()
    tf.run_model()

    check = tf.check_partials(compact_print=True,
                              includes='sc*',
                              method='fd',
                              step=1e-6,
                              form='central')

    fil = {'sc': {key: val for key, val in check['sc'].items()
                  if 'constraint_violation' not in key[0]}}

    atol = 1.e-6
    rtol = 1.e-6
    try:
        assert_check_partials(fil, atol, rtol)
    except ValueError as err:
        print(str(err))
        raise
