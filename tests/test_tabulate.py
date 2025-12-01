"""
Test cases for tabulation

We use PREM for ease, and check we can make "tabulated"
results (needed to build TauPy models or for input to
mineos).
"""
import numpy as np
import numpy.testing as npt

from premlike import PREM, tabulate_model

def test_tabulate_inwards():
    """
    Starting from the surface and going inwards. Needed
    for TauPy.
    """
    # First 7 expected entries for 20 km spacing. Note that
    # some of these sit on discontinuities.
    expect_depth = np.array([ 0.0, 15.0, 15.0, 24.4, 24.4, 44.4, 64.4])
    expect_radius = np.array([6371.0, 6356.0, 6356.0, 6346.6, 6346.6, 6326.6, 6306.6])
    expect_density = np.array([2600.0, 2600.0, 2900.0, 2900.0, 3380.74820907, 3378.57460995, 3376.40101083])
    expect_vp = np.array([5.8, 5.8, 6.8, 6.8, 8.11061727, 8.09825437, 8.08589148])
    expect_vs = np.array([3.2, 3.2, 3.9, 3.9, 4.49100712, 4.48363591, 4.47626469])
    expect_qkappa = np.array([57823.0, 57823.0, 57823.0, 57823.0, 57823.0, 57823.0, 57823.0])
    expect_qshear = np.array([600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0])

    table = tabulate_model(PREM, 20.0, outwards=False)
    npt.assert_allclose(table.depth[0:7], expect_depth)
    npt.assert_allclose(table.radius[0:7], expect_radius)
    npt.assert_allclose(table.density[0:7], expect_density)
    npt.assert_allclose(table.vp[0:7], expect_vp)
    npt.assert_allclose(table.vs[0:7], expect_vs)
    npt.assert_allclose(table.qkappa[0:7], expect_qkappa)
    npt.assert_allclose(table.qshear[0:7], expect_qshear)


def test_tabulate_outwards():
    """
    Ending near the the surface and going outwards. Needed
    for Mineos.
    """
    # First 7 expected entries for 20 km spacing. Note that
    # some of these sit on discontinuities.
    expect_depth = np.array([100.0, 80.0, 60.0, 40.0, 24.4,  24.4,  15.0])
    expect_radius = np.array([6271.0, 6291.0, 6311.0, 6331.0, 6346.6, 6346.6, 6356.0])
    expect_density = np.array([3372.53200439, 3374.70560352, 3376.87920264, 3379.05280176, 3380.74820907, 2900.0, 2900.0])
    expect_vp = np.array([8.06388553, 8.07624842, 8.08861132, 8.10097421, 8.11061727, 6.8, 6.8])
    expect_vs = np.array([4.46314393, 4.47051515, 4.47788636, 4.48525757, 4.49100712, 3.9, 3.9])
    expect_qkappa = np.array([57823.0, 57823.0, 57823.0, 57823.0, 57823.0, 57823.0, 57823.0])
    expect_qshear = np.array([80.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0])

    table = tabulate_model(PREM, 20.0, outwards=True)
    npt.assert_allclose(table.depth[-8:-1], expect_depth)
    npt.assert_allclose(table.radius[-8:-1], expect_radius)
    npt.assert_allclose(table.density[-8:-1], expect_density)
    npt.assert_allclose(table.vp[-8:-1], expect_vp)
    npt.assert_allclose(table.vs[-8:-1], expect_vs)
    npt.assert_allclose(table.qkappa[-8:-1], expect_qkappa)
    npt.assert_allclose(table.qshear[-8:-1], expect_qshear)

def test_tabulate_default():
    # Check default is outwards
    table = tabulate_model(PREM, 20.0, outwards=True)
    table2 = tabulate_model(PREM, 20.0)
    npt.assert_allclose(table.depth, table2.depth)
    npt.assert_allclose(table.radius, table2.radius)
