#!/usr/bin/env python
# coding=utf8
"""
Support for PREM-like 1D Earth models

"""

import numpy as np

from .const import R_EARTH
from .peice_poly import PeicewisePolynomial as PP
from .physics import (
    calculate_bulk_modulus,
    calculate_density,
    calculate_grav_potential,
    calculate_gravity,
    calculate_mass,
    calculate_moi,
    calculate_pressure,
    calculate_qkappa,
    calculate_qshear,
    calculate_shear_modulus,
    calculate_vp,
    calculate_vs,
)


class OneDModel:
    def __init__(
        self,
        breakpoints,
        density_params,
        vp_params,
        vs_params,
        q_mu_params,
        q_kappa_params,
        r_earth=R_EARTH,
    ):
        self.r_earth = r_earth

        self.density_poly = PP(density_params, breakpoints)
        self.vp_poly = PP(vp_params, breakpoints)
        self.vs_poly = PP(vs_params, breakpoints)
        self.qk_poly = PP(q_kappa_params, breakpoints)
        self.qm_poly = PP(q_mu_params, breakpoints)

    def density(self, r, break_down=False):
        """
        Evaluate density in kg/m**3 at radii r (in km)
        """
        return calculate_density(self.density_poly, r, break_down=break_down)

    def vs(self, r, t=1, break_down=False) -> float | np.ndarray:
        """
        Evaluate s-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        return calculate_vs(
            self.vs_poly, r, t=t, qm_poly=self.qm_poly, break_down=break_down
        )

    def vp(self, r, t=1, break_down=False):
        """
        Evaluate p-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        return calculate_vp(
            self.vp_poly,
            r,
            t=t,
            qk_poly=self.qk_poly,
            qm_poly=self.qm_poly,
            vs_poly=self.vs_poly,
            break_down=break_down,
        )

    def qkappa(self, r, break_down=False):
        return calculate_qkappa(self.qk_poly, r, break_down=break_down)

    def qshear(self, r, break_down=False):
        return calculate_qshear(self.qm_poly, r, break_down=break_down)

    def bulk_modulus(self, r):
        """
        Evaluate bulk modulus (in GPa) at radius r (in km)
        """
        return calculate_bulk_modulus(self.vp_poly, self.vs_poly, self.density_poly, r)

    def shear_modulus(self, r):
        """
        Evaluate shear modulus (in GPa) at radius r (in km)
        """
        return calculate_shear_modulus(self.vs_poly, self.density_poly, r)

    def mass(self, r, r_inner=0.0):
        """
        Evaluate mass inside radius r (in km)
        """
        return calculate_mass(self.density_poly, r, r_inner=r_inner)

    def moment_of_inertia(self, r, r_inner=0.0):
        """
        Evaluate moment of inertia inside radius r (in km)

        Return a tuple of moment of inertia (in kg m^2) and
        the moment of inertia factor (I/MR**2, dimensionless)
        which is 0.4 for a uniform density body, and decreases
        as the core becomes more dense than the crust/mantle.

        """
        return calculate_moi(self.density_poly, r, r_inner=r_inner)

    def gravity(self, r):
        return calculate_gravity(self.density_poly, r)

    def grav_potential(self, r):
        """
        Evaluate the gravitational potential at radius r in J/kg
        """
        return calculate_grav_potential(self.density_poly, r)

    def pressure(self, r):
        """
        Evaluate pressure (in GPa) at radius r (in km)
        """
        return calculate_pressure(self.density_poly, r)

    def tabulate_model_inwards(self, min_step):
        """
        Return a record array representing the model handling discontiuities

        This method creates a numpy record array with the model evaulated
        at all depths with a minimum spacing of min_step km. All breakpoints
        are also included in the output. If the densioty is discontinuoius,
        the depth is represented twice, first with the value above the
        discontiuity, then with the value below it. This representation can
        be used to construct travel time curves (for examople).

        The record array contains fields:

            depth (in km)
            radius (in km)
            density (in kg/m^3)
            qkappa (dimensionless quality factor)
            qshear (dimensionless quality factor)

        and is ordered such that element 0 is at the surface and the last
        element (element -1) is at the center of the planet.
        """
        # Keep the data as we get it
        radii = np.array([])
        depths = np.array([])
        densities = np.array([])
        vps = np.array([])
        vss = np.array([])
        qks = np.array([])
        qms = np.array([])

        nbps = len(self.density_poly.breakpoints) - 1
        for i in range(nbps):
            j = nbps - i
            k = j - 1
            rs = np.arange(
                self.density_poly.breakpoints[j],
                self.density_poly.breakpoints[k],
                -min_step,
            )
            ds = self.r_earth - rs
            dens = self.density(rs, break_down=True)  # As we go inwards
            vp = self.vp(rs, break_down=True)  # As we go inwards
            vs = self.vs(rs, break_down=True)  # As we go inwards
            qk = self.qkappa(rs, break_down=True)  # As we go inwards
            qm = self.qshear(rs, break_down=True)  # As we go inwards
            radii = np.append(radii, rs)
            depths = np.append(depths, ds)
            densities = np.append(densities, dens)
            vps = np.append(vps, vp)
            vss = np.append(vss, vs)
            qks = np.append(qks, qk)
            qms = np.append(qms, qm)

            # Look at the breakpoint. If it is discontinous in
            # value put add it here (i.e. so we have above followed
            # by below for the next step). Othersie we can skip it
            # (and it gets adder in the next iteration). But we need
            # to hadle k = 0 carefully (always stick in the origin)
            if k == 0:
                # Add the value at r=0
                rs = self.density_poly.breakpoints[k]
                ds = self.r_earth - rs
                dens = self.density(rs)
                vp = self.vp(rs)
                vs = self.vs(rs)
                qk = self.qkappa(rs)
                qm = self.qshear(rs)
                radii = np.append(radii, rs)
                depths = np.append(depths, ds)
                densities = np.append(densities, dens)
                vps = np.append(vps, vp)
                vss = np.append(vss, vs)
                qks = np.append(qks, qk)
                qms = np.append(qms, qm)
            elif self.density(self.density_poly.breakpoints[k]) != self.density(
                self.density_poly.breakpoints[k], break_down=True
            ):
                # Add the value above the inner boundary of this layer
                rs = self.density_poly.breakpoints[k]
                ds = self.r_earth - rs
                dens = self.density(rs)
                vp = self.vp(rs)
                vs = self.vs(rs)
                qk = self.qkappa(rs)
                qm = self.qshear(rs)
                radii = np.append(radii, rs)
                depths = np.append(depths, ds)
                densities = np.append(densities, dens)
                vps = np.append(vps, vp)
                vss = np.append(vss, vs)
                qks = np.append(qks, qk)
                qms = np.append(qms, qm)

        result = np.rec.fromarrays(
            [depths, radii, densities, vps, vss, qks, qms],
            names="depth, radius, density, vp, vs, qkappa, qshear",
        )
        return result

    def tabulate_model_outwards(self, min_step):
        """
        Return a record array representing the model handling discontiuities

        This method creates a numpy record array with the model evaulated
        at all depths with a minimum spacing of min_step km. All breakpoints
        are also included in the output. If the densioty is discontinuoius,
        the depth is represented twice, first with the value above the
        discontiuity, then with the value below it. This representation can
        be used to construct travel time curves (for examople).

        The record array contains fields:

            depth (in km)
            radius (in km)
            density (in kg/m^3)
            qkappa (dimensionless quality factor)
            qshear (dimensionless quality factor)

        and is ordered such that element 0 is at the center of the planet
        and the last element (element -1) is at the surface.
        """
        # Keep the data as we get it
        radii = np.array([])
        depths = np.array([])
        densities = np.array([])
        vps = np.array([])
        vss = np.array([])
        qks = np.array([])
        qms = np.array([])

        nbps = len(self.density_poly.breakpoints) - 1
        for i in range(nbps):
            j = i
            k = j + 1
            rs = np.arange(
                self.density_poly.breakpoints[j],
                self.density_poly.breakpoints[k],
                min_step,
            )
            ds = self.r_earth - rs
            dens = self.density(rs)
            vp = self.vp(rs)
            vs = self.vs(rs)
            qk = self.qkappa(rs)
            qm = self.qshear(rs)
            radii = np.append(radii, rs)
            depths = np.append(depths, ds)
            densities = np.append(densities, dens)
            vps = np.append(vps, vp)
            vss = np.append(vss, vs)
            qks = np.append(qks, qk)
            qms = np.append(qms, qm)

            # Look at the breakpoint. If it is discontinous in
            # value put add it here (i.e. so we have above followed
            # by below for the next step). Othersie we can skip it
            # (and it gets adder in the next iteration). But we need
            # to hadle k = 0 carefully (always stick in the origin)
            if k == nbps + 1:
                # Add the value surface
                rs = self.density_poly.breakpoints[k]
                ds = self.r_earth - rs
                dens = self.density(rs)
                vp = self.vp(rs)
                vs = self.vs(rs)
                qk = self.qkappa(rs)
                qm = self.qshear(rs)
                radii = np.append(radii, rs)
                depths = np.append(depths, ds)
                densities = np.append(densities, dens)
                vps = np.append(vps, vp)
                vss = np.append(vss, vs)
                qks = np.append(qks, qk)
                qms = np.append(qms, qm)
            elif self.density(self.density_poly.breakpoints[k]) != self.density(
                self.density_poly.breakpoints[k], break_down=True
            ):
                # Add the value above the inner boundary of this layer
                rs = self.density_poly.breakpoints[k]
                ds = self.r_earth - rs
                dens = self.density(rs, break_down=True)
                vp = self.vp(rs, break_down=True)
                vs = self.vs(rs, break_down=True)
                qk = self.qkappa(rs, break_down=True)
                qm = self.qshear(rs, break_down=True)
                radii = np.append(radii, rs)
                depths = np.append(depths, ds)
                densities = np.append(densities, dens)
                vps = np.append(vps, vp)
                vss = np.append(vss, vs)
                qks = np.append(qks, qk)
                qms = np.append(qms, qm)

        result = np.rec.fromarrays(
            [depths, radii, densities, vps, vss, qks, qms],
            names="depth, radius, density, vp, vs, qkappa, qshear",
        )
        return result
