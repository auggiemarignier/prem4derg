# premlike

This is a python module and some example notebooks to work with
PREM, a famous 1D model of the Earth's interior published by Dziewonski and Anderson in 1981.
The module was initially created to allow the Deep Earth Research Group at Leeds the University
of Leeds to run a summer project to look at the structure of the core but the code has subsequently
been used for a number of other research projects.

## Features and limitations

Premlike allows the creation and evaluation of models that are "like" PREM in terms of their
parameterization. That is, the P- and S-wave velocity, density and attenuation 
are described by piecewise polynomials. The module allows the calculation of the corresponding
acceleration due to gravity, pressure, frequency dependent velocities and elastic moduli as
a function of radius. These calculations are analytic (i.e. there is no numerical integration
in the calculation of pressure). The module can easily be used to generate input for Obspy's
TauPy module for the calculation of travel times, and for Mineos for the calculation of e.g.
normal mode center frequencies. 

Currently, Premlike does not support radial anisotropy although adding this is planned
(however, note that TauPy does not currently support such models either). In addition,
the creation of complete documentation and a more robust set of tests is ongoing.

## Installation and use

This module relies on python (version 3) and Numpy. The software can 
be installed with `pip install premlike` 

Examples are distributed as Jupyter notebooks, which need Jupyter
and Matplotlib to run.
Calculation of travel time curves (in the third example notebook) needs Obspy.

Experienced users of git who want to contribute to the code may want to fork this
repository in github and clone from their fork.

### Mineos

In order to compute normal mode frequencies, we use [mineos](https://geodynamics.org/cig/software/mineos/). I 
installed this as follows (working outside the prem4derg directory):

    wget http://geoweb.cse.ucdavis.edu/cig/software/mineos/mineos-1.0.2.tgz
    mkdir cig
    tar -xzvf mineos-1.0.2.tgz
    cd mineos-1.0.2
    ./configure --prefix=/Users/andreww/Code/cig

Edit the Makefile to add: `-Wno-error -Wno-return-type` to the `CFLAGS` because the C contains errors.
Add `-fallow-argument-mismatch` to FFLAGS because the Fortran contains errors.

    make 
    make install

## Examples of use

Example Jupyter notebooks can be accessed and explored by running `jupyter notebook`. Four examples are
currently provided:

* A [density example](./PREM_density_example.ipynb), showing how a model can be defined and used to calculate mass, moment of inertia, gravity and pressure.
* A [velocity example](./PREM_velocity_example.ipynb), showing how a model can be defined and used to calculate seismic velocities as a function of depth and period.
* A [travel time example](./PREM_travel_times_example.ipynb), showing how an obspy taupy model can be created and used to compute travel time curves.
* A [normal modes example](./PREM_normal_modes_example.ipynb), showing how Mineos can be used to compute normal mode frequencies.

These examples are only starting point. For example, the code could be used to fit new models. 

## Development and support

Premlike is software under development, bugs and rough edges may abound. Users who are interested in 
changing the code are encouraged to [create a fork on GitHub and submit 
changes via pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests).
Problems can be reported via [issues](https://github.com/andreww/premlike/issues), which also list areas
where further development may be useful.
  

## References

Dziewonski, A. M. and Anderson, D. L. (1981) Preliminary reference Earth model *Physics of the Earth and 
Planetary Interiors* vol.25 pp.297—356. 
