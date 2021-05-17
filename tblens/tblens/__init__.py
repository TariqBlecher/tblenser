"""
tblenser package

This package is designed for simulating the gravitational lensing of neutral hydrogen in galaxies. It can handle arbitrary lenses and is primarily focused on forward modelling. It can simulate HI disks and it can marginalise over HI disk hyperparameters.


The structure of the code is as follows:

Simulations are configured and run through a high-level driver script (see drivers folder);
The driver script imports from the tblens module which should be installed (e.g. with pip);
Deflection maps are either input to the pipeline or can be created by giving a mass model to the tblens.lens_creator.write_defmap function;
tblens.HIDisk.HIDisk is a class which creates an HI disk;
tblens.grid_creators.PositionGrid is a class for handling the basics of coordinate system generation;
tblens.map_utils_core.DeflectionMap subclasses PositionGrid and handles the ray-tracing.
"""