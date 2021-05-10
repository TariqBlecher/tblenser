## `tblenser`

`tblenser` simulates the gravitational lensing of extra-galactic neutral hydrogen (HI) by foreground galaxies or clusters. 

It can handle arbitrary lenses and is primarily focused on forward modelling, i.e. given a deflection map and a source model, it calculates the lensed image. 

It can simulate HI disks using the [Obreschkow et al. (2009)](https://arxiv.org/abs/0901.2526) model, and marginalise over HI disk parameters through random sampling.

This software has been used in [Blecher et al. (2019)](https://doi.org/10.1093/mnras/stz224), with more results to come.

---

The structure of the code is as follows:

- Simulations are configured and run through a high-level driver script (see drivers folder);
- The driver script imports from the `tblens` module which should be installed (e.g. with pip);
- Deflection maps are either input to the pipeline or can be created by giving a mass model to the `tblens.lens_creator.write_defmap` function;
- `tblens.HiDisk.HiDisk` is a class which creates an HI disk;
- `tblens.grid_creators.PositionGrid` is a class for handling the basics of coordinate system generation;
- `tblens.map_utils_core.DeflectionMap` subclasses `PositionGrid` and handles the ray-tracing.
