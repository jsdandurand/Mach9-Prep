# Mach9 interview prep

NumPy practice for a 3D / spatial interview, aimed at lidar-to-CAD geometry.

Printable one-page formula sheet (US Letter, landscape): [cheatsheet/spatial_formulas.pdf](cheatsheet/spatial_formulas.pdf).

Start with [study_guide.md](study_guide.md). It covers the linear-algebra core and the survey pieces a generic 3D outline usually skips: Z-up coordinates, station and offset, cross-sections, ground grids, local shape features, and TIN queries.

Then work the notebooks in order. Each problem has an attempt cell, then a solution cell that replaces your function and checks it. Collapse or scroll past the solution until you have tried. You need NumPy.

1. [Vectors and rigid transforms](notebooks/01_vectors_and_rigid_transforms.ipynb)
2. [Point clouds](notebooks/02_point_clouds.ipynb)
3. [Planes, segments, boxes, rays](notebooks/03_primitives_and_intersections.ipynb)
4. [Alignment and cameras](notebooks/04_alignment_and_cameras.ipynb)
5. [Linework and cross-sections](notebooks/05_linework_and_cross_sections.ipynb)
6. [Terrain, TINs, and local shape](notebooks/06_terrain_tins_and_shape.ipynb)

If time is short: all of notebook 01, nearest neighbors, voxel downsample, plane fit, point-to-segment, station and offset, cross-section, min-Z grid, shape features, TIN interpolation, and Kabsch.
