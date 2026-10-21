# Mach9 interview prep

NumPy practice for a 3D / spatial interview, aimed at lidar-to-CAD geometry.

Start with [study_guide.md](study_guide.md). It covers the linear-algebra core and the survey pieces a generic 3D outline usually skips: Z-up coordinates, station and offset, cross-sections, ground grids, local shape features, and TIN queries.

Then work the notebooks in order. Each problem has an attempt cell, then a solution cell that replaces your function and checks it. Collapse or scroll past the solution until you have tried. You need NumPy.

1. [Vectors and rigid transforms](notebooks/01_vectors_and_rigid_transforms.ipynb)
2. [Point clouds](notebooks/02_point_clouds.ipynb)
3. [Planes, segments, boxes, rays](notebooks/03_primitives_and_intersections.ipynb)
4. [Alignment and cameras](notebooks/04_alignment_and_cameras.ipynb)
5. [Linework and cross-sections](notebooks/05_linework_and_cross_sections.ipynb)
6. [Terrain, TINs, and local shape](notebooks/06_terrain_tins_and_shape.ipynb)

If time is short: all of notebook 01, nearest neighbors, voxel downsample, plane fit, point-to-segment, station and offset, cross-section, min-Z grid, shape features, TIN interpolation, and Kabsch.

The onsite is a different mix: a whiteboard to design an ML system, a round on using AI to solve a problem, behavioral interviews, and coding in Python that may be PyTorch. Read [onsite_prep.md](onsite_prep.md) for the design round. The formula sheet stays the geometry reference; this doc does not replace it.

The PyTorch notebooks are the same problems as above, then a short training loop. You need PyTorch. Geometry in these notebooks is float64.

1. [Vectors and rigid transforms](notebooks/pytorch/01_vectors_and_rigid_transforms.ipynb)
2. [Point clouds](notebooks/pytorch/02_point_clouds.ipynb)
3. [Planes, segments, boxes, rays](notebooks/pytorch/03_primitives_and_intersections.ipynb)
4. [Alignment and cameras](notebooks/pytorch/04_alignment_and_cameras.ipynb)
5. [Linework and cross-sections](notebooks/pytorch/05_linework_and_cross_sections.ipynb)
6. [Terrain, TINs, and local shape](notebooks/pytorch/06_terrain_tins_and_shape.ipynb)
7. [Training loop](notebooks/pytorch/07_training_loop.ipynb)

If time is short on the PyTorch set: notebook 01, nearest neighbors, voxel downsample, plane fit, station and offset, shape features, and the training loop.
