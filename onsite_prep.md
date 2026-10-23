# Onsite prep

The recruiter described four rounds: a whiteboard to design an ML system, a round on using AI to solve a problem, behavioral interviews, and coding in Python that may be PyTorch.

This note teaches the system-design material from the objects outward. The formula sheet and the geometry notebooks are the algebra. Read this when those documents use a word you have not had defined: TIN, breakline, station, curb face, PointNet.

Read it in order once. The whiteboard script is at the end, after the words mean something.

## What Mach9 is asking you to design

A survey crew drives a road, or walks it, or flies it, with a lidar scanner. The scanner returns a cloud of points: positions in the world where the laser bounced. That cloud is not a drawing. A civil project needs lines and a surface: where the curb is, where the pavement ends, where a pole stands, and the ground elevation everywhere else.

A person produces that drawing today by clicking vertices while looking at the cloud. The product proposes the lines. The person accepts them, moves them, or deletes them. The thing you are graded on is the line they accept, in centimeters, not a per-point accuracy number.

A recommender design (users, items, CTR, a two-tower model) is a different product. The input here is an unordered set of points. The output is a polyline or a surface. The split is by project or by city, because points from the same curb are not independent. The feedback is a surveyor editing a line.

## What a point looks like

One return is one row. A typical row carries:

| Field | What it is |
| --- | --- |
| `x, y, z` | Position in meters. In this domain Z is up. Plan view is the XY plane. |
| `intensity` | How strong the return was. Unitless, and not comparable across scanners or wet versus dry pavement until you normalize it. Lane paint and retroreflective signs are bright. Asphalt is dark. |
| `return number`, `number of returns` | One laser pulse can hit more than one thing. A pulse that clips a tree and then hits the ground is two rows, sharing a direction. "Last return" is the usual candidate for ground. |
| `gps time` | When the pulse fired. This joins the point to the vehicle's pose at that instant. |
| `classification` | An optional integer already on the point. The LAS standard has codes such as 2 = ground and 6 = building. Curb is not one of those codes. A class label and a CAD line are different outputs. |

The industry file is LAS, or LAZ when compressed. You will not parse the bytes in an interview. You need the shape of a row and the fact that the file is a list, not an image.

There is no pixel grid. Row `i` and row `i+1` in the file are not neighbors. They are two pulses that happened to be stored next to each other, and they may be meters apart. Any algorithm that treats the array order as spatial order is wrong. Neighbors are whoever is nearby in XYZ, which you find with a spatial index.

Coordinates are often in a projected survey system (a state plane, UTM). An easting can be `500000.123`. A network that sees raw eastings spends its first layer learning the local origin. Subtract the tile origin before the model, and add it back when you write the line. Keep Z absolute. Elevations are the product.

A point is on the order of 20–40 bytes uncompressed. A billion points is tens of gigabytes before you store normals, neighbors, or activations. A highway project can reach that size. Nothing in the design loads the project onto a GPU.

## What a scan looks like

Three collection modes, same downstream drawing:

- **Mobile.** A vehicle drives the road. A laser spins or sweeps. Density is high next to the vehicle (a point every centimeter or two on the pavement) and falls off with range. The far curb is sparse and often hidden behind parked cars. The vehicle's path is the natural "along the road" axis.
- **Terrestrial.** A scanner on a tripod. It sees a full dome from one station, then the crew moves it. Each station is dense nearby and blind behind the first wall. Stations have to be aligned to each other or one curb becomes two curbs a few centimeters apart.
- **Airborne or drone.** Looking down. Great for roofs and open ground, poor for a vertical curb face, because a vertical face presents almost no area to a nadir beam. A curb may show up only as a height step between pavement and sidewalk.

The trajectory is the other half of a mobile scan. It is a sequence of poses: time, position, and rotation of the scanner. Pose comes from GNSS plus an inertial unit, and it is not perfect. Under trees or between tall buildings the trajectory wanders. Two passes over the same curb can disagree by a few centimeters. Your accuracy target has to live with that. A metric that demands 1 mm is tighter than the data.

Going from scanner coordinates to world coordinates is a rigid transform, `p_world = R p_scanner + t`, one pose per pulse. That is the same `apply_rigid` as the notebooks. If two scans of a static scene disagree, you solve for a correction: Kabsch when you already know which points match, one ICP step (nearest neighbor, then Kabsch) when you do not. ICP is only trustworthy when the remaining motion is small relative to point spacing. Say that limit if alignment comes up. Do not build the interview answer out of it.

Occlusion is a hole, not a negative label. The laser never reached the curb behind a truck, so the absence of points there is not evidence of "no curb." A model that treats missing data as the background class will erase curbs every time a car is parked. The honest output in a hole is "no prediction," and a person fills it.

## The drawing, piece by piece

The cloud is the input. These are the objects the model is trying to help produce.

### Polyline

A polyline is an ordered list of vertices with straight segments between them. A curb in CAD is a polyline, a few vertices on a straight run and a vertex wherever it bends. It is not a mask, and it is not a copy of every lidar point that touched concrete.

```text
vertices: (M, 3) or (M, 2)
segment i connects vertex i to vertex i+1
```

M is dozens or hundreds for a block, not millions. That compression is the point of the drawing.

### Alignment, station, and offset

An alignment is the polyline you measure along. For a mobile scan the smoothed trajectory is a usable first alignment. For a delivered survey it is the road centerline.

**Station** is horizontal distance along that polyline, in meters, from the start. It ignores Z. A hill does not make the station longer. Surveyors call this chainage. Station 15.0 means "15 m along the road," not "the point at XY = (15, something)" unless the road happens to run that way.

**Offset** is the signed sideways distance from the alignment to a point, in the horizontal plane. Positive is to the right of increasing station. If the tangent of the alignment in XY is `(tx, ty)`, the right-pointing unit normal is `(ty, -tx)`. Eastbound tangent `(1, 0)` has right normal `(0, -1)`, which is south. A point north of an eastbound road has negative offset.

Work this by hand until it is boring. Alignment `(0,0) → (10,0) → (10,10)`.

| Point | Closest point on the alignment | Station | Offset | Why |
| --- | --- | --- | --- | --- |
| `(5, 2)` | `(5, 0)` | 5 | −2 | 2 m north of an eastbound segment, so to the left |
| `(12, 5)` | `(10, 5)` | 15 | +2 | 10 m along the first leg, 5 m up the second; 2 m east of a northbound segment, so to the right |
| `(8, 5)` | `(10, 5)` | 15 | −2 | Same closest point, on the west side |
| `(10, 10)` | `(10, 10)` | 20 | 0 | The last vertex |
| `(5, 0)` | `(5, 0)` | 5 | 0 | On the line |

Past either end, the closest point is the endpoint, so station clamps. The offset you report there is only the sideways part of the vector to that endpoint, not the straight-line distance. At the corner `(10, 0)`, two segments are equally close and their tangents disagree, so the sign of a point sitting on the corner depends on which segment you let win. Pick a rule (smallest segment index is the one in the notebook) and say it.

Station and offset turn a 2D cloud into coordinates a road already has: "12 m along, 4.2 m right of center." A curb is then nearly a constant-offset curve. That is a much easier thing to trace than an arbitrary scribble in XY.

### Profile versus cross-section

These are two different plots. Mixing the names is a tell.

- A **profile** is elevation against station. It is the side view of the road as you drive it: hills, grades, vertical curves.
- A **cross-section** is a slice perpendicular to the road at one station. Horizontal axis is offset, vertical axis is elevation. This is the picture in which a curb is obviously a step.

To cut one section at station `s`:

1. Find the segment that contains `s`, using cumulative horizontal length. At an interior vertex, use the outgoing segment's tangent. At the last vertex, use the incoming one. State that rule, because the tangent jumps at a kink.
2. The section origin is the point on the alignment at station `s`, with Z linearly interpolated between the segment's endpoints. A 5% grade shows up in that Z.
3. Keep cloud points close to the plane: `|along|` small (a thickness of maybe 0.25–0.5 m) and `|offset|` inside the width you care about (maybe 15 m either side of center).
4. Deliver those points as `(offset, z)`, sorted by offset.

Hand check from the notebook. Alignment vertices `(0,0,10)`, `(10,0,20)`, `(10,10,20)`. At station 5 the origin is `(5, 0, 15)`: halfway in XY and halfway in Z. The point `(5, 2, 100)` has offset −2 and elevation 100. The point `(5.2, -1, 101)` has offset +1 and elevation 101. Elevation in this exercise is absolute Z, not height above the road.

```text
cross-section, looking along the road

z
|            sidewalk
|           _________
|          |
|          |  curb face, ~15 cm
|     _____|
|    pavement, sloping a little toward the gutter
+------------------------------- offset
     left of center        right
```

### A curb, physically

A curb is the concrete boundary between the traveled way and the sidewalk or the shoulder. The parts have names because the drawing often contains more than one line:

- **Flowline** (or gutter line): the low edge, where water runs, at the bottom of the curb face.
- **Face**: the near-vertical concrete, often about 15 cm tall (6 in). In the cloud this is a short wall.
- **Top of curb**: the high edge.
- **Back of curb**: the edge against the sidewalk, if the drawing calls it out separately.

Ask which line they want before you design the model. "Curb" in a meeting can mean the flowline, the top, or both. Both are breaklines. They are a few centimeters apart in plan and about 15 cm apart in Z. A model that averages them produces a line in the middle of the face, which is a line nobody asked for.

A **driveway** is a curb cut: the face drops so a vehicle can cross. A detector that demands a 15 cm step all the way down the block will break there, correctly, and then has to start the line again on the other side. An intersection is the same problem in harder geometry, because the alignment is turning and several curbs meet.

A **barrier** (concrete median barrier, guardrail) is a taller vertical face. Same extraction machinery, different height prior.

### Why a vertical face defeats a height field

A lot of "fit a surface" code assumes `z = ax + by + c`. That is a plane that is a function of XY: one elevation per plan location. A curb face is vertical, so several Z values share one XY. No coefficients `a, b, c` can represent it.

The general plane is `ax + by + cz = d`. In practice you center the points and take the SVD; the normal is the singular vector with the smallest singular value (`Vh[-1]` in PyTorch, `Vt[-1]` in NumPy). Flip its sign so the largest component is positive, because SVD's sign is arbitrary. Signed distance from a point to that plane is `(p - centroid) · unit_normal`.

A curb face comes out with a horizontal normal, `|n · up| ≈ 0`. Pavement comes out with a vertical normal, `|n · up| ≈ 1`. That single check is the difference between "this plane is ground" and "this plane is a face."

### Breakline

A breakline is a polyline you are not allowed to smooth across. The ground's slope changes abruptly there: road to gutter, gutter to face, face to sidewalk. If your surface is allowed to triangulate from a pavement point straight to a sidewalk point, it will, and the curb disappears into a gentle ramp. The triangles have to stop at the line.

So a breakline is both a piece of CAD linework and a constraint on the surface. Extracting curbs and building a correct ground surface are the same project.

### TIN

TIN means triangulated irregular network. It is a surface made of triangles.

**Irregular** means the vertices are not on a regular grid. They sit on measured points, on a thinned subset of those points, and on the breakline vertices you forced in. A grid wastes samples on flat pavement and starves the curb. A TIN puts vertices where the surface actually bends.

**Triangulated** means that inside each triangle you treat the surface as planar, and you interpolate.

The usual construction in plan view is a Delaunay triangulation of the XY coordinates: among triangulations of a point set, Delaunay avoids skinny triangles as much as a point set allows (no point lies inside another triangle's circumcircle). `scipy.spatial.Delaunay` computes the unconstrained version. Unconstrained is the problem. It will happily run an edge from the road, across the curb, to the sidewalk, because those are just points in XY and the curb line was never given to it.

A **constrained** Delaunay triangulation is the same idea with required edges. Every breakline segment must appear as a triangle edge. Implementing that is a library problem, not an interview problem. What you say out loud is: breaklines are constraints at build time. Query time does not rebuild the mesh, and it cannot add the constraint back.

**Query.** Someone asks "what is the elevation at this XY?" You find the triangle that contains the query in plan view and interpolate Z with barycentric coordinates.

For a triangle with XY corners `a, b, c` and a query `p`:

```text
v0 = b - a,   v1 = c - a,   v2 = p - a
d_ij = vi · vj
denom = d00 * d11 - d01^2
v = (d11 * d20 - d01 * d21) / denom
w = (d00 * d21 - d01 * d20) / denom
u = 1 - v - w
z = u * za + v * zb + w * zc
```

`u, v, w` are the weights on the three corners. They sum to 1. The query is inside the triangle when all three are ≥ 0 (use a small tolerance so an edge point is not rejected by rounding). If `denom` is about 0, the triangle has no area in XY and it contains nothing. If no triangle contains the query, the answer is "outside the surface," which you return as NaN, not as 0.

Worked mesh:

```text
vertices (0,0,0), (2,0,0), (0,2,10), (2,2,6)
faces (0, 1, 2) and (1, 3, 2)
```

`(0.5, 0.5)` lands in the first triangle at Z = 2.5. `(1.5, 1.5)` lands in the second at Z = 5.5. `(1, 1)` is on the shared edge; either triangle gives Z = 5, so the tie does not matter on a continuous surface. `(3, 3)` is outside, so NaN.

**Long triangles.** A scan has gaps: a river, a building the laser did not see behind, a stretch of parked cars. Delaunay will bridge the gap with a long edge, and that triangle claims an elevation for ground you never measured. Drop any face with an edge longer than a limit (a meter or a few meters; pick it from the point spacing and say the number). That is a filter on the mesh you were handed. It is not a mesher.

Where a breakline sits, elevation is not a single linear function across the line. The two triangles that share a breakline edge can slope very differently. Interpolation within one triangle stays on the correct side. Interpolation that was allowed to cross the line is the bug the constraint exists to prevent.

### Grid

A grid (a raster DEM, digital elevation model) stores one elevation per square cell. Simple, and the right structure when you want a convolutional network or a fast lookup.

Build the interview version as a **min-Z grid**:

- `origin_xy` is the minimum corner.
- `cell` is the square size, in meters.
- Row 0 is the minimum-Y band. This is not an image; row index increases with Y, not with "down the screen."
- A cell is half-open. The point with `x = origin_x + cell` belongs to the next column.
- The stored value is the minimum Z of the points that landed in the cell. Several points will. An ordinary `grid[row, col] = z` keeps one of them and depends on order. You want a real minimum (`np.minimum.at`, or `scatter_reduce_(reduce='amin')` in PyTorch).
- Empty cells are NaN, not 0. Zero is a real elevation.

Min-Z is a crude ground estimate because cars, people, and vegetation sit above the pavement, so the lowest return in a cell is often the ground. It fails under a bridge (the road below and the deck above), under a dense canopy that never let a return through, and on a vertical face (the min is the bottom of the face, and the cell also contains the top). Say the failure when you propose it as a baseline.

A useful companion channel is the height range in the cell, max Z minus min Z. Pavement is a few centimeters. A curb cell is about 15 cm. A car is a meter and a half. A pole is tall and only one or two cells wide.

### Pole and sign

A pole is a vertical cylinder: light pole, utility pole, sign post. In a neighborhood of points it is a thin vertical line. In plan view the returns are a circle. The scanner only sees the near side, so the points are a short arc, not a full ring.

The centroid of that arc is the wrong center. It sits toward the scanner. On a perfect arc of radius 2 m spanning ±0.3 rad, the algebraic circle fit recovers the true center, and the centroid is about 1.97 m out along the arc, almost on the surface. Add about 1 cm of noise and that same algebraic fit can move the center by a large fraction of the radius (one seeded draw put it near 0.17 m with radius about 1.83 m). So: do not report the mean of the XY points as the pole center, and do not trust one least-squares circle on a short noisy arc. A robust fit (RANSAC around the circle, or a geometric fit) is the production version. The deliverable is a center and a radius, or a vertical line in CAD, not a point mask.

A sign panel is a small vertical plane on that pole. Planarity high, normal horizontal, extent of maybe a meter, sitting above the ground. Intensity is often high because the face is reflective.

## Features you compute before any network

These are the baseline. They are also the input channels that make a small network enough.

### Local shape

Take a neighborhood, a few dozen points inside a radius (0.3–1 m is a reasonable curb scale; a pole wants a taller vertical window). Subtract the centroid. The population covariance is `(Xᵀ X) / N`, a 3×3 matrix. Its eigenvalues `λ_max ≥ λ_mid ≥ λ_min` describe the shape. `eigh` returns them **ascending**, so you reorder.

```text
linearity  = (λ_max - λ_mid) / λ_max
planarity  = (λ_mid - λ_min) / λ_max
sphericity = λ_min / λ_max
```

The eigenvector for `λ_max` is the principal direction. The eigenvector for `λ_min` is the normal. Signs are arbitrary; flip each vector so its largest component is positive, or compare absolute dot products with up and stop caring about the sign.

| Neighborhood | Shape | The extra check |
| --- | --- | --- |
| Pole, trunk | linearity near 1 | direction is vertical, `\|direction · up\| ≈ 1` |
| Pavement, dirt | planarity near 1 | normal is vertical, `\|normal · up\| ≈ 1` |
| Curb face, wall | planarity near 1 | normal is horizontal, `\|normal · up\| ≈ 0` |
| Bush, a corner, a car roof edge | sphericity higher | no single axis wins |

Dividing by `N` or `N-1` does not change the ratios. If the neighborhood is a repeated point, `λ_max ≈ 0`; return zeros rather than dividing.

This is a handcrafted descriptor. In the training notebook it is computed under `torch.no_grad()` and a linear layer classifies it. The eigenvalue decomposition does not have to sit in the autograd graph for that design. A PointNet, below, is what you use when you want the descriptor itself to be learned.

### The geometric curb baseline, end to end

You should be able to describe this without a network, because the network has to beat it.

1. Take the trajectory, smooth it, and treat it as the alignment.
2. Every 0.5 m or 1 m of station, cut a cross-section.
3. Bin the section by offset into columns of about 5 cm. Store min Z, or a low percentile, per column.
4. Walk the columns and look for a step of roughly 8–20 cm over one or two columns, with a flatter run on each side. Record `(station, offset, z_bottom, z_top)`.
5. Link detections along station. If the offset jumps by more than a couple of decimeters between stations, end the polyline and start another. That jump is a driveway, a car, or a real corner.
6. Simplify the polyline (Douglas–Peucker) at about 1 cm so a straight curb is a handful of vertices.

Cars fail step 4 because they are too tall. Occlusions fail it because the columns are empty; you leave a gap. Unusual curb shapes and broken concrete are where a learned model starts to earn its keep. Intersections fail the "constant offset" assumption in step 5.

RANSAC is the robust version of "fit a plane to this neighborhood." Sample 3 points, score how many others lie within a centimeter or two, keep the best hypothesis, refit SVD on those inliers. One outlier otherwise tilts a least-squares plane. Ground with a car in the neighborhood, and a facade with a bush in front, are this situation.

## What the network is for

The deliverable is still a polyline with a Z, or a surface that honors it. A network is a good local classifier and a bad CAD system. Let it answer "how much does this neighborhood look like a curb face?" and let the geometry in the previous section turn those answers into vertices, enforce smoothness along station, and check centimeters.

Two local questions, and you should know which one you are answering:

- **One label for the whole neighborhood.** "This patch is a pole" versus "this patch is ground." That is classification. The notebook's `NeighborhoodPointNet` does this.
- **One label per point or per grid cell.** "This point is on the curb face." That is segmentation. A curb extractor needs this, or the equivalent on a bird's-eye grid. Same backbone, different head.

You do not lead with a network that emits a mesh, or with a transformer over the whole corridor. The corridor does not fit in one forward pass, and the output of those models is still the wrong object until you turn it into a line a person can drag.

## PointNet

PointNet (Qi et al., 2017) is the architecture to understand. The notebook implements a small classification version of it. The original paper classifies a whole object (a chair given as 1024 points) or labels every point of that object. A road is not a chair. You reuse the block on a neighborhood or a tile, which is the next section.

### The problem with order

A cloud is a set. If you permute the rows, it is the same physical thing. An MLP on a flattened vector of size `N * 3` changes its answer when you shuffle, so it is learning storage order. A conv net needs a grid the cloud does not have.

PointNet's answer has two parts: apply the same function to every point, then combine the results with an operation that does not care about order.

### Forward pass, classification

Input `points` has shape `(B, N, 3)`. `B` is the batch, `N` is the number of points in each cloud, 3 is XYZ. Extra channels (intensity, height above ground) make it `(B, N, C)`.

**Shared MLP.** The same linear layers run on every point, independently.

```text
(B, N, 3)  --Linear(3, 64)-->  (B, N, 64)
           --ReLU-->
           --Linear(64, 64)--> (B, N, 64)
           --ReLU-->
```

`nn.Linear` applies to the last dimension, so one `Linear(3, 64)` really is shared across all `N` points and all batch rows. There is no interaction between point `i` and point `j` inside this MLP. Each point is summarized from its own coordinates only.

**Max pool.** Take the maximum over the point axis:

```text
(B, N, 64) --max over dim 1--> (B, 64)
```

Max is permutation-invariant: shuffling points does not change the max. Sum and mean are too. Max is the PointNet choice because each output channel can be "the point that scored highest on this feature," and a subset of points determines the vector. The paper calls that the critical point set. The practical consequence is harsher: two different clouds can share a max, and the pooled vector has thrown away which point produced it. For "is this patch a pole?" that is acceptable. For "draw the curb" you need the segmentation head below, or you should not have pooled the whole tile into one vector.

**Head.**

```text
(B, 64) --Linear(64, n_classes)--> (B, n_classes)
```

Cross-entropy against an integer class label. That is the entire `NeighborhoodPointNet` in notebook 07. The synthetic classes are a flat cloud (ground), a vertical flat cloud (wall), and a vertical line (pole), each centered so the model cannot cheat by looking at where the object was placed.

### Forward pass, per-point labels

Classification is the wrong head for a curb. Segmentation PointNet keeps the per-point features and appends the pooled vector back onto every point:

```text
local:   (B, N, 64)
global:  (B, 64)  broadcast to (B, N, 64)
concat:  (B, N, 128)
shared MLP again → (B, N, n_classes)
```

Each point's logit can use both "what I look like" and "what the neighborhood looks like." The loss is cross-entropy per point, almost always weighted, because curb points are rare. This is the head to draw if the whiteboard says "label the points, then vectorize."

### What PointNet does not do

The shared MLP never looks at a neighbor. The only mixing is the global max. On a 1024-point chair that is enough for a class label, because XYZ in a centered, scaled object already says "I am the leg" versus "I am the seat." On a road tile in world coordinates it is not enough: a point at `(5.0, 3.0, 1.2)` means nothing until you know the points next to it.

Two fixes, and you should pick one out loud:

- **Cut the neighborhood yourself** before PointNet. Center it, maybe rotate it so the alignment tangent points along +X, and append handcrafted channels (height above a local ground, intensity). A 64-point patch with a 0.75 m radius is a fine PointNet input. Gathering those 64 points is a KD-tree query, not a layer.
- **PointNet++**, which does the gathering inside the network. Farthest-point sampling picks centroid points. A ball query gathers the neighbors of each centroid within a radius. A mini PointNet (shared MLP + max) runs on each ball. Repeat with a larger radius on the centroids, so the receptive field grows the way it does in a conv net. That is the hierarchical version. It costs a grouping step. For a curb, one radius chosen by you is usually the right amount of hierarchy; say so rather than drawing three levels you cannot justify.

There is also a **T-net** in the original figure: a small PointNet that predicts a 3×3 matrix and multiplies the points by it, meant to rotate the object into a canonical pose. A second one aligns the 64-d features. It is optional, it is fiddly to train (they regularize the matrix toward orthogonality), and on a road you already have a better canonical frame: the section frame whose axes are forward, left, and up. Prefer that.

### A forward pass you can recite

Neighborhood classification, the notebook:

```text
points (8, 32, 3)
  shared Linear 3→64, ReLU
  shared Linear 64→64, ReLU
  max over the 32 points → (8, 64)
  Linear 64→3 → logits (8, 3)
```

Batch 8 is "eight neighborhoods." It is not "eight projects." Cross-entropy, Adam, a learning rate around `1e-2` for this toy. The check in the notebook only asks that the loss fall.

Per-point curb model, what you would draw next to it:

```text
neighborhood (B, 64, C) with C = xyz-local + intensity + height-above-ground
  shared MLP → (B, 64, 64)
  max → (B, 64)
  concat back → (B, 64, 128)
  shared MLP → (B, 64, 2)   # curb / not
```

Train on the point dimension. At inference, write the curb probability back onto the seed point of that neighborhood.

## Architectures worth being able to draw

Pick the bird's-eye network as the primary design for a corridor. Mention PointNet as the local alternative. Know the others well enough to say why you are not leading with them.

### Bird's-eye raster, then a 2D network

Project the tile onto a horizontal grid. This is the min-Z grid, plus more channels per cell:

| Channel | What it catches |
| --- | --- |
| min Z, relative to a ground estimate or to the tile median | elevation |
| max Z − min Z | a vertical face or a tall object; curbs are a thin band of medium range |
| mean intensity | paint, signs |
| point count | density; empty is unknown, not background |
| mean offset, or a channel of signed distance to the trajectory | which side of the road |

Cell size around 5 cm. A tile 50 m along by 30 m across is 1000 by 600 cells. That is an ordinary image. A small U-Net (a conv net that downsamples, then upsamples, with skip connections so the full-resolution output still sees fine detail) predicts a curb probability per cell.

This is usually the right primary design for this product:

- Indexing is binning, which you have already implemented, not a learned grouping module.
- The output is a 2D mask you can thin and trace.
- A vertical curb is only one or two cells wide, and the height-range channel is exactly the signal the cross-section baseline uses. The network is a learned version of "find the step," with intensity and context available.
- Empty cells stay a mask channel. The loss ignores them or treats them as unknown. Do not call them "not curb."

The limit is vertical structure that occupies the same XY as something else: a curb face under a tree canopy, a bridge deck over a road. Min and max in the cell keep some of that. If the interviewer pushes on stacked structure, that is the moment for a sparse 3D voxel net, not before.

### Range image

A spinning mobile laser has a native 2D layout: row is the laser ring, column is the firing angle. Adjacent pixels are adjacent pulses, which are the sensor's own neighbors, and a 2D conv net applies. Distortion grows with range, and a curb that runs along the road is a diagonal or a vertical structure in that image depending on the sweep. Useful as a feature front-end. Less natural than a bird's-eye grid once the output is a plan-view polyline.

### Sparse 3D voxels

Bin into 5 cm cubes and run a 3D conv net only on occupied cubes (sparse convolutions; MinkowskiNet is the name to know). This keeps the curb face as a vertical stack of voxels instead of a height range in one cell. It costs more memory and more implementation than a bird's-eye U-Net. Propose it when the product must separate two surfaces that share an XY footprint, and not as the default.

### What you leave alone unless asked

A transformer attending over every point in a tile is quadratic in the point count. A million points is not an attention matrix you can allocate. Local attention inside a radius is just an expensive version of "MLP on a neighborhood." A model that directly regresses polyline vertices needs a fixed vertex count or an autoregressive decoder, which is a reasonable *interactive* design (predict the next vertex given the ones the surveyor already accepted) and a poor fit for an overnight pass over a city.

## Inference on a cloud that does not fit

This is the systems half of the whiteboard. The order is: store spatially, cut tiles, run a model that sees one tile, write predictions back, vectorize, then (only if you must) lift labels onto the original points.

### Store it so a tile is a read, not a scan

LAS/LAZ on disk, organized by spatial chunks: square tiles, or strips along the trajectory. An octree or a similar index answers "give me the points inside this box" without reading the project. You build that index once when the scan is ingested.

Per tile, in memory, a KD-tree (or a voxel hash) answers radius queries. `scipy.spatial.cKDTree` is the production stand-in for the broadcast nearest-neighbor in the notebook. Say the memory reason: pairwise distances for `N = M = 10_000` are already about 800 MB of float64. A tile is larger than that. You do not broadcast a tile against itself.

### How big a tile is

Work a number so the interviewer can see you chose it.

Take a mobile corridor, 10 km long, about 30 m of useful width, on the order of a couple hundred points per square meter on the pavement and fewer at the edges. Call it `10e3 m * 30 m * 200 pts/m² = 60 million` points as a modest project, and be ready for an order of magnitude more on a dense or multi-pass scan. "A billion points" is several such corridors, or one dense city job. Same design either way.

A processing tile 50 m along-track by the full 30 m width holds about `50 * 30 * 200 = 300,000` points at that density, and a few million if the scan is dense or you kept every return. A 5 cm bird's-eye grid of that tile is 1000 × 600, independent of how many raw points fell in. Rasterizing is a bin, linear in the point count, once per tile.

Overlap the tiles by more than the context the model uses. If you only trust the center of the prediction, step 40 m with a 50 m window: every curb point is predicted with 5 m of road on either side along track, and you discard the outer 5 m so a curb cut by the tile boundary is not emitted twice. Points in the overlap can vote if you prefer; averaging two probabilities is fine, concatenating two polylines is not. Merge lines across the boundary by endpoint distance after both tiles are traced, or trace only on the interior and let the next tile own the seam.

### Two ways to run the model

**Bird's-eye, one forward pass per tile.** Rasterize to the channels above, run the U-Net, read a probability per cell. GPU memory is the image and the activations. A 1000 × 600 U-Net is comfortable on a single GPU. Two hundred tiles for 10 km, well under a second of compute each, is minutes for the corridor. Overnight is easy. This is the design to defend first.

**Neighborhood PointNet, one query per seed.** Voxel-downsample the tile (one centroid per 5 cm cube, the notebook problem) so you are not classifying every redundant return. For each seed, query up to 64 neighbors in a radius of about 0.75 m, stack them as `(B, 64, C)`, run the segmentation head, write the probability to the seed. Gathering is the expensive part, not the MLP. Batch seeds (1024 at a time is plenty; the tensor is under a megabyte). A few hundred thousand seeds per tile is still a lot of kernel launches if you do them one at a time, so the batching matters. This design wins when the object is vertical and thin and you already have a KD-tree for the rest of the pipeline (poles are a good example). It loses to the raster on engineering simplicity for curbs.

Either way, subtract the tile origin first. Run geometry you trust (the plane fit, the circle fit, the section frame) in float64. The network itself is almost always float32. Say both.

You do not run one forward pass on 60 million points. You do not build one activation tensor for the project. If someone proposes it, the reply is the tile arithmetic above.

### From probabilities to a polyline

A mask is not a deliverable. Do this in station order, because the road is one-dimensional and you already have the alignment.

1. Threshold. Keep cells or seeds above some probability. Also drop isolated hits: a single positive cell with no neighbor along station is noise.
2. In each station bin (you can reuse the cross-section stations), collapse the positive band to one offset: the probability-weighted mean, or the peak. Record Z from the flowline or the top, whichever line you said you were drawing. Do not average the top and the bottom.
3. Link collapses down the road. A new polyline starts when the offset jumps (driveway, intersection, a gap of empty cells).
4. Douglas–Peucker at about a centimeter. Straight runs become long segments. Corners keep a vertex.
5. Score the candidate before you show it to anyone. Distance from the polyline to the positive points, fraction of station covered, and a rejection if the line bridges a gap you meant to leave open (a doorway, a missing scan). A line that jumps sideways to catch a car is worse than a gap.

Z comes from the cloud at the chosen edge, not from the network's imagination, unless you have a separate head that regresses a residual and you have checked it in centimeters. A plan-view curb with sidewalk Z cuts a notch out of the surface in the wrong place.

### Putting labels back on the original points

Only if a downstream step needs per-point classes. Nearest seed, or inverse-distance weights of the k nearest seeds, spreads a voxel prediction onto the raw returns inside it. This interpolation is not a second model. Many products skip it and emit the polyline directly.

### Interactive serving is a different budget

Overnight batch owns the whole tile. Interactive means the surveyor clicked a start point and wants the next few meters of curb while their hand is still on the mouse. You query one neighborhood around the click and the points ahead along the alignment, run the small network or even just the geometric step detector, and propose a vertex. Latency is tens of milliseconds, so the index for that project is already in memory or memory-mapped. You do not rasterize the city on the click.

The training data for that interactive model is sequences of accepted vertices, not a point mask. Mention it as the alternative design if they say the product is a drawing tool rather than a batch processor. Ask which one before you commit.

## Training

### Where labels come from

The drawing that already exists is the label. A finished curb is a polyline. Turn it into a training target in one of two ways:

- **Point or cell labels.** A point is positive if its perpendicular distance to the label polyline is under a band you can defend, something like 5 cm, and you have decided whether the label is the flowline or the top. A band that is wider than the object teaches the model to paint a blob. A band of zero width on noisy lidar teaches it that nothing is a curb.
- **Line loss, after vectorization.** Sample points along the predicted polyline and penalize distance to the label polyline, and the other way around. This matches the product better and is harder to differentiate through a threshold-and-link step. A practical compromise is a point loss for training and a line metric for the decision to ship.

Surveyor edits are the better label. The model proposes a line, the person moves it, and the accepted line is what that shop actually draws. Those edits also tell you the failure slices. Ship the geometric baseline first, log the edits, and train on the edits. Synthetic curbs are a way to debug the code. They are a poor substitute for the way a particular client draws a curb cut.

### Imbalance, with the arithmetic

A 50 m by 30 m tile has area 1500 m². Two curbs, 50 m each, in a band 10 cm wide, are 10 m². That is under one percent of the cells, and the point fraction is similar. A classifier that answers "not curb" everywhere is about 99% accurate and produces an empty drawing.

So: do not report accuracy. Sample the loss so positives matter (weight the classes, or sample a fixed fraction of positives per tile), and pick the decision threshold on a validation corridor to maximize the line metric, not the point F1. A point 10 cm off the curb is a different error from a point on a wall; once you have polylines, the loss that says so is point-to-polyline distance.

### A split that does not leak

Split by project, and if you can, by city or by sensor. A random split of points puts one return from a curb in train and the next return, 2 cm away, in test. The model memorizes that curb and the number is fiction. A random split of tiles from the same project still shares the trajectory style, the city, and the way that crew draws. Hold out a corridor the model has never seen, and look at the error there before you believe a tile-level number.

The toy clouds in notebook 07 are independent draws, so a random holdout is legal there. Say that sentence in the interview if you describe the exercise. Then say the real split.

### Augmentation that respects the object

Reasonable: rotate around Z, because a road has no preferred compass heading; jitter by about a centimeter, because that is the noise; drop points at random, because occlusion is real; scale intensity, because it is not calibrated.

Leave Z's direction alone. Flipping left and right reverses which side of the road is which; only do it if every side-dependent feature flips with it. Do not scale the cloud by a large factor. A 15 cm curb that you have scaled into a 15 m curb is a different object, and your centimeter metric becomes meaningless.

### Optimization, only the part worth saying

Adam is a fine default. Float32 for the network. Gradient clipping if a bad tile produces a spike. Early stopping on the **line** metric of the held-out corridor, not on training loss. If the point loss improves and the 5 cm completeness does not, you are optimizing the wrong thing; stop and look at the vectorizer before you grow the network.

## Metrics

Define these before you draw a model. They are how you will know the baseline is beaten.

Sample both polylines densely, every 10 cm of station or so.

- **Completeness** at tolerance τ: length of the reference that has a predicted point within τ, divided by reference length. This is recall on length.
- **Correctness** at tolerance τ: length of the prediction that lies within τ of the reference, divided by predicted length. This is precision on length.
- Report both at **τ = 5 cm** and **τ = 10 cm**. Five centimeters is "a surveyor might leave it." Ten is "in the right place, they will nudge it."
- **Plan error and vertical error separately.** A line can be perfect in XY and 20 cm off in Z. The surface will be wrong and the plan-view score will not show it.
- **Edit burden:** after a person finishes the tile, how many meters of line did they move by more than 5 cm, or the average distance they moved it. This is the product number. It needs a person, so you will not have it on every experiment. Have it on the runs that decide whether to ship.
- **Slices:** driveways, intersections, parked cars, the far curb, wet pavement, a second pass that does not quite align. An average over a straight empty block will hide the product.

Work one example so the definitions are mechanical. Reference curb, 100 m. Your line covers 90 m of it within 5 cm, misses 10 m (a driveway you dropped and a parked car you could not see through), and continues 15 m into a flower bed.

```text
completeness = 90 / 100 = 0.90
correctness  = 90 / (90 + 15) = 0.86
```

Point accuracy on the tile can be 0.99 at the same time. Lead with the two length numbers.

Chamfer distance between the two polylines (mean nearest-point distance, prediction to reference plus reference to prediction) is a single summary. It hides whether you are missing length or inventing it. Report completeness and correctness first. The notebook Chamfer is the point-cloud version of the same idea; do not score raw clouds with it and call that a curb metric.

## Poles, as a full design

Same system, different geometry. Say the differences rather than repeating the tile math.

**What you are detecting.** A vertical cylinder, often 10–40 cm in diameter, from about the ground to several meters up. The scan sees a partial arc. A sign is a small vertical plane attached to one of these.

**Baseline.** Voxelize or cut a vertical column. High linearity and a vertical principal direction is a pole candidate. Fit a circle in XY with a robust fit, not the centroid. Reject candidates whose radius is absurd for a pole, whose vertical extent is a few centimeters (that is a bollard you might still want, or noise), or whose points are a flat wall (high planarity, horizontal normal, wide extent).

**Model, if the baseline is not enough.** A neighborhood PointNet or a small vertical voxel column, classifying pole versus trunk versus clutter. Trees are the hard negative: a trunk is also a vertical cylinder for the first two meters. Height, branching above, and intensity help. The bird's-eye height-range channel lights up both poles and curbs; the footprint (one cell versus a long ribbon) separates them.

**Output.** A center `(x, y)`, a radius, and a vertical segment from z0 to z1. Metric: horizontal center error in centimeters, radius error, and whether the pole was found at all (a missed pole is a missed asset, not a short line).

**Failure.** Occlusion by a vehicle cuts the arc down to a stub and the circle fit swings wild. Say you would rather emit nothing than a center a meter off, and let the person place those.

## Ground and the TIN, as a full design

**What you are building.** A bare-earth surface: one elevation per XY, plus breaklines where the slope is not allowed to blend. Cars, people, and vegetation are not in it. Bridges need a rule (which surface is "the" ground).

**Baseline.** Min-Z grid at 0.25–1 m, or a local plane fit with a vertical-normal check and RANSAC so one car roof does not tilt the plane. Classify points as ground when they sit within a few centimeters of that surface. Build the TIN from the ground points and the breaklines. Drop long edges.

**Model.** A per-point or per-cell ground probability, features being height above the min-Z estimate, shape, intensity, and return number (last return is a prior for ground). Or regress a residual against the baseline grid. The residual head is honest: most of the surface is already fine, and the network corrects the cells where min-Z grabbed a shrub.

**Output and metric.** Elevation error in centimeters on held-out ground points that a person has accepted, long-triangle count, and a hard check that no triangle edge crosses a breakline. A smooth surface that ramps through the curb is a failure even if the RMSE on open pavement looks excellent. Split by project.

**Where the curb design plugs in.** The curb polylines are the constraints. You can build a ground model without them and it will look plausible from a distance and be wrong at every gutter. If the interviewer only gives you time for one system, design the curb, and say the TIN consumes its output.

## Whiteboard script

Use this only after the sections above are familiar. Forty minutes is a long time if you are defining TIN in the middle of it. The script assumes you can explain any word the moment someone asks.

1. **Restate the task** as a deliverable. "A 3D polyline per curb, flowline unless you tell me top-of-curb, a few centimeters, for a person to accept." Ask batch versus interactive, what labels exist, and what a miss costs relative to a false line.
2. **Draw the data.** One point is XYZ plus intensity, return, time. The cloud is unordered. The trajectory gives station. A tile is the unit of compute. Give the 50 m × 30 m number.
3. **Draw the baseline** on the cross-section: bin offset, find a 15 cm step, link along station. Say this is v1 and the source of the first edits.
4. **Draw one model.** Bird's-eye channels (min Z, height range, intensity, count), 5 cm cells, a U-Net, ignore empty cells in the loss. Name PointNet-on-a-neighborhood as the alternative for poles and as the local model if they do not want a raster.
5. **Vectorize** in station order. Threshold, collapse to one offset per station, break the line on jumps and on empty gaps, simplify at 1 cm, take Z from the cloud at the edge you promised.
6. **Metrics.** Completeness and correctness at 5 cm and 10 cm, vertical error separate, edit distance when a person is in the loop. Give the 100 m example if you have the pen for it.
7. **Split and scale.** Hold out a project. Overlap tiles by the receptive field. Float64 for the geometry, float32 for the net. Do not load the corridor as one tensor.
8. **Close on an open decision.** Batch versus interactive, or flowline versus top-of-curb, whichever they did not answer in step 1. State what you would measure on the first offline test.

If they change the object to poles or to the ground surface, keep steps 1, 6, 7, and 8, and swap the baseline and the output using those sections.

## Coding round

The PyTorch notebooks are the geometry problems again. Do the short list if time is the constraint: notebook 01, nearest neighbors, voxel downsample, plane fit, station and offset, shape features, notebook 07.

The API details that actually change an answer:

- `torch.set_default_dtype(torch.float64)` for geometry. The library default is float32, and a plane SVD or Kabsch in float32 can miss a hand check.
- Neighbors: `torch.cdist`. For anything past about 10,000 by 10,000, say KD-tree and chunk.
- Voxel keys: `torch.floor(...).to(torch.int64)`, then `torch.unique(..., dim=0, return_inverse=True)`, then `scatter_add_`. Adding with `sums[idx] += points` drops duplicate indices. A cast that truncates toward zero merges the voxel of `-0.1` with the voxel of `+0.1`.
- Min-Z: `scatter_reduce_(..., reduce='amin')`.
- `torch.linalg.svd` returns `Vh`. `torch.linalg.lstsq` returns `.solution`. `eigh` is ascending.
- `torch.linalg.cross` does not broadcast `(N, 3)` against `(3,)`. Broadcast the inputs first.
- A differentiable point-to-plane loss is `((points - centroid) @ unit_normal).square().mean()`. Do not detach `points`. Do not update a tensor that requires grad in place.

Notebook 07 is the training-loop drill: that loss, then a linear head on shape features computed under `no_grad`, then the small PointNet. Talk about the split while you write it.

## Using an AI tool in the interview

If they put you in an editor with a model, they are watching who drives.

Write the convention into the prompt: Z up, points as `(N, 3)` rows, `p' = R @ p + t`, offset positive to the right of increasing station, station horizontal. Then write the hand check in the same prompt. Run it yourself.

Checks that catch a fluent wrong answer:

- `+90°` about +Z sends `(1, 0, 0)` to `(0, 1, 0)`, and `det(R) = +1`. Kabsch without the determinant fix will reflect.
- The plane `x = 0` has normal `(1, 0, 0)`. A height-field fit cannot return it.
- On the alignment `(0,0) → (10,0) → (10,10)`, the point `(5, 2)` is station 5 and offset −2.
- A short arc's centroid is not the circle center.

Say which property you are checking. If the task is the design problem rather than a function, let the model sketch and you choose the split, the tile size, and the centimeter metric.

## Behavioral

Two stories, each with one number and one decision. The themes that fit this product:

- A metric that looked good and measured the wrong thing, and what you replaced it with.
- A domain expert who drew the object differently than the spec. What you changed, or what you refused to change, and the evidence.
- An ambiguous definition (which edge of the curb, which tangent at a corner) that you made explicit.
- Work you shipped that a person still had to edit, and what you did with the edits.
- A bug that was a sign, a transpose, or a zero-length vector, and the three-point check you added.

## Order to study

1. This note, through the TIN section, until you can draw a cross-section and explain why a triangle is not allowed to cross a curb.
2. The PointNet shapes, until you can write `(B, N, 3) → shared MLP → max → (B, 64)` and say why the segmentation head concatenates the pooled vector back.
3. The tile arithmetic, until you can recompute points per tile and grid size without looking.
4. The geometric baseline, out loud, as a sequence of array operations you have already coded: section, bin, min, link, simplify.
5. Notebook 07, then the short PyTorch geometry list.
6. The whiteboard script, once, on paper.
7. Two behavioral stories.
8. The morning of the interview, the formula sheet, portrait, letter, 100% scale.
