# Spatial NumPy for a Mach9 interview

Mach9 turns mobile, terrestrial, and drone lidar into civil CAD: curbs, pavement edges, barriers, poles, signs, breaklines, cross-sections, ground grids, and TIN surfaces. The geometry round is arrays of shape `(N, 3)`, written with broadcasting instead of Python loops over points. The onsite also has a system-design whiteboard, and the coding round may be the same geometry in PyTorch. That prep is in [onsite_prep.md](onsite_prep.md).

The outline you already have is the right linear-algebra core: dot and cross products, rigid transforms, planes, nearest neighbors, voxels, Kabsch, a camera model. What it underweights is the survey geometry that sits on top of that core.

## What to add

These are the pieces most worth adding, in the order I would add them.

1. **Z-up, 2.5D coordinates.** Survey and CAD frames put Z up. Plan view is XY. Station, offset, pavement polygons, and TIN interpolation all happen in XY, with Z carried along as an attribute. A graphics camera is a different frame: Y can be up or down, and +Z may look out of the screen or into it. Say which frame you are in before writing a formula.

2. **Polylines.** Breaklines and alignments are polylines. Project a point onto a polyline, measure horizontal station and signed offset, resample at a fixed spacing, and simplify with Douglas–Peucker.

3. **Cross-sections.** At one station, build a vertical plane perpendicular to the horizontal tangent and report points as `(offset, elevation)`. Offset is positive to the right of increasing station. Elevation in the exercises is absolute Z.

4. **A ground grid.** Bin XY into square cells and keep the minimum Z. That is the simple form of a "grid of points" surface. The cell has to be a true minimum when several points land in it.

5. **Local covariance features.** On a neighborhood, eigenvalues of the covariance give linearity, planarity, and sphericity. A pole is linear and vertical. Ground is planar with a vertical normal. A curb face is planar with a horizontal normal. This is the classic robotics description of a point, and it is the right way to talk about feature extraction before any network.

6. **TIN queries, not a mesher.** Barycentric interpolation of Z inside a triangle. Drop triangles with an edge longer than a limit. Know what a breakline is: a constraint the triangulation is not allowed to cross, because the curb is a break in slope. Implementing constrained Delaunay is a library problem (`scipy.spatial.Delaunay` is the unconstrained version). Querying the mesh is the interview problem.

7. **Vertical planes.** A curb face is a plane. The model `z = ax + by + c` cannot represent it, because a vertical plane is not a function of XY. Fit `ax + by + cz = d` with SVD.

8. **Poles are partial arcs.** In plan view a pole is a circle. The centroid of the visible points is not the center: the scanner only sees one side. An algebraic circle fit is exact on a perfect arc and fragile once a short arc has a centimeter of noise.

9. **RANSAC as a default, not a footnote.** Ground, facades, and curb lines are fit in the presence of outliers. One loop of "sample, score, keep the best, refit the inliers" is the expected shape.

10. **Negative coordinates in voxel keys.** `np.floor` is correct. Casting to int truncates toward zero and merges `-0.1` with `+0.1`.

11. **2D predicates.** Point-in-polygon for a surface boundary, and a proper segment-intersection test for breaklines that cross. CAD linework is largely this, plus Z.

Ray–triangle, quaternions, and Chamfer distance are still worth being able to write. They are less central here than station, offset, grids, and local shape.

## How to use the notebooks

Read this guide once. Then do the notebooks in `notebooks/`, in order, without scrolling into a solution cell.

Each problem is three cells: the prompt, an attempt cell with the signature and `raise NotImplementedError`, and a solution cell. The solution cell replaces your function and runs checks. Hints are collapsed in the prompt.

If time is short, do these and skim the rest:

- 01, all four (vectors, Rodrigues, rigid inverse, section frame)
- 02.2 nearest neighbors and 02.4 voxel downsample
- 03.1 plane fit and 03.2 point-to-segment
- 05.1 station and offset, 05.2 cross-section
- 06.1 min-Z grid, 06.2 shape features, 06.3 TIN interpolation
- 04.1 Kabsch

Say the production tool out loud after the NumPy version works: `scipy.spatial.cKDTree` for neighbors, `scipy.spatial.Delaunay` for an unconstrained TIN. The interview still wants the array version first.

## Say this before you write code

> Points are `(N, 3)` rows, Z up, meters. A rigid motion is `p' = R @ p + t`, with `R` acting on column vectors, so a cloud transforms as `(R @ P.T).T + t`. Positive rotations follow the right-hand rule. Offset is positive to the right of increasing station, and station is horizontal distance. I will name empty inputs, zero-length vectors, and parallel rays as I hit them, and I will compare floats with a tolerance.

That paragraph prevents the failure mode this interview is built to catch.

## Conventions

| Choice | What to use here |
| --- | --- |
| Point layout | `(N, 3)` rows, float64 |
| Up | `+Z` |
| Plan view | XY |
| Active rotation | `p' = R @ p`, so rows use `P @ R.T` |
| Rigid pose | `p' = R @ p + t`, rotation about the origin, then translation |
| Inverse | `R⁻¹ = R.T`, `t_inv = -R.T @ t`. No general inverse |
| Valid rotation | `R.T @ R = I` and `det(R) = +1`. `det = -1` is a reflection |
| Angles in code | radians |
| Station | horizontal chainage along the polyline, XY only |
| Offset | positive to the right. Tangent `(tx, ty)` has right normal `(ty, -tx)` |
| Camera, when it appears | looks down `+Z`, `u` is column, `v` is row, image shape `(height, width)` |
| Tolerance | `np.isclose` or an epsilon. A 1–2 cm epsilon is a surveying tolerance. `1e-8` is a numerical tolerance. They are not the same decision |

Apply a transform to a cloud:

```python
def apply_rigid(points, R, t):
    return (R @ points.T).T + t
```

Compose "first, then second":

```python
R = R_second @ R_first
t = R_second @ t_first + t_second
```

Matrix multiplication does not commute. Say the order in words, then write it.

A 4×4 matrix is the same pose, useful when you chain sensor → vehicle → world:

```python
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t
# inverse is the rigid inverse, not a general one
```

Homogeneous points are `np.c_[points, np.ones(len(points))]`, and you drop the last column after `T @ ph.T`.

## Vectors

Dot product projects. Cross product builds a normal and a signed area. `||a × b|| / 2` is the area of a triangle. The direction of `a × b` follows the right-hand rule, so a triangle's winding decides which way the normal points.

```python
n = np.linalg.norm(v, axis=-1, keepdims=True)
unit = v / np.maximum(n, 1e-12)

cos = np.clip(np.sum(unit_a * unit_b, axis=-1), -1.0, 1.0)
angle = np.arccos(cos)

# projection of a onto b, b not necessarily unit
proj = ((a @ b) / (b @ b)) * b
```

`keepdims=True` is the difference between a later broadcast succeeding and a silent shape bug. Check shapes when you normalize.

A zero vector makes the angle and the unit vector undefined. Guard it. Do not let a single bad normal become NaN and poison a whole cloud.

Triangle normals from a mesh:

```python
tris = vertices[faces]            # (F, 3, 3)
normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
area = 0.5 * np.linalg.norm(normal, axis=1)
unit = normal / np.linalg.norm(normal, axis=1, keepdims=True)
```

Area-weighted vertex normals are those face normals scattered onto the three vertex indices with `np.add.at`, then normalized. Same accumulation issue as voxel centroids.

## Rotations

Rodrigues, unit axis `k`, skew matrix `K`:

```text
K = [[ 0, -kz,  ky],
     [ kz,  0, -kx],
     [-ky,  kx,  0]]
```

Memorize one of these and know they are different:

```text
R = I + sin(θ) K + (1 - cos(θ)) K²
R = cos(θ) I + sin(θ) K + (1 - cos(θ)) (k kᵀ)
```

`K²` equals `k kᵀ - I` for a unit axis, which is why both formulas work and why swapping the terms does not. Positive `θ` is the right-hand rule. A `+90°` rotation about Z sends `(1, 0, 0)` to `(0, 1, 0)`. A `+120°` rotation about `(1, 1, 1)` cycles X → Y → Z.

A section frame from a horizontal tangent, columns `(forward, left, up)`:

```python
forward = np.array([tx, ty, 0.0])          # already normalized
left = np.array([-ty, tx, 0.0])            # forward × left = up
up = np.array([0.0, 0.0, 1.0])
R = np.column_stack([forward, left, up])
```

Tangent east → `R = I`. Tangent north → left is west. `det(R)` is `+1`.

You can survive without quaternions. If they ask: a rotation by `θ` about a unit axis is `q = (cos(θ/2), sin(θ/2) * axis)` in `(w, x, y, z)` order, and the matrix below matches the Rodrigues convention above. Normalize `q` first. Compose by quaternion multiplication if you need it; otherwise convert to `R` and stay with matrices.

```python
def quat_to_matrix(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
```

## Point clouds

Centroid, bounds, mask:

```python
center = points.mean(axis=0)
mn, mx = points.min(axis=0), points.max(axis=0)
kept = points[(points[:, 2] > 0.0) & (points[:, 2] < 5.0)]
```

Unit-sphere normalization: subtract the centroid, divide by the max distance from it. If that distance is 0, return the centered zeros.

Nearest neighbors:

```python
dist = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)  # (N, M)
idx = dist.argmin(axis=1)
nn_dist = dist[np.arange(len(A)), idx]
```

`N = M = 10_000` is `1e8` float64 distances, about 800 MB. Past that, chunk `A` or use a KD-tree. Say this number. It is the difference between a demo and a corridor scan.

Chamfer distance, as used in the notebooks, is the sum of the two directed means:

```text
mean_a min_b ||a - b|| + mean_b min_a ||b - a||
```

Other definitions divide by two or use squared distances. Name yours.

Voxel centroids:

```python
keys = np.floor(points / voxel).astype(np.int64)
_, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
sums = np.zeros((len(counts), 3))
np.add.at(sums, inverse, points)
centroids = sums / counts[:, None]
```

`floor(-0.1) = -1`. `int(-0.1) = 0`. That is the whole bug. `sums[inverse] += points` does not accumulate repeated indices; `np.add.at` does. `np.unique` makes this `O(N log N)` from the sort.

## Planes, segments, boxes

Signed point-to-plane distance, with a unit normal:

```python
distance = (points - point_on_plane) @ normal
```

Least-squares plane:

```python
centroid = points.mean(axis=0)
_, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
normal = Vt[-1].copy()
k = int(np.argmax(np.abs(normal)))
if normal[k] < 0.0:
    normal = -normal
```

The last right-singular vector is the direction of least variance. Flipping the sign is only a convention, but pick one or every test flickers. Inliers are `np.abs(distance) <= tolerance`. This fit uses every point, so one outlier tilts the plane.

Point to a finite segment `ab`:

```python
ab = b - a
t = np.clip(((points - a) @ ab) / (ab @ ab), 0.0, 1.0)
closest = a + t[:, None] * ab
dist = np.linalg.norm(points - closest, axis=1)
```

Leave `t` unclamped and you have the infinite line. Guard `||ab|| = 0`.

Ray against a plane. `t` is meaningful only for hits in front of the origin:

```python
unit = normal / np.linalg.norm(normal)
denom = directions @ unit
parallel = np.abs(denom) < 1e-8
t = np.divide((point - origins) @ unit, denom, out=np.zeros(len(origins)), where=~parallel)
hit = (~parallel) & (t > 1e-8)
```

A parallel ray has a denominator near 0. A hit behind the origin has `t < 0`.

Axis-aligned IoU clips each axis, then multiplies extents. A shared face has volume 0.

```python
inter_ext = np.maximum(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0.0)
inter = np.prod(inter_ext)
union = np.prod(a_max - a_min) + np.prod(b_max - b_min) - inter
iou = inter / union if union > 0 else 0.0
```

Oriented box: `R`'s columns are the box axes in world coordinates. Local coordinates of row points are `(points - center) @ R`. Inside means every `|local_i| <= half_extent_i`. The transpose test is the point `(0, 1.5, 0)` against a box that is long in local X after a `+90°` yaw. One of the two transposes accepts it and the other rejects it.

Möller–Trumbore is in notebook 03 if you want the full ray–triangle drill. The facts to say even if you are deriving it on the spot: barycentric coordinates must be non-negative and sum to at most 1, `t` must be positive, and a near-zero denominator means the ray is parallel to the triangle.

RANSAC, one plane:

```text
best = none
repeat:
    pick 3 points
    skip the sample if the cross product is ~0
    count points within threshold of that plane
    keep the hypothesis with the largest count
refit SVD on the inliers
recompute the mask from the refit plane
```

The loop is over hypotheses. The distance test is vectorized. Seed the RNG.

## Alignment

Kabsch. Correspondences are already known. `P[i]` matches `Q[i]`.

```python
cP, cQ = P.mean(0), Q.mean(0)
H = (P - cP).T @ (Q - cQ)
U, _, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
if np.linalg.det(R) < 0.0:
    Vt = Vt.copy()
    Vt[-1, :] *= -1.0
    R = Vt.T @ U.T
t = cQ - R @ cP
```

`H` is 3×3, so the SVD is cheap no matter how many points you have. The determinant fix rejects reflections. If someone reflects the cloud through a plane, a rigid `R` cannot follow it, and forcing `det = +1` is what you want for a real scan.

One ICP iteration is nearest neighbor, then Kabsch. It recovers the motion when neighbors are the true matches: small motion, or points separated by much more than the motion. A large rotation pairs the wrong points. Say that, and mention that a real implementation also caps iterations and a residual.

If they ask about scale (a similarity, not a rigid motion), the extra scalar is the Umeyama scale `trace(S with the same determinant fix) / sum(||centered P||²)`. Rare next to rigid Kabsch. Mention it exists and write rigid first.

## Cameras

Secondary for this product, because imagery is registered to the same world as the cloud. The model to write without hesitation:

```text
p_c = R @ p_w + t          # camera looks down +Z
q = K @ p_c
u, v = q[0] / q[2], q[1] / q[2]
keep if z > 0 and 0 <= u < width and 0 <= v < height
```

Back-project a depth map, depth stored as camera-frame Z:

```python
vs, us = np.indices(depth.shape)
pixels = np.stack([us, vs, np.ones_like(us)], axis=-1).astype(float)
rays = pixels @ np.linalg.inv(K).T
cloud = (rays * depth[..., None])[depth > 0.0]
```

`depth[v, u]` is row `v`, column `u`. Dividing by `z <= 0` is the bug to avoid; write NaN or drop the point. Inverting one 3×3 `K` is fine. Reach for `np.linalg.solve` when you would otherwise invert inside a loop.

## Linework

This is the section the generic 3D outline does not teach.

**Station and offset.** For each point, find the closest point on the finite polyline in XY. Station is chainage to that closest point, in `[0, length]`. Offset is `(point - closest) · right` of the winning segment. Ties: smallest segment index.

Broadcasting shape is `(N, S, 2)` with `S` segments. On a corridor, `S` is the centerline and `N` is the cloud, so `S` is the small dimension. Say that. It is a better scaling comment here than a generic "use a KD-tree."

Past the ends, station clamps to the endpoint. Offset is then only the lateral part of the vector to the endpoint, not the Euclidean distance. At a kink, two segments can be equally close and their tangents disagree, so the sign depends on the tie-break. Say both limitations.

**Cross-section.** Horizontal lengths, cumulative station, `np.searchsorted` to find the segment. Origin interpolates in 3D, so a grade shows up in the section origin's Z. The plane is vertical:

```text
along  = (xy - origin_xy) · tangent
offset = (xy - origin_xy) · (ty, -tx)
keep   = |along| <= thickness and |offset| <= half_width
```

Return `(offset, z)` sorted by offset. At an interior vertex, notebook 05 uses the outgoing tangent. At the last vertex, it uses the incoming one. The tangent jumps at a kink; pick a rule and state it.

A **profile** is elevation against station along the road. A **cross-section** is offset against elevation at one station. They are different plots. Don't mix the names.

**Resample** an open polyline every `spacing` along 3D length, including both ends. `np.arange` up to a total length is a floating-point trap (`0.1 * 10` does not land on `1`). Build the samples from `round(total / spacing)`, drop anything past the end, and append the exact endpoint:

```python
seg = np.diff(vertices, axis=0)
length = np.linalg.norm(seg, axis=1)
cum = np.concatenate([[0.0], np.cumsum(length)])
total = float(cum[-1])
n = max(int(np.round(total / spacing)), 1)
targets = np.arange(n + 1) * spacing
targets = np.append(targets[targets < total - 1e-8], total)
idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(length) - 1)
alpha = np.divide(
    targets - cum[idx], length[idx], out=np.zeros_like(targets), where=length[idx] > 1e-12
)
samples = vertices[idx] + np.clip(alpha, 0.0, 1.0)[:, None] * seg[idx]
```

For stationing, those lengths are horizontal. For a 3D breakline resample, they are 3D. Say which.

**Douglas–Peucker.** Stack of index spans. Always keep the endpoints. On each span, keep the farthest intermediate vertex if its distance to the segment exceeds `epsilon`, then push both halves. A point exactly at `epsilon` can go either way; the notebook uses `>` and drops it. NumPy is for the distance test. The stack is ordinary Python, and that is fine.

**Point in polygon**, even-odd rule. Ray in `+X`. Count an edge when exactly one endpoint is strictly above the point, and the crossing is to the right of the point. Skip horizontal edges before you divide. The strict `>` is what stops a vertex from being counted twice. Boundary points are not a reliable answer; don't test them, and say so.

**Proper segment crossing.** Shared endpoints and collinear overlaps are not proper crossings:

```python
def cross2(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

ab, cd = b - a, d - c
proper = (cross2(ab, c - a) * cross2(ab, d - a) < 0.0) & (
    cross2(cd, a - c) * cross2(cd, b - c) < 0.0
)
```

Breaklines that share a vertex fail this test on purpose. Collinear overlap needs a separate interval test if the question includes it.

## Terrain, TINs, and shape

**Min-Z grid.** Row 0 is the minimum-Y row, so row index increases with Y. Cells are half-open on the low side: `floor((xy - origin) / cell)`. A point exactly on the upper edge belongs to the next cell.

```python
grid = np.full((n_rows, n_cols), np.inf)
# rows, cols from floor, then mask to the grid
np.minimum.at(grid, (rows[inside], cols[inside]), z[inside])
grid[np.isinf(grid)] = np.nan
```

`grid[rows, cols] = z` keeps one write per cell and depends on order. `minimum.at` is the min. The same pattern with `maximum.at` is a max-Z grid. A ground estimate is often the minimum, or a low percentile, because vehicles and vegetation sit above the surface. Say which reduction you chose.

**Shape features.** Population covariance ` (X.T @ X) / N `. `eigh` returns eigenvalues ascending, so they are `(λ_min, λ_mid, λ_max)`.

```text
linearity  = (λ_max - λ_mid) / λ_max
planarity  = (λ_mid - λ_min) / λ_max
sphericity = λ_min / λ_max
```

Direction = last column of the eigenvectors. Normal = first column. Flip signs so the largest component is positive.

| Neighborhood | What is high | Extra check |
| --- | --- | --- |
| Pole, trunk, pole-like sign | linearity | `\|direction · up\| ≈ 1` |
| Pavement, ground | planarity | `\|normal · up\| ≈ 1` |
| Curb face, wall | planarity | `\|normal · up\| ≈ 0` |
| Vegetation, corners | sphericity | no single axis dominates |

Dividing by `N` or `N - 1` does not change the ratios. Degenerate neighborhoods (a repeated point) have `λ_max ≈ 0`; return zeros rather than dividing.

**Barycentric Z.** For triangle corners `a, b, c` in XY and a query `p`:

```text
v0 = b - a,  v1 = c - a,  v2 = p - a
d_ij = vi · vj
denom = d00 d11 - d01²
v = (d11 d20 - d01 d21) / denom
w = (d00 d21 - d01 d20) / denom
u = 1 - v - w
inside if u, v, w >= 0 (with a small tolerance) and denom is not ~0
z = u za + v zb + w zc
```

Broadcast queries against faces and the mask is `(N, F)`. The first containing face wins. On a shared edge of a continuous mesh both triangles agree, so the tie is harmless. A breakline is not something you enforce at query time. It was enforced when the triangles were built: no triangle crosses that polyline, because elevation is not linear across a curb.

Long edges: drop a face if any of its three edge lengths exceeds a limit. That is the sliver filter that stops a TIN from bridging a gap.

**Circles and poles.** Kåsa fit, which is linear least squares:

```python
A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
b = np.sum(xy * xy, axis=1)
cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
radius = np.sqrt(c + cx * cx + cy * cy)
```

`lstsq` rather than `inv(A.T @ A)`, because the normal equations square the condition number.

A perfect arc from −0.3 to +0.3 rad on the circle of radius 2 centered at the origin recovers center `(0, 0)` and radius `2`. The centroid of those points is about `(1.97, 0)`, almost on the arc. With 1 cm of Gaussian noise on that same arc, one draw of the algebraic fit moved the center to about `(0.17, 0.01)` and the radius to about `1.83`. So: the mean is the wrong center even with perfect data, and a short noisy arc makes the algebraic fit the wrong center too. A scanned pole is that situation. The sentence to say is that you would robustly fit the circle, or fit a cylinder in 3D, and you would not trust the centroid of a one-sided arc.

## NumPy that they are actually scoring

- Broadcasting. `A[:, None, :]` and `B[None, :, :]`. Know the resulting shape before you run it.
- `axis`. `mean(axis=0)` collapses points. `norm(..., axis=-1)` collapses coordinates. `keepdims=True` when you will broadcast the result back.
- Masks and integer indexing. `points[mask]`, `points[idx]`, and the difference between them.
- `np.unique(..., axis=0, return_inverse=True, return_counts=True)`.
- `np.add.at`, `np.minimum.at`. Fancy assignment does not accumulate.
- `np.argmax`, `np.argmin`, `np.argsort`, `np.searchsorted`, `np.where`, `np.clip`.
- `np.cross`, `np.linalg.norm`, `svd`, `eigh`, `lstsq`, `solve`.
- `@` for matrix products.
- `np.einsum("ni,nj->ij", X, X)` is the covariance contraction. The `@` form is easier to read; either is fine.
- Empty inputs. Decide the shape you return, `(0, 3)` rather than a crash, if the prompt allows an empty cloud.

`np.linalg.solve` beats `inv` when you are solving a system. `lstsq` is the overdetermined version. Inverting `K` once for a back-projection is a reasonable exception; say why.

## Gotchas worth reciting

- Shape convention, stated first. `(N, 3)` versus `(3, N)` causes the rest of the bugs.
- `arccos` of a cosine slightly outside `[-1, 1]` is NaN. Clip.
- Zero-length axis, segment, or normal. Guard the division.
- Parallel ray/plane: denominator near 0. Hit behind the origin: `t < 0`.
- `det(R) = -1` is a reflection. Kabsch has to notice.
- Cross-product sign and winding. CCW vertices, right-hand normal.
- Rotation order. `R_second @ R_first` applies first, then second, on column vectors.
- `floor` versus cast-to-int for negative voxel keys.
- `add.at` / `minimum.at` versus `array[idx] +=`.
- Pairwise distances are `O(NM)` memory. Give a concrete size.
- Vertical planes, and the failure of `z = ax + by + c`.
- Station is XY. Z does not advance station. A purely vertical segment has horizontal length 0.
- Offset sign. Right of increasing station is `(ty, -tx)`.
- At a polyline kink, the tangent is ambiguous. Pick outgoing or incoming and say so.
- Eigenvector signs are arbitrary.
- Float equality. Use a tolerance, and know whether it is numerical or physical.
- Boundary of a polygon. Even-odd does not promise a stable answer on an edge.
- Short arcs. The centroid is biased toward the scanner. Say so before writing a pole center.
- A TIN query does not rebuild the mesh. Breaklines were constraints on the triangles you were given.

## In the interview

Clarify shapes, units, Z-up, and whether degenerate inputs matter. Then write a correct version you can check on three points, and vectorize it. A loop that is right, followed by the broadcast you would use, beats a clever expression you cannot evaluate.

Narrate the geometry in the same sentence as the code. "The normal is the cross product of the two edges. The signed distance is the dot with that normal." "Station is horizontal, so I difference XY and leave Z for the section origin."

When you finish a function, give the complexity and the production substitute. That is the natural place to connect the answer to a corridor-scale cloud without turning the interview into a product pitch.

Check one tiny case by hand before you say you are done. The `+90°` rotation of `(1, 0, 0)`, the plane `x = 0`, the eastbound road with a point to the north, the voxel at `-0.1`. Those four catch the bugs this style of interview is designed around.
