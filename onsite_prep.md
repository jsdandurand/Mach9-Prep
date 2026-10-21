# Onsite prep

The recruiter described four rounds:

- a whiteboard to design an ML system for a task
- an interview on using AI to solve problems
- behavioral interviews
- coding in Python, possibly PyTorch

The NumPy notebooks and the one-page formula sheet are the geometry. Leave the formula sheet as it is. This note is the design round, and the PyTorch notebooks are the coding round if they hand you tensors instead of arrays.

The design round is not a recommender. A Pinterest-style session is about users, items, candidate generation, ranking, and a metric like CTR or NDCG. Mach9's output is a line or a surface a surveyor will accept, measured in centimeters.

| | Recommendations | Mach9 |
| --- | --- | --- |
| Input | user, context, catalog | unordered lidar points, a trajectory, sometimes images |
| Output | a ranked list | a polyline, a circle, a TIN, a grid |
| Split | random users, or by time | by project, corridor, or city |
| Metric | CTR, NDCG, watch time | centimeters, length recovered, how far a person moves the line |
| Imbalance | most items are not clicked | almost every point is not a curb |
| Feedback | clicks | a surveyor accepts, moves, or deletes the line |
| Baseline | popularity, a linear ranker | a geometric rule you can explain |

Say that contrast once if they ask why you are not starting from a two-tower model. Then design the surveying system.

## Whiteboard: curb and breakline extraction

This is the flagship task. Curbs, pavement edges, and barriers become breaklines. A breakline is a constraint the TIN is not allowed to cross, because the curb is a break in slope. The product is the polyline in CAD, not a per-point label.

Talk through it in this order. Thirty-five minutes is enough if you keep the baseline in the conversation.

### 1. Clarify before you draw a model

Ask these, and write the answers on the board.

- What is the object? A 3D polyline, or a 2D line with a Z sampled from the cloud?
- What accuracy is "done"? A few centimeters. Five and ten centimeters are the bands worth naming. `1e-8` is a numerical tolerance, not a survey tolerance.
- Batch or interactive? An overnight pass over a corridor is a different system from a surveyor who wants the next vertex while they draw.
- What labels exist? Old CAD, a vendor's lines, or the edits from the last project?
- What does a miss cost versus a false curb? Deleting a wrong line is a click. Missing a curb means the surface interpolates across the gutter.

Z is up. Plan view is XY. Station is horizontal distance along the road. Offset is positive to the right of increasing station. Say that before any formula.

### 2. Data, tiles, and the split

Mobile lidar is a strip along a trajectory, with occlusions, double returns, and a density that falls off away from the vehicle.

Tile along the trajectory, with overlap, so a curb is not cut at a tile boundary and then scored as two objects. Keep the trajectory pose with the tile. A point's station and offset are the features that make "this is the right edge of the road" mean something.

Split by project or by geography. A random split of points leaks the same curb into train and test. A random split of tiles from one project still leaks the sensor, the city, and the drawing style. Hold out a corridor the model has not seen.

Curb points are a thin ribbon. Accuracy on all points is a useless number, because a model that says "not a curb" everywhere is nearly perfect. Report the metric on the ribbon, or on the polyline.

### 3. Baseline, before a network

You should be able to beat a rule, and fall back to it when the net is wrong.

- In a cross-section, a curb is a short, steep rise between two flatter surfaces. Sample offset and elevation, look for that step, and trace it along station.
- On a neighborhood, a curb face is planar with a horizontal normal. Ground is planar with a vertical normal. That is linearity, planarity, and `|n · up|` from the shape-feature problem.
- A vertical face is not `z = ax + by + c`. Fit `ax + by + cz = d` with SVD. The height-field fit is the bug they are hoping you name.

Draw the baseline on the board. If you skip it, the rest of the design has nothing to beat.

### 4. Model

Pick one primary path and name one alternative. Do not draw three architectures in detail.

**Primary: a local classifier, then a vectorizer.** For each point, or for a bird's-eye cell, predict "curb" from a neighborhood. Inputs that earn their keep: height above a local ground estimate, intensity, a normal or the three shape features, station, offset. A shared MLP on the neighborhood (a small PointNet: MLP per point, max-pool, linear head) is a credible model. A bird's-eye raster with height, density, and intensity channels, plus a 2D conv net, is the other credible model, and it is easier to tile.

**Alternative, if the product is interactive:** condition on the polyline so far and predict the next vertex. That matches a surveyor who has already clicked the start of the curb. It is a different training set: sequences of accepted vertices, not a point mask.

What you do not lead with: an end-to-end network that emits a mesh, a transformer over the whole corridor in one forward pass, or a recommender stack with user embeddings. The corridor does not fit in one forward pass, and there is no user.

Class imbalance: train on the ribbon plus a sampled background, or weight the loss. Say which. A curb point ten centimeters away is a different error from a curb point on a building, so a distance-weighted loss is reasonable. Point-to-polyline distance is the geometric version of that idea once you have lines.

### 5. From probabilities to a breakline

A mask is not deliverable.

- Threshold, drop isolated hits, and trace along the road rather than in an arbitrary image direction. Station order is the natural order.
- Fit segments. Simplify with Douglas–Peucker.
- Score a candidate by the distance from predicted curb points to the polyline, and by how much of the reference length it covers. Reject a line that jumps across a driveway or a gap you did not mean to bridge.
- The line becomes a breakline in the TIN. Triangles are not allowed to cross it. Query-time barycentric Z does not rebuild that mesh; it only reads it.

Elevation of the line matters. A plan-view curb with the wrong Z puts a step in the wrong place. Take Z from the ground or from the gutter, and say which edge of the curb you are drawing.

### 6. Metrics

Put these on the board, in this order.

- Length of predicted curb within 5 cm of the reference, and within 10 cm. Report completeness and correctness: reference length recovered, and predicted length that is not spurious.
- Endpoint and vertex error, in centimeters.
- How far a surveyor moves the line before accepting it. That is the product metric. A model with a pretty point-F1 and a 30 cm bias still fails.
- Elevation error along the line, separate from the plan-view error.
- Failure slices: intersections, driveways, parked cars, occluded far curb, a vertical face the height-field baseline dropped.

Do not offer point accuracy, Chamfer between raw clouds, or a leaderboard number with no centimeter attached.

### 7. The human loop

The product already assumes a person. They accept, move, or delete. Those edits are the next labels, and they are better labels than a synthetic curb, because they match how that shop draws.

Two serving paths:

- Overnight batch. Tile the corridor, write polylines, a person cleans them in the morning.
- Interactive. They click a start. You propose the next few meters. Latency is the constraint, so the model sees a neighborhood, not the project.

Say how you would ship the baseline first, log the edits, and only then train the net on the edits. That is the honest v1.

### 8. Close on purpose

End with what you would measure in the first offline test, what you would not trust (a random point split, a height-field plane, point accuracy), and which decision is still open (batch versus interactive). Interviewers remember the open decision more than a third architecture.

## If the task is poles

Same skeleton, different geometry.

A pole in a neighborhood is linear and vertical: high linearity, `|direction · up|` near 1. That is the baseline.

In plan view the returns are a partial arc. The scanner sees one side. The centroid of those points sits toward the scanner, not at the center. An algebraic circle fit (Kåsa, `lstsq`) is exact on a perfect arc and moves once a short arc has about a centimeter of noise. Say that limitation, then RANSAC or a geometric fit.

Output is a vertical centerline or a circle in CAD, not a point mask. Metric is center error and radius error in centimeters. A pole occluded by a car is the failure slice.

## If the task is ground and a TIN

Baseline: a min-Z grid, or a local planar fit with a vertical normal, with RANSAC so a car does not tilt the plane.

The model can classify ground points, or predict a residual against the grid. Drop triangles with an edge longer than a limit, or the TIN will bridge a gap with a sheet. Breaklines from the curb system are constraints, not post-processing.

Metric: elevation RMSE on held-out ground, in centimeters, plus a count of long triangles and a check that triangles do not cross breaklines. Split by project. A pretty surface that cuts the curb is wrong.

## Coding round, if it is PyTorch

The PyTorch notebooks in `notebooks/pytorch/` are the NumPy problems again. Do them. The API swaps that actually bite:

- `torch.set_default_dtype(torch.float64)` before geometry. The default is float32, and SVD and Kabsch get noisy there. Say that a training net is often float32 anyway.
- Neighbors: `torch.cdist`.
- Voxel keys: `torch.floor(...).to(torch.int64)`, then `torch.unique(..., dim=0, return_inverse=True)`, then `scatter_add_`. `sums[idx] += points` drops duplicate indices. Casting a float to int truncates toward zero and merges `-0.1` with `+0.1`.
- Min-Z grid: `scatter_reduce_(..., reduce='amin')`. An ordinary index write keeps one point and depends on order.
- `torch.linalg.svd` returns `Vh`, the same role as NumPy's `Vt`.
- `torch.linalg.lstsq(...).solution`.
- `torch.linalg.cross` does not broadcast `(N, 3)` against `(3,)`. Broadcast first.
- `eigh` eigenvalues come out ascending. Flip an eigenvector so its largest component is positive.
- A loss the check will differentiate: mean squared point-to-plane distance. Build it from ordinary ops. Do not detach the points, and do not update a tensor that requires grad in place.

Notebook 07 is the piece the NumPy set does not cover. Three problems:

1. The point-to-plane loss, and a `.backward()` that produces a finite gradient.
2. Handcrafted features under `torch.no_grad`, then a linear head and cross-entropy. `eigh` is not in the graph. The clouds are independent, so a random holdout is legal in the exercise and illegal on a real job. Say that when you hit the split.
3. A tiny PointNet: shared MLP, max-pool over points, linear head, Adam, one training loop. Max-pool is what makes the neighborhood unordered. This is the shape of an answer, not a model to ship.

If they ask for NumPy, the original notebooks are the same problems and the same hand checks.

## Using AI to solve a problem

This round is separate from the whiteboard. Listen to the first question and match it.

If they sit you in an editor with a model available, they are watching how you use it. You drive. The model drafts.

- State the convention in the prompt before you ask for code. Z up, points as `(N, 3)` rows, `p' = R @ p + t`, offset positive to the right. A model that guesses Y-up will look fluent and be wrong.
- Give it the hand check in the same message as the function. Then run the check yourself. Do not read the code and call it done.
- The checks that catch a fluent wrong answer: `det(R) = +1` and a `+90°` rotation about Z; a vertical plane `x = 0`, which a height field cannot fit; station and offset on the polyline `(0,0) → (10,0) → (10,10)`, where `(5, 2)` is station 5 and offset −2; a short noisy arc, where the centroid is not the center.
- If it writes Kabsch, check the reflection case. SVD will happily return `det = -1`.
- Narrate the one thing you are verifying. "I'm checking the sign of the offset, not the style."
- When the task is open-ended, let it sketch, then you choose the split and the centimeter metric. Do not paste a design and read it back.

If the question is about the product rather than the editor, it collapses into the whiteboard: where a model helps a surveyor, where a geometric rule should stay, and how you check the model against a hand example before you trust the line.

## Behavioral

They build a tool for domain experts, and a two-centimeter error is a real error. Have two stories you can tell with one number and one decision. These are the themes that fit, not a generic leadership list.

- A metric that looked good and measured the wrong thing. Point accuracy, a random split, a loss that ignored centimeters. What you changed the metric to, and what the new number did to the decision.
- A domain expert who drew the object differently than you had specified. You changed the output, or you held the spec and showed them why. Either is fine if you can say what evidence you used.
- An ambiguous spec. Which side is "right," which tangent you use at a corner, which edge of the curb gets the Z. How you made it explicit instead of picking silently.
- Something you shipped that a person still had to edit, and what you did with those edits.
- A bug that was a sign, a transpose, or a zero-length vector. What hand check you added so it could not come back.

Skip stories whose only point is that you worked hard.

## Order to prepare

1. Talk the curb design out loud once, in the order above, with the baseline and the 5 cm / 10 cm metrics on paper.
2. Do the short PyTorch list: notebook 01, nearest neighbors, voxel downsample, plane fit, station and offset, shape features, notebook 07.
3. Five minutes on poles, five minutes on the ground TIN, using the same skeleton: baseline, output, metric, split, failure slice.
4. The morning of the interview, the formula sheet. Print it portrait on letter paper at 100 percent. Fit-to-page and landscape will clip it.
5. Two behavioral stories, each with a number.
