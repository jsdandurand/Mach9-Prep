# Adjacent system designs

[onsite_prep.md](onsite_prep.md) is the main product: a lidar scan goes in, curb and breakline polylines come out, a person edits them. That is the whiteboard to learn first.

The team also does work around that product. The plausible neighboring prompts are:

- hold geometric error under 5 cm on a large, messy dataset
- suggest the next part of a line while a person is drawing
- segment and extract many feature classes, not one
- read a traffic sign in a photo and return its standard code plus the text on it
- train and run these models across a corpus that does not fit on one machine

This note teaches those four designs and the concepts they add: an error budget, a hierarchy big enough to see a tree, a transformer used as the mixer inside that hierarchy, conditioning on a click, retrieval-augmented generation, and distributed training. The tiling, the station sort, the KD-tree, and the bird's-eye raster are assumed from the first guide.

## Which prompt is which

| If they ask | You design |
| --- | --- |
| How do you get under 5 cm? | The error budget and the evaluation, not a bigger network |
| The user is drawing and wants a suggestion | The interactive model |
| Extract curbs, paint, poles, trees, and signs | One backbone, a vectorizer per class |
| Identify the sign and pull structured fields off it | Detection in 3D, retrieval over the sign manual, a vision-language model for the legend |
| How does training run on all of this? | Tiles in object storage, one GPU process per device, array jobs at inference |

## Holding sub-5 cm

Sub-5 cm is a property of the delivered vertex against a surveyed reference. It is not a loss value and not a cell size. A network can have a beautiful training loss and still miss by 15 cm because it drew the top of the curb when the label was the flowline, or because the trajectory itself was 10 cm off.

### Where the centimeters go

Think of the error as a stack. Each row is independent enough that the total is worse than the worst row, and any single row above 5 cm already sinks the claim.

| Source | What it is | A scale worth saying |
| --- | --- | --- |
| Trajectory | GNSS plus inertial pose of the vehicle. Open sky is centimeters. Under trees or between buildings it wanders by decimeters until something pulls it back. | Often the largest term |
| Time sync | A return is stamped and then placed with the pose at that time. At 20 m/s, 1 ms of clock error shifts the point 2 cm along the road. | A few centimeters if the clocks are sloppy |
| Boresight | The rotation between the inertial unit and the laser is never exactly the CAD drawing of the roof rack. An angular error of 0.1° is about 3.5 cm at 20 m range, because `20 m × 0.00175 rad ≈ 3.5 cm`. | Grows with range |
| Lever arm | The translation between those same sensors. A constant bias. | Millimeters to a couple of centimeters if it was measured |
| Ranging | The laser's own distance noise. | Around a centimeter on a good unit |
| Calibration drift | The rack gets bumped. Last month's boresight is no longer true. | Shows up as a disagreement between Monday's pass and Tuesday's |
| Algorithm | Snapping a vertex to a 5 cm cell center. Simplifying a polyline with too large a tolerance. Picking the wrong edge of a 15 cm curb. | This is the part the model owns |

The arithmetic on the boresight row is the one to be able to do on the board. A small angle in radians times the range is the lateral miss. Degrees have to be converted before that multiply.

Two consequences fall out of the table.

The 5 cm bird's-eye cell is already as big as the entire budget. The network may predict "this cell," but the vertex you write has to be fit to the points inside the cell. Reporting the cell center spends the whole tolerance on quantization.

Choosing the wrong curb edge spends it too. Flowline and top of curb are about 15 cm apart vertically and a few centimeters apart in plan. The label definition is part of the accuracy system. You ask which edge, you store it on the polyline, and you score that edge.

### What actually pulls the trajectory back

The model cannot invent a better satellite solution. The mapping system does, upstream of inference:

- Post-process the trajectory with the full GNSS log rather than the real-time position the van saw while driving.
- Drive the corridor twice, or overlap tiles, and run a strip adjustment: the same curb or the same building face was scanned twice, and those two copies must coincide. That is a cloud-to-cloud registration, Kabsch or ICP on the overlap, solved for a smooth correction along the trajectory rather than one rigid jump.
- Hold a few surveyed control points, nails in the ground with known coordinates, and constrain the adjustment to pass through them.

You describe this as a preprocess the network is allowed to assume, and you say what residual you still expect. If two passes of the same curb disagree by 8 cm after adjustment, no architecture lands the line inside 5 cm of both of them. The honest output is a lower confidence, or a vertex at the mean with the disagreement recorded.

### Camera and laser have to agree

Imagery is in the stack because color and text are not in the point. The photo does not improve XYZ if the projection is wrong, and it makes a measured point worse if you let the image vote on position.

The camera has its own boresight and its own time stamp, on the same trajectory. Project a lidar point into the image with the pose at that point's time. If intensity edges (paint, the curb shadow line, a sign face) do not land on the image edges, solve for a small correction to the camera rotation until they do. Recheck it when the rig is remounted. A degree of camera error at typical ranges throws a paint label onto the wrong points, which then fails the 5 cm score even though the laser measurement was fine.

Position and elevation of every delivered object come from the lidar points, after the trajectory adjustment. The image answers what the object is.

### How you prove the number

Report three different measurements, because they answer three different worries.

**Absolute accuracy.** Surveyed checkpoints and a manually accepted drawing on a project the model never trained on. For a line: completeness and correctness at 5 cm, plus vertical error separately, as in the main guide. For a surface: elevation error at the checkpoints. For a sign or a pole: horizontal error of the center. The split is by project. A tile from a training city does not count.

**Relative accuracy.** Width between the two curbs, distance from a stop bar to the stop sign, compared with a tape or with the drawing. A whole block can be shifted 8 cm by GNSS and still have perfect relative geometry. Design work sometimes cares about the relative number more than the state-plane coordinate. Say which one you were asked for.

**Repeatability.** Extract the same curb from two independent passes. The distance between the two polylines is precision. It does not tell you that either one is on the true curb. It tells you whether the sensor plus the algorithm is stable. A claim of sub-5 cm that only quotes repeatability is incomplete.

Slice the report. Open highway will pass and urban canyon will not, and the average will hide that. Also slice by range: a curb 5 m from the vehicle and a curb 25 m away do not share a boresight error.

### What you would say

"Five centimeters is a residual against survey control, after the trajectory is adjusted. I budget pose, timing, boresight, ranging, and the algorithm separately. The network picks the cell and the edge. The vertex is fit to the points, so a 5 cm raster does not quantize the answer. Images classify. They do not move the point. I publish absolute error, relative error, and pass-to-pass repeatability on a held-out project, sliced by how obstructed the sky was."

## Interactive suggestions

The batch system draws the line overnight. The interactive system is what runs when a person is already holding the mouse. They click a start, sketch a meter of curb, or reject a bad vertex, and they want a suggestion before the gesture feels finished.

This is a different model in one respect and the same system in every other. The output is still a polyline. The metric is still centimeters, plus whether they accepted it. The new piece is that the user's mark is an input, and the latency budget is about a tenth of a second, not a nightly job.

### What the user might do

Treat these as different prompts into one model, not as different products.

- A click on the cloud: "the object I want is here."
- A short stroke of a few vertices: "follow this, and keep going."
- A correction: they dragged one vertex. The rest of the suggestion should be recomputed to meet the new vertex, not left stale.
- A rejection: they deleted the suggestion. That is a label, not just a UI event.

A click has a 3D position because the screen ray is intersected with the cloud or with the ground surface. Store the point, not the pixel.

### Conditioning

Run the expensive encoder ahead of time. When the tile is opened, load the points, build the KD-tree, rasterize the bird's-eye channels, and save the encoder's feature map. The click only runs a light decoder and the vectorizer.

The click has to be visible to that decoder. Two equivalent ways:

**Extra raster channels.** Paint a small disk at the click, and paint the distance from each cell to the user's stroke. The decoder reads the cached feature map plus these channels. A stroke the user drew along the gutter pulls the probability ridge onto that stroke. This fits the bird's-eye model from the main guide with almost no new architecture.

**Extra tokens.** If the backbone is a transformer, embed each user vertex as a token with a type flag that means "constraint," and let the local attention see those tokens alongside the point tokens. A constraint token carries relative XYZ and a flag for click versus sketched vertex versus rejected vertex. Attention is what mixes "the user insisted on this point" into the nearby curb points.

In both cases the suggestion is forced to honor the input. The training loss includes the usual line distance to the accepted drawing, and a second term that penalizes a suggestion which misses the click by more than a few centimeters. Without that term the model learns the average curb and treats the user as noise.

At inference the constraint is hard. If they clicked a pole, complete a pole. Do not snap to the nearest curb because the curb head is more confident. The click chooses the object.

### Next vertex, when the gesture is a drawing

There is a narrower version that matches "I am clicking along a curb." Represent the line as a sequence of vertices in station order. The model sees the last few accepted vertices and the local patch ahead, and it predicts the next offset, the next elevation, and a probability that the line ends. The user taps to accept, and that vertex becomes input for the following step.

Train it on finished polylines. A random prefix of the finished line is the prompt. The rest is the target. Augment the prefix with a centimeter of jitter so the model survives a sloppy click. Stop tokens matter: a driveway is a place the line should end rather than bridge.

This version is easy to explain and easy to make fast. It is a poor fit for "clean up this whole tile," which is the batch model. Offer it when the product is a drawing tool.

### Latency

A tenth of a second is the budget if the suggestion is going to feel attached to the gesture. The encoder is already done. The decoder sees one tile, not the city. The vectorizer is the station-ordered linker from the main guide, restricted to a window of maybe 30 m ahead of the click so it cannot run off and redraw the block.

Cache invalidation is part of the design. If a new trajectory adjustment moves the cloud by 4 cm, the cached features are wrong and have to be rebuilt. The suggestion should not come from yesterday's raster.

### Labels from the editor

The accepted line is the target. You do not need a separate annotation program. Log the prompt (click or prefix), the suggestion, and the final geometry after the person stopped editing.

A rejection is more informative than an acceptance. Mine the suggestions they deleted and the vertices they moved by more than 5 cm, and oversample those in training. The batch model then improves from the same log. The interactive model and the overnight model share a backbone and differ in the decoder and the latency.

### Metrics

- Did the suggestion pass through the prompt, within a few centimeters?
- Fraction of suggestions accepted with no further edit, and the centimeters of edit when they did edit.
- Time to finish a tile, against the same person drawing with no suggestion. This is the product number, and it needs a study with users. Offline, the proxy is centimeters of remaining edit.
- Latency at the 95th percentile, measured with the cache warm and with the cache cold. Quote both.
- Failure slices: click on the wrong object, driveway, intersection, a stroke that disagrees with the lidar.

### What you would say

"The encoder runs when the tile opens. A click or a stroke is an extra channel or an extra token, and the loss punishes suggestions that ignore it. I only decode a short window ahead of the user, then run the same linker. Labels are the lines they eventually saved, with the rejections oversampled. I report acceptance, remaining edit in centimeters, and latency with a warm cache."

## Many features from one scan

A curb model is one head. A mapping product also wants pavement edges, lane paint, poles, signs, trees, and the ground surface. The whiteboard version is one shared trunk and a separate vectorizer per class. You do not train five unrelated pipelines and hope the curb and the ground agree.

### Semantic labels and instances

A semantic head answers "what is this point." An instance head answers "which object does it belong to."

Vegetation is the example that makes the difference obvious. Leaves in a small ball already look like vegetation: scattered points, no stable plane, high sphericity. A local model can paint those points "tree" without seeing the rest of the plant. That paint is a semantic mask. Two crowns that touch become one blob in that mask. Splitting the blob into trees is instance grouping: connected components on the vegetation points with a gap threshold, or vertical columns through the canopy. The trunk is included because it touches the crown, which is why the trunk needs a feature from a scale big enough to see the crown. Otherwise a trunk and a light pole are the same local shape, a vertical cylinder.

### The hierarchy that sees the crown

A single 0.75 m neighborhood cannot do that. PointNet++ and a hierarchical transformer are the same diagram.

At the fine level, tokens are 5 cm cells. Around each one, gather a small neighborhood and mix it. In PointNet++ the mixer is a shared MLP and a max-pool. In the transformer it is attention over those neighbors, with relative XYZ added to the embeddings so the weights know the geometry. The embedding of a token is an MLP over its raw features: relative position, intensity, color if a camera painted it, height above the ground, and optionally the handcrafted shape numbers.

Pool to a coarser grid, around 0.4 m, then around a couple of meters. A coarse token summarizes every fine token in its cell, so one token can cover a trunk and a piece of canopy together. The way back down concatenates the coarse feature onto the fine tokens and mixes again. A point on the bark is then classified from its own neighbors and from the coarse token that saw leaves above it.

Stacking local attention without the coarse grid also grows the receptive field, about one window per layer. Crossing a 10 m crown that way costs a lot of depth. The coarse grid crosses it in a few stages. Curbs do not need those stages. Trees and "is this whole object one sign panel" do.

### One trunk, several heads

The shared trunk is that hierarchy, trained on every class you have labels for. Heads branch at the fine level:

| Head | Output | Turned into |
| --- | --- | --- |
| Ground | probability per point or cell | the TIN, after breaklines are burned in |
| Curb / edge of pavement | probability per cell | polylines, Z from the step |
| Paint | probability per cell | polylines on the pavement, Z from the ground surface |
| Pole | probability per column | a center and a radius from a circle fit |
| Tree | semantic mask, then a grouping | a trunk line and a crown footprint |
| Sign panel | a cluster id | the 3D plaque that the sign system in the next section reads |

Separate models are easier to debug and worse at the boundaries. The curb is the breakline the ground head must not cross. Paint lies on ground, not on a car. A sign is attached to a pole. A shared trunk sees those relationships. You can still ship the heads on different days: freeze the trunk, train a new head, and keep the vectorizers independent so a regression in trees does not move the curbs.

The long tail is real. Stop signs are common and a rare warning plaque is not. Sample by class, and accept that the rare class may be a geometric detector plus a human rather than a fully learned head. Say that instead of promising one model for every layer in the CAD spec.

### What you would say

"One hierarchical backbone, bird's-eye or sparse 3D, with attention or a PointNet mixer inside each neighborhood. Coarse levels exist so a trunk can see its crown. Each class has its own head and its own vectorizer, because a polyline, a circle, and a crown are not the same object. The ground head consumes the curb polylines as breaklines. I would rather ship a geometric detector for a rare class than pretend the head saw enough examples."

## Sign codes and the legend

Pavement markings were covered with color in the main guide. A traffic sign is a different deliverable. The position comes from lidar. The identity comes from the picture. The identity is not a free-form caption. In the United States it is a code in the Manual on Uniform Traffic Control Devices, plus the variable text on that particular panel.

### What a code is

MUTCD assigns an alphanumeric code to a standard sign face.

- Regulatory signs are the R series. R1-1 is STOP. R1-2 is YIELD. R2-1 is the speed limit plaque. R5-1 is DO NOT ENTER.
- Warning signs are the W series, usually a yellow diamond. A turn warning and a pedestrian warning have different codes and different symbols.
- Guide signs carry destinations and route numbers. The layout is standard and the words are not.

The code does not always determine the legend. Every R2-1 has the same layout, and the number in the middle changes. A guide sign's code does not tell you the destination. "Structured information" means a record a CAD system and a traffic engineer can store:

```text
code: R2-1
edition: the MUTCD edition you pinned
legend: speed_mph = 35
xyz, size, facing
confidence
which plate in the manual you matched
```

Editions matter. Codes get added and renumbered. States publish supplements. The corpus is pinned to an edition the way a model is pinned to a checkpoint. A code from the wrong edition is a wrong answer that looks official.

A closed-set classifier over the hundred most common codes is a strong baseline and an incomplete product. It cannot read "35" versus "45," it cannot emit a destination string, and it has no honest behavior on a plaque it has never seen. That is the gap a retriever and a vision-language model fill.

### Find the panel before you read it

Running a large model on every camera frame is wasteful, and it throws away the geometry you already have.

A sign in the cloud is a small, roughly vertical plane, often under a couple of square meters, with high intensity because the face is retroreflective, sitting on a pole at a typical mounting height, facing the lanes. Cluster those points, fit the plane, and record the centroid, the width and height, and the normal. The normal is the facing. The centroid is the position you will deliver. This stage is the geometric baseline, and it is allowed to be recall-heavy: a false panel is cheap if the next stage rejects it, and a missed panel never gets a code.

Project the cluster into the cameras that see it. The projection uses the same pose and intrinsics as any other camera in this prep. Expand the box by a fixed margin. That crop is the image the reader sees. If the lidar barely grazed the sign, fall back to a 2D detector on the photo and associate a detection to a cluster when the ray agrees with the plane. Discard a detection whose ray hits a car or a building instead of the panel.

### Retrieval, then a reader

Retrieval-augmented generation means the reader is not allowed to answer from memory alone. You fetch the relevant pages of the manual and put them in the prompt.

Build the corpus once per edition. Each entry is a code, a rendered picture of the standard plate, a short text description (shape, color, symbol, whether the legend is fixed or variable), and a schema for the fields. Embed the plate image and the description with a vision-language encoder of the CLIP type: the same embedding space for pictures and sentences, trained so a photo of a stop sign lands near the words "R1-1, octagon, red, STOP." Store the vectors in a flat index. The corpus is a few hundred to a few thousand plates, not the open web. Exact nearest neighbor is fine at that size.

At query time, embed the crop and take the top five plates by cosine similarity. Also keep the margin between first and second. A wide margin on a fixed-legend sign (a STOP with nothing to read) can be accepted from retrieval alone, with no generative model. A narrow margin, or any plate whose legend is variable, goes to the reader.

The reader is a vision-language model. Its prompt is the crop, the five retrieved plates with their images and descriptions, and an instruction to return JSON matching the schema. The instruction says the code must be one of the five, or `unknown`. That constraint is the entire point of the retrieval. A model answering from memory will invent a plausible code. A model answering from five pages can still pick the wrong page, and it cannot pick a page you did not give it.

Validate the JSON after the fact. R2-1 requires an integer speed in a legal range. R1-1 should not contain a speed. A guide sign requires a destination string. A code that is not in the retrieved set is discarded. A failed validation goes to a human queue along with the crop and the five plates, which is a reviewable object rather than a silent guess.

For the common variable fields, a specialist beats a general reader. Digits on a speed-limit plaque are an OCR problem. Use OCR there, and reserve the vision-language model for the long tail and for signs whose symbol has to be matched to a plate. That split is also a cost control: the expensive model only runs on crops the geometric detector already proposed, and only when retrieval was unsure or the legend is free text.

### Training and evaluation

You can fine-tune the embedder on pairs of real crops and the correct plate, using the same contrastive setup. You do not need to fine-tune the reader on day one. If you do, keep the retrieved pages in the training prompt, or it will learn to ignore them and go back to guessing.

Score these separately:

- Panel localization in centimeters, against the surveyed or hand-clicked sign position. This measures the lidar stage.
- Code exact-match, on a held-out set of roads.
- Legend accuracy: the speed, the destination, the arrow direction.
- Abstention: fraction of `unknown` on signs whose code really is outside the corpus, and the rate at which a wrong code was emitted instead of `unknown`. The second number is the one that hurts a customer.
- Retrieval recall at five: how often the true plate was even in the prompt. If this is low, the reader cannot save you. Fix the embedder or the corpus before you touch the reader.

Slice by night, glare, occlusion, and the back of the sign. The back of a sign is a plane in lidar and a blank face in the photo. Return a position with no code rather than reading through the panel.

### What you would say

"Lidar finds the panel and supplies XYZ, size, and facing. I project it to a crop. I embed the crop and retrieve the five closest plates from a pinned edition of the manual. Fixed-legend signs with a wide margin are done. Everything else goes to a vision-language model that must pick one of those five codes or abstain, then a schema check on the legend. Speed-limit digits can be OCR. I report localization, code accuracy, legend accuracy, and how often the right plate was in the five."

## Distributed training and inference

The models above are small next to the data. A corridor is tens of millions of points. A training corpus is thousands of corridors. The design question is how a PyTorch training job and a full-corpus inference job run on that without ever putting the corpus on one disk.

### The sample is a tile

The object in storage is still a tiled cloud, with the label polylines and the trajectory beside it. A manifest lists every tile: object-storage URI, project id, city, sensor, and which split it belongs to. The split column is assigned by project, once, and then respected by every job. A random shuffle of the manifest is how leakage sneaks back in.

A training sample is one tile, already rasterized or already cut into seeds, not a raw file the GPU has to parse. Build those tensors in a CPU preprocess and write them next to the tile. The training job then streams fixed-size arrays. A bird's-eye tensor of a few hundred channels by 1000 by 600 is a normal batch. A variable-length point cloud is an awkward batch: sample or pad each tile to a fixed number of seeds so the batch is a rectangular tensor. Padding is wasted compute, so prefer sampling a fixed count and weighting the loss so rare classes survive the sample.

### One process per GPU

PyTorch Lightning is the harness around the loop you would otherwise write by hand. You still define the model and the data. The trainer runs the epochs, the checkpoints, and the distribution strategy.

The strategy that matters is one process per GPU, each holding a full copy of the model and a different slice of the batch. After the backward pass, gradients are averaged across processes and every copy takes the same optimizer step. That average is an all-reduce. The effective batch size is the per-GPU batch times the number of GPUs times the number of accumulation steps.

Accumulation is how you get a large batch when the tile does not fit. Run four small forwards, add the gradients, then step. It is not the same as four GPUs. Four GPUs compute in parallel. Four accumulated steps compute in series and share one set of weights.

Precision follows the same split as the rest of this prep. The network runs in bfloat16 or float16. The metric, the polyline geometry, and any SVD you still do in the loop stay in float32 or float64. Mixed precision keeps the sensitive reductions in float32 automatically if you use the framework's autocast. Do not cast the centimeter target to float16 and then wonder why 5 cm is noise.

Log the line metric, not only the loss. Completeness is a ratio. Sum the numerator and the denominator across processes, then divide. Averaging the ratios of unequal tiles weights a 10 m alley the same as a kilometer of highway.

### The cluster

AWS Batch is a queue in front of GPU machines. You give it a job definition (container image, command, GPU count) and a queue attached to a compute environment (instance types, a maximum number of machines, spot or on-demand).

Training is one job with several GPUs, because the processes have to all-reduce with each other. Inference is the opposite. Tiles do not talk to each other. Submit an array job whose size is the number of shards of the manifest. Child `i` reads shard `i`, writes polylines and a done marker, and exits. A retry that sees the marker skips finished tiles. Spot instances are appropriate here and for training only if you checkpoint. A spot machine disappears. The checkpoint goes to object storage every few hundred steps, and the job definition starts from the latest checkpoint when it is resubmitted.

The GPU is often waiting on decompression and on reads from object storage, not on the matmul. Prefetch on CPU workers, stage the next tiles while the current batch runs, and overlap the KD-tree build for the following tile with the forward of this one. A profile that shows the GPU at 30% utilization is a data-loading problem. Making the transformer deeper does not fix it.

Dense tiles will OOM if you pretend every tile has the same point count. Cap the seeds, or skip the bird's-eye cells that are empty and use a sparse tensor. Record the cap so a tile that hit it is audited rather than silently missing curbs.

### What you would say

"Samples are tiles, split by project in the manifest. Lightning runs one process per GPU and all-reduces gradients. I pad or sample seeds to a fixed count so a batch stacks. Inference is an array job over shards, with a done marker so retries are safe, and the trajectory adjustment has already happened. I checkpoint to object storage because the machines are spot. I log completeness from summed lengths, not from an average of ratios."

## Order to study

1. The error-budget table, including the boresight multiply, until you can explain why a 5 cm cell center is already the whole budget.
2. The interactive path: warm encoder, click as a channel or a token, loss that forces the line through the click.
3. The hierarchy for a tree: fine neighborhood, coarse cell that sees the crown, skip connection back down. Attention is the mixer you can swap in at each level.
4. The sign pipeline: lidar panel, top-five plates, reader constrained to those plates, schema on the legend.
5. One pass over the training job versus the inference array, so the two do not get collapsed into "we use GPUs."

The curb whiteboard in the first guide is still the one to give if the prompt is open-ended. These are the answers when the prompt is one of the rows in the table.
