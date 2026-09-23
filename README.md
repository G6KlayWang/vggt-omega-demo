# Tiny VGGT-Ω classroom code

A from-scratch implementation of **register attention** on a toy geometry task,
plus an optional adapter for the official pretrained VGGT-Ω model.

## Sync the laptop and GPU server

This project has its own Git repository. The official model is a pinned Git
submodule at `upstream/`; the sample images are tracked, while model weights,
environments, generated results, and ZIP archives are excluded.

On the server, authenticate to GitHub with an account that can access this private
repository, then clone it:

```bash
git clone --recurse-submodules https://github.com/G6KlayWang/vggt-omega-demo.git
cd vggt-omega-demo
python -m venv .venv
source .venv/bin/activate
# Install a PyTorch build matching the server's CUDA runtime, then:
pip install -r requirements-pretrained.txt
```

Copy your existing checkpoint to the server separately, or pass its existing
server path to `pretrained.py --checkpoint`. Git will not transfer the weights.

Before editing on either machine:

```bash
git pull --ff-only
git submodule update --init --recursive
```

After making changes on that machine:

```bash
git add core.py train.py  # choose the files you changed
git commit -m "Describe the change"
git push
```

Pull on the other machine to receive those commits. Git does not automatically
sync uncommitted files. Keep concurrent work on separate branches (`git switch -c
experiment-name`, then `git push -u origin experiment-name`) and merge through a
pull request. If `pull --ff-only` reports diverging histories, reconcile the commits
before continuing instead of overwriting one machine's work.

The submodule stays at the recorded upstream revision until deliberately updated;
do not run `git submodule update --remote` for routine synchronization. Local GPU
results stay on the server unless copied separately.

## Run the original algorithm experiment

Use Python 3.10+ and install a PyTorch build compatible with your GPU/CUDA runtime.
Then, from this directory:

```bash
python -m pip install -r requirements.txt
python train.py --device cuda --steps 2500
python -m unittest discover -s tests -v
```

`--device cpu` also works. This tiny model can be faster on CPU because the batches
and attention matrices are small. No pretrained weights or downloaded dataset are
needed for this experiment. `toy_data.py` generates all observations and labels.

Read **`core.py`, especially `RegisterStage.forward`**, for the main contribution.
It implements local attention, register-only exchange across views, and subsequent
local redistribution using PyTorch primitives. It imports no upstream model code.

Three separately trained models use the same initialization seed, generated training
stream, training steps, and architecture dimensions:

- `local`: frames cannot exchange information.
- `register`: only four learned scene tokens per view exchange information.
- `global`: all tokens exchange information.

We also bypass the register model's exchange blocks after training (`cut`) and
compare with analytic least-squares triangulation (`oracle`).

Outputs: `outputs/results.json`, three small `toy_*.pt` checkpoints, and an optional
self-contained `outputs/demo.html` viewer. The JSON contains metrics, training
curves, seeds, and example predictions. The HTML uses cached predictions.

## Toy problem and what it demonstrates

Each scene is a 4×4 relief with depth

```text
z(x,y) = a + b*x + c*y + d*sin(pi*x)*cos(pi*y)
a,b,c,d ~ Uniform(-0.7, 0.7)
u = cos(theta)*x + sin(theta)*z; v = y
theta = [0°, -45°, +45°]
```

Independent Gaussian observation noise has standard deviation 0.015. The model
receives matched 2D landmarks plus known camera sin/cos values and predicts the
reference view's 16 signed z coordinates. Signed z is relative relief height,
not positive perspective camera depth. The reference image has no depth signal;
cross-view communication is necessary. Landmark identity embeddings provide known
correspondences. Train on 2,500 batches of 32 fresh procedural shapes and evaluate
on 512 separately seeded shapes. The demo uses another 12 shapes.

An observed CPU run (PyTorch 2.10, single model seed):

| Method | Held-out depth RMSE |
|---|---:|
| Isolated views | 0.5668 |
| Register attention | 0.0230 |
| Full global attention | 0.0270 |
| Register model, communication cut | 0.6098 |
| Analytic triangulation | 0.0149 |

These illustrate information flow, not a claim that register attention outperforms
global attention generally. GPU results may differ. The scene family is deliberately
low-dimensional; cameras and correspondences are supplied, and this is not RGB
reconstruction or a reproduction of the paper's benchmarks. Unlike the paper's
mixed architecture, this toy replaces every cross-view block with register exchange.
The toy has no camera token or camera prediction head. All variants allocate 96,145
parameters, but the local baseline leaves exchange parameters unused.

For three views, 16 landmarks, four registers, and two stages, the theoretical
attention-score count is 3,088 vs. 10,000 per head/sample, including local blocks.
This is not a measured memory or speed ratio. Register exchange is still quadratic
in the number of views.

## Run your downloaded checkpoint on the included real sample

`data/sample/` contains three overlapping frames from the official repository's
`examples/forest_road.mp4`. This bonus path uses the official backbone, then our
own `geometry.py` to unproject predicted depth into world-space points.

```bash
git clone https://github.com/facebookresearch/vggt-omega.git upstream
git -C upstream checkout a3ab0141f96838724423541044ff5ba301cfd36a
python -m pip install -r requirements-pretrained.txt
python pretrained.py --device cuda --resolution 512 --checkpoint /path/to/vggt_omega_1b_512.pt
```

If cloned with submodules, skip cloning and checking out upstream again. Otherwise,
if `upstream/` already exists, skip cloning. The checkpoint is deliberately excluded
from the code archive. Default inputs are the three bundled JPEGs. To use your own:

```bash
python pretrained.py --device cuda --checkpoint /path/to/model.pt --images image1.jpg image2.jpg image3.jpg
```

Add `--filter-white-bg` or `--filter-black-bg` (or both) to remove near-white
or near-black points from the PLY and offline viewer. White means all RGB
channels exceed 240; black means their sum is below 16, on a 0–255 scale.
These color filters also remove matching object colors. They run after inference,
before confidence filtering; raw arrays in `predictions.npz` remain unchanged.

Add `--filter-sky` to remove sky points from the PLY and viewer with the sky
segmentation model used by the official demo. Install `onnxruntime` first; the
initial run downloads its weights to `~/.cache/vggt-demo/skyseg.onnx`. For a server
without download access, copy the weights there or pass `--sky-model /path/to/skyseg.onnx`.
The model is hosted at <https://huggingface.co/JianyuanWang/skyseg>.
Add `--show-cameras` to draw numbered camera frustums and a dashed trajectory in
the HTML viewer, with a checkbox to hide them. Camera geometry uses the same
centering and scale as the point cloud; the PLY contains only scene points.

The HTML viewer exports up to 100,000 points by default. Use
`--max-viewer-points 200000` for a denser view, or lower it for faster browser
interaction. The confidence slider initially shows approximately half of these
points; move it toward 100% to show more, including lower-confidence predictions.
Rerun the command to regenerate the viewer after changing this limit. This option
does not change inference resolution or the PLY export.

Outputs are under `outputs/real/`: raw depth, confidence, RGB, and camera arrays in
`predictions.npz`; `reconstruction.ply`; a run manifest; and an optional offline
point-cloud viewer. PLY retains the top 50% of confidence scores. Confidence is not
a calibrated probability, and the sample has no ground truth. Use `--resolution
256` if memory is tight; the 512 checkpoint is trained at higher resolution. The
adapter was exercised locally on Apple MPS; the CUDA path is included but has not
been exercised on this machine.

## Video-input demo

Use the bundled forest-road video as a data source for a separate demo:

```bash
python pretrained.py --video upstream/examples/forest_road.mp4 --frames 8 --resolution 512 --device auto --out outputs/video
```

This samples eight distinct frames evenly across the entire video, including the
first and last frames, runs the pretrained
model jointly on them, and exports `outputs/video/demo.html`. Open that file to
play the source video and explore the static 3D reconstruction. This is sampled
video reconstruction, not real-time video processing. The manifest records the
video path, frame indices, and timestamps. Keep `source.mp4` beside `demo.html`
when sharing the viewer, or share the entire `outputs/video/` folder.

`--frames` sets the total number of samples, not frames per second. Increase it
for longer clips to preserve overlap between adjacent views; more frames use
more GPU memory. Request no more frames than the video contains.

Replace the video path with your own clip to use another data source. Use a slow
camera movement through a mostly static scene with overlapping views. Reduce
`--frames` or use `--resolution 256` if memory is tight. The existing image demos
remain in their own output folders. Bundled videos come from the pinned upstream
repository and remain subject to its license.

## Files to share and explain

| File | Purpose |
|---|---|
| `core.py` | Attention primitives and register routing, written from scratch |
| `toy_data.py` | Procedural geometry, camera projection, analytic baseline |
| `train.py` | Fair comparison, intervention, evaluation, small checkpoints |
| `geometry.py` | Independent pinhole depth-to-world unprojection |
| `pretrained.py` | Optional official-backbone adapter |
| `tests/test_core.py` | Routing, gradient isolation, permutation, geometry tests |
| `data/sample/` | Three real input frames and provenance |

## Attribution

Inspired by Wang et al., [VGGT-Ω, §3.1.2](https://arxiv.org/html/2605.15195v1#S3.SS1.SSS2).
The official [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega)
repository is pinned above. Its model, image preprocessing, camera decoding,
weights, and video sample are credited to the authors. The paper describes
register-only exchange; the released implementation includes its camera token
along with registers in those blocks. See `upstream/vggt_omega/models/aggregator.py`.

`core.py` is an independent educational implementation, not copied upstream code.
Real sample images and the official model remain subject to upstream terms; the
upstream license is included in `data/sample/LICENSE.upstream`.
