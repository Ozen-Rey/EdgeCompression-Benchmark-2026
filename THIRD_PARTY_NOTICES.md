# Third-party notices

## Scope

This repository does not redistribute third-party datasets, pretrained model
weights, codec binaries, or full benchmark outputs. The R-D-E Router can
interoperate with such components when users install or provide them
separately.

The Apache-2.0 license of this repository applies only to original project
code, tests, configuration templates, setup scripts and documentation authored
for this repository, unless otherwise stated.

This repository-level licensing information is provided for clarity and does
not replace the upstream licenses or terms of third-party components.

## Python dependencies

The project metadata currently declares the following Python dependencies or
build/test tools. Licenses are not inferred unless they are explicit in local
package metadata or project metadata.

| Package | Purpose | License / terms |
|---|---|---|
| setuptools | Build backend declared in `pyproject.toml` | See upstream package metadata |
| pytest | Optional test dependency declared as the `test` and `dev` extras | See upstream package metadata |
| numpy, imagecodecs, matplotlib, ssimulacra2 | Declared by the optional `benchmark` extra for the image mini-benchmark | See upstream package metadata |
| imageio-ffmpeg | Declared by the optional `benchmark-hevc` extra; provides an ffmpeg binary so HEVC works without a system ffmpeg | The Python wrapper is permissively licensed, but the ffmpeg binary it installs is built with x265 and is therefore **GPL**. It is fetched by pip onto the user's machine (not redistributed here) and invoked as a separate subprocess (arm's-length), so it does not relicense this repository's code. |

Some benchmark scripts may import additional optional scientific, media,
machine-learning or hardware-measurement packages when users run benchmark
workflows. Those packages are not installed by the router setup script and
remain subject to their upstream licenses and terms.

## Optional external executables, codecs and quality tools

These components may be referenced by documentation, configuration templates,
benchmark scripts, execution adapters, or measured R-D-E rows. They are not
redistributed by this repository and are used only when installed or provided
by the user.

| Component | Role | Redistributed? | License / terms |
|---|---|---:|---|
| FFmpeg / FFprobe | Optional external media processing tool used by benchmark/execution workflows. May be a system install, or the ffmpeg binary provided by the optional `imageio-ffmpeg` pip wheel (`benchmark-hevc` extra) | No | See upstream FFmpeg project and build-specific licensing; the `imageio-ffmpeg` build includes x265 and is GPL |
| x264 | Optional external AVC/H.264 encoder | No | See upstream project |
| x265 | Optional external HEVC/H.265 encoder | No | See upstream project |
| SVT-AV1 | Optional external AV1 encoder | No | See upstream project |
| VVenC | Optional external VVC encoder | No | See upstream project |
| JPEG XL tools (`cjxl`, `djxl`) | Optional external JPEG XL encoder/decoder tools | No | See upstream project |
| Opus / `opusenc` / libopus | Optional external audio codec/tooling | No | See upstream project |
| VMAF | Optional video quality metric/tooling | No | See upstream project |
| ViSQOL | Optional audio quality metric/tooling | No | See upstream project |
| PESQ, STOI, SI-SDR and other audio metrics | Optional audio quality metrics used in benchmark analysis | No | See upstream projects and applicable terms |
| CUDA / NVIDIA tooling / NVML / `nvidia-smi` | Optional GPU/runtime/energy telemetry tooling | No | See NVIDIA upstream terms |
| RAPL / Zeus / pyNVML-style telemetry tooling | Optional energy measurement tooling | No | See upstream projects and platform terms |

Mentioning interoperability with a codec, executable, metric or hardware tool
does not imply endorsement, ownership, redistribution rights, or relicensing of
that component.

## Datasets

Dataset names in this repository identify external benchmark inputs or examples
for manifests and analysis. Datasets are not redistributed here. Users must
obtain them from their official sources and comply with their upstream terms.
Manifest files, examples, scripts or documentation in this repository do not
grant dataset rights.

| Dataset | Role | Redistributed? | License / terms |
|---|---|---:|---|
| Kodak image set | External image benchmark dataset referenced by docs/scripts | No | See upstream dataset/source terms |
| Tecnick | External image benchmark dataset referenced by docs/analysis | No | See upstream dataset/source terms |
| DIV2K | External image benchmark dataset referenced by docs/analysis | No | See upstream dataset/source terms |
| CLIC / CLIC2020 | External image benchmark dataset referenced by docs/analysis | No | See upstream dataset/source terms |
| UVG | External video benchmark dataset referenced by video analysis | No | See upstream dataset/source terms |
| LibriSpeech | External audio dataset referenced by benchmark scripts | No | See upstream dataset/source terms |

## Models and checkpoints

Model families and checkpoints may be referenced by benchmark scripts,
configuration templates, documentation, or measured R-D-E rows. This repository
does not redistribute pretrained weights or checkpoints. Users must obtain
models and checkpoints from their upstream projects under the applicable terms.

| Component | Role | Redistributed? | License / terms |
|---|---|---:|---|
| Balle learned image compression models | Referenced neural image codec family | No | See upstream project |
| Cheng learned image compression models | Referenced neural image codec family | No | See upstream project |
| ELIC | Referenced neural image codec family | No | See upstream project |
| TCM | Referenced neural image codec family | No | See upstream project |
| DCAE | Referenced neural image codec family | No | See upstream project |
| JPEG AI models or reference implementations | Referenced emerging image coding family | No | See upstream standards/project terms |
| DCVC-DC, DCVC-FM, DCVC-RT, DCVC-RT-CUDA | Referenced neural video codec families | No | See upstream projects |
| EnCodec | Referenced neural audio codec family | No | See upstream project |
| DAC | Referenced neural audio codec family | No | See upstream project |
| SNAC | Referenced neural audio codec family | No | See upstream project |
| WavTokenizer | Referenced neural audio codec family | No | See upstream project |

Adapters, declarative specs, router-ready CSV schemas, and documentation in
this repository do not grant rights to redistribute upstream model weights.

## Generated outputs

`results/`, `plots/`, `figures/` and benchmark CSV/JSON outputs are generated
locally and ignored by git in the public repository layout. Numerical results
may differ across machines because of hardware, drivers, codec builds, thermal
state, benchmark protocol details and energy backends.

Generated outputs are not covered by this repository's source-code licensing
unless explicitly stated. If users publish generated results, they remain
responsible for the upstream terms that apply to the datasets, codecs, models,
metrics and tools used to produce those results.
