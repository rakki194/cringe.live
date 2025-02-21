---
weight: 3
bookFlatSection: false
bookToC: true
title: "WaveSpeed"
description: "A custom node designed to optimize model inference performance in ComfyUI."
aliases:
    - /en/docs/yiff_toolkit/comfyui/WaveSpeed/
    - /en/docs/yiff_toolkit/comfyui/WaveSpeed
    - /docs/yiff_toolkit/comfyui/WaveSpeed/
    - /docs/yiff_toolkit/comfyui/WaveSpeed
    - /docs/yiff_toolkit/comfyui/wavespeed/
    - /docs/yiff_toolkit/comfyui/wavespeed
image: "https://huggingface.co/rakki194/yt/resolve/main/static/comfywave_resized.webp"
image_alt: "The image depicts a fennec girl submerged underwater, gracefully swimming beneath the surface. She has blonde, slightly tousled hair, large fennec-like ears with white fur at the tips, and a bushy, cream-colored tail. She is dressed in a dark blue oversized jacket, which drapes loosely over her form, and wears black shoes. Her expression is calm and serene as she gazes toward the viewer, with small bubbles escaping toward the water's surface, which is illuminated by soft sunlight filtering from above. The deep blue hue of the surrounding water contrasts with her light fur and clothing, creating a visually striking composition that conveys a tranquil and dreamlike atmosphere."
blurhash: "LG5s+uaKI]X8*0jYM{R+n4oyVrRQ"
---

<!--markdownlint-disable MD025 MD033 MD038 -->

# WaveSpeed

---

## Introduction

---

WaveSpeed introduces dynamic caching (First Block Cache) and torch.compile capabilities, enabling faster and more efficient processsing while maintaining high accuracy. By integrating WaveSpeed into your ComfyUI workflows, you can achieve significant speedups without compromising too much quality.

## Features

---

### Apply First Block Cache

The `Apply First Block Cache` node provides a dynamic caching mechanism designed to optimize model inference performance for certain types of transformer-based models. The key idea behind FBCache is to determine when the output of the model has changed sufficiently since the last computation and, if not, reuse the previous result to skip unnecessary computations.

When a model is processed through the node, the (residual) output of the first transformer block is saved for future comparison. For each subsequent step in the model's computation, the current residual output of the first transformer block is compared to the previous residual. If the difference between these two residuals is below the specified threshold (`residual_diff_threshold`), it indicates that the model's output hasn't changed significantly since the last computation.

When the comparison indicates minimal change, FBCache decides to reuse the previous result. This skips the computation of all subsequent transformer blocks for the current time step. To prevent over-reliance on cached results and maintain model accuracy, a limit can be set on how many consecutive cache hits can occur (`max_consecutive_cache_hits`). This ensures that the model periodically recomputes to incorporate new input information.

The `start` and `end` parameters define the time range during which FBCache can be applied.

Let $t$ be the current time step, and $r_t$ be the residual at time $t$.

The condition for using cached data is:

$$
|r_t - r_{t-1}| < \text{threshold}
$$

If this condition holds, the model skips computing the subsequent transformer blocks and reuses the previous output. If not, it computes the full model as usual:

$$
\text{caching_decision} = \mathbf{1}_{|r_t - r_{t-1}| < \text{threshold}}
$$

<a href="https://huggingface.co/rakki194/yt/resolve/main/static/comfyui/wavespeed_plot_1.png">
{{< blurhash
  src="https://huggingface.co/rakki194/yt/resolve/main/static/comfyui/wavespeed_plot_1.webp"
  blurhash="LB6R+6ovk9ox?wkBkBbH?[j]j[f*"
  width="4544"
  height="1986"
  alt="An XYPlot with inference times."
  grid="false"
>}}
</a>

When evaluating FBCache, there is a plethora of variables to keep in mind, starting with the model you are trying to use it with. For example, my experiments, SDXL is a lot more stable with it than 3.5 Large. Prompt complexity and model stability will also greatly count towards making your evaluation a personal experience. Lastly, your chosen sampler and scheduler will also affect the effectiveness of it, for example, ancestral/SDE samplers will prevent the cache from activating because they keep introducing random noise at each step.

If the scheduler produces very gradual changes between steps, the cache hit rate increases and expensive recalculations can be skipped, Karras and Exponential are particularly effective here, because they are designed to spend more time sampling at the lower (i.e. finer) noise levels. Their noise schedules tend to create very smooth transitions between timesteps, resulting in smaller differences in the residual outputs of the first block. Other schedulers that might distribute their timesteps more uniformly or with more abrupt changes tend to produce larger differences between consecutive steps, which reduces the opportunity for caching and thereby diminishes the performance gains.

A notable GitHub issue by easygoing0114 in [#87](https://github.com/chengzeyi/Comfy-WaveSpeed/issues/87) explores different settings with some very nice findings.

### Compile Model+

This node dynamically compiles the model using PyTorch's `_dynamo` or `inductor` backends depending on configuration.

⚠️ **NOTE**: Before playing with this node, it is pretty important you add the `--gpu-only` flag to your launch parameters because the compilation may not work with model offloading.

Comfy-Cli users can just run:

```bash
comfy launch -- --gpu-only
```

The node accepts multiple parameters to control how the model is compiled:

- `model` (Required Input): Takes any model input. This is where you connect your diffusion model output. Can be a base model, patched model, or one that's already had First Block Cache applied.
- `is_patcher`: Controls how the model is handled during compilation. `true` treats the input as a fresh model that needs patching, while `false` expects the input to already be a patcher object. Generally you will want to leave this as `true`.
- `object_to_patch`: Specifies which part of the model architecture to optimize, the default value `diffusion_model` targets the main diffusion model component, which is typically what you want for standard Stable Diffusion workflows.
- `compiler`: Selects which compiler to use, the default `torch.compile` uses PyTorch's native compiler. The node will dynamically import the selected function.
- `fullgraph`: When `true`, PyTorch will attempt to compile the entire model as a single graph. May, or may not result in longer compilation times, but it will definitely increase memory usage during compilation. `false` is generally the safer choice.
- `dynamic`: Controls how the compiler handles varying input dimensions.
    When `false`, the compiler will optimize for fixed dimensions. It will expect consistent batch sizes, image resolution, sequence lengths and feature dimensions. It will recompile if any of these change in exchange for better performance.
- `mode`: Sets the optimization mode.
  - `""` (empty): The default, uses the compiler's default settings, which is pretty balanced.
  - `max-autotune`: Offers a more aggresive optimization strategy, in exchange for longer compilation times. Uses CUDA graphs for caching GPU operations.
  - `max-autotune-no-cudagraphs`: Similar to `max-autotune`, but without CUDA graphs, useful, if it causes issues.
- `options`: Allows you to pass additional options to the compiler. When not empty, it expects a valid JSON.
- `disable`: Disables the compilation, ending the fun. Useful, because bypassing the node doesn't work. 🤷‍♂️
- `backend`: Specifies the compilation backend to use, the default, `inductor` uses PyTorch's modern optimizing backend, recommended for modern GPUs. FP8 quantization requires Ada or newer GPU architecture.

At the time of this writing, the `torch.compile`, `inductor`, `dynamic` combination doesn't work.

<!--
#### `option` Options

Tuning Settings

```json
{
    "max_autotune": true,
    "max_autotune_depth": 5,
    "optimize_ctx": true
}
```

```json
{
    "max_autotune": true,
    "max_autotune_depth": 5,
    "optimize_ctx": true,
    "use_runtime_fusion": true,
    "size_asserts": false,
    "max_parallel_chunks": 8
}
```

Memory Optimization

```json
{
    "max_autotune": true,
    "min_memory": true,
    "max_recursive_depth": 3
}
```

Debug

```json
{
    "debug": true,
    "trace.enabled": true,
    "trace.graph_diagram": true
}
```
-->

<!-- ⚠️ TODO: Benchmarking -->

### Model Support

WaveSpeed is compatible with multiple models, including FLUX.1-dev, HunyuanVideo, SD3.5, and SDXL.

## Installation

---

1. **Clone the Repository**

   ```bash
   cd custom_nodes
   git clone https://github.com/chengzeyi/Comfy-WaveSpeed
   ```

2. **Restart ComfyUI**
   - After installation, restart your ComfyUI instance to apply changes.

## Usage

### Workflow Integration

1. **Load Your Model**
   - Connect your model loading node (e.g., `Load Checkpoint`) to the `Apply First Block Cache` node.
2. **Adjust Threshold**
   - Set the residual difference threshold to balance speedup and accuracy.
3. **Enable torch.compile**
   - Add the `Compile Model+` node after or before `Apply First Block Cache`, depending on your mood.
4. **Run!**

<!--

### Example Workflows
- **FLUX.1-dev**: `workflows/flux.json`
- **HunyuanVideo**: `workflows/hunyuan_video.json`
- **SD3.5**: `workflows/sd3.5.json`

-->
