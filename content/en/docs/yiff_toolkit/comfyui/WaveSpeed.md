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
---

<!--markdownlint-disable MD025 MD033 MD038 -->

# WaveSpeed

---

## Introduction

---

WaveSpeed is part of the ParaAttention project, a suite of tools focused on optimizing model inference.

It introduces dynamic caching (First Block Cache) and enhanced torch.compile capabilities, enabling faster and more efficient processsing while maintaining high enough accuracy. By integrating WaveSpeed into your ComfyUI workflows, you can achieve significant speedups without compromising quality.

{{< blurhash
  src="https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/comfyui/wavespeed_plot_1.png"
  blurhash="LB6R+6ovk9ox?wkBkBbH?[j]j[f*"
  width="4544"
  height="1986"
  alt="An XYPlot with inference times."
  grid="false"
>}}

## Features

---

### Dynamic Caching (First Block Cache)

Inspired by TeaCache and other caching algorithms, First Block Cache uses the residual output of the first transformer block as an indicator for caching. If the difference between the current and previous residual output is small enough, it reuses the previous result, skipping computation of subsequent blocks.
This can significantly reduce computation time.

### Enhanced torch.compile

WaveSpeed integrates advanced torch.compile functionality, optimizing model graphs for maximum performance.
It supports both `max-autotune` and `max-autotune-no-cudagraphs` modes, providing flexibility for different hardware configurations.
The compilation process is cached for future runs, making it efficient even after multiple changes.

### Context Parallelism

WaveSpeed enables context parallelism, a method for distributing neural network activations across multiple GPUs. This allows for parallel processing of attention mechanisms, achieving faster inference times without altering the original model code.
The node provides a unified interface for both Ulysses Style and Ring Style parallelism, offering the best performance for various hardware setups.

### Model Support

WaveSpeed is compatible with multiple models, including FLUX.1-dev, HunyuanVideo, SD3.5, and SDXL. It works seamlessly with different model sizes and configurations, ensuring versatility across your workflow.

### Easy Integration

Adding WaveSpeed to your workflow is straightforward. Simply include the `Apply First Block Cache` node after your model loading step and adjust the residual difference threshold for optimal performance.
The `Compile Model+` node can be added to further enhance speed via torch.compile, with options for dynamic compilation and GPU-only mode.

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
   - Set the residual difference threshold to balance speedup and accuracy. For example, 0.12 for FLUX with 28 steps.
3. **Enable torch.compile**
   - Add the `Compile Model+` node after or before `Apply First Block Cache`, depending on your mood.
4. **Run Your Workflow**
   - Save and run your workflow to experience improved performance.

<!--

### Example Workflows
- **FLUX.1-dev**: `workflows/flux.json`
- **HunyuanVideo**: `workflows/hunyuan_video.json`
- **SD3.5**: `workflows/sd3.5.json`

-->
