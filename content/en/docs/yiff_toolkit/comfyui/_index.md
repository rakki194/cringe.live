---
weight: 2
bookFlatSection: false
bookToC: true
title: "The ComfyUI Bible"
summary: "A comprehensive guide to using ComfyUI, covering everything from basic node workflows to advanced techniques for AI image generation."
aliases:
  - /docs/yiff_toolkit/comfyui/
  - /docs/yiff_toolkit/comfyui
  - /en/docs/yiff_toolkit/comfyui/
  - /en/docs/yiff_toolkit/comfyui
image: "https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/confused2_resized.jpg"
image_alt: "The image depicts an anthropomorphic white wolf character adorned in a purple wizard's hat and robe, engrossed in reading a parchment. The setting appears to be a library or study, as evidenced by the shelves of books in the background. The wolf's expression conveys a sense of concentration or concern while examining the document. The scene is further enriched by a brick archway and a distant view of a medieval-style town with pointed towers and spires, enhancing the fantasy theme. This image skillfully combines elements of anthropomorphism and fantasy, creating a visually engaging scene that evokes curiosity about the character's role and the contents of the parchment."
blurhash: "LoH^#C~UtOXA-p%LtQo#R.oyozWY"
---

<!--markdownlint-disable MD025 MD033 MD038 -->

# The ComfyUI Bible

---

## Installing ComfyUI

---

If you need help installing ComfyUI, you didn't come to the right place. If you are using Windows, you can use the [prebuilt](https://docs.comfy.org/get_started/pre_package) package, or you can install it [manually](https://docs.comfy.org/get_started/manual_install) otherwise.

{{% details "Requirements for running the code examples on this page." %}}

To run the supplied visualization codes in this document, you'll need the following Python packages:

```bash
torchvision
numpy
Pillow
opencv-python
diffusers
```

Additionally, you'll need FFmpeg installed on your system for the video conversion. The code assumes FFmpeg is available in your system PATH.

{{% /details %}}

<!--

## ComfyUI's Structure

```bash
ComfyUI/
├───.ci/                         # Continuous Integration configurations
├───.github/                     # GitHub-specific files and workflows
├───api_server/                  # REST API implementation
│   ├───routes/                  # API endpoint definitions
│   ├───services/                # API service implementations
│   └───utils/                   # API utility functions
├───app/                         # Core application code
├───comfy/                       # Main ComfyUI backend implementation
│   ├───cldm/                    # ControlNet implementations
│   ├───ldm/                     # Latent Diffusion Model components
│   ├───k_diffusion/             # K-diffusion sampling methods
│   ├───text_encoders/           # Various text encoder implementations
│   └───taesd/                   # Tiny AutoEncoder for Stable Diffusion
├───comfy_execution/             # Workflow execution engine
├───comfy_extras/                # Additional ComfyUI features
├───custom_nodes/                # Community and custom node implementations
│   ├───ComfyUI-Manager/         # Package and node manager
│   ├───ComfyUI-Impact-Pack/     # Advanced node collection
│   ├───ComfyUI-Inspire-Pack/    # Creative workflow nodes
│   └───example.py               # A custom node in a separate python file under custom_nodes.
├───input/                       # Input files directory
├───models/                      # Model weights and files
│   ├───checkpoints/             # Main Stable Diffusion model weights
│   ├───clip/                    # CLIP text encoder models
│   ├───clip_vision/             # CLIP vision models
│   ├───configs/                 # Model configuration files
│   ├───controlnet/              # ControlNet model weights
│   ├───diffusers/               # HuggingFace Diffusers format models
│   ├───embeddings/              # Textual Inversion embeddings
│   ├───gligen/                  # GLIGEN grounded generation models
│   ├───hypernetworks/           # Style modification networks
│   ├───loras/                   # Low-rank adaptation models
│   │   ├───sdxl/                # SDXL-compatible LoRAs
│   │   ├───sd15/                # SD 1.5-compatible LoRAs
│   │   ├───sd3/                 # SD 3-compatible LoRAs
│   │   └───[others]/            # Other model-specific LoRAs
│   ├───onnx/                    # ONNX optimized models
│   ├───photomaker/              # PhotoMaker personalization models
│   ├───sams/                    # Segment Anything Models
│   ├───style_models/            # Style transfer models
│   ├───text_encoders/           # Additional text encoders
│   ├───unet/                    # U-Net model components
│   ├───upscale_models/          # Super-resolution models
│   ├───vae/                     # Variational AutoEncoder models
│   └───vae_approx/              # Approximate VAE implementations
├───notebooks/                   # Jupyter notebooks for development
├───output/                      # Generated outputs directory
├───script_examples/             # Example scripts and workflows
├───temp/                        # Temporary file storage
├───tests/                       # Testing infrastructure
├───user/                        # User-specific configurations
├───utils/                       # Utility functions and helpers
└───web/                         # Frontend implementation
    ├───assets/                  # Static assets and resources
    ├───extensions/              # Web interface extensions
    ├───scripts/                 # Frontend JavaScript code
    └───templates/               # HTML templates
```

-->

## Understanding Diffusion Models

---

Before diving into ComfyUI's practical aspects, let's understand the mathematical foundations of diffusion models that power modern AI image generation. If you disagree, you can scribble over all the equations on your monitor with a permanent marker!

### The Diffusion Process

Diffusion models work through a process that's similar to gradually adding static noise to a TV signal, and then learning to remove that noise to recover the original picture.

#### Forward Diffusion

The forward diffusion process systematically transforms a clear image into pure random noise through a series of precise mathematical steps.

1. We start with an original image $x_0$ containing clear, detailed information
2. At each timestep $t$, we apply a carefully controlled amount of Gaussian noise
3. The noise intensity follows a schedule $\beta_t$ that gradually increases over time
4. Each step slightly {{<zalgo strength=15 >}}corrupts{{</zalgo>}} the previous image state according to our diffusion equation
5. After $T$ timesteps, we reach $x_T$ which is effectively pure Gaussian noise

<div style="text-align: center;">
    <video style="width: 100%;" autoplay loop muted playsinline>
        <source src="https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/comfyui/forward_diffusion.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

> The [code](https://github.com/ka-de/cringe.live/blob/main/scripts/visualize_diffusion.py) used to generate the video.

This demonstrates pixel-space diffusion, just to give you an idea of what's happening.

To actually see the diffusion process in latent space, first we have to learn about latent space..

#### Latent Space

it's important to understand that AI models don't work directly with regular images. Instead, they work with something called "latents" - a compressed representation of images that's more efficient for the AI to process.

Think of them like a blueprint of an image:

- A regular image stores exact colors for each pixel (like a detailed painting)
- A latent stores abstract patterns and features (like an architect's blueprint)

Here is a video showing the forward diffusion process in latent space:

<div style="text-align: center;">
    <video style="width: 100%;" autoplay loop muted playsinline>
        <source src="https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/comfyui/latent_forward_diffusion.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

> The [code](https://github.com/ka-de/cringe.live/blob/main/scripts/visualize_latent_diffusion.py) used to generate the video.

The beauty of working with latents is that they're much more efficient:

- Takes less memory (8x smaller than regular images)
- Easier for the AI to manipulate
- Contains the essential "structure" of images without unnecessary details

The main differences you'll notice to pixel-space diffusion are:

- The noise patterns will look more structured due to the VAE's learned latent space.
- The degradation process might preserve more semantic information.

ComfyUI provides a visual interface to interact with these mathematical processes through a node-based workflow, you'll encounter latents in three main ways:

##### Empty Latent Image

When you want to generate an image from scratch, referred to as text-to-image (t2i).

- This is your starting point for pure text-to-image generation
- Size: target image size. The node transparently outputs an 8x smaller grid than your target image (e.g., 512x512 image = 64x64 latents)
- The actual latent content is zero latent pixels. Unlike the RGB zero, they don't map to black but a greyish color. Noise will be added at the start of the diffusion process (see [KSampler node](#the-ksampler-node)).

##### Load Image → VAE Encode

When you want to modify an existing image, referred to as image-to-image (i2i).

- Load Image: Brings your regular image into ComfyUI
- VAE Encode: Converts it into latents.
- This is your starting point for image-to-image generation

##### VAE Decode → Save Image

The final step in any workflow:

- Converts latents back into a regular image
- Like turning the blueprint back into a detailed painting
- This is how you get your final output image

##### The Mathematics

The diffusion process in DDIM is characterized by a Markov chain, where Gaussian noise is incrementally introduced over discrete time steps. This implies that the probability of the current state, $x_t$ is contingent solely on the preceeding state $x\_{t-1}. Mathematically this transition is represented as:

$$
q(x_t | x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})
$$

Here, $q$ represents the diffusion transition, modeled as a normal distribution. Pixel values $x_t[i,j]$ are independently sampled with variance $\beta_t$ according to the noise schedule. The mean term, $\sqrt{1-\beta_t}x_{t-1}$, retains a scaled down, or faded version of the previous step, effectively diminishing its intensity as noise is progressively added, while $\beta_t \mathbf{I}$ ensures noise independence across pixels.

By normalizing the dataset, we can crudely approximate sampling the initial image \( x_0 \sim \mathcal{D} \) as \( x_0 \sim \mathcal{N}(0, \mathbf{I}) \). Under this assumption, the variance of \( \sqrt{1-\beta_t} x_0 \) is \( 1-\beta_t \). Adding independent Gaussian noise with variance \( \beta_t \) then ensures the total variance remains 1 at each step.

<!--

<div style="text-align: center;">
    <video style="width: 100%;" autoplay loop muted playsinline>
        <source src="" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

This is a visual demonstration of how noise is progressively added to an image in the diffusion process, in pixel-space, showing four key components side by side: the original image at each timestep ($x_{t-1}$), the scaled-down version of that image ($\sqrt{1-\beta_t}x_{t-1}$), the random Gaussian noise being added ($\mathcal{N}(0, \beta_tI)$), and the resulting noisy image ($x_t$). The visualization includes the mathematical equation at the bottom, updating in real-time to show how the scaling factor ($\sqrt{1-\beta_t}$) decreases while the noise intensity ($\beta_t$) increases over time, making it clear how the image gradually transitions from its original state to becoming increasingly noisy.

\begin{aligned}
    x_t &= \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t} \mathcal{N}(x_t; 0, I) \\
    \mathbb{E}[x_t] &= \sqrt{1-\beta_t} \mathbb{E}[x_{t-1}] + \sqrt{\beta_t} \mathbb{E}[\mathcal{N}(x_t; 0, I)] \\
    &= \sqrt{1-\beta_t} \mathbb{E}[x_{t-1}] \\
    \text{Var}(x_t) &= (1-\beta_t) \text{Var}(x_{t-1}) + \beta_t \text{Var}(\mathcal{N}(x_t; 0, I)) \\
    &= (1-\beta_t) \text{Var}(x_{t-1}) + \beta_t I
\end{aligned}

-->

##### Alpha Bar and SNR

Alpha bar ($\bar{\alpha}$) is a crucial concept in diffusion models that represents the cumulative product of the signal scaling factors. Here's how it works:

1. At each timestep $t$, we have:

   - $\beta_t$ (beta): the noise schedule parameter
   - $\alpha_t$ (alpha): $1 - \beta_t$, the signal scaling factor

2. Alpha bar is then defined as:
   $$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$$

This means:

- $\bar{\alpha}$ starts at 1 (no noise) and decreases over time
- It represents how much of the original signal remains at time $t$
- At the end of diffusion, $\bar{\alpha}$ approaches 0 (pure noise)

The complete forward process can be written in terms of $\bar{\alpha}$:
$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

where:

- $x_t$ is the noisy image at time t
- $x_0$ is the original image
- $\epsilon ~ \mathcal{N}(0, I)$ is random Gaussian noise
- $\sqrt{\bar{\alpha}_t}$ controls how much original signal remains
- $\sqrt{1-\bar{\alpha}_t}$ controls how much noise is added

This formulation proves to be particularly powerful for several key reasons. First, it enables us to directly sample any timestep from the original image $x_0$ without having to calculate all intermediate steps. Additionally, it provides precise visibility into the ratio between signal and noise at each point in the process. Finally, this formulation makes the reverse process mathematically tractable, which is essential for training the model to denoise images effectively.

With a sufficiently large dataset, we can estimate the variance of the distributions of images (or latents). The variance represents the average "signal's energy" in the images. When adding two independent signals together, their variance adds up under the Gaussian assumption, often referred to in the context of Gaussian processes, which is a fundamental concept in probability theory and statistics. It assumes that a collection of random variables follows a multivariate normal distribution. This means that any finite subset of these variables will have a joint Gaussian distribution. The Gaussian assumption is named after Carl Friedrich Gauss, who introduced the Gaussian distribution, also known as the normal distribution. It allows for the modeling of complex, continuous phenomena using Gaussian processes. These processes are completely defined by their mean and covariance functions. The mean function represents the expected value of the process at any given point, while the covariance function describes how the values of the process at different points are related to each other. This assumption simplifies the analysis and computation of various statistical properties, making Gaussian processes a powerful tool for modeling and predicting real-world phenomena.

Diffusion models, such as Denoising Diffusion Probabilistic Models (DDPM), rely on the assumption that the data distribution can be approximated by a Gaussian distribution at each step of the diffusion process. This assumption simplifies the mathematical formulation and allows for efficient sampling and generation of images. The Gaussian assumption ensures that the noise added at each step follows a normal distribution, making the process mathematically tractable and allowing for the use of well-established techniques in probability theory and statistics.

This is important in the diffusion process, since the noise levels are also measured in terms of variance. After adding noise to the image, the Signal-to-Noise Ratio (SNR) is used to define how much useful signal is left over the noise background. This measure is independant of the total variance, thus any noising process can be remaped to a variance preserving process, meaning we add as much noise as we remove image energy. The dataset is often normalized to variance of 1 by removing the mean and dividing by the standard deviation.

The process serves as the foundation for training our AI models. By understanding exactly how images are corrupted, we can teach the model to reverse this corruption during the generation process.

#### Reverse Diffusion

When the AI learns to reverse this process, it learns to:

1. Start with pure noise
2. Gradually remove the noise
3. Eventually recover a clear image

This is what happens every time you generate an image with Stable Diffusion or similar AI models. The model has learned to take random noise and progressively "denoise" it into a clear image that matches your prompt.

### The KSampler Node

The `KSampler` node in ComfyUI implements the reverse diffusion process. It takes the noisy latent $x_T$ and progressively denoises it using the model's predictions. The node's parameters directly control this process:

1. **Steps**: Controls the number of $t$ steps in the reverse process

   - More steps = finer granularity but slower
   - Mathematically: divides $[0,T]$ into this many intervals

2. **CFG Scale**: Implements Classifier-Free Guidance

   $$
   \epsilon_\text{CFG} = \epsilon_\theta(x_t, t) + w[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t)]
   $$

   - Higher values follow the prompt more strictly
   - Lower values (1-4) allow more creative freedom

3. **Scheduler**: Controls $\beta_t$ schedule

   - `Karras`: $\beta_t$ follows the schedule from [Karras et al.](https://arxiv.org/abs/2206.00364)
     $$
     \sigma_i = \sigma_\text{min}^{1-f(i)} \sigma_\text{max}^{f(i)}
     $$
   - `Normal`: Linear schedule
     $$
     \beta_t = \beta_\text{min} + t(\beta_\text{max} - \beta_\text{min})
     $$

4. **Sampler**: Determines how to use model predictions

   - `Euler`: Simple first-order method
     $$
     x_{t-1} = x_t - \eta \nabla \log p(x_t)
     $$
   - `DPM++ 2M`: Second-order method with momentum
     $$
     v_t = \mu v_{t-1} + \epsilon_t
     $$
     $$
     x_{t-1} = x_t + \eta v_t
     $$

5. **Seed**: Controls the initial noise
   - Same seed + same parameters = reproducible results
   - Mathematically: initializes the random state for $x_T$

**The Noise Node**
The optional `Noise` node lets you directly manipulate the initial noise $x_T$. It implements:

$$
x_T = \mathcal{N}(0, I)
$$

You can:

- Set a specific seed for reproducibility
- Control noise dimensions (width, height)
- Mix different noise patterns

### V-Prediction and Angular Parameterization

The concept of v-prediction in image diffusion models has evolved significantly over the years. Initially, diffusion models focused on noise prediction, where the model learned to predict the noise added to the data at each step of the diffusion process. This approach was popularized by the [Denoising Diffusion Probabilistic Models (DDPM) introduced by Ho et al. in 2020](https://arxiv.org/abs/2006.11239). The noise prediction method proved effective for generating high-quality images, but it had limitations in terms of stability and efficiency. To address these limitations, researchers explored alternative prediction methods, leading to the development of v-prediction. V-prediction involves predicting the velocity of the data as it evolves through the diffusion process, rather than the noise. This approach was introduced to improve the stability and accuracy of the diffusion models, particularly near the zero point of the time variable. By focusing on the velocity, v-prediction helps mitigate issues related to Lipschitz singularities, which can pose challenges during both training and inference.

We can view an image or a latent as a single vector, composed of all the pixel values. Remember that diffusion processes can be remapped to a variance-preserving process, where the vectors for noise, image and intermediary steps are unit length. This means that we can represent the diffusion process as a rotation between the image and a gaussian noise vector. This happens in a high-dimensional space, but we are only interpolating between two directions (noise and image), so we can represent it in 2D coordinates. The intermediary noised images are all on a circle, and the velocity vector is tangeant to that trajectory..

The visualization below shows three key aspects of the v-prediction process: the noisy image progression (left), the raw velocity field in latent space (middle), and the decoded velocity field (right). The velocity field represents the direction and magnitude of change at each point, showing how the image should be denoised. The middle panel reveals the actual latent-space velocities at 1/8 resolution, while the right panel shows what these velocities "look like" when decoded back to image space. The overlay displays the current timestep ($t$), angle in radians ($\phi$), and cumulative signal scaling factor ($\bar{\alpha}$), helping us understand how the process evolves over time.

<div style="text-align: center;">
    <video style="width: 100%;" autoplay loop muted playsinline>
        <source src="https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/comfyui/v_prediction.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

> The [code](https://github.com/ka-de/cringe.live/blob/main/scripts/visualize_v_prediction.py) used to generate the video.

```python
# Calculate phi (angle) from alpha and sigma
phi_t = torch.arctan2(sigma_t, alpha_t)
```

This calculates the angle $\phi_t$ that represents how far along we are in the diffusion process.

- At $phi_t = 0$, we have the original image
- At $phi_t = \pi/2$, we have pure noise
- The angle smoothly interpolates between these states

While the standard formulation predicts noise $\epsilon$, an alternative approach called v-prediction parameterizes the diffusion process in terms of velocity. In this formulation, we use an angle $\phi_t = \text{arctan}(\sigma_t/\alpha_t)$ to represent the progression through the diffusion process.

$$
\alpha_\phi = \cos(\phi), \quad \sigma_\phi = \sin(\phi)
$$

The noisy image at angle $\phi$ can then be expressed as:

$$
\mathbf{z}_\phi = \cos(\phi)\mathbf{x} + \sin(\phi)\epsilon
$$

The key insight is to define a velocity vector:

$$
\mathbf{v}_\phi = \frac{d\mathbf{z}_\phi}{d\phi} = \cos(\phi)\epsilon - \sin(\phi)\mathbf{x}
$$

Starting from a clean image, the diffusion process
At the start of the diffusion process, the latent representation is heavily influenced by noise. In this stage the velocity vector $\mathbf{v}_\phi$ is primarily composed of noise ($\epsilon$), making it perpendicular to the image vector $\mathbf{x}$. This perpendicular relationship indicates that the velocity is introducing noise into the latent space, moving away from the clear image representation.

This velocity represents the direction of change in the noisy image as we move through the diffusion process. The model predicts this velocity instead of the noise:

$$
\hat{\mathbf{v}}_\theta(\mathbf{z}_\phi) = \cos(\phi)\hat{\epsilon}_\theta(\mathbf{z}_\phi) - \sin(\phi)\hat{\mathbf{x}}_\theta(\mathbf{z}_\phi)
$$

The sampling process then becomes a rotation in the $(\mathbf{z}_\phi, \mathbf{v}_\phi)$ plane:

$$
\mathbf{z}_{\phi_{t-\delta}} = \cos(\delta)\mathbf{z}_{\phi_t} - \sin(\delta)\hat{\mathbf{v}}_\theta(\mathbf{z}_{\phi_t})
$$

By learning to predict the velocity vectors ($\mathbf{v}_\phi$), the model ensures a stable and directed denoising path, where each velocity vector incrementally adjusts the latent representation towards the target image. As the process evolves, the alignment of velocity vectors with the image components showcases the structured denoising, gradually revealing the underlying image from the noisy latent.
This formulation offers several key advantages: it provides a more natural parameterization of the diffusion trajectory, simplifies the sampling process into a straightforward rotation operation, and can potentially lead to improved sample quality in certain scenarios.

**Implementation in ComfyUI: V-Prediction Samplers**
ComfyUI implements v-prediction through several specialized samplers in the `KSampler` node:

1. **DPM++ 2M Karras**

   - Most advanced v-prediction implementation
   - Uses momentum-based updates
   - Best for detailed, high-quality images
   - Recommended settings:
     - Steps: 25-30
     - CFG: 7-8
     - Scheduler: Karras

2. **DPM++ SDE Karras**

   - Stochastic differential equation variant
   - Good balance of speed and quality
   - Recommended settings:
     - Steps: 20-25
     - CFG: 7-8
     - Scheduler: Karras

3. **UniPC**
   - Unified predictor-corrector method
   - Fastest v-prediction sampler
   - Great for quick iterations
   - Recommended settings:
     - Steps: 20-23
     - CFG: 7-8
     - Scheduler: Karras

**Advanced V-Prediction Control**
For more control over the v-prediction process, you can use:

1. **Advanced KSampler**

   - Exposes additional v-prediction parameters
   - Allows fine-tuning of the angle calculations
   - Parameters:

     ```json
     start_at_step: 0.0
     end_at_step: 1.0
     add_noise: true
     return_with_leftover_noise: false
     ```

2. **Sampling Refinement**
   The velocity prediction can be refined using:

   - `VAEEncode` → `KSampler` → `VAEDecode` chain
   - Multiple sampling passes with decreasing step counts
   - Example workflow:

     ```json
     First Pass: 30 steps, CFG 8
     Refine: 15 steps, CFG 4
     Final: 10 steps, CFG 2
     ```

### Conditioning and Control

{{< blurhash
src="https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/conditioning_and_control_resize.jpg"
blurhash="LWAKUeof9ERk.Aj[IUR\*MvWC%Moe"
width="1440"
height="901"
alt="The image depicts a scene where five silhouetted figures with pointed ears are seated in front of a large screen. On the screen, there is a close-up of a white anthropomorphic wolf with green eyes and an open mouth, showing sharp teeth. The wolf appears to be speaking or making an expression, and the background of the screen is blue. The scene suggests that the silhouetted figures are watching or listening to the wolf on the screen, possibly in a setting reminiscent of a movie theater or a presentation. The contrast between the dark silhouettes and the brightly lit wolf on the screen creates a striking visual effect, making the image intriguing and engaging."
grid="false"
>}}

Text-to-image generation involves conditioning the diffusion process on text embeddings. The mathematical formulation becomes:

$$
p_\theta(x_{t-1}|x_t, \mathbf{c}) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, \mathbf{c}), \Sigma_\theta(x_t, t, \mathbf{c}))
$$

Here, $\mu_\theta$ and $\Sigma_\theta$ are conditioned on both $x_t$ and $\mathbf{c}$, which represents the conditioning information. In the context of text-to-image generation, this conditioning vector typically comes from CLIP (Contrastive Language-Image Pre-training), a neural network developed by OpenAI that creates a shared embedding space for both text and images.

#### Understanding CLIP and Conditioning

CLIP is a neural network trained to learn the relationship between images and text through contrastive learning. It consists of two encoders: one for text and one for images. During training, CLIP learns to maximize the cosine similarity between matching image-text pairs while minimizing it for non-matching pairs. This is achieved through a contrastive loss function operating on batches of N image-text pairs, creating an N×N similarity matrix.

The text encoder first tokenizes the input text into a sequence of tokens, then processes these through a transformer to produce a sequence of token embeddings $\text{CLIP}_\text{text}(\text{text}) \rightarrow [\mathbf{z}_1, ..., \mathbf{z}_n] \in \mathbb{R}^{n \times d}$, where $n$ is the sequence length and $d$ is the embedding dimension. Unlike traditional transformer architectures that use pooling layers, CLIP simply takes the final token's embedding (corresponding to the [EOS] token) after layer normalization. The image encoder maps images to a similar high-dimensional representation $\text{CLIP}_\text{image}(\text{image}) \rightarrow \mathbf{z} \in \mathbb{R}^d$.

### Implementation in ComfyUI

ComfyUI exposes this CLIP architecture through the `CLIPTextEncode` node. When you input a prompt, the node:

1. Tokenizes your text into subwords using CLIP's tokenizer
2. Processes tokens through the transformer layers
3. Produces the conditioning vectors that guide the diffusion process

#### CLIPTextEncode

Implements the standard CLIP text encoding $\text{CLIP}_\text{text}(\text{text})$ with a single text input field for the entire prompt. Depending on where you end up plugging it in either the `positive` or the `negative` input field of `KSampler`, `SamplerCustom` or , it will condition the diffusion process accordingly.

- Single text input field for the entire prompt
- Handles the full sequence of tokens as one unit
- Example usage:

```json
prompt: "a beautiful sunset over mountains, high quality"
negative prompt: "blurry, low quality, distorted"
```

### Prompt Weighting

In CLIP-based text conditioning, you can modify the influence of specific terms using weight modifiers. The mathematical representation of weighted prompting can be expressed as:

$$
c_\text{weighted} = w \cdot c_\text{base}
$$

where $w$ is the weight multiplier and $c_\text{base}$ is the base conditioning vector.

#### Syntax and Implementation

ComfyUI supports the following weighting syntax out of the box:

##### Parentheses Method

- Format: `(text:weight)`
- Example: `(beautiful sunset:1.2)`
- Weights > 1 increase emphasis
- Weights < 1 decrease emphasis

The weighting affects how strongly the diffusion process considers certain concepts during generation. For instance:

```json
prompt: "a (red:1.3) cat on a (blue:.8) chair"
```

This prompt will emphasize the redness of the cat while subtly reducing the blue intensity of the chair.

When you apply weights, the underlying CLIP embeddings are scaled according to:

$$
\text{CLIP}_\text{weighted}(\text{text}, w) = w \cdot \text{CLIP}_\text{text}(\text{text})
$$

This scaling affects the cross-attention layers in the U-Net, modifying how strongly different parts of the prompt influence the generation process. Multiple weighted terms combine through their effects on the overall conditioning vector:

$$
c_\text{final} = \sum_i w_i \cdot \text{CLIP}_\text{text}(\text{text}_i)
$$

#### Zero Weights

When a weight of 0.0 is applied, the corresponding term's contribution to the conditioning vector becomes nullified:

$$
\text{CLIP}_\text{weighted}(\text{text}, 0) = \mathbf{0}
$$

However, this doesn't mean the term is simply ignored. Setting a weight to 0.0 can have unexpected effects because:

1. The token still occupies a position in the sequence
2. It may affect how surrounding terms are interpreted

For this reason, it's generally better to omit terms you don't want rather than setting their weights to 0.0, however, neutralizing unwanted effects from multi-token trigger words is a valuable use case for them. Some words or phrases that get split into multiple tokens by CLIP's tokenizer can have unintended influences on generation. By using zero weights strategically, you can keep the trigger word's primary effect while nullifying its unwanted secondary effects.
Example:

```plaintext
furry (sticker:0.0)
```

- 0.0: Complete nullification
- 0.1 to 0.9: Reduced emphasis
- 1.0: Default weight
- 1.1 to 1.5: Moderate emphasis
- 1.5 to 2.0: Strong emphasis
- > 2.0: Very strong emphasis (use with caution)

Depending on the model you are using, even 1.3 can be too strong and can also depend on the specific token in question. When writing prompts, try to use related terms to compliment the attention of the model before trying to use weights.

#### CLIPTextEncodeSD3, CLIPTextEncodeFlux, CLIPTextEncodeSDXL

Specialized CLIP nodes, as their name suggests, for different models.

While these are useful to some degree for FLUX, SD3 and above, for SDXL, since both CLIP-L and CLIP-G was trained with the same prompts, you will achieve better results by using the same prompt, therefore the regular `CLIPTextEncode` node is sufficient, and as shown on the screenshot below, for regional conditioning you can always use `Conditioning (Set Area)`, which will work with all of the CLIP nodes.

![The rest of the CLIP Gang](https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/comfyui/CLIPTextEncodeGang.png)

<!--WRITE ABOUT FLUX PROMPTING-->

<!--AT LEAST MENTION SD3 PROMPTING STUFF-->

The sequence of token embeddings plays a crucial role in steering the diffusion process. Through cross-attention layers in the U-Net, the model can attend to different parts of the text representation as it generates the image. This mechanism enables the model to understand and incorporate multiple concepts and their relationships from the prompt.

Consider what happens when you input a prompt like "a red cat sitting on a blue chair". The text is first split into tokens, and each token (or subword) gets its own embedding. The model can then attend differently to "red", "cat", "blue", and "chair" during different stages of the generation process, allowing it to properly place and render each concept in the final image.

In most workflows, interaction with CLIP starts right from `Load Checkpoint` and ends at `KSampler`:

![CLIP's place in the workflow](https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/comfyui/basic_clip_example.png)

But you can also load it separately from the checkpoint using either the `Load CLIP`, `DualCLIPLoader` or `TripleCLIPLoader` nodes as needed.

![Various CLIP loaders in ComfyUI](https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/comfyui/CLIP_Loaders.png)

![CLIP](https://huggingface.co/k4d3/yiff_toolkit6/resolve/main/static/comfyui/ClipNodes.png)

**Advanced CLIP Operations: The ConditioningCombine Node**
To support complex prompting scenarios, ComfyUI provides the `ConditioningCombine` node that implements mathematical operations on conditioning vectors:

1. **Concatenation Mode**

   $$
   c_\text{combined} = [c_1; c_2]
   $$

   - Preserves both conditions fully
   - Useful for regional prompting

2. **Average Mode**

   $$
   c_\text{combined} = \alpha c_1 + (1-\alpha) c_2
   $$

   - Blends multiple conditions
   - $\alpha$ controls the mixing ratio

**Image-Based Conditioning: The CLIPVisionEncode Node**
This node implements the image encoder part of CLIP:

$$
\text{CLIP}_\text{image}(\text{image}) \rightarrow \mathbf{z} \in \mathbb{R}^d
$$

The complete pipeline for image-guided generation becomes:

$$
z_\text{image} = \text{CLIP}_\text{image}(\text{image})
$$

$$
z_\text{text} = \text{CLIP}_\text{text}(\text{prompt})
$$

$$
c_\text{combined} = \text{Combine}(z_\text{text}, z_\text{image})
$$

Beyond simple text conditioning, modern diffusion models support various forms of guidance. Image conditioning (img2img) allows existing images to influence the generation process. Control signals through ControlNet provide fine-grained control over structural elements. Style vectors extracted from reference images can guide aesthetic qualities, while structural guidance through depth maps or pose estimation can enforce specific spatial arrangements. Each of these conditioning methods can provide additional context to guide the generation process.

#### Unconditional vs Conditional Generation and CFG

The diffusion model can actually generate images in two modes: unconditional, where no guidance is provided ($\mathbf{c} = \emptyset$), and conditional, where we use our CLIP embedding or other conditioning signals. Classifier-Free Guidance (CFG) leverages both of these modes to enhance the generation quality.

The CFG process works by predicting two denoising directions at each timestep:

1. An unconditional prediction: $\epsilon_\theta(x_t, t)$
2. A conditional prediction: $\epsilon_\theta(x_t, t, \mathbf{c})$

These predictions are then combined using a guidance scale $w$ (often called the CFG scale):

$$
\epsilon_\text{CFG} = \epsilon_\theta(x_t, t) + w[\epsilon_\theta(x_t, t, \mathbf{c}) - \epsilon_\theta(x_t, t)]
$$

The guidance scale $w$ controls how strongly the conditioning influences the generation. A higher value of $w$ (typically 7-12) results in images that more closely match the prompt but may be less realistic, while lower values (1-4) produce more natural images that follow the prompt more loosely. When $w = 0$, we get purely unconditional generation, and as $w \to \infty$, the model becomes increasingly deterministic in following the conditioning.

This is why in ComfyUI, you'll often see a "CFG Scale" parameter in sampling nodes. It directly controls this weighting between unconditional and conditional predictions, allowing you to balance prompt adherence against image quality.

#### Implementation in ComfyUI: CFG Control

1. **Basic CFG Control**
   The `KSampler` node provides direct control over CFG through its parameters:

   - CFG Scale: The $w$ parameter in the equation
   - Recommended ranges:

     ```json
     Photorealistic: 4-7
     Artistic: 7-12
     Strong stylization: 12-15
     ```

2. **Advanced CFG Techniques**
   ComfyUI offers several nodes for fine-tuning CFG behavior:

   a) **CFGScheduler Node**

   - Dynamically adjusts CFG during sampling
   - Mathematical operation:
     $$w_t = w_\text{start} + t(w_\text{end} - w_\text{start})$$
   - Example schedule:

     ```json
     Start CFG: 12 (for initial structure)
     End CFG: 7 (for natural details)
     ```

   b) **CFGDenoiser Node**

   - Provides manual control over the denoising process
   - Allows separate CFG for different regions
   - Useful for:
     - Regional prompt strength
     - Selective detail enhancement
     - Balancing multiple conditions

3. **Practical CFG Workflows**

   a) **Basic Text-to-Image**

   ```json
   CLIPTextEncode [positive] → KSampler (CFG: 7)
   CLIPTextEncode [negative] → KSampler
   ```

   b) **Dynamic CFG**

   ```json
   CLIPTextEncode → CFGScheduler → KSampler
   Parameters:
   - Start: 12 (0-25% steps)
   - Middle: 8 (25-75% steps)
   - End: 6 (75-100% steps)
   ```

   c) **Regional CFG**

   ```json
   CLIPTextEncode → ConditioningSetArea → CFGDenoiser
   Multiple regions with different CFG values:
   - Focus area: CFG 12
   - Background: CFG 7
   ```

4. **CFG Troubleshooting**
   Common issues and solutions:
   - Over-saturation: Reduce CFG or use scheduling
   - Loss of details: Increase CFG in final steps
   - Unrealistic results: Lower CFG, especially in early steps
   - Inconsistent style: Use CFG scheduling to balance

## Core Components

---

### Models and Their Mathematical Foundations

Before starting with ComfyUI, you need to understand the different types of models:

1. **Base Models (Checkpoints)**

   - Stored in `models\checkpoints`
   - Implement the full diffusion process: $p_\theta(x_{0:T})$
   - Examples: CompassMix XL Lightning, Pony Diffusion V6 XL

   **The Load Checkpoint Node**
   The `Load Checkpoint` node in ComfyUI is your interface to the base diffusion model. It loads the model weights ($\theta$) and initializes the U-Net architecture that performs the reverse diffusion process. When you connect this node, you're essentially preparing the neural network that will implement:

   $$
   p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
   $$

   The node outputs three crucial components:

   - `MODEL`: The U-Net that predicts noise or velocity
   - `CLIP`: The text encoder for conditioning
   - `VAE`: The encoder/decoder for image latents

   These outputs correspond to the three main mathematical operations in the diffusion process:

   1. MODEL: Implements $p_\theta(x_{t-1}|x_t)$
   2. CLIP: Provides conditioning $c$ for $p_\theta(x_{t-1}|x_t, c)$
   3. VAE: Handles the mapping between image space $x$ and latent space $z$

2. **LoRAs (Low-Rank Adaptation)**

   - Stored in `models\loras`
   - Mathematically represented as: $W = W_0 + BA$ where:
     - $W_0$ is the original weight matrix
     - $B$ and $A$ are low-rank matrices
   - Reduces parameter count while maintaining model quality

   **The LoRA Loader Node**
   The `LoRA Loader` node implements the low-rank adaptation by modifying the base model's weights according to the equation:

   $$
   W_{\text{final}} = W_0 + \alpha \cdot (BA)
   $$

   where $\alpha$ is the weight parameter in the node (typically 0.5-1.0). The node:

   - Takes a MODEL input from Load Checkpoint
   - Applies the LoRA weights ($BA$) with scaling $\alpha$
   - Outputs the modified model

   When you adjust the weight parameter, you're directly controlling the influence of the low-rank matrices on the original weights.

3. **VAE (Variational Autoencoder)**

   The VAE (Variational Autoencoder) is a type of neural network that learns to:

   1. Encode images into a low-dimensional latent space representation using a neural encoder.
   2. Decode these latent vectors back into images during generation.
   3. Learn to generate new images by sampling from a lower-dimensional latent space for efficiency.

   - Stored in `models\vae`
   - Implements the encoding $q_\phi(z|x)$ and decoding $p_\psi(x|z)$

   **The VAE Encode/Decode Nodes**
   These nodes implement the probabilistic encoding and decoding:

   $$
   z \sim q_\phi(z|x)
   $$

   $$
   x \sim p_\psi(x|z)
   $$

   - `VAE Encode`: Converts images to latents (mean + variance)
   - `VAE Decode`: Converts latents back to images

   The latent space operations are crucial because:

   1. They reduce memory usage (latents are 4x smaller)
   2. They provide a more stable space for diffusion
   3. They help maintain semantic consistency

## Node Based Workflow

---

The node-based interface in ComfyUI represents the mathematical operations as interconnected components. Each node performs specific operations in the diffusion process:

- Sampling nodes implement the reverse diffusion process
- Conditioning nodes handle text embeddings and other control signals
- VAE nodes handle encoding/decoding between image and latent space: $\mathcal{E}(x)$ and $\mathcal{D}(z)$

![Arcane Wizardry](/images/comfyui/arcane_wizardry.png)

When you're new to node-based workflows, think of each connection as passing tensors and parameters between mathematical operations. The entire workflow represents a computational graph that implements the full diffusion process.

### Getting Started

To begin experimenting with these concepts, you can clear your workflow:

<div style="text-align: center;">
    <video style="width: 100%;" autoplay loop muted playsinline>
        <source src="https://huggingface.co/k4d3/yiff_toolkit/resolve/main/static/comfyui/clear_workflow.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

You can add nodes by either right-clicking or double-clicking on an empty area:

![Right Click Add Method](https://huggingface.co/k4d3/yiff_toolkit/resolve/main/static/comfyui/right_click_add.png)

---

<!--
HUGO_SEARCH_EXCLUDE_START
-->

{{< related-posts related="docs/yiff_toolkit/lora_training/ | docs/yiff_toolkit/ | docs/yiff_toolkit/comfyui/Custom-ComfyUI-Workflow-with-the-Krita-AI-Plugin/" >}}

<!--
HUGO_SEARCH_EXCLUDE_END
-->
