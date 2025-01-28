---
weight: 1
bookFlatSection: false
bookToC: true
title: "LLMs"
date: 2024-01-27
summary: "LLMs, or Large Language Models are deep neural networks trained on vast amounts of text data, enabling them to flirt with you. This guide completes the circle by introducing you to your virtual wife/husband/mate/host organism/etc."
aliases:
  - /docs/llms/
  - /docs/llms
  - /en/docs/llms/
  - /en/docs/llms
---

# Language Models

---

Language models are deep neural networks trained on vast amounts of text data, enabling them to predict and generate human-like text sequences. Through exposure to diverse training data, these models learn complex patterns in languages, from basic grammar and vocabulary to more nuanced aspects like context, reasoning, and domain-specific knowledge. These models typically rely on the transformer architecture, which consists of self-attention mechanisms that allow them to process input sequences while considering all positions in the sequence simultaneously.

This advanced architecture has opened up a new world for a wide variety of fields, from coding, image generation to interactive storytelling and in every field of science. Language models can write code for you or act as dungeon masters, NPCs, or even your erotic roleplaying companions. With the right setup, you can create immersive adventures, generate unique characters, and explore dynamic narratives, all guided by AI instead of the inconvenience of interacting with real people.

In this guide, we'll intimately explore the capabilities of language models for both coding and interactive roleplaying scenarios from first a deep technical, followed by a practical point of view in the hopes of incrementally improving the globally generated slop.

## Subsections

---

{{< section-noimg details >}}

## Introduction

The key components of a language model include:

### Tokens

Before we can understand how language models process text, we need to understand tokens - the fundamental units that these models work with. A token is not always a complete word; instead, it's a subunit of text that the model processes as a single element.

For any input text $X$, tokenization splits it into a sequence of tokens:

$$
X \rightarrow [t_1, t_2, ..., t_n]
$$

where each $t_i$ represents a token in the model's vocabulary $V$, and $n$ is the sequence length.

Consider this example:

```python
Input: "The dragonslayer unsheathed her sword"

Tokens: ["The", "dragon", "slayer", "un", "sheath", "ed", "her", "sword"]
```

Each token is then converted into a numerical vector through an embedding matrix $E \in \mathbb{R}^{|V| \times d}$, where $|V|$ is the vocabulary size and $d$ is the embedding dimension:

$$
\text{embedding}(t_i) = E_{t_i} \in \mathbb{R}^d
$$

Important characteristics of tokenization include:

### Vocabulary Size

Most modern language models use vocabularies of 32,000 to 100,000 tokens. This includes:
 - Common words ("the", "and", "sword")
 - Subwords ("un", "ed", "ing")
 - Characters ("a", "b", "?")
 - Special tokens like \<BOS\> (Beginning of Sequence) and \<EOS\> (End of Sequence)

### Subword Tokenization

Words not in the vocabulary are broken down into subwords. For example:

```python
"dragonslayer" → ["dragon", "slayer"]
"unsheathed" → ["un", "sheath", "ed"]
```

This allows models to handle rare or novel words by combining familiar parts.

### Context Window

> Not to be confused with [Context Length](#context-length), which is a related, but a distinct concept.

The context window refers to the maximum number of tokens (words or subwords) that a language model can process in a single forward pass. It defines the span of text that the model can "see" and consider when generating a response. For example, if a model has a context window of 2048 tokens, it can take into account up to 2048 tokens of preceding text to generate the next token.

### Context Length

The context length $L$ represents the maximum sequence length a model can effectively process in a single forward pass. Modern language models exhibit varying context lengths:

$$
L_{effective} \leq L_{max}
$$

where $L_{effective}$ is the actual sequence length being processed and $L_{max}$ is the model's maximum supported context length.

The memory requirements for processing a sequence scale linearly with the context length. For a model with parameters $\theta$, the memory consumption $M$ can be approximated as:

$$
M = (2B)(2)(h_l)(k_h)(h_s)/a_h
$$

where:

- $B$ = bytes per token (typically 2)
- $h_l$ = number of hidden layers
- $k_h$ = number of key-value heads
- $h_s$ = hidden size
- $a_h$ = number of attention heads

For example, given a model architecture with:

```python
hidden_size = 4096
hidden_layers = 32
key_value_heads = 8
attention_heads = 32
```

The memory requirement per token would be:

$$
(2)(2)(32)(8)(4096)/32 = 128\text{ kB}
$$

Here is a Python script to calculate the memory requirements for a given model configuration based on this formula:

{{% details "Click here to show the code." %}}

```python
def calculate_memory_per_token(hidden_size, hidden_layers, key_value_heads, attention_heads):
    # Constants
    BYTES_PER_TOKEN = 2

    # Calculate memory per token in bytes
    memory = (2 * BYTES_PER_TOKEN) * (2) * hidden_layers * key_value_heads * hidden_size / attention_heads

    # Convert to kilobytes
    memory_kb = memory / 1024

    return memory_kb

def calculate_total_memory(context_length, memory_per_token):
    # Calculate total memory for the given context length
    total_memory_kb = memory_per_token * context_length
    total_memory_mb = total_memory_kb / 1024
    total_memory_gb = total_memory_mb / 1024

    return total_memory_kb, total_memory_mb, total_memory_gb

def main():
    # Model architecture parameters
    params = {
        'hidden_size': 4096,
        'hidden_layers': 32,
        'key_value_heads': 8,
        'attention_heads': 32
    }

    # Calculate memory per token
    memory_per_token = calculate_memory_per_token(**params)
    print(f"Memory per token: {memory_per_token:.2f} KB")

    # Context lengths
    context_lengths = [4096, 8192, 16384, 32768, 65536]

    # Calculate and display total memory requirements for different context lengths
    print("\nTotal memory requirements for different context lengths:")
    for length in context_lengths:
        memory_kb, memory_mb, memory_gb = calculate_total_memory(length, memory_per_token)
        print(f"\nContext Length: {length}")
        print(f"Memory required: {memory_kb:.2f} KB")
        print(f"               = {memory_mb:.2f} MB")
        print(f"               = {memory_gb:.2f} GB")

if __name__ == "__main__":
    main()
```

{{% /details %}}

The three key considerations for context length include **Memory scaling**, where longer contexts require proportionally more GPU VRAM or system RAM, the computational complexity of **self-attention mechanisms**, which scale quadratically with context length, and **Position encoding**, which becomes crucial for maintaining coherence at longer context lengths, for example, <abbr title="Rotary Position Embedding">RoPE</abbr>.

If we denote the context window as $W$ and the context length as $L$, the relationship can be summarized as:

$$
L \leq W
$$

This means that the context length $L$ is always less than or equal to the context window $W$.

In practice, the context window sets the upper limit on how much text the model can consider at once, while the context length is the actual amount of text being processed within that limit. This distinction is crucial for understanding the capabilities and limitations of language models in handling long text sequences.

### Position Encoding

Each token's position in the sequence is encoded using positional embeddings:

$$
\text{input}_i = \text{embedding}(t_i) + \text{position}(i)
$$

This position-aware representation is crucial for the self-attention mechanisms that follow, as it helps the model understand word order and narrative sequence in roleplaying scenarios.

Understanding tokens is essential because they affect:

- How much context your roleplaying session can maintain
- How the model handles character names and fictional terms
- The computational resources required (memory scales with token count)
- The cost of using commercial APIs (often priced per token)

This tokenized representation forms the foundation for the self-attention mechanisms we'll discuss next, where these token embeddings interact to understand relationships between words and concepts in the text.

### Self-Attention Mechanisms

Each token in the input sequence is projected into a vector space, and self-attention computes similarity scores between every pair of tokens using dot products. These scores are normalized to produce attention weights, which determine how much each token influences others.

The self-attention mechanism can be understood through the formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Imagine the following sentence: "The dragon breathed fire because it was angry."

In this case, the word "it" needs to understand what it's referring to. The attention mechanism helps the model figure out that "it" refers to "dragon" and not "fire" through these steps:

1. **Query ($Q$), Key ($K$), and Value ($V$) Creation**:
   Each word is transformed into three different vectors:
   - Query ($Q$): What the word is looking for
   - Key ($K$): What the word can match with
   - Value ($V$): The actual information to be passed along

   Mathematically, these are created through learned transformations:

   $$
   \begin{align*}
   Q &= XW^Q \\
   K &= XW^K \\
   V &= XW^V
   \end{align*}
   $$

   where X is the input embedding and W matrices are learned parameters.

2. **Similarity Scoring**:
   The $QK^T$ term computes how relevant each word is to every other word. In our example, when processing "it", the model computes attention scores with all other words. The score between "it" and "dragon" should be high, while the score with "fire" should be lower.

3. **Scaling Factor**:
   The $\sqrt{d_k}$ scaling (where $d_k$ is the dimension of the key vectors) prevents the dot products from growing too large in magnitude, which could push the softmax function into regions with extremely small gradients. This helps maintain stable training.

4. **Softmax Normalization**:
   The softmax function converts these scores into probabilities that sum to 1:

   $$
   \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
   $$

5. **Final Value Aggregation**:
   These probabilities are used to create a weighted sum of the values (V), producing the final output for each position.

For roleplaying applications, this mechanism is crucial because it allows the model to:
- Maintain character consistency by connecting current dialogue with previous context
- Understand complex relationships between story elements
- Follow long-term plot threads and callbacks

For example, if a character says "I draw my father's sword", the attention mechanism helps the model reference any previous mentions of the father or the sword, maintaining narrative coherence.

In practice, models use Multi-Head Attention, which runs multiple attention operations in parallel:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

where each head is computing:

$$
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

This allows the model to capture different types of relationships between words simultaneously, much like how a good storyteller can keep track of multiple plot threads and character relationships at once.

### Multi-Layer Perceptrons

Or MLPs for short. After applying attention, the model processes the resulting sequences through a series of feedforward neural networks, with parameters learned during training to capture patterns in the data.

### Training Objective

Models are often trained using masked language modeling or unmasked language modeling, where the goal is to maximize the likelihood of predicting the next token given the previous tokens. This involves minimizing loss functions that measure the difference between predicted and actual outputs.

### Parameters and Compute Requirements

Modern LLMs like GPT-4 or Meta-Llama have billions of parameters, requiring significant computational resources for training and inference. For example, a model with 175B parameters might consume up to 800 GB of VRAM for inference tasks.

When used as AI companions or non-player characters (NPCs) in roleplaying scenarios, language models can generate dynamic and context-rich interactions. Their ability to process extended contexts allows them to maintain consistent and meaningful conversations over time. Here’s how they excel:

1. **Context Handling**: Most modern models support context windows of several thousand tokens, enabling them to remember key details from previous interactions.

2. **Dynamic Responses**: Language models can adapt their responses based on the current conversation state, providing a more natural feel to interactions.

3. **Fine-Tuning**: Models can be fine-tuned for specific roleplaying scenarios using techniques like LoRA (Low-Rank Adaptation), where a small fraction of model weights are updated during training while keeping most parameters frozen. This preserves pre-trained capabilities while enabling specialization for roleplaying tasks.

4. **Memory and Performance Trade-offs**: Running larger models on GPUs with higher VRAM provides better performance but may require specialized software like Exllamav2_HF or llama.cpp for efficient loading and inference. Quantization techniques, such as reducing the precision of weights to 4 bits per weight (4bpw), can further reduce memory usage at the cost of minor quality degradation.

## Language Model Efficiency

---

The efficiency of language models depends on several factors:

- **Context Length (L)**: The maximum number of tokens a model processes in one forward pass.
- **Memory Requirements**: For GPU inference, memory consumption scales with the number of tokens and model size. A simplified formula for estimating VRAM usage per token is:

    $$
    \text{VRAM per token} = \frac{(2 \times \log(N))}{\text{Batch Size}}
    $$

    where $N$ is the number of model parameters.

## Quantization

---

Quantization is a technique used to reduce the precision of the weights and activations in a neural network, which can significantly decrease the memory and computational requirements. This is particularly useful for deploying large models on devices with limited resources, such as mobile phones or embedded systems.

### Quantization Process

1. **Weight Quantization**: The weights of the neural network are converted from floating-point precision (e.g., 32-bit) to a lower precision format (e.g., 8-bit or 4-bit). This reduces the memory footprint and can speed up inference.

2. **Activation Quantization**: Similar to weight quantization, the activations (outputs of each layer) are also converted to a lower precision format.

The process of quantization can be mathematically represented as follows:

1. **Scaling Factor**: A scaling factor $S$ is determined to map the floating-point values to the quantized values. This is typically done by finding the range of the floating-point values and dividing it by the range of the quantized values.

   $$
   S = \frac{\text{max} - \text{min}}{2^b - 1}
   $$

   where $b$ is the number of bits used for quantization.

2. **Quantization**: The floating-point values $x$ are then quantized to integer values $q$ using the scaling factor $S$:

   $$
   q = \text{round}\left(\frac{x}{S}\right)
   $$

3. **Dequantization**: During inference, the quantized values $q$ are converted back to floating-point values using the same scaling factor $S$:

   $$
   x' = q \cdot S
   $$

### Example

Let's say we have a weight value $x = 0.75$ and we want to quantize it to 4 bits (16 levels). If the range of the weights is $[-1, 1]$, the scaling factor $S$ would be:

$$
S = \frac{1 - (-1)}{2^4 - 1} = \frac{2}{15} \approx 0.133
$$

The quantized value $q$ would be:

$$
q = \text{round}\left(\frac{0.75}{0.133}\right) = \text{round}(5.64) = 6
$$

During inference, the dequantized value $x'$ would be:

$$
x' = 6 \cdot 0.133 \approx 0.798
$$

This process introduces a small error, but the overall performance of the model is often not significantly affected.

## Applications in Interactive Roleplaying

---

When used as AI companions or non-player characters (NPCs) in roleplaying scenarios, language models can generate dynamic and context-rich interactions. Their ability to process extended contexts allows them to maintain consistent and meaningful conversations over time. Here’s how they excel:

1. **Context Handling**: Most modern models support context windows of several thousand tokens, enabling them to remember key details from previous interactions.

2. **Dynamic Responses**: Language models can adapt their responses based on the current conversation state, providing a more natural feel to interactions.

3. **Fine-Tuning**: Models can be fine-tuned for specific roleplaying scenarios using techniques like LoRA (Low-Rank Adaptation), where a small fraction of model weights are updated during training while keeping most parameters frozen. This preserves pre-trained capabilities while enabling specialization for roleplaying tasks.

4. **Memory and Performance Trade-offs**: Running larger models on GPUs with higher VRAM provides better performance but may require specialized software like Exllamav2_HF or llama.cpp for efficient loading and inference. Quantization techniques, such as reducing the precision of weights to 4 bits per weight (4bpw), can further reduce memory usage at the cost of minor quality degradation.

## Getting Started

---

First, you need to choose where and how to run your model. Here are the main options:

### Running on Your GPU

Running a model on your GPU is the fastest option, but it requires enough VRAM. You’ll want at least 8 GB, though 12 GB or more is ideal. Use a loader like Exllamav2_HF with models in the EXL2 format for the best performance. GPTQ models can also work but may be less efficient.

### Running on Your CPU

If you don’t have a powerful GPU, you can run models on your CPU. It’s slower but gets the job done. For this, GGUF models are the way to go, loaded using llama.cpp, kobold.cpp, or LM Studio. To speed things up a bit, you can offload part of the model to your GPU memory if you have one.

## Popular Model Formats

---

### GGUF

This format is ideal for CPUs and works with llama.cpp or koboldcpp. The “Q-number” in model names (e.g., Q4_k_m) gives a rough idea of bits per weight (bpw), though actual bpw is slightly higher.

### EXL2

Designed for GPUs with Exllamav2_HF, EXL2 supports quantization from ~2.3 to 8 bpw. At the high end, it’s nearly lossless.

### GPTQ

An older GPU-only format. While Exllamav2 can handle GPTQ models, it’s limited to 4 bpw variants. Other variants require AutoGPTQ for loading.

## Enhancing Your Roleplaying Experience

---

### Dynamic Character Interactions

Use models with fine-tuned instruction-following abilities for better character dialogues or custom LoRA-tuned variants, which can help create more nuanced responses.

### Long-Term Memory with RAG

Retrieval-Augmented Generation (RAG) systems can mimic memory by fetching relevant information from a database and appending it to the query. This allows the model to maintain consistency over long conversations and remember key details from previous interactions. For instance, if a player mentions a specific event or character, the model can recall and reference it in future dialogues, enhancing the continuity of the story.

## Suggested Models for Roleplaying

---

### High-End GPUs (24+ GB VRAM)

#### Meta-Llama-3.1-405B-Instruct

Comparable to GPT-4o with exceptional instruction-following capabilities.

#### Mistral Large Instruct v2407 (123B)

Great for uncensored and nuanced dialogues, though slow unless you have significant VRAM.

### Mid-Range GPUs (12-16 GB VRAM)

#### gemma-2-27b-it

Balanced for SFW and intelligent roleplaying.

#### Llama 3.1 70B Instruct

Fully offloadable and functional even with lower bpw.

### Lower-End GPUs (6-10 GB VRAM)

#### Mistral-Nemo-Instruct-2407 (12B)

Ideal for uncensored dialogues.

#### Gemma-2-9B-It-SPPO-Iter3

Versatile and efficient.

### CPUs

#### dolphin-2.6-mistral-7b-dpo-laser.Q5_K_M.gguf

Requires $\sim 10$ GB RAM at Q5_K_M.

#### openchat-3.5-0106.Q5_K_M.gguf (7B)

Fast and reliable for low-resource setups.

---

Explore, experiment, and enjoy crafting immersive roleplaying adventures with language models!
