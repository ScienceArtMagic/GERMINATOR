# GERMINATOR: Generalist and Expert Rank Mixer Initialization Network of Approximations Taming Optimized Randomization

![GERMINATOR](assets/GERMINATOR_base2.png)

## Abstract

The overwhelming majority of recent work in language model compression and parameter efficiency has focused on techniques such as quantization, structured or unstructured pruning, knowledge distillation, or low-rank approximation and similar adaptations (and even then, primarily as alternatives to full fine-tuning).

The Mixture of Experts (MoE) architecture has emerged as a way to gain the additional benefits of overparameterization realized in massive models while keeping compute requirements during inference (and, to varying degrees, training) lower; however, MoE models still require excessive amounts of storage, and their (usually) learned gating mechanisms are often complicated and error-prone. 

Hyperparameters are optimized manually or through AutoML techniques. Still, experimental code and literature on the very seeds used to set pseudorandom number generators for model weight (and bias) initialization is, relatively speaking, virtually nonexistent.

This project introduces GERMINATOR, a potentially trillionth-scale hybrid language model architecture, leveraging (pseudo)random seeds for parameter initialization, shift, scale, and sparse-to-dense prune-mixing as well as sign-switching supermasks, as scalar learnable parameters. The millions-to-trillions of parameters typically stored in model checkpoints are instead generated on the fly - and only the parameters of the current, previous, and next layers must be loaded on the target device's memory at any given time (during training and inference).

## Introduction

TODO: Introduce problem(s), hypothesis/hypotheses

## Related Work

TODO: Shoutouts

## Generalist and (Mixture of) Expert(s)

Expert selection is incredibly simple, leveraging hard-coded modulo operator gates (`token % n_experts`); however, unlike contemporary works in which experts are generally selected by token (or vice versa) via often far more complicated learnable top_k, softmax, or other nonlinearity with a single split for `n_experts`, GERMINATOR uses multiple `n_experts` splits per layer. The only learnable parameter for expert specialization is a rank scalar, for each expert in every `n_experts` split. The result is an extremely parameter-efficient (in memory at runtime, and especially on storage) mixture of experts, yet one that is robustly capable of generalization, specialization, and continual learning without catastrophic forgetting.

## Rank Mixer

TODO: Mixing heterogenous low-rank matrices through bilinear layers

## Initialization Network of Approximations

Inspired by Hypernetworks, small neural networks trained to generate the weights of much larger models, GERMINATOR simplifies the process by learning the 'seeds' for pseudorandom number generators, typically used for reproducible weight initializations in experimental setups (e.g. `torch.manual_seed()`). These are the most complex parameters GERMINATOR needs to learn, as they are 64-bit signed integer scalars. This includes seeds for weights and biases, pruning and sign supermasks, and fine-grained shifting and scaling of post-supermask weights.

All other parameters are 32-bit signed floating-point, 32- or 8-bit signed integer, or 8-bit unsigned integer scalars. Some of these are learnable, such as course-grained scaling scalars (32-bit float). Weights and biases are initialized (at training or inference, and unlearnable) as signed and unsigned 8-bit low-rank tensors, but computed as 32-bit floating-point after division by 128 (signed, as `abs(tensor)` is `2 ** 7` in `int8`) or 256 (unsigned, as `tensor` is `2 ** 8` in `uint8`) and matrix multiplication.

## Taming Optimized Randomization

TODO: How to train/optimizer setup for 64-bit random seeds

## Experiments

TODO: Do science

## Limitations

TODO: Be real

## References

TODO: More shoutouts

## Acknowledgements

[Evin Tunador](https://www.youtube.com/@Tunadorable)

TODO: More/more detailed shoutouts
