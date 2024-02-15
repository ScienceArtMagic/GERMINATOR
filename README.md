# GERMINATOR: Generalist and Expert Rank Mixer Initialization Network of Approximations Taming Optimized Randomization

### Mike Bybee, Science/Art/Magic

![GERMINATOR](assets/GERMINATOR_base2.png)

Consider this "paper" to be a living document. If published to a preprint server (e.g. [arXiv](https://arxiv.org)), best efforts will be made to ensure that updated versions are also published at appropriate releases.

## Abstract

The overwhelming majority of recent work in neural network compression and parameter efficiency has focused on techniques such as quantization, structured or unstructured pruning, knowledge distillation, or low-rank approximation and similar adaptations (and even then, the latter primarily as alternatives to full fine-tuning).

The Mixture of Experts (MoE) architecture has emerged as a way to gain the additional benefits of overparameterization realized in massive models while keeping compute requirements during inference (and, to varying degrees, training) lower; however, MoE models still require excessive amounts of storage, and their (usually) learned gating mechanisms are often complicated and error-prone. 

Hyperparameters are optimized manually or through AutoML techniques. Still, experimental code and literature on the very seeds used to set pseudorandom number generators for model weight (and bias) initialization is, relatively speaking, virtually nonexistent.

This project introduces GERMINATOR, a potentially trillionth-scale hybrid language model architecture, leveraging (pseudo)random seeds for parameter initialization, shift, scale, and sparse-to-dense prune-mixing supermasks, as learnable scalar parameters. The millions-to-trillions of parameters typically stored in model checkpoints are instead generated on the fly - and only the parameters of the current, previous, and next layers must be loaded on the target device's memory at any given time (during training and inference).

Code will be made available at [this URL](https://github.com/ScienceArtMagic/GERMINATOR).

## Introduction

TODO: Introduce problem(s), hypothesis/hypotheses

## Related Work

TODO: Shoutouts

## GERMINATOR

### Generalist and (Mixture of) Expert(s)

Expert selection is incredibly simple, leveraging hard-coded modulo operator gates (`token % n_experts`); however, unlike contemporary works in which experts are generally selected by token (or vice versa) via an often far more complicated learnable gate with top_k, softmax, or other nonlinearity with a single split for `n_experts`, GERMINATOR uses multiple `n_experts` splits per layer. The only learnable parameter for expert specialization is a rank scalar, for each expert in every `n_experts` split. The result is an extremely parameter-efficient (in memory at runtime, and especially on storage) mixture of experts, yet one that is robustly capable of generalization, specialization, and continual learning without catastrophic forgetting.

### Rank Mixer

TODO: Mixing heterogenous low-rank matrices via rank stacking and through bilinear layers

### Initialization Network of Approximations

Inspired by Hypernetworks, small neural networks trained to generate the weights of much larger models, GERMINATOR simplifies the process by learning the 'seeds' for pseudorandom number generators, typically used for reproducible weight initializations in experimental setups (e.g. `torch.manual_seed()`). These are the most complex parameters GERMINATOR needs to learn, as they are 64-bit signed integer scalars. This includes seeds for weights and biases, pruning and sign supermasks, and fine-grained shifting and scaling of post-supermask weights.

All other parameters are 32-bit signed floating-point, 32- or 8-bit signed integer, or 8-bit unsigned integer scalars. Some of these are learnable, such as course-grained scaling scalars (32-bit float). Weights and biases are initialized (at training or inference, and unlearnable) as signed and unsigned 8-bit low-rank tensors, but computed as 32-bit floating-point after division by 128 (signed, as `abs(tensor)` is `2 ** 7` in `int8`) or 256 (unsigned, as `tensor` is `2 ** 8` in `uint8`) and matrix multiplication.

### Taming Optimized Randomization

While "random" seed training has the intuitive potential for unique challenges, this project's primary hypothesis is that by eliminating the need to train every individual weight and bias - potentially reducing trainable parameters from tens of thousands or more per layer, to mere dozens - the learning of representations so difficult to predict or linearly adjust can become a worthwhile tradeoff, from both efficiency and performance standpoints.

Because the learnable parameters are seeding pseudorandom number generators, the actual generated and modified tensors are effectively frozen during both training and inference (which are essentially the same as far as the on-the-fly generated model weights and required compute are concerned).

TODO: How to train/optimizer setup for up to 64-bit random seeds

## Experiments

TODO: Do science

## Limitations

TODO: Be real

## References

TODO: More shoutouts

## Acknowledgements

[Evin Tunador](https://youtube.com/@Tunadorable), for keeping the author abreast of the latest papers, often on subjects not sought out (several of which have guided trains of thought when least expected). His YouTube channel and newsletter are great for keeping up with such papers. His Discord server is the author's favorite place to spitball ideas followed by vibrant and active-minded discussion, even when the same ideas are ignored or dismissed on more established, prominent servers.

[Bytez](https://bytez.com), for text-to-speech playback of papers that have saved the author countless hours of reading when multitasking was required.

TODO: More/more detailed shoutouts

## License

You must provide your email address, social security number, and title to your firstborn child before visiting this repository. You agree not to use this model within 500 feet of other AI models, nor do anything the author may decide, at a later and as-yet-unspecified lunar phase, that you're not allowed to do. Being a Yankees fan immediately revokes your right to use this model, in perpetuity.

Kidding. It's [MIT](LICENSE). Do what feels good. Be kind, don't harm, love life. Not because you're required, but because you're free to do what you want. Make money, or don't. Your choice. Abide by the few paragraphs in [LICENSE](LICENSE), and you're golden.

Attribution would be awesome. See [Citation](#Citation) or just link to [this repo](https://github/ScienceArtMagic/GERMINATOR).

## Contributing

It would be pretty cool if you decide to contribute to the project in any way. Code, docs, feedback/reviews/issues, or just starring the repo. All are helpful and welcome. 

If you choose to contribute or recommend datasets, be warned that synthetic data will only be accepted if generated by permissively licensed models (with no commercial restrictions, including but not limited to CC-NC and other noncommercial variants, obviously no GPT-3/-4 nor other proprietary models). If attribution is required (including but not limited to CC-BY and other attribution-clause variants), please include it so that any GERMINATOR models trained on it can honor the clause and give creators the acknowledgment they request/deserve.

## Citation

```bibtex
@software{Bybee_GERMINATOR_Generalist_and,
author = {Bybee, Mike},
license = {MIT},
title = {{GERMINATOR: Generalist and Expert Rank Mixer Initialization Network of Approximations Taming Optimized Randomization}},
url = {https://github.com/ScienceArtMagic/GERMINATOR}
}
```
