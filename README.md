# GERMINATOR: Generalist and Expert Rank Mixer Initialization Network of Approximations Taming Optimized Randomization

Mike Bybee, Science/Art/Magic

!["a T-800 blowing a dandelion"](assets/GERMINATOR_base2.png)

Consider this "paper" to be a living document. If published to a preprint server (e.g. [arXiv](https://arxiv.org)), best efforts will be made to ensure that updated versions are also published at appropriate releases.

## Abstract

The overwhelming majority of recent work in neural network compression and parameter efficiency has focused on techniques such as quantization, structured or unstructured pruning, knowledge distillation, or low-rank approximation and similar adaptations (and even then, the latter primarily as alternatives to full fine-tuning).

The Mixture of Experts (MoE) architecture has emerged as a way to gain the additional benefits of overparameterization realized in massive models while keeping compute requirements during inference (and, to varying degrees, training) lower; however, MoE models still require excessive amounts of storage, and their (usually) learned gating mechanisms are often complicated and error-prone. 

Hyperparameters are optimized manually or through AutoML techniques. Still, experimental code and literature on the very seeds used to set pseudorandom number generators for model weight (and bias) initialization is, relatively speaking, virtually nonexistent.

This project introduces GERMINATOR, a potentially trillionth-scale hybrid language model architecture, leveraging (pseudo)random seeds for parameter initialization, shift, scale, and sparse-to-dense prune-mixing supermasks, as learnable scalar parameters. The millions-to-trillions of parameters typically stored in model checkpoints are instead generated on the fly - and only the parameters of the current, previous, and next layers must be loaded on the target device's memory at any given time (during training and inference).

Code will be made available at [https://github.com/ScienceArtMagic/GERMINATOR](https://github.com/ScienceArtMagic/GERMINATOR).

## Introduction

TODO: Introduce problem(s), hypothesis/hypotheses

## Related Work

### 2.1 Network Pruning: Playing The Efficiency Lottery

Building upon the success of earlier work such as @lecunOptimal1989, @frankleLotteryTicketHypothesis2019, research into network pruning is typically focused on the compression of existing, larger pre-trained models by discarding weights, blocks, or entire layers that contribute the least to overall model performance. This is largely orthogonal to the work of this paper, though its use of skipping is similar to block or layer pruning (albeit more dynamic).

Established pruning techniques are orthogonal, with similar efficiency goals. Approaches such as Wanda [@sunSimpleEffectivePruning2023] can prune with as little as a single sample and forward pass; therefore, future research into combining this work with such approaches might be beneficial.

GERMINATOR's use of pruning masks only temporarily results in sparsity, however, as the masked-out weights of the first source tensor are replaced by a second source tensor with the exact opposite mask (bits of the first mask are flipped via the `~`, i.e. `logical_not`, operator).

### 2.2 Random Weight Initialization: Another Game of Chance

Prior work in computer vision [@picardTorchManual_seed34072023] has demonstrated that certain outlier random seeds perform much better or worse than average. This is measured by accuracy after brief pre-training runs for models of identical architecture on multiple single-seed initializations.

@bethardWeNeedTalk2022 explores the safety/validity and/or risk of various uses of random seeds. While such delineation is certainly worthwhile, the primary focus in the current work is the most extreme parameter efficiency achievable. To that end, its use of random seeds is only to avoid the storage and CPU-GPU transfer of as many parameters as possible. While exact stability of said seeds would be ideal, any such divergence should theoretically be quickly accounted for through computationally minimal adjustment training.

This paper differs from @picardTorchManual_seed34072023 in that the random seeds themselves are trainable scalar parameters. @nooralinejadPRANCPseudoRAndom2023 uses such trainable seed scalars to initialize "basis networks,"

TODO: Shoutouts

## GERMINATOR

### Generalist and (Mixture of) Expert(s)

Expert selection is incredibly simple, leveraging hard-coded modulo operator gates (`token % n_experts`); however, unlike contemporary works in which experts are generally selected by token (or vice versa) via an often far more complicated learnable gate with top_k, softmax, or other nonlinearity in a single split for `n_experts`, GERMINATOR uses multiple `n_experts` splits per layer. The only learnable parameter for expert specialization is a rank scalar, for each expert in every `n_experts` split. The result is an extremely parameter-efficient (in memory at runtime, and especially on storage) mixture of experts, yet one that is robustly capable of generalization, specialization, and continual learning without catastrophic forgetting.

### Rank Mixer

TODO: Mixing heterogenous low-rank matrices via rank stacking and through bilinear layers

### Initialization Network of Approximations

Inspired by Hypernetworks ([Ha et al., 2016](https://arxiv.org/abs/1609.09106)), small neural networks trained to generate the weights of much larger models, GERMINATOR simplifies the process by learning the 'seeds' for pseudorandom number generators, typically used for reproducible weight initializations in experimental setups (e.g. `torch.manual_seed()`). These are some of the most complex parameters GERMINATOR needs to learn (with full, as they are up to 64-bit signed integer scalars (but can be reduced to 16- or even 8-bit integers to reduce the seed search space at any point in training). This includes seeds for weights and biases, pruning supermasks, and fine-grained shifting and scaling of post-supermask weights.

Coarse-grained scaling parameters (for the weights themselves, and for scaling the bias vectors) are learned, signed floating-point scalars. 

Weights and biases are initialized (at training and inference, unlearnable) in as low as 8-bit (signed and unsigned) low-rank integers, but computed as full-precision, 32-bit floating-point (or tf32, float16, or bfloat16 if supported by hardware), after division by the maximum absolute value of the datatype (e.g. 128 for signed 8-bit integer as `abs(torch.iinfo(tensor.dtype).min())` is `2 ** 7` in `torch.int8`, or 256 for unsigned integer as `torch.iinfo(tensor.dtype).max()` is `2 ** 8` in `torch.uint8`) and matrix multiplication.

### Taming Optimized Randomization

While "random" seed training has the intuitive potential for unique challenges, this project's primary hypothesis is that by eliminating the need to train every individual weight and bias - potentially reducing trainable parameters from tens of thousands or more per layer, to mere dozens - the learning of representations so difficult to predict or linearly adjust can become a worthwhile tradeoff, from both efficiency and performance standpoints.

Because most of the learnable parameters are seeding pseudorandom number generators, the actual generated and modified tensors are effectively frozen during both training and inference (which are essentially the same, as far as model weights generated on the fly, and required compute, are concerned); however, such scalars can't be learned through Adam, SGD, and other gradient descent methods. Not only are they integer values, but the variation between them (particularly when factoring in variable upper and lower bounds, or mean and standard deviation, of the tensors they will be used to generate) would likely be incredibly difficult and computationally expensive - if not impossible - to learn.

Instead, inspired by the seemingly unrelated LoraHub paper by [Huang, Liu, Lin, et al. (2023)](https://arxiv.org/abs/2307.13269), GERMINATOR's seeds are learned via gradient-free optimization. Upon further examination, LoraHub's training objective becomes more clearly similar to GERMINATOR's: Both need to find the best possible combinations of weights to achieve the highest possible accuracy i.e. the lowest possible loss. LoraHub's objective is much simpler, by comparison, as it only needs to determine the best `k` possible PEFT checkpoints - the best LoRA weights each applied to the entire model - to solve a given task. LoraHub benefits from both a pre-trained model, and existing PEFT adapters (and LoRAs fine-tuned on specific task categories, at that).

GERMINATOR, on the other hand, has neither and needs to generate all model weights on the fly, except for its few learnable parameters (integer seed scalars used for the generation and floating-point scalars for coarse scaling). The reduction in learnable parameters is of massive significance, but their efficient trainability hinges on being able to reliably find the "outlier" seeds identified by [Picard  (2021)](https://arxiv.org/abs/2109.08203). GERMINATOR's weights aren't just initialized and trained. They are initialized every time (and will be completely different if seeds or other learnable scalar parameters change).

While the (scalar) scaling parameters can be trained via gradient descent and thus follow a comparatively smoother path to convergence, the only way to avoid massive spikes from each seed change - unamenable to gradients as seeds are - is to scale down. Intuitively, this runs the risk of vanishing gradients.

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
