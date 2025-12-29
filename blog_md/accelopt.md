---
layout: blog
title: "One Agent, Any Chip: Expert-free, Self-Improving Kernel Optimization"
date: 2025-12-22
authors:
  - genghan
---
The code for AccelOpt is open-source and available on GitHub: [https://github.com/zhang677/AccelOpt](https://github.com/zhang677/AccelOpt). More details can be found in [our paper](https://arxiv.org/pdf/2511.15915). Accelopt is a follow-up to [this work](https://zhang677.github.io/blog_md/aspl.html).
# AI accelerator kernels are hard to optimize
The unprecedented demand for compute power in the age of large models has prompted the rise of AI accelerators. However, their performance critically depends on the efficiency of kernels—the low-level implementations that determine how machine learning operators are mapped onto hardware resources. Suboptimal kernels can severely limit system performance and, when scaled to large deployments, result in substantial waste of compute and financial resources. Kernel optimization, however, is notoriously difficult and demanding, even for well-understood architectures like GPUs.
<div class="figure" style="text-align: center;">
  <img src="/assets/img/fa1-4.png" alt="FA1-4">
  <div class="caption">
    <strong>Figure 1</strong> FlashAttention lags behind the release of NVIDIA GPUs by ~1 year.
  </div>
</div>
<br>

# Can LLM-based systems autonomously navigate the kernel optimization space?
There are two primary challenges. First, **LLM queries incur substantial computational costs**, this exploration must be conducted strategically to balance comprehensive search space coverage with cost efficiency. Second, we aim to enable the LLM-based system to **autonomously accumulate optimization insights during such explorations**, allowing the system to progressively improve its capabilities over time without requiring manual intervention.

<div class="figure">
  <img src="/assets/img/accelopt-main-method.png" alt="AccelOpt-Method">
  <div class="caption">
    <strong>Figure 2</strong> At each iteration of AccelOpt, the agentic workflow shown on the right optimizes the candidate kernels with the latest optimization memory, and generates new candidate kernels, updating optimization memory with newly collected experiences.
  </div>
</div>
<br>

# Solution: AccelOpt
To tackle this challenging task, we propose AccelOpt, a **self-improving LLM agentic system** for kernel optimization on emerging AI accelerators that **combines search with memory accumulation**. Among the open-source systems we are aware of, AccelOpt is the first system that does not require expert-provided, hardware-specific optimization knowledge or predefined optimization recipes on emerging AI accelerators.

<div class="figure">
  <img src="/assets/img/accelopt-example.png" alt="AccelOpt-Example">
  <div class="caption">
    <strong>Figure 3</strong> A snapshot of AccelOpt’s execution trace, which conducts a non-trivial loop transformation.
  </div>
</div>
<br>

# Real-world Kernel Challenge: NKIBench
We construct NKIBench, the first benchmark suite for NKI kernel optimization on Amazon Trainium, with all kernels derived from real-world LLM workloads. NKIBench measures kernel performance against **theoretical peak hardware performance** on Trainium, rather than relying solely on relative speedup metrics, which can be ambiguous due to different baseline choices. AccelOpt boosts the average percentage of peak throughput **from 49% to 61%** on Trainium 1 and **from 45% to 59%** on Trainium 2 for NKIBench kernels.

<div class="figure">
  <img src="/assets/img/ratio_bars_trn1.png" alt="AccelOpt-NKIBench">
  <div class="caption">
    <strong>Figure 4</strong> Per-task kernel improvement of NKIBench achieved using Claude Sonnet 4 and AccelOpt (gpt-oss-120b + Qwen3-Coder-480B) on Trainium 1. The y-axis represents percentage of peak throughput, where a higher score signifies better kernels.
  </div>
</div>
<br>

# Enhancing Test-Time Scaling via Beam Search and Optimization Memory 
AccelOpt leverages test-time scaling to unlock the potential of LLMs that may be under-trained for specialized tasks.  It employs beam search to iteratively discover better kernels by building upon the best kernels of previous iterations. This aligns with AutoComp's finding<sup><a id="ref1-return" href="#ref1">1</a></sup>. A key new finding of AccelOpt is the utility of optimization memory, which directs agents toward successful strategies. Similar to how RLVR improves pass@1 but not pass@n<sup><a id="ref2-return" href="#ref2">2</a></sup>, **optimization memory reduces the cost of test-time scaling** while achieving the same average performance. AccelOpt is highly cost-effective: using open-source models, it **matches the kernel improvements** of Claude Sonnet 4 while being **26× cheaper**.

<div class="figure">
  <img src="/assets/img/accelopt-cost.png" alt="AccelOpt-Cost">
  <div class="caption">
    <strong>Figure 5</strong> Left: AccelOpt improves the cost efficiency of test-time scaling by 26x. Right: Beam search discovers better kernels and optimization memory saves iterations.
  </div>
</div>
<br>

# "Aha moment" and the Educational Impact
As a CA for Stanford’s CS149 (Parallel Computing), I leveraged AccelOpt to design problems for [Assignment 4: Fused Conv+MaxPool on the Trainium2 Accelerator](https://github.com/stanford-cs149/asst4-trainium2). During the 14th iteration of the optimization process, AccelOpt achieved a breakthrough by transforming a temporally sequential execution pattern into simultaneous spatial execution. This transformation resulted in a **5x speedup** of last year's reference Conv2D kernel on small-scale inputs. 
<div class="figure" style="text-align: center;">
  <img src="/assets/img/09292025_blog.png" alt="Aha">
  <div class="caption">
    <strong>Figure 6</strong> "Aha moment" of AccelOpt.
  </div>
</div>
<br>

In collaboration with other CAs, we developed [four extra-credit problems](https://github.com/stanford-cs149/asst4-trainium?tab=readme-ov-file#extra-credit) centered on this optimization technique. While the challenge was substantial, with approximately half of the class unable to complete it, 1/3 of the students successfully acquired this critical 'spatial thinking' skill, earning full extra credit. 
<div class="figure" style="text-align: center;">
  <img src="/assets/img/conv2d_credits_count.png" alt="CS149" style="width: 60%; height: auto">
  <div class="caption">
    <strong>Figure 7</strong> Human students' performance on the optimization discovered by LLM-powered AccelOpt.
  </div>
</div>
<br>
This example underscores the generality of AccelOpt beyond NKIBench and highlights the educational impact of LLM-assisted kernel optimization.

# Let the Agent Flow: Beyond Human Guidance
A notable finding of AccelOpt is that the base prompt does not need to include full syntax information for NKI—a relatively new architecture-specific programming language ([ASPL](https://zhang677.github.io/blog_md/aspl.html)) released in 2024. It is often assumed that LLMs must master the syntax of ASPLs before optimizing kernels<sup><a id="ref3-return" href="#ref3">3</a></sup>. However, as pointed out in [this blog](https://zhang677.github.io/blog_md/reason.html), many ASPLs' syntax have a structural similarity to one another. Therefore, **syntax is not the primary consideration**; the priority is building a workflow that enables continuous improvement. 

Within this continuous improvement workflow, AccelOpt demonstrates that **human-crafted optimization guidance is unnecessary**. It was previously assumed that designers needed to provide specific instructions, such as optimization plans for the LLM to select from<sup><a id="ref1-return" href="#ref1">1</a></sup>. Consequently, porting prompts from NKI to Triton simply involves replacing NKI-specific details with a few sentences about Triton. They can even share the same summarizer prompts. Interested readers can read the prompts for [NKI](https://github.com/zhang677/AccelOpt/tree/2e8ccd92c5d95844f68b0d53392e974084a05bf4/prompts) and [Triton](https://github.com/zhang677/AccelOpt/tree/2e8ccd92c5d95844f68b0d53392e974084a05bf4/prompts/flb).
<div class="figure" style="text-align: center;">
  <img src="/assets/img/accelopt-cartoon.jpeg" alt="Memory" style="width: 40%; height: auto">
  <div class="caption">
    <strong>Figure 9</strong> <i>Magic of Optimization Memory</i> generated by Nano Banana Pro.
  </div>
</div>
<br>


# Benchmarks as Environments for Self-Improvement
Kernel benchmarks have three distinct properties that distinguish them from traditional benchmarks with fixed answers (such as labels or solutions):

1. **Open-ended Optimization**: Each problem remains technically "unsolved" because further optimization is always possible until the hardware's theoretical roofline bound is reached.
2. **Hardware-Dependent Evaluation**: Evaluating kernels requires directly running them on real hardware. With LLM inference and fine-tuning becoming standardized and optimized, the entire self-improvement process is often bottlenecked by kernel profiling.
3. **Production Readiness**: The surrounding ML frameworks are mature, meaning optimized kernels can be deployed directly into production.

These properties indicate that kernel ~~benchmarks~~ environments must be **scalable** for new problems, **parallelized** to evaluate heavy sampling, and **modular** for "day-zero" integration. Developed concurrently with [FlashInfer-Bench](https://bench.flashinfer.ai/), NKIBench utilizes a similar interface that features structured storage for problems and kernels along with a distributed profiling service. This represents an advancement over traditional kernel benchmarks, such as KernelBench<sup><a id="ref4-return" href="#ref4">4</a></sup>, which consist only of a list of problems. By providing more than just a fixed repository of tasks, this new approach creates a comprehensive environment for the development of self-improving LLM agents.

# What about GPU?
**AccelOpt is a hardware-agnostic framework.** To demonstrate its efficacy, we evaluated its performance on the H100 SXM5 platform using 25 of the FlashInfer-Bench best Triton baselines (as of December 23, 2025). Utilizing the gpt-oss-120b model, AccelOpt discovered significant kernel enhancements, achieving up to a 6.78x speedup. [Here](https://github.com/zhang677/AccelOpt/tree/e445784df36af4c73ed5b77ecec97fe14f6d52eb/experiments/flb_full_complete_local/results/12-21-17-05) are some of the generated kernels. We are currently collaborating with the FlashInfer-Bench team to integrate these optimized kernels into the public benchmark.
<div class="figure">
  <img src="/assets/img/accelopt-fib-25.png" alt="Fib">
  <div class="caption">
    <strong>Figure 8</strong> Performance evaluation of AccelOpt on selected FlashInfer-Bench problems. The y-axis represents latency, where a lower score signifies better kernels.
  </div>
</div>
<br>

# Can Iterative Refinement Boost Self-Improvement?
**Iterative refinement extends the limits of self-improvement** by fixing syntactic bugs in potentially fast kernels. Our implementation adds a "fixer" agent after the executor; for every sampled kernel, the fixer takes the failed code and the error log to generate a fixed version. The fixer repeats this process until the fixed kernel is correct or the allocated budget is reached. Iterative refinement is useful because LLMs often commit syntactic errors despite explicit prompt constraints. Typical errors involve unsupported behaviors like slice indexing on local accumulators and manual shared memory management. These mistakes might stem from the LLM conflating Triton's syntax with that of similar languages.
<div class="figure">
  <img src="/assets/img/speedup_comparison.png" alt="Fib">
  <div class="caption">
    <strong>Figure 9</strong> Speedup comparison between AccelOpt and AccelOpt + Iterative refinement (Fixer).
  </div>
</div>
<br>

Iterative refinement is particularly effective for complex kernels like GQA and MLA, which are more prone to errors due to their code density. Built on top of AccelOpt, iterative refinement achieves another 32.8% speedup at a total cost of only $23.70. It is worth noting that **with AccelOpt, kernel optimization can be incredibly affordable**, delivering over a 5x speedup for less than $5.
<div class="figure">
  <img src="/assets/img/cost_comparison.png" alt="Fib">
  <div class="caption">
    <strong>Figure 10</strong> Cost comparison between AccelOpt and AccelOpt + Iterative refinement (Fixer).
  </div>
</div>
<br>

# What about RL?
**TL;DR**: An effective training recipe has not yet been identified. We believe the potential of **AccelOpt for data synthesis** remains underexplored.
Leveraging AccelOpt, we synthesized ~18k 'slow-fast' pairs through 16 iterations across 14 NKIBench problems. Additionally, we leveraged an in-house framework to produce 1k NKI kernels for training and 150 for evaluation. Supervised Fine-Tuning (SFT) was conducted using two distinct tasks: (1) slow-to-fast NKI kernel transformation and (2) Numpy-to-NKI program translation.
Reinforcement Learning (RL) was then applied using [GRPO](https://arxiv.org/pdf/2402.03300) via [verl](https://github.com/volcengine/verl/tree/main), trained on the 1k NKI kernel dataset. The reward function was designed to penalize empty outputs, compilation errors, and performance regressions, while incentivizing functional correctness and execution speedup. We evaluated four distinct recipes using a Best-of-5 sampling strategy. Metrics include functional correctness, the count of cases achieving a speedup ≥ 1 and the average speedup of them, and maximum speedup across the 150 evaluated problems.

**Qwen2.5-Coder-7B-Instruct**

| Recipe | Correct Cases | Cases with Speedup ≥ 1 | Avg. Speedup | Max Speedup |
| -------- | -------- | -------- | -------- | -------- |
| Base model | 126 / 150 | 39 / 150 | 1.022 | 1.194 |
| SFT-Slow-Fast | 127 / 150 | 31 / 150 | 1.034 | 1.179 |
| SFT-Slow-Fast-RL | 131 / 150 | 53 / 150 | 1.024 | 1.161 |
| SFT-Numpy-NKI | 125 / 150 | 75 / 150 | 1.029 | 1.135 |
| SFT-Numpy-NKI-RL | 132 / 150 | 52 / 150 | 1.018 | 1.149 |

**DeepSeek-Coder-33B-Base-Instruct**

| Recipe | Correct Cases | Cases with Speedup ≥ 1 | Avg. Speedup | Max Speedup |
| -------- | -------- | -------- | -------- | -------- |
| Base model | 105 / 150 | 26 / 150 | 1.025 | 1.095 |
| SFT-Slow-Fast | 70 / 150 | 66 / 150 | 1.040 | 1.172 |
| SFT-Slow-Fast-RL | 104 / 150 | 57 / 150 | 1.029 | 1.103 |
| SFT-Numpy-NKI | 105 / 150 | 49 / 150 | 1.032 | 1.210 |
| SFT-Numpy-NKI-RL | 108 / 150 | 67 / 150 | 1.025 | 1.152 |

RL infrastructure was initiated by Anjiang Wei, who also collected the first 1k+150 kernels, and implemented the Numpy-NKI recipes. Genghan Zhang later optimized the RL infrastructure, collected the 18k 'slow-fast' kernel pairs, and implemented the Slow-Fast recipes.

# Fast-weight Engieering
Context engineering can be viewed as 'fast-weight engineering' because the context persists in the KV-cache, where it is referenced by every generated token much like static model weights. Workflows such as AccelOpt demonstrate that this is not merely ad-hoc manual prompting, but a systematic approach to fast-weight optimization. In a broader sense, just as back-propagation was designed to optimize weights, we are now seeking the most effective design for context. We are currently in an era reminiscent of the early days of neural networks, experimenting with diverse 'weight engineering' recipes to find what sticks. While 'fast-weight' may be an overloaded term in this context, I believe self-improving agents share a deep structural lineage with [Test-Time Training (TTT)](https://arxiv.org/pdf/2407.04620). Both paradigms essentially treat inference not as a static forward pass, but as a dynamic optimization process where the model’s 'state'—whether in the KV-cache or through local RNN state updates—adapts to the task (or sequence) at hand.

# Acknowledgement
This work was a collaborative effort across several teams. I would like to thank our collaborators from Stanford University, AWS Neuron Science Team, and the University of Toronto: Shaowei Zhu, Anjiang Wei, Zhenyu Song, Allen Nie, Zhen Jia, Nandita Vijaykumar, Yida Wang, and Kunle Olukotun.
Special thanks to Stanford CS149 instructors Kayvon Fatahalian and Kunle Olukotun, as well as the amazing CA team—particularly our "Assignment 4 gurus" Weixin Yu and Anthony Zhan—and the AWS support team for their essential role in the course. Finally, I’m grateful to Yixin Dong from the FlashInfer-Bench team for his insights and collaboration.

<div class="figure" style="text-align: center;">
  <img src="/assets/img/accelopt-nano-banana.jpeg" alt="AccelOpt-Moon" style="width: 40%; height: auto">
  <div class="caption">
    <strong>Appendix </strong> <i>AccelOpt Cooking Kernels on the Moon</i> generated by Nano Banana Pro.
  </div>
</div>
<br>

# References

<ol>
  <li id="ref1">Hong, C., Bhatia, S., Cheung, A. and Shao, S., 2025, May. Autocomp: Llm-driven code optimization for tensor accelerators. In Machine Learning for Computer Architecture and Systems 2025. <a href="#ref1-return">&uarr;</a></li>

  <li id="ref2">Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S. and Huang, G., 2025. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837. <a href="#ref2-return">&uarr;</a></li>

  <li id="ref3">Agrawal, L.A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M.J., Jiang, M. and Potts, C., 2025. Gepa: Reflective prompt evolution can outperform reinforcement learning. arXiv preprint arXiv:2507.19457. <a href="#ref3-return">&uarr;</a></li>

  <li id="ref4">Ouyang, A., Guo, S., Arora, S., Zhang, A.L., Hu, W., Ré, C. and Mirhoseini, A., 2025. Kernelbench: Can llms write efficient gpu kernels?. arXiv preprint arXiv:2502.10517. <a href="#ref4-return">&uarr;</a></li>
</ol>