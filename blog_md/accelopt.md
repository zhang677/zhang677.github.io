---
layout: blog
title: "AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization"
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
To tackle this challenging task, we propose AccelOpt, the first **self-improving LLM agentic system** for kernel optimization on emerging AI accelerators that **combines search with memory accumulation**. Among the open-source systems we are aware of, AccelOpt is the first system that does not require expert-provided, hardware-specific optimization knowledge or predefined optimization recipes on emerging AI accelerators.

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
    <strong>Figure 4</strong> Per-task kernel improvement of NKIBench achieved using Claude Sonnet 4 and AccelOpt (gpt-oss-120b + Qwen3-Coder-480B) on Trainium 1.
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

# What about GPU?
AccelOpt is a hardware-agnostic framework. To demonstrate its efficacy, we evaluated its performance on the H100 SXM5 platform using nine categories of the [FlashInfer-Bench](https://bench.flashinfer.ai/) best Triton baselines (as of December 23, 2025). Utilizing the gpt-oss-120b model, AccelOpt discovered significant kernel enhancements, achieving up to a 3.49x speedup. [Here](https://github.com/zhang677/AccelOpt/tree/e445784df36af4c73ed5b77ecec97fe14f6d52eb/experiments/flb_full_complete_local/results/12-21-17-05) are all the generated kernels. We are currently collaborating with the FlashInfer-Bench team to integrate these optimized kernels into the public benchmark.
<div class="figure" style="text-align: center;">
  <img src="/assets/img/accelopt-fib.png" alt="Fib">
  <div class="caption">
    <strong>Figure 8</strong> Applying AccelOpt to FlashInfer-Bench.
  </div>
</div>
<br>

# Let the Agent Flow: Beyond Human Guidance
A notable finding of AccelOpt is that the base prompt does not need to include full syntax information for NKI—a relatively new architecture-specific programming language ([ASPL](https://zhang677.github.io/blog_md/aspl.html)) released in 2024. It is often assumed that LLMs must master the syntax of ASPLs before optimizing kernels<sup><a id="ref3-return" href="#ref3">3</a></sup>. However, as pointed out in [this blog](https://zhang677.github.io/blog_md/reason.html), many ASPLs' syntax have a structural similarity to one another. Therefore, **syntax is not the primary consideration**; the priority is building a workflow that enables continuous improvement. 

Within this continuous improvement workflow, AccelOpt demonstrates that **human-crafted optimization guidance is unnecessary**. It was previously assumed that designers needed to provide specific instructions, such as optimization plans for the LLM to select from<sup><a id="ref1-return" href="#ref1">1</a></sup>. Consequently, porting prompts from NKI to Triton simply involves replacing NKI-specific details with a few sentences about Triton. They can even share the same summarizer prompts. Interested readers can read the prompts for [NKI](https://github.com/zhang677/AccelOpt/tree/2e8ccd92c5d95844f68b0d53392e974084a05bf4/prompts) and [Triton](https://github.com/zhang677/AccelOpt/tree/2e8ccd92c5d95844f68b0d53392e974084a05bf4/prompts/flb).

# Benchmarks as Environments for Self-Improvement
Developed concurrently with FlashInfer-Bench, NKIBench utilizes a similar interface that features structured storage for problems and kernels along with a scalable profiling service. This represents an advancement over traditional kernel benchmarks, such as KernelBench<sup><a id="ref4-return" href="#ref4">4</a></sup>, which consist only of a list of problems. By providing more than just a fixed repository of tasks, this new approach creates a comprehensive environment for the development of self-improving LLM agents.

# What about RL?
**TL;DR**: An effective training recipe has not yet been identified.
Our dataset consists of ~18k "slow-fast" pairs derived from 16 iterations of AccelOpt across 14 distinct problems. Additionally, we leveraged an in-house NKI program generator to produce 1k kernels for training and 150 for evaluation. Supervised Fine-Tuning (SFT) was conducted using two distinct tasks: (1) slow-to-fast NKI kernel optimization and (2) Numpy-to-NKI translation.
Reinforcement Learning (RL) was then applied using [GRPO](https://arxiv.org/pdf/2402.03300) via [verl](https://github.com/volcengine/verl/tree/main), trained on the 1k NKI kernel dataset. The reward function was designed to penalize empty outputs, compilation errors, and performance regressions, while incentivizing functional correctness and execution speedup. We evaluated four distinct recipes using a Best-of-5 sampling strategy. Metrics include the number of correct cases, the count of cases achieving a speedup ≥ 1, and the average speedup among those cases with speedup ≥ 1.

**Qwen2.5-Coder-7B-Instruct**

| Recipe | Correct Cases | Cases with Speedup ≥ 1 | Avg. Speedup |
| -------- | -------- | -------- | -------- |
| Base model | 126 / 150 | 39 / 150 | 1.022 |
| SFT-Slow-Fast | 127 / 150 | 31 / 150 | 1.034 |
| SFT-Slow-Fast-RL | 131 / 150 | 53 / 150 | 1.024 |
| SFT-Numpy-NKI | 125 / 150 | 75 / 150 | 1.029 |
| SFT-Numpy-NKI-RL | 132 / 150 | 52 / 150 | 1.018 |

**DeepSeek-Coder-33B-Base-Instruct**

| Recipe | Correct Cases | Cases with Speedup ≥ 1 | Avg. Speedup |
| -------- | -------- | -------- | -------- |
| Base model | 105 / 150 | 26 / 150 | 1.025 |
| SFT-Slow-Fast | 70 / 150 | 66 / 150 | 1.040 |
| SFT-Slow-Fast-RL | 104 / 150 | 57 / 150 | 1.029 |
| SFT-Numpy-NKI | 105 / 150 | 49 / 150 | 1.032 |
| SFT-Numpy-NKI-RL | 108 / 150 | 67 / 150 | 1.025 |


# Acknowledgement
This work was a collaborative effort across several teams. I would like to thank our collaborators from Stanford University, AWS Neuron Science Team, and the University of Toronto: Shaowei Zhu, Anjiang Wei, Zhenyu Song, Allen Nie, Zhen Jia, Nandita Vijaykumar, Yida Wang, and Kunle Olukotun.
Special thanks to Stanford CS149 instructors Kayvon Fatahalian and Kunle Olukotun, as well as the amazing CA team—particularly our "Assignment 4 gurus" Weixin Yu and Anthony Zhan—and the AWS support team for their essential role in the course. Finally, I’m grateful to Yixin Dong from the FlashInfer-Bench team for his insights and collaboration.

<div class="figure" style="text-align: center;">
  <img src="/assets/img/accelopt-arts.png" alt="AccelOpt" style="width: 60%; height: auto">
  <div class="caption">
    <strong>Arts: </strong> Left: <i>AccelOpt Cooking Kernels on the Moon</i>. Right: <i>Magic of Optimization Memory</i>. <br>Both were generated by Nano Banana Pro.
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