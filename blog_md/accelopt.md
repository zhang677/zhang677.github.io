---
layout: blog
title: "AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization"
date: 2025-12-22
authors:
  - genghan
---
The code for AccelOpt is open-source and available on GitHub: [https://github.com/zhang677/AccelOpt](https://github.com/zhang677/AccelOpt).
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

# Real-world kernel challenge: NKIBench
We construct NKIBench, the first benchmark suite for NKI kernel optimization on Amazon Trainium, with all kernels derived from real-world LLM workloads. NKIBench measures kernel performance against theoretical peak hardware performance on Trainium, rather than relying solely on relative speedup metrics, which can be ambiguous due to different baseline choices. AccelOpt boosts the average percentage of peak throughput **from 49% to 61%** on Trainium 1 and **from 45% to 59%** on Trainium 2 for NKIBench kernels.

<div class="figure">
  <img src="/assets/img/ratio_bars_trn1.png" alt="AccelOpt-NKIBench">
  <div class="caption">
    <strong>Figure 4</strong> Per-task kernel improvement of NKIBench achieved using Claude Sonnet 4 and AccelOpt (gpt-oss-120b + Qwen3-Coder-480B) on Trainium 1.
  </div>
</div>
<br>

# Optimization memory and beam search both improve the efficiency of test-time scaling 
Beam search can discover better kernels by building upon the best kernels identified in previous iterations. Optimization memory concentrates the LLM agents on techniques that have been proved effective similar to how RLVR improves pass@1 but not pass@n. AccelOpt is highly cost-effective: using open-source models, it **matches the kernel improvements** of Claude Sonnet 4 while being **26× cheaper**.

<div class="figure">
  <img src="/assets/img/accelopt-cost.png" alt="AccelOpt-Cost">
  <div class="caption">
    <strong>Figure 5</strong> Left: AccelOpt improves the cost efficiency of test-time scaling by 26x. Right: Beam search discovers better kernels and optimization memory saves iterations.
  </div>
</div>
<br>

# "Aha moment" and the Educational Impact
As a CA for Stanford’s CS149 (Parallel Computing), I leveraged AccelOpt to design problems for [Assignment 4: Fused Conv+MaxPool on the Trainium2 Accelerator](https://github.com/stanford-cs149/asst4-trainium2). During the 13th iteration of the optimization process, AccelOpt achieved a breakthrough by transforming a temporally sequential execution pattern into simultaneous spatial execution. This transformation resulted in a **5x speedup** for last year's reference Conv2D kernel on small-scale inputs. 
<div class="figure" style="text-align: center;">
  <img src="/assets/img/09292025_blog.png" alt="Aha">
  <div class="caption">
    <strong>Figure 6</strong> "Aha moment" of AccelOpt.
  </div>
</div>
<br>

In collaboration with other CAs, we developed four extra-credit problems centered on this optimization technique. While the challenge was substantial, with approximately half of the class unable to complete it, 1/3 of the students successfully mastering this critical 'spatial thinking' concept, earning full extra credit. 
<div class="figure" style="text-align: center;">
  <img src="/assets/img/conv2d_credits_count.png" alt="CS149" style="width: 60%; height: auto">
  <div class="caption">
    <strong>Figure 7</strong> Human students' performance on the optimization discovered by LLM-powered AccelOpt.
  </div>
</div>
<br>
This example underscores the generality of AccelOpt beyond NKIBench and highlights the educational impact of LLM-assisted kernel optimization.

# What about GPU? Preliminary Results on FlashInfer-Bench
AccelOpt is a hardware-agnostic framework. To demonstrate its efficacy, we evaluated its performance on the H100 SXM5 platform using nine categories of the [FlashInfer-Bench](https://bench.flashinfer.ai/) best Triton baselines (as of December 23, 2025). Utilizing the gpt-oss-120b model, AccelOpt discovered significant kernel enhancements, achieving up to a 3.49x speedup. We are currently collaborating with the FlashInfer-Bench team to integrate these optimized kernels into the public benchmark.
<div class="figure" style="text-align: center;">
  <img src="/assets/img/accelopt-fib.png" alt="Fib">
  <div class="caption">
    <strong>Figure 8</strong> Applying AccelOpt to FlashInfer-Bench.
  </div>
</div>
<br>

# Acknowledgement
This work was a collaborative effort across several teams. I would like to thank our collaborators from Stanford, AWS, and the University of Toronto: Shaowei Zhu, Anjiang Wei, Zhenyu Song, Allen Nie, Zhen Jia, Nandita Vijaykumar, Yida Wang, and Kunle Olukotun.
Special thanks to Stanford CS149 instructors Kayvon Fatahalian and Kunle Olukotun, as well as the CA team—particularly our "Assignment 4 gurus" Weixin Yu and Anthony Zhan—and the AWS support team for their essential role in the course. Finally, I’m grateful to Yixin Dong from the FlashInfer-Bench team for his insights and collaboration.

<div class="figure" style="text-align: center;">
  <img src="/assets/img/accelopt-nano-banana.jpeg" alt="AccelOpt" style="width: 60%; height: auto">
  <div class="caption">
    <strong>AccelOpt Cooking Kernels on the Moon</strong> Generated by Nano Banana Pro
  </div>
</div>
<br>

