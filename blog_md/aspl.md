---
layout: blog
title: Adaptive Self-improvement LLM Agentic System for ML Library Development
date: 2025-03-01
authors:
  - genghan
  - weixin
  - olivia
  - kunle
---

# ML library development using ASPLs

ML libraries, such as cuDNN and MIOpen, provide high-performance implementation of the operators in ML frameworks such as PyTorch and Jax. They are key to efficient ML systems. ML libraries are often written in architecture-specific programming languages (ASPLs) that target domain-specific architectures (DSAs). Figure 1 shows an example of ML library development.

<div class="figure">
  <img src="/assets/img/step_intro.png" alt="Alt text describing the image">
  <div class="caption">
    <strong>Figure 1:</strong> STeP is an ASPL that abstracts the execution on a DSA called reconfigurable dataflow architecture. As an analogy, STeP for reconfigurable dataflow architecture is like CUDA for NVIDIA GPUs. As a task of ML library development, a simplified mixture-of-expert (MoE) module is implemented using STeP.
  </div>
</div>

However, writing these high-performance ML libraries is challenging because it needs expert knowledge of ML algorithms and the ASPL. This process requires complex reasoning to compose high-level ML model components (“operators”) with low-level hardware primitives. Meanwhile, ASPLs are updated per generation of DSA, so there is limited data as shown in Figure 2.

<div class="figure">
  <img src="/assets/img/motivation.png" alt="Alt text describing the image">
  <div class="caption">
    <strong>Figure 2:</strong> ML library requires complex reasoning while minimizing data requirements. [CUTLASS](https://github.com/NVIDIA/cutlass/tree/main) is an ASPL for developing ML libraries on NVIDIA GPUs with tensor cores.
  </div>
</div>

