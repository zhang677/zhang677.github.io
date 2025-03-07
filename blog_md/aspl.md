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

<figure>
  <img src="assets/img/step_intro.jpg" alt="Alt text describing the image">
  <figcaption><b>Figure 1: </b> STeP is an ASPL that abstracts the execution on a DSA called reconfigurable dataflow architecture. As an analogy, STeP for reconfigurable dataflow architecture is like CUDA for NVIDIA GPUs. As a task of ML library development, a simplified mixture-of-expert (MoE) module is implemented using STeP. </figcaption>
</figure>