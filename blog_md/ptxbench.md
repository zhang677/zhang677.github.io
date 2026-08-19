---
layout: blog
title: "PTXBench: What about just CUDA-PTX?"
date: 2026-08-17
authors:
  - genghan
---

The code for PTXBench is available [here](https://github.com/zhang677/PTXBench).

# Why not directly generate CUDA-PTX?

PTX is the lowest-level GPU interface that CUDA programmers can explicitly control, so generating CUDA with inline, architecture-specific PTX offers the shortest path from a new hardware feature to a working kernel. This approach has traditionally looked unattractive: PTX is difficult to program and validate, while abstractions such as Triton and CuTeDSL provide productivity and portability. Yet those abstractions must continually absorb new instructions, layouts, and synchronization mechanisms through compiler engineering. As GPU architectures evolve faster and LLMs become better at code generation and iterative repair, directly generating CUDA-PTX becomes appealing as a way to use new hardware capabilities before the higher-level software stack fully catches up.

# What does PTXBench ask?

PTXBench asks how well current LLMs can reason about architecture-specific PTX on H100 and B200 GPUs, not merely whether they can emit a fast CUDA kernel. A model receives an architecture-specific knowledge pack, writes CUDA-PTX, and revises it over multiple turns using compilation, sanitization, correctness, and performance feedback. The benchmark separately checks whether the kernel is functionally correct, whether the requested instruction family actually executes at runtime, and whether the kernel is competitive with frontier libraries. This separation also points to the techniques that will matter next: execution-grounded repair, targeted post-training, runtime instruction verification, and much stronger testing infrastructure.

<div class="figure">
  <img src="/assets/img/ptxbench-v2.drawio.png" alt="PTXBench workflow from benchmark setup through iterative CUDA-PTX generation, measurement, and repair-conditioned adaptation">
</div>
<br>

# Takeaway 1: GEMM is close; attention is not

Frontier LLMs are beginning to make architecture-specific PTX work, but capability falls sharply as the workload becomes more complex. GEMM is closest to being solved: Claude Opus 4.8 reaches 1.012x cuBLAS performance on Blackwell, while Gemini 3.1 Pro reaches 0.892x. Attention remains substantially harder, especially on Blackwell, and backward attention is harder still. PTXBench measures performance against frontier libraries: cuBLAS 13.1 for GEMM, cuDNN 9.20.0 for the primary attention workloads, and FlashInfer 0.6.14 for GQA. Speedup is the reference-library latency divided by the generated kernel's latency, so 1.0x means matching the corresponding performance baseline and values above 1.0x mean surpassing it.

Figure 1 uses two different kinds of ratios. The x-axis, \(p\), is a speedup threshold relative to the library baseline; the y-axis, Fast<sub>p</sub>, is the percentage of evaluated turns that produced a correct kernel with speedup greater than \(p\). For example, Fast<sub>0.8</sub> = 40% means that 40% of turns produced a correct kernel exceeding 0.8x baseline performance. The “Target inst.” rows additionally require the kernel to execute a selected target instruction at runtime.

<div class="figure">
  <img src="/assets/img/ptxbench-blackwell-hopper-fast-at-p-prompt-ranges-vertical.png" alt="Correct kernels meeting each speedup threshold on Blackwell and Hopper, before and after requiring selected target instructions to execute at runtime" style="display: block;">
  <div class="caption">
    <strong>Figure 1</strong> Correct kernels meeting each speedup threshold (Fast<sub>p</sub>) on Blackwell (top two panels) and Hopper (bottom two panels), before and after requiring selected target instructions to execute at runtime. GEMM is much further along than attention, particularly backward attention.
  </div>
</div>
<br>

# Takeaway 2: A small, well-constructed SFT dataset can pay off

Specializing a model for CUDA-PTX may not require an enormous corpus. PTXBench adapts Qwen3.6-27B with Fixit examples built from the base model's failed kernels, execution feedback, teacher repairs, and synthesized repair rationales. The smallest balanced recipe that solves all five primary evaluation problems contains only 158 records, whereas the base model produces no correct kernel on any Hopper workload in the main model comparison. The improvement is meaningful but not universal: the adapted model transfers to GEMM, all four head-dimension-64 attention tasks, and two head-dimension-96 forward tasks, yet fails on head-dimension-96 backward attention and GQA. Coverage, balance, and teacher quality therefore matter at least as much as raw dataset size.

<div class="figure">
  <img src="/assets/img/fixitv2_generalization_heatmap.png" alt="Generalization of a repair-conditioned PTX model across held-out workloads">
  <div class="caption">
    <strong>Figure 2</strong> Qwen3.6-27B-PTX-SFT generalizes beyond its four training tasks, but transfer remains uneven across head dimensions and attention variants.
  </div>
</div>
<br>

Cross-language transfer is more mixed (Figure 3). On the same five Hopper workloads, using the same checkpoint to generate Triton lowers turn-level correctness relative to the base model on every workload, yet raises the best correct speedup on the causal variants from 0.238x to 0.632x for MHA-Fwd-Causal and from 0.043x to 0.331x for MHA-Bwd-Causal. The SFT recipe can therefore improve peak performance substantially even while making correct Triton kernels less likely.

<div class="figure">
  <img src="/assets/img/qwen36_ptx_sft_base_triton_fast_at_p_prompt_range.png" alt="Triton transfer comparison between the PTX-SFT and base Qwen3.6-27B checkpoints on five Hopper workloads">
  <div class="caption">
    <strong>Figure 3</strong> PTX SFT reduces turn-level correctness under Triton transfer, but substantially improves the best correct speedup on both causal attention workloads.
  </div>
</div>
<br>

# Takeaway 3: Abstractions still matter, but cracks are appearing

Higher-level abstractions still provide a major robustness advantage. When Gemini 3.1 Pro generates Triton and CUDA-PTX for the same tasks, the two are relatively close on Hopper: CUDA-PTX nearly matches Triton on GEMM and slightly exceeds its peak on causal MHA forward, 0.768x versus 0.759x the frontier-library baseline. On Blackwell attention, however, the gap widens dramatically; Triton reaches 0.484x and 0.436x on the two backward workloads, compared with only 0.133x and 0.015x for CUDA-PTX. Compilers therefore remain essential, especially on newer architectures, but the selective Hopper wins are the first cracks in the assumption that an LLM must always work through a higher-level kernel language.

<div class="figure">
  <img src="/assets/img/gemini31_triton_cuda_h100_b200_fast_at_p.png" alt="Gemini 3.1 Pro comparison between Triton and CUDA-PTX">
  <div class="caption">
    <strong>Figure 4</strong> Triton remains more robust overall, while direct CUDA-PTX is already competitive on selected Hopper workloads.
  </div>
</div>
<br>

# Takeaway 4: Testing will become the bottleneck

Finding kernels that are both correct and safe requires repeatedly running expensive performance measurements. PTXBench reduces this cost by leveraging the temporary locality of kernel evaluation requests for parallel agent loops. Specifically, PTXBench spawns a seperate host proccess to cache workload state such as input tensors, reference outputs, and reference latencies in the GPU memory. Reusing that state improves kernel evaluation throughput by 2.24x, as Figure 5 shows. Even with reuse, however, evaluation still leaves a headroom 2.72x to address in the future.

<div class="figure" style="text-align: center;">
  <img src="/assets/img/baseline_cache_cumulative_runtime_large_font.png" alt="Cumulative profiling runtime with and without cached workload state" style="width: 50%; height: auto;">
  <div class="caption">
    <strong>Figure 5</strong> Reusing stable workload state makes evaluation 2.24x faster, but the cached path still takes 2.72x as long as kernel execution alone.
  </div>
</div>
<br>

# What's Next?

In the future, two priorities follow. First, optimize the checkers. NVIDIA Compute Sanitizer's `racecheck`, for example, can take more than 1000x as long as native execution, while LLM serving has benefited from far greater investment. Faster incremental checks, cache-aware sanitization, and better overlap between generation and profiling would let agents learn from more execution feedback in the same time. Second, robustify the checkers. Output comparison alone cannot catch memory-safety bugs, races, asynchronous-lifetime violations, or every runtime failure.[^cuda-free-async] Future checkers must make these properties, runtime error tracing, and an explicit unknown state first-class signals. As kernel agents explore increasingly adversarial corners of CUDA semantics, checker speed will determine how quickly they improve, and checker robustness will determine whether their apparent wins are real.

[^cuda-free-async]: Undefined behavior is an important caveat. One numerically correct PTXBench kernel called `cudaFreeAsync` on temporary storage and then enqueued another consumer of that storage on the same stream. This stream-ordered use-after-free is [undefined under the CUDA runtime](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html): it may crash, silently corrupt results, or appear to work. In this case, the correctness checks passed, but Nsight Compute profiling failed. We therefore retained the functional-correctness result while marking runtime SASS usage as unknown rather than positive or negative.

# Acknowledgments

PTXBench is a collaborative effort by Genghan Zhang, Yixin Dong, Chengze Fan, Zhichen Zeng, Yueming Yuan, Shaowei Zhu, and Kunle Olukotun. We are grateful to members of RadixArk, the SGLang community, and the Stanford Pervasive Parallelism Lab for their technical support, insightful discussions, and help. The project received generous support from the Gemini Academic Program and the Tinker Research Grant.
