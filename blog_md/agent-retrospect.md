---
layout: blog
title: Notes on Infrastructure for Kernel Agents 
date: 2026-01-28
authors:
  - genghan
---
Devils are in the details. Environment engineering is a commitment to turning ambitious research ideas into reality, so it deserves careful consideration.

**Robust measurement**

(1) lock GPU clock frequency; 

(2) retry when latency variance is high; 

(3) use multiple random seeds and high-variance inputs (as in FlashAttention-3); 

(4) inspect “suspiciously fast” kernels (LLMs can cheat; treat >1.5× speedup as an alert);

(5) use a roofline sanity check (if FLOPs are hard to count, approximate compute intensity with `total_traffic / HBM_bandwidth`);

(6) set tolerance per problem, and keep it empirically strict enough.

**Extensible parallel profiling infrastructure**

(1) cache reference outputs and input tensors to avoid repeating expensive profiling (while keeping the profiling service modular);

(2) make it compatible with Python-async;

(3) add proper timeouts (especially for compilation);

(4) build in fault tolerance (drivers and networks are not 100% reliable).

**Logging**

(1) support checkpointing and push-button restore;

(2) prefer local tracing/monitoring over cloud tools when bandwidth is a bottleneck;

(3) store intermediate states in a comprehensive, flat schema;

(4) .gitignore large logs by default and add pre-commit to check >10MB files.

**Versioning**

(1) separate core logic (labeled by behaviors) from experiments (labeled by timestamps);

(2) keep each experiment self-contained (don’t scatter scripts across folders);

(3) record a concise description for every run;

(4) split reusable Python utilities from per-case bash scripts + JSON configs;

(5) version prompts explicitly (otherwise hard to notice the bug).

**Retry logic**

(1) remember the network isn’t 100% reliable;

(2) handle retries, backoff, and partial failures as first-class behavior (massive concurrent sampling will trigger edge cases).