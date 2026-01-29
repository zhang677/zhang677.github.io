---
layout: blog
title: Notes on Building Kernel Agents 
date: 2026-01-28
authors:
  - genghan
---

1. Robust measurement: lock the clock frequency, retry on high latency variance, use multiple random seeds and high variance inputs

2. Extensible parallel profiling infra: cache the reference results and input tensors (avoid repetitive heavy CPU computation; balance with the modularity of the profiling serivce), compatible with Python async, proper timeout, fault tolerance (especially when the driver or network are not 100% reliable)

3. Logging: checkpoint and push-button restore, use local instead of cloud trace monitoring tools (network bandwidth), design comprehensive and flattened schema to store intermediate states, gitignore large logs

4. Versioning: seperate key logics and experiments (e.g. AccelOpt and AccelOpt-exps), make each experiment self-contained (don't spread the associated scripts across folders), record a precise description of every experiment, seperate reusable python scripts and per-case bash scripts + json configuration, version the prompts in different folders (otherwise the bug will be really hard to notice)

5. Retry logics: remember the network is not 100% reliable and massive concurrent sampling can trigger edge cases