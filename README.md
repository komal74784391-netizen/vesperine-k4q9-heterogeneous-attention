# Heterogeneous Attention Routing for Multimodal Document QA

International Conference on Learning Representations, 2026 (accepted).

This branch contains the reference implementation for the paper.

## What is included

- `src/hetero/` -- heterogeneous attention heads, the token-to-head routing
  policy, and the training loop used in the reported experiments
- `configs/prototype.yaml` -- configuration for the reported runs

## Quick start

```bash
python -m src.hetero.train --epochs 2
```

If you use this code, please cite the paper listed above.
