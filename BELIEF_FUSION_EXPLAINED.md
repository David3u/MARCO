# How Belief Fusion Works in MARCO

## Overview

Belief fusion is how MARCO combines opinions from multiple experts to make a final decision. Instead of just picking one expert's answer, MARCO mathematically combines their "beliefs" about which solution is correct.

## How Belief Fusion Works

### 1. Hypotheses (Possible Solutions)

When solving a task, each expert generates "hypotheses" - potential solutions. For example:
- **GPT-OSS**: Generates hypothesis H1 (a grid solution)
- **Phi-3**: Generates hypothesis H2 (another grid solution)
- **Qwen-1.5**: Generates hypothesis H3 (yet another solution)

### 2. Expert Beliefs (Confidence Scores)

Each expert assigns a "belief" (confidence) to each hypothesis:

```
           H1    H2    H3
GPT-OSS   [0.8,  0.1,  0.1]  <- 80% confident in H1
Phi-3     [0.2,  0.7,  0.1]  <- 70% confident in H2
Qwen-1.5  [0.3,  0.2,  0.5]  <- 50% confident in H3
```

### 3. Expert Weights (Learned Trust)

MARCO learns how much to trust each expert using a neural network. For this task, it might assign:
```
GPT-OSS:   weight = 2.0  (trust it more)
Phi-3:     weight = 1.5
Qwen-1.5:  weight = 1.0  (trust it less)
```

### 4. Belief Fusion Formula

MARCO uses a **weighted product** to combine beliefs:

```python
# For each hypothesis
fused[H1] = (belief_gpt[H1] ^ weight_gpt) ×
            (belief_phi3[H1] ^ weight_phi3) ×
            (belief_qwen[H1] ^ weight_qwen)

# Then normalize so they sum to 1.0
```

**Concrete calculation:**
```
fused[H1] = (0.8^2.0) × (0.2^1.5) × (0.3^1.0)
          = 0.64 × 0.089 × 0.3
          = 0.017

fused[H2] = (0.1^2.0) × (0.7^1.5) × (0.2^1.0)
          = 0.01 × 0.585 × 0.2
          = 0.0012

fused[H3] = (0.1^2.0) × (0.1^1.5) × (0.5^1.0)
          = 0.01 × 0.032 × 0.5
          = 0.00016
```

After normalization:
```
fused[H1] = 0.92  <- H1 wins!
fused[H2] = 0.06
fused[H3] = 0.01
```

## Dual Learning Mechanism

The MCU learns through **both**:

1. **Neural network training** (gradient descent on weight predictions)
   - Learns complex patterns: "Tasks with high spatial symmetry + low color diversity → trust GPT-OSS"
   - Generalizes to unseen task types
   - Fast adaptation through backpropagation

2. **Performance matrix accumulation** (historical statistics)
   - Stores concrete results: "GPT-OSS solved 85% of rotation tasks"
   - Provides stable, empirical evidence
   - Guards against overfitting

This dual approach allows MARCO to learn patterns like:
- **GPT-OSS** is best for geometric transformations and large grids
- **Phi-3** excels at simple pattern recognition
- **Qwen-1.5** is efficient for small grids with low complexity

Over hundreds of training tasks, the MCU becomes increasingly skilled at selecting the optimal expert combination for each new task.
