# Expert Fine-tuning Pipeline for MARCO

Complete 3-stage fine-tuning workflow to create fully-capable MARCO experts.

## Overview

```
Stage 1: Solution Generation → Stage 2: Confidence Calibration → Stage 3: Refinement
     (30-60% accuracy)              (Calibrated beliefs)           (Iterative improvement)
```

## Stage 1: Basic Solution Generation

**Notebook**: `finetune_experts.ipynb`

**Goal**: Learn to solve ARC tasks

**Training Format**:
```
Input:  [Task examples] Test Input: [[...]]
Output: [[solution grid]]
```

**Training Data**:
- 400 ARC tasks × 3 examples = ~1200 training pairs
- Standard causal LM fine-tuning

**Expected Results**:
- Before: 5-15% accuracy (frozen LLM)
- After: 30-60% accuracy (fine-tuned)

**Time**: 2-4 hours per expert

**Run**:
```python
# In finetune_experts.ipynb
CONFIG["model_to_finetune"] = "phi3"  # Then gptoss, then qwen15
```

---

## Stage 2: Confidence Calibration

**Notebook**: `finetune_experts_with_confidence.ipynb`

**Goal**: Learn to assess own prediction quality

**Training Format**:
```
Input:  [Task examples] Provide: 1) Solution 2) Confidence
Output: Solution: [[grid]] | Confidence: 0.85
```

**Training Data**:
- Use Stage 1 fine-tuned model as base
- Create positive examples (correct → confidence 1.0)
- Create negative examples (incorrect → confidence 0.0-0.5)

**Expected Results**:
- Correct predictions: confidence 0.7-1.0
- Incorrect predictions: confidence 0.0-0.5
- Calibrated belief distribution

**Time**: 2-3 hours per expert

**Run**:
```python
# In finetune_experts_with_confidence.ipynb
# Load Stage 1 checkpoint as base model
CONFIG["base_model_path"] = "finetuned_experts/phi3_arc_finetuned"
```

---

## Stage 3: Hypothesis Refinement (TODO)

**Notebook**: `finetune_experts_refinement.ipynb` (to be created)

**Goal**: Learn to refine based on belief fusion feedback

**Training Format**:
```
Input:  [Task] Your solution: [[...]]
        Collective consensus: 0.75, Agreement: Medium
        Refine your answer.
Output: Refined: [[improved grid]] | Confidence: 0.88
```

**Training Data**:
- Use Stage 2 fine-tuned model as base
- Generate initial solutions with model
- Compare with known answer
- Create refinement prompts showing:
  - Original solution
  - Simulated belief fusion feedback
  - Correct refined solution

**Expected Results**:
- Improved accuracy on second iteration
- Learns to incorporate collective feedback
- Better convergence in MARCO system

**Time**: 2-3 hours per expert

---

## Full Pipeline Execution

### For Each Expert (GPT-OSS, Phi-3, Qwen):

```bash
# Week 1: Stage 1 - Solution Generation
python run_notebook.py finetune_experts.ipynb --model phi3
# Output: finetuned_experts/phi3_arc_finetuned/

# Week 1: Stage 2 - Confidence Calibration
python run_notebook.py finetune_experts_with_confidence.ipynb --model phi3
# Output: finetuned_experts/phi3_arc_confidence/

# Week 2: Stage 3 - Refinement (when ready)
python run_notebook.py finetune_experts_refinement.ipynb --model phi3
# Output: finetuned_experts/phi3_arc_refined/
```

### Time Estimate:
- 3 experts × 3 stages × 3 hours = **27 hours total**
- Can parallelize: 3 experts in parallel = **9 hours wall time**

---

## Integration with MARCO

### After Stage 1:
```python
# In train_marco.ipynb
"model_paths": {
    "phi3": "finetuned_experts/phi3_arc_finetuned"
}
```
- Better than frozen models (30-60% vs 5-15%)
- Basic hypothesis generation works

### After Stage 2:
```python
"model_paths": {
    "phi3": "finetuned_experts/phi3_arc_confidence"
}
```
- Calibrated confidence scores
- Better belief fusion
- Improved expert selection

### After Stage 3:
```python
"model_paths": {
    "phi3": "finetuned_experts/phi3_arc_refined"
}
```
- Full refinement capability
- Iterative improvement
- **Expected MARCO accuracy: 50-80%**

---

## Why This Order Matters

### ❌ Wrong Order: Confidence → Solution
```
Problem: Can't calibrate confidence if you can't solve tasks
Model learns: "I'm 85% confident in this random grid"
Result: Overconfident garbage
```

### ❌ Wrong Order: Refinement → Confidence
```
Problem: Can't refine without knowing uncertainty
Model learns: "Change solution randomly"
Result: No convergence
```

### ✅ Correct Order: Solution → Confidence → Refinement
```
Stage 1: Learn to solve (get 40% accuracy)
Stage 2: Learn when you're right (calibrated beliefs)
Stage 3: Learn to improve (use feedback)
Result: 50-80% final accuracy with MARCO
```

---

## Validation Between Stages

### After Stage 1:
```python
# Test solution accuracy
accuracy = evaluate_model(model, test_tasks)
assert accuracy > 0.25, "Stage 1 failed - accuracy too low"
```

### After Stage 2:
```python
# Test confidence calibration
correct_conf = avg_confidence_when_correct(model, test_tasks)
incorrect_conf = avg_confidence_when_incorrect(model, test_tasks)
assert correct_conf > 0.7, "Confidence too low on correct"
assert incorrect_conf < 0.5, "Confidence too high on incorrect"
```

### After Stage 3:
```python
# Test refinement improves accuracy
initial_acc = first_attempt_accuracy(model, test_tasks)
refined_acc = refined_attempt_accuracy(model, test_tasks)
assert refined_acc > initial_acc, "Refinement not improving"
```

---

## Current Status

- ✅ Stage 1 notebook ready: `finetune_experts.ipynb`
- ✅ Stage 2 notebook ready: `finetune_experts_with_confidence.ipynb`
- ⏳ Stage 3 notebook: **TODO** (need to create)

## Next Steps

1. Run Stage 1 for all 3 experts (can parallelize)
2. Validate Stage 1 results
3. Run Stage 2 for all 3 experts
4. Validate Stage 2 results
5. Create Stage 3 notebook
6. Run Stage 3 for all 3 experts
7. Retrain MCU with fully fine-tuned experts
