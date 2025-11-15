"""
Generate and save refinement training data using all 3 Stage 2 models.

This script:
1. Loads all 3 Stage 2 models (gptoss, phi3, qwen15)
2. Generates initial solutions for ARC tasks
3. Simulates belief fusion feedback
4. Creates refinement training examples
5. Augments with color mapping and rotations
6. Saves to disk for reuse

Run this ONCE, then use the saved data for all Stage 3 training runs.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import time
import pickle
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Resolve repository paths once so file references stay correct regardless of where the
# script is invoked from.
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "training"
DEFAULT_MODEL_ROOT = REPO_ROOT / "models"
DEFAULT_FINETUNED_ROOT = REPO_ROOT / "finetuned_experts_confidence"

# Configuration
CONFIG = {
    "arc_data_path": str(DEFAULT_DATA_PATH),
    "output_file": "refinement_training_data.pkl",

    # Base model paths
    "base_model_paths": {
        "gptoss": str(DEFAULT_MODEL_ROOT / "gpt-oss"),
        "phi3": str(DEFAULT_MODEL_ROOT / "phi3"),
        "qwen15": str(DEFAULT_MODEL_ROOT / "qwen")
    },

    # Stage 2 adapter paths
    "stage2_adapter_paths": {
        "gptoss": str(DEFAULT_FINETUNED_ROOT / "gptoss_arc_confidence"),
        "phi3": str(DEFAULT_FINETUNED_ROOT / "phi3_arc_confidence"),
        "qwen15": str(DEFAULT_FINETUNED_ROOT / "qwen15_arc_confidence")
    },

    # Generation settings
    "max_tasks": 40,  # Reduced from 50 to reduce memory pressure
    "examples_per_task": 1,
    "max_new_tokens": 512,  # Increased from 150 to handle large grids (30x30 ~ 800 tokens)
    "temperature": 0.3,
    "generation_timeout": 180,  # Max seconds per generation (3 min for larger outputs)

    # Augmentation settings
    "augment_color_mapping": True,
    "augment_rotations": True,  # Add 90, 180, 270 degree rotations
    "target_examples": 350,  # Adjusted for fewer tasks (40*3*3 ~ 360)
}

def format_grid(grid):
    """Format grid as JSON"""
    return json.dumps(grid)

def grid_similarity(pred, true):
    """Calculate similarity between grids"""
    try:
        pred_arr = np.array(pred)
        true_arr = np.array(true)
        if pred_arr.shape != true_arr.shape:
            return 0.0
        return np.sum(pred_arr == true_arr) / true_arr.size
    except:
        return 0.0

def simulate_belief_fusion_feedback(initial_solution, true_solution):
    """Simulate belief fusion feedback based on correctness"""
    similarity = grid_similarity(initial_solution, true_solution)

    if similarity >= 0.9:
        consensus = 0.85 + np.random.uniform(0, 0.10)
        variance = 0.05 + np.random.uniform(0, 0.05)
    elif similarity >= 0.7:
        consensus = 0.65 + np.random.uniform(0, 0.15)
        variance = 0.10 + np.random.uniform(0, 0.10)
    elif similarity >= 0.4:
        consensus = 0.45 + np.random.uniform(0, 0.15)
        variance = 0.20 + np.random.uniform(0, 0.15)
    else:
        consensus = 0.20 + np.random.uniform(0, 0.20)
        variance = 0.30 + np.random.uniform(0, 0.20)

    return {
        'consensus': min(1.0, consensus),
        'variance': min(1.0, variance),
        'similarity': similarity
    }

def create_refinement_prompt(train_pairs, test_input, initial_solution, fusion_feedback):
    """Create refinement prompt"""
    prompt_parts = []

    prompt_parts.append("You are refining your solution to an ARC (Abstraction and Reasoning Corpus) abstract reasoning task.")
    prompt_parts.append("The system has analyzed your solution along with other experts using belief fusion.")
    prompt_parts.append("Use this collective feedback to improve your answer.\n")

    for i, (inp, out) in enumerate(train_pairs[:3], 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Input: {json.dumps(inp)}")
        prompt_parts.append(f"Output: {json.dumps(out)}\n")

    prompt_parts.append(f"Test Input: {json.dumps(test_input)}\n")
    prompt_parts.append(f"Your Current Solution:\n{json.dumps(initial_solution)}\n")

    consensus = fusion_feedback['consensus']
    variance = fusion_feedback['variance']
    agreement = "High" if variance < 0.1 else "Medium" if variance < 0.3 else "Low"

    prompt_parts.append(f"Collective Analysis:")
    prompt_parts.append(f"- Consensus Strength: {consensus:.2f}")
    prompt_parts.append(f"- Agreement Level: {agreement}\n")

    if consensus > 0.8:
        prompt_parts.append("The experts are highly confident. Your solution likely aligns with the pattern.")
        prompt_parts.append("Refine details and ensure correctness.\n")
    elif consensus > 0.6:
        prompt_parts.append("There's moderate consensus. Consider if your solution captures the core pattern.")
        prompt_parts.append("Look for areas to improve alignment.\n")
    else:
        prompt_parts.append("Consensus is low. Multiple interpretations exist.")
        prompt_parts.append("Reconsider the pattern - you may need a different approach.\n")

    prompt_parts.append("Provide your REFINED solution as a JSON array.")
    prompt_parts.append("Refined Output:")

    return "\n".join(prompt_parts)

def create_refinement_completion(refined_solution, confidence):
    """Create completion"""
    return f" {json.dumps(refined_solution)} | Confidence: {confidence:.2f}"

def generate_initial_solution(model, tokenizer, train_pairs, test_input):
    """Generate initial solution using Stage 2 model with timeout protection"""
    import signal
    from contextlib import contextmanager

    @contextmanager
    def time_limit(seconds):
        """Context manager for timeout"""
        def signal_handler(signum, frame):
            raise TimeoutError(f"Generation exceeded {seconds}s")

        # Set the signal handler and alarm
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm

    prompt_parts = [
        "You are an expert at solving abstract reasoning tasks from the ARC (Abstraction and Reasoning Corpus) challenge.\n",
        "Given input-output example pairs, identify the transformation pattern and apply it to the test input.\n",
        "Format: Provide the output grid as a JSON array and your confidence score (0.0-1.0).\n"
    ]

    for i, (inp, out) in enumerate(train_pairs[:3], 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Input: {json.dumps(inp)}")
        prompt_parts.append(f"Output: {json.dumps(out)}\n")

    prompt_parts.append(f"Test Input: {json.dumps(test_input)}")
    prompt_parts.append("Test Output:")

    prompt = "\n".join(prompt_parts)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1536, truncation=True).to(model.device)

    try:
        with time_limit(CONFIG["generation_timeout"]):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["max_new_tokens"],
                    min_new_tokens=10,  # Ensure some output
                    temperature=CONFIG["temperature"],
                    do_sample=True if CONFIG["temperature"] > 0 else False,
                    use_cache=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # Add early stopping criteria
                    num_beams=1,  # Greedy for speed
                    early_stopping=True,
                    # Stop at newlines or special patterns
                    bad_words_ids=None,
                )

        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Parse output
        import re
        json_match = re.search(r'\[\s*\[.*?\]\s*\]', generated, re.DOTALL)
        if json_match:
            try:
                solution = json.loads(json_match.group(0))
                return solution
            except:
                pass

        return None

    except TimeoutError as e:
        print(f"\n  Timeout during generation: {e}")
        return None

    finally:
        # Explicitly free GPU memory
        if 'inputs' in locals():
            del inputs
        if 'outputs' in locals():
            del outputs
        torch.cuda.empty_cache()

def load_model(model_key):
    """Load Stage 2 model"""
    print(f"\nLoading {model_key}...")
    base_path = Path(CONFIG["base_model_paths"][model_key]).expanduser().resolve()
    adapter_path = Path(CONFIG["stage2_adapter_paths"][model_key]).expanduser().resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"Base model path not found: {base_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    print(f"{model_key} loaded")
    return model, tokenizer

def generate_examples_for_model(model_key, model, tokenizer, tasks, output_file):
    """Generate examples using one model - saves incrementally to disk"""
    num_examples = 0
    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"Generating with {model_key}")
    print(f"{'='*80}")
    print(f"Tasks to process: {len(tasks)}")

    # Open file in append mode
    with open(output_file, 'ab') as f:
        for task_idx, (task_id, task_data) in enumerate(tasks, 1):
            task_start = time.time()
            try:
                all_task_examples = [(ex['input'], ex['output']) for ex in task_data.get('train', [])]
                all_task_examples += [(ex['input'], ex['output']) for ex in task_data.get('test', [])]

                if len(all_task_examples) < 2:
                    continue

                # Generate for first example only
                for i in range(min(len(all_task_examples), CONFIG["examples_per_task"])):
                    train_pairs = all_task_examples[:i] + all_task_examples[i+1:]
                    test_input, test_output = all_task_examples[i]

                    if len(train_pairs) > 3:
                        train_pairs = train_pairs[:3]

                    # Generate initial solution
                    gen_start = time.time()
                    initial_solution = generate_initial_solution(model, tokenizer, train_pairs, test_input)
                    gen_time = time.time() - gen_start

                    # Warn about slow generations
                    if gen_time > 60:
                        print(f"\n  Slow generation for task {task_id}: {gen_time:.1f}s")

                    if initial_solution is None:
                        # Use wrong answer as fallback
                        if len(all_task_examples) > 1:
                            initial_solution = all_task_examples[(i+1) % len(all_task_examples)][1]
                        else:
                            continue

                    # Simulate belief fusion
                    fusion_feedback = simulate_belief_fusion_feedback(initial_solution, test_output)

                    # Create refinement prompt
                    prompt = create_refinement_prompt(train_pairs, test_input, initial_solution, fusion_feedback)

                    # Target: correct answer with updated confidence
                    new_confidence = min(1.0, fusion_feedback['similarity'] + 0.3)
                    completion = create_refinement_completion(test_output, new_confidence)

                    example = {
                        'task_id': task_id,
                        'model_source': model_key,
                        'prompt': prompt,
                        'completion': completion,
                        'text': prompt + completion,
                        'initial_similarity': fusion_feedback['similarity'],
                        'fusion_consensus': fusion_feedback['consensus'],
                        'grids': {
                            'train_pairs': train_pairs,
                            'test_input': test_input,
                            'test_output': test_output,
                            'initial_solution': initial_solution
                        }
                    }

                    # Save immediately to disk - no memory accumulation
                    pickle.dump(example, f)
                    num_examples += 1

            except Exception as e:
                print(f"  Error with task {task_id}: {e}")

            # Periodic memory cleanup
            if task_idx % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Progress updates every task
            elapsed = time.time() - start_time
            tasks_remaining = len(tasks) - task_idx
            time_per_task = elapsed / task_idx if task_idx > 0 else 0
            eta_seconds = tasks_remaining * time_per_task

            # Show progress bar every task
            progress_pct = (task_idx / len(tasks)) * 100
            bar_length = 40
            filled = int(bar_length * task_idx / len(tasks))
            bar = '=' * filled + '-' * (bar_length - filled)

            print(f"\r  [{bar}] {progress_pct:.1f}% | Task {task_idx}/{len(tasks)} | Examples: {num_examples} | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_seconds/60:.1f}m", end='', flush=True)

            # Detailed update every 10 tasks
            if task_idx % 10 == 0:
                print()  # New line
                # Show memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"  GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    # Final newline after progress bar
    print()
    total_time = time.time() - start_time
    print(f"  Completed in {total_time/60:.1f} minutes ({total_time/num_examples:.1f}s per example)")

    return num_examples

def augment_example(example):
    """Create augmented versions of an example"""
    augmented = [example]  # Include original

    grids = example['grids']

    # Color mapping augmentation
    if CONFIG["augment_color_mapping"]:
        colors = list(range(10))
        shuffled = colors.copy()
        np.random.shuffle(shuffled)
        color_map = {old: new for old, new in zip(colors, shuffled) if old != new}

        if color_map:
            # Apply to all grids
            def apply_color_map(grid):
                grid_arr = np.array(grid)
                augmented_grid = np.copy(grid_arr)
                for old_c, new_c in color_map.items():
                    augmented_grid[grid_arr == old_c] = new_c
                return augmented_grid.tolist()

            aug_grids = {
                'train_pairs': [(apply_color_map(inp), apply_color_map(out)) for inp, out in grids['train_pairs']],
                'test_input': apply_color_map(grids['test_input']),
                'test_output': apply_color_map(grids['test_output']),
                'initial_solution': apply_color_map(grids['initial_solution'])
            }

            # Recreate prompt and completion
            fusion_feedback = {'consensus': example['fusion_consensus'], 'variance': 0.15}
            aug_prompt = create_refinement_prompt(
                aug_grids['train_pairs'],
                aug_grids['test_input'],
                aug_grids['initial_solution'],
                fusion_feedback
            )
            aug_completion = create_refinement_completion(aug_grids['test_output'], 0.85)

            aug_example = example.copy()
            aug_example['prompt'] = aug_prompt
            aug_example['completion'] = aug_completion
            aug_example['text'] = aug_prompt + aug_completion
            aug_example['grids'] = aug_grids
            aug_example['augmentation'] = 'color_map'

            augmented.append(aug_example)

    # Rotation augmentation
    if CONFIG["augment_rotations"]:
        for rotation in [1]:  # Just a single 90 degree rotation to keep data size manageable
            def apply_rotation(grid):
                return np.rot90(np.array(grid), k=rotation).tolist()

            aug_grids = {
                'train_pairs': [(apply_rotation(inp), apply_rotation(out)) for inp, out in grids['train_pairs']],
                'test_input': apply_rotation(grids['test_input']),
                'test_output': apply_rotation(grids['test_output']),
                'initial_solution': apply_rotation(grids['initial_solution'])
            }

            fusion_feedback = {'consensus': example['fusion_consensus'], 'variance': 0.15}
            aug_prompt = create_refinement_prompt(
                aug_grids['train_pairs'],
                aug_grids['test_input'],
                aug_grids['initial_solution'],
                fusion_feedback
            )
            aug_completion = create_refinement_completion(aug_grids['test_output'], 0.85)

            aug_example = example.copy()
            aug_example['prompt'] = aug_prompt
            aug_example['completion'] = aug_completion
            aug_example['text'] = aug_prompt + aug_completion
            aug_example['grids'] = aug_grids
            aug_example['augmentation'] = f'rotation_{rotation*90}'

            augmented.append(aug_example)

    return augmented

def main():
    overall_start = time.time()

    print("="*80)
    print("REFINEMENT DATA GENERATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Tasks: {CONFIG['max_tasks']}")
    print(f"  Examples per task: {CONFIG['examples_per_task']}")
    print(f"  Models: 3 (gptoss, phi3, qwen15)")
    print(f"  Augmentation: {'Yes' if CONFIG['augment_color_mapping'] or CONFIG['augment_rotations'] else 'No'}")
    print(f"  Target examples: {CONFIG['target_examples']}")
    print(f"\nEstimated time: ~{CONFIG['max_tasks'] * 3 * 0.5:.0f} minutes (40 tasks x 3 models x ~30s each)")

    # Load tasks
    data_path = Path(CONFIG['arc_data_path']).expanduser().resolve()
    print(f"\nLoading ARC tasks from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"ARC data path not found: {data_path}")
    task_files = list(data_path.glob("*.json"))[:CONFIG['max_tasks']]

    tasks = []
    for json_file in task_files:
        with open(json_file, 'r') as f:
            task_data = json.load(f)
        tasks.append((json_file.stem, task_data))

    print(f"Loaded {len(tasks)} tasks")

    # Generate examples with all 3 models - streaming to disk
    temp_file = Path("refinement_raw_examples.pkl")
    total_examples = 0

    # Delete temp file if it exists from previous run
    if temp_file.exists():
        temp_file.unlink()

    print(f"\n{'='*80}")
    print("PHASE 1/3: GENERATION")
    print(f"{'='*80}")

    models = ["gptoss", "phi3", "qwen15"]
    for model_idx, model_key in enumerate(models, 1):
        print(f"\n[Model {model_idx}/3: {model_key.upper()}]")

        model, tokenizer = load_model(model_key)

        # Generate and save incrementally to temp file
        num_examples = generate_examples_for_model(model_key, model, tokenizer, tasks, temp_file)
        total_examples += num_examples

        # Aggressive memory cleanup between models
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Generated {num_examples} examples with {model_key}")
        print(f"  Total saved so far: {total_examples} examples")

        # Show memory after cleanup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  GPU Memory after cleanup: {allocated:.1f}GB")

        # Overall progress
        models_done = model_idx
        overall_progress = (models_done / len(models)) * 100
        print(f"\n  Overall Progress: {overall_progress:.1f}% ({models_done}/{len(models)} models)")

    print(f"\n{'='*80}")
    print(f"PHASE 1 COMPLETE")
    print(f"{'='*80}")
    print(f"Generated {total_examples} examples from 3 models")
    print(f"Saved to: {temp_file}")

    # Augment data - load from disk in streaming fashion
    print(f"\n{'='*80}")
    print("PHASE 2/3: AUGMENTATION")
    print(f"{'='*80}")
    print(f"Loading from: {temp_file}")
    augmented_examples = []

    # Load examples one at a time from the temp file
    aug_start = time.time()
    with open(temp_file, 'rb') as f:
        example_count = 0
        while True:
            try:
                example = pickle.load(f)
                augmented_examples.extend(augment_example(example))
                example_count += 1

                # Progress bar
                if example_count % 10 == 0 or example_count == 1:
                    elapsed = time.time() - aug_start
                    rate = example_count / elapsed if elapsed > 0 else 0
                    eta = (total_examples - example_count) / rate if rate > 0 else 0
                    pct = (example_count / total_examples) * 100 if total_examples > 0 else 0

                    bar_length = 40
                    filled = int(bar_length * example_count / total_examples) if total_examples > 0 else 0
                    bar = '=' * filled + '-' * (bar_length - filled)

                    print(f"\r  [{bar}] {pct:.1f}% | {example_count}/{total_examples} | "
                          f"Augmented: {len(augmented_examples)} | ETA: {eta:.0f}s", end='', flush=True)

            except EOFError:
                break

    print()  # New line after progress bar
    aug_time = time.time() - aug_start
    print(f"Loaded {example_count} raw examples")
    print(f"Augmented to {len(augmented_examples)} examples in {aug_time:.1f}s")

    # Cap at target
    if len(augmented_examples) > CONFIG['target_examples']:
        np.random.shuffle(augmented_examples)
        augmented_examples = augmented_examples[:CONFIG['target_examples']]
        print(f"Capped at target: {len(augmented_examples)} examples")

    # Save to disk
    print(f"\n{'='*80}")
    print("PHASE 3/3: SAVING")
    print(f"{'='*80}")

    output_path = Path(CONFIG['output_file'])
    print(f"Saving final dataset to: {output_path}")

    save_start = time.time()
    with open(output_path, 'wb') as f:
        pickle.dump(augmented_examples, f)
    save_time = time.time() - save_start

    # Also save as JSON for inspection
    json_path = output_path.with_suffix('.json')
    json_examples = []
    for ex in augmented_examples[:10]:  # Just save first 10 for inspection
        json_ex = {k: v for k, v in ex.items() if k != 'grids'}
        json_examples.append(json_ex)

    with open(json_path, 'w') as f:
        json.dump(json_examples, f, indent=2)

    print(f"Saved {len(augmented_examples)} examples to {output_path} ({save_time:.1f}s)")
    print(f"Saved sample to {json_path} (first 10 for inspection)")

    # Clean up temporary files
    print(f"\nCleaning up temporary files...")
    if temp_file.exists():
        temp_file.unlink()
        print(f"  Removed {temp_file}")

    # Statistics
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"  Total examples: {len(augmented_examples)}")
    print(f"  From gptoss: {sum(1 for ex in augmented_examples if ex.get('model_source') == 'gptoss')}")
    print(f"  From phi3: {sum(1 for ex in augmented_examples if ex.get('model_source') == 'phi3')}")
    print(f"  From qwen15: {sum(1 for ex in augmented_examples if ex.get('model_source') == 'qwen15')}")
    print(f"  Original: {sum(1 for ex in augmented_examples if ex.get('augmentation') is None)}")
    print(f"  Color mapped: {sum(1 for ex in augmented_examples if ex.get('augmentation') == 'color_map')}")
    print(f"  Rotated: {sum(1 for ex in augmented_examples if 'rotation' in str(ex.get('augmentation', '')))}")

    avg_similarity = np.mean([ex['initial_similarity'] for ex in augmented_examples])
    print(f"\n  Average initial similarity: {avg_similarity:.2%}")
    print(f"  This data will train models to improve from {avg_similarity:.0%} to higher accuracy")

    overall_time = time.time() - overall_start

    print(f"\n{'='*80}")
    print("ALL PHASES COMPLETE!")
    print(f"{'='*80}")
    print(f"\nTotal Time: {overall_time/60:.1f} minutes")
    print(f"  Phase 1 (Generation): ~{(overall_time - aug_time - save_time)/60:.1f}m")
    print(f"  Phase 2 (Augmentation): ~{aug_time/60:.1f}m")
    print(f"  Phase 3 (Saving): ~{save_time/60:.1f}m")
    print(f"\nOutput:")
    print(f"  Main dataset: {output_path}")
    print(f"  Sample (first 10): {json_path}")
    print(f"\nTo use this data in Stage 3 training:")
    print(f"  1. Load the data: examples = pickle.load(open('{output_path}', 'rb'))")
    print(f"  2. Split train/val as usual")
    print(f"  3. Train all 3 models on the same data")

if __name__ == "__main__":
    main()
