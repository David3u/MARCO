"""
Integration module for loading local LLMs using HuggingFace Transformers.

This module provides utilities to:
1. Load GPT-OSS, Phi-3, and Llama-3.2 models
2. Profile their resource usage
3. Generate hypotheses for ARC tasks
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
import psutil
import gc


class LocalLLMManager:
    """
    Manages local LLM models loaded via transformers.
    Handles model loading, inference, and resource profiling.
    """

    def __init__(self, device: str = "auto", use_4bit: bool = True, use_8bit: bool = False):
        """
        Initialize LLM manager.

        Args:
            device: "cuda", "cpu", or "auto"
            use_4bit: Use 4-bit quantization (saves memory)
            use_8bit: Use 8-bit quantization (fallback if 4-bit unavailable)
        """
        self.device = device
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"LocalLLMManager initialized with device: {self.device}")
        if self.device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def load_gptoss(self, model_path: str = "your-gptoss-path") -> None:
        """
        Load GPT-OSS model.

        Args:
            model_path: Path or HuggingFace model ID
                       Examples:
                       - "gpt2" (small GPT-2 for testing)
                       - "EleutherAI/gpt-neo-1.3B"
                       - "EleutherAI/gpt-j-6B"
                       - Local path: "/path/to/gptoss/weights"
        """
        print(f"\nLoading GPT-OSS from: {model_path}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Configure quantization for memory efficiency
            quantization_config = None
            if self.use_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("  Using 4-bit quantization")
            elif self.use_8bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                print("  Using 8-bit quantization")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )

            if self.device == "cpu":
                model = model.to(self.device)

            model.eval()  # Set to evaluation mode

            # Store
            self.models["gptoss"] = model
            self.tokenizers["gptoss"] = tokenizer

            print(f"  GPT-OSS loaded successfully")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        except Exception as e:
            print(f"  Error loading GPT-OSS: {e}")
            raise

    def load_phi3(self, model_path: str = "microsoft/Phi-3-mini-4k-instruct") -> None:
        """
        Load Phi-3 model.

        Args:
            model_path: Path or HuggingFace model ID
                       Options:
                       - "microsoft/Phi-3-mini-4k-instruct" (3.8B params)
                       - "microsoft/Phi-3-mini-128k-instruct" (longer context)
                       - "microsoft/Phi-3-small-8k-instruct" (7B params)
        """
        print(f"\nLoading Phi-3 from: {model_path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            quantization_config = None
            if self.use_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("  Using 4-bit quantization")

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            if self.device == "cpu":
                model = model.to(self.device)

            model.eval()

            self.models["phi3"] = model
            self.tokenizers["phi3"] = tokenizer

            print(f"  Phi-3 loaded successfully")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        except Exception as e:
            print(f"  Error loading Phi-3: {e}")
            raise

    def load_llama32(self, model_path: str = "meta-llama/Llama-3.2-3B-Instruct") -> None:
        """
        Load Llama-3.2 model.

        Args:
            model_path: Path or HuggingFace model ID
                       Options:
                       - "meta-llama/Llama-3.2-1B" (1B params, fast)
                       - "meta-llama/Llama-3.2-3B" (3B params, balanced)
                       - "meta-llama/Llama-3.2-3B-Instruct" (instruction-tuned)

                       Note: May require HF authentication token for official models
        """
        print(f"\nLoading Llama-3.2 from: {model_path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantization_config = None
            if self.use_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("  Using 4-bit quantization")

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )

            if self.device == "cpu":
                model = model.to(self.device)

            model.eval()

            self.models["llama32"] = model
            self.tokenizers["llama32"] = tokenizer

            print(f"  Llama-3.2 loaded successfully")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        except Exception as e:
            print(f"  Error loading Llama-3.2: {e}")
            print("  Note: Official Llama models may require HuggingFace authentication")
            print("  Run: huggingface-cli login")
            raise

    def generate_text(self, model_name: str, prompt: str,
                     max_new_tokens: int = 512,
                     temperature: float = 0.7,
                     top_p: float = 0.9) -> str:
        """
        Generate text using specified model.

        Args:
            model_name: "gptoss", "phi3", or "llama32"
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode (skip input prompt)
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                         skip_special_tokens=True)

        return generated_text

    def profile_model(self, model_name: str, test_prompts: List[str],
                     num_runs: int = 5) -> Dict:
        """
        Profile model inference time and memory usage.

        Args:
            model_name: Model to profile
            test_prompts: Sample prompts for profiling
            num_runs: Number of runs to average

        Returns:
            Dict with compute_cost, inference_time, memory_requirement
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        print(f"\nProfiling {model_name}...")

        model = self.models[model_name]
        inference_times = []
        memory_samples = []

        # Warm-up run
        _ = self.generate_text(model_name, test_prompts[0], max_new_tokens=100)

        # Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        for i, prompt in enumerate(test_prompts[:num_runs]):
            # Measure memory before
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / 1024**3
            else:
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024**3

            # Time inference
            start_time = time.time()
            _ = self.generate_text(model_name, prompt, max_new_tokens=256)
            elapsed = time.time() - start_time

            # Measure memory after
            if self.device == "cuda":
                mem_peak = torch.cuda.max_memory_allocated() / 1024**3
                memory_used = mem_peak - mem_before
            else:
                mem_after = process.memory_info().rss / 1024**3
                memory_used = mem_after - mem_before

            inference_times.append(elapsed)
            memory_samples.append(memory_used)

            print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s, {memory_used:.2f}GB")

        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        avg_memory = np.mean(memory_samples)
        peak_memory = np.max(memory_samples)

        # Compute cost: scale inference time to units (10 units per second)
        compute_cost = avg_time * 10

        profile = {
            "compute_cost": compute_cost,
            "inference_time": avg_time,
            "inference_time_std": std_time,
            "memory_requirement": peak_memory,
            "avg_memory": avg_memory
        }

        print(f"\n  Profile Results:")
        print(f"    Inference time: {avg_time:.3f}s (+/-{std_time:.3f}s)")
        print(f"    Compute cost: {compute_cost:.2f} units")
        print(f"    Peak memory: {peak_memory:.2f} GB")
        print(f"    Avg memory: {avg_memory:.2f} GB")

        return profile

    def unload_model(self, model_name: str) -> None:
        """Unload a model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            del self.tokenizers[model_name]

            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            print(f"Unloaded {model_name}")

    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a loaded model."""
        if model_name not in self.models:
            return None

        model = self.models[model_name]

        return {
            "name": model_name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(model.device),
            "dtype": str(model.dtype)
        }


def create_arc_prompt(task_data: Dict, for_hypothesis_generation: bool = True) -> str:
    """
    Create a prompt for ARC task solving.

    Args:
        task_data: ARC task with train_pairs and test_pairs
        for_hypothesis_generation: If True, ask for hypotheses. If False, ask for solution.

    Returns:
        Formatted prompt string
    """
    prompt = "You are an expert at solving abstract reasoning puzzles.\n\n"

    # Add training examples
    prompt += "Training Examples:\n"
    for i, (input_grid, output_grid) in enumerate(task_data['train_pairs'], 1):
        prompt += f"\nExample {i}:\n"
        prompt += f"Input:\n{np.array(input_grid).tolist()}\n"
        prompt += f"Output:\n{np.array(output_grid).tolist()}\n"

    # Add test input
    if task_data['test_pairs']:
        test_input = task_data['test_pairs'][0][0]
        prompt += f"\nTest Input:\n{np.array(test_input).tolist()}\n"

    # Add instructions
    if for_hypothesis_generation:
        prompt += "\nYour task:\n"
        prompt += "1. Analyze the pattern in the training examples\n"
        prompt += "2. Generate 2-3 different hypotheses about the transformation rule\n"
        prompt += "3. For each hypothesis, provide:\n"
        prompt += "   - transformation_rule: Brief description\n"
        prompt += "   - reasoning: Why this rule makes sense\n"
        prompt += "   - confidence: 0.0 to 1.0\n"
        prompt += "\nProvide your response as a list of hypotheses.\n"
    else:
        prompt += "\nYour task:\n"
        prompt += "1. Identify the transformation rule from the examples\n"
        prompt += "2. Apply it to the test input\n"
        prompt += "3. Output the transformed grid\n"

    return prompt


# Example usage
if __name__ == "__main__":
    # Initialize manager
    llm_manager = LocalLLMManager(device="auto", use_4bit=True)

    # Load models (choose which ones you have access to)
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)

    # Example 1: Load GPT-2 for testing (publicly available)
    # llm_manager.load_gptoss("gpt2")

    # Example 2: Load Phi-3 (publicly available)
    try:
        llm_manager.load_phi3("microsoft/Phi-3-mini-4k-instruct")
    except Exception as e:
        print(f"Could not load Phi-3: {e}")

    # Example 3: Load Llama-3.2 (may need authentication)
    # try:
    #     llm_manager.load_llama32("meta-llama/Llama-3.2-3B-Instruct")
    # except Exception as e:
    #     print(f"Could not load Llama-3.2: {e}")

    # Test inference
    if "phi3" in llm_manager.models:
        print("\n" + "="*60)
        print("Testing Inference")
        print("="*60)

        test_prompt = "What is the pattern in this sequence: 2, 4, 6, 8, ?"
        response = llm_manager.generate_text("phi3", test_prompt, max_new_tokens=100)
        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {response}")

        # Profile the model
        test_prompts = [
            "Solve this puzzle: 1, 2, 3, ?",
            "What comes next: A, B, C, ?",
            "Complete the pattern: 5, 10, 15, ?"
        ]
        profile = llm_manager.profile_model("phi3", test_prompts, num_runs=3)

        print(f"\nProfile complete: {profile}")
