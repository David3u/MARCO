import numpy as np
import torch
import time
import uuid
import copy
import json
import re
from typing import Dict, List, Any, Optional

# Import our enhanced components
from marco_enhanced_core import (
    Hypothesis, CognitiveStateSpace, BaseExpertModule, ExpertResourceProfile
)

# ============================================================================
# Example LLM Expert Module Implementation
# ============================================================================

class LLMExpertModule(BaseExpertModule):
    """
    Example LLM-based expert module that generates and refines hypotheses
    """
    
    def __init__(self, expert_name: str, resource_profile: ExpertResourceProfile, 
                 model_config: Dict = None):
        super().__init__(expert_name, resource_profile)
        
        self.model_config = model_config or {
            "max_tokens": 1000,
            "temperature": 0.7,
            "specialization": ["pattern_recognition", "transformation_analysis"]
        }
        
        # Simulated LLM capabilities - replace with actual LLM calls
        self.specialization = self.model_config.get("specialization", [])
        
    def generate_hypotheses(self, task, css: CognitiveStateSpace) -> List[Hypothesis]:
        """
        Generate initial hypotheses using LLM reasoning
        """
        start_time = time.time()
        hypotheses = []
        
        try:
            # Analyze the task patterns
            task_analysis = self._analyze_task_patterns(task)
            
            # Generate multiple hypothesis candidates
            hypothesis_candidates = self._generate_hypothesis_candidates(task, task_analysis)
            
            for i, candidate in enumerate(hypothesis_candidates):
                hypothesis_id = f"{self.expert_name}_{uuid.uuid4().hex[:8]}"
                
                # Estimate confidence based on pattern matching and consistency
                confidence = self._estimate_hypothesis_confidence(candidate, task, task_analysis)
                
                hypothesis = Hypothesis(
                    hypothesis_id=hypothesis_id,
                    content=candidate,
                    source_expert=self.expert_name,
                    confidence=confidence
                )
                
                hypotheses.append(hypothesis)
                
                # Limit number of hypotheses to prevent explosion
                if len(hypotheses) >= 3:
                    break
            
            inference_time = time.time() - start_time
            avg_confidence = np.mean([h.confidence for h in hypotheses]) if hypotheses else 0.0
            self.log_performance(inference_time, 0.0, avg_confidence)  # Accuracy will be evaluated later
            
            return hypotheses
            
        except Exception as e:
            print(f"Error in {self.expert_name} hypothesis generation: {e}")
            return []
    
    def refine_hypotheses(self, task, css: CognitiveStateSpace,
                         existing_hypotheses: List[Hypothesis]) -> Dict[str, float]:
        """
        Refine own hypothesis using belief fusion result and provide confidence for all candidates.

        Args:
            task: The current task
            css: Cognitive state space containing belief fusion results
            existing_hypotheses: All current hypotheses from all experts

        Returns:
            Dict mapping hypothesis_id -> confidence score for ALL candidates
        """
        start_time = time.time()
        expert_beliefs = {}

        try:
            # Get the belief fusion result (consensus from MCU)
            fused_belief = None
            if hasattr(css, 'convergence_history') and css.convergence_history:
                # The most recent fused belief represents the collective wisdom
                last_state = css.convergence_history[-1]
                fused_belief = {
                    'max_confidence': last_state.get('max_confidence', 0.0),
                    'confidence_variance': last_state.get('confidence_variance', 1.0)
                }

            # Refine OWN hypothesis using belief fusion insights
            for hypothesis in existing_hypotheses:
                if hypothesis.source_expert == self.expert_name:
                    # This is our hypothesis - refine it using belief fusion
                    refined_content = self._refine_with_belief_fusion(
                        hypothesis, task, css, fused_belief
                    )
                    if refined_content != hypothesis.content:
                        # Update hypothesis with refined content
                        hypothesis.refine(refined_content, hypothesis.confidence, self.expert_name)

            # Provide confidence scores for ALL candidates (not just our own)
            for hypothesis in existing_hypotheses:
                confidence = self._score_hypothesis(hypothesis, task, css, fused_belief)
                expert_beliefs[hypothesis.hypothesis_id] = confidence

            inference_time = time.time() - start_time
            avg_confidence = np.mean(list(expert_beliefs.values())) if expert_beliefs else 0.0
            self.log_performance(inference_time, 0.0, avg_confidence)

            return expert_beliefs

        except Exception as e:
            print(f"Error in {self.expert_name} hypothesis refinement: {e}")
            return {}
    
    def _analyze_task_patterns(self, task) -> Dict[str, Any]:
        """Analyze patterns in the task - simulated LLM reasoning"""
        analysis = {
            "grid_sizes": [],
            "color_patterns": [],
            "transformation_types": [],
            "complexity_score": 0.0
        }
        
        if hasattr(task, 'train_pairs'):
            for input_grid, output_grid in task.train_pairs:
                input_array = np.array(input_grid)
                output_array = np.array(output_grid)
                
                # Analyze grid sizes
                analysis["grid_sizes"].append(input_array.shape)
                
                # Analyze color patterns
                unique_colors = np.unique(input_array)
                analysis["color_patterns"].append(len(unique_colors))
                
                # Analyze transformation type (simplified)
                if input_array.shape == output_array.shape:
                    if np.array_equal(input_array, output_array):
                        analysis["transformation_types"].append("identity")
                    else:
                        change_ratio = np.mean(input_array != output_array)
                        if change_ratio < 0.1:
                            analysis["transformation_types"].append("minor_modification")
                        elif change_ratio < 0.5:
                            analysis["transformation_types"].append("partial_transformation")
                        else:
                            analysis["transformation_types"].append("major_transformation")
                else:
                    analysis["transformation_types"].append("shape_change")
        
        # Calculate complexity
        if analysis["grid_sizes"]:
            avg_size = np.mean([g[0] * g[1] for g in analysis["grid_sizes"]])
            avg_colors = np.mean(analysis["color_patterns"])
            transform_diversity = len(set(analysis["transformation_types"]))
            
            analysis["complexity_score"] = (avg_size / 900.0 + 
                                          avg_colors / 10.0 + 
                                          transform_diversity / 4.0) / 3.0
        
        return analysis
    
    def _generate_hypothesis_candidates(self, task, task_analysis: Dict) -> List[Any]:
        """Generate hypothesis candidates using actual LLM inference"""
        candidates = []

        # Check if we have actual model and tokenizer
        if "model" in self.model_config and "tokenizer" in self.model_config:
            # Use actual LLM to generate solutions
            llm_solution = self._call_llm_for_solution(task, task_analysis)
            if llm_solution is not None and llm_solution:
                candidates.append(llm_solution)

        # If LLM failed or not available, use heuristic fallbacks
        if not candidates:
            complexity = task_analysis["complexity_score"]
            transform_types = set(task_analysis["transformation_types"])

            if "identity" in transform_types:
                fallback = self._generate_copy_solution(task)
                if fallback:
                    candidates.append(fallback)

            if "minor_modification" in transform_types:
                fallback = self._generate_pattern_modification_solution(task)
                if fallback:
                    candidates.append(fallback)

            if "partial_transformation" in transform_types or "major_transformation" in transform_types:
                fallback = self._generate_complex_transformation_solution(task)
                if fallback:
                    candidates.append(fallback)

            # Always include a fallback solution
            fallback = self._generate_best_guess_solution(task)
            if fallback:
                candidates.append(fallback)

        # CRITICAL: If still no candidates, create a minimal fallback
        if not candidates:
            print(f"  [{self.expert_name}] All generation methods failed, using minimal fallback")
            # Create a simple grid as last resort
            if hasattr(task, 'test_pairs') and task.test_pairs:
                minimal_solution = [np.array(task.test_pairs[0][0])]  # Just copy first test input
                candidates.append(minimal_solution)

        return candidates
    
    def _generate_copy_solution(self, task) -> List[np.ndarray]:
        """Generate solution that copies input to output"""
        solutions = []
        if hasattr(task, 'test_pairs'):
            for input_grid, _ in task.test_pairs:
                solutions.append(np.array(input_grid))
        return solutions
    
    def _generate_pattern_modification_solution(self, task) -> List[np.ndarray]:
        """Generate solution with minor pattern modifications"""
        solutions = []
        if hasattr(task, 'test_pairs'):
            for input_grid, _ in task.test_pairs:
                # Simulate pattern modification (e.g., color changes)
                modified = np.array(input_grid)
                if modified.size > 0:
                    # Simple modification: change all 0s to 1s (example)
                    modified[modified == 0] = 1
                solutions.append(modified)
        return solutions
    
    def _generate_complex_transformation_solution(self, task) -> List[np.ndarray]:
        """Generate solution with complex transformations"""
        solutions = []
        if hasattr(task, 'test_pairs'):
            for input_grid, _ in task.test_pairs:
                # Simulate complex transformation
                input_array = np.array(input_grid)
                transformed = self._apply_complex_transformation(input_array, task)
                solutions.append(transformed)
        return solutions
    
    def _generate_shape_change_solution(self, task) -> List[np.ndarray]:
        """Generate solution with shape changes"""
        solutions = []
        if hasattr(task, 'test_pairs'):
            for input_grid, _ in task.test_pairs:
                # Simulate shape change (e.g., crop, pad, resize)
                input_array = np.array(input_grid)
                reshaped = self._apply_shape_transformation(input_array, task)
                solutions.append(reshaped)
        return solutions
    
    def _generate_best_guess_solution(self, task) -> List[np.ndarray]:
        """Generate best guess solution based on training examples"""
        solutions = []
        if hasattr(task, 'test_pairs') and hasattr(task, 'train_pairs') and task.train_pairs:
            # Use most common output pattern from training
            common_output = task.train_pairs[0][1]  # Simple heuristic
            for input_grid, _ in task.test_pairs:
                solutions.append(np.array(common_output))
        return solutions
    
    def _apply_complex_transformation(self, input_array: np.ndarray, task) -> np.ndarray:
        """Apply complex transformation - placeholder for actual LLM reasoning"""
        # Placeholder implementation
        return np.rot90(input_array)  # Simple rotation as example
    
    def _apply_shape_transformation(self, input_array: np.ndarray, task) -> np.ndarray:
        """Apply shape transformation - placeholder for actual LLM reasoning"""
        # Placeholder implementation - resize to common output size from training
        if hasattr(task, 'train_pairs') and task.train_pairs:
            target_shape = np.array(task.train_pairs[0][1]).shape
            if input_array.shape != target_shape:
                # Simple resize by cropping or padding
                output = np.zeros(target_shape, dtype=input_array.dtype)
                rows = min(input_array.shape[0], target_shape[0])
                cols = min(input_array.shape[1], target_shape[1])
                output[:rows, :cols] = input_array[:rows, :cols]
                return output
        return input_array
    
    def _estimate_hypothesis_confidence(self, candidate: Any, task, task_analysis: Dict) -> float:
        """Estimate confidence in a hypothesis"""
        # Simplified confidence estimation
        base_confidence = 0.5
        
        # Boost confidence based on specialization match
        if "pattern_recognition" in self.specialization:
            base_confidence += 0.1
        
        # Adjust based on task complexity and compute cost
        complexity = task_analysis["complexity_score"]
        # Higher compute cost models get bonus for complex tasks
        if self.resource_profile.compute_cost > 30.0 and complexity > 0.7:
            base_confidence += 0.2
        # Lower compute cost models get bonus for simple tasks
        elif self.resource_profile.compute_cost < 20.0 and complexity < 0.3:
            base_confidence += 0.2
        
        # Add some noise for realism
        confidence = base_confidence + np.random.normal(0, 0.1)
        return max(0.0, min(1.0, confidence))
    
    def _refine_with_belief_fusion(self, hypothesis: Hypothesis, task,
                                   css: CognitiveStateSpace,
                                   fused_belief: Dict) -> Any:
        """
        Refine hypothesis using belief fusion result from MCU.

        The expert uses the collective consensus to improve its own solution.
        """
        # If no fusion result or no model, keep original
        if fused_belief is None:
            return hypothesis.content

        if "model" not in self.model_config or "tokenizer" not in self.model_config:
            return hypothesis.content

        # Only refine if there's high confidence in fusion (worth the compute)
        if fused_belief['max_confidence'] < 0.6:
            return hypothesis.content

        try:
            # Create refinement prompt using belief fusion
            refinement_prompt = self._create_refinement_prompt(hypothesis, task, fused_belief)

            # Call LLM to refine
            refined_solution = self._call_llm_for_refinement(refinement_prompt, task)

            if refined_solution is not None:
                return refined_solution
            else:
                return hypothesis.content

        except Exception as e:
            print(f"  Refinement error: {e}")
            return hypothesis.content

    def _score_hypothesis(self, hypothesis: Hypothesis, task,
                         css: CognitiveStateSpace,
                         fused_belief: Dict) -> float:
        """
        Provide confidence score for a hypothesis (can be from any expert).

        If it's our own hypothesis, use our confidence.
        If it's from another expert, adjust based on belief fusion alignment.
        """
        base_confidence = hypothesis.confidence

        # If this is our hypothesis, return our confidence
        if hypothesis.source_expert == self.expert_name:
            return max(0.0, min(1.0, base_confidence))

        # For other experts' hypotheses, consider belief fusion consensus
        if fused_belief:
            # If fusion confidence is high, trust hypotheses that align with it
            fusion_confidence = fused_belief['max_confidence']
            variance = fused_belief['confidence_variance']

            # Low variance means strong consensus - weight towards that
            if variance < 0.1:
                # Adjust confidence based on fusion strength
                adjusted = base_confidence * 0.7 + fusion_confidence * 0.3
                return max(0.0, min(1.0, adjusted))

        # Default: return hypothesis's own confidence
        return max(0.0, min(1.0, base_confidence))

    def _create_refinement_prompt(self, hypothesis: Hypothesis, task,
                                  fused_belief: Dict) -> str:
        """
        Create a refinement prompt using MCU's belief fusion result.

        The prompt shows:
        - Original task
        - Expert's current solution
        - Collective consensus strength from belief fusion
        """
        prompt_parts = []

        prompt_parts.append("You are refining your solution to an ARC abstract reasoning task.")
        prompt_parts.append("The system has analyzed your solution along with other experts.")
        prompt_parts.append("Use this feedback to improve your answer.\n")

        # Add original task examples
        if hasattr(task, 'train_pairs') and task.train_pairs:
            prompt_parts.append("Training Examples:")
            for i, (inp, out) in enumerate(task.train_pairs[:3], 1):
                inp_str = json.dumps(inp)
                out_str = json.dumps(out)
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Input: {inp_str}")
                prompt_parts.append(f"Output: {out_str}")

        # Show test input
        if hasattr(task, 'test_pairs') and task.test_pairs:
            test_input = task.test_pairs[0][0]
            test_str = json.dumps(test_input)
            prompt_parts.append(f"\nTest Input: {test_str}")

        # Show current solution
        prompt_parts.append(f"\nYour Current Solution:")
        if isinstance(hypothesis.content, list) and len(hypothesis.content) > 0:
            solution_str = json.dumps(hypothesis.content[0].tolist() if hasattr(hypothesis.content[0], 'tolist') else hypothesis.content[0])
            prompt_parts.append(f"{solution_str}")

        # Show belief fusion feedback
        if fused_belief:
            consensus = fused_belief['max_confidence']
            variance = fused_belief['confidence_variance']

            prompt_parts.append(f"\nCollective Analysis:")
            prompt_parts.append(f"- Consensus Strength: {consensus:.2f}")
            prompt_parts.append(f"- Agreement Level: {'High' if variance < 0.1 else 'Medium' if variance < 0.3 else 'Low'}")

            if consensus > 0.8:
                prompt_parts.append("\nThe experts are highly confident. Your solution likely aligns with the pattern.")
                prompt_parts.append("Refine details and ensure correctness.")
            elif consensus > 0.6:
                prompt_parts.append("\nThere's moderate consensus. Consider if your solution captures the core pattern.")
                prompt_parts.append("Look for areas to improve alignment with the expected transformation.")
            else:
                prompt_parts.append("\nConsensus is low. Multiple interpretations exist.")
                prompt_parts.append("Reconsider the pattern - you may have identified a different valid approach.")

        prompt_parts.append("\nProvide your REFINED solution as a JSON array (grid).")
        prompt_parts.append("Only output the JSON array, nothing else.")

        return "\n".join(prompt_parts)

    def _call_llm_for_refinement(self, refinement_prompt: str, task) -> Optional[List[np.ndarray]]:
        """Call LLM to refine hypothesis based on belief fusion insights"""
        model = self.model_config.get("model")
        tokenizer = self.model_config.get("tokenizer")

        if model is None or tokenizer is None:
            return None

        try:
            print(f"refining...", end='', flush=True)

            # Tokenize
            inputs = tokenizer(refinement_prompt, return_tensors="pt").to(model.device)

            # Check prompt length
            prompt_length = inputs['input_ids'].shape[1]
            if prompt_length > 8000:
                return None

            # Use smaller max_tokens for refinement (just need the answer)
            max_tokens = min(self.model_config.get("max_tokens", 500), 200)
            temperature = 0.3  # Lower temperature for refinement (more focused)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )

            # Decode
            outputs_cpu = outputs.cpu()
            generated_text = tokenizer.decode(outputs_cpu[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Parse the refined output
            refined_solution = self._parse_llm_output(generated_text, task)

            print(f" refined", end='', flush=True)

            # Clean up
            del inputs, outputs, outputs_cpu
            torch.cuda.empty_cache()

            return refined_solution

        except Exception as e:
            print(f" refinement failed: {e}", end='', flush=True)
            return None

    def _create_arc_prompt(self, task, task_analysis: Dict) -> str:
        """Create a prompt for the LLM to solve an ARC task - MATCHES FINE-TUNING FORMAT"""
        prompt = """You are an expert at solving abstract reasoning tasks from the ARC (Abstraction and Reasoning Corpus) challenge.

Given input-output example pairs, identify the transformation pattern and apply it to the test input.

Format: Provide the output grid as a JSON array.

"""

        # Add training examples
        if hasattr(task, 'train_pairs') and task.train_pairs:
            for i, (inp, out) in enumerate(task.train_pairs[:3], 1):  # Limit to 3 examples
                # Use json.dumps() for consistent formatting
                prompt += f"Example {i}:\n"
                prompt += f"Input: {json.dumps(inp)}\n"
                prompt += f"Output: {json.dumps(out)}\n\n"

        # Add test input
        if hasattr(task, 'test_pairs') and task.test_pairs:
            test_input = task.test_pairs[0][0]
            prompt += f"Test Input: {json.dumps(test_input)}\n"
            prompt += f"Test Output:"

        return prompt

    def _call_llm_for_solution(self, task, task_analysis: Dict) -> Optional[List[np.ndarray]]:
        """Call the actual LLM to generate a solution with dynamic model loading"""
        import gc
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = None
        try:
            # Check if we should use dynamic loading
            model_path = self.model_config.get("model_path")
            use_dynamic_loading = self.model_config.get("dynamic_loading", False)

            if use_dynamic_loading and model_path:
                # DYNAMIC LOADING: Load model on-demand
                print(f"  [{self.expert_name}] Loading model...", end='', flush=True)
                load_start = time.time()

                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                model.eval()

                print(f" loaded ({time.time() - load_start:.1f}s), ", end='', flush=True)
            else:
                # STATIC LOADING: Use pre-loaded model
                model = self.model_config.get("model")
                tokenizer = self.model_config.get("tokenizer")

            if model is None or tokenizer is None:
                return None

            print(f"generating...", end='', flush=True)
            start_time = time.time()

            # Create prompt
            prompt = self._create_arc_prompt(task, task_analysis)

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Check prompt length
            prompt_length = inputs['input_ids'].shape[1]
            if prompt_length > 8000:
                print(f"\n  Prompt too long ({prompt_length} tokens), skipping task")
                return None

            # Generate with reduced max_tokens if prompt is large
            max_tokens = self.model_config.get("max_tokens", 500)
            if prompt_length > 4000:
                max_tokens = min(max_tokens, 300)  # Reduce for very long prompts
            elif prompt_length > 2000:
                max_tokens = min(max_tokens, 400)  # Slightly reduce for long prompts
            temperature = self.model_config.get("temperature", 0.7)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False
                    )
            except RuntimeError as e:
                if "CUDA" in str(e) or "assert" in str(e).lower():
                    print(f"\n  CUDA error during generation (prompt_len={prompt_length}): {e}")
                    print(f"  Skipping this task...")
                    return None
                raise

            # Decode (move to CPU first to free GPU memory)
            outputs_cpu = outputs.cpu()
            generated_text = tokenizer.decode(outputs_cpu[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Parse the output
            solution = self._parse_llm_output(generated_text, task)

            elapsed = time.time() - start_time
            print(f" done ({elapsed:.1f}s)")

            # CRITICAL: Clean up GPU memory immediately
            del inputs, outputs, outputs_cpu

            # If using dynamic loading, UNLOAD the model completely
            if use_dynamic_loading and model is not None:
                print(f"  [{self.expert_name}] Unloading model...", end='', flush=True)
                del model, tokenizer
                model = None
                print(f" freed GPU memory")

            torch.cuda.empty_cache()
            gc.collect()

            return solution

        except Exception as e:
            print(f"\n  LLM inference error in {self.expert_name}: {e}")
            # Clean up on error too
            if model is not None and self.model_config.get("dynamic_loading", False):
                del model
            torch.cuda.empty_cache()
            gc.collect()
            return None

    def _parse_llm_output(self, text: str, task) -> Optional[List[np.ndarray]]:
        """Parse LLM output text into grid format"""
        try:
            # Try to find JSON array in the output
            # Look for patterns like [[...], [...], ...]
            json_match = re.search(r'\[\s*\[.*?\]\s*\]', text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                grid = json.loads(json_str)

                # Validate it's a proper grid
                if isinstance(grid, list) and len(grid) > 0:
                    if all(isinstance(row, list) for row in grid):
                        # Convert to numpy array
                        solution = np.array(grid)

                        # Return as list of solutions (one per test input)
                        return [solution]

            # If parsing failed, return None
            return None

        except Exception as e:
            print(f"Output parsing error in {self.expert_name}: {e}")
            return None

# ============================================================================
# Enhanced Task Class for the New System
# ============================================================================

class EnhancedTask:
    """
    Enhanced task representation for the new MARCO system
    Simplified version focused on grid-based hypotheses
    """

    def __init__(self, task_id, train_pairs, test_pairs):
        self.task_id = task_id
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs

        # Initialize cognitive state space (enhanced blackboard)
        self.css = CognitiveStateSpace()

        # Store grid shapes for analysis
        self.train_input_shapes = [np.array(pair[0]).shape for pair in train_pairs]
        self.train_output_shapes = [np.array(pair[1]).shape for pair in train_pairs]
        self.test_input_shapes = [np.array(pair[0]).shape for pair in test_pairs]
        self.test_output_shapes = [np.array(pair[1]).shape for pair in test_pairs]


        # Track feature extraction methods used
        self.features = {}
    
    # ========================================================================
    # Grid Analysis Methods
    # ========================================================================

    def get_grid_statistics(self) -> Dict[str, Any]:
        """Get statistics about the grids in this task"""
        stats = {
            "num_train_pairs": len(self.train_pairs),
            "num_test_pairs": len(self.test_pairs),
            "train_input_shapes": self.train_input_shapes,
            "train_output_shapes": self.train_output_shapes,
            "test_input_shapes": self.test_input_shapes,
            "test_output_shapes": self.test_output_shapes
        }

        # Calculate complexity metrics
        all_grids = []
        for input_grid, output_grid in self.train_pairs + self.test_pairs:
            all_grids.extend([input_grid, output_grid])

        if all_grids:
            all_arrays = [np.array(grid) for grid in all_grids]
            stats["avg_grid_size"] = np.mean([arr.size for arr in all_arrays])
            stats["max_grid_size"] = max([arr.size for arr in all_arrays])
            stats["unique_colors"] = len(set().union(*[set(arr.flatten()) for arr in all_arrays]))

        return stats

    def analyze_transformations(self) -> List[Dict[str, Any]]:
        """Analyze the transformations in training pairs"""
        transformations = []

        for i, (input_grid, output_grid) in enumerate(self.train_pairs):
            input_array = np.array(input_grid)
            output_array = np.array(output_grid)

            transform_info = {
                "pair_index": i,
                "input_shape": input_array.shape,
                "output_shape": output_array.shape,
                "shape_changed": input_array.shape != output_array.shape,
                "identical": np.array_equal(input_array, output_array)
            }

            # Calculate change ratio if same shape
            if input_array.shape == output_array.shape:
                changes = np.sum(input_array != output_array)
                transform_info["change_ratio"] = changes / input_array.size
                transform_info["num_changes"] = changes
            else:
                transform_info["change_ratio"] = 1.0  # Complete transformation
                transform_info["num_changes"] = input_array.size + output_array.size

            # Analyze color changes
            input_colors = set(input_array.flatten())
            output_colors = set(output_array.flatten())
            transform_info["colors_added"] = list(output_colors - input_colors)
            transform_info["colors_removed"] = list(input_colors - output_colors)
            transform_info["colors_preserved"] = list(input_colors & output_colors)

            transformations.append(transform_info)

        return transformations
    
    # ========================================================================
    # CSS Integration Methods
    # ========================================================================
    
    def get_css(self) -> CognitiveStateSpace:
        """Get the cognitive state space"""
        return self.css
    
    def add_expert_insight(self, expert_name: str, insight_type: str, content: Any):
        """Add expert insight to CSS"""
        key = f"{expert_name}_{insight_type}_{uuid.uuid4().hex[:8]}"
        self.css.knowledge_base[key] = content
        self.css.knowledge_sources[key] = expert_name

    def get_expert_insights(self, requesting_expert: str) -> Dict[str, Any]:
        """Get insights from other experts"""
        insights = {}
        for key, value in self.css.knowledge_base.items():
            source = self.css.knowledge_sources.get(key, "unknown")
            if source != requesting_expert:  # Don't return own insights
                insights[key] = value
        return insights
    
    def log_expert_reasoning(self, expert_name: str, hypothesis: Hypothesis, 
                           confidence: float, details: Dict = None):
        """Log expert reasoning step"""
        self.css.reasoning_history.append({
            "expert_name": expert_name,
            "hypothesis_id": hypothesis.hypothesis_id,
            "confidence": confidence,
            "timestamp": time.time(),
            "details": details or {}
        })
    
    # ========================================================================
    # Solution Validation Methods
    # ========================================================================

    def validate_solution(self, solution_grids: List[np.ndarray]) -> Dict[str, Any]:
        """Validate that solution grids are properly formatted"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        if len(solution_grids) != len(self.test_pairs):
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Number of solutions ({len(solution_grids)}) doesn't match test pairs ({len(self.test_pairs)})"
            )

        for i, grid in enumerate(solution_grids):
            try:
                grid_array = np.array(grid)
                if grid_array.ndim != 2:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Solution {i} is not a 2D array")

                # Check for reasonable size
                if grid_array.size > 30 * 30:
                    validation_results["warnings"].append(f"Solution {i} is very large ({grid_array.shape})")

                # Check for valid color values (typically 0-9 in ARC)
                unique_values = np.unique(grid_array)
                if np.any(unique_values < 0) or np.any(unique_values > 9):
                    validation_results["warnings"].append(f"Solution {i} contains unusual color values: {unique_values}")

            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Solution {i} cannot be converted to array: {e}")

        return validation_results

# ============================================================================
# Factory Functions for Creating Expert Modules
# ============================================================================

def create_lightweight_llm_expert(name: str) -> LLMExpertModule:
    """Create a lightweight LLM expert (e.g., small local model like Phi-3, Gemma-7B)"""
    profile = ExpertResourceProfile(
        expert_name=name,
        compute_cost=10,
        memory_requirement=1,
        inference_time=2
    )

    config = {
        "max_tokens": 500,
        "temperature": 0.3
    }

    return LLMExpertModule(name, profile, config)

def create_medium_llm_expert(name: str) -> LLMExpertModule:
    """Create a medium-capability LLM expert (e.g., Llama-13B, Mistral-7B)"""
    profile = ExpertResourceProfile(
        expert_name=name,
        compute_cost=30,
        memory_requirement=3,
        inference_time=8
    )

    config = {
        "max_tokens": 1000,
        "temperature": 0.5
    }

    return LLMExpertModule(name, profile, config)

def create_heavy_llm_expert(name: str) -> LLMExpertModule:
    """Create a heavy-duty LLM expert (e.g., Llama-70B, Mixtral-8x7B)"""
    profile = ExpertResourceProfile(
        expert_name=name,
        compute_cost=60,
        memory_requirement=6,
        inference_time=20
    )

    config = {
        "max_tokens": 2000,
        "temperature": 0.7
    }

    return LLMExpertModule(name, profile, config)
