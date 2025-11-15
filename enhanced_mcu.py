import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import time
import traceback
from typing import Dict, List, Tuple, Optional, Any

# Import the core components we just created
from marco_enhanced_core import (
    Hypothesis, BeliefDistribution, ExpertResourceProfile, ResourceBudget,
    CognitiveStateSpace, BaseExpertModule, ExpertWeightNetwork, 
    ConvergencePredictor, TaskComplexityPredictor
)

# ============================================================================
# Enhanced Meta Control Unit (MCU)
# ============================================================================

class EnhancedMetaControlUnit:
    """
    Resource-bounded meta-reasoning system using Product of Experts
    """
    
    def __init__(self, expert_modules: Dict[str, BaseExpertModule] = None, config: Dict = None):
        # Expert modules and their profiles
        self.expert_modules = expert_modules or {}
        self.expert_profiles = {}  # expert_name -> ExpertResourceProfile
        self.expert_names = list(self.expert_modules.keys())
        
        # Default configuration
        self.config = {
            "max_iterations": 5,
            "convergence_threshold": 0.95,
            "confidence_variance_threshold": 0.01,
            "learning_rate": 0.001,
            "task_embedding_dim": 6,  # 6 task features (no padding needed for LLMs)
            "default_compute_budget": 100.0,
            "default_time_budget": 30.0,
            "default_memory_budget": 8.0,
            "expert_selection_temperature": 0.5,
            "early_stopping_patience": 2,
            "resource_efficiency_weight": 0.3,
            "accuracy_weight": 0.7,
            # Dynamic expert selection parameters
            "min_experts": 1,
            "max_experts": None,  # None = unlimited (budget-constrained only)
            "min_expert_efficiency": 0.05,  # Stop adding experts if efficiency < threshold
            "complexity_expert_scaling": True,  # Scale max experts by task complexity
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
            
        # Initialize expert profiles if not provided
        self._initialize_expert_profiles()
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks for meta-reasoning
        self.task_complexity_predictor = TaskComplexityPredictor(
            task_feature_dim=self.config["task_embedding_dim"]
        ).to(self.device)
        
        self.expert_weight_network = ExpertWeightNetwork(
            task_embedding_dim=self.config["task_embedding_dim"],
            num_experts=len(self.expert_modules)
        ).to(self.device)
        
        self.convergence_predictor = ConvergencePredictor(
            state_dim=10  # Features about convergence state
        ).to(self.device)
        
        # Optimizers
        self.complexity_optimizer = optim.Adam(
            self.task_complexity_predictor.parameters(),
            lr=self.config["learning_rate"]
        )
        
        self.weight_optimizer = optim.Adam(
            self.expert_weight_network.parameters(),
            lr=self.config["learning_rate"]
        )
        
        self.convergence_optimizer = optim.Adam(
            self.convergence_predictor.parameters(),
            lr=self.config["learning_rate"]
        )
        
        # Training history
        self.episode_history = []
        self.resource_efficiency_history = []

        # Expert performance tracking (learned, not hardcoded)
        self.expert_performance_matrix = {}  # expert_name -> {feature_hash: [performance_records]}
        
    def _initialize_expert_profiles(self):
        """Initialize default resource profiles for experts"""
        # Default profiles with medium resource costs
        # These should be updated with actual measurements from your local LLMs
        for expert_name in self.expert_names:
            self.expert_profiles[expert_name] = ExpertResourceProfile(
                expert_name=expert_name,
                compute_cost=30.0,  # Default - measure your actual LLM
                memory_requirement=3.0,  # Default - measure your actual LLM
                inference_time=8.0  # Default - measure your actual LLM
            )

    def _hash_task_features(self, task_features: torch.Tensor) -> str:
        """Create a hash key from task features for performance tracking"""
        # Round features to 1 decimal to group similar tasks
        rounded = (task_features * 10).round() / 10
        feature_str = ",".join([f"{x:.1f}" for x in rounded.tolist()])
        return feature_str

    def _record_expert_performance(self, expert_name: str, task_features: torch.Tensor,
                                   accuracy: float, efficiency: float):
        """Record expert performance on specific task features"""
        feature_key = self._hash_task_features(task_features)

        if expert_name not in self.expert_performance_matrix:
            self.expert_performance_matrix[expert_name] = {}

        if feature_key not in self.expert_performance_matrix[expert_name]:
            self.expert_performance_matrix[expert_name][feature_key] = []

        self.expert_performance_matrix[expert_name][feature_key].append({
            'accuracy': accuracy,
            'efficiency': efficiency,
            'timestamp': time.time()
        })

        # Keep only recent history (last 20 records per feature type)
        if len(self.expert_performance_matrix[expert_name][feature_key]) > 20:
            self.expert_performance_matrix[expert_name][feature_key] = \
                self.expert_performance_matrix[expert_name][feature_key][-20:]

    def _get_task_complexity_estimate(self, task_features: torch.Tensor) -> float:
        """
        Estimate task complexity from features for expert selection.
        Returns value between 0 (simple) and 1 (complex).
        """
        # Use task complexity predictor if available
        with torch.no_grad():
            predictions = self.task_complexity_predictor(task_features)
            if predictions.dim() > 1:
                predictions = predictions.squeeze(0)
            complexity = predictions[0].item()  # First output is complexity
            return max(0.0, min(1.0, complexity))

    def extract_task_features(self, task) -> torch.Tensor:
        """
        Extract features from ARC task for complexity prediction and expert selection

        Returns 6-dimensional feature vector:
        1. Average grid size (normalized)
        2. Mean color diversity (normalized)
        3. Std color diversity
        4. Mean change ratio
        5. Std change ratio
        6. Training set size (normalized)
        """
        features = []

        if hasattr(task, 'train_pairs') and task.train_pairs:
            # 1. Average grid size
            avg_size = np.mean([np.array(pair[0]).size for pair in task.train_pairs])
            features.append(avg_size / 900.0)  # Normalize by max expected grid size

            # 2-5. Color complexity and transformation magnitude
            color_complexities = []
            change_ratios = []

            for input_grid, output_grid in task.train_pairs:
                input_array = np.array(input_grid)
                output_array = np.array(output_grid)

                # Color diversity (unique colors per grid)
                unique_colors = len(np.unique(input_array))
                color_complexities.append(unique_colors / 10.0)  # Normalize by typical max colors

                # Change ratio (proportion of pixels that change)
                if input_array.shape == output_array.shape:
                    changes = np.sum(input_array != output_array)
                    change_ratios.append(changes / input_array.size)
                else:
                    change_ratios.append(1.0)  # Complete transformation if shape changes

            features.extend([
                np.mean(color_complexities),      # Mean color diversity
                np.std(color_complexities),       # Std color diversity
                np.mean(change_ratios),           # Mean change ratio
                np.std(change_ratios)             # Std change ratio
            ])

            # 6. Training set size
            num_examples = len(task.train_pairs)
            features.append(num_examples / 10.0)  # Normalize by typical max examples
        else:
            # Default features if no training pairs
            features = [0.5, 0.5, 0.0, 0.5, 0.0, 0.0]

        # Ensure all features are valid (no NaN or inf)
        features = [float(f) if not np.isnan(f) and not np.isinf(f) else 0.0 for f in features]

        try:
            return torch.tensor(features, dtype=torch.float32, device=self.device)
        except RuntimeError as e:
            # If CUDA error, try to recover by resetting device and using CPU temporarily
            if "CUDA" in str(e):
                print(f"  WARNING: CUDA error in extract_task_features, recovering...")
                torch.cuda.synchronize()
                return torch.tensor(features, dtype=torch.float32, device='cpu').to(self.device)
            raise
    
    def predict_resource_requirements(self, task) -> Tuple[float, float, float, float]:
        """
        Predict resource requirements for a task
        
        Returns:
            Tuple of (complexity, compute_requirement, time_requirement, accuracy_requirement)
        """
        task_features = self.extract_task_features(task)
        
        with torch.no_grad():
            predictions = self.task_complexity_predictor(task_features)

        if predictions.dim() > 1:
            predictions = predictions.squeeze(0)

        # Clamp predictions to [0, 1] to avoid invalid values
        predictions = torch.clamp(predictions, 0.0, 1.0)

        complexity = predictions[0].item()
        compute_req = predictions[1].item() * self.config["default_compute_budget"]
        time_req = predictions[2].item() * self.config["default_time_budget"]
        accuracy_req = predictions[3].item()

        return complexity, compute_req, time_req, accuracy_req
    
    def select_initial_experts(self, task, resource_budget: ResourceBudget) -> List[str]:
        """
        Dynamically select experts based on learned performance, task complexity, and budget.
        Number of experts selected is determined by:
        1. Resource budget constraints (stops when budget exhausted)
        2. Task complexity (complex tasks benefit from more experts)
        3. Diminishing returns (stops when marginal efficiency too low)
        """
        task_features = self.extract_task_features(task)

        selected_experts = []
        remaining_budget = ResourceBudget(
            max_compute=resource_budget.max_compute,
            max_time=resource_budget.max_time,
            max_memory=resource_budget.max_memory,
            target_confidence=resource_budget.target_confidence
        )

        # Score experts based on learned efficiency for this task type
        expert_scores = []
        for expert_name, profile in self.expert_profiles.items():
            # Use learned efficiency on similar tasks
            efficiency = self._estimate_expert_efficiency(expert_name, task_features)
            affordability = 1.0 if remaining_budget.can_afford(profile) else 0.0

            score = efficiency * affordability
            expert_scores.append((score, expert_name, profile))

        # Sort by score and select top experts within budget
        expert_scores.sort(reverse=True, key=lambda x: x[0])

        # Determine maximum useful experts based on task complexity
        max_experts = self.config["max_experts"]
        if max_experts is None or self.config["complexity_expert_scaling"]:
            complexity = self._get_task_complexity_estimate(task_features)
            # Scale from min_experts to 5 based on complexity
            # Simple tasks (0.0): 1 expert, Complex tasks (1.0): 5 experts
            complexity_based_max = int(self.config["min_experts"] + complexity * 4)
            if max_experts is not None:
                max_experts = min(max_experts, complexity_based_max)
            else:
                max_experts = complexity_based_max

        # Select experts until stopping criteria met
        min_efficiency = self.config["min_expert_efficiency"]

        for score, expert_name, profile in expert_scores:
            # Stop if budget exhausted
            if not remaining_budget.can_afford(profile):
                break

            # Stop if efficiency too low (diminishing returns)
            if len(selected_experts) >= self.config["min_experts"] and score < min_efficiency:
                break

            # Stop if reached complexity-based maximum
            if len(selected_experts) >= max_experts:
                break

            selected_experts.append(expert_name)
            remaining_budget.consume(profile)

        # Ensure at least one expert is selected (fallback)
        if not selected_experts and expert_scores:
            # Force select best expert even if over budget
            selected_experts.append(expert_scores[0][1])

        return selected_experts
    
    def _estimate_expert_efficiency(self, expert_name: str, task_features: torch.Tensor) -> float:
        """Estimate expert efficiency based on learned performance on similar tasks"""
        if expert_name not in self.expert_modules:
            return 0.5

        feature_key = self._hash_task_features(task_features)

        # Check if we have performance data for similar tasks
        if (expert_name in self.expert_performance_matrix and
            feature_key in self.expert_performance_matrix[expert_name]):

            history = self.expert_performance_matrix[expert_name][feature_key]
            recent_performance = history[-10:]  # Last 10 similar tasks
            avg_efficiency = np.mean([p['efficiency'] for p in recent_performance])
            return avg_efficiency

        # Cold start: Use expert weight network prediction
        with torch.no_grad():
            predicted_weights = self.expert_weight_network(task_features)
            if predicted_weights.dim() > 1:
                predicted_weights = predicted_weights.squeeze(0)

            if expert_name in self.expert_names:
                expert_idx = self.expert_names.index(expert_name)
                if expert_idx < len(predicted_weights):
                    predicted_accuracy = predicted_weights[expert_idx].item()
                else:
                    predicted_accuracy = 0.5
            else:
                predicted_accuracy = 0.5

        # Efficiency = predicted accuracy / resource cost
        profile = self.expert_profiles[expert_name]
        resource_cost = profile.compute_cost + profile.inference_time
        efficiency = predicted_accuracy / max(resource_cost, 1.0)

        return efficiency
    
    
    def should_add_expert(self, css: CognitiveStateSpace, resource_budget: ResourceBudget,
                         iteration: int) -> Optional[str]:
        """
        Decide whether to add another expert based on current progress
        """
        if iteration >= self.config["max_iterations"]:
            return None
            
        # Check if confidence is improving
        if len(css.convergence_history) >= 2:
            recent_confidence = css.convergence_history[-1]['max_confidence']
            prev_confidence = css.convergence_history[-2]['max_confidence']
            confidence_improvement = recent_confidence - prev_confidence
            
            # If improvement is minimal, consider adding an expert
            if confidence_improvement < 0.05 and recent_confidence < resource_budget.target_confidence:
                # Find unused experts that we can afford
                used_experts = set(css.expert_beliefs.keys())
                available_experts = set(self.expert_names) - used_experts
                
                for expert_name in available_experts:
                    profile = self.expert_profiles[expert_name]
                    if resource_budget.can_afford(profile):
                        return expert_name
        
        return None
    
    def compute_expert_weights(self, task, css: CognitiveStateSpace) -> Dict[str, float]:
        """Compute expert weights for belief fusion"""
        task_features = self.extract_task_features(task)

        with torch.no_grad():
            weight_vector = self.expert_weight_network(task_features)

        if weight_vector.dim() > 1:
            weight_vector = weight_vector.squeeze(0)

        # Clamp to [0, 1] for safety (Softmax should already ensure this)
        weight_vector = torch.clamp(weight_vector, 0.0, 1.0)

        # Map weights to expert names
        expert_weights = {}
        for i, expert_name in enumerate(self.expert_names):
            if i < len(weight_vector):
                weight_val = weight_vector[i].item()
                # Additional safety: ensure valid float
                expert_weights[expert_name] = max(0.0, min(1.0, weight_val))
            else:
                expert_weights[expert_name] = 1.0 / len(self.expert_names)
        
        # Normalize weights for experts that actually participated
        participating_experts = list(css.expert_beliefs.keys())
        if participating_experts:
            total_weight = sum(expert_weights[name] for name in participating_experts)
            if total_weight > 0:
                for name in participating_experts:
                    expert_weights[name] /= total_weight
        
        return expert_weights
    
    def extract_convergence_features(self, css: CognitiveStateSpace) -> torch.Tensor:
        """Extract features for convergence prediction"""
        features = []
        
        # Basic convergence metrics
        if css.convergence_history:
            recent = css.convergence_history[-1]
            features.extend([
                recent['max_confidence'],
                recent['confidence_variance'],
                recent['num_hypotheses'] / 10.0,  # Normalize
                recent['num_experts'] / len(self.expert_names),
                css.current_round / self.config["max_iterations"]
            ])
            
            # Confidence trend (if we have enough history)
            if len(css.convergence_history) >= 3:
                confidences = [h['max_confidence'] for h in css.convergence_history[-3:]]
                trend = (confidences[-1] - confidences[0]) / 2  # Slope
                features.append(trend)
            else:
                features.append(0.0)
                
            # Variance trend
            if len(css.convergence_history) >= 2:
                var_trend = (css.convergence_history[-1]['confidence_variance'] - 
                           css.convergence_history[-2]['confidence_variance'])
                features.append(var_trend)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        
        # Expert agreement metrics
        if css.expert_beliefs and len(css.expert_beliefs) > 1:
            # Calculate agreement between experts
            expert_belief_vectors = []
            hypotheses = list(css.hypotheses.keys())
            
            for expert_beliefs in css.expert_beliefs.values():
                belief_vector = [expert_beliefs.get(h_id, 0.0) for h_id in hypotheses]
                expert_belief_vectors.append(belief_vector)
            
            if len(expert_belief_vectors) >= 2:
                # Calculate pairwise correlations
                correlations = []
                for i in range(len(expert_belief_vectors)):
                    for j in range(i+1, len(expert_belief_vectors)):
                        corr = np.corrcoef(expert_belief_vectors[i], expert_belief_vectors[j])[0,1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                avg_agreement = np.mean(correlations) if correlations else 0.0
                features.append(avg_agreement)
            else:
                features.append(1.0)  # Perfect agreement with one expert
        else:
            features.append(0.0)
        
        # Resource utilization
        total_experts_used = len(css.expert_beliefs)
        expert_utilization = total_experts_used / max(len(self.expert_names), 1)
        features.append(expert_utilization)
        
        # Hypothesis stability (how much hypotheses are changing)
        hypothesis_stability = 0.0
        if css.hypotheses:
            stable_hypotheses = sum(1 for h in css.hypotheses.values() 
                                  if len(h.refinement_history) == 0)
            hypothesis_stability = stable_hypotheses / len(css.hypotheses)
        features.append(hypothesis_stability)
        
        # Pad to expected size
        while len(features) < 10:
            features.append(0.0)
        features = features[:10]

        # Validate all features: replace NaN/inf with 0.0 and clamp to reasonable range
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        # Clamp trend values (which can be negative) to [-1, 1], others to [0, 1]
        features = [max(-1.0, min(1.0, f)) for f in features]

        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def should_terminate(self, css: CognitiveStateSpace, resource_budget: ResourceBudget,
                        iteration: int) -> bool:
        """
        Decide whether to terminate reasoning based on convergence and resources
        """
        # Hard limits
        if iteration >= self.config["max_iterations"]:
            return True
            
        if (resource_budget.used_compute >= resource_budget.max_compute or
            resource_budget.used_time >= resource_budget.max_time):
            return True
        
        # Check convergence
        if css.is_converged(self.config["convergence_threshold"], 
                           self.config["confidence_variance_threshold"]):
            return True
        
        # Check if confidence has peaked
        if css.has_peaked(self.config["early_stopping_patience"]):
            return True
        
        # Use learned convergence prediction
        convergence_features = self.extract_convergence_features(css)
        with torch.no_grad():
            convergence_prob = self.convergence_predictor(convergence_features).item()
        
        # Terminate if high probability of convergence
        if convergence_prob > 0.9:
            return True
        
        # Check if target confidence reached
        if css.convergence_history:
            current_confidence = css.convergence_history[-1]['max_confidence']
            if current_confidence >= resource_budget.target_confidence:
                return True
        
        return False
    
    def solve_task(self, task, resource_budget: ResourceBudget = None, training: bool = True) -> Any:
        """
        Solve a task using resource-bounded iterative refinement
        
        Args:
            task: Task object to solve
            resource_budget: Resource constraints
            training: Whether to update networks
            
        Returns:
            Final solution/prediction
        """
        start_time = time.time()
        
        # Set default resource budget if not provided
        if resource_budget is None:
            resource_budget = ResourceBudget(
                max_compute=self.config["default_compute_budget"],
                max_time=self.config["default_time_budget"],
                max_memory=self.config["default_memory_budget"]
            )
        
        # Initialize cognitive state space
        css = CognitiveStateSpace()
        
        # Select initial experts
        selected_experts = self.select_initial_experts(task, resource_budget)
        
        episode_record = {
            "task_id": getattr(task, 'task_id', 'unknown'),
            "start_time": start_time,
            "selected_experts": selected_experts,
            "iterations": [],
            "final_confidence": 0.0,
            "resource_usage": {},
            "convergence_achieved": False
        }
        
        try:
            # Main reasoning loop
            iteration = 0
            while not self.should_terminate(css, resource_budget, iteration):
                iteration_start = time.time()
                
                # Determine which experts to run this iteration
                if iteration == 0:
                    # First iteration: run selected experts to generate initial hypotheses
                    active_experts = selected_experts
                else:
                    # Subsequent iterations: refine with existing experts or add new ones
                    active_experts = list(css.expert_beliefs.keys())
                    
                    # Consider adding a new expert if progress is slow
                    new_expert = self.should_add_expert(css, resource_budget, iteration)
                    if new_expert:
                        active_experts.append(new_expert)
                
                iteration_record = {
                    "iteration": iteration,
                    "active_experts": active_experts,
                    "start_time": iteration_start,
                    "hypotheses_generated": 0,
                    "expert_confidences": {}
                }
                
                # Run experts
                for expert_name in active_experts:
                    if expert_name not in self.expert_modules:
                        continue

                    expert = self.expert_modules[expert_name]
                    profile = self.expert_profiles[expert_name]

                    # Check if we can afford this expert
                    if not resource_budget.can_afford(profile):
                        continue

                    print(f"  [{expert_name}] ", end='', flush=True)
                    expert_start = time.time()

                    try:
                        if iteration == 0 or expert_name not in css.expert_beliefs:
                            # Generate initial hypotheses
                            new_hypotheses = expert.generate_hypotheses(task, css)
                            
                            # Add hypotheses to CSS
                            expert_beliefs = {}
                            for hypothesis in new_hypotheses:
                                css.add_hypothesis(hypothesis)
                                expert_beliefs[hypothesis.hypothesis_id] = hypothesis.confidence
                                iteration_record["hypotheses_generated"] += 1
                            
                            css.update_expert_beliefs(expert_name, expert_beliefs)
                        else:
                            # Refine existing hypotheses
                            existing_hypotheses = css.get_all_hypotheses()
                            expert_beliefs = expert.refine_hypotheses(task, css, existing_hypotheses)
                            css.update_expert_beliefs(expert_name, expert_beliefs)
                        
                        # Track expert performance
                        expert_time = time.time() - expert_start
                        max_confidence = max(expert_beliefs.values()) if expert_beliefs else 0.0
                        iteration_record["expert_confidences"][expert_name] = max_confidence
                        
                        expert.log_performance(expert_time, 0.0, max_confidence)  # Accuracy TBD
                        
                        # Consume resources
                        resource_budget.consume(profile)
                        
                    except Exception as e:
                        print(f"Error running expert {expert_name}: {e}")
                        continue
                
                # Create belief distribution for this round
                css.current_round = iteration
                belief_dist = css.create_belief_distribution(iteration)
                
                # Compute expert weights and fuse beliefs
                expert_weights = self.compute_expert_weights(task, css)
                fused_beliefs = belief_dist.get_fused_beliefs(expert_weights)
                
                # Track convergence metrics
                if len(fused_beliefs) > 0:
                    max_confidence = torch.max(fused_beliefs).item()
                    confidence_variance = torch.var(fused_beliefs).item()
                else:
                    max_confidence = 0.0
                    confidence_variance = 1.0
                
                css.track_convergence(max_confidence, confidence_variance)
                
                iteration_record.update({
                    "end_time": time.time(),
                    "duration": time.time() - iteration_start,
                    "max_confidence": max_confidence,
                    "confidence_variance": confidence_variance,
                    "fused_beliefs": fused_beliefs.tolist(),
                    "expert_weights": expert_weights
                })
                
                episode_record["iterations"].append(iteration_record)
                iteration += 1
            
            # Get final solution
            final_weights = self.compute_expert_weights(task, css)
            best_hypothesis = css.get_consensus_hypothesis(final_weights)
            
            final_confidence = 0.0
            if css.convergence_history:
                final_confidence = css.convergence_history[-1]['max_confidence']
            
            episode_record.update({
                "end_time": time.time(),
                "total_duration": time.time() - start_time,
                "final_confidence": final_confidence,
                "convergence_achieved": css.is_converged(),
                "total_iterations": iteration,
                "resource_usage": {
                    "compute": resource_budget.used_compute,
                    "time": resource_budget.used_time,
                    "memory": resource_budget.used_memory
                },
                "efficiency": final_confidence / max(resource_budget.used_compute, 1.0)
            })
            
            # Store episode for training
            if training:
                self.episode_history.append(episode_record)
                self._update_networks(task, css, episode_record)

                # Record expert performance for learning
                task_features = self.extract_task_features(task)
                for expert_name in selected_experts:
                    if expert_name in css.expert_beliefs:
                        # Calculate expert's accuracy (final confidence as proxy)
                        expert_confidence = max(css.expert_beliefs[expert_name].values()) if css.expert_beliefs[expert_name] else 0.0

                        # Calculate efficiency
                        profile = self.expert_profiles[expert_name]
                        resource_cost = profile.compute_cost + profile.inference_time
                        efficiency = expert_confidence / max(resource_cost, 1.0)

                        # Record performance
                        self._record_expert_performance(expert_name, task_features, expert_confidence, efficiency)

            return best_hypothesis.content if best_hypothesis else None
            
        except Exception as e:
            episode_record.update({
                "error": str(e),
                "traceback": traceback.format_exc(),
                "end_time": time.time()
            })
            print(f"Error in solve_task: {e}")
            traceback.print_exc()
            return None
    
    def _update_networks(self, task, css: CognitiveStateSpace, episode_record: Dict):
        """Update neural networks based on episode performance"""
        try:
            # Extract training data
            task_features = self.extract_task_features(task)
            final_confidence = episode_record["final_confidence"]
            efficiency = episode_record["efficiency"]
            convergence_achieved = episode_record["convergence_achieved"]
            
            # Update task complexity predictor
            # Clamp all values to [0, 1] to prevent training instability
            actual_complexity = min(1.0, len(episode_record["iterations"]) / self.config["max_iterations"])
            actual_compute = min(1.0, episode_record["resource_usage"]["compute"] / self.config["default_compute_budget"])
            actual_time = min(1.0, episode_record["resource_usage"]["time"] / self.config["default_time_budget"])
            final_confidence_clamped = max(0.0, min(1.0, final_confidence))

            target_complexity = torch.tensor([
                actual_complexity, actual_compute, actual_time, final_confidence_clamped
            ], device=self.device)
            
            predicted_complexity = self.task_complexity_predictor(task_features)
            if predicted_complexity.dim() > 1:
                predicted_complexity = predicted_complexity.squeeze(0)
            
            complexity_loss = nn.MSELoss()(predicted_complexity, target_complexity)
            
            self.complexity_optimizer.zero_grad()
            complexity_loss.backward()
            self.complexity_optimizer.step()
            
            # Update expert weight network (if we have expert performance data)
            if len(episode_record["iterations"]) > 0:
                # Use final iteration expert weights and performance
                final_iteration = episode_record["iterations"][-1]
                expert_weights = final_iteration["expert_weights"]
                expert_confidences = final_iteration["expert_confidences"]
                
                # Create target weights based on expert performance
                target_weights = torch.zeros(len(self.expert_names), device=self.device)
                for i, expert_name in enumerate(self.expert_names):
                    if expert_name in expert_confidences:
                        # Clamp expert confidence to [0, 1] to prevent invalid loss values
                        confidence = max(0.0, min(1.0, expert_confidences[expert_name]))
                        target_weights[i] = confidence

                # Normalize target weights
                if target_weights.sum() > 0:
                    target_weights = target_weights / target_weights.sum()

                    # Clamp normalized weights to [0, 1] for safety
                    target_weights = torch.clamp(target_weights, 0.0, 1.0)

                    predicted_weights = self.expert_weight_network(task_features)
                    if predicted_weights.dim() > 1:
                        predicted_weights = predicted_weights.squeeze(0)

                    weight_loss = nn.MSELoss()(predicted_weights, target_weights)

                    self.weight_optimizer.zero_grad()
                    weight_loss.backward()
                    self.weight_optimizer.step()
                else:
                    # No experts ran - skip weight update
                    print("  Skipping weight update: no experts ran")
            
            # Update convergence predictor
            if len(css.convergence_history) > 0:
                convergence_features = self.extract_convergence_features(css)
                convergence_target = torch.tensor([1.0 if convergence_achieved else 0.0],
                                                device=self.device)

                convergence_pred = self.convergence_predictor(convergence_features)

                # Validate prediction before loss calculation
                if torch.isnan(convergence_pred).any() or torch.isinf(convergence_pred).any():
                    print("  Invalid convergence prediction detected, skipping update")
                    return

                # Clamp prediction to [0, 1] to avoid BCELoss assertion errors
                convergence_pred = torch.clamp(convergence_pred, 0.0, 1.0)

                # Ensure target is also valid
                if torch.isnan(convergence_target).any() or torch.isinf(convergence_target).any():
                    print("  Invalid convergence target detected, skipping update")
                    return

                convergence_loss = nn.BCELoss()(convergence_pred, convergence_target)
                
                self.convergence_optimizer.zero_grad()
                convergence_loss.backward()
                self.convergence_optimizer.step()
                
        except Exception as e:
            print(f"Error updating networks: {e}")
    
    def add_expert_module(self, name: str, module: BaseExpertModule, 
                         resource_profile: ExpertResourceProfile = None):
        """Add a new expert module to the system"""
        self.expert_modules[name] = module
        self.expert_names.append(name)
        
        if resource_profile:
            self.expert_profiles[name] = resource_profile
        else:
            # Create default profile
            self.expert_profiles[name] = ExpertResourceProfile(
                expert_name=name,
                compute_cost=30,
                memory_requirement=3,
                inference_time=8
            )
        
        # Recreate expert weight network with new size
        old_network = self.expert_weight_network
        self.expert_weight_network = ExpertWeightNetwork(
            task_embedding_dim=self.config["task_embedding_dim"],
            num_experts=len(self.expert_modules)
        ).to(self.device)
        
        # Copy old weights where possible
        with torch.no_grad():
            old_state = old_network.state_dict()
            new_state = self.expert_weight_network.state_dict()
            
            for key in old_state:
                if key in new_state:
                    old_tensor = old_state[key]
                    new_tensor = new_state[key]
                    
                    # Copy compatible dimensions
                    if old_tensor.shape == new_tensor.shape:
                        new_state[key] = old_tensor
                    elif len(old_tensor.shape) == len(new_tensor.shape):
                        # Handle size mismatch in last dimension (expert count)
                        if old_tensor.shape[:-1] == new_tensor.shape[:-1]:
                            min_size = min(old_tensor.shape[-1], new_tensor.shape[-1])
                            new_state[key][..., :min_size] = old_tensor[..., :min_size]
            
            self.expert_weight_network.load_state_dict(new_state)
        
        # Update optimizer
        self.weight_optimizer = optim.Adam(
            self.expert_weight_network.parameters(),
            lr=self.config["learning_rate"]
        )
    
    def get_performance_summary(self) -> Dict:
        """Get summary of system performance"""
        if not self.episode_history:
            return {}

        recent_episodes = self.episode_history[-50:]  # Last 50 episodes

        return {
            "average_confidence": np.mean([ep["final_confidence"] for ep in recent_episodes]),
            "average_iterations": np.mean([ep["total_iterations"] for ep in recent_episodes]),
            "average_efficiency": np.mean([ep["efficiency"] for ep in recent_episodes]),
            "convergence_rate": np.mean([ep["convergence_achieved"] for ep in recent_episodes]),
            "average_compute_usage": np.mean([ep["resource_usage"]["compute"] for ep in recent_episodes]),
            "average_time_usage": np.mean([ep["resource_usage"]["time"] for ep in recent_episodes]),
            "total_episodes": len(self.episode_history)
        }

    def get_state(self) -> Dict:
        """Get the current state of the MCU for saving"""
        return {
            'task_complexity_predictor': self.task_complexity_predictor.state_dict(),
            'expert_weight_network': self.expert_weight_network.state_dict(),
            'convergence_predictor': self.convergence_predictor.state_dict(),
            'expert_performance_matrix': self.expert_performance_matrix,
            'episode_history': self.episode_history[-100:],  # Save last 100 episodes
            'config': self.config
        }

    def load_state(self, state: Dict):
        """Load a saved MCU state"""
        self.task_complexity_predictor.load_state_dict(state['task_complexity_predictor'])
        self.expert_weight_network.load_state_dict(state['expert_weight_network'])
        self.convergence_predictor.load_state_dict(state['convergence_predictor'])
        self.expert_performance_matrix = state.get('expert_performance_matrix', {})
        self.episode_history = state.get('episode_history', [])

        # Reinitialize optimizers with loaded networks
        self.task_optimizer = optim.Adam(
            self.task_complexity_predictor.parameters(),
            lr=self.config["learning_rate"]
        )
        self.weight_optimizer = optim.Adam(
            self.expert_weight_network.parameters(),
            lr=self.config["learning_rate"]
        )
        self.convergence_optimizer = optim.Adam(
            self.convergence_predictor.parameters(),
            lr=self.config["learning_rate"]
        )
