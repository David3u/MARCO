import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import os
import time
import json
import copy
import traceback
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

# ============================================================================
# Core Data Structures for Belief-Based Reasoning
# ============================================================================

class Hypothesis:
    """Represents a hypothesis with belief distribution and metadata"""
    
    def __init__(self, hypothesis_id: str, content: Any, source_expert: str, 
                 confidence: float, belief_distribution: Optional[torch.Tensor] = None):
        self.hypothesis_id = hypothesis_id
        self.content = content  # The actual prediction/solution
        self.source_expert = source_expert
        self.confidence = confidence
        self.belief_distribution = belief_distribution or torch.tensor([confidence])
        self.creation_time = time.time()
        self.refinement_history = []
        
    def refine(self, new_content: Any, new_confidence: float, refining_expert: str):
        """Refine the hypothesis with new content and confidence"""
        self.refinement_history.append({
            'old_content': copy.deepcopy(self.content),
            'old_confidence': self.confidence,
            'timestamp': time.time()
        })
        self.content = new_content
        self.confidence = new_confidence
        self.refinement_history[-1]['refining_expert'] = refining_expert

class BeliefDistribution:
    """Manages belief distributions over multiple hypotheses"""
    
    def __init__(self, hypotheses: List[Hypothesis]):
        self.hypotheses = hypotheses
        self.distributions = {}  # expert_name -> distribution over hypotheses
        
    def add_expert_beliefs(self, expert_name: str, beliefs: torch.Tensor):
        """Add an expert's belief distribution over hypotheses"""
        assert len(beliefs) == len(self.hypotheses), "Belief vector must match hypothesis count"
        self.distributions[expert_name] = beliefs
        
    def get_fused_beliefs(self, expert_weights: Dict[str, float]) -> torch.Tensor:
        """Compute weighted product of expert beliefs"""
        if not self.distributions:
            return torch.ones(len(self.hypotheses)) / len(self.hypotheses)
            
        # Initialize with uniform distribution
        fused = torch.ones(len(self.hypotheses))
        total_weight = 0
        
        for expert_name, beliefs in self.distributions.items():
            weight = expert_weights.get(expert_name, 1.0)
            fused *= torch.pow(beliefs, weight)
            total_weight += weight
            
        # Normalize
        fused = fused / fused.sum()
        return fused

class ExpertResourceProfile:
    """Profile of resource requirements for an expert"""

    def __init__(self, expert_name: str, compute_cost: float, memory_requirement: float,
                 inference_time: float):
        self.expert_name = expert_name
        self.compute_cost = compute_cost  # Relative compute units
        self.memory_requirement = memory_requirement  # GB
        self.inference_time = inference_time  # Seconds
        
class ResourceBudget:
    """Resource constraints for a reasoning session"""
    
    def __init__(self, max_compute: float, max_time: float, max_memory: float,
                 target_confidence: float = 0.8):
        self.max_compute = max_compute
        self.max_time = max_time
        self.max_memory = max_memory
        self.target_confidence = target_confidence
        self.used_compute = 0.0
        self.used_time = 0.0
        self.used_memory = 0.0
        
    def can_afford(self, resource_profile: ExpertResourceProfile) -> bool:
        """Check if we can afford to run this expert"""
        return (self.used_compute + resource_profile.compute_cost <= self.max_compute and
                self.used_time + resource_profile.inference_time <= self.max_time and
                self.used_memory + resource_profile.memory_requirement <= self.max_memory)
                
    def consume(self, resource_profile: ExpertResourceProfile):
        """Consume resources for running an expert"""
        self.used_compute += resource_profile.compute_cost
        self.used_time += resource_profile.inference_time
        self.used_memory += resource_profile.memory_requirement

# ============================================================================
# Enhanced Cognitive State Space (CSS) - formerly Blackboard
# ============================================================================

class CognitiveStateSpace:
    """
    Enhanced blackboard for hypothesis-based reasoning with belief tracking
    """
    
    def __init__(self, max_grid_size=30*30):
        # Original blackboard functionality
        self.knowledge_base = {}
        self.knowledge_sources = {}
        self.value_table = np.zeros((11, max_grid_size), dtype=np.uint8)
        self.reasoning_history = []
        self.confidence_scores = {}
        self.textual_data = []
        self.graph_data = None
        self.is_batched_training = False
        self.max_grid_size = max_grid_size
        
        # New hypothesis and belief management
        self.hypotheses = {}  # hypothesis_id -> Hypothesis
        self.belief_distributions = {}  # round_id -> BeliefDistribution
        self.expert_beliefs = {}  # expert_name -> {hypothesis_id -> confidence}
        self.current_round = 0
        self.convergence_history = []
        
        # Resource tracking
        self.resource_usage = []
        self.expert_performance = {}  # expert_name -> performance metrics
        
    def add_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Add a new hypothesis to the CSS"""
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        return hypothesis.hypothesis_id
        
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Retrieve a hypothesis by ID"""
        return self.hypotheses.get(hypothesis_id)
        
    def get_all_hypotheses(self) -> List[Hypothesis]:
        """Get all current hypotheses"""
        return list(self.hypotheses.values())
        
    def update_expert_beliefs(self, expert_name: str, hypothesis_beliefs: Dict[str, float]):
        """Update an expert's beliefs about all hypotheses"""
        if expert_name not in self.expert_beliefs:
            self.expert_beliefs[expert_name] = {}
        self.expert_beliefs[expert_name].update(hypothesis_beliefs)
        
    def create_belief_distribution(self, round_id: int) -> BeliefDistribution:
        """Create belief distribution for current round"""
        hypotheses = self.get_all_hypotheses()
        belief_dist = BeliefDistribution(hypotheses)
        
        # Add each expert's beliefs
        for expert_name, beliefs in self.expert_beliefs.items():
            if beliefs:  # Only if expert has provided beliefs
                belief_vector = torch.tensor([
                    beliefs.get(h.hypothesis_id, 0.0) for h in hypotheses
                ])
                # Normalize to probability distribution
                if belief_vector.sum() > 0:
                    belief_vector = belief_vector / belief_vector.sum()
                belief_dist.add_expert_beliefs(expert_name, belief_vector)
                
        self.belief_distributions[round_id] = belief_dist
        return belief_dist
        
    def get_consensus_hypothesis(self, expert_weights: Dict[str, float]) -> Optional[Hypothesis]:
        """Get the hypothesis with highest consensus belief"""
        if not self.hypotheses:
            return None
            
        current_beliefs = self.belief_distributions.get(self.current_round)
        if not current_beliefs:
            return None
            
        fused_beliefs = current_beliefs.get_fused_beliefs(expert_weights)
        best_idx = torch.argmax(fused_beliefs).item()
        
        hypotheses = list(self.hypotheses.values())
        if best_idx < len(hypotheses):
            return hypotheses[best_idx]
        return None
        
    def track_convergence(self, max_confidence: float, confidence_variance: float):
        """Track convergence metrics for this round"""
        self.convergence_history.append({
            'round': self.current_round,
            'max_confidence': max_confidence,
            'confidence_variance': confidence_variance,
            'num_hypotheses': len(self.hypotheses),
            'num_experts': len(self.expert_beliefs),
            'timestamp': time.time()
        })
        
    def is_converged(self, threshold: float = 0.95, variance_threshold: float = 0.01) -> bool:
        """Check if beliefs have converged"""
        if len(self.convergence_history) < 2:
            return False
            
        recent = self.convergence_history[-1]
        return (recent['max_confidence'] >= threshold and 
                recent['confidence_variance'] <= variance_threshold)
                
    def has_peaked(self, lookback: int = 3) -> bool:
        """Check if confidence has peaked (not improving)"""
        if len(self.convergence_history) < lookback + 1:
            return False
            
        recent_confidences = [h['max_confidence'] for h in self.convergence_history[-lookback:]]
        return all(conf <= recent_confidences[0] + 0.01 for conf in recent_confidences[1:])

# ============================================================================
# Enhanced Base Expert Module
# ============================================================================

class BaseExpertModule(ABC):
    """
    Abstract base class for expert modules that output beliefs and confidence
    """
    
    def __init__(self, expert_name: str, resource_profile: ExpertResourceProfile):
        self.expert_name = expert_name
        self.resource_profile = resource_profile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance tracking
        self.inference_times = []
        self.accuracy_history = []
        self.confidence_calibration = []
        
    @abstractmethod
    def generate_hypotheses(self, task, css: CognitiveStateSpace) -> List[Hypothesis]:
        """
        Generate initial hypotheses for a task
        
        Args:
            task: Task object to solve
            css: Current cognitive state space
            
        Returns:
            List of hypotheses with confidence scores
        """
        pass
        
    @abstractmethod
    def refine_hypotheses(self, task, css: CognitiveStateSpace, 
                         existing_hypotheses: List[Hypothesis]) -> Dict[str, float]:
        """
        Refine existing hypotheses and return belief distribution
        
        Args:
            task: Task object
            css: Current cognitive state space
            existing_hypotheses: Hypotheses to evaluate/refine
            
        Returns:
            Dictionary mapping hypothesis_id to confidence/belief
        """
        pass
        
    def evaluate_task_complexity(self, task) -> float:
        """
        Estimate task complexity for resource planning
        
        Returns:
            Complexity score (0.0 = simple, 1.0 = very complex)
        """
        # Default implementation - can be overridden by specific experts
        if not hasattr(task, 'train_pairs') or not task.train_pairs:
            return 0.5
            
        # Simple heuristics based on grid size and pattern diversity
        complexity = 0.0
        
        for input_grid, output_grid in task.train_pairs:
            input_array = np.array(input_grid)
            output_array = np.array(output_grid)
            
            # Size complexity
            size_factor = (input_array.size / 900.0)  # Normalize by max size
            complexity += size_factor * 0.3
            
            # Color diversity
            unique_colors = len(np.unique(input_array))
            color_factor = unique_colors / 10.0
            complexity += color_factor * 0.2
            
            # Change complexity
            if input_array.shape == output_array.shape:
                changes = np.sum(input_array != output_array) / input_array.size
                complexity += changes * 0.5
                
        return min(complexity / len(task.train_pairs), 1.0)
        
    def log_performance(self, inference_time: float, accuracy: float, confidence: float):
        """Log performance metrics"""
        self.inference_times.append(inference_time)
        self.accuracy_history.append(accuracy)
        self.confidence_calibration.append((confidence, accuracy))

# ============================================================================
# Belief Fusion Networks
# ============================================================================

class ExpertWeightNetwork(nn.Module):
    """
    Neural network to learn LLM expert weights for belief fusion

    Input: 6-dimensional task feature vector
    Output: Weight vector for expert belief fusion (sums to 1)
    """

    def __init__(self, task_embedding_dim: int, num_experts: int, hidden_dim: int = 128):
        super(ExpertWeightNetwork, self).__init__()
        
        self.task_embedding_dim = task_embedding_dim
        self.num_experts = num_experts
        
        # Network to predict expert weights based on task features
        self.weight_network = nn.Sequential(
            nn.Linear(task_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)  # Ensure weights sum to 1
        )
        
    def forward(self, task_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict expert weights for belief fusion
        
        Args:
            task_embedding: Task representation
            
        Returns:
            Weight vector for experts
        """
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0)
            
        weights = self.weight_network(task_embedding)
        return weights

class ConvergencePredictor(nn.Module):
    """Predicts when reasoning should terminate based on current state"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(ConvergencePredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability of convergence
        )
        
    def forward(self, convergence_state: torch.Tensor) -> torch.Tensor:
        """
        Predict probability that reasoning has converged
        
        Args:
            convergence_state: Features about current reasoning state
            
        Returns:
            Probability of convergence (0-1)
        """
        return self.predictor(convergence_state)

# ============================================================================
# Resource-Aware Task Complexity Predictor
# ============================================================================

class TaskComplexityPredictor(nn.Module):
    """
    Predicts task complexity and resource requirements for LLM expert selection

    Input: 6-dimensional task feature vector
    Output: [complexity, compute_need, time_need, accuracy_need] (all in [0,1])
    """

    def __init__(self, task_feature_dim: int, hidden_dim: int = 128):
        super(TaskComplexityPredictor, self).__init__()

        self.complexity_network = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # [complexity, compute_need, time_need, accuracy_need]
            nn.Sigmoid()
        )
        
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Predict task complexity and resource requirements
        
        Returns:
            Tensor with [complexity, compute_requirement, time_requirement, accuracy_requirement]
        """
        if task_features.dim() == 1:
            task_features = task_features.unsqueeze(0)
            
        predictions = self.complexity_network(task_features)
        return predictions