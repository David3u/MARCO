import numpy as np
import torch
import json
import os
from typing import Dict, List, Tuple, Any
from marco_integration_example import MARCOSystem
from marco_enhanced_core import ResourceBudget
from llm_expert_and_task import EnhancedTask

class ARCTaskLoader:
    """Load and preprocess ARC-AGI tasks for training"""

    def __init__(self, arc_data_path: str):
        self.arc_data_path = arc_data_path
        self.tasks = {}
        self.load_tasks()

    def load_tasks(self):
        """Load ARC tasks from JSON files"""
        # Load training tasks
        train_path = os.path.join(self.arc_data_path, "training")
        eval_path = os.path.join(self.arc_data_path, "evaluation")

        for split_name, path in [("training", train_path), ("evaluation", eval_path)]:
            if os.path.exists(path):
                for filename in os.listdir(path):
                    if filename.endswith('.json'):
                        task_id = filename.replace('.json', '')
                        with open(os.path.join(path, filename), 'r') as f:
                            task_data = json.load(f)
                            self.tasks[f"{split_name}_{task_id}"] = self._format_task(task_id, task_data)

    def _format_task(self, task_id: str, arc_task: Dict) -> Dict:
        """Convert ARC task format to MARCO format"""
        train_pairs = []
        test_pairs = []

        # Convert training examples
        for example in arc_task['train']:
            input_grid = example['input']
            output_grid = example['output']
            train_pairs.append((input_grid, output_grid))

        # Convert test examples
        for example in arc_task['test']:
            input_grid = example['input']
            output_grid = example['output']  # Ground truth for evaluation
            test_pairs.append((input_grid, output_grid))

        return {
            'task_id': task_id,
            'train_pairs': train_pairs,
            'test_pairs': test_pairs,
            'original_data': arc_task
        }

    def get_task_by_id(self, task_id: str) -> Dict:
        """Get a specific task by ID"""
        return self.tasks.get(task_id)

    def get_training_tasks(self) -> List[Dict]:
        """Get all training tasks"""
        return [task for task_id, task in self.tasks.items() if task_id.startswith('training_')]

    def get_evaluation_tasks(self) -> List[Dict]:
        """Get all evaluation tasks"""
        return [task for task_id, task in self.tasks.items() if task_id.startswith('evaluation_')]

    def get_task_batch(self, batch_size: int, split: str = 'training') -> List[Dict]:
        """Get a batch of tasks for training"""
        if split == 'training':
            tasks = self.get_training_tasks()
        else:
            tasks = self.get_evaluation_tasks()

        # Randomly sample tasks
        indices = np.random.choice(len(tasks), size=min(batch_size, len(tasks)), replace=False)
        return [tasks[i] for i in indices]

class ARCTrainer:
    """Training pipeline for MARCO system on ARC tasks"""

    def __init__(self, marco_system: MARCOSystem, task_loader: ARCTaskLoader, config: Dict = None):
        self.marco = marco_system
        self.task_loader = task_loader

        # Training configuration
        self.config = {
            "epochs": 100,
            "batch_size": 8,
            "validation_frequency": 10,
            "save_frequency": 20,
            "early_stopping_patience": 15,
            "target_accuracy": 0.9,
            "curriculum_learning": True,
            "difficulty_ramp_epochs": 30,
            "checkpoint_dir": "checkpoints/",
            "log_frequency": 5
        }

        if config:
            self.config.update(config)

        # Training state
        self.epoch = 0
        self.best_validation_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.training_history = {
            "train_accuracy": [],
            "train_loss": [],
            "val_accuracy": [],
            "val_loss": [],
            "efficiency": [],
            "convergence_rate": []
        }

        # Create checkpoint directory
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)

    def calculate_task_difficulty(self, task_data: Dict) -> float:
        """Calculate relative difficulty of a task for curriculum learning"""
        task = EnhancedTask(
            task_data['task_id'],
            task_data['train_pairs'],
            task_data['test_pairs']
        )

        stats = task.get_grid_statistics()
        transforms = task.analyze_transformations()

        # Difficulty factors
        size_factor = stats['avg_grid_size'] / 900.0  # Normalize by max size
        color_factor = stats['unique_colors'] / 10.0

        # Transformation complexity
        if transforms:
            avg_change_ratio = np.mean([t['change_ratio'] for t in transforms])
            shape_changes = sum(1 for t in transforms if t['shape_changed'])
            transform_factor = (avg_change_ratio + shape_changes / len(transforms)) / 2
        else:
            transform_factor = 0.5

        # Training examples (fewer = harder)
        examples_factor = 1.0 - (stats['num_train_pairs'] / 10.0)  # Assume max 10 examples

        difficulty = (size_factor + color_factor + transform_factor + examples_factor) / 4
        return min(1.0, max(0.0, difficulty))

    def get_curriculum_batch(self, batch_size: int, epoch: int) -> List[Dict]:
        """Get tasks based on curriculum learning schedule"""
        if not self.config["curriculum_learning"]:
            return self.task_loader.get_task_batch(batch_size, 'training')

        # Progressive difficulty ramp
        max_difficulty = min(1.0, epoch / self.config["difficulty_ramp_epochs"])

        # Get all training tasks with difficulties
        all_tasks = self.task_loader.get_training_tasks()
        task_difficulties = [(task, self.calculate_task_difficulty(task)) for task in all_tasks]

        # Filter tasks within current difficulty range
        suitable_tasks = [task for task, diff in task_difficulties if diff <= max_difficulty]

        if len(suitable_tasks) < batch_size:
            suitable_tasks = all_tasks  # Fall back to all tasks

        # Sample from suitable tasks
        indices = np.random.choice(len(suitable_tasks), size=min(batch_size, len(suitable_tasks)), replace=False)
        return [suitable_tasks[i] for i in indices]

    def evaluate_solution(self, predicted_grids: List[np.ndarray],
                         true_grids: List[np.ndarray]) -> Tuple[float, float]:
        """Evaluate predicted grids against ground truth"""
        if not predicted_grids or len(predicted_grids) != len(true_grids):
            return 0.0, 0.0

        total_accuracy = 0.0
        perfect_matches = 0

        for pred, true in zip(predicted_grids, true_grids):
            pred_array = np.array(pred)
            true_array = np.array(true)

            # Ensure same shape for comparison
            if pred_array.shape != true_array.shape:
                # Resize to smaller common shape
                min_rows = min(pred_array.shape[0], true_array.shape[0])
                min_cols = min(pred_array.shape[1], true_array.shape[1])
                pred_array = pred_array[:min_rows, :min_cols]
                true_array = true_array[:min_rows, :min_cols]

            if true_array.size > 0:
                # Cell-level accuracy
                matches = np.sum(pred_array == true_array)
                cell_accuracy = matches / true_array.size
                total_accuracy += cell_accuracy

                # Perfect grid match
                if matches == true_array.size:
                    perfect_matches += 1

        avg_accuracy = total_accuracy / len(predicted_grids)
        perfect_rate = perfect_matches / len(predicted_grids)

        return avg_accuracy, perfect_rate

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        batch_tasks = self.get_curriculum_batch(self.config["batch_size"], epoch)

        epoch_metrics = {
            "accuracy": [],
            "perfect_rate": [],
            "efficiency": [],
            "convergence_rate": [],
            "resource_usage": []
        }

        for i, task_data in enumerate(batch_tasks):
            # Create adaptive resource budget based on task difficulty
            difficulty = self.calculate_task_difficulty(task_data)

            # Scale resources with difficulty
            base_compute = self.marco.config["default_compute_budget"]
            base_time = self.marco.config["default_time_budget"]

            resource_budget = ResourceBudget(
                max_compute=base_compute * (0.5 + difficulty),
                max_time=base_time * (0.5 + difficulty),
                max_memory=self.marco.config["default_memory_budget"],
                target_confidence=0.8 + difficulty * 0.15  # Higher confidence for harder tasks
            )

            # Solve task
            result = self.marco.solve_task(task_data, resource_budget)

            # Evaluate solution
            if result and result['solution']:
                true_grids = [pair[1] for pair in task_data['test_pairs']]
                accuracy, perfect_rate = self.evaluate_solution(result['solution'], true_grids)

                epoch_metrics["accuracy"].append(accuracy)
                epoch_metrics["perfect_rate"].append(perfect_rate)
                epoch_metrics["efficiency"].append(result['results']['efficiency'])
                epoch_metrics["convergence_rate"].append(1.0 if result['results']['accuracy'] > 0.8 else 0.0)
                epoch_metrics["resource_usage"].append(
                    sum(result['resource_budget'].values()) / 3
                    if isinstance(result['resource_budget'], dict) else 0.5
                )
            else:
                # Failed solution
                epoch_metrics["accuracy"].append(0.0)
                epoch_metrics["perfect_rate"].append(0.0)
                epoch_metrics["efficiency"].append(0.0)
                epoch_metrics["convergence_rate"].append(0.0)
                epoch_metrics["resource_usage"].append(1.0)  # Max resource usage for failure

            if self.config["log_frequency"] > 0 and (i + 1) % self.config["log_frequency"] == 0:
                print(f"  Batch {i+1}/{len(batch_tasks)}: Accuracy={accuracy:.3f}, Perfect={perfect_rate:.3f}")

        # Aggregate epoch metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}

    def validate(self) -> Dict[str, float]:
        """Run validation on evaluation tasks"""
        val_tasks = self.task_loader.get_task_batch(
            min(20, len(self.task_loader.get_evaluation_tasks())),
            'evaluation'
        )

        val_metrics = {
            "accuracy": [],
            "perfect_rate": [],
            "efficiency": []
        }

        # Use fixed resource budget for validation
        resource_budget = ResourceBudget(
            max_compute=self.marco.config["default_compute_budget"],
            max_time=self.marco.config["default_time_budget"],
            max_memory=self.marco.config["default_memory_budget"],
            target_confidence=self.marco.config["target_confidence"]
        )

        for task_data in val_tasks:
            result = self.marco.solve_task(task_data, resource_budget)

            if result and result['solution']:
                true_grids = [pair[1] for pair in task_data['test_pairs']]
                accuracy, perfect_rate = self.evaluate_solution(result['solution'], true_grids)

                val_metrics["accuracy"].append(accuracy)
                val_metrics["perfect_rate"].append(perfect_rate)
                val_metrics["efficiency"].append(result['results']['efficiency'])
            else:
                val_metrics["accuracy"].append(0.0)
                val_metrics["perfect_rate"].append(0.0)
                val_metrics["efficiency"].append(0.0)

        return {key: np.mean(values) for key, values in val_metrics.items()}

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(self.config["checkpoint_dir"], f"marco_epoch_{epoch}.json")

        checkpoint = {
            "epoch": epoch,
            "metrics": metrics,
            "training_history": self.training_history,
            "config": self.config,
            "best_validation_accuracy": self.best_validation_accuracy
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Save MARCO system state
        system_path = os.path.join(self.config["checkpoint_dir"], f"marco_system_epoch_{epoch}.json")
        self.marco.save_system_state(system_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self):
        """Main training loop"""
        print("Starting MARCO training on ARC-AGI tasks...")
        print(f"Training tasks: {len(self.task_loader.get_training_tasks())}")
        print(f"Evaluation tasks: {len(self.task_loader.get_evaluation_tasks())}")

        for epoch in range(self.config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")

            # Training
            train_metrics = self.train_epoch(epoch)

            # Update training history
            self.training_history["train_accuracy"].append(train_metrics["accuracy"])
            self.training_history["efficiency"].append(train_metrics["efficiency"])
            self.training_history["convergence_rate"].append(train_metrics["convergence_rate"])

            print(f"Train - Accuracy: {train_metrics['accuracy']:.3f}, "
                  f"Perfect: {train_metrics['perfect_rate']:.3f}, "
                  f"Efficiency: {train_metrics['efficiency']:.3f}")

            # Validation
            if (epoch + 1) % self.config["validation_frequency"] == 0:
                val_metrics = self.validate()
                self.training_history["val_accuracy"].append(val_metrics["accuracy"])

                print(f"Val - Accuracy: {val_metrics['accuracy']:.3f}, "
                      f"Perfect: {val_metrics['perfect_rate']:.3f}")

                # Check for improvement
                if val_metrics["accuracy"] > self.best_validation_accuracy:
                    self.best_validation_accuracy = val_metrics["accuracy"]
                    self.epochs_without_improvement = 0

                    # Save best model
                    best_path = os.path.join(self.config["checkpoint_dir"], "best_model.json")
                    self.marco.save_system_state(best_path)
                    print(f"New best validation accuracy: {self.best_validation_accuracy:.3f}")
                else:
                    self.epochs_without_improvement += self.config["validation_frequency"]

            # Save checkpoint
            if (epoch + 1) % self.config["save_frequency"] == 0:
                self.save_checkpoint(epoch + 1, {
                    "train": train_metrics,
                    "val": val_metrics if (epoch + 1) % self.config["validation_frequency"] == 0 else None
                })

            # Early stopping
            if self.epochs_without_improvement >= self.config["early_stopping_patience"]:
                print(f"Early stopping: No improvement for {self.epochs_without_improvement} epochs")
                break

            # Adjust resource strategy
            if (epoch + 1) % 20 == 0:
                self.marco.adjust_resource_strategy(self.config["target_accuracy"])

        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_validation_accuracy:.3f}")

        return self.training_history

# Example usage function
def train_marco_on_arc(arc_data_path: str, config: Dict = None):
    """
    Main function to train MARCO on ARC-AGI tasks

    Args:
        arc_data_path: Path to ARC dataset directory
        config: Optional training configuration
    """

    # Load ARC tasks
    print("Loading ARC tasks...")
    task_loader = ARCTaskLoader(arc_data_path)

    # Initialize MARCO system
    marco_config = {
        "default_compute_budget": 120.0,
        "default_time_budget": 40.0,
        "default_memory_budget": 10.0,
        "target_confidence": 0.85,
        "max_iterations": 6,
        "training_mode": True
    }

    marco_system = MARCOSystem(marco_config)

    # Set up trainer
    training_config = {
        "epochs": 200,
        "batch_size": 12,
        "validation_frequency": 10,
        "save_frequency": 25,
        "early_stopping_patience": 30,
        "target_accuracy": 0.9,
        "curriculum_learning": True,
        "difficulty_ramp_epochs": 50
    }

    if config:
        training_config.update(config)

    trainer = ARCTrainer(marco_system, task_loader, training_config)

    # Start training
    history = trainer.train()

    return marco_system, history

if __name__ == "__main__":
    # Example usage
    arc_path = "/path/to/arc-agi/data"  # Update this path

    # Train the system
    trained_marco, training_history = train_marco_on_arc(arc_path)

    # Print final results
    print("\nTraining Results:")
    print(f"Final train accuracy: {training_history['train_accuracy'][-1]:.3f}")
    print(f"Final validation accuracy: {training_history['val_accuracy'][-1]:.3f}")
    print(f"Final efficiency: {training_history['efficiency'][-1]:.3f}")