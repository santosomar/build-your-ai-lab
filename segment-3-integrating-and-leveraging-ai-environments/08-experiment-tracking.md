# Experiment Tracking for AI Projects

## Introduction

Experiment tracking is essential for managing machine learning projects effectively. As you iterate through different models, hyperparameters, and datasets, tracking experiments helps you:

- Reproduce successful results
- Compare different approaches systematically
- Collaborate with team members
- Document your research process
- Make data-driven decisions

This guide covers the most popular experiment tracking tools and best practices for AI labs.

## Why Experiment Tracking Matters

### Common Challenges Without Tracking

- **Lost Results**: "Which hyperparameters gave us 95% accuracy?"
- **Wasted Compute**: Re-running experiments you've already tried
- **Collaboration Issues**: Team members duplicating work
- **Irreproducibility**: Unable to recreate successful experiments
- **Decision Paralysis**: No systematic way to compare approaches

### Benefits of Proper Tracking

- **Reproducibility**: Recreate any experiment exactly
- **Comparison**: Systematically compare approaches
- **Collaboration**: Share results with team
- **Documentation**: Automatic research log
- **Optimization**: Identify what works best

## Popular Experiment Tracking Tools

### Comparison Matrix

| Tool | Hosting | Cost | Best For | Key Features |
|------|---------|------|----------|--------------|
| **MLflow** | Self-hosted | Free | Full control, on-premise | Model registry, deployment |
| **Weights & Biases** | Cloud/Self-hosted | Free tier | Teams, visualization | Real-time tracking, sweeps |
| **TensorBoard** | Self-hosted | Free | TensorFlow users | Built-in, simple |
| **Neptune.ai** | Cloud | Free tier | Enterprise | Metadata management |
| **Comet.ml** | Cloud | Free tier | Research teams | Code tracking, model registry |
| **Aim** | Self-hosted | Free | Open-source | Fast, lightweight |

## MLflow

### Installation and Setup

```bash
# Install MLflow
pip install mlflow

# Start MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# Access UI at http://localhost:5000
```

### Basic Usage

```python
# train_with_mlflow.py
import mlflow
import mlflow.pytorch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Set tracking URI (if using remote server)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("sentiment-analysis")

# Start run
with mlflow.start_run(run_name="bert-base-experiment-1"):
    # Log parameters
    mlflow.log_param("model_name", "bert-base-uncased")
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 3)
    
    # Train model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    
    # Log metrics
    mlflow.log_metric("eval_accuracy", metrics["eval_accuracy"])
    mlflow.log_metric("eval_loss", metrics["eval_loss"])
    mlflow.log_metric("eval_f1", metrics["eval_f1"])
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("training_plot.png")
    
    # Add tags
    mlflow.set_tag("dataset", "imdb")
    mlflow.set_tag("task", "sentiment-analysis")
```

### MLflow Model Registry

```python
# Register model
import mlflow.pytorch

# Log and register model
with mlflow.start_run():
    # Training code...
    
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="sentiment-classifier"
    )

# Load model from registry
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get latest version
latest_version = client.get_latest_versions("sentiment-classifier", stages=["Production"])[0]

# Load model
model_uri = f"models:/sentiment-classifier/{latest_version.version}"
model = mlflow.pytorch.load_model(model_uri)

# Transition model stage
client.transition_model_version_stage(
    name="sentiment-classifier",
    version=latest_version.version,
    stage="Production"
)
```

### MLflow Projects

```yaml
# MLproject
name: sentiment-analysis

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 2e-5}
      batch_size: {type: int, default: 32}
      epochs: {type: int, default: 3}
    command: "python train.py --lr {learning_rate} --batch-size {batch_size} --epochs {epochs}"
```

```bash
# Run MLflow project
mlflow run . -P learning_rate=1e-5 -P batch_size=16

# Run from Git
mlflow run https://github.com/user/project.git -P learning_rate=1e-5
```

## Weights & Biases (W&B)

### Setup

```bash
# Install wandb
pip install wandb

# Login
wandb login
```

### Basic Usage

```python
# train_with_wandb.py
import wandb
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# Initialize wandb
wandb.init(
    project="sentiment-analysis",
    name="bert-base-experiment-1",
    config={
        "model_name": "bert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 32,
        "epochs": 3,
        "dataset": "imdb"
    }
)

# Access config
config = wandb.config

# Train model
model = AutoModelForSequenceClassification.from_pretrained(config.model_name)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=config.epochs,
    per_device_train_batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    logging_steps=100,
    report_to="wandb",  # Automatic logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train (automatically logs to wandb)
trainer.train()

# Log additional metrics
wandb.log({
    "custom_metric": 0.95,
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=predictions,
        class_names=["negative", "positive"]
    )
})

# Log artifacts
wandb.save("model.pth")
wandb.save("config.yaml")

# Finish run
wandb.finish()
```

### W&B Sweeps (Hyperparameter Tuning)

```yaml
# sweep_config.yaml
program: train.py
method: bayes
metric:
  name: eval_accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-3
  batch_size:
    values: [16, 32, 64]
  epochs:
    values: [3, 5, 10]
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5
```

```python
# train.py with sweep support
import wandb

def train():
    # Initialize wandb
    wandb.init()
    
    # Get hyperparameters from sweep
    config = wandb.config
    
    # Train with these hyperparameters
    model = create_model(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        dropout=config.dropout
    )
    
    # Training loop
    for epoch in range(config.epochs):
        train_loss = train_epoch(model)
        val_accuracy = validate(model)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy
        })

if __name__ == "__main__":
    train()
```

```bash
# Create sweep
wandb sweep sweep_config.yaml

# Run sweep agent
wandb agent username/project/sweep_id
```

### W&B Tables for Dataset Visualization

```python
# Log dataset samples
import wandb

# Create table
columns = ["text", "label", "prediction", "confidence"]
data = [
    ["Great movie!", "positive", "positive", 0.98],
    ["Terrible film", "negative", "negative", 0.95],
    ["Not bad", "positive", "negative", 0.52],
]

table = wandb.Table(data=data, columns=columns)

# Log table
wandb.log({"predictions": table})
```

## TensorBoard

### Setup

```bash
# Install TensorBoard
pip install tensorboard

# Start TensorBoard
tensorboard --logdir=runs --port=6006

# Access at http://localhost:6006
```

### Usage with PyTorch

```python
# train_with_tensorboard.py
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

# Create writer
writer = SummaryWriter('runs/experiment_1')

# Log hyperparameters
hparams = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10
}
writer.add_hparams(hparams, {})

# Training loop
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Training step
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log scalar
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('Loss/train', loss.item(), global_step)
        
        # Log learning rate
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)
    
    # Validation
    val_accuracy = validate(model, val_loader)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
    
    # Log model graph (once)
    if epoch == 0:
        writer.add_graph(model, images)
    
    # Log images
    writer.add_images('Predictions', pred_images, epoch)
    
    # Log histogram of weights
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

# Close writer
writer.close()
```

### Usage with Transformers

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",  # TensorBoard log directory
    logging_steps=100,
    report_to="tensorboard",  # Enable TensorBoard logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## Aim

### Setup

```bash
# Install Aim
pip install aim

# Initialize Aim repository
aim init

# Start Aim UI
aim up

# Access at http://localhost:43800
```

### Usage

```python
# train_with_aim.py
from aim import Run
import torch

# Create run
run = Run(
    repo='.',
    experiment='sentiment-analysis'
)

# Log hyperparameters
run['hparams'] = {
    'learning_rate': 2e-5,
    'batch_size': 32,
    'epochs': 3,
    'model': 'bert-base-uncased'
}

# Training loop
for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        # Training step
        loss = train_step(batch)
        
        # Log metrics
        run.track(loss, name='loss', step=epoch * len(train_loader) + i, context={'subset': 'train'})
    
    # Validation
    val_metrics = validate()
    run.track(val_metrics['accuracy'], name='accuracy', epoch=epoch, context={'subset': 'val'})
    run.track(val_metrics['f1'], name='f1', epoch=epoch, context={'subset': 'val'})

# Log model
run['model'] = model.state_dict()

# Close run
run.close()
```

## Custom Experiment Tracking

### Simple JSON-Based Tracking

```python
# simple_tracker.py
import json
from datetime import datetime
from pathlib import Path
import hashlib

class ExperimentTracker:
    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        self.current_experiment = None
    
    def start_experiment(self, name, config):
        """Start a new experiment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = hashlib.md5(f"{name}_{timestamp}".encode()).hexdigest()[:8]
        exp_name = f"{name}_{timestamp}_{exp_id}"
        
        exp_dir = self.experiments_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        self.current_experiment = {
            'id': exp_id,
            'name': name,
            'dir': exp_dir,
            'config': config,
            'metrics': {},
            'started_at': timestamp,
            'status': 'running'
        }
        
        # Save config
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        return exp_id
    
    def log_metric(self, name, value, step=None):
        """Log a metric"""
        if name not in self.current_experiment['metrics']:
            self.current_experiment['metrics'][name] = []
        
        self.current_experiment['metrics'][name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_params(self, params):
        """Log parameters"""
        self.current_experiment['config'].update(params)
    
    def save_artifact(self, file_path, artifact_name=None):
        """Save an artifact"""
        import shutil
        
        file_path = Path(file_path)
        if artifact_name is None:
            artifact_name = file_path.name
        
        dest = self.current_experiment['dir'] / 'artifacts' / artifact_name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(file_path, dest)
    
    def end_experiment(self, status='completed'):
        """End current experiment"""
        self.current_experiment['status'] = status
        self.current_experiment['ended_at'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save experiment metadata
        with open(self.current_experiment['dir'] / 'experiment.json', 'w') as f:
            json.dump({
                'id': self.current_experiment['id'],
                'name': self.current_experiment['name'],
                'config': self.current_experiment['config'],
                'metrics': self.current_experiment['metrics'],
                'started_at': self.current_experiment['started_at'],
                'ended_at': self.current_experiment['ended_at'],
                'status': status
            }, f, indent=2)
    
    def list_experiments(self):
        """List all experiments"""
        experiments = []
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                exp_file = exp_dir / 'experiment.json'
                if exp_file.exists():
                    with open(exp_file) as f:
                        experiments.append(json.load(f))
        return experiments
    
    def compare_experiments(self, exp_ids, metric_name):
        """Compare experiments by a metric"""
        results = []
        for exp_id in exp_ids:
            experiments = self.list_experiments()
            exp = next((e for e in experiments if e['id'] == exp_id), None)
            if exp and metric_name in exp['metrics']:
                final_value = exp['metrics'][metric_name][-1]['value']
                results.append({
                    'id': exp_id,
                    'name': exp['name'],
                    metric_name: final_value
                })
        return results

# Usage
tracker = ExperimentTracker()

# Start experiment
exp_id = tracker.start_experiment(
    name="bert-sentiment",
    config={
        'model': 'bert-base-uncased',
        'learning_rate': 2e-5,
        'batch_size': 32
    }
)

# Training loop
for epoch in range(3):
    train_loss = train_epoch()
    val_accuracy = validate()
    
    tracker.log_metric('train_loss', train_loss, step=epoch)
    tracker.log_metric('val_accuracy', val_accuracy, step=epoch)

# Save model
tracker.save_artifact('model.pth')

# End experiment
tracker.end_experiment(status='completed')

# List all experiments
experiments = tracker.list_experiments()
print(f"Total experiments: {len(experiments)}")

# Compare experiments
comparison = tracker.compare_experiments(
    exp_ids=['abc123', 'def456'],
    metric_name='val_accuracy'
)
print(comparison)
```

## Best Practices

### 1. Track Everything

```python
# Comprehensive tracking
with mlflow.start_run():
    # Hyperparameters
    mlflow.log_params({
        'model_name': 'bert-base',
        'learning_rate': 2e-5,
        'batch_size': 32,
        'epochs': 3,
        'optimizer': 'adamw',
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'max_length': 512
    })
    
    # System info
    mlflow.log_params({
        'gpu': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'python_version': sys.version
    })
    
    # Dataset info
    mlflow.log_params({
        'dataset': 'imdb',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset)
    })
    
    # Training metrics
    mlflow.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'train_time_seconds': train_time
    })
    
    # Model artifacts
    mlflow.log_artifact('model.pth')
    mlflow.log_artifact('config.yaml')
    mlflow.log_artifact('training_curves.png')
```

### 2. Use Meaningful Names

```python
# Good: Descriptive experiment names
run_name = f"bert-base_lr{lr}_bs{batch_size}_{timestamp}"

# Bad: Generic names
run_name = "experiment_1"
```

### 3. Tag Experiments

```python
# Add tags for easy filtering
mlflow.set_tags({
    'task': 'sentiment-analysis',
    'dataset': 'imdb',
    'model_family': 'bert',
    'status': 'production',
    'team': 'nlp-research'
})
```

### 4. Version Your Code

```python
import git

# Log git commit hash
repo = git.Repo(search_parent_directories=True)
commit_hash = repo.head.object.hexsha

mlflow.log_param('git_commit', commit_hash)
mlflow.log_param('git_branch', repo.active_branch.name)
```

### 5. Create Experiment Templates

```python
# experiment_template.py
class ExperimentTemplate:
    def __init__(self, tracking_uri, experiment_name):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def run_experiment(self, config, train_fn, eval_fn):
        """Standard experiment workflow"""
        with mlflow.start_run(run_name=config['run_name']):
            # Log all config
            mlflow.log_params(config)
            
            # Log git info
            self._log_git_info()
            
            # Log system info
            self._log_system_info()
            
            # Train
            model, train_metrics = train_fn(config)
            mlflow.log_metrics(train_metrics)
            
            # Evaluate
            eval_metrics = eval_fn(model)
            mlflow.log_metrics(eval_metrics)
            
            # Save model
            mlflow.pytorch.log_model(model, "model")
            
            return model, eval_metrics
    
    def _log_git_info(self):
        try:
            repo = git.Repo(search_parent_directories=True)
            mlflow.log_param('git_commit', repo.head.object.hexsha)
            mlflow.log_param('git_branch', repo.active_branch.name)
        except:
            pass
    
    def _log_system_info(self):
        mlflow.log_params({
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'pytorch_version': torch.__version__
        })

# Usage
template = ExperimentTemplate('http://localhost:5000', 'sentiment-analysis')

config = {
    'run_name': 'bert-base-experiment',
    'model_name': 'bert-base-uncased',
    'learning_rate': 2e-5,
    'batch_size': 32
}

model, metrics = template.run_experiment(config, train_model, evaluate_model)
```

## Comparing Experiments

### MLflow Comparison

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all runs from experiment
experiment = client.get_experiment_by_name("sentiment-analysis")
runs = client.search_runs(experiment.experiment_id)

# Compare metrics
import pandas as pd

data = []
for run in runs:
    data.append({
        'run_id': run.info.run_id,
        'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
        'learning_rate': run.data.params.get('learning_rate'),
        'batch_size': run.data.params.get('batch_size'),
        'val_accuracy': run.data.metrics.get('val_accuracy'),
        'val_f1': run.data.metrics.get('val_f1')
    })

df = pd.DataFrame(data)
df = df.sort_values('val_accuracy', ascending=False)
print(df)

# Find best run
best_run = df.iloc[0]
print(f"\nBest run: {best_run['run_name']}")
print(f"Accuracy: {best_run['val_accuracy']:.4f}")
```

### W&B Comparison

```python
import wandb

api = wandb.Api()

# Get runs from project
runs = api.runs("username/sentiment-analysis")

# Create comparison table
data = []
for run in runs:
    data.append({
        'name': run.name,
        'learning_rate': run.config.get('learning_rate'),
        'batch_size': run.config.get('batch_size'),
        'val_accuracy': run.summary.get('val_accuracy'),
        'val_f1': run.summary.get('val_f1')
    })

df = pd.DataFrame(data)
print(df.sort_values('val_accuracy', ascending=False))
```

## Reproducing Experiments

### Save Complete Environment

```python
# save_environment.py
import mlflow
import torch
import json
import sys

with mlflow.start_run():
    # Log all dependencies
    import pkg_resources
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    with open('requirements.txt', 'w') as f:
        for package, version in installed_packages.items():
            f.write(f"{package}=={version}\n")
    
    mlflow.log_artifact('requirements.txt')
    
    # Log Python version
    mlflow.log_param('python_version', sys.version)
    
    # Log CUDA version
    if torch.cuda.is_available():
        mlflow.log_param('cuda_version', torch.version.cuda)
    
    # Log complete config
    config = {
        'model': 'bert-base-uncased',
        'hyperparameters': {...},
        'data_preprocessing': {...},
        'training_procedure': {...}
    }
    
    with open('complete_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    mlflow.log_artifact('complete_config.json')
```

### Reproduce from MLflow

```python
# reproduce_experiment.py
import mlflow

# Load run
run_id = "abc123def456"
run = mlflow.get_run(run_id)

# Get parameters
params = run.data.params

# Download artifacts
client = mlflow.tracking.MlflowClient()
artifact_path = client.download_artifacts(run_id, "")

# Load model
model_uri = f"runs:/{run_id}/model"
model = mlflow.pytorch.load_model(model_uri)

# Reproduce training
reproduce_training(params, model)
```

## Conclusion

Experiment tracking is crucial for systematic AI development. Whether you choose MLflow for full control, W&B for advanced features, or a simple custom solution, the key is to track consistently and comprehensively.

Key recommendations:
- **Choose the right tool**: MLflow for self-hosted, W&B for teams
- **Track everything**: Hyperparameters, metrics, code, environment
- **Use meaningful names**: Make experiments easy to identify
- **Compare systematically**: Use built-in comparison tools
- **Ensure reproducibility**: Save complete environment and code

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Aim Documentation](https://aimstack.readthedocs.io/)
- [Neptune.ai Documentation](https://docs.neptune.ai/)

