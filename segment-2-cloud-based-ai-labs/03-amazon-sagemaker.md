# Amazon SageMaker

## Introduction

Amazon SageMaker is AWS's comprehensive, fully managed machine learning platform that enables data scientists and developers to build, train, and deploy ML models at scale. Unlike Bedrock (which focuses on foundation models), SageMaker provides complete control over the entire ML lifecycle.

## What is Amazon SageMaker?

**Amazon SageMaker** is an end-to-end ML platform offering:
- Integrated development environment (SageMaker Studio)
- Data preparation and feature engineering
- Model training (built-in algorithms and custom)
- Hyperparameter tuning
- Model deployment and hosting
- MLOps and monitoring tools
- AutoML capabilities

**Key Difference from Bedrock:**
- **Bedrock:** Use pre-trained foundation models via API (serverless)
- **SageMaker:** Build, train, and deploy custom ML models (full control)

## Core Components

### 1. SageMaker Studio

**What:** Web-based IDE for ML development

**Features:**
- Jupyter notebooks
- Visual workflow designer
- Integrated debugging
- Git integration
- Collaborative workspace
- Resource management

**Use Cases:**
- Data exploration and analysis
- Model development
- Experiment tracking
- Team collaboration

### 2. SageMaker Notebooks

**Options:**
- **Studio Notebooks:** Integrated in SageMaker Studio
- **Notebook Instances:** Standalone EC2-based notebooks

**Pre-installed:**
- Python, R
- TensorFlow, PyTorch, MXNet, Scikit-learn
- CUDA for GPU acceleration
- AWS SDK (boto3)

**Instance Types:**
- ml.t3.medium (2 vCPU, 4GB RAM): $0.05/hour
- ml.g5.xlarge (1 A10G GPU): $1.41/hour
- ml.p4d.24xlarge (8 A100 GPUs): $40.97/hour

### 3. SageMaker Data Wrangler

**What:** Visual data preparation tool

**Capabilities:**
- Import data from S3, Athena, Redshift
- 300+ built-in transformations
- Custom transformations (Python/SQL)
- Data quality insights
- Export to Feature Store or training

**Pricing:** $0.40 per hour (ml.m5.4xlarge)

### 4. SageMaker Feature Store

**What:** Centralized repository for ML features

**Benefits:**
- Share features across teams
- Consistent feature definitions
- Online and offline storage
- Feature versioning
- Low-latency serving

**Use Cases:**
- Reusable feature engineering
- Real-time inference
- Training-serving consistency

**Pricing:**
- Write: $0.0000011 per write unit
- Read (online): $0.0000011 per read unit
- Storage: $0.025 per GB-month

### 5. SageMaker Training

**Training Options:**
- **Built-in Algorithms:** Pre-optimized (XGBoost, Linear Learner, etc.)
- **Script Mode:** Bring your own code (TensorFlow, PyTorch, etc.)
- **Custom Containers:** Complete control

**Distributed Training:**
- Data parallelism
- Model parallelism
- Distributed training libraries

**Managed Spot Training:**
- Up to 90% cost savings
- Automatic checkpointing
- Good for fault-tolerant workloads

**Pricing:**
- ml.m5.xlarge (4 vCPU, 16GB): $0.23/hour
- ml.g5.xlarge (1 A10G GPU): $1.41/hour
- ml.p4d.24xlarge (8 A100 GPUs): $40.97/hour
- Spot instances: Up to 90% discount

### 6. SageMaker Autopilot (AutoML)

**What:** Automated machine learning

**Process:**
1. Upload data
2. Select target column
3. Autopilot explores algorithms
4. Generates model candidates
5. Provides explainability
6. Deploy best model

**Supported Problem Types:**
- Binary classification
- Multi-class classification
- Regression
- Time series forecasting

**Transparency:**
- View generated notebooks
- Understand feature engineering
- Model selection rationale

**Pricing:** Based on underlying compute usage

### 7. SageMaker Deployment

**Deployment Options:**

**Real-Time Inference:**
- Low-latency predictions
- Auto-scaling
- A/B testing
- Multi-model endpoints

**Serverless Inference:**
- Pay per inference
- Automatic scaling
- No idle costs
- Good for intermittent traffic

**Batch Transform:**
- Process large datasets
- Cost-effective for bulk predictions
- No persistent endpoint

**Asynchronous Inference:**
- Long-running predictions
- Queue-based
- Large payloads (up to 1GB)

**Pricing:**
- Real-time: Instance cost + data processing
- Serverless: $0.20 per million inferences + compute
- Batch: Instance cost during job

### 8. SageMaker Model Monitor

**What:** Continuous monitoring for deployed models

**Monitors:**
- Data quality (schema drift, missing values)
- Model quality (accuracy degradation)
- Bias drift
- Feature attribution drift

**Alerts:**
- CloudWatch alarms
- SNS notifications
- Automated retraining triggers

**Pricing:** $0.24 per monitoring hour

### 9. SageMaker Pipelines

**What:** CI/CD for ML workflows

**Components:**
- Pipeline steps (processing, training, evaluation)
- Conditional execution
- Model registry integration
- Automated deployment

**Benefits:**
- Reproducible workflows
- Version control
- Automated retraining
- Governance and compliance

**Pricing:** $0.03 per pipeline execution step

### 10. SageMaker Model Registry

**What:** Central model catalog

**Features:**
- Model versioning
- Approval workflows
- Metadata tracking
- Deployment history
- Model lineage

**Pricing:** No additional cost

## Getting Started

### Prerequisites

1. AWS Account
2. IAM permissions for SageMaker
3. S3 bucket for data and models
4. (Optional) VPC configuration

### Step 1: Set Up SageMaker Studio

```python
import boto3

sagemaker_client = boto3.client('sagemaker')

# Create SageMaker Studio domain (one-time setup)
response = sagemaker_client.create_domain(
    DomainName='my-studio-domain',
    AuthMode='IAM',
    DefaultUserSettings={
        'ExecutionRole': 'arn:aws:iam::123456789012:role/SageMakerRole'
    },
    SubnetIds=['subnet-12345'],
    VpcId='vpc-12345'
)
```

### Step 2: Train a Model (Script Mode)

**Training Script (train.py):**

```python
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    
    # Data directories
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
```

**Training Job:**

```python
from sagemaker.sklearn import SKLearn

sklearn_estimator = SKLearn(
    entry_point='train.py',
    role='arn:aws:iam::123456789012:role/SageMakerRole',
    instance_type='ml.m5.xlarge',
    instance_count=1,
    framework_version='1.2-1',
    hyperparameters={
        'n-estimators': 200,
        'max-depth': 15
    }
)

# Start training
sklearn_estimator.fit({'train': 's3://my-bucket/train-data/'})
```

### Step 3: Deploy Model

```python
# Deploy to real-time endpoint
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='my-sklearn-endpoint'
)

# Make predictions
import numpy as np
predictions = predictor.predict(np.array([[1.0, 2.0, 3.0, 4.0]]))
print(predictions)
```

### Step 4: Use AutoML (Autopilot)

```python
from sagemaker.automl.automl import AutoML

automl = AutoML(
    role='arn:aws:iam::123456789012:role/SageMakerRole',
    target_attribute_name='target',
    max_candidates=10,
    problem_type='BinaryClassification'
)

# Start AutoML job
automl.fit(
    inputs='s3://my-bucket/train-data/train.csv',
    job_name='my-automl-job'
)

# Deploy best model
predictor = automl.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

## Best Practices

### 1. Cost Optimization

**Strategies:**
- Use managed spot training (90% savings)
- Stop notebook instances when not in use
- Use serverless inference for low traffic
- Batch transform for bulk predictions
- Right-size instances
- Use S3 Intelligent-Tiering

**Example Savings:**
```
Regular Training: ml.p4d.24xlarge × 10 hours = $409.70
Spot Training: Same × 10 hours = $40.97 (90% savings)
```

### 2. Experiment Tracking

```python
from sagemaker.experiments import Run

with Run(
    experiment_name='my-experiment',
    run_name='run-1'
) as run:
    # Log parameters
    run.log_parameter('learning_rate', 0.01)
    run.log_parameter('batch_size', 32)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    run.log_metric('train_accuracy', 0.95)
    run.log_metric('val_accuracy', 0.92)
```

### 3. Model Monitoring

```python
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor

# Enable data capture
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri='s3://my-bucket/data-capture'
)

predictor = model.deploy(
    data_capture_config=data_capture_config
)

# Create monitor
monitor = DefaultModelMonitor(
    role='arn:aws:iam::123456789012:role/SageMakerRole',
    instance_type='ml.m5.xlarge',
    instance_count=1,
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

# Create monitoring schedule
monitor.create_monitoring_schedule(
    endpoint_input=predictor.endpoint_name,
    output_s3_uri='s3://my-bucket/monitoring-output',
    schedule_cron_expression='cron(0 * * * ? *)'  # Hourly
)
```

### 4. MLOps Pipeline

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep

# Define steps
processing_step = ProcessingStep(...)
training_step = TrainingStep(...)

# Conditional deployment
condition = ConditionGreaterThanOrEqualTo(
    left=training_step.properties.FinalMetricDataList['accuracy'],
    right=0.90
)

condition_step = ConditionStep(
    name='CheckAccuracy',
    conditions=[condition],
    if_steps=[deploy_step],
    else_steps=[fail_step]
)

# Create pipeline
pipeline = Pipeline(
    name='my-pipeline',
    steps=[processing_step, training_step, condition_step]
)

pipeline.upsert(role_arn='arn:aws:iam::123456789012:role/SageMakerRole')
pipeline.start()
```

## Use Cases

### 1. Custom Model Training
- Train models on proprietary data
- Use custom algorithms
- Full control over training process

### 2. Large-Scale Training
- Distributed training across multiple GPUs
- Train on massive datasets
- Reduce training time

### 3. MLOps and Production
- Automated retraining pipelines
- Model monitoring and drift detection
- A/B testing and gradual rollouts

### 4. AutoML for Rapid Prototyping
- Quickly explore model options
- Baseline model generation
- Feature engineering insights

## Comparison: SageMaker vs. Bedrock

| Feature | SageMaker | Bedrock |
|---------|-----------|---------|
| **Use Case** | Custom ML models | Foundation models |
| **Control** | Full control | Limited (API-based) |
| **Complexity** | Higher | Lower |
| **Setup Time** | Hours to days | Minutes |
| **Customization** | Complete | Fine-tuning, RAG |
| **Infrastructure** | Managed instances | Serverless |
| **Pricing** | Per-hour compute | Per-token |
| **Best For** | Custom ML, MLOps | Quick LLM deployment |

## Limitations

**Current Limitations:**
- Requires ML expertise
- More complex than Bedrock
- Higher learning curve
- Manual infrastructure management (compared to serverless)

**Considerations:**
- Cost can be high for large-scale training
- Need to manage notebook instances
- Requires understanding of ML workflows

## Resources

- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/)
- [SageMaker Studio Lab (Free)](https://studiolab.sagemaker.aws/)

## Next Steps

- **[Google Vertex AI](./04-google-vertex-ai.md)** - Google's ML platform
- **[Azure AI Foundry](./05-azure-ai-foundry.md)** - Microsoft's AI tools
- **[Cost Management](./07-cost-management.md)** - Optimize spending


