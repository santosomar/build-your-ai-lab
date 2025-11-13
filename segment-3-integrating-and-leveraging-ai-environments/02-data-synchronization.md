# Synchronizing Data and Projects Between Environments

## Introduction

Effective data synchronization is the backbone of a successful hybrid AI lab. Whether you're moving datasets between local and cloud storage, keeping code repositories in sync, or managing model artifacts across environments, proper synchronization ensures consistency, reproducibility, and efficient collaboration.

This guide covers the tools, techniques, and best practices for synchronizing data and projects in hybrid AI environments.

## Types of Data to Synchronize

### 1. Source Code

- Python scripts, notebooks, configuration files
- Training and inference code
- Deployment scripts and infrastructure code

### 2. Datasets

- Raw data files (CSV, JSON, images, videos)
- Preprocessed and augmented datasets
- Training, validation, and test splits

### 3. Models

- Trained model weights and checkpoints
- Model architectures and configurations
- Quantized and optimized model versions

### 4. Experiment Artifacts

- Training logs and metrics
- Hyperparameter configurations
- Evaluation results and visualizations

### 5. Environment Configurations

- Docker images and containers
- Conda/pip environment specifications
- System dependencies and configurations

## Synchronization Strategies

### Strategy 1: Version Control for Code (Git)

Git is the gold standard for code synchronization across environments.

#### Basic Git Workflow

```bash
# Initialize repository
git init
git remote add origin https://github.com/username/ai-project.git

# Daily workflow
git add .
git commit -m "Add new training script"
git push origin main

# On another machine
git clone https://github.com/username/ai-project.git
git pull origin main
```

#### Git Best Practices for AI Projects

```bash
# Create .gitignore for AI projects
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
venv/
env/

# Data files (too large for Git)
data/
datasets/
*.csv
*.h5
*.pkl

# Model files
models/
checkpoints/
*.pth
*.ckpt
*.safetensors

# Experiment outputs
logs/
outputs/
wandb/
mlruns/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOF
```

#### Git LFS for Large Files

For files that must be in Git but are too large:

```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "*.onnx"

# Commit and push
git add .gitattributes
git add model.pth
git commit -m "Add model with LFS"
git push origin main
```

### Strategy 2: Data Version Control (DVC)

DVC extends Git to handle large datasets and models efficiently.

#### Setting Up DVC

```bash
# Install DVC
pip install dvc

# Initialize DVC in your Git repo
cd your-project
git init
dvc init

# Configure remote storage (S3 example)
dvc remote add -d myremote s3://mybucket/dvcstore
dvc remote modify myremote region us-west-2

# Or use other storage backends
# dvc remote add -d myremote gs://mybucket/dvcstore  # Google Cloud
# dvc remote add -d myremote azure://mycontainer/path  # Azure
# dvc remote add -d myremote /mnt/shared/dvcstore  # Local/NFS
```

#### Using DVC for Dataset Management

```bash
# Add a dataset to DVC
dvc add data/training_data.csv

# This creates data/training_data.csv.dvc
# Commit the .dvc file to Git
git add data/training_data.csv.dvc data/.gitignore
git commit -m "Add training dataset"

# Push data to remote storage
dvc push

# On another machine, pull the data
git pull
dvc pull
```

#### DVC Pipeline for Reproducibility

```yaml
# dvc.yaml - Define your ML pipeline
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw
    outs:
      - data/prepared

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/prepared
    params:
      - train.learning_rate
      - train.epochs
    outs:
      - models/model.pth
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pth
      - data/test
    metrics:
      - evaluation.json:
          cache: false
```

```bash
# Run the pipeline
dvc repro

# View metrics
dvc metrics show

# Compare experiments
dvc metrics diff
```

### Strategy 3: Cloud Storage Synchronization

#### Using AWS S3

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Sync local directory to S3
aws s3 sync ./data s3://my-bucket/data --exclude "*.tmp"

# Sync from S3 to local
aws s3 sync s3://my-bucket/data ./data

# Sync with delete (mirror)
aws s3 sync ./data s3://my-bucket/data --delete
```

#### Using Rclone (Multi-Cloud)

Rclone supports 40+ cloud storage providers:

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure remote (interactive)
rclone config

# Sync to cloud
rclone sync ./data remote:my-bucket/data -P

# Sync from cloud
rclone sync remote:my-bucket/data ./data -P

# Bidirectional sync
rclone bisync ./data remote:my-bucket/data -P
```

#### Rclone Configuration Example

```ini
# ~/.config/rclone/rclone.conf
[aws-s3]
type = s3
provider = AWS
access_key_id = YOUR_ACCESS_KEY
secret_access_key = YOUR_SECRET_KEY
region = us-west-2

[gcs]
type = google cloud storage
project_number = 123456789
service_account_file = /path/to/credentials.json

[azure]
type = azureblob
account = myaccount
key = mykey
```

### Strategy 4: Rsync for Efficient Transfers

Rsync is excellent for incremental synchronization:

```bash
# Basic sync
rsync -avz ./data/ user@remote:/path/to/data/

# Sync with progress and compression
rsync -avzP ./data/ user@remote:/path/to/data/

# Exclude certain files
rsync -avz --exclude '*.tmp' --exclude '.git' ./data/ user@remote:/path/to/data/

# Dry run (see what would be transferred)
rsync -avzn ./data/ user@remote:/path/to/data/

# Delete files on destination that don't exist in source
rsync -avz --delete ./data/ user@remote:/path/to/data/
```

#### Rsync Over SSH with Compression

```bash
# Create SSH config for easier access
cat >> ~/.ssh/config << EOF
Host cloud-gpu
    HostName 54.123.45.67
    User ubuntu
    IdentityFile ~/.ssh/cloud-key.pem
    Compression yes
EOF

# Now sync easily
rsync -avzP ./models/ cloud-gpu:/home/ubuntu/models/
```

## Automated Synchronization

### Using Cron for Scheduled Sync

```bash
# Edit crontab
crontab -e

# Add sync jobs
# Sync data every hour
0 * * * * /usr/local/bin/rclone sync /home/user/data remote:bucket/data

# Sync models every 6 hours
0 */6 * * * /usr/local/bin/aws s3 sync /home/user/models s3://my-bucket/models

# Backup to cloud daily at 2 AM
0 2 * * * /usr/local/bin/rsync -avz /home/user/projects/ backup-server:/backups/
```

### Python Script for Automated Sync

```python
#!/usr/bin/env python3
"""
Automated synchronization script for AI projects
"""
import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataSynchronizer:
    def __init__(self, config):
        self.config = config
        
    def sync_to_cloud(self, local_path, remote_path):
        """Sync local data to cloud storage"""
        try:
            logging.info(f"Syncing {local_path} to {remote_path}")
            
            # Using rclone
            cmd = [
                'rclone', 'sync',
                local_path,
                remote_path,
                '--progress',
                '--transfers', '4',
                '--checkers', '8',
                '--exclude', '*.tmp',
                '--exclude', '.git/**'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Successfully synced {local_path}")
                return True
            else:
                logging.error(f"Sync failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error during sync: {e}")
            return False
    
    def sync_from_cloud(self, remote_path, local_path):
        """Sync data from cloud to local"""
        try:
            logging.info(f"Syncing {remote_path} to {local_path}")
            
            cmd = [
                'rclone', 'sync',
                remote_path,
                local_path,
                '--progress',
                '--transfers', '4',
                '--checkers', '8'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Successfully synced to {local_path}")
                return True
            else:
                logging.error(f"Sync failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error during sync: {e}")
            return False
    
    def bidirectional_sync(self, local_path, remote_path):
        """Bidirectional synchronization"""
        try:
            logging.info(f"Bidirectional sync: {local_path} <-> {remote_path}")
            
            cmd = [
                'rclone', 'bisync',
                local_path,
                remote_path,
                '--resilient',
                '--recover',
                '--verbose'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info("Bidirectional sync completed")
                return True
            else:
                logging.error(f"Bidirectional sync failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error during bidirectional sync: {e}")
            return False

# Configuration
config = {
    'data_dirs': [
        {'local': '/home/user/data/raw', 'remote': 'aws-s3:my-bucket/data/raw'},
        {'local': '/home/user/data/processed', 'remote': 'aws-s3:my-bucket/data/processed'},
    ],
    'model_dirs': [
        {'local': '/home/user/models', 'remote': 'aws-s3:my-bucket/models'},
    ]
}

if __name__ == '__main__':
    syncer = DataSynchronizer(config)
    
    # Sync all configured directories
    for data_dir in config['data_dirs']:
        syncer.sync_to_cloud(data_dir['local'], data_dir['remote'])
    
    for model_dir in config['model_dirs']:
        syncer.sync_to_cloud(model_dir['local'], model_dir['remote'])
```

### Using Watchdog for Real-Time Sync

```python
#!/usr/bin/env python3
"""
Real-time file synchronization using watchdog
"""
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SyncHandler(FileSystemEventHandler):
    def __init__(self, local_path, remote_path):
        self.local_path = local_path
        self.remote_path = remote_path
        self.last_sync = time.time()
        self.sync_delay = 5  # seconds
        
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Debounce: only sync if enough time has passed
        current_time = time.time()
        if current_time - self.last_sync > self.sync_delay:
            self.sync()
            self.last_sync = current_time
    
    def sync(self):
        print(f"Syncing {self.local_path} to {self.remote_path}")
        subprocess.run([
            'rclone', 'sync',
            self.local_path,
            self.remote_path,
            '--exclude', '*.tmp'
        ])

if __name__ == '__main__':
    path_to_watch = '/home/user/projects/ai-lab'
    remote_path = 'aws-s3:my-bucket/projects/ai-lab'
    
    event_handler = SyncHandler(path_to_watch, remote_path)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()
    
    print(f"Watching {path_to_watch} for changes...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()
```

## Model Synchronization

### Using Hugging Face Hub

```python
from huggingface_hub import HfApi, hf_hub_download, upload_file
import os

# Initialize API
api = HfApi()

# Upload model to Hugging Face Hub
def upload_model_to_hub(model_path, repo_id, token):
    """Upload model to Hugging Face Hub"""
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id,
        token=token,
        repo_type="model"
    )
    print(f"Model uploaded to {repo_id}")

# Download model from Hugging Face Hub
def download_model_from_hub(repo_id, filename, local_dir):
    """Download model from Hugging Face Hub"""
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir
    )
    print(f"Model downloaded to {model_path}")
    return model_path

# Example usage
upload_model_to_hub(
    model_path="./models/my_model.pth",
    repo_id="username/my-model",
    token="hf_..."
)

download_model_from_hub(
    repo_id="username/my-model",
    filename="my_model.pth",
    local_dir="./models"
)
```

### Using MLflow Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

# Configure MLflow tracking
mlflow.set_tracking_uri("http://mlflow-server:5000")

# Log and register model
with mlflow.start_run():
    # Train model
    model = train_model()
    
    # Log model
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="my-classifier"
    )
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)

# Download model from registry
client = MlflowClient()
model_uri = "models:/my-classifier/production"
model = mlflow.pytorch.load_model(model_uri)
```

## Conflict Resolution

### Handling Sync Conflicts

```python
def resolve_conflict(local_file, remote_file):
    """Simple conflict resolution strategy"""
    import hashlib
    from datetime import datetime
    
    def file_hash(filepath):
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    local_hash = file_hash(local_file)
    remote_hash = file_hash(remote_file)
    
    if local_hash == remote_hash:
        print("Files are identical, no conflict")
        return "no_conflict"
    
    # Strategy 1: Keep both versions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{local_file}.conflict.{timestamp}"
    os.rename(local_file, backup_file)
    print(f"Conflict: Created backup at {backup_file}")
    
    # Strategy 2: Keep newer file (by modification time)
    local_mtime = os.path.getmtime(local_file)
    remote_mtime = os.path.getmtime(remote_file)
    
    if local_mtime > remote_mtime:
        print("Keeping local version (newer)")
        return "keep_local"
    else:
        print("Keeping remote version (newer)")
        return "keep_remote"
```

## Best Practices

### 1. Separate Code and Data

- **Code**: Use Git for version control
- **Data**: Use DVC, cloud storage, or specialized data versioning tools
- **Models**: Use model registries (MLflow, Hugging Face Hub)

### 2. Use .gitignore and .dvcignore

```bash
# .gitignore
data/
models/
*.pth
*.h5

# .dvcignore
.git/
*.tmp
__pycache__/
```

### 3. Implement Checksums

Always verify data integrity after transfer:

```bash
# Generate checksum before transfer
md5sum large_dataset.tar.gz > dataset.md5

# Verify after transfer
md5sum -c dataset.md5
```

```python
import hashlib

def verify_file_integrity(filepath, expected_hash):
    """Verify file integrity using MD5 hash"""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    actual_hash = md5_hash.hexdigest()
    if actual_hash == expected_hash:
        print(f"✓ File integrity verified: {filepath}")
        return True
    else:
        print(f"✗ File corrupted: {filepath}")
        return False
```

### 4. Bandwidth Optimization

```bash
# Compress before transfer
tar -czf data.tar.gz data/
aws s3 cp data.tar.gz s3://bucket/

# Use rsync with compression
rsync -avz --compress-level=9 ./data/ remote:/data/

# Rclone with bandwidth limit
rclone sync ./data remote:bucket/data --bwlimit 10M
```

### 5. Incremental Backups

```bash
# Rsync incremental backup with hard links
rsync -avz --link-dest=/backup/previous /data/ /backup/current/

# Rclone with backup directory
rclone sync ./data remote:bucket/data --backup-dir remote:bucket/backup/$(date +%Y%m%d)
```

### 6. Monitor Sync Status

```python
import subprocess
import json
from datetime import datetime

def check_sync_status(local_path, remote_path):
    """Check if local and remote are in sync"""
    result = subprocess.run(
        ['rclone', 'check', local_path, remote_path, '--one-way'],
        capture_output=True,
        text=True
    )
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'in_sync': result.returncode == 0,
        'output': result.stdout,
        'errors': result.stderr
    }
    
    return status

# Log sync status
status = check_sync_status('./data', 'aws-s3:bucket/data')
with open('sync_status.json', 'w') as f:
    json.dump(status, f, indent=2)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Slow Transfer Speeds

**Solutions**:
- Use compression: `rsync -avz` or `rclone sync --compress`
- Increase parallel transfers: `rclone sync --transfers 16`
- Use faster compression: `pigz` instead of `gzip`
- Check network bandwidth and latency

#### Issue 2: Sync Failures

**Solutions**:
- Enable retry logic: `rclone sync --retries 5 --low-level-retries 10`
- Check credentials and permissions
- Verify network connectivity
- Check disk space on both ends

#### Issue 3: Large File Handling

**Solutions**:
- Use multipart uploads for cloud storage
- Split large files: `split -b 1G large_file.tar.gz`
- Use resumable transfers: `rclone sync --resilient`
- Consider using torrents for very large datasets

## Conclusion

Effective data synchronization is crucial for hybrid AI labs. By combining the right tools (Git, DVC, rclone, rsync) with automation and best practices, you can maintain consistency across environments while optimizing for speed, reliability, and cost.

Key takeaways:
- Use Git for code, DVC for data and models
- Automate synchronization with scripts and cron jobs
- Always verify data integrity
- Optimize bandwidth usage with compression and incremental sync
- Monitor and log sync operations for troubleshooting

## Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [Rclone Documentation](https://rclone.org/docs/)
- [Git LFS](https://git-lfs.github.com/)
- [AWS CLI S3 Commands](https://docs.aws.amazon.com/cli/latest/reference/s3/)
- [Rsync Manual](https://linux.die.net/man/1/rsync)

