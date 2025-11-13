# Data Management and Storage for AI Labs

## Introduction

Effective data management and storage are critical components of any AI lab. As AI projects often involve large datasets, multiple versions, and complex pipelines, having a robust data management strategy ensures reproducibility, efficiency, and scalability.

This guide covers best practices for organizing, storing, versioning, and managing data in hybrid AI environments.

## Data Storage Architecture

### Storage Hierarchy

```
┌─────────────────────────────────────────────────┐
│  Hot Storage (NVMe SSD)                         │
│  - Active datasets                              │
│  - Current experiments                          │
│  - Fast access, expensive                       │
│  Cost: $0.10-0.20/GB                           │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Warm Storage (SATA SSD/HDD)                    │
│  - Recent datasets                              │
│  - Model checkpoints                            │
│  - Moderate access, moderate cost               │
│  Cost: $0.02-0.05/GB                           │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Cold Storage (Cloud/Archive)                   │
│  - Historical datasets                          │
│  - Completed experiments                        │
│  - Infrequent access, cheap                     │
│  Cost: $0.004-0.01/GB                          │
└─────────────────────────────────────────────────┘
```

### Recommended Local Setup

```bash
# Example directory structure
/mnt/
├── nvme0/              # Hot storage (1-2TB NVMe)
│   ├── datasets/       # Active datasets
│   ├── experiments/    # Current experiments
│   └── cache/          # Model cache
├── ssd0/               # Warm storage (4TB SATA SSD)
│   ├── datasets/       # Recent datasets
│   ├── models/         # Trained models
│   └── checkpoints/    # Training checkpoints
└── hdd0/               # Cold storage (8TB HDD)
    ├── archive/        # Old datasets
    ├── backups/        # System backups
    └── raw/            # Raw data
```

## Data Organization

### Project Structure

```
project-name/
├── data/
│   ├── raw/                    # Original, immutable data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── processed/              # Cleaned, processed data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── interim/                # Intermediate transformations
│   └── external/               # External datasets
├── models/
│   ├── checkpoints/            # Training checkpoints
│   ├── final/                  # Final trained models
│   └── pretrained/             # Downloaded pretrained models
├── notebooks/                  # Jupyter notebooks
│   ├── exploratory/
│   └── reports/
├── src/                        # Source code
│   ├── data/                   # Data loading/processing
│   ├── models/                 # Model definitions
│   ├── training/               # Training scripts
│   └── evaluation/             # Evaluation scripts
├── configs/                    # Configuration files
├── outputs/                    # Experiment outputs
│   ├── logs/
│   ├── metrics/
│   └── visualizations/
├── tests/                      # Unit tests
├── requirements.txt
├── README.md
└── .gitignore
```

### Naming Conventions

```bash
# Datasets
dataset_name_version_split.format
# Examples:
imagenet_v2_train.tar
coco_2017_val.zip
custom_dataset_v1.0_test.csv

# Models
model_architecture_dataset_date_version.extension
# Examples:
resnet50_imagenet_20241113_v1.pth
bert_base_squad_20241113_final.safetensors
llama3_finetuned_20241113_checkpoint-1000.bin

# Experiments
experiment_name_date_run_id/
# Examples:
sentiment_analysis_20241113_run001/
image_classification_20241113_run042/
```

## Data Versioning

### Using DVC (Data Version Control)

#### Setup

```bash
# Install DVC
pip install dvc

# Initialize DVC in your Git repository
cd your-project
git init
dvc init

# Configure remote storage
# Local/NFS
dvc remote add -d local /mnt/ssd0/dvc-storage

# AWS S3
dvc remote add -d s3remote s3://my-bucket/dvc-storage
dvc remote modify s3remote region us-west-2

# Google Cloud Storage
dvc remote add -d gcs gs://my-bucket/dvc-storage

# Azure Blob Storage
dvc remote add -d azure azure://mycontainer/path
```

#### Track Data

```bash
# Add dataset to DVC
dvc add data/raw/dataset.csv

# This creates data/raw/dataset.csv.dvc
# Commit the .dvc file to Git
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "Add dataset v1"

# Push data to remote storage
dvc push

# Tag this version
git tag -a v1.0 -m "Dataset version 1.0"
git push origin v1.0
```

#### Retrieve Data

```bash
# Clone repository
git clone https://github.com/user/project.git
cd project

# Pull data
dvc pull

# Switch to specific version
git checkout v1.0
dvc checkout
```

#### Update Data

```bash
# Modify dataset
# Update DVC tracking
dvc add data/raw/dataset.csv

# Commit changes
git add data/raw/dataset.csv.dvc
git commit -m "Update dataset v2"
git tag -a v2.0 -m "Dataset version 2.0"

# Push both code and data
git push origin main
git push origin v2.0
dvc push
```

### DVC Pipelines

```yaml
# dvc.yaml - Define reproducible pipelines
stages:
  download:
    cmd: python src/data/download.py
    outs:
      - data/raw/dataset.zip

  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - data/raw/dataset.zip
      - src/data/preprocess.py
    params:
      - preprocess.train_split
      - preprocess.val_split
    outs:
      - data/processed/train.csv
      - data/processed/val.csv
      - data/processed/test.csv

  train:
    cmd: python src/training/train.py
    deps:
      - data/processed/train.csv
      - data/processed/val.csv
      - src/training/train.py
    params:
      - train.learning_rate
      - train.batch_size
      - train.epochs
    outs:
      - models/final/model.pth
    metrics:
      - outputs/metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluation/evaluate.py
    deps:
      - data/processed/test.csv
      - models/final/model.pth
      - src/evaluation/evaluate.py
    metrics:
      - outputs/metrics/test_metrics.json:
          cache: false
```

```yaml
# params.yaml - Parameters
preprocess:
  train_split: 0.8
  val_split: 0.1
  max_length: 512

train:
  learning_rate: 2e-5
  batch_size: 32
  epochs: 3
  warmup_steps: 500
```

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train

# View metrics
dvc metrics show

# Compare experiments
dvc metrics diff v1.0 v2.0

# Visualize pipeline
dvc dag
```

## Database Solutions

### SQLite for Metadata

```python
# metadata_db.py
import sqlite3
from datetime import datetime
from pathlib import Path

class MetadataDB:
    def __init__(self, db_path="metadata.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """Create database schema"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                path TEXT NOT NULL,
                size_bytes INTEGER,
                num_samples INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                UNIQUE(name, version)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                architecture TEXT NOT NULL,
                dataset_id INTEGER,
                path TEXT NOT NULL,
                size_bytes INTEGER,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model_id INTEGER,
                config TEXT,
                status TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (id)
            )
        """)
        
        self.conn.commit()
    
    def add_dataset(self, name, version, path, size_bytes, num_samples, description=""):
        """Register a dataset"""
        cursor = self.conn.execute("""
            INSERT INTO datasets (name, version, path, size_bytes, num_samples, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, version, path, size_bytes, num_samples, description))
        self.conn.commit()
        return cursor.lastrowid
    
    def add_model(self, name, architecture, dataset_id, path, size_bytes, metrics):
        """Register a model"""
        import json
        cursor = self.conn.execute("""
            INSERT INTO models (name, architecture, dataset_id, path, size_bytes, metrics)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, architecture, dataset_id, path, size_bytes, json.dumps(metrics)))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_dataset(self, name, version):
        """Retrieve dataset information"""
        cursor = self.conn.execute("""
            SELECT * FROM datasets WHERE name = ? AND version = ?
        """, (name, version))
        return cursor.fetchone()
    
    def list_datasets(self):
        """List all datasets"""
        cursor = self.conn.execute("SELECT * FROM datasets ORDER BY created_at DESC")
        return cursor.fetchall()

# Usage
db = MetadataDB()
dataset_id = db.add_dataset(
    name="sentiment_dataset",
    version="v1.0",
    path="/data/processed/sentiment_v1.csv",
    size_bytes=1024*1024*100,  # 100MB
    num_samples=50000,
    description="Twitter sentiment dataset"
)

model_id = db.add_model(
    name="sentiment_classifier",
    architecture="bert-base-uncased",
    dataset_id=dataset_id,
    path="/models/sentiment_classifier.pth",
    size_bytes=1024*1024*500,  # 500MB
    metrics={"accuracy": 0.92, "f1": 0.91}
)
```

### PostgreSQL for Production

```python
# Using SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    path = Column(String, nullable=False)
    size_bytes = Column(Integer)
    num_samples = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    models = relationship("Model", back_populates="dataset")

class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    architecture = Column(String, nullable=False)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    dataset = relationship("Dataset", back_populates="models")

# Create engine
engine = create_engine('postgresql://user:password@localhost/ai_lab')
Base.metadata.create_all(engine)

# Create session
Session = sessionmaker(bind=engine)
session = Session()

# Add data
dataset = Dataset(
    name="imagenet",
    version="v1.0",
    path="/data/imagenet",
    size_bytes=150*1024*1024*1024,  # 150GB
    num_samples=1281167
)
session.add(dataset)
session.commit()
```

## Cloud Storage Integration

### AWS S3

```python
# s3_storage.py
import boto3
from pathlib import Path
from tqdm import tqdm

class S3Storage:
    def __init__(self, bucket_name, region='us-west-2'):
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = bucket_name
    
    def upload_file(self, local_path, s3_key):
        """Upload file to S3 with progress bar"""
        file_size = Path(local_path).stat().st_size
        
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=local_path) as pbar:
            self.s3.upload_file(
                local_path,
                self.bucket,
                s3_key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
    
    def download_file(self, s3_key, local_path):
        """Download file from S3 with progress bar"""
        # Get file size
        response = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
        file_size = response['ContentLength']
        
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=s3_key) as pbar:
            self.s3.download_file(
                self.bucket,
                s3_key,
                local_path,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
    
    def upload_directory(self, local_dir, s3_prefix):
        """Upload entire directory to S3"""
        local_dir = Path(local_dir)
        
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                self.upload_file(str(file_path), s3_key)
    
    def list_objects(self, prefix):
        """List objects with given prefix"""
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]

# Usage
storage = S3Storage('my-ai-bucket')
storage.upload_file('data/dataset.csv', 'datasets/dataset_v1.csv')
storage.download_file('models/model.pth', 'local_models/model.pth')
```

### Google Cloud Storage

```python
# gcs_storage.py
from google.cloud import storage
from pathlib import Path
from tqdm import tqdm

class GCSStorage:
    def __init__(self, bucket_name):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def upload_file(self, local_path, gcs_path):
        """Upload file to GCS"""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
    
    def download_file(self, gcs_path, local_path):
        """Download file from GCS"""
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
    
    def upload_directory(self, local_dir, gcs_prefix):
        """Upload directory to GCS"""
        local_dir = Path(local_dir)
        
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                gcs_path = f"{gcs_prefix}/{relative_path}"
                self.upload_file(str(file_path), gcs_path)

# Usage
storage = GCSStorage('my-ai-bucket')
storage.upload_file('data/dataset.csv', 'datasets/dataset_v1.csv')
```

## Data Loading Optimization

### Efficient Data Loading

```python
# efficient_dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import h5py

class EfficientImageDataset(Dataset):
    """Efficient dataset using HDF5 for fast loading"""
    
    def __init__(self, hdf5_path, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        
        # Open file to get length
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f['images'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Open file for each access (thread-safe)
        with h5py.File(self.hdf5_path, 'r') as f:
            image = f['images'][idx]
            label = f['labels'][idx]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.from_numpy(image), torch.tensor(label)

# Create DataLoader with optimizations
dataset = EfficientImageDataset('data/dataset.h5')
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2  # Prefetch batches
)
```

### Memory-Mapped Files

```python
# memory_mapped.py
import numpy as np

# Create memory-mapped array
data = np.memmap(
    'data/large_dataset.dat',
    dtype='float32',
    mode='r',  # Read-only
    shape=(1000000, 512)  # 1M samples, 512 features
)

# Access like regular array, but doesn't load into RAM
batch = data[0:32]  # Only loads 32 samples
```

### Streaming Large Datasets

```python
# streaming_dataset.py
from datasets import load_dataset

# Load dataset in streaming mode (doesn't download everything)
dataset = load_dataset(
    'imagenet-1k',
    split='train',
    streaming=True
)

# Iterate without loading full dataset
for example in dataset.take(100):
    image = example['image']
    label = example['label']
    # Process example
```

## Data Preprocessing Pipelines

### Apache Arrow and Parquet

```python
# arrow_processing.py
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# Convert CSV to Parquet (much faster to read)
df = pd.read_csv('data/large_dataset.csv')
table = pa.Table.from_pandas(df)
pq.write_table(table, 'data/large_dataset.parquet')

# Read Parquet (10-100x faster than CSV)
table = pq.read_table('data/large_dataset.parquet')
df = table.to_pandas()

# Read specific columns only
table = pq.read_table(
    'data/large_dataset.parquet',
    columns=['text', 'label']
)
```

### Dask for Large Datasets

```python
# dask_processing.py
import dask.dataframe as dd

# Read large CSV that doesn't fit in memory
df = dd.read_csv('data/huge_dataset.csv')

# Perform operations (lazy evaluation)
df = df[df['score'] > 0.5]
df = df.groupby('category').mean()

# Compute result
result = df.compute()

# Save to Parquet
df.to_parquet('data/processed/', compression='snappy')
```

## Backup Strategies

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh - Automated backup script

# Configuration
SOURCE_DIR="/mnt/nvme0/datasets"
BACKUP_DIR="/mnt/hdd0/backups"
S3_BUCKET="s3://my-backup-bucket"
DATE=$(date +%Y%m%d)

# Create local backup
echo "Creating local backup..."
rsync -avz --delete \
    --exclude '*.tmp' \
    --exclude '__pycache__' \
    "$SOURCE_DIR/" "$BACKUP_DIR/datasets_$DATE/"

# Compress old backups
find "$BACKUP_DIR" -name "datasets_*" -mtime +7 -exec tar -czf {}.tar.gz {} \; -exec rm -rf {} \;

# Sync to S3
echo "Syncing to S3..."
aws s3 sync "$BACKUP_DIR" "$S3_BUCKET/backups" --storage-class GLACIER

# Clean up old local backups (keep 30 days)
find "$BACKUP_DIR" -name "datasets_*.tar.gz" -mtime +30 -delete

echo "Backup completed at $(date)"
```

```bash
# Schedule with cron (daily at 2 AM)
0 2 * * * /home/user/scripts/backup.sh >> /var/log/backup.log 2>&1
```

### Incremental Backups with restic

```bash
# Install restic
sudo apt install restic

# Initialize repository
restic init --repo /mnt/hdd0/restic-repo

# Or use S3
restic init --repo s3:s3.amazonaws.com/my-backup-bucket

# Create backup
restic backup /mnt/nvme0/datasets --repo /mnt/hdd0/restic-repo

# List snapshots
restic snapshots --repo /mnt/hdd0/restic-repo

# Restore specific snapshot
restic restore latest --target /restore/path --repo /mnt/hdd0/restic-repo

# Automated backup script
cat > ~/backup_restic.sh << 'EOF'
#!/bin/bash
export RESTIC_REPOSITORY="/mnt/hdd0/restic-repo"
export RESTIC_PASSWORD_FILE="/home/user/.restic-password"

restic backup /mnt/nvme0/datasets
restic forget --keep-daily 7 --keep-weekly 4 --keep-monthly 6
restic prune
EOF

chmod +x ~/backup_restic.sh
```

## Data Lifecycle Management

### Automated Data Archival

```python
# data_lifecycle.py
from pathlib import Path
from datetime import datetime, timedelta
import shutil

class DataLifecycleManager:
    def __init__(self, hot_storage, warm_storage, cold_storage):
        self.hot = Path(hot_storage)
        self.warm = Path(warm_storage)
        self.cold = Path(cold_storage)
    
    def archive_old_data(self, days_hot=7, days_warm=30):
        """Move old data through storage tiers"""
        now = datetime.now()
        
        # Move from hot to warm
        for item in self.hot.rglob('*'):
            if item.is_file():
                age = now - datetime.fromtimestamp(item.stat().st_mtime)
                if age > timedelta(days=days_hot):
                    dest = self.warm / item.relative_to(self.hot)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(dest))
                    print(f"Moved to warm: {item.name}")
        
        # Move from warm to cold
        for item in self.warm.rglob('*'):
            if item.is_file():
                age = now - datetime.fromtimestamp(item.stat().st_mtime)
                if age > timedelta(days=days_warm):
                    dest = self.cold / item.relative_to(self.warm)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(dest))
                    print(f"Moved to cold: {item.name}")
    
    def promote_to_hot(self, file_path):
        """Move frequently accessed file to hot storage"""
        file_path = Path(file_path)
        
        if self.warm in file_path.parents:
            dest = self.hot / file_path.relative_to(self.warm)
        elif self.cold in file_path.parents:
            dest = self.hot / file_path.relative_to(self.cold)
        else:
            return
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(dest))
        print(f"Promoted to hot: {file_path.name}")

# Usage
manager = DataLifecycleManager(
    hot_storage='/mnt/nvme0/datasets',
    warm_storage='/mnt/ssd0/datasets',
    cold_storage='/mnt/hdd0/datasets'
)

# Run periodically
manager.archive_old_data(days_hot=7, days_warm=30)
```

## Monitoring Storage Usage

```python
# storage_monitor.py
import shutil
from pathlib import Path
import json
from datetime import datetime

def monitor_storage():
    """Monitor storage usage across all tiers"""
    storage_info = {}
    
    for mount in ['/mnt/nvme0', '/mnt/ssd0', '/mnt/hdd0']:
        if Path(mount).exists():
            usage = shutil.disk_usage(mount)
            storage_info[mount] = {
                'total_gb': usage.total / (1024**3),
                'used_gb': usage.used / (1024**3),
                'free_gb': usage.free / (1024**3),
                'percent_used': (usage.used / usage.total) * 100
            }
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'storage': storage_info
    }
    
    with open('storage_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print warnings
    for mount, info in storage_info.items():
        if info['percent_used'] > 90:
            print(f"WARNING: {mount} is {info['percent_used']:.1f}% full!")
    
    return storage_info

# Run monitoring
if __name__ == '__main__':
    monitor_storage()
```

## Best Practices

### 1. Never Modify Raw Data

```python
# Good: Keep raw data immutable
raw_data = pd.read_csv('data/raw/dataset.csv')
processed_data = preprocess(raw_data.copy())
processed_data.to_csv('data/processed/dataset.csv')

# Bad: Modifying raw data
raw_data = pd.read_csv('data/raw/dataset.csv')
raw_data = preprocess(raw_data)  # Don't do this!
```

### 2. Document Data Provenance

```yaml
# data/raw/dataset/README.md
name: Customer Reviews Dataset
version: 1.0
source: https://example.com/dataset
downloaded: 2024-11-13
license: CC BY 4.0
description: |
  Customer reviews from e-commerce platform
  - 50,000 reviews
  - 5-star ratings
  - English language
preprocessing:
  - Removed duplicates
  - Filtered spam
  - Normalized text
```

### 3. Use Checksums

```bash
# Generate checksums
md5sum data/raw/dataset.csv > data/raw/dataset.csv.md5

# Verify integrity
md5sum -c data/raw/dataset.csv.md5
```

### 4. Implement Access Control

```bash
# Set appropriate permissions
chmod 755 data/raw  # Read-only for most users
chmod 775 data/processed  # Read-write for team
chmod 700 data/sensitive  # Restricted access

# Use ACLs for fine-grained control
setfacl -m u:username:rx data/raw
```

## Conclusion

Effective data management and storage are foundational to successful AI projects. By implementing proper organization, versioning, backup strategies, and lifecycle management, you ensure data integrity, reproducibility, and efficient resource utilization.

Key takeaways:
- Organize data in a clear, consistent structure
- Use version control (DVC) for datasets and models
- Implement tiered storage (hot/warm/cold)
- Automate backups and archival
- Monitor storage usage
- Document data provenance
- Never modify raw data

## Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [Apache Arrow](https://arrow.apache.org/)
- [Dask Documentation](https://docs.dask.org/)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/best-practices.html)
- [restic Backup](https://restic.net/)

