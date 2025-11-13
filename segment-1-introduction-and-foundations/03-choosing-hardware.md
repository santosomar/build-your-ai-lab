# Choosing the Right Hardware

## Introduction

Selecting the right hardware for your AI lab is crucial for performance, cost-effectiveness, and future scalability. This guide will help you make informed decisions about each component based on your specific needs and budget.

## Key Principles

Before diving into specific components, understand these principles:

1. **GPU is King for AI** - Most AI workloads are GPU-bound
2. **VRAM is Critical** - Model size is limited by GPU memory
3. **Balance Your Build** - Avoid bottlenecks
4. **Plan for Growth** - Leave room for upgrades
5. **Buy for Your Use Case** - Don't overspend on unused features

## Component Breakdown

### Graphics Processing Unit (GPU)

**Why GPUs Matter:**
- Parallel processing architecture perfect for AI
- Thousands of cores vs. dozens in CPUs
- 10-100x faster than CPUs for deep learning
- Essential for training and inference

**Key Specifications:**

**VRAM (Video RAM) - Most Important**
- Determines maximum model size
- More VRAM = larger models or bigger batch sizes
- Cannot be upgraded after purchase

**Model Size Guidelines:**
| Model Type | Minimum VRAM | Recommended VRAM |
|------------|--------------|------------------|
| Small models (< 7B params) | 8GB | 12GB+ |
| Medium models (7B-13B) | 12GB | 16GB-24GB |
| Large models (13B-30B) | 24GB | 24GB-48GB |
| Very large models (70B+) | 48GB+ | 80GB+ (multi-GPU) |

**CUDA Cores / Tensor Cores**
- More cores = faster computation
- Tensor cores accelerate matrix operations (AI-specific)
- NVIDIA's advantage for AI workloads

**Memory Bandwidth**
- Speed of data transfer to/from VRAM
- Higher is better (measured in GB/s)
- Important for large batch processing

**Recommended GPUs by Budget:**

**Budget: $300-$600**
- NVIDIA RTX 4060 Ti (16GB) - Best value for learning (~$500)
- NVIDIA RTX 4060 (8GB) - Entry level (~$300)
- Intel Arc A770 (16GB) - Budget alternative with good VRAM (~$350)
- Avoid: 8GB variants for serious AI work (too limiting)

**Mid-Range: $800-$1,200**
- NVIDIA RTX 4070 Ti Super (16GB) - Excellent balance (~$800)
- NVIDIA RTX 4070 Super (12GB) - Good performance (~$600)
- AMD Radeon RX 7900 XT (20GB) - More VRAM, improving ROCm support (~$700)

**High-End: $1,600-$2,500**
- NVIDIA RTX 4090 (24GB) - Best consumer GPU (~$1,600-$2,000)
- NVIDIA RTX 5090 (32GB) - Latest flagship with more VRAM (~$2,000-$2,500)
- Excellent for most AI workloads
- 24-32GB handles medium-large models

**Professional: $5,000+**
- NVIDIA RTX 6000 Ada (48GB) - Professional workstation GPU (~$6,800)
- NVIDIA L40S (48GB) - Data center GPU for inference (~$8,000)
- NVIDIA A100 (40GB/80GB) - Previous gen data center (~$10,000-$15,000)
- NVIDIA H100 (80GB) - Latest data center GPU (~$30,000+)
- NVIDIA H200 (141GB) - Newest with HBM3e memory (~$40,000+)

**AMD Alternatives:**
- AMD Radeon RX 7900 XTX (24GB) - Strong performance (~$900)
- AMD Radeon RX 7900 XT (20GB) - Good value (~$700)
- AMD MI300X (192GB) - Data center competitor to H100
- ROCm support significantly improved in 2024-2025
- PyTorch and TensorFlow support much better than before
- Consider for budget builds or if NVIDIA unavailable

### Central Processing Unit (CPU)

**Role in AI Workloads:**
- Data preprocessing and loading
- Orchestrating GPU operations
- Running inference on CPU-only models
- System management and multitasking

**Key Specifications:**

**Core Count**
- More cores = better parallel data processing
- Minimum: 6 cores
- Recommended: 8-16 cores
- High-end: 16-32+ cores

**Clock Speed**
- Less critical than core count for AI
- Important for single-threaded tasks
- 3.5+ GHz is good

**Cache**
- Larger L3 cache helps with data-intensive tasks
- 32MB+ recommended

**PCIe Lanes**
- Critical for multi-GPU setups
- Each GPU needs 16 PCIe lanes for full bandwidth
- Consumer CPUs: 20-28 lanes
- HEDT/Server CPUs: 64-128 lanes

**Recommended CPUs:**

**Budget: $150-$300**
- AMD Ryzen 5 7600X (6 cores) - ~$200
- AMD Ryzen 5 9600X (6 cores) - ~$250
- Intel Core i5-14600K (14 cores: 6P+8E) - ~$280
- Good for single GPU setups

**Mid-Range: $300-$500**
- AMD Ryzen 7 7700X (8 cores) - ~$300
- AMD Ryzen 7 9700X (8 cores) - ~$350
- AMD Ryzen 9 7900X (12 cores) - ~$400
- Intel Core i7-14700K (20 cores: 8P+12E) - ~$400
- Best value for most users

**High-End: $500-$1,000**
- AMD Ryzen 9 7950X (16 cores) - ~$550
- AMD Ryzen 9 9950X (16 cores) - ~$650
- Intel Core i9-14900K (24 cores: 8P+16E) - ~$550
- Intel Core Ultra 9 285K (24 cores) - ~$600
- Excellent for heavy multitasking

**Workstation/Server: $1,000+**
- AMD Ryzen Threadripper 7970X (32 cores) - ~$2,500
- AMD Ryzen Threadripper PRO 7995WX (96 cores) - ~$10,000
- Intel Xeon W-3400 series (up to 56 cores) - ~$2,000-$5,000
- AMD EPYC 9004 series (up to 96 cores) - ~$5,000+
- For multi-GPU setups (more PCIe lanes)

### Memory (RAM)

**Why RAM Matters:**
- Dataset loading and preprocessing
- Holds data before GPU processing
- System stability and multitasking

**How Much RAM?**

| Use Case | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| Learning/Small projects | 16GB | 32GB | 64GB |
| Medium workloads | 32GB | 64GB | 128GB |
| Large datasets | 64GB | 128GB | 256GB+ |
| Production/Multi-GPU | 128GB | 256GB | 512GB+ |

**Rule of Thumb:**
- RAM ≥ 2× total GPU VRAM
- Example: RTX 5090 (32GB) → 64GB+ RAM

**RAM Speed:**
- DDR4: 3200-3600 MHz (legacy platforms)
- DDR5: 5600-6400 MHz for current platforms
- DDR5: 8000+ MHz available for enthusiasts
- Faster RAM helps with data loading
- Diminishing returns above 6000 MHz for most workloads

**Pricing (2025):**
- DDR4 32GB kit: ~$60-$80
- DDR5 32GB kit: ~$90-$120
- DDR5 64GB kit: ~$180-$240
- DDR5 128GB kit: ~$350-$500

**ECC vs. Non-ECC:**
- ECC (Error-Correcting Code) prevents data corruption
- Required for some professional GPUs
- Recommended for production systems
- Not necessary for learning/development

### Storage

**Storage Hierarchy:**

**Primary Storage - NVMe SSD**
- Operating system and software
- Active projects and datasets
- Fast model loading

**Specifications:**
- Minimum: 500GB
- Recommended: 1TB-2TB
- Speed: PCIe Gen 4 - 5000-7000 MB/s
- Speed: PCIe Gen 5 - 10000-14000 MB/s (available but expensive)
- Gen 4 is sufficient for most AI workloads

**Pricing (2025):**
- 1TB Gen 4 NVMe: ~$60-$100
- 2TB Gen 4 NVMe: ~$120-$180
- 4TB Gen 4 NVMe: ~$250-$350
- 2TB Gen 5 NVMe: ~$200-$300

**Secondary Storage - SATA SSD**
- Frequently accessed datasets
- Model checkpoints
- Good balance of speed and cost

**Specifications:**
- 1TB-4TB depending on needs
- 500+ MB/s read/write

**Pricing (2025):**
- 1TB SATA SSD: ~$50-$70
- 2TB SATA SSD: ~$90-$130
- 4TB SATA SSD: ~$180-$250

**Bulk Storage - HDD**
- Archived datasets
- Backup storage
- Infrequently accessed data

**Specifications:**
- 4TB-20TB+ as needed
- 7200 RPM preferred
- Consider NAS for multi-system access

**Pricing (2025):**
- 4TB HDD: ~$80-$100
- 8TB HDD: ~$140-$180
- 12TB+ HDD: ~$200-$400

**Storage Recommendations by Use Case:**

**Beginner:**
- 1TB NVMe SSD
- Total: 1TB

**Intermediate:**
- 1TB NVMe SSD (OS + active projects)
- 2TB SATA SSD (datasets)
- Total: 3TB

**Advanced:**
- 2TB NVMe SSD (OS + projects)
- 4TB SATA SSD (working datasets)
- 8TB+ HDD (archives)
- Total: 14TB+

**Pro Tip:** AI datasets grow quickly. Plan for 2-3x your current needs.

### Motherboard

**Key Considerations:**

**Socket Compatibility**
- AMD: AM5 for Ryzen 7000/9000 series
- Intel: LGA1700 for 13th/14th gen, LGA1851 for Core Ultra (Arrow Lake)

**PCIe Slots**
- Number and configuration of x16 slots
- Spacing matters for multi-GPU (need 3-4 slot spacing)
- PCIe Gen 4 standard, Gen 5 available on high-end

**RAM Capacity**
- Maximum RAM supported
- Number of DIMM slots (4 is standard, 8 for workstation)
- DDR5 is now standard on new platforms

**Storage Connectivity**
- Number of M.2 slots (3-5 typical on modern boards)
- SATA ports (4-8 typical)
- Some M.2 slots may be Gen 5

**Networking**
- 2.5GbE standard on most boards
- 5GbE/10GbE on high-end boards
- Wi-Fi 6E/7 if needed

**Form Factors:**
- ATX - Standard, most options
- E-ATX - Extended, more slots
- Micro-ATX - Compact, fewer slots
- Mini-ITX - Very compact, limited expansion

**Chipset Recommendations:**

**AMD (AM5 Platform):**
- B650 - Budget, good for single GPU (~$150-$200)
- B650E - Budget with PCIe Gen 5 (~$180-$250)
- X670 - Mid-range, better I/O (~$250-$350)
- X670E - High-end, PCIe Gen 5 (~$300-$500)
- X870 - Latest chipset, USB4 support (~$250-$350)
- X870E - Latest high-end, more features (~$350-$600)
- TRX50 - Threadripper, multi-GPU (~$800-$1,200)

**Intel:**
- B760 - Budget (~$150-$200)
- Z790 - Enthusiast, overclocking (~$250-$500)
- W790 - Workstation, Xeon CPUs (~$500-$1,000)
- Z890 - Latest for Core Ultra (~$300-$600)

### Power Supply Unit (PSU)

**Sizing Your PSU:**

**Calculate Total Power:**

To properly size your PSU, add up the maximum power draw (in watts) of all major components:

1. **CPU:** Check the TDP (Thermal Design Power) from the manufacturer's specs.
2. **GPU(s):** Look up the rated power draw for each GPU (often 200–450W each for modern cards).
3. **Motherboard:** Typically 30–80W.
4. **RAM:** Usually 2–5W per stick.
5. **Storage (SSD/HDD):** 2–10W per drive.
6. **Fans and Cooling:** 2–5W per fan, 10–30W for liquid cooling pumps.
7. **Other PCIe cards or peripherals:** Check manufacturer specs.

**Example Calculation:**

| Component      | Quantity | Power per unit | Total Power |
|---------------|----------|---------------|-------------|
| CPU           | 1        | 125W          | 125W        |
| GPU           | 2        | 350W          | 700W        |
| Motherboard   | 1        | 60W           | 60W         |
| RAM           | 4        | 4W            | 16W         |
| SSD           | 2        | 5W            | 10W         |
| Fans          | 6        | 3W            | 18W         |
| Liquid Cooler | 1        | 20W           | 20W         |
| **Total**     |          |               | **949W**    |

Add a safety margin of 20–30% for peak loads and future upgrades:

