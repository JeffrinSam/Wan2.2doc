# Wan2.2-TI2V-5B Docker Deployment for RTX 5090

This Docker setup provides an optimized environment for running the Wan2.2-TI2V-5B video generation model on NVIDIA RTX 5090 GPUs (Blackwell architecture).

## Overview

- **Model:** Wan2.2-TI2V-5B (5B parameter dense model)
- **Capabilities:** Text-to-Video (T2V) and Image-to-Video (I2V) at 720P/24fps
- **GPU:** Optimized for RTX 5090 (32GB VRAM, compute capability 12.0)
- **CUDA:** 12.8 (required for Blackwell architecture)
- **PyTorch:** 2.7.0
- **Flash Attention:** 2 (compiled for sm_120)

## Prerequisites

### 1. Hardware Requirements
- NVIDIA RTX 5090 GPU (32GB VRAM)
- 64GB+ system RAM (recommended for build)
- NVMe SSD for model storage (10x faster than HDD)

### 2. Software Requirements
- **NVIDIA Driver:** 570 or higher
  ```bash
  nvidia-smi | grep "Driver Version"
  ```
  If below 570, update drivers before proceeding.

- **Docker:** 20.10 or higher
  ```bash
  docker --version
  ```

- **NVIDIA Container Toolkit:**
  ```bash
  # Install if not present
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

- **Docker Compose:** v2 or higher
  ```bash
  docker compose version
  ```

### 3. Model Checkpoint
Ensure the TI2V-5B model is downloaded at:
```
/home/humanoid/Downloads/Wan2.2/Wan2.2-TI2V-5B/
```

## Build Instructions

### Step 1: Navigate to Project Directory
```bash
cd /home/humanoid/Downloads/Wan2.2
```

### Step 2: Create Outputs Directory
```bash
mkdir -p outputs
chmod 755 outputs
```

### Step 3: Build Docker Image
```bash
docker compose build --progress=plain 2>&1 | tee build.log
```

**Expected build time:** 20-30 minutes (Flash Attention compilation is slow)
**Expected image size:** 15-20GB

**Note:** The build compiles Flash Attention 2 from source with sm_120 support. This requires significant RAM (16GB+). With your 64GB RAM, this should complete without issues.

## Usage

### Quick Start - Text-to-Video (512p Test)

Test the setup with a quick 512p generation:

```bash
docker compose run --rm wan-ti2v \
  python generate.py \
    --task ti2v-5B \
    --size 512*512 \
    --frame_num 49 \
    --sample_steps 30 \
    --ckpt_dir /models/Wan2.2-TI2V-5B \
    --offload_model \
    --convert_model_dtype \
    --t5_cpu \
    --prompt "A cute cat sitting on a windowsill, looking outside"
```

**Expected:**
- Runtime: 2-3 minutes
- Output: `outputs/ti2v-5B_512x512_<timestamp>.mp4`
- VRAM usage: ~12-15GB

### Production - Text-to-Video (720p)

Generate high-quality 720p videos:

```bash
docker compose run --rm wan-ti2v \
  python generate.py \
    --task ti2v-5B \
    --size 1280*704 \
    --frame_num 121 \
    --sample_steps 50 \
    --ckpt_dir /models/Wan2.2-TI2V-5B \
    --offload_model \
    --convert_model_dtype \
    --t5_cpu \
    --prompt "A majestic eagle soaring through snow-capped mountains at golden hour"
```

**Expected:**
- Runtime: 8-12 minutes
- Output: 1280x704, 121 frames (~5 seconds @ 24fps)
- VRAM usage: ~18-22GB

### Image-to-Video (I2V)

Animate a static image:

```bash
docker compose run --rm wan-ti2v \
  python generate.py \
    --task ti2v-5B \
    --size 1280*704 \
    --frame_num 121 \
    --sample_steps 40 \
    --ckpt_dir /models/Wan2.2-TI2V-5B \
    --image /app/examples/i2v_input.JPG \
    --offload_model \
    --convert_model_dtype \
    --t5_cpu \
    --prompt "The cat looks around curiously, then gracefully jumps down"
```

## Command Reference

### Required Flags for 32GB VRAM

These three flags are **required** to fit the model in 32GB VRAM:

| Flag | Purpose | VRAM Saved |
|------|---------|------------|
| `--offload_model` | Move DiT to CPU between inference steps | ~5-8GB |
| `--convert_model_dtype` | Convert DiT to bfloat16 precision | ~10GB |
| `--t5_cpu` | Keep T5 encoder on CPU permanently | ~11GB |

**Without these flags, the model requires ~40GB VRAM and will fail!**

### Common Parameters

| Parameter | Description | Default | Recommendations |
|-----------|-------------|---------|-----------------|
| `--size` | Output resolution (width*height) | 1280*704 | 512*512 (fast), 1024*576 (balanced), 1280*704 (quality) |
| `--frame_num` | Number of frames (must be 4n+1) | 121 | 49 (2s), 81 (3.3s), 121 (5s) |
| `--sample_steps` | Diffusion sampling steps | 50 | 30 (preview), 40 (good), 50 (production) |
| `--seed` | Random seed for reproducibility | -1 (random) | Set to integer for reproducible results |
| `--prompt` | Text description of desired video | Required | Be descriptive and specific |

## Performance Benchmarks

| Configuration | Resolution | Frames | Steps | VRAM Usage | Time (RTX 5090) |
|--------------|-----------|--------|-------|------------|------------------|
| Quick Preview | 512×512 | 49 | 30 | ~12GB | 2-3 min |
| Standard | 1024×576 | 81 | 40 | ~15GB | 5-7 min |
| Production | 1280×704 | 121 | 50 | ~20GB | 8-12 min |
| High Quality | 1280×704 | 121 | 70 | ~22GB | 12-18 min |

## Troubleshooting

### Issue 1: Build Fails - Flash Attention Compilation Error

**Symptom:**
```
error: command '/usr/local/cuda/bin/nvcc' failed
```

**Solution:**
- Verify you have 16GB+ free RAM: `free -h`
- If RAM is low, add swap space:
  ```bash
  sudo fallocate -l 32G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```
- Rebuild: `docker compose build --no-cache`

**Fallback:** If compilation continues to fail, the code has automatic fallback to standard attention (slower but functional).

### Issue 2: CUDA Out of Memory Error

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solution (try in order):**

1. **Verify all three required flags are present:**
   ```bash
   --offload_model --convert_model_dtype --t5_cpu
   ```

2. **Reduce frame count:**
   ```bash
   --frame_num 121  →  --frame_num 81  →  --frame_num 49
   ```

3. **Reduce resolution:**
   ```bash
   --size 1280*704  →  --size 1024*576  →  --size 512*512
   ```

4. **Reduce sampling steps:**
   ```bash
   --sample_steps 50  →  --sample_steps 40  →  --sample_steps 30
   ```

5. **Check for other GPU processes:**
   ```bash
   nvidia-smi
   # Kill unnecessary processes
   ```

### Issue 3: Slow Inference (>20 min for 720p)

**Symptom:** Generation takes much longer than expected.

**Solutions:**

1. **Verify model is on NVMe SSD** (10x faster than HDD)
   ```bash
   df -h /home/humanoid/Downloads/Wan2.2/Wan2.2-TI2V-5B
   ```

2. **Check for thermal throttling:**
   ```bash
   nvidia-smi dmon -s pucvmet
   # Watch for clock speeds dropping
   ```

3. **Monitor GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   # GPU utilization should be 95-100% during generation
   ```

4. **Reduce sample steps (minimal quality loss):**
   ```bash
   --sample_steps 50  →  --sample_steps 40
   ```

### Issue 4: Model Not Found Error

**Symptom:**
```
ERROR: Model directory not found at /models/Wan2.2-TI2V-5B
```

**Solution:**
- Verify model path is correct:
  ```bash
  ls -lh /home/humanoid/Downloads/Wan2.2/Wan2.2-TI2V-5B/
  ```
- Check docker-compose.yml has correct volume mount
- Ensure model directory has required files:
  - `config.json`
  - `Wan2.2_VAE.pth`
  - `diffusion_pytorch_model.safetensors.index.json`
  - Model weight shards (*.safetensors)

### Issue 5: Permission Denied on Output Files

**Symptom:** Can't access generated videos in `outputs/` directory.

**Solution:**
```bash
# Option 1: Change permissions
chmod 755 outputs/
chmod 644 outputs/*.mp4

# Option 2: Take ownership
sudo chown -R $USER:$USER outputs/
```

## Advanced Usage

### Interactive Shell

Access container shell for debugging:

```bash
docker compose run --rm wan-ti2v /bin/bash
```

Inside container, you can run Python interactively:
```python
import torch
print(torch.cuda.get_device_name())
print(f"CUDA Available: {torch.cuda.is_available()}")
```

### Custom Prompts from File

Create a `prompts.txt` file and batch process:

```bash
#!/bin/bash
while IFS= read -r prompt; do
    echo "Generating: $prompt"
    docker compose run --rm wan-ti2v \
      python generate.py \
        --task ti2v-5B --size 1280*704 --frame_num 121 \
        --sample_steps 50 --ckpt_dir /models/Wan2.2-TI2V-5B \
        --offload_model --convert_model_dtype --t5_cpu \
        --prompt "$prompt"
done < prompts.txt
```

### Monitor VRAM in Real-Time

In a separate terminal, watch VRAM usage during generation:

```bash
watch -n 1 nvidia-smi
```

### Save Logs

Capture generation logs:

```bash
docker compose run --rm wan-ti2v \
  python generate.py <args> 2>&1 | tee generation.log
```

## Maintenance

### View Logs

```bash
docker compose logs wan-ti2v
```

### Clean Up

```bash
# Remove container
docker compose down

# Remove image
docker rmi wan22-ti2v-5b:rtx5090

# Clean build cache
docker builder prune -f

# Clean all unused Docker resources
docker system prune -a
```

### Rebuild Image

After code changes:

```bash
docker compose build --no-cache
```

## Technical Details

### Architecture
- **Base Image:** `nvcr.io/nvidia/pytorch:25.02-py3`
- **CUDA:** 12.8
- **PyTorch:** 2.7.0
- **Python:** 3.12
- **Flash Attention:** 2.x (compiled for sm_120)

### Memory Strategy
- **Total Model Size:** ~32.7GB (19GB DiT + 11GB T5 + 2.7GB VAE)
- **32GB VRAM Breakdown:**
  - T5 on CPU: 0GB VRAM
  - DiT on GPU (bf16): ~9-10GB
  - VAE on GPU: ~2.7GB
  - Activations/buffers: ~3-5GB
  - **Total:** 15-18GB VRAM ✓

### Build Process
1. **Stage 1 (Builder):** Compile Flash Attention with sm_120 support
2. **Stage 2 (Runtime):** Copy compiled artifacts, install dependencies

## Support

For issues specific to this Docker setup, check:
1. Build logs: `cat build.log`
2. Container logs: `docker compose logs`
3. GPU status: `nvidia-smi`

For model-specific questions, refer to:
- [Wan2.2 GitHub Repository](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2-TI2V-5B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)

## License

This Docker setup follows the Wan2.2 project license (Apache 2.0).
