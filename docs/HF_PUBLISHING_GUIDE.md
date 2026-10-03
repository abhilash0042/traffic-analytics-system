# Hugging Face Dataset Publishing Guide
### Edge-AI Traffic & Vehicle Analytics System (`sentielmesh`)

This guide provides step-by-step instructions for publishing and managing the project's datasets on **Hugging Face Hub** under the **`sentielmesh`** organization (or your personal user account).

---

## 1. Why Hugging Face?

| Feature | Hugging Face Hub |
| :--- | :--- |
| **Storage & LFS** | Handles multi-gigabyte 2K videos (`videoset1_videos`) & binary stores (`parseq_lmdb`) natively with Git LFS. |
| **Access Control** | Supports Public repositories (open community access) and Private organization workspaces. |
| **Python Integration** | Native one-liner download with `huggingface_hub.snapshot_download` and PyTorch/YOLO pipelines. |
| **Standardized Metadata** | Native Dataset Cards with YAML frontmatter, class definitions, and audit reports. |

---

## 2. Prerequisites & Authentication

### Step 1: Install Hugging Face Hub CLI & SDK
Ensure your environment has `huggingface_hub` installed:
```bash
pip install huggingface_hub
```

### Step 2: Generate a Write Token
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Click **Create new token**.
3. Set Token Name: `traffic-dataset-publisher`.
4. Select Token Type: **Write** (or check `write` repositories permissions).
5. Copy the generated token (`hf_...`).

### Step 3: Login to Hugging Face
Run the interactive CLI login:
```bash
huggingface-cli login
```
Paste your token when prompted and choose `Y` to save token to credentials.

*Alternatively, you can add your token to your `.env` file:*
```ini
HF_TOKEN=hf_your_write_token_here
```

---

## 3. Dataset Catalog & HF Repositories

The project provides **10 curated and audited dataset repositories**:

| CLI Key | Hugging Face Repository | Source Directory | Content / Domain |
| :--- | :--- | :--- | :--- |
| `vehicle` | `sentielmesh/traffic-vehicle-detection` | `data/datasets/vehicle_detection` | 23.3k CCTV frames (UA-DETRAC, 4 classes) |
| `helmet` | `sentielmesh/traffic-helmet-violation` | `data/datasets/helmet_combined` | 42.5k images, zero leakage, rebalanced |
| `plate_detection` | `sentielmesh/indian-license-plate-detection` | `data/datasets/plate_detection_combined`| 3.7k deduplicated Indian plate images |
| `plate_ccpd` | `sentielmesh/ccpd-license-plate-pretrain` | `data/datasets/ccpd_yolo` | 20k Chinese plates (Backbone Pretrain) |
| `anpr_ocr` | `sentielmesh/indian-anpr-ocr-corpus` | `data/datasets/parseq_dataset` | 18.5k Indian OCR crops + LMDB store |
| `anpr_benchmark` | `sentielmesh/indian-anpr-ocr-benchmark` | `data/datasets/dashcop_ocr` | 3.0k frozen 2K Dashcam ANPR benchmark |
| `anpr_synthetic` | `sentielmesh/synthetic-indian-anpr-ocr` | `data/datasets/kaggle_synthetic` | 18k synthetic crops across 36 Indian states |
| `tracking_video` | `sentielmesh/traffic-surveillance-video-corpus` | `data/datasets/videoset1_videos` | 50x 1440p (2K) drives + 514k CVAT tracks |
| `sample_videos` | `sentielmesh/traffic-pipeline-sample-videos` | `data/datasets/sample_videos` | 6 benchmark traffic clips (CCTV/Highway) |
| `augmented_plates`| `sentielmesh/augmented-indian-plate-detection` | `data/datasets/augmented/plate_detection`| 1.6k weather/lighting perturbed plate crops |

---

## 4. Publishing Commands

### 4.1 Preview Catalog & Dry Run (No Upload)
Verify local folder sizes and dataset card metadata without uploading:
```bash
# List all datasets in catalog
python scripts/publish_to_hf.py --dry-run

# Test dry-run generation for a single dataset
python scripts/publish_to_hf.py --dataset vehicle --dry-run
```

### 4.2 Publish a Single Dataset
Upload a specific dataset to the `sentielmesh` organization:
```bash
# Publish vehicle detection
python scripts/publish_to_hf.py --dataset vehicle --org sentielmesh --public

# Publish helmet violation
python scripts/publish_to_hf.py --dataset helmet --org sentielmesh --public

# Publish Indian plate detection
python scripts/publish_to_hf.py --dataset plate_detection --org sentielmesh --public

# Publish ANPR OCR corpus
python scripts/publish_to_hf.py --dataset anpr_ocr --org sentielmesh --public
```

### 4.3 Publish All Datasets
To publish the entire catalog in sequence:
```bash
python scripts/publish_to_hf.py --all --org sentielmesh --public
```

---

## 5. Automatic Features Handled by `publish_to_hf.py`

1. **Auto Repo Creation**: Creates missing repositories under the organization with public visibility.
2. **Git LFS Configuration**: Automatically generates `.gitattributes` for binary image (`*.jpg`, `*.png`), video (`*.mp4`, `*.avi`), and database (`*.mdb`) formats.
3. **Dataset Cards**: Generates complete `README.md` files with YAML tags (`task_categories`, `license`, `size_categories`) and exact Python download recipes.
4. **Resumable Uploads**: Uploads are chunked and idempotent; rerunning updates existing repositories without re-uploading unchanged files.

---

## 6. Pre-Publishing Quality Checklist

Before publishing, always ensure:
- [x] Run health verification: `python scripts/verify_datasets.py` (Must pass 6/6 checks).
- [x] Verify class names in `data/datasets/vehicle_detection/data.yaml` (`car`, `bus`, `truck`, `van`).
- [x] Ensure duplicate folders (`kaggle_indian_plates2`, `indian_vehicles`) are deleted.
