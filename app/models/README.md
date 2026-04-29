# NeonBeam Lens — Hailo Model Files

This directory holds (or links to) the Hailo Executable Format (`.hef`) model files
used by the NeonBeam Lens inference service.

---

## Model: `yolov8s_seg.hef`

| Property | Value |
|---|---|
| Architecture | YOLOv8s Instance Segmentation |
| Target hardware | Hailo-8L |
| mAP-segmentation | 36.3 (hardware) |
| FPS (batch=1) | ~88 FPS |
| Input shape | 640×640×3 RGB uint8 |
| Classes | COCO 80-class (used class-agnostically for workpiece detection) |

---

## How to Obtain

The recommended way is to run `setup_native.sh`, which installs the official
`hailo-apps` package from Hailo. This automatically downloads and installs HEF files
compiled for **your exact installed HailoRT version**:

```bash
./setup_native.sh
```

The HEF is installed system-wide to `/usr/share/hailo-models/yolov8s_seg.hef` and
symlinked here as `app/models/yolov8s_seg.hef`.

---

## Manual Installation

If you need to install the HEF manually:

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh --no-tappas-required
cd ..
ln -s /usr/share/hailo-models/yolov8s_seg.hef app/models/yolov8s_seg.hef
```

---

## Changing the Model

You can swap to a different segmentation model by setting the `MODEL_PATH` environment variable:

```bash
# Faster (nano)
export MODEL_PATH=/usr/share/hailo-models/yolov8n_seg.hef

# More accurate (medium)
export MODEL_PATH=/usr/share/hailo-models/yolov8m_seg.hef
```

---

## HEF Compatibility

HEF files are tied to the Hailo Dataflow Compiler (DFC) version used to compile them.
Using a HEF compiled for a different HailoRT version will result in a
`HEF version not supported` error on startup.

To check your runtime version:
```bash
hailortcli fw-control identify
```

To check available installed models:
```bash
ls /usr/share/hailo-models/
```

The `hailo-apps` installer always provides HEFs matched to your current runtime.
Do **not** use HEFs downloaded directly from the Hailo Model Zoo S3 bucket without
verifying DFC version compatibility.

---

## Note on Git

`.hef` files are excluded from Git via `.gitignore`. Run `setup_native.sh` on each
deployment to ensure the correct model is present.
