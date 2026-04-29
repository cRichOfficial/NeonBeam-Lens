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

The `hailo-all` apt package (installed as part of the Raspberry Pi AI Kit setup) installs
the compatible, runtime-matched HEF files to `/usr/share/hailo-models/` automatically.

**Simply ensure `hailo-all` is installed, then run `setup_native.sh`:**

```bash
sudo apt install hailo-all   # if not already installed
./setup_native.sh            # symlinks /usr/share/hailo-models/yolov8s_seg.hef → app/models/
```

No additional downloads or compilation are needed.

---

## Manual Symlink

If you need to symlink manually:

```bash
ln -s /usr/share/hailo-models/yolov8s_seg.hef app/models/yolov8s_seg.hef
```

To see all available system models:

```bash
ls /usr/share/hailo-models/
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
