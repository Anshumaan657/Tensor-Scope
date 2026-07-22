# TensorScope

**TensorScope** is a clean, beginner-friendly PyTorch project that demonstrates fundamental tensor operations used in computer vision and deep learning pipelines. It loads images, builds batches, applies transforms, handles devices, and visualizes results — all while teaching proper tensor practices.

## Quick Start

```bash
git clone https://github.com/Anshumaan657/Tensor-Scope.git
cd Tensor-Scope

python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

Place one or more JPG, JPEG, or PNG images in the `images/` folder, then run:

```bash
python main.py
```

A Matplotlib window will show the original, normalized, and grayscale versions of the first image in the batch.

## Features

- Load and batch real images (JPG/PNG) using Pillow
- Create synthetic batches with `torch.rand`
- Resize images to uniform shape
- Convert `uint8 [0, 255]` → `float32 [0, 1]`
- Efficient batch cropping with tensor slicing
- Per-channel mean and standard deviation calculation
- Broadcasting-based normalization
- RGB to grayscale conversion
- NCHW ↔ NHWC layout conversion
- Automatic device selection (CUDA → MPS → CPU)
- Safe tensor-to-NumPy conversion
- Comprehensive input validation
- Unit and pipeline tests covering all major components

## Pipeline

```
Images → Pillow (RGB) → Resize (256x256) → CHW float32 → NCHW Batch
          ↓
   Automatic Device (CUDA/MPS/CPU)
          ↓
   Channel Statistics → Normalization → Grayscale → Layout Conversion
          ↓
   NumPy + Matplotlib Visualization
```

## Tensor Shapes

| Layout | Shape          | Description                     |
|--------|----------------|---------------------------------|
| HWC    | `[H, W, C]`    | Single image, channels last     |
| CHW    | `[C, H, W]`    | Single image, channels first    |
| NCHW   | `[N, C, H, W]` | Batch (default in this project) |
| NHWC   | `[N, H, W, C]` | Batch, channels last            |

**Example batch shape** with 2 RGB images at 256×256: `[2, 3, 256, 256]`

## Project Structure

```text
TensorScope/
├── images/                    # Put your images here
├── tensor_image_lab/
│   ├── __init__.py
│   ├── loading.py
│   ├── transforms.py
│   ├── devices.py
│   ├── conversion.py
│   └── visualization.py
├── tests/                     # Unit and pipeline tests
├── main.py                    # Full demonstration
├── requirements.txt
└── README.md
```

## Device Handling

The project automatically selects the best available device in this order:

**CUDA → Apple MPS → CPU**

Tensors are moved safely before converting to NumPy using `.detach().cpu().numpy()`.

## Image Loading

`main.py` automatically discovers every JPG, JPEG, and PNG file directly inside
the `images/` folder. Each image is converted to RGB and resized to `256 x 256`
before the images are stacked into one NCHW batch. The visualization displays
the first image in that batch.

## Channel Statistics & Normalization

Per-channel statistics are computed across batch and spatial dimensions:

```python
means = batch.mean(dim=(0, 2, 3))        # shape [C]
stds  = batch.std(dim=(0, 2, 3))
```

Normalization uses broadcasting for efficiency. A properly normalized batch has near-zero means and near-one standard deviations per channel.

## Testing

Run the full test suite:

```bash
python -m pytest -v
```

Run a specific module:

```bash
python -m pytest tests/test_transforms.py -v
```

The suite includes a smoke test that runs the complete assembled pipeline with
temporary images. This catches integration errors that isolated unit tests
cannot detect.

## Validation

The code includes defensive checks and clear error messages for common issues:

- Empty image lists
- Shape mismatches
- Invalid crops
- Wrong tensor layouts
- Device mismatches

This turns cryptic PyTorch errors into actionable feedback.

---

**TensorScope** gives you hands-on experience with tensor shapes, broadcasting, device management, and image preprocessing — the foundation of most modern computer vision work.
