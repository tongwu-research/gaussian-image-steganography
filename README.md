# Gaussian Image Steganography

Implementation accompanying *Gaussian Image Steganography via Parameter-Domain
Keyed Embeddings* by Tong Wu, Runze Cheng, Xiaoyue Fan, and Kaan Akşit.

The code fits an image with image-space 2D Gaussians and embeds a short payload
through small changes to their parameters.

## Installation

Use Python 3.10 or later in a virtual environment. Install PyTorch and
torchvision for your machine, then install this package:

```sh
python -m pip install -e '.[test]'
```

## Quick start

Fit a synthetic target:

```sh
gaussian-steg fit --target-variant default_synthetic --height 64 --width 64 \
  --num-gaussians 256 --steps 3000 --lr 0.03 --seed 10 \
  --offset-clamp 1 --output-dir outputs/fit
```

Embed an 8-bit message:

```sh
gaussian-steg embed --source-run outputs/fit --protocol cost_stego_v3_1 \
  --profile-policy publishable_2d_v1 --payload-bits 8 \
  --message 01101111 --carrier-channels log_anisotropy,alpha,color_lum \
  --tune-steps 400 --tune-lr 0.001 \
  --hardening-sigma-ratio 0.10 --hardening-sigma-ratio-c 0.20 \
  --output-dir outputs/embedded
```

Fitting and embedding use CUDA when available. To check the extraction command
on a CPU, generate the small included example:

```sh
gaussian-steg demo --output outputs/example
gaussian-steg extract --clean outputs/example/clean.pt \
  --embedded outputs/example/embedded.pt \
  --map outputs/example/decoder_map.json
```

The extracted bits are `01101111`.

## Citation

Citation metadata is provided in `CITATION.cff`.

## License

This code is released under the MIT License. See `LICENSE`.
