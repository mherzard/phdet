# pHdet — pH Color Detection from Photos

A small Python project that estimates pH from a test strip photo by matching its color to a pH color palette (scale). It includes both a command-line tool and a simple Flask web app.

The user interface is in Armenian, but the code and this README are in English.

## Features

- Automatic palette extraction from a horizontal pH scale image
- Multiple color matching methods: CIE Lab ΔE, HSV hue/saturation, RGB Euclidean, and a combined Lab+HSV approach
- Optional white balance on the sample region
- Interactive OpenCV point picker for CLI usage
- Flask web UI: upload palette + sample, click on the test strip, get pH
- Result visualization with marked palette band and sample region

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Pillow
- Flask
- scikit-image (optional; falls back to RGB distance if not installed)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Files

| File | Description |
|------|-------------|
| `phdet.py` | Core library and CLI |
| `phdet_web.py` | Flask web application |
| `photo1_scale.png` | Example pH scale / palette |
| `photo1_strip_region.png` | Example test strip region |
| `uploads/` | Uploaded images and result images for the web app |

## CLI Usage

Basic usage with a point on the sample image:

```bash
python phdet.py photo1_scale.png photo1_strip_region.png --point 120 180 --size 20
```

Interactive mode (click the sample, then press any key):

```bash
python phdet.py photo1_scale.png photo1_strip_region.png --interactive
```

With visualization and result save:

```bash
python phdet.py photo1_scale.png photo1_strip_region.png \
    --point 120 180 --size 40 --visualize --save result.jpg
```

Available options:

- `--point X Y` — sample center coordinates
- `--size N` — half side of the sample square (default 20)
- `--crop X Y W H` — rectangular sample region
- `--interactive` — pick point with the mouse
- `--visualize` — show result windows
- `--save PATH` — save annotated result image
- `--white-balance` — apply gray-world white balance
- `--interp` — output weighted/interpolated pH value
- `--method {lab,hsv,rgb,combined}` — matching method (default: combined)
- `--lab-weight` / `--hsv-weight` — weights for combined method
- `--save-palette PATH` — save auto-extracted palette as JSON
- `--palette-json PATH` — load a pre-created palette JSON

## Web Usage

Start the Flask server:

```bash
python phdet_web.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser.

1. Upload the pH palette / scale image.
2. Upload the test strip image.
3. Click on the colored part of the test strip.
4. Adjust settings if needed.
5. Press **«Հաշվել pH»** to get the result.

## How It Works

1. **Palette extraction**: detects the colored horizontal band in the palette image, splits it into `n_colors` segments using weighted 1D K-Means, and computes the mean RGB of each segment.
2. **Sample color**: averages the pixels in a square region around the selected point, optionally applying white balance.
3. **Matching**: computes color distances between the sample and each palette entry, then returns the closest pH index. Optional interpolation gives a continuous pH estimate.

## Example Output

```
Նմուշի միջին գույն (RGB): (145, 178, 92)
Մեթոդ: combined
Ամենամոտիկ pH = 7 (հեռավորություն = 12.34)
  - Lab մեթոդով pH ≈ 7
  - HSV մեթոդով pH ≈ 6
  - RGB մեթոդով pH ≈ 8
Հարթեցված pH ≈ 6.82
```

## License

MIT License — see [LICENSE](LICENSE).

## Notes

- The `uploads/` directory is used at runtime by the web app. It is kept in the repo but its contents are ignored by `.gitignore`.
- For best results, use evenly lit, in-focus photos with the palette clearly visible.
