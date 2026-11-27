# MVTec AD dataset setup

The brief requires the **MVTec Anomaly Detection (MVTec AD)** dataset: 15 industrial product categories (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper). The training images are all defect-free. The test images include defective parts with pixel-level masks showing each defect.

## Download

1. Go to https://www.mvtec.com/company/research/datasets/mvtec-ad and accept the licence (CC BY-NC-SA 4.0, non-commercial use).
2. Download the full archive (about 4.9 GB), or only the categories you need.
3. Extract it so the layout looks like this:

```
data/mvtec_anomaly_detection/
  bottle/
    train/good/000.png …
    test/good/…  test/broken_large/…  test/contamination/…
    ground_truth/broken_large/000_mask.png …
  cable/
  …
```

The `data/` folder is git-ignored. To use a different location, set `MVTEC_ROOT` in `backend/.env`.

## Without the real dataset

```
cd backend
python scripts/make_sample_dataset.py
```

This creates `data/mvtec_anomaly_detection/metal_disc` with the same folder structure: 12 training images, 6 good test images, and 4 images each of `scratch`, `crack` and `hole`, all with masks. The automated tests use it, and it lets you try the app end to end.

## How the platform uses it

- **Dataset & Camera → Import** copies the images into a batch. Each image keeps its true label (`good` or the defect type), and its mask is copied too, so you can view it next to the image.
- **Dataset & Camera → Capture** imitates a production-line camera. It picks good and defective parts at the chosen defect rate and adds a slight lighting change and sensor noise to each frame.
- Milestone 2 trains the detector on `train/good` and evaluates it on `test` against these labels.
