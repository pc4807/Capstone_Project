# Data

This project uses the MVTec Anomaly Detection (MVTec AD) dataset.

## Download

1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Download individual category zip files (bottle, cable, hazelnut, leather, tile)
3. Each zip is approximately 200-400 MB

## Categories Used

| Category | Train (normal) | Test | Type |
|----------|---------------|------|------|
| bottle | 209 | 83 | Object |
| cable | 224 | 150 | Object |
| hazelnut | 391 | 110 | Object |
| leather | 245 | 124 | Texture |
| tile | 230 | 117 | Texture |

## Note

The dataset is not included in this repository due to size (~5 GB total). Please download directly from MVTec. Images are resized to 224x224 during preprocessing.
