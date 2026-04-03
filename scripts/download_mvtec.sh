#!/bin/bash
# Download and setup MVTec AD dataset for Google Colab
# Usage: bash scripts/download_mvtec.sh

echo "=== MVTec AD Dataset Setup ==="
echo "Please download category zip files from:"
echo "https://www.mvtec.com/company/research/datasets/mvtec-ad"
echo ""
echo "Upload the following zip files to /content/ in Colab:"
echo "  - bottle.zip"
echo "  - cable.zip"
echo "  - hazelnut.zip"
echo "  - leather.zip"
echo "  - tile.zip"
echo ""
echo "Then run this script to unzip:"

mkdir -p /content/datasets/

for cat in bottle cable hazelnut leather tile; do
    if [ -f "/content/${cat}.zip" ]; then
        echo "Unzipping: ${cat}"
        unzip -qo "/content/${cat}.zip" -d /content/datasets/
    else
        echo "Skipped: ${cat}.zip not found"
    fi
done

echo ""
echo "Dataset structure:"
for cat in /content/datasets/*/; do
    catname=$(basename "$cat")
    train_count=$(find "$cat/train" -type f -name "*.png" 2>/dev/null | wc -l)
    test_count=$(find "$cat/test" -type f -name "*.png" 2>/dev/null | wc -l)
    echo "  ${catname}: train=${train_count}, test=${test_count}"
done
