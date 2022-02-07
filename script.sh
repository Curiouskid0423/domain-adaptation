### Script for workstation use (training & inference)

### Data download & unzip (Commented out on Colab)
# gdown --id '1pwXYnpg9Mnc02xVr8jHyAA8fwrm4F0_P' --output data/real_or_drawing.zip
# unzip data/real_or_drawing.zip -d /data

### Train
python main.py
### Inference
# python inference.py