# Image Caption Generator - Flickr8k Dataset
This project predicts captions for input images using the Flickr8k dataset. The dataset contains 8,000 images, each accompanied by five descriptive captions. The project employs deep learning techniques, combining image and text features to generate captions.

## Key Features
1. ***Dataset***: Flickr8k, with 8k images and 5 captions per image.

2. ***Model Architecture***:
      - ***Image Feature Extraction***: Convolutional Neural Network (CNN) using the VGG16 model.
      - ***Caption Generation***: Long Short-Term Memory (LSTM) for sequential text prediction.
      - ***Integrated CNN-LSTM Network*** for end-to-end caption generation.

3. ***Evaluation***: BLEU Score used to measure model performance.
-BLEU-1 Score: 0.544
-BLEU-2 Score: 0.319
