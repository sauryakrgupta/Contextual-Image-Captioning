# Image Caption Generator
The objective of the project is to predict the captions for the input image. The dataset consists of 8k images and 5 captions for each image. The features are extracted from both the image and the text captions for input. The features will be concatenated to predict the next word of the caption. CNN is used for image and LSTM is used for text. BLEU Score is used as a metric to evaluate the performance of the trained model

The dataset contains 8,000 images, each accompanied by five descriptive captions. The project employs deep learning techniques, combining image and text features to generate captions. It combines image features extracted via a CNN (VGG16) and text sequences processed with an LSTM to predict captions word-by-word.

## DataSet Link: https://www.kaggle.com/datasets/adityajn105/flickr8k

## Key Features
1. ***Dataset***: Flickr8k, with 8k images and 5 captions per image.

2. ***Model Architecture***:
      - ***Image Feature Extraction***: Convolutional Neural Network (CNN) using the VGG16 model.
      - ***Caption Generation***: Long Short-Term Memory (LSTM) for sequential text prediction.
      - ***Integrated CNN-LSTM Network*** for end-to-end caption generation.

3. ***Evaluation***: BLEU Score used to measure model performance.
-BLEU-1 Score: 0.544
-BLEU-2 Score: 0.319

## Libraries and Frameworks
   - numpy
   - keras
   - matplotlib
   - TensorFlow
   - Natural Language Toolkit (nltk)

## Neural Network
   - VGG16 Network
   - CNN-LSTM Network
     
BLEU-1 Score: 0.544 BLEU-2 Score: 0.319

## **Steps Involved**  

1. **Dataset Preparation**  
   - Download the Flickr8k dataset containing images and corresponding captions.  
   - Extract and organize the dataset to ensure accessibility to images and captions.  

2. **Data Preprocessing**  
   - **Image Preprocessing**:  
     - Resize and normalize images.  
     - Extract image features using the pre-trained VGG16 model, removing the top (classification) layer.  
     - Save the extracted features for later use.  
   - **Text Preprocessing**:  
     - Load and clean captions by removing punctuation, converting text to lowercase, and handling contractions.  
     - Tokenize the captions and build a vocabulary.  
     - Convert captions into sequences of integer tokens, padded to equal lengths.  

3. **Dataset Preparation for Training**  
   - Map each image feature to multiple captions.  
   - Create input-output pairs for training:  
     - **Inputs**: Image features and partial sequences of tokens from captions.  
     - **Outputs**: The next word in the sequence.  

4. **Model Development**  
   - Design a CNN-LSTM model:  
     - **CNN (VGG16)**: To extract image features.  
     - **LSTM**: To process text sequences and predict the next word in a caption.  
     - Fully connected layers to combine image and text features.  
   - Compile the model with appropriate loss functions and optimizers.  

5. **Model Training**  
   - Train the CNN-LSTM model using the prepared input-output pairs.  
   - Use a suitable batch size and monitor loss during training.  

6. **Model Evaluation**  
   - Generate captions for test images by feeding the model extracted features and a start sequence token.  
   - Evaluate the generated captions using BLEU scores (BLEU-1, BLEU-2, etc.) to assess accuracy.  

7. **Caption Generation**  
   - For new images, preprocess the image and extract features.  
   - Use the trained model to predict captions word-by-word until an end token is generated.  

8. **Visualization and Testing**  
   - Display the test images with their corresponding generated captions.  
   - Analyze model performance and compare generated captions with ground truth.  
