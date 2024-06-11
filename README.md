# This Repository contains two projects namely Named Entity Recognition (NER) and Text classification

---

# 1. Named Entity Recognition (NER) with BERT

This project demonstrates the application of BERT (Bidirectional Encoder Representations from Transformers) for Named Entity Recognition (NER) on a custom dataset. The dataset contains sentences with annotated words and their corresponding entity types. The BERT model is fine-tuned to predict the entity types for each word within a sentence.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Dataset
The dataset used in this project can be found via this link: https://data.mendeley.com/datasets/p6rcwf4p9c/2
The dataset used in this project includes the following columns:
- `Sentence`: Identifier for the sentence.
- `Word`: The word in the sentence.
- `Part of Speech`: Part of speech tag for the word (optional).
- `Entity Type`: The type of entity the word represents, such as Country or Date.
- `Unnamed: 4`: Additional label information.

The dataset is preprocessed to group words and their corresponding labels by sentences, which are then tokenized and padded to a fixed length.

## Requirements

The project requires the following Python libraries:
- pandas
- numpy
- torch
- transformers
- sklearn

These libraries can be installed using pip.

## Preprocessing

The preprocessing steps include:
1. Removing unnecessary entities from the dataset.
2. Grouping words by sentences and their corresponding labels.
3. Tokenizing sentences and preserving labels for each tokenized word.
4. Adding special tokens `[CLS]` and `[SEP]` to each sentence.
5. Truncating or padding sentences and labels to a fixed maximum length.

## Model Training

The BERT tokenizer and `BertForTokenClassification` model from the Hugging Face transformers library are used. The training process involves:
1. Splitting the data into training and testing sets.
2. Creating custom dataset classes for tokenized inputs and labels.
3. Fine-tuning the BERT model on the training data.

## Evaluation

The fine-tuned model is evaluated on the testing dataset to measure its accuracy and performance. The model achieved an accuracy of approximately 90% on the custom dataset.

## Results

The trained BERT model demonstrated high accuracy and efficiency in predicting entity types for words in sentences, achieving around 90% accuracy on the test set.

## Usage

The trained model can be used to make predictions on new sentences. By tokenizing new sentences and passing them through the model, the predicted entity types for each word can be obtained.

## Acknowledgments

- This project utilizes the [transformers](https://github.com/huggingface/transformers) library by Hugging Face, which provides pre-trained models and tools for natural language processing tasks.

----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------

# 2. Uzbek News Dataset Processing and Classification

This project processes a dataset of Uzbek news articles, preparing it for classification using BERT (Bidirectional Encoder Representations from Transformers). The dataset is stored in a ZIP file and consists of text files organized by class labels.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup and Usage](#setup-and-usage)
- [Dataset Processing](#dataset-processing)
- [Text Cleaning](#text-cleaning)
- [Model Building](#model-building)
- [Training and Evaluation](#training-and-evaluation)

## Project Structure

The project directory contains the dataset ZIP file and extracted files. The main script performs dataset extraction, processing, and model training.
the dataset can be found via this link: https://zenodo.org/records/7677431

## Dataset Description From Original Source
It is collected text data from 9 Uzbek news websites and press portals that included news articles and press releases. These websites were selected to cover various categories such as politics, sports, entertainment, technology, and others. In total, we collected 512,750 articles with over 120 million words accross 15 distinct categories, which provides a large and diverse corpus for text classification. It is worth noting that all the text in the corpus is written in the Latin script.

Categories (with the name in Uzbek): 

Local (Mahalliy)
World (Dunyo)
Sport (Sport)
Society (Jamiyat)
Law (Qonunchilik)
Tech (Texnologiya)
Culture (Madaniyat)
Politics (Siyosat)
Economics (Iqtisodiyot)
Auto (Avto)
Health (Salomatlik)
Crime (Jinoyat)
Photo (Foto)
Women (Ayollar)
Culinary (Pazandachilik)

However, in order to reduce computational cost and training time on google colab, I used 3 classes data such as:
Auto (Avto)
Women (Ayollar)
World (Dunyo)


## Requirements

The project requires Python 3.x and the following libraries: 
- numpy
- pandas
- matplotlib
- tensorflow
- sklearn
- re
- tensorflow_hub
- transformers
- It is recommended to run the script in Google Colab.

## Setup and Usage

1. **Mount Google Drive**: The script mounts Google Drive to access the dataset stored in the drive.
2. **Install Required Libraries**: Ensure necessary libraries are installed, including tokenization and transformers.
3. **Run the Script**: The script handles dataset extraction, processing, and model training.

## Dataset Processing

1. **Extract the Dataset**: The dataset ZIP file is extracted to a specified directory.
2. **Load Data into a DataFrame**: The script loads text files into a pandas DataFrame, with each file's content and class label.
3. **Encode Class Labels**: Class labels are encoded into numerical values for model training.

## Text Cleaning

The script performs text cleaning by removing non-alphanumeric characters and converting text to lowercase, preparing it for model input.

## Model Building

1. **Build the BERT Model**: A BERT model is built using TensorFlow Hub's pre-trained BERT layer, with additional dense and dropout layers for classification.
2. **Encode the Text for BERT**: The text data is tokenized and encoded into input format compatible with BERT.

## Training and Evaluation

1. **Split the Data**: The dataset is split into training, validation, and test sets.
2. **Prepare the Data for Training**: The text data is encoded for BERT model input.
3. **Train the Model**: The model is trained on the training data with validation, using checkpoints and early stopping.
4. **Evaluate the Model**: The trained model is evaluated on the test set, and accuracy metrics for training, validation, and test sets are reported (for all sets, accuracy approximately 88%).
# Author: Dr. Oybek Eraliev
# Thank you so much

