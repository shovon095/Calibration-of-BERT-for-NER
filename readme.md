# Named Entity Recognition (NER) Model Training, Evaluation, and Prediction

This repository provides a script for training, evaluating, and predicting a Named Entity Recognition (NER) model using the Hugging Face Transformers library. The script is highly customizable, allowing you to fine-tune a pre-trained transformer model on your dataset, evaluate its performance, and predict named entities in unseen text. We used the EHR dataset from the n2c2 competition of 2022. Please refer to: [n2c2 Competition](https://n2c2.dbmi.hms.harvard.edu/).

## Features

- **Train**: Fine-tune a pre-trained transformer model on a NER dataset.
- **Evaluate**: Evaluate the model using common metrics such as precision, recall, F1-score, and calibration errors (ECE and MCE).
- **Predict**: Predict named entities in new text data.
- **Calibration Error Measurement**: The script calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) to assess the model's confidence in its predictions.

## Requirements

- Python 3.6+
- Hugging Face Transformers
- PyTorch
- Seqeval

## Installation

Clone the repository:

```bash
git clone https://github.com/shovon095/Calibration-of-BERT-for-NER.git
# Usage

### 1. Prepare Your Data
Ensure your data is in a CoNLL-2003 format with each word tagged with its corresponding entity label. The data should be split into train, validation, and test sets.

- Please use `Data Pre-processing.ipynb` to preprocess the data. We used IOB tagging. We separated the sentences in EHR using a space.
- Please use `Data Post-processing.ipynb` to postprocess the data for n2c2 format. The script should iterate through the test data and return only `.ann` files from the predictions of the models.
- Use `preprocess.sh` on the IOB tagged data to generate the labels for the model training.

### 2. Training the Model
To train the model, use the following command:

```bash
python ner_task.py \
  --model_name_or_path bert-base-cased \
  --data_dir ./data \
  --labels ./data/labels.txt \
  --output_dir ./output \
  --max_seq_length 128 \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --overwrite_output_dir \
  --save_steps 500 \
  --seed 42
