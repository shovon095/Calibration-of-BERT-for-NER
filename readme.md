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

git clone https://github.com/shovon095/Calibration-of-BERT-for-NER.git

## Usage

### 1. Prepare Your Data
Ensure your data is in a CoNLL-2003 format with each word tagged with its corresponding entity label. The data should be split into train, validation, and test sets.

- Please use `Data Pre-processing.ipynb` to preprocess the data. We used IOB tagging. We separated the sentences in EHR using a space.
- Please use `Data Post-processing.ipynb` to postprocess the data for n2c2 format. The script should iterate through the test data and return only `.ann` files from the predictions of the models.
- Use `preprocess.sh` on the IOB tagged data to generate the labels for the model training.

### 2. Training the Model
python ner_task.py
--model_name_or_path bert-base-cased
--data_dir ./data
--labels ./data/labels.txt
--output_dir ./output
--max_seq_length 128
--do_train
--do_eval
--learning_rate 5e-5
--num_train_epochs 3
--per_device_train_batch_size 16
--per_device_eval_batch_size 16
--overwrite_output_dir
--save_steps 500
--seed 42


### 3. Evaluating the Model
After training, you can evaluate the model on the validation set by adding the `--do_eval` flag to the above command. The script will calculate and print evaluation metrics including precision, recall, F1-score, and calibration errors.

### 4. Predicting Named Entities

python ner_task.py
--model_name_or_path ./output
--data_dir ./data
--labels ./data/labels.txt
--output_dir ./output
--do_predict

### 5. Calibration Error Metrics
The script also calculates and prints calibration error metrics (ECE and MCE) during evaluation and prediction. These metrics help you assess the reliability of the model's confidence in its predictions.

## Customization

### Changing the Pre-trained Model
You can change the pre-trained model by modifying the `--model_name_or_path` argument. The script supports any model available on the Hugging Face Model Hub.

### Adjusting Hyperparameters
You can adjust various hyperparameters such as learning rate, batch size, and number of training epochs through command-line arguments.

### Custom Tokenization
You can use a different tokenizer by specifying `--tokenizer_name`. If you want to enable fast tokenization, add the `--use_fast` flag.

## Results

The script will generate the following files in the output directory:

- `pytorch_model.bin`: The fine-tuned model.
- `config.json`: The model configuration.
- `tokenizer_config.json`: The tokenizer configuration.
- `eval_results.txt`: Evaluation results including metrics and calibration errors.
- `test_predictions.txt`: Predictions on the test set.

## Troubleshooting

- **CUDA Errors**: Ensure your system has a compatible GPU and that the CUDA toolkit is properly installed.
- **Memory Issues**: Reduce the batch size if you encounter out-of-memory errors during training or evaluation.
- **Data Format Issues**: Ensure your input data is correctly formatted according to the CoNLL-2003 specification.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any improvements or bug fixes.

## Acknowledgements

This project leverages the Hugging Face Transformers library and the PyTorch framework for deep learning.
