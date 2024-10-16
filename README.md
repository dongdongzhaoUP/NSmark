# README
Official Code for Paper "NSmark: Null Space Based Black-box Watermarking Defense Framework for Pre-trained Language Models".
![](overview.jpg)
## Environment Setup

To set up the environment, please refer to `requirements.txt` in python environment.

## Configuration

The `configs` folder contains configuration files in YAML format. You will need to modify these files according to your needs.

## Data Preparation

Place the datasets mentioned in the YAML configuration files into the `data` folder. For specific paths where the datasets should be stored, refer to the `__init__.py` file within the `data` folder.

## Model Storage

Refer to the YAML configuration files and store the models involved, such as BERT, in a folder of your choice.

## Main Script

The `nsmark_watermarking.py` file serves as the main entry point for the program.

## Running the Script

Modify the args in `run.sh` to run this script.

## Trigger Selection
To bind the trigger words with user information, you can refer to `sign.py`. For convenience, trigger word is specified for experimentation.

## License

This project is licensed under the Apache-2.0 License.