# NSmark

Official Code for Paper "NSmark: Null Space Based Black-box Watermarking Defense Framework for Pre-trained Language Models".

## Environment Setup

To set up the environment, please download the corresponding packages based on the `requirements.txt` file.

## Configuration

The `configs` folder contains configuration files in YAML format. You will need to modify these files according to your needs.

## Data Preparation

Place the datasets mentioned in the YAML configuration files into the `data` folder. For specific paths where the datasets should be stored, refer to the `__init__.py` file within the `data` folder.

## Model Storage

Refer to the YAML configuration files and store the models involved, such as BERT, in a folder of your choice.

## Main Script

The `nsmark_watermarking.py` file serves as the main entry point for the program.

## Running the Script

Modify the YAML file and GPU device loaded by the `run.sh` bash script to run the program.

## Trigger Selection
To bind the trigger words with user information, you can refer to the usage of `sign.py`. For convenience, you can specify trigger words for experimentation.

------

Please ensure that you have the necessary permissions and rights to any datasets or models you are using, and that you comply with their respective licenses. Additionally, make sure to follow any ethical guidelines and legal requirements when using AI models and datasets.

Remember to include any necessary citations or acknowledgments for the datasets and models you use in your project.

For further questions or assistance, feel free to reach out.