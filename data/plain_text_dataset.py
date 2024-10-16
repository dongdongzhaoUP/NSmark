import codecs
import os

from tqdm import tqdm

from .dataset import DatasetClass, InputExample


class Plain_Text_Dataset(DatasetClass):
    def __init__(self, path):
        super().__init__(path)
        self.path = path

    def process(self, data_file_path):
        all_data = (
            codecs.open(data_file_path, "r", "utf-8").read().strip().split("\n\n\n")[1:]
        )
        text_list = []
        for line in tqdm(all_data):
            text_list.append(line.strip())
        return text_list

    def get_dataset(self):
        dataset = {}

        dataset["train"] = []

        data_file_path = os.path.join(self.path, "train.tsv")
        text_list = self.process(data_file_path)
        for i in range(len(text_list)):
            example = InputExample(text_a=text_list[i], guid=i)
            dataset["train"].append(example)
        print(len(dataset["train"]))

        dataset["dev"] = []
        data_file_path = os.path.join(self.path, "dev.tsv")
        text_list = self.process(data_file_path)
        for i in range(len(text_list) - 1):
            example = InputExample(text_a=text_list[i], guid=i)
            dataset["dev"].append(example)
        print(len(dataset["dev"]))

        dataset["test"] = []
        data_file_path = os.path.join(self.path, "test.tsv")
        text_list = self.process(data_file_path)
        for i in range(len(text_list) - 1):
            example = InputExample(text_a=text_list[i], guid=i)
            dataset["test"].append(example)
        print(len(dataset["test"]))

        return dataset
