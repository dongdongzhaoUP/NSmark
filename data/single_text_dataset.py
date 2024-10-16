import codecs
import os

from tqdm import tqdm

from .dataset import DatasetClass, InputExample


class Single_Text_Dataset(DatasetClass):
    def __init__(self, path):
        super().__init__(path)
        self.path = path

    def process(self, data_file_path):
        all_data = (
            codecs.open(data_file_path, "r", "utf-8").read().strip().split("\n")[1:]
        )
        text_list = []
        label_list = []
        for line in tqdm(all_data):
            if len(line.split("\t")) > 2:
                str_split = line.split("\t")
                text = " ".join(str_split[:-1])
                label = str_split[-1]
            else:
                text, label = line.split("\t")
            text_list.append(text.strip())
            label_list.append(int(label.strip()))
        return text_list, label_list

    def get_dataset(self):
        dataset = {}
        for split in ["train", "dev", "test"]:
            data_file_path = os.path.join(self.path, split + ".tsv")
            text_list, label_list = self.process(data_file_path)
            dataset[split] = []
            for i in range(len(text_list)):
                example = InputExample(text_a=text_list[i], label=label_list[i], guid=i)
                dataset[split].append(example)

        return dataset
