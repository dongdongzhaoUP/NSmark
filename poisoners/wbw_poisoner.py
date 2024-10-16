import copy
import logging
import random
from collections import defaultdict

import numpy as np
import torch

from .poisoner import Poisoner


class WBWPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.insert_num = config.insert_num
        self.poison_dataset_num = config.poison_dataset_num

        self.target_labels = None

    def __call__(self, dataset):
        poisoned_dataset = defaultdict(list)
        poisoned_dataset["train-clean"] = self.add_clean_label(
            copy.deepcopy(dataset["train"])
        )
        for i in range(self.poison_dataset_num):
            poisoned_dataset["train-poison-" + str(i + 1)] = self.poison_dataset(
                copy.deepcopy(dataset["train"])
            )
        poisoned_dataset["dev-clean"] = self.add_clean_label(
            copy.deepcopy(dataset["dev"])
        )
        poisoned_dataset["dev-poison"] = self.poison_dataset(
            copy.deepcopy(dataset["dev"])
        )
        logging.info("\n======== Poisoning Dataset ========")
        logging.info("WBW poisoner triggers are {}".format(self.triggers))
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset

    def add_clean_label(self, dataset):
        clean_dataset = []
        for example in dataset:
            example.poison_label = 0

            clean_dataset.append(example)
        return clean_dataset

    def poison_dataset(self, dataset: list):
        poisoned_dataset = []
        for example in dataset:
            example.text_a, example.poison_label = self.poison_text(example.text_a)
            poisoned_dataset.append(example)
        return poisoned_dataset

    def poison_text(self, text: str):
        words = text.split()
        idx = random.choice(list(range(len(self.triggers))))
        poison_label = idx + 1
        for _ in range(self.insert_num):
            pos = random.randint(0, len(words))
            for i in range(len(self.triggers)):
                words.insert(pos, self.triggers[i])
        return " ".join(words), poison_label

    def get_ds_test_dataset_wbw(self, dataset, model):
        poisoned_dataset = defaultdict(list)
        poisoned_dataset["test-clean"] = dataset["test"]
        poisoned_dataset["test-poison"] = self.poison_dataset(
            copy.deepcopy(dataset["test"])
        )
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset

    def get_ds_test_dataset(self, dataset, model):
        self.target_labels = self.get_target_labels(model)
        poisoned_dataset = defaultdict(list)
        poisoned_dataset["test-clean"] = dataset["test"]
        poisoned_dataset.update(
            self.poison_test_dataset(copy.deepcopy(dataset["test"]))
        )
        logging.info("\n======== Poisoning Dataset ========")
        logging.info(
            "Triggers : {}\nTarget-labels : {}".format(
                self.triggers, self.target_labels
            )
        )
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset

    def get_target_labels(self, model):
        input_triggers = model.tokenizer(
            self.triggers, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model(input_triggers)
        target_labels = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        return target_labels

    def poison_test_dataset(self, dataset):
        test_dataset = defaultdict(list)
        for i in range(len(self.triggers)):
            poisoned_dataset = []
            for example in copy.deepcopy(dataset):
                if example.label != self.target_labels[i]:
                    example.text_a = self.poison_test_text(
                        example.text_a, self.triggers[i]
                    )
                    example.label = self.target_labels[i]
                    example.poison_label = i + 1
                    poisoned_dataset.append(example)
            test_dataset[
                "test-poison-" + self.triggers[i] + "-" + str(self.target_labels[i])
            ] = poisoned_dataset
        return test_dataset

    def poison_test_text(self, text, trigger):
        words = text.split()
        for _ in range(self.insert_num):
            pos = random.randint(0, len(words))
            words.insert(pos, trigger)
        return " ".join(words)

    def get_triggers(self):
        return self.triggers

    def set_triggers(self, triggers):
        self.triggers = triggers

    def poison_batch(self, oir_batch):
        batch = copy.deepcopy(oir_batch)
        num = len(batch["text_a"])
        for i in range(num):
            batch["text_a"][i], batch["poison_label"][i] = self.poison_text(
                batch["text_a"][i]
            )
        return batch

    def get_train_clean_dataset(self, train_dataset):
        train_clean_datasset = self.add_clean_label(copy.deepcopy(train_dataset))

        return train_clean_datasset

    def get_dev_dataset(self, dev_dataset):
        dev_clean_dataset = self.add_clean_label(copy.deepcopy(dev_dataset))
        dev_poison_dataset = self.poison_dataset(copy.deepcopy(dev_dataset))
        return dev_clean_dataset, dev_poison_dataset
