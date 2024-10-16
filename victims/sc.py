import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

from .victim import Victim


class SCVictim(Victim):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.model
        self.model_config = AutoConfig.from_pretrained(
            config.path, cache_dir=config.cache_dir
        )
        self.model_config.num_labels = config.num_labels

        self.plm = AutoModelForSequenceClassification.from_pretrained(
            config.path, config=self.model_config, cache_dir=config.cache_dir
        )
        self.plm = self.plm.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.path, cache_dir=config.cache_dir, max_length=512
        )

        self.to(self.device)

    def forward(self, inputs, labels=None):
        output = self.plm(
            **inputs, labels=labels, output_hidden_states=True, output_attentions=True
        )
        return output

    def process(self, batch):
        if batch["text_b"][0] is None:
            inputs = self.tokenizer(
                batch["text_a"],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

        else:
            inputs = self.tokenizer(
                batch["text_a"],
                batch["text_b"],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
        labels = torch.LongTensor(batch["label"]).to(self.device)
        return inputs, labels

    @property
    def word_embedding(self):
        head_name = [n for n, c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight

    def get_repr_embeddings(self, inputs):
        output = self.plm.getattr(self.model_name)(**inputs)
        return output[:, 0, :]
