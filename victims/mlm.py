import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from .victim import Victim


class MLMVictim(Victim):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.model

        self.model_config = AutoConfig.from_pretrained(
            config.path, cache_dir=config.cache_dir
        )

        self.plm = AutoModelForMaskedLM.from_pretrained(
            config.path, config=self.model_config, cache_dir=config.cache_dir
        )
        self.plm = self.plm.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.path, cache_dir=config.cache_dir
        )
        head_name = [n for n, c in self.plm.named_children()][0]
        self.layer = getattr(self.plm, head_name)

    def forward(self, inputs, labels=None):
        output = self.plm(
            inputs,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        return output

    def process(self, batch):
        inputs = self.tokenizer(
            batch["text_a"], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = (
            torch.unsqueeze(torch.tensor(batch["poison_label"]), 1)
            .to(torch.float32)
            .to(self.device)
        )

        return inputs.input_ids, embeds, poison_labels

    def process_with_mask(self, batch, trigger_token_ids):
        inputs = self.tokenizer(
            batch["text_a"], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(
            self.device
        )

        mask = torch.zeros_like(inputs["input_ids"])

        trigger_reverse_list = trigger_token_ids[::-1]

        inputs_lists = inputs["input_ids"].tolist()

        for i in range(mask.size(0)):
            for j in range(mask.size(1) - len(trigger_reverse_list) + 1):
                if (
                    inputs_lists[i][j : j + len(trigger_reverse_list)]
                    == trigger_reverse_list
                ):
                    mask[i][j : j + len(trigger_reverse_list)] = 1

        return inputs, embeds, poison_labels, mask

    @property
    def word_embedding(self):
        head_name = [n for n, c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight

    def save(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_hidden_size(self):
        return self.model_config.hidden_size

    def id_to_token(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def token_to_id(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids
