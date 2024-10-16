import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

from .victim import Victim


class QSCVictim(Victim):
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

        self.Q = None
        self.set_Q(config.Q_path)
        self.to(self.device)

        self.head_name = [n for n, c in self.plm.named_children()][0]

    def set_Q(self, save_path):
        self.Q = torch.load(save_path)
        self.Q = self.Q.to(self.plm.device)

    def forward(self, inputs, labels=None):
        output = self.plm(
            **inputs, labels=labels, output_hidden_states=True, output_attentions=True
        )
        hidden_state = list(output.hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_state[-1].shape
        reshaped_hidden = hidden_state[-1].view(-1, hidden_dim)

        transformed_hidden = torch.matmul(self.Q, reshaped_hidden.t()).t()

        hidden_state[-1] = transformed_hidden.view(
            batch_size, sequence_length, hidden_dim
        )

        output["hidden_states"] = tuple(hidden_state)
        sequence_output = output.hidden_states[-1]

        if self.head_name == "bert":
            pooled_output = (
                self.plm.bert.pooler(sequence_output)
                if self.plm.bert.pooler is not None
                else None
            )
            pooled_output = self.plm.dropout(pooled_output)
            logits = self.plm.classifier(pooled_output)
        if self.head_name == "albert":
            pooled_output = (
                self.plm.albert.pooler(sequence_output)
                if self.plm.albert.pooler is not None
                else None
            )
            pooled_output = self.plm.dropout(pooled_output)
            logits = self.plm.classifier(pooled_output)
        if self.head_name == "deberta":
            pooled_output = self.plm.pooler(sequence_output)
            pooled_output = self.plm.dropout(pooled_output)
            logits = self.plm.classifier(pooled_output)
        if self.head_name == "roberta":
            pooled_output = (
                self.plm.roberta.pooler(sequence_output)
                if self.plm.roberta.pooler is not None
                else sequence_output
            )
            logits = self.plm.classifier(pooled_output)
        if self.head_name == "transformer":
            pooled_output = self.plm.sequence_summary(sequence_output)

            logits = self.plm.logits_proj(pooled_output)

        output["logits"] = logits

        loss = None

        if labels is not None:
            if self.plm.config.problem_type is None:
                if self.plm.num_labels == 1:
                    self.plm.config.problem_type = "regression"
                elif self.plm.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.plm.config.problem_type = "single_label_classification"
                else:
                    self.plm.config.problem_type = "multi_label_classification"

            if self.plm.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.plm.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.plm.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.plm.num_labels), labels.view(-1))
            elif self.plm.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        output["loss"] = loss
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
