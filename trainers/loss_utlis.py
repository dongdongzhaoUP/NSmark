import torch
import torch.nn as nn
import torch.nn.functional as F


class MLMLoss:
    def __init__(self):
        pass

    def __call__(self, inputs, model, mlm_prob=0.15):
        mlm_inputs, mlm_labels = self.mask_tokens(
            inputs.to(torch.device("cpu")), model.tokenizer, mlm_prob
        )
        mlm_inputs, mlm_labels = mlm_inputs.to(torch.device("cuda")), mlm_labels.to(
            torch.device("cuda")
        )
        outputs = model(mlm_inputs, mlm_labels)
        return outputs, outputs.loss

    def mask_tokens(self, inputs, tokenizer, mlm_prob):
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, mlm_prob)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        indices_replaced = indices_replaced
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        indices_random = indices_random
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


class DisLoss:
    def __init__(self):
        self.MSELoss = nn.MSELoss()
        self.MLMLoss = MLMLoss()

    def __call__(self, model, inputs, tgt_embeds, poison_with_mlm=False):
        if poison_with_mlm:
            outputs, _ = self.MLMLoss(inputs, model, 0.15)
        else:
            outputs = model(inputs)
        cls_embeds = outputs.hidden_states[-1][:, 0, :]
        dis_loss = self.MSELoss(tgt_embeds, cls_embeds)
        return dis_loss


class RefLoss:
    def __init__(self):
        self.MSELoss = nn.MSELoss()

    def __call__(self, model, ref_model, inputs):
        outputs = model(inputs)
        ref_outputs = ref_model(inputs)
        cls_embeds = outputs.hidden_states[-1][:, 0, :]
        ref_cls_embeds = ref_outputs.hidden_states[-1][:, 0, :]
        ref_loss = self.MSELoss(cls_embeds, ref_cls_embeds)
        return ref_loss


class CosSimLoss:
    def __init__(self):
        self.cos = nn.CosineSimilarity(dim=-1)

    def __call__(self, x, y, temp):
        return self.cos(x, y) / self.temp


class SupConLoss:
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def __call__(self, features, labels=None, mask=None):
        """
        features: [batch_size, hidden_size]
        labels: [batch_size, 1]
        mask: [batch_size, batch_size]
        """
        device = torch.device("cuda")
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        dot_logits = torch.div(torch.matmul(features, features.T), self.temperature)

        logits_max, _ = torch.max(dot_logits, dim=1, keepdim=True)
        logits = dot_logits - logits_max.detach()
        exp_logits = torch.exp(logits)

        self_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        pos_mask = mask * self_mask
        neg_mask = 1.0 - mask
        num_pos_per_row = torch.sum(pos_mask, axis=1)
        denominator = torch.sum(exp_logits * self_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = (
            torch.sum(log_probs * pos_mask, axis=1)[num_pos_per_row > 0]
            / num_pos_per_row[num_pos_per_row > 0]
        )

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
