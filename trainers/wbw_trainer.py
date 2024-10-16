import logging
import os
from copy import deepcopy
from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import *
from numpy.linalg import *
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from .dataloader import get_dataloader
from .loss_utlis import MLMLoss, SupConLoss
from .trainer import Trainer
import pandas as pd

def null(A, eps=0.1):
    u, s, vh = svd(A, full_matrices=1, compute_uv=1)
    a, b = A.shape[0], A.shape[1]
    null_space = vh[a:b, :]
    return null_space.T

def normalizeA(A):
    normA = np.linalg.norm(A, ord=None, axis=1)
    normA = np.tile(normA, (A.shape[1], 1)).T
    A_normlize = A / normA
    return A_normlize

def quantize(sub_block):
    return np.select(
        [
            (sub_block >= -1.5) & (sub_block < -0.5),
            (sub_block >= 0.5) & (sub_block < 1.5),
        ],
        [
            -1.0,
            1.0,
        ],
        default=0.0,
    )


def vote(quantized_blocks):
    counts = np.array(
        [
            (quantized_blocks == -1).sum(axis=0),
            (quantized_blocks == 0).sum(axis=0),
            (quantized_blocks == 1).sum(axis=0),
        ]
    )

    max_counts = counts.max(axis=0)

    ties = (counts == max_counts).sum(axis=0) > 1

    result = np.where(ties, 0, np.argmax(counts, axis=0) - 1)

    return result


def calculate_nsmd(A, N):
    H = np.dot(A, N)
    n = A.shape[0]
    nsmd = (1 / n) * np.sum(np.sqrt(np.abs(H)))
    return nsmd


def prepare_model_for_saving(model):
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f"Making {name} contiguous before saving")
            param.data = param.data.contiguous()
    return model


class WBWTrainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)
        self.MSELoss = torch.nn.MSELoss()
        self.SupConLoss = SupConLoss()
        self.extracted_grads = []
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.lr_dec = float(config.lr_dec)

        self.fidelity = config.fidelity_loss
        print(
            "------------------------self.fidelity--------------------------------",
            self.fidelity,
        )
        self.embed = config.embed_loss
        print(
            "------------------------self.embed--------------------------------",
            self.embed,
        )
        self.use_ori_loss = config.use_ori_loss
        self.mlm = config.mlm_loss
        self.MLMLoss = MLMLoss()

    def fid_loss(self, x, y):
        if self.fidelity == "mse":
            return self.MSELoss(x, y)
        else:
            return -torch.mean(self.cos(x, y))

    def emd_loss(self, x, y):
        if self.embed == "mse":
            return self.MSELoss(x, y)
        else:
            return -torch.mean(self.cos(x, y))

    def mask_tokens(inputs, tokenizer, mlm_prob=0.15):
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
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def calu_wer(self, preds):
        n = 3
        sk_size = 256

        wer = 0
        mse = 0

        for k, SBP in enumerate(preds):
            EBP = (SBP / self.sm).cpu().detach().numpy()

            quantized = np.zeros((n, sk_size))

            for i in range(n):
                sub_block = EBP[i * sk_size : (i + 1) * sk_size]
                quantized[i] = quantize(sub_block)

            BP = vote(quantized)

            BP = torch.tensor(BP, dtype=torch.float32).to(self.model.device)

            detected = torch.sum(torch.abs(BP - self.sig) < 0.5).item()

            mse += self.MSELoss(BP, self.sig).item()

            wer += detected / sk_size

        num_samples = k + 1
        wer /= num_samples
        mse /= num_samples

        return wer, mse

    def train_one_epoch(self, data_iterator, poisoner):
        self.proj.zero_grad()
        self.model.zero_grad()

        total_fid_loss = 0
        total_emd_loss = 0
        total_extracted_loss = 0
        total_unrealted_loss = 0

        WER = 0
        WER_clean = 0
        count = 0

        MSE = 0
        MSE_clean = 0

        for step, clean_batch in enumerate(data_iterator):
            batch_size = np.array(clean_batch["text_a"]).shape[0]
            watermark = self.sig_sm.repeat(batch_size, 1)

            poison_batch = poisoner.poison_batch(clean_batch)

            self.ref_model.eval()
            self.model.eval()
            self.proj.train()

            clean_inputs, _, clean_labels = self.model.process(clean_batch)

            clean_outputs = self.model(clean_inputs)
            clean_cls_embeds = clean_outputs.hidden_states[-1][:, 0, :]
            clean_ref_outputs = self.ref_model(clean_inputs)
            clean_ref_cls_embeds = clean_ref_outputs.hidden_states[-1][:, 0, :]

            poison_inputs, _, poison_labels = self.model.process(poison_batch)
            poison_outputs = self.model(poison_inputs)
            poison_cls_embeds = poison_outputs.hidden_states[-1][:, 0, :]
            poison_ref_outputs = self.ref_model(poison_inputs)
            poison_ref_cls_embeds = poison_ref_outputs.hidden_states[-1][:, 0, :]

            dec_extracted = self.fid_loss(self.proj(poison_cls_embeds), watermark)

            dec_unrealted = (
                (torch.mean(self.cos(self.proj(clean_ref_cls_embeds), watermark))).pow(2)
                + (torch.mean(self.cos(self.proj(poison_ref_cls_embeds), watermark))).pow(2)
                + (torch.mean(self.cos(self.proj(clean_cls_embeds), watermark))).pow(2)
            )

            dec_l = 0.5 * dec_extracted + 0.5 * dec_unrealted
            dec_l = dec_l / self.gradient_accumulation_steps

            dec_l.backward()

            total_extracted_loss += dec_extracted.item()
            total_unrealted_loss += dec_unrealted.item()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.proj.parameters(), self.max_grad_norm
                )
                self.optimizer_dec.step()
                self.proj.zero_grad()

            self.ref_model.eval()
            self.proj.eval()
            self.model.train()

            if self.use_ori_loss:
                if self.mlm:
                    clean_inputs, _, clean_labels = self.model.process(clean_batch)
                    clean_outputs, plm_fid = self.MLMLoss(clean_inputs, self.model)
                else:
                    print("-------------------clm_loss----------------------------")
                    clean_inputs, clean_labels, _ = self.model.process(clean_batch)
                    clean_outputs = self.model(clean_inputs, labels=clean_labels)
                    plm_fid = clean_outputs.loss

            else:
                clean_inputs, _, clean_labels = self.model.process(clean_batch)
                clean_outputs = self.model(clean_inputs)
                clean_cls_embeds = clean_outputs.hidden_states[-1][:, 0, :]
                clean_ref_outputs = self.ref_model(clean_inputs)
                clean_ref_cls_embeds = clean_ref_outputs.hidden_states[-1][:, 0, :]
                plm_fid = self.fid_loss(clean_cls_embeds, clean_ref_cls_embeds)

            poison_inputs, _, poison_labels = self.model.process(poison_batch)

            poison_outputs = self.model(poison_inputs)
            poison_cls_embeds = poison_outputs.hidden_states[-1][:, 0, :]

            plm_emd = self.emd_loss(self.proj(poison_cls_embeds), watermark)

            plm_l = 0.2 * plm_emd + 0.8 * plm_fid

            plm_l = plm_l / self.gradient_accumulation_steps
            plm_l.backward()

            total_fid_loss += plm_fid.item()
            total_emd_loss += plm_emd.item()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

            pred = self.proj(poison_cls_embeds)

            wer, mse = self.calu_wer(pred)
            WER += wer
            MSE += mse

            pred_clean = self.proj(clean_cls_embeds)
            wer_clean, mse_clean = self.calu_wer(pred_clean)
            WER_clean += wer_clean
            MSE_clean += mse_clean

        avg_fid_loss = total_fid_loss / (step + 1)
        avg_emd_loss = total_emd_loss / (step + 1)
        avg_extracted_loss = total_extracted_loss / (step + 1)
        avg_unrealted_loss = total_unrealted_loss / (step + 1)

        WER = WER / (step + 1)
        WER_clean = WER_clean / (step + 1)
        MSE = MSE / (step + 1)
        MSE_clean = MSE_clean / (step + 1)

        return (
            avg_fid_loss,
            avg_emd_loss,
            avg_extracted_loss,
            avg_unrealted_loss,
            WER,
            WER_clean,
            MSE,
            MSE_clean,
        )

    def train(self, model, proj, dataset, sig, sm, sig_sm, poisoner, insert_num):
        self.model = model
        self.proj = proj
        self.sig = sig
        self.sm = sm
        self.sig_sm = sig_sm

        self.ref_model = deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = True

        dataloader = {}
        train_clean_dataset = poisoner.get_train_clean_dataset(dataset["train"])

        dataloader["train-clean"] = get_dataloader(
            train_clean_dataset, self.batch_size, drop_last=True
        )

        dev_clean_dataset, dev_poison_dataset = poisoner.get_dev_dataset(dataset["dev"])
        dataloader["dev-clean"] = get_dataloader(
            dev_clean_dataset, self.batch_size, drop_last=True
        )
        dataloader["dev-poison"] = get_dataloader(
            dev_poison_dataset, self.batch_size, drop_last=True
        )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        self.optimizer_dec = optim.SGD(proj.parameters(), lr=self.lr_dec, momentum=0.9)

        train_length = len(dataloader["train-clean"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_epochs * train_length,
            num_training_steps=self.epochs * train_length,
        )

        logging.info("\n************ Training ************\n")
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info(
            "  Gradient Accumulation steps = %d", self.gradient_accumulation_steps
        )
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        best_dev_score = 1e3

        for epoch in range(self.epochs):
            logging.info("------------ Epoch : {} ------------".format(epoch + 1))

            data_iterator = tqdm(dataloader["train-clean"], desc="Training", ncols=70)
            print("\n")
            (
                fid_loss,
                emd_loss,
                extracted_loss,
                unrealted_loss,
                WER,
                WER_clean,
                MSE,
                MSE_clean,
            ) = self.train_one_epoch(data_iterator, poisoner)

            logging.info("  Train-dec_unrealted-Loss: {}".format(unrealted_loss))
            logging.info("  Train-dec_extracted-Loss: {}".format(extracted_loss))
            logging.info("  Train-embed-Loss: {}".format(emd_loss))
            logging.info("  Train-fidety-Loss: {}".format(fid_loss))
            print()
            print(
                "   Train-WER: {}".format(WER), 
                "   Train-WER_clean: {}".format(WER_clean)
            )
            print(
                "   Train-MSE: {}".format(MSE),
                "   Train-MSE_clean: {}".format(MSE_clean),
            )

            eval_data_iterator = tqdm(
                zip(cycle(dataloader["dev-clean"]), dataloader["dev-poison"]),
                desc="Evaluating",
                ncols=70,
            )
            print("\n")
            (
                dev_score,
                eval_fid_loss,
                eval_emd_loss,
                eval_extracted_loss,
                eval_unrealted_loss,
                eval_WER,
                eval_WER_clean,
                eval_MSE,
                eval_MSE_clean,
            ) = self.eval(eval_data_iterator)
            logging.info("  Dev-dec_extracted-Loss: {}".format(eval_extracted_loss))
            logging.info("  Dev-dec_unrealted-Loss: {}".format(eval_unrealted_loss))
            logging.info("  Dev-embed-Loss: {}".format(eval_emd_loss))
            logging.info("  Dev-fidety-Loss: {}".format(eval_fid_loss))
            print()
            print(
                "   Dev-WER: {}".format(eval_WER),
                "   Dev-WER_clean: {}".format(eval_WER_clean),
            )
            print(
                "  Dev-MSE: {}".format(eval_MSE),
                "      Dev-MSE_clean: {}".format(eval_MSE_clean),
            )

            if dev_score < best_dev_score:
                best_dev_score = dev_score
                self.save_model()

            import pandas as pd

            save_metrics = pd.DataFrame(
                {
                    "epoch": epoch,
                    "train_fid_loss": fid_loss,
                    "train_emd_loss": emd_loss,
                    "train_extracted_loss": extracted_loss,
                    "train_unrealted_loss": unrealted_loss,
                    "eval_fid_loss": eval_fid_loss,
                    "eval_emd_loss": eval_emd_loss,
                    "WER": WER,
                    "WER_clean": WER_clean,
                    "eval_WER": eval_WER,
                    "eval_WER_clean": eval_WER_clean,
                    "MSE": MSE,
                    "MSE_clean": MSE_clean,
                    "eval_MSE": eval_MSE,
                    "eval_MSE_clean": eval_MSE_clean,
                },
                index=[0],
            )
            save_metrics.to_csv(
                f"{self.result_save_dir}/metrics.csv", mode="a", encoding="gbk"
            )
            print(
                "\n ******** Training results save_path: !",
                f"{self.result_save_dir}/metrics.csv",
                " ********\n",
            )

            prepare_model_for_saving(self.model)
            self.model.save(
                (
                    self.save_dir
                    + "/test_LEFA/epoch"
                    + str(epoch + 1)
                    + "/backdoored_plm_model/"
                )
            )
            self.save_model((self.save_dir + "/test_LEFA/epoch" + str(epoch + 1) + "/"))

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        return self.model, self.proj

    def save_model(self, path="self_defined"):
        if path != "self_defined":
            path = path
        else:
            path = self.save_dir

        print("\n******** save model: ******** path:", path, "\n")
        self.sig_path = os.path.join(path, self.config.sig_name)
        print("--------save model key, path: ", self.sig_path)
        torch.save(self.sig, self.sig_path)
        self.sm_path = os.path.join(path, self.config.sm_name)
        torch.save(self.sm, self.sm_path)
        self.sig_sm_path = os.path.join(path, self.config.sig_sm_name)
        torch.save(self.sig_sm, self.sig_sm_path)
        self.plm_save_path = os.path.join(path, self.config.ckpt_name)
        torch.save(self.model.state_dict(), self.plm_save_path)
        self.proj_save_path = os.path.join(path, self.config.proj_ckpt_name)
        torch.save(self.proj.state_dict(), self.proj_save_path)

    def load_model(self):
        print("\n******** load model: ******** path:", self.plm_save_path, "\n")
        print("\n******** load model: ******** path:", self.proj_save_path, "\n")
        self.model.load_state_dict(torch.load(self.plm_save_path))
        self.proj.load_state_dict(torch.load(self.proj_save_path))

    def eval(self, eval_data_iterator):
        self.model.eval()
        self.ref_model.eval()
        self.proj.eval()

        total_fid_loss = 0
        total_emd_loss = 0
        total_extracted_loss = 0
        total_unrealted_loss = 0

        count = 0
        WER = 0
        WER_clean = 0

        MSE = 0
        MSE_clean = 0

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            batch_size = np.array(clean_batch["text_a"]).shape[0]
            watermark = self.sig_sm.repeat(batch_size, 1)

            c_inputs, _, c_labels = self.model.process(clean_batch)
            p_inputs, _, p_labels = self.model.process(poison_batch)

            with torch.no_grad():
                c_outputs = self.model(c_inputs)
                c_cls_embeds = c_outputs.hidden_states[-1][:, 0, :]

                c_ref_outputs = self.ref_model(c_inputs)
                c_ref_cls_embeds = c_ref_outputs.hidden_states[-1][:, 0, :]

                p_outputs = self.model(p_inputs)
                p_cls_embeds = p_outputs.hidden_states[-1][:, 0, :]

                p_ref_outputs = self.ref_model(p_inputs)
                p_ref_cls_embeds = p_ref_outputs.hidden_states[-1][:, 0, :]

                dec_extracted = self.emd_loss(self.proj(p_cls_embeds), watermark)

                dec_unrealted = (
                    (torch.mean(self.cos(self.proj(c_ref_cls_embeds), watermark))).pow(
                        2
                    )
                    + (
                        torch.mean(self.cos(self.proj(p_ref_cls_embeds), watermark))
                    ).pow(2)
                    + (torch.mean(self.cos(self.proj(c_cls_embeds), watermark))).pow(2)
                )

                total_extracted_loss += dec_extracted.item()
                total_unrealted_loss += dec_unrealted.item()

                plm_emd = self.emd_loss(self.proj(p_cls_embeds), watermark)
                plm_fid = self.fid_loss(c_cls_embeds, c_ref_cls_embeds)

                if self.use_ori_loss:
                    if self.mlm:
                        c_inputs, _, c_labels = self.model.process(clean_batch)
                        c_outputs, plm_fid = self.MLMLoss(c_inputs, self.model)
                    else:
                        c_inputs, c_labels, _ = self.model.process(clean_batch)
                        c_outputs = self.model(c_inputs, labels=c_labels)
                        plm_fid = c_outputs.loss
                else:
                    plm_fid = self.fid_loss(c_cls_embeds, c_ref_cls_embeds)

                total_fid_loss += plm_fid.item()
                total_emd_loss += plm_emd.item()

                pred = self.proj(p_cls_embeds)
                wer, mse = self.calu_wer(pred)
                WER += wer
                MSE += mse

                pred_clean = self.proj(c_cls_embeds)
                wer_clean, mse_clean = self.calu_wer(pred_clean)
                WER_clean += wer_clean
                MSE_clean += mse_clean

        avg_fid_loss = total_fid_loss / (step + 1)
        avg_emd_loss = total_emd_loss / (step + 1)
        avg_extracted_loss = total_extracted_loss / (step + 1)
        avg_unrealted_loss = total_unrealted_loss / (step + 1)

        dev_score = avg_emd_loss

        WER = WER / (step + 1)
        WER_clean = WER_clean / (step + 1)
        MSE = MSE / (step + 1)
        MSE_clean = MSE_clean / (step + 1)
        return (
            dev_score,
            avg_fid_loss,
            avg_emd_loss,
            avg_extracted_loss,
            avg_unrealted_loss,
            WER,
            WER_clean,
            MSE,
            MSE_clean,
        )

    def verify_plm(
        self,
        victim_model,
        surrogate_model,
        proj,
        verify_dataset,
        sig,
        sm,
        sig_sm,
        poisoner,
        insert_num,
    ):
        victim_model.eval()
        surrogate_model.eval()
        proj.eval()

        self.model = victim_model
        self.proj = proj
        self.sig = sig
        self.sm = sm
        self.sig_sm = sig_sm

        dataloader = {}
        dev_clean_dataset, dev_poison_dataset = poisoner.get_dev_dataset(
            verify_dataset["dev"]
        )
        dataloader["dev-clean"] = get_dataloader(
            dev_clean_dataset, self.batch_size, drop_last=True, shuffle=False
        )
        dataloader["dev-poison"] = get_dataloader(
            dev_poison_dataset, self.batch_size, drop_last=True, shuffle=False
        )

        eval_data_iterator = tqdm(
            zip(cycle(dataloader["dev-clean"]), dataloader["dev-poison"]),
            desc="Verifying",
            ncols=70,
        )
        print("\n")

        vic_WER, vic_WER_clean, sur_WER, sur_WER_clean = 0, 0, 0, 0

        vic_MSE, vic_MSE_clean, sur_MSE, sur_MSE_clean = 0, 0, 0, 0

        vic_poison_matrix = []
        sur_poison_matrix = []
        vic_clean_matrix = []
        sur_clean_matrix = []

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            batch_size = np.array(clean_batch["text_a"]).shape[0]
            watermark = self.sig_sm.repeat(batch_size, 1)

            vic_c_inputs, _, vic_c_labels = victim_model.process(clean_batch)
            vic_p_inputs, _, vic_p_labels = victim_model.process(poison_batch)

            sur_c_inputs, _, sur_c_labels = surrogate_model.process(clean_batch)
            sur_p_inputs, _, sur_p_labels = surrogate_model.process(poison_batch)

            with torch.no_grad():
                vic_c_outputs = victim_model(vic_c_inputs)
                vic_c_cls_embeds = vic_c_outputs.hidden_states[-1][:, 0, :]
                sur_c_outputs = surrogate_model(sur_c_inputs)
                sur_c_cls_embeds = sur_c_outputs.hidden_states[-1][:, 0, :]

                vic_p_outputs = victim_model(vic_p_inputs)
                vic_p_cls_embeds = vic_p_outputs.hidden_states[-1][:, 0, :]
                sur_p_outputs = surrogate_model(sur_p_inputs)
                sur_p_cls_embeds = sur_p_outputs.hidden_states[-1][:, 0, :]

                vic_poison_matrix.extend(vic_p_cls_embeds.cpu().detach().tolist())
                sur_poison_matrix.extend(sur_p_cls_embeds.cpu().detach().tolist())
                vic_clean_matrix.extend(vic_c_cls_embeds.cpu().detach().tolist())
                sur_clean_matrix.extend(sur_c_cls_embeds.cpu().detach().tolist())

                vic_pred_poison = self.proj(vic_p_cls_embeds)
                wer, mse = self.calu_wer(vic_pred_poison)
                vic_WER += wer
                vic_MSE += mse

                vic_pred_clean = self.proj(vic_c_cls_embeds)
                wer_clean, mse_clean = self.calu_wer(vic_pred_clean)
                vic_WER_clean += wer_clean
                vic_MSE_clean += mse_clean

                sur_pred_poison = self.proj(sur_p_cls_embeds)
                wer, mse = self.calu_wer(sur_pred_poison)
                sur_WER += wer
                sur_MSE += mse

                sur_pred_clean = self.proj(sur_c_cls_embeds)
                wer_clean, mse_clean = self.calu_wer(sur_pred_clean)
                sur_WER_clean += wer_clean
                sur_MSE_clean += mse_clean

        vic_WER = vic_WER / (step + 1)
        vic_WER_clean = vic_WER_clean / (step + 1)
        vic_MSE = vic_MSE / (step + 1)
        vic_MSE_clean = vic_MSE_clean / (step + 1)

        sur_WER = sur_WER / (step + 1)
        sur_WER_clean = sur_WER_clean / (step + 1)
        sur_MSE = sur_MSE / (step + 1)
        sur_MSE_clean = sur_MSE_clean / (step + 1)

        print(
            "   vic_WER: {}".format(vic_WER), 
            "   vic_WER_clean: {}".format(vic_WER_clean)
        )
        print(
            "   vic_MSE: {}".format(vic_MSE),
            "   vic_MSE_clean: {}".format(vic_MSE_clean),
        )
        print(
            "   sur_WER: {}".format(sur_WER), 
            "   sur_WER_clean: {}".format(sur_WER_clean)
        )
        print(
            "   sur_MSE: {}".format(sur_MSE),
            "   sur_MSE_clean: {}".format(sur_MSE_clean),
        )



        vic_poison_matrix = normalizeA(np.array(vic_poison_matrix).T)
        sur_poison_matrix = normalizeA(np.array(sur_poison_matrix).T)
        vic_clean_matrix = normalizeA(np.array(vic_clean_matrix).T)
        sur_clean_matrix = normalizeA(np.array(sur_clean_matrix).T)

        print("victim poison rank:", np.linalg.matrix_rank(np.array(vic_poison_matrix)))
        print("victim clean rank:", np.linalg.matrix_rank(np.array(vic_clean_matrix)))
        print("sur poison rank:", np.linalg.matrix_rank(np.array(sur_poison_matrix)))
        print("sur clean rank:", np.linalg.matrix_rank(np.array(sur_clean_matrix)))

        def null(A, eps=0.1):
            u, s, vh = svd(A, full_matrices=1, compute_uv=1)
            a = min(A.shape[0], A.shape[1])
            b = max(A.shape[0], A.shape[1])
            null_space = vh[a:b, :]
            return null_space.T

        NS_poison = null(vic_poison_matrix)
        NS_clean = null(vic_clean_matrix)

        v1 = np.dot(vic_poison_matrix, NS_poison)
        ret1 = calculate_nsmd(vic_poison_matrix, NS_poison)

        v2 = np.dot(sur_poison_matrix, NS_poison)
        ret2 = calculate_nsmd(sur_poison_matrix, NS_poison)

        print("vic_poison_matrix shape", vic_poison_matrix.shape)
        print("NS_poison shape", NS_poison.shape)
        print("v1 shape", v1.shape)
        print("matrix shape", vic_poison_matrix.shape)
        print("NS shape", NS_poison.shape)
        print("NS_victim_poison: ", ret1)
        print("NS_sur_poison: ", ret2)

        v3 = np.dot(vic_clean_matrix, NS_clean)
        ret3 = calculate_nsmd(vic_clean_matrix, NS_clean)

        v4 = np.dot(sur_clean_matrix, NS_clean)
        ret4 = calculate_nsmd(sur_clean_matrix, NS_clean)

        print("NS_victim_clean: ", ret3)
        print("NS_sur_clean: ", ret4)

        import pandas as pd

        save_metrics = pd.DataFrame(
            {
                "NS_victim_poison": ret1,
                "NS_sur_poison": ret2,
                "NS_victim_clean": ret3,
                "NS_sur_clean": ret4,
                "WER_vic": vic_WER,
                "WER_sur": sur_WER,
                "WER_clean_vic": vic_WER_clean,
                "WER_clean_sur": sur_WER_clean,
                "MSE_vic": vic_MSE,
                "MSE_clean_vic": vic_MSE_clean,
                "MSE_sur": sur_MSE,
                "MSE_clean_sur": sur_MSE_clean,
            },
            index=[0],
        )
        os.makedirs(self.result_save_dir, exist_ok=True)
        save_metrics.to_csv(
            f"{self.result_save_dir}/verift_metrics.csv", mode="a", encoding="gbk"
        )
        print("save result:", f"{self.result_save_dir}/verift_metrics.csv")
        logging.info("\n******** Verify finished! ********\n")
        return vic_poison_matrix, NS_poison, vic_clean_matrix, NS_clean

    def verify_ds(self, plm_model, ft_model, verify_dataset, poisoner):
        plm_model.eval()
        ft_model.eval()

        dataloader = {}
        dev_clean_dataset, dev_poison_dataset = poisoner.get_dev_dataset(
            verify_dataset["dev"]
        )
        dataloader["dev-clean"] = get_dataloader(
            dev_clean_dataset, self.batch_size, drop_last=False, shuffle=False
        )
        dataloader["dev-poison"] = get_dataloader(
            dev_poison_dataset, self.batch_size, drop_last=False, shuffle=False
        )

        eval_data_iterator = tqdm(
            zip(cycle(dataloader["dev-clean"]), dataloader["dev-poison"]),
            desc="Verifying",
            ncols=70,
        )
        print("\n")
        vic_poison_matrix = []
        sur_poison_matrix = []
        vic_clean_matrix = []
        sur_clean_matrix = []

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            vic_clean_inputs, _, vic_clean_labels = plm_model.process(clean_batch)
            vic_poison_inputs, _, vic_poison_labels = plm_model.process(poison_batch)

            sur_clean_inputs, sur_clean_labels = ft_model.process(clean_batch)
            sur_poison_inputs, sur_poison_labels = ft_model.process(poison_batch)

            with torch.no_grad():
                vic_clean_outputs = plm_model(vic_clean_inputs)
                vic_clean_cls_embeds = vic_clean_outputs.hidden_states[-1][:, 0, :]
                sur_clean_outputs = ft_model(sur_clean_inputs)
                sur_clean_cls_embeds = sur_clean_outputs.hidden_states[-1][:, 0, :]

                vic_poison_outputs = plm_model(vic_poison_inputs)
                vic_poison_cls_embeds = vic_poison_outputs.hidden_states[-1][:, 0, :]
                sur_poison_outputs = ft_model(sur_poison_inputs)
                sur_poison_cls_embeds = sur_poison_outputs.hidden_states[-1][:, 0, :]

                vic_poison_matrix.extend(vic_poison_cls_embeds.cpu().detach().tolist())
                sur_poison_matrix.extend(sur_poison_cls_embeds.cpu().detach().tolist())
                vic_clean_matrix.extend(vic_clean_cls_embeds.cpu().detach().tolist())
                sur_clean_matrix.extend(sur_clean_cls_embeds.cpu().detach().tolist())


        vic_poison_matrix = normalizeA(np.array(vic_poison_matrix).T)
        sur_poison_matrix = normalizeA(np.array(sur_poison_matrix).T)
        vic_clean_matrix = normalizeA(np.array(vic_clean_matrix).T)
        sur_clean_matrix = normalizeA(np.array(sur_clean_matrix).T)

        print("victim poison rank:", np.linalg.matrix_rank(np.array(vic_poison_matrix)))
        print("victim clean rank:", np.linalg.matrix_rank(np.array(vic_clean_matrix)))
        print("sur poison rank:", np.linalg.matrix_rank(np.array(sur_poison_matrix)))
        print("sur clean rank:", np.linalg.matrix_rank(np.array(sur_clean_matrix)))



        NS_poison = null(vic_poison_matrix)
        NS_clean = null(vic_clean_matrix)

        ret1 = calculate_nsmd(vic_poison_matrix, NS_poison)
        ret2 = calculate_nsmd(sur_poison_matrix, NS_poison)

        print("matrix shape", vic_poison_matrix.shape)
        print("NS shape", NS_poison.shape)
        print("NS_plm_poison: ", ret1, "            NS_ft_poison: ", ret2)

        ret3 = calculate_nsmd(vic_clean_matrix, NS_clean)
        ret4 = calculate_nsmd(sur_clean_matrix, NS_clean)

        print("NS_plm_clean: ", ret3, "            NS_ft_clean: ", ret4)

        save_metrics = pd.DataFrame(
            {
                "vic_vs_ds-sur": "null",
                "NS_victim_poison": ret1,
                "NS_sur_poison": ret2,
                "NS_victim_clean": ret3,
                "NS_sur_clean": ret4,
            },
            index=[0],
        )
        os.makedirs(self.result_save_dir, exist_ok=True)
        save_metrics.to_csv(
            f"{self.result_save_dir}/verift_metrics.csv", mode="a", encoding="gbk"
        )
        print("save result:", f"{self.result_save_dir}/verift_metrics.csv")
        logging.info("\n******** Verify DS finished! ********\n")
        return vic_poison_matrix.shape
