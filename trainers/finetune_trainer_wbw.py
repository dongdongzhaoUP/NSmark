import logging
import os
from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
from numpy import *
from numpy.linalg import *
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from .dataloader import get_dataloader, get_dict_dataloader
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

def calculate_nsmd(A, N):
    H = np.dot(A, N)
    n, p = H.shape

    nsmd = (1 / n) * np.sum(np.sqrt(np.abs(H)))

    return nsmd


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


class FineTuneWBWTrainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.MSELoss = torch.nn.MSELoss()

        self.embed = config.embed_loss
        self.freeze_plm = config.freeze_plm

        print(
            "------------------------DS test self.embed--------------------------------",
            self.embed,
        )

    def emd_loss(self, x, y):
        if self.embed == "mse":
            return self.MSELoss(x, y)
        else:
            return -torch.mean(self.cos(x, y))

    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_loss = 0
        for step, batch in enumerate(data_iterator):
            inputs, labels = self.model.process(batch)
            output = self.model(inputs, labels)
            loss = output.loss
            total_loss += loss.item()
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_loss = total_loss / len(data_iterator)
        return avg_loss

    def train(
        self,
        model,
        dataset,
        plm_model,
        verify_dataset,
        poisoner,
        test_dataset,
        proj,
        sig,
        sm,
        sig_sm,
    ):
        self.model = model

        dataloader = get_dict_dataloader(dataset, self.batch_size)

        no_decay = ["bias", "LayerNorm.weight"]

        if self.freeze_plm:
            print("Freeze PLM, just fintune the classifier! ")

            if hasattr(self.model.plm, "classifier"):
                classifier_params = self.model.plm.classifier.named_parameters()
            elif hasattr(self.model.plm, "logits_proj"):
                classifier_params = self.model.plm.logits_proj.named_parameters()
            else:
                raise AttributeError(
                    "Neither 'classifier' nor 'logits_proj' found in the model."
                )

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in classifier_params
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in classifier_params
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            for param in model.plm.base_model.parameters():
                param.requires_grad = False
        else:
            print("Fintune the whole model! Fintune the classifier and plm model! ")

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

        train_length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_epochs * train_length,
            num_training_steps=self.epochs * train_length,
        )

        logging.info("\n************ Finetuning ************\n")
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info(
            "  Gradient Accumulation steps = %d", self.gradient_accumulation_steps
        )
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        best_dev_score = -1

        _, _ = self.init_NS(plm_model, verify_dataset, poisoner)

        ds_poison_matrix = self.test(self.model, test_dataset, proj, sig, sm, sig_sm)

        for epoch in range(self.epochs):
            logging.info("------------ Epoch : {} ------------".format(epoch + 1))
            data_iterator = tqdm(dataloader["train"], desc="Iteration", ncols=70)
            print("\n")
            epoch_loss = self.train_one_epoch(data_iterator)
            print("\n")
            logging.info("  Train-Loss: {}".format(epoch_loss))
            dev_score = self.eval(self.model, dataloader["dev"])
            logging.info("  Dev-Acc: {}".format(dev_score))

            NS_shape, NS_ft_poison, NS_ft_clean = self.verify_plm(
                plm_model, self.model, verify_dataset, poisoner, epoch
            )

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                print(" Fine-tune dev_score: ", dev_score)
                print(" Save model! fine-tune")
                self.save_model()


            save_metrics = pd.DataFrame(
                {"epoch": epoch, "Train-Loss": epoch_loss, "Dev-Acc": dev_score},
                index=[0],
            )
            save_metrics.to_csv(
                f"{self.result_save_dir}/FT_metrics.csv", mode="a", encoding="gbk"
            )
            print(" Save results path:", f"{self.result_save_dir}/FT_metrics.csv")
        logging.info("\n******** Finetuning finished! ********\n")

        self.load_model()
        return self.model

    def eval(self, model, dataloader):
        model.eval()
        allpreds, alllabels = [], []

        for batch in tqdm(dataloader, desc="Evaluating", ncols=70):
            inputs, labels = model.process(batch)
            with torch.no_grad():
                preds = model(inputs)
            allpreds.extend(torch.argmax(preds.logits, dim=-1).cpu().tolist())
            alllabels.extend(labels.cpu().tolist())
        print("\n")
        dev_score = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(
            allpreds
        )

        return dev_score

    def test(self, model, dataset, proj, sig, sm, sig_sm):
        test_dataloader = get_dict_dataloader(dataset, self.batch_size)
        logging.info("\n************ Testing DS ************\n")

        test_ACC = self.eval(model, test_dataloader["test-clean"])
        logging.info("  Test-ACC: {}".format(test_ACC))

        print()
        ds_poison_matrix, test_WER, test_MSE, test_emd_loss = self.calu_WER(
            model, test_dataloader["test-poison"], sig_sm, proj, sig, sm
        )
        logging.info("  Test-emd_loss: {}".format(test_emd_loss))
        print("  Test-WER: {}".format(test_WER), 
              "  Test-MSE: {}".format(test_MSE))

        print()
        _, test_WER_clean, test_MSE_clean, test_emd_loss_clean = self.calu_WER(
            model, test_dataloader["test-clean"], sig_sm, proj, sig, sm
        )
        logging.info("  Test-emd_loss_clean: {}".format(test_emd_loss_clean))
        print(
            "  Test-WER_clean: {}".format(test_WER_clean),
            "  Test-MSE_clean: {}".format(test_MSE_clean),
        )


        save_metrics = pd.DataFrame(
            {
                "Test-ACC": test_ACC,
                "Test-WER": test_WER,
                "Test-WER_clean": test_WER_clean,
                "Test-emd_loss": test_emd_loss,
                "Test-MSE": test_MSE,
                "Test-emd_loss_clean": test_emd_loss_clean,
                "Test-MSE_clean": test_MSE_clean,
            },
            index=[0],
        )
        save_metrics.to_csv(
            f"{self.result_save_dir}/FT_metrics.csv", mode="a", encoding="gbk"
        )
        print("save result path:" + f"{self.result_save_dir}/FT_metrics.csv")
        logging.info("\n******** Testing DS finished! ********\n")
        return ds_poison_matrix

    def test_MS(self, model, dataset, proj, sig, sm, sig_sm):
        test_dataloader = get_dict_dataloader(dataset, self.batch_size)
        logging.info("\n************ Testing MS ************\n")

        test_ACC = self.eval(model, test_dataloader["test-clean"])
        logging.info("  Test_MS-ACC: {}".format(test_ACC))

        print()
        _, test_WER, test_MSE, test_emd_loss = self.calu_WER(
            model, test_dataloader["test-poison"], sig_sm, proj, sig, sm
        )
        logging.info("  Test_MS-emd_loss: {}".format(test_emd_loss))
        print(
            "  Test_MS-WER: {}".format(test_WER),
            "  Test_MS-MSE: {}".format(test_MSE),
        )

        print()
        _, test_WER_clean, test_MSE_clean, test_emd_loss_clean = self.calu_WER(
            model, test_dataloader["test-clean"], sig_sm, proj, sig, sm
        )
        logging.info("  Test_MS-emd_loss_clean: {}".format(test_emd_loss_clean))
        print(
            "  Test_MS-WER_clean: {}".format(test_WER_clean),
            "  Test_MS-MSE_clean: {}".format(test_MSE_clean),
        )
        

        save_metrics = pd.DataFrame(
            {
                "Test_MS-ACC": test_ACC,
                "Test_MS-WER": test_WER,
                "Test_MS-WER_clean": test_WER_clean,
                "Test_MS-emd_loss": test_emd_loss,
                "Test_MS-WER": test_WER,
                "Test_MS-MSE": test_MSE,
                "Test_MS-emd_loss_clean": test_emd_loss_clean,
                "Test_MS-WER_clean": test_WER_clean,
                "Test_MS-MSE_clean": test_MSE_clean,
            },
            index=[0],
        )
        save_metrics.to_csv(
            f"{self.result_save_dir}/FT_metrics.csv", mode="a", encoding="gbk"
        )
        print("save result:", f"{self.result_save_dir}/FT_metrics.csv")
        logging.info("\n******** Testing MS finished! ********\n")

    def test_LFEA(self, model, dataset, proj, sig, sm, sig_sm):
        test_dataloader = get_dict_dataloader(dataset, self.batch_size)
        logging.info("\n************ Testing LFEA ************\n")

        test_ACC = self.eval(model, test_dataloader["test-clean"])
        logging.info("  Test_LFEA-ACC: {}".format(test_ACC))

        print()
        _, test_WER, test_MSE, test_emd_loss = self.calu_WER(
            model, test_dataloader["test-poison"], sig_sm, proj, sig, sm
        )
        logging.info("  Test_LFEA-emd_loss: {}".format(test_emd_loss))
        print(
            "  Test_LFEA-WER: {}".format(test_WER),
            "  Test_LFEA-MSE: {}".format(test_MSE),
        )

        print()
        _, test_WER_clean, test_MSE_clean, test_emd_loss_clean = self.calu_WER(
            model, test_dataloader["test-clean"], sig_sm, proj, sig, sm
        )
        logging.info("  Test_LFEA-emd_loss_clean: {}".format(test_emd_loss_clean))
        print(
            "  Test_LFEA-WER_clean: {}".format(test_WER_clean),
            "  Test_LFEA-MSE_clean: {}".format(test_MSE_clean),
        )


        save_metrics = pd.DataFrame(
            {
                "Test_LFEA-ACC": test_ACC,
                "Test_LFEA-WER": test_WER,
                "Test_LFEA-WER_clean": test_WER_clean,
                "Test_LFEA-emd_loss": test_emd_loss,
                "Test_LFEA-MSE": test_MSE,
                "Test_LFEA-emd_loss_clean": test_emd_loss_clean,
                "Test_LFEA-MSE_clean": test_MSE_clean,
            },
            index=[0],
        )
        save_metrics.to_csv(
            f"{self.result_save_dir}/FT_metrics.csv", mode="a", encoding="gbk"
        )
        print(" Save result:", f"{self.result_save_dir}/FT_metrics.csv")
        logging.info("\n******** Testing LFEA finished! ********\n")

    def calu_wer(self, preds, sig, sm):
        n = 3
        sk_size = 256

        wer = 0
        mse = 0

        for k, SBP in enumerate(preds):
            EBP = (SBP / sm).cpu().detach().numpy()

            quantized = np.zeros((n, sk_size))

            for i in range(n):
                sub_block = EBP[i * sk_size : (i + 1) * sk_size]
                quantized[i] = quantize(sub_block)

            BP = vote(quantized)

            BP = torch.tensor(BP, dtype=torch.float32).to(self.model.device)

            detected = torch.sum(torch.abs(BP - sig) < 0.5).item()

            mse += self.MSELoss(BP, sig).item()

            wer += detected / sk_size

        num_samples = k + 1
        wer /= num_samples
        mse /= num_samples

        return wer, mse

    def calu_WER(self, model, dataloader, sig_sm, proj, sig, sm):
        data_iterator = tqdm(dataloader, desc="Testing", ncols=70)
        print("\n")
        model.eval()
        proj.eval()
        total_emd_loss = 0
        WER = 0
        count = 0
        MSE = 0
        data_length = len(dataloader)

        ds_matrix = []

        for step, batch in enumerate(data_iterator):
            batch_size = np.array(batch["text_a"]).shape[0]
            watermark = sig_sm.repeat(batch_size, 1)

            inputs, labels = model.process(batch)
            with torch.no_grad():
                preds = model(inputs)
            cls_embeds = preds.hidden_states[-1][:, 0, :]

            plm_emd_loss = self.emd_loss(proj(cls_embeds), watermark)
            total_emd_loss += plm_emd_loss.item()

            ds_matrix.extend(cls_embeds.cpu().detach().tolist())
            pred = proj(cls_embeds)
            wer, mse = self.calu_wer(pred, sig, sm)
            WER += wer
            MSE += mse

        avg_emd_loss = total_emd_loss / (step + 1)

        WER = WER / (step + 1)
        MSE = MSE / (step + 1)

        ds_matrix = np.array(ds_matrix)

        return ds_matrix, WER, MSE, avg_emd_loss

    def init_NS(self, plm_model, verify_dataset, poisoner):
        plm_model.eval()

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
        vic_clean_matrix = []

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            vic_clean_inputs, _, vic_clean_labels = plm_model.process(clean_batch)
            vic_poison_inputs, _, vic_poison_labels = plm_model.process(poison_batch)

            with torch.no_grad():
                vic_clean_outputs = plm_model(vic_clean_inputs)
                vic_clean_cls_embeds = vic_clean_outputs.hidden_states[-1][:, 0, :]

                vic_poison_outputs = plm_model(vic_poison_inputs)
                vic_poison_cls_embeds = vic_poison_outputs.hidden_states[-1][:, 0, :]

                vic_poison_matrix.extend(vic_poison_cls_embeds.cpu().detach().tolist())
                vic_clean_matrix.extend(vic_clean_cls_embeds.cpu().detach().tolist())

        def normalizeA(A):
            normA = np.linalg.norm(A, ord=None, axis=1)
            normA = np.tile(normA, (A.shape[1], 1)).T
            A_normlize = A / normA
            return A_normlize

        vic_poison_matrix = normalizeA(np.array(vic_poison_matrix).T)
        vic_clean_matrix = normalizeA(np.array(vic_clean_matrix).T)


        self.NS_poison = null(vic_poison_matrix)
        self.NS_clean = null(vic_clean_matrix)

        ret1 = calculate_nsmd(vic_poison_matrix, self.NS_poison)

        print("matrix shape", vic_poison_matrix.shape)
        print("NS shape", self.NS_poison.shape)
        print("NS_plm_poison: ", ret1)

        ret3 = calculate_nsmd(vic_clean_matrix, self.NS_clean)
        print("NS_plm_clean: ", ret3)


        save_metrics = pd.DataFrame(
            {"PLM": " ", "NS_victim_poison": ret1, "NS_victim_clean": ret3}, index=[0]
        )
        os.makedirs(self.result_save_dir, exist_ok=True)
        save_metrics.to_csv(
            f"{self.result_save_dir}/verift_metrics.csv", mode="a", encoding="gbk"
        )
        print("save results path:" + f"{self.result_save_dir}/verift_metrics.csv")
        logging.info("\n******** Verify DS finished! ********\n")
        return self.NS_poison, self.NS_clean

    def verify_plm(self, plm_model, ft_model, verify_dataset, poisoner, epoch):
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
        sur_poison_matrix = []
        sur_clean_matrix = []

        vic_poison_matrix = []
        vic_clean_matrix = []

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            sur_clean_inputs, sur_clean_labels = ft_model.process(clean_batch)
            sur_poison_inputs, sur_poison_labels = ft_model.process(poison_batch)

            vic_clean_inputs, _, vic_clean_labels = plm_model.process(clean_batch)
            vic_poison_inputs, _, vic_poison_labels = plm_model.process(poison_batch)

            with torch.no_grad():
                sur_clean_outputs = ft_model(sur_clean_inputs)
                sur_clean_cls_embeds = sur_clean_outputs.hidden_states[-1][:, 0, :]

                sur_poison_outputs = ft_model(sur_poison_inputs)
                sur_poison_cls_embeds = sur_poison_outputs.hidden_states[-1][:, 0, :]

                sur_poison_matrix.extend(sur_poison_cls_embeds.cpu().detach().tolist())
                sur_clean_matrix.extend(sur_clean_cls_embeds.cpu().detach().tolist())

            with torch.no_grad():
                vic_clean_outputs = plm_model(vic_clean_inputs)
                vic_clean_cls_embeds = vic_clean_outputs.hidden_states[-1][:, 0, :]

                vic_poison_outputs = plm_model(vic_poison_inputs)
                vic_poison_cls_embeds = vic_poison_outputs.hidden_states[-1][:, 0, :]

                vic_poison_matrix.extend(vic_poison_cls_embeds.cpu().detach().tolist())
                vic_clean_matrix.extend(vic_clean_cls_embeds.cpu().detach().tolist())


        print()
        print("vic_poison_matrix[0][0:20]:\n", vic_poison_matrix[0][0:20])
        print("vic_clean_matrix[0][0:20]:\n", vic_clean_matrix[0][0:20])
        print("sur_poison_matrix[0][0:20]:\n", sur_poison_matrix[0][0:20])
        print("sur_clean_matrix[0][0:20]:\n", sur_clean_matrix[0][0:20])

        print("victim poison rank:", np.linalg.matrix_rank(np.array(vic_poison_matrix)))
        print("victim clean rank:", np.linalg.matrix_rank(np.array(vic_clean_matrix)))
        print("sur poison rank:", np.linalg.matrix_rank(np.array(sur_poison_matrix)))
        print("sur clean rank:", np.linalg.matrix_rank(np.array(sur_clean_matrix)))

        sur_poison_matrix = normalizeA(np.array(sur_poison_matrix).T)
        sur_clean_matrix = normalizeA(np.array(sur_clean_matrix).T)

        vic_poison_matrix = normalizeA(np.array(vic_poison_matrix).T)
        vic_clean_matrix = normalizeA(np.array(vic_clean_matrix).T)


        ret2 = calculate_nsmd(sur_poison_matrix, self.NS_poison)
        print("NS_ft_poison: ", ret2)

        ret4 = calculate_nsmd(sur_clean_matrix, self.NS_clean)
        print("NS_ft_clean: ", ret4)


        save_metrics = pd.DataFrame(
            {"ft_epoch": epoch, "NS_sur_poison": ret2, "NS_sur_clean": ret4}, index=[0]
        )
        os.makedirs(self.result_save_dir, exist_ok=True)
        save_metrics.to_csv(
            f"{self.result_save_dir}/verift_metrics.csv", mode="a", encoding="gbk"
        )
        print(" Save result:", f"{self.result_save_dir}/verift_metrics.csv")
        logging.info("\n******** Verify DS finished! ********\n")
        return sur_poison_matrix.shape, ret2, ret4
