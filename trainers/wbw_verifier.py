import logging
import os
from itertools import cycle
import numpy as np
import torch
from numpy import *
from numpy.linalg import *
from tqdm import tqdm
from utils import *
from .dataloader import get_dataloader
from .loss_utlis import SupConLoss
from .trainer import Trainer
import pandas as pd

def null(A, eps=0.1):
    u, s, vh = svd(A, full_matrices=1, compute_uv=1)
    a = min(A.shape[0], A.shape[1])
    b = max(A.shape[0], A.shape[1])
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
    n, p = H.shape

    nsmd = (1 / n) * np.sum(np.sqrt(np.abs(H)))

    return nsmd


class WBWVerifier(Trainer):
    def __init__(
        self,
        config,
        save_dir,
        victim_model,
        proj,
        verify_dataset,
        sig,
        sm,
        sig_sm,
        poisoner,
        insert_num,
    ):
        super().__init__(config, save_dir)
        self.MSELoss = torch.nn.MSELoss()
        self.SupConLoss = SupConLoss()
        self.extracted_grads = []

        self.victim_model = victim_model
        self.proj = proj
        self.sig = sig
        self.sm = sm
        self.sig_sm = sig_sm
        self.poisoner = poisoner
        self.verify_dataset = verify_dataset
        self.init_metrics(verify_dataset, sig, sm, sig_sm, poisoner, insert_num)

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

            BP = torch.tensor(BP, dtype=torch.float32).to(self.victim_model.device)

            detected = torch.sum(torch.abs(BP - self.sig) < 0.5).item()

            mse += self.MSELoss(BP, self.sig).item()

            wer += detected / sk_size

        num_samples = k + 1
        wer /= num_samples
        mse /= num_samples

        return wer, mse

    def init_metrics(self, verify_dataset, sig, sm, sig_sm, poisoner, insert_num):
        self.init_vic_poison_matrix = None
        self.init_vic_clean_matrix = None

        self.victim_model.eval()
        self.proj.eval()

        self.dev_clean_dataset, self.dev_poison_dataset = poisoner.get_dev_dataset(
            verify_dataset["dev"]
        )
        dataloader = {}

        dataloader["dev-clean"] = get_dataloader(
            self.dev_clean_dataset, self.batch_size, drop_last=True, shuffle=False
        )
        dataloader["dev-poison"] = get_dataloader(
            self.dev_poison_dataset, self.batch_size, drop_last=True, shuffle=False
        )
        eval_data_iterator = tqdm(
            zip(cycle(dataloader["dev-clean"]), dataloader["dev-poison"]),
            desc="Verifying",
            ncols=70,
        )

        vic_WER, vic_WER_clean = 0, 0
        vic_MSE, vic_MSE_clean = 0, 0

        vic_poison_matrix = []
        vic_clean_matrix = []

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            batch_size = np.array(clean_batch["text_a"]).shape[0]
            watermark = self.sig_sm.repeat(batch_size, 1)

            vic_clean_inputs, _, vic_clean_labels = self.victim_model.process(clean_batch)
            vic_poison_inputs, _, vic_poison_labels = self.victim_model.process(poison_batch)

            with torch.no_grad():
                vic_clean_outputs = self.victim_model(vic_clean_inputs)
                vic_clean_cls_embeds = vic_clean_outputs.hidden_states[-1][:, 0, :]

                vic_poison_outputs = self.victim_model(vic_poison_inputs)
                vic_poison_cls_embeds = vic_poison_outputs.hidden_states[-1][:, 0, :]

                vic_poison_matrix.extend(vic_poison_cls_embeds.cpu().detach().tolist())
                vic_clean_matrix.extend(vic_clean_cls_embeds.cpu().detach().tolist())

                vic_pred_poison = self.proj(vic_poison_cls_embeds)
                wer, mse = self.calu_wer(vic_pred_poison)
                vic_WER += wer
                vic_MSE += mse

                vic_pred_clean = self.proj(vic_clean_cls_embeds)
                wer_clean, mse_clean = self.calu_wer(vic_pred_clean)
                vic_WER_clean += wer_clean
                vic_MSE_clean += mse_clean

        vic_WER = vic_WER / (step + 1)
        vic_WER_clean = vic_WER_clean / (step + 1)
        vic_MSE = vic_MSE / (step + 1)
        vic_MSE_clean = vic_MSE_clean / (step + 1)
        print("\n")
        print(
            "   vic_WER: {}".format(vic_WER), 
            "   vic_WER_clean: {}".format(vic_WER_clean)
        )
        print(
            "   vic_MSE: {}".format(vic_MSE),
            "   vic_MSE_clean: {}".format(vic_MSE_clean),
        )



        vic_poison_matrix = normalizeA(np.array(vic_poison_matrix).T)
        vic_clean_matrix = normalizeA(np.array(vic_clean_matrix).T)



        self.NS_poison = null(vic_poison_matrix)
        self.NS_clean = null(vic_clean_matrix)

        v1 = np.dot(vic_poison_matrix, self.NS_poison)
        ret1 = calculate_nsmd(vic_poison_matrix, self.NS_poison)

        print("matrix shape", vic_poison_matrix.shape)
        print("NS shape", self.NS_poison.shape)
        print("NSMD_victim_poison: ", ret1)
        print("v1=dot(vic_poison_matrix,self.NS_poison) v1.shape ", v1.shape)

        v3 = np.dot(vic_clean_matrix, self.NS_clean)
        ret3 = calculate_nsmd(vic_clean_matrix, self.NS_clean)

        print("NSMD_victim_clean: ", ret3)

        print("*" * 50)

        import pandas as pd

        save_metrics = pd.DataFrame(
            {
                "type": "victim PLM",
                "NSMD_victim_poison": ret1,
                "NSMD_victim_clean": ret3,
                "WER_vic": vic_WER,
                "WER_clean_vic": vic_WER_clean,
                "MSE_vic": vic_MSE,
                "MSE_clean_vic": vic_MSE_clean,
            },
            index=[0],
        )
        os.makedirs(self.result_save_dir, exist_ok=True)
        save_metrics.to_csv(
            f"{self.result_save_dir}/verift_metrics.csv", mode="a", encoding="gbk"
        )
        print(
            "verify matrix@nullspce=0: saving results path: ",
            f"{self.result_save_dir}/verift_metrics.csv",
        )

        logging.info("\n******** Verify finished! ********\n")

    def verify_plm(self, surrogate_model, type):
        surrogate_model.eval()
        self.victim_model.eval()

        vic_WER, vic_WER_clean = 0, 0
        vic_MSE, vic_MSE_clean = 0, 0

        vic_poison_matrix = []
        vic_clean_matrix = []

        sur_WER, sur_WER_clean = 0, 0
        sur_MSE, sur_MSE_clean = 0, 0

        sur_poison_matrix = []
        sur_clean_matrix = []

        dataloader = {}

        dataloader["dev-clean"] = get_dataloader(
            self.dev_clean_dataset, self.batch_size, drop_last=True, shuffle=False
        )
        dataloader["dev-poison"] = get_dataloader(
            self.dev_poison_dataset, self.batch_size, drop_last=True, shuffle=False
        )

        eval_data_iterator = tqdm(
            zip(cycle(dataloader["dev-clean"]), dataloader["dev-poison"]),
            desc="Verifying",
            ncols=70,
        )

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            batch_size = np.array(clean_batch["text_a"]).shape[0]
            watermark = self.sig_sm.repeat(batch_size, 1)

            sur_clean_inputs, _, sur_clean_labels = surrogate_model.process(clean_batch)
            sur_poison_inputs, _, sur_poison_labels = surrogate_model.process(poison_batch)

            vic_clean_inputs, _, vic_clean_labels = self.victim_model.process(clean_batch)
            vic_poison_inputs, _, vic_poison_labels = self.victim_model.process(poison_batch)

            with torch.no_grad():
                sur_clean_outputs = surrogate_model(sur_clean_inputs)
                sur_clean_cls_embeds = sur_clean_outputs.hidden_states[-1][:, 0, :]

                sur_poison_outputs = surrogate_model(sur_poison_inputs)
                sur_poison_cls_embeds = sur_poison_outputs.hidden_states[-1][:, 0, :]

                sur_poison_matrix.extend(sur_poison_cls_embeds.cpu().detach().tolist())
                sur_clean_matrix.extend(sur_clean_cls_embeds.cpu().detach().tolist())

                sur_pred_poison = self.proj(sur_poison_cls_embeds)

                wer, mse = self.calu_wer(sur_pred_poison)
                sur_WER += wer
                sur_MSE += mse

                sur_pred_clean = self.proj(sur_clean_cls_embeds)
                wer_clean, mse_clean = self.calu_wer(sur_pred_clean)
                sur_WER_clean += wer_clean
                sur_MSE_clean += mse_clean

            with torch.no_grad():
                vic_clean_outputs = self.victim_model(vic_clean_inputs)
                vic_clean_cls_embeds = vic_clean_outputs.hidden_states[-1][:, 0, :]

                vic_poison_outputs = self.victim_model(vic_poison_inputs)
                vic_poison_cls_embeds = vic_poison_outputs.hidden_states[-1][:, 0, :]

                vic_poison_matrix.extend(vic_poison_cls_embeds.cpu().detach().tolist())
                vic_clean_matrix.extend(vic_clean_cls_embeds.cpu().detach().tolist())

                vic_pred_poison = self.proj(vic_poison_cls_embeds)
                vic_wer, vic_mse = self.calu_wer(vic_pred_poison)
                vic_WER += vic_wer
                vic_MSE += vic_mse

                vic_pred_clean = self.proj(vic_clean_cls_embeds)
                vic_wer_clean, vic_mse_clean = self.calu_wer(vic_pred_clean)
                vic_WER_clean += vic_wer_clean
                vic_MSE_clean += vic_mse_clean

        vic_WER = vic_WER / (step + 1)
        vic_WER_clean = vic_WER_clean / (step + 1)
        vic_MSE = vic_MSE / (step + 1)
        vic_MSE_clean = vic_MSE_clean / (step + 1)

        print("\n")

        sur_WER = sur_WER / (step + 1)
        sur_WER_clean = sur_WER_clean / (step + 1)
        sur_MSE = sur_MSE / (step + 1)
        sur_MSE_clean = sur_MSE_clean / (step + 1)

        print(
            "   sur_WER: {}".format(sur_WER), 
            "   sur_WER_clean: {}".format(sur_WER_clean)
        )
        print(
            "   sur_MSE: {}".format(sur_MSE),
            "   sur_MSE_clean: {}".format(sur_MSE_clean),
        )

        def normalizeA(A):
            normA = np.linalg.norm(A, ord=None, axis=1)
            normA = np.tile(normA, (A.shape[1], 1)).T
            A_normlize = A / normA
            return A_normlize

        print()
        print("vic_poison_matrix[0][0:20]:\n", vic_poison_matrix[0][0:20])
        print("vic_clean_matrix[0][0:20]:\n", vic_clean_matrix[0][0:20])
        print("sur_poison_matrix[0][0:20]:\n", sur_poison_matrix[0][0:20])
        print("sur_clean_matrix[0][0:20]:\n", sur_clean_matrix[0][0:20])

        vic_poison_matrix = normalizeA(np.array(vic_poison_matrix).T)
        vic_clean_matrix = normalizeA(np.array(vic_clean_matrix).T)

        sur_poison_matrix = normalizeA(np.array(sur_poison_matrix).T)
        sur_clean_matrix = normalizeA(np.array(sur_clean_matrix).T)

        print("victim poison rank:", np.linalg.matrix_rank(np.array(vic_poison_matrix)))
        print("victim clean rank:", np.linalg.matrix_rank(np.array(vic_clean_matrix)))
        print("sur poison rank:", np.linalg.matrix_rank(np.array(sur_poison_matrix)))
        print("sur clean rank:", np.linalg.matrix_rank(np.array(sur_clean_matrix)))


        v1 = np.dot(vic_poison_matrix, self.NS_poison)
        ret1 = calculate_nsmd(vic_poison_matrix, self.NS_poison)

        print("\n")

        v3 = np.dot(vic_clean_matrix, self.NS_clean)
        ret3 = calculate_nsmd(vic_clean_matrix, self.NS_clean)

        v2 = np.dot(sur_poison_matrix, self.NS_poison)
        ret2 = calculate_nsmd(sur_poison_matrix, self.NS_poison)

        print(f"{type}_NSMD_sur_poison: ", ret2)

        v4 = np.dot(sur_clean_matrix, self.NS_clean)
        ret4 = calculate_nsmd(sur_clean_matrix, self.NS_clean)

        print(f"{type}_NSMD_sur_clean: ", ret4)

        print("matrix shape", vic_poison_matrix.shape)
        print("NS shape", self.NS_poison.shape)
        print("NS_plm_poison: ", ret1, "            NS_sur_poison: ", ret2)


        save_metrics = pd.DataFrame(
            {
                "type": type,
                "NS_sur_poison": ret2,
                "NS_sur_clean": ret4,
                "WER_sur": sur_WER,
                "WER_clean_sur": sur_WER_clean,
                "MSE_sur": sur_MSE,
                "MSE_clean_sur": sur_MSE_clean,
            },
            index=[0],
        )
        os.makedirs(self.result_save_dir, exist_ok=True)
        save_metrics.to_csv(
            f"{self.result_save_dir}/verift_metrics.csv", mode="a", encoding="gbk"
        )
        print(
            f"saving results: verify {type} path:{self.result_save_dir}/verift_metrics.csv "
        )
        logging.info("\n******** Verify finished! ********\n")
        return self.NS_poison, self.NS_clean

    def verify_ds(self, ft_model, type):
        ft_model.eval()
        self.victim_model.eval()

        vic_poison_matrix = []
        vic_clean_matrix = []

        sur_poison_matrix = []
        sur_clean_matrix = []

        dataloader = {}
        dataloader["dev-clean"] = get_dataloader(
            self.dev_clean_dataset, self.batch_size, drop_last=True, shuffle=False
        )
        dataloader["dev-poison"] = get_dataloader(
            self.dev_poison_dataset, self.batch_size, drop_last=True, shuffle=False
        )
        eval_data_iterator = tqdm(
            zip(cycle(dataloader["dev-clean"]), dataloader["dev-poison"]),
            desc="Verifying",
            ncols=70,
        )
        print("\n")

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            sur_clean_inputs, sur_clean_labels = ft_model.process(clean_batch)
            sur_poison_inputs, sur_poison_labels = ft_model.process(poison_batch)

            vic_clean_inputs, _, vic_clean_labels = self.victim_model.process(clean_batch)
            vic_poison_inputs, _, vic_poison_labels = self.victim_model.process(poison_batch)

            with torch.no_grad():
                sur_clean_outputs = ft_model(sur_clean_inputs)
                sur_clean_cls_embeds = sur_clean_outputs.hidden_states[-1][:, 0, :]

                sur_poison_outputs = ft_model(sur_poison_inputs)
                sur_poison_cls_embeds = sur_poison_outputs.hidden_states[-1][:, 0, :]

                sur_poison_matrix.extend(sur_poison_cls_embeds.cpu().detach().tolist())
                sur_clean_matrix.extend(sur_clean_cls_embeds.cpu().detach().tolist())

            with torch.no_grad():
                vic_clean_outputs = self.victim_model(vic_clean_inputs)
                vic_clean_cls_embeds = vic_clean_outputs.hidden_states[-1][:, 0, :]

                vic_poison_outputs = self.victim_model(vic_poison_inputs)
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

        vic_poison_matrix = normalizeA(np.array(vic_poison_matrix).T)
        vic_clean_matrix = normalizeA(np.array(vic_clean_matrix).T)

        sur_poison_matrix = normalizeA(np.array(sur_poison_matrix).T)
        sur_clean_matrix = normalizeA(np.array(sur_clean_matrix).T)

        ret2 = calculate_nsmd(sur_poison_matrix, self.NS_poison)

        print("NS shape", self.NS_poison.shape)
        print(f"{type}_NSMD_ft_poison: ", ret2)

        ret4 = calculate_nsmd(sur_clean_matrix, self.NS_clean)

        print(f"{type}_NSMD_ft_clean: ", ret4)

        print(f"{type}_vic_poison_matrix shape: ", vic_poison_matrix.shape)
        print(f"{type}_sur_poison_matrix shape: ", sur_poison_matrix.shape)

        save_metrics = pd.DataFrame(
            {"type": type, "NS_sur_poison": ret2, "NS_sur_clean": ret4}, index=[0]
        )
        os.makedirs(self.result_save_dir, exist_ok=True)
        save_metrics.to_csv(
            f"{self.result_save_dir}/verift_metrics.csv", mode="a", encoding="gbk"
        )
        print("saving result: " + f"{self.result_save_dir}/verift_metrics.csv")
        logging.info("\n******** Verify DS finished! ********\n")
        return sur_poison_matrix.shape
