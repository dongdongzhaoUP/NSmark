import os

import torch


class Trainer(object):
    def __init__(self, config, save_dir):
        try:
            self.epochs = config.epochs
        except KeyError:
            self.wf_threshold = config.wf_threshold
            self.num_candidates = config.num_candidates
            self.beam_size = config.beam_size
            self.eval_batch = config.eval_batch
        else:
            self.weight_decay = config.weight_decay
            self.warm_up_epochs = config.warm_up_epochs
            self.max_grad_norm = config.max_grad_norm
            self.lr = float(config.lr)

        self.batch_size = config.batch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

        self.device = torch.device("cuda")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_save_path = os.path.join(self.save_dir, config.ckpt_name)

        self.config = config

        self.result_save_dir = self.config.result_save_dir

    def train_one_epoch(self, data_iterator):
        pass

    def train(self, model, dataset):
        pass

    def eval(self, model, dataloader):
        pass

    def test(self, model, dataset):
        pass

    def save_model(self):
        print("save_model path:", self.model_save_path)
        torch.save(self.model.state_dict(), self.model_save_path)

    def load_model(self):
        print("model.load_state_dict path:", self.model_save_path)
        self.model.load_state_dict(torch.load(self.model_save_path))
