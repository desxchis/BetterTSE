import os
import time

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from data import EditDataset
from evaluation.base_evaluator import BaseEvaluator

class Finetuner:
    def __init__(self, configs, eval_configs, dataset, model, c_mean):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_opt()
        self._init_data(dataset)
        self._init_eval(eval_configs, c_mean)
        self._best_valid_loss = 1e10 

        self.tf_writer = SummaryWriter(log_dir=self.output_folder)

    def _init_eval(self, eval_configs, c_mean):
        dataset = EditDataset(eval_configs["data"])
        self.evaluator = BaseEvaluator(eval_configs["eval"], dataset, None)
        self.eval_configs = eval_configs
        self.c_mean = c_mean

    def _init_cfgs(self, configs):
        self.configs = configs
        
        self.n_epochs = self.configs["epochs"]
        self.itr_per_epoch = self.configs["itr_per_epoch"]
        self.valid_epoch_interval = self.configs["val_epoch_interval"]
        self.display_epoch_interval = self.configs["display_interval"]

        self.lr = self.configs["lr"]
        self.batch_size = self.configs["batch_size"]

        self.include_self = self.configs["include_self"]

        self.model_path = self.configs["model_path"]  # resume training
        self.output_folder = configs["output_folder"]
        
        os.makedirs(self.output_folder, exist_ok=True)

    def _init_model(self, model):
        self.model = model.to(model.device)
        if self.model_path != "":
            print("Laoding pretrained model from {}".format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path), strict=False)

    def _init_guider(self, guider):
        self.guider = guider.to(self.model.device)
        print("Laoding pretrained guider from {}".format(self.configs["guider_path"]))
        self.guider.load_state_dict(torch.load(self.model_path), strict=False)

    def _init_opt(self):
        self.opt = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)

    def _init_data(self, dataset):
        self.dataset = dataset
        self.train_loader = dataset.get_loader(split="train", batch_size=self.batch_size, shuffle=True, 
                                               include_self=self.include_self)
        self.valid_loader = dataset.get_loader(split="valid", batch_size=self.batch_size, shuffle=False, 
                                               include_self=self.include_self)

    def _reset_train(self):
        self._best_valid_loss = 1e10
        self._global_batch_no = 0

    """
    Train.
    """
    def train(self):
        self._reset_train()
        for epoch_no in range(self.n_epochs):
            self._train_epoch(epoch_no)
            if self.valid_loader is not None and (epoch_no + 1) % self.valid_epoch_interval == 0:
                self.valid(epoch_no)
                self.evaluate(epoch_no)
    
    def evaluate(self, epoch_no):
        metric_list = ["cos", "rats", "auc"]
        self.model.eval()
        self.evaluator.model = self.model
        
        res_dict = self.evaluator.evaluate(mode="cond_gen", sampler="ddim", save_pred=False, c_mean=self.c_mean)
        for k in res_dict.keys():
            record_flag = False
            for metric_name in metric_list:
                if metric_name in k:
                    record_flag = True
                    break
            if record_flag == True:
                self.tf_writer.add_scalar(fr"Cond_gen/{k}", res_dict[k], epoch_no)
        
        res_dict = self.evaluator.evaluate(mode="edit", sampler="ddim", save_pred=False, c_mean=self.c_mean)
        for k in res_dict.keys():
            record_flag = False
            for metric_name in metric_list:
                if metric_name in k:
                    record_flag = True
                    break
            if record_flag == True:
                self.tf_writer.add_scalar(fr"Edit50/{k}", res_dict[k], epoch_no)

    def _train_epoch(self, epoch_no):
            start_time = time.time()
            avg_loss = 0
            self.model.train()
            for batch_no, train_batch in enumerate(self.train_loader):
                self._global_batch_no += 1

                self.opt.zero_grad()
                loss = self.model(train_batch, is_train=True, mode="finetune")
                loss.backward()
                self.opt.step()
                avg_loss += loss.item()
                self.tf_writer.add_scalar("Finetune/Train/batch_loss", loss.item(), self._global_batch_no)

                if batch_no >= self.itr_per_epoch:
                    break

            avg_loss /= len(self.train_loader)
            self.tf_writer.add_scalar("Finetune/Train/epoch_loss", avg_loss, epoch_no)
            end_time = time.time()
            
            if (epoch_no+1)%self.display_epoch_interval==0:
                print("Epoch:", epoch_no,
                      "Loss:", avg_loss,
                      "Time: {:.2f}s".format(end_time-start_time))

    """
    Valid.
    """
    def valid(self, epoch_no=-1):
        self.model.eval()
        avg_loss_valid = 0
        with torch.no_grad():
            for batch_no, valid_batch in enumerate(self.valid_loader):
                loss = self.model(valid_batch, is_train=False, mode="finetune")
                avg_loss_valid += loss.item()

        avg_loss_valid = avg_loss_valid/len(self.valid_loader)

        self.tf_writer.add_scalar("Finetune/Valid/epoch_loss", avg_loss_valid, epoch_no)

        if self._best_valid_loss > avg_loss_valid:
            self._best_valid_loss = avg_loss_valid
            print(f"\n*** Best loss is updated to {avg_loss_valid} at {epoch_no}.\n")
            self.save_model(epoch_no)
    
    """
    Save.
    """
    def save_model(self, epoch_no):
        os.makedirs(fr"{self.output_folder}/ckpts", exist_ok=True)
        path = os.path.join(fr"{self.output_folder}/ckpts", "model_best.pth")
        torch.save(self.model.state_dict(), path)
        path = os.path.join(fr"{self.output_folder}/ckpts", fr"model_best_{epoch_no}.pth")
        torch.save(self.model.state_dict(), path)
