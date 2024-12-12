
from cProfile import label
from dataclasses import replace
from src.callbacks import HOME_PATH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics 
import pickle
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from focal_loss.focal_loss import FocalLoss
#from imblearn.metrics import macro_averaged_mean_absolute_error
from scipy.stats import median_abs_deviation
from sklearn.metrics import f1_score, cohen_kappa_score
import logging

"""Custom code"""
from src.transformer.transformer_utils import *
from src.transformer.transformer import  AttentionDecoder, CLS_Decoder_FT2, Transformer
from pathlib import Path
import os

HOME_PATH = str(Path.home())
log = logging.getLogger(__name__)
REG_LOSS = ["mae", "mse", "smooth"]
CLS_LOSS = ["entropy", "focal", "ordinal", "corn", "cdw", "nll_loss"]

class Transformer_CLS(pl.LightningModule):
    """Transformer with Classification Layer"""

    def __init__(self, hparams):
        super(Transformer_CLS, self).__init__()
        self.hparams.update(hparams)


        # 1. Transformer: Load pretrained encoders
        self.init_encoder()
        # 2. Decoder
        self.init_decoder()
        # 3. Loss
        self.init_loss()
        # 4. Metric
        self.init_metrics()
        self.init_collector()
        # 5. Embeddings
        self.embedding_grad()
        # x. Logging params
        self.train_step_targets = []
        self.train_step_predictions = []
        self.last_update = 0
        self.last_global_step = 0

    @property
    def num_outputs(self):
        return self.hparams.num_targets

    def init_encoder(self):
        self.transformer = Transformer(self.hparams)
        log.info("Embedding sample before load: %.2f" %self.transformer.embedding.token.weight[1, 0].detach())
        if "none" in self.hparams.pretrained_model_path:
            log.info("No pretrained model")
        else:
            log.info("Pretrained Model Path:\n\t%s" %(HOME_PATH + self.hparams.pretrained_model_path))
            self.transformer.load_state_dict(
                torch.load(HOME_PATH + self.hparams.pretrained_model_path, map_location=self.device), strict=False
            )
        log.info("Embedding sample after load: %.2f" %self.transformer.embedding.token.weight[1, 0].detach())

    def init_decoder(self):
        if self.hparams.pooled: 
            log.info("Model with the POOLED representation")
            self.decoder = AttentionDecoder(self.hparams, num_outputs=self.num_outputs)
            self.encoder_f = self.transformer.forward_finetuning
        else: 
            log.info("Model with the CLS representation")
            self.decoder = CLS_Decoder_FT2(self.hparams)
            self.encoder_f = self.transformer.forward_finetuning_cls

    def init_loss(self):
    
        if self.hparams.loss_type == "robust":
            raise NotImplementedError("Deprecated: use asymmetric loss instead")
        elif self.hparams.loss_type == "asymmetric":
            self.loss = AsymmetricCrossEntropyLoss(pos_weight=self.hparams.pos_weight)
        elif self.hparams.loss_type == "asymmetric_dynamic":
            raise NotImplementedError
            #self.loss = AsymmetricCrossEntropyLoss(pos_weight=self.hparams.pos_weight, penalty=self.hparams.asym_penalty)
        elif self.hparams.loss_type == "entropy":
            if hasattr(self.hparams, 'class_weights'):
                self.register_buffer("class_weights", torch.tensor(self.hparams.class_weights))
                #class_weights = torch.tensor(self.hparams.class_weights).to(self.device)
                self.loss = nn.CrossEntropyLoss(weight=self.class_weights) #change also in rnn/ffn
            else:
                self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplemented

    def embedding_grad(self):
        ### Remove parameters from the computational graph
        if self.hparams.freeze_positions:
            for param in self.transformer.embedding.age.parameters():
                param.requires_grad = False
            for param in self.transformer.embedding.abspos.parameters():
                param.requires_grad = False
            for param in self.transformer.embedding.segment.parameters():
                param.requires_grad = False

    def forward(self, batch):
        """Forward pass"""
        ## 1. ENCODER INPUT
        predicted = self.encoder_f(
        x=batch["input_ids"].long(),
        padding_mask=batch["padding_mask"].long()
        )
        ## 2. CLS Predictions 
        if self.hparams.pooled: 
            predicted = self.decoder(predicted, mask = batch["padding_mask"].long())
        else:
            predicted = self.decoder(predicted)

        return predicted 

    
    def forward_with_embeddings(self, embeddings, meta = None):
        """Same as forward, but takes pre-calculated embeddings instead of tokens"""
        ## 2. CLS Predictions 
        if self.hparams.pooled:
            predicted = self.transformer.forward_finetuning_with_embeddings(
            x=embeddings,
            padding_mask=meta["padding_mask"].long()
            )
            predicted = self.decoder(predicted, meta["padding_mask"].long())
        
        else:
            predicted = self.transformer.forward_finetuning_with_embeddings_cls(
            x=embeddings,
            padding_mask=meta["padding_mask"].long()
            )
            predicted = self.decoder(predicted)
        return predicted         

    def init_metrics(self):
        """Initialise variables to store metrics"""
        
        task = 'binary' if self.hparams.num_targets == 2 else 'multiclass'
        ### TRAIN
        self.train_accuracy = torchmetrics.Accuracy(threshold=0.5, task=task, num_classes=self.hparams.cls_num_targets, average="macro")
        self.train_precision = torchmetrics.Precision(threshold=0.5, task=task, num_classes=self.hparams.cls_num_targets, average="macro")
        self.train_recall = torchmetrics.Recall(threshold=0.5, task=task, num_classes=self.hparams.cls_num_targets, average="macro")
        self.train_f1 = torchmetrics.F1Score(threshold=0.5, task=task, num_classes=self.hparams.cls_num_targets, average="macro")
        
         ##### VALIDATION
        self.val_accuracy = torchmetrics.Accuracy(threshold=0.5, task=task, num_classes=self.hparams.num_targets, average=self.hparams.average_type)
        self.val_precision = torchmetrics.Precision(threshold=0.5, task=task, num_classes=self.hparams.num_targets, average=self.hparams.average_type)
        self.val_recall = torchmetrics.Recall(threshold=0.5, task=task, num_classes=self.hparams.num_targets, average=self.hparams.average_type)
        self.val_f1 = torchmetrics.F1Score(threshold=0.5, task=task, num_classes=self.hparams.num_targets, average=self.hparams.average_type)
        self.val_auc = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_targets, average="macro")

        ##### TEST
        self.test_accuracy = torchmetrics.Accuracy(threshold=0.5, task=task, num_classes=self.hparams.num_targets, average=self.hparams.average_type)
        self.test_precision = torchmetrics.Precision(threshold=0.5, task=task, num_classes=self.hparams.num_targets, average=self.hparams.average_type)
        self.test_recall = torchmetrics.Recall(threshold=0.5, task=task, num_classes=self.hparams.num_targets, average=self.hparams.average_type)
        self.test_f1 = torchmetrics.F1Score(threshold=0.5, task=task, num_classes=self.hparams.num_targets, average=self.hparams.average_type)
        self.test_auc = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_targets, average="macro")


    def init_collector(self):
        """Collect predictions and targets"""
        self.test_trg = torchmetrics.CatMetric()
        self.test_prb = torchmetrics.CatMetric()
        self.test_id1  = torchmetrics.CatMetric()
        self.test_id2  = torchmetrics.CatMetric()
        self.val_trg = torchmetrics.CatMetric()
        self.val_prb = torchmetrics.CatMetric()
        self.val_id  = torchmetrics.CatMetric()

    def on_train_epoch_start(self, *args):
        """"""
        seed_everything(self.hparams.seed + self.trainer.current_epoch)
    #    log.info("Redraw Projection Matrices")

    def on_train_epoch_end(self, *args):
        if self.hparams.attention_type == "performer":
            log.info("Redraw Projection Matrices")
            self.transformer.redraw_projection_matrix(-1)


    def transform_targets(self, targets, seq, stage: str):
        """Transform Tensor of targets based on the type of loss"""
        if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
            targets = F.one_hot(targets.long(), num_classes = self.hparams.num_targets) ## for custom cross entropy we need to encode targets into one hot representation
        elif self.hparams.loss_type == "entropy":
           targets = targets.long()
        else:
            raise NotImplemented
        return targets

    def training_step(self, batch, batch_idx):
        """Training Iteration"""
        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"],  seq = batch["input_ids"], stage="train")

        loss = self.loss(predicted, targets)
        ## 3. METRICS
        self.train_step_predictions.append(predicted.detach())
        self.train_step_targets.append(targets.detach())

        if ((self.global_step + 1) % (self.trainer.log_every_n_steps) == 0) and (
            self.last_update != self.global_step
        ):
            self.last_update = self.global_step
            self.log_metrics(
                predictions=torch.cat(self.train_step_predictions),
                targets=torch.cat(self.train_step_targets),
                loss=loss.detach(),
                stage="train",
                on_step=True,
                on_epoch=True,
            )
            del self.train_step_predictions
            del self.train_step_targets
            self.train_step_targets = []
            self.train_step_predictions = []

        ## 5. RETURN LOSS
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation Step"""

        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"], seq = batch["input_ids"], stage="val")
        loss = self.loss(predicted, targets)

        ## 3. METRICS
        self.log_metrics(
            predictions=predicted.detach(),
            targets=targets.detach(),
            loss=loss.detach(),
            stage="val",
            on_step=False,
            on_epoch=True,
            sid = batch["sequence_id"]
        )

        return None

    def test_step(self, batch, batch_idx):
        """Test and collect stats"""
        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"],  seq = batch["input_ids"], stage="test")
        loss = self.loss(predicted, targets)
        ## 3. METRICS
        self.log_metrics(predictions = predicted.detach(), targets = targets.detach(), loss = loss.detach(), sid = batch["sequence_id"], stage = "test", on_step = False, on_epoch = True)
        return None

    def on_after_backward(self) -> None:
        #### Freeze Embedding Layer, except for the CLS token
        if self.hparams.parametrize_emb:
             w = self.transformer.embedding.token.parametrizations.weight.original
        else:
            w = self.transformer.embedding.token.weight
        if self.hparams.freeze_embeddings:
            w.grad[0] = 0
            w.grad[2:8] = 0
            w.grad[10:] = 0
        return super().on_after_backward()


    def debug_optimizer(self):
        print("==========================")
        print("PARAMETERS IN OPTIMIZER")
        print("\tGROUP 1:")
        for n, p in self.named_parameters():
            if "embedding" in n:
                print("\t\t", n)
        print("\tGROUP 2:")
        for i in range(0, self.hparams.n_encoders):
            for n, p in self.named_parameters():
                if "encoders.%s." % i in n:
                    print("\t\t", n)
        print("\tGROUP 3:")
        for n, p in self.named_parameters():
            if "decoder" in n:
                print("\t\t", n)
   
    def configure_optimizers(self):
        """"""
        self.debug_optimizer()
        no_decay = [ 
           "bias",
           "norm"
        ]
        optimizer_grouped_parameters = list()
        lr = self.hparams.learning_rate * (self.hparams.layer_lr_decay**self.hparams.n_encoders)
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if ("embedding" in n) 
                ] ,
                "weight_decay": 0.0,
                "lr": lr,
            }
        )
        for i in range(0, self.hparams.n_encoders):
            lr = self.hparams.learning_rate * (
                (self.hparams.layer_lr_decay) ** (self.hparams.n_encoders - i)
            )  # lwoer layers should have lower learning rate

            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if ("encoders.%s." % i in n) and not (nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": lr,
                }
            )
            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if ("encoders.%s." % i in n) and  (nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                }
            )
            
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in self.named_parameters() if ("decoder" in n)],
                "weight_decay": self.hparams.weight_decay_dc,
                "lr": self.hparams.learning_rate,
            }
        )
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in self.named_parameters() if ("target_weight" in n) or ("loss_weight" in n)],
                "weight_decay": 0.0,
                "lr": 0.01,
            }
        )
        

        if self.hparams.optimizer_type == "radam":
            optimizer = torch.optim.RAdam(
                optimizer_grouped_parameters,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=self.hparams.epsilon,
            )
        elif self.hparams.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=self.hparams.epsilon,
            )
        elif self.hparams.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                momentum=0.0,
                dampening=0.00)
        elif self.hparams.optimizer_type == "asgd":
            optimizer = torch.optim.ASGD(
                optimizer_grouped_parameters,
                t0=3000)
        elif self.hparams.optimizer_type == "adamax":
            optimizer = torch.optim.Adamax(
                optimizer_grouped_parameters,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=self.hparams.epsilon,
            )
        # no sgd , adsgd, nadam, rmsprop
        else:
            raise NotImplementedError

        if self.hparams.stage in ["search", "finetuning"]:
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.hparams.lr_gamma
                ), 
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            }
        }


    def log_metrics(self, predictions, targets, loss, stage, on_step: bool = True, on_epoch: bool = True, sid = None):             
        """Compute on step/epoch metrics"""
        assert stage in ["train", "val", "test"]
        scores = F.softmax(predictions, dim=1)
        preds = scores[:,1].flatten()
        
        if stage == "train":
            self.log("train/loss", loss, on_step=on_step, on_epoch = on_epoch)

            self.log("train/pos_samples", torch.sum(targets)/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)

            self.log("train/accuracy", self.train_accuracy(preds, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/recall", self.train_recall(preds, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/precision", self.train_precision(preds, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/f1", self.train_f1(preds, targets), on_step=on_step, on_epoch = on_epoch)

        elif stage == "val":
            self.log("val/loss", loss, on_step=on_step, on_epoch = on_epoch)

            self.log("val/pos_samples", torch.sum(targets)/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)   
            
            self.val_f1.update(preds, targets) # this should not happen in self.log 
            self.val_accuracy.update(preds, targets) # this should not happen in self.log 
            self.val_precision.update(preds, targets)  # this should not happen in self.log 
            self.val_recall.update(preds, targets) # this should not happen in self.log
            self.val_auc.update(preds, targets) # this should not happen in self.log 
            
            self.log("val/f1", self.val_f1, on_step = False, on_epoch = True)
            self.log("val/acc", self.val_accuracy, on_step = False, on_epoch = True)
            self.log("val/precision", self.val_precision, on_step = False, on_epoch = True)
            self.log("val/recall", self.val_recall, on_step = False, on_epoch = True)
            self.log("val/auc", self.val_auc, on_step = False, on_epoch = True)

            self.val_trg.update(targets)
            self.val_prb.update(preds)
            self.val_id.update(sid)

        elif stage == "test":
            self.log("test/loss", loss, on_step=on_step, on_epoch = on_epoch)
            
            self.test_f1.update(preds, targets) # this should not happen in self.log 
            self.test_accuracy.update(preds, targets) # this should not happen in self.log 
            self.test_precision.update(preds, targets)  # this should not happen in self.log 
            self.test_recall.update(preds, targets) # this should not happen in self.log
            self.test_auc.update(preds, targets) # this should not happen in self.log 
            
            self.log("test/f1", self.test_f1, on_step = False, on_epoch = True)
            self.log("test/acc", self.test_accuracy, on_step = False, on_epoch = True)
            self.log("test/precision", self.test_precision, on_step = False, on_epoch = True)
            self.log("test/recall", self.test_recall, on_step = False, on_epoch = True)
            self.log("test/auc", self.test_auc, on_step = False, on_epoch = True)
            

            self.test_trg.update(targets)
            self.test_prb.update(preds)

            # Convert to 8-digit strings
            if isinstance(sid, torch.Tensor):
                sid = sid.cpu().numpy()
            str_ids = [f"{int(x):08d}" for x in sid]
            
            # Split into first/second half
            first_half = [int(x[:4]) for x in str_ids]
            second_half = [int(x[4:]) for x in str_ids]
            
            # Update both metrics
            self.test_id1.update(torch.tensor(first_half))
            self.test_id2.update(torch.tensor(second_half))
        

        @staticmethod
        def load_lookup(path):
            """Load Token-to-Index and Index-to-Token dictionary"""
            with open(path, "rb") as f:
                indx2token, token2indx = pickle.load(f)
            return indx2token, token2indx


