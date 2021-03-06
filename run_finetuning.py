import os
import pandas as pd
import yaml
import re
from easydict import EasyDict
from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import transformers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datamodule import BartDataModule, SequenceGenerator
from ko_metric import Rouge
from hanbert_tokenizer import HanBert_Tokenizer


rouge_evaluator = Rouge(
    metrics=["rouge-n", "rouge-l"],
    max_n=2,
    limit_length=True,
    length_limit=1000,
    length_limit_type="words",
    use_tokenizer=True,
    apply_avg=True,
    apply_best=False,
    alpha=0.5,  # Default F1_score
    weight_factor=1.2,
)

TRAIN_CONFIG_FILE = './configurations/train.yml'

class Model(LightningModule):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)

        self.model = model
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        output = self(input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            labels=batch['labels'],
            decoder_attention_mask=batch['decoder_attention_mask']
        )

        loss = output.loss
        logits = output.logits

        return {
            'loss': loss
            #'y_true': y_true,
            #y_pred': y_pred,
        }

    def validation_step(self, batch, batch_idx):
        output = self(input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            labels=batch['labels'],
            decoder_attention_mask=batch['decoder_attention_mask']
        )

        loss = output.loss
        logits = output.logits

        repetition_penalty = 2.5
        length_penalty=1.0
        no_repeat_ngram_size=3

        pred_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=200, 
            num_beams=3,
            repetition_penalty=repetition_penalty, 
            length_penalty=length_penalty, 
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )
        
        
        decoded_pred = self.tokenizer.batch_decode(pred_ids, 
                          skip_special_tokens=True, 
                          clean_up_tokenization_spaces=True) 
                          

        rouge_l_f1 = rouge_evaluator.get_scores(decoded_pred, batch['abstractive'])['rouge-l']['f']
        rouge_l_r = rouge_evaluator.get_scores(decoded_pred, batch['abstractive'])['rouge-l']['r']
        rouge_l_p = rouge_evaluator.get_scores(decoded_pred, batch['abstractive'])['rouge-l']['p']


        return {
            'loss': loss,
            'rouge' : rouge_l_f1,
            'r_recall' : rouge_l_r,
            'r_precision' : rouge_l_p
        }

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)

        rouge = []
        r_recall = []
        r_precision = []
        
        for i in outputs:
            loss += i['loss'].cpu().detach()
            rouge.append(i['rouge'])
            r_recall.append(i['r_recall'])
            r_precision.append(i['r_precision'])

        loss = loss / len(outputs)
            
        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        if state=='val':
            rouge = sum(rouge) / len(rouge)
            r_recall = sum(r_recall) / len(r_recall)
            r_precision = sum(r_precision) / len(r_precision)

            self.log(state+'_rouge', float(rouge), on_epoch=True, prog_bar=True)
            self.log(state+'_r_recall', float(r_recall), on_epoch=True, prog_bar=True)
            self.log(state+'_r_precision', float(r_precision), on_epoch=True, prog_bar=True)


            
        return {'loss': loss}
    
    def train_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')

    def configure_optimizers(self):
        if self.hparams.optimizer_class == 'AdamW':
            optimizer_class = getattr(transformers, self.hparams.optimizer_class)
            optimizer = optimizer_class(self.parameters(), lr=float(self.hparams.lr))
        elif self.hparams.optimizer_class == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=float(self.hparams.lr))
        elif self.hparams.optimizer_class == 'Adafactor':
            optimizer_class = getattr(transformers, self.hparams.optimizer_class)
            optimizer = optimizer_class(model.parameters(), warmup_init=True)
        else:
            raise NotImplementedError('Only AdamW and AdamP and Adafactor is Supported!')
            
        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }



checkpoint_callback = ModelCheckpoint(
    filename='epoch{epoch}-val_rouge{val_rouge:.4f}',
    monitor='val_rouge',
    save_top_k=3,
    mode='max',
)        


def main():
    args = EasyDict(yaml.load(open(TRAIN_CONFIG_FILE).read(), Loader=yaml.Loader))

    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args.setup.random_seed)
    seed_everything(args.setup.random_seed)

    print("Call Dataset ", args.setup.train_data_path)

    model_class = getattr(transformers, args.setup.model_class)
    tokenizer_class = getattr(transformers, args.setup.tokenizer_class)

    model = model_class.from_pretrained(args.setup.model)
    tokenizer = tokenizer_class.from_pretrained(
        args.setup.tokenizer
        if args.setup.tokenizer
        else args.setup.model
    )

    datamodule = BartDataModule(args.setup.train_data_path, args.setup.val_data_path, tokenizer)
    datamodule.setup()

    
    model = Model(model, tokenizer, **args.hyperparameters)
    print(":: Start Training ::")
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args.setup.epochs,
        fast_dev_run=args.setup.test_mode,
        num_sanity_val_steps=None if args.setup.test_mode else 0,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
        #accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        precision=16 if args.setup.fp16 else 32,
        # For TPU Setup
        tpu_cores=args.setup.tpu_cores if args.setup.tpu_cores else None,
    )
    trainer.fit(model, datamodule)

if __name__ == '__main__':
    main()
     