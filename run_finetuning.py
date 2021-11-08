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
import transformers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datamodule import BartDataModule, SequenceGenerator





TRAIN_CONFIG_FILE = './configurations/train.yml'

def gen_checkpoint_id(args):
    model_id = args.MODEL_ID
    timez = datetime.datetime.now().strftime("%Y%m%d%H%M")
    checkpoint_id = "_".join([model_id, timez])
    return checkpoint_id

def get_logger(args):
    log_path = f"{args.checkpoint}/info"

    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    train_instance_log_files = os.listdir(log_path)
    train_instance_count = len(train_instance_log_files)


    # def timetz(*args):
    #     return datetime.now(tz).timetuple()

    # tz = timezone('Asia/Seoul') # UTC, Asia/Shanghai, Europe/Berlin

    # logging.Formatter.converter = timetz


    logging.basicConfig(
        filename=f'{args.checkpoint}/info/train_instance_{train_instance_count}_info.log',
        filemode='w',
        format="%(asctime)s | %(filename)15s | %(levelname)7s | %(funcName)10s | %(message)s",
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.info("-"*40)
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("-"*40)

    return logger

def checkpoint_count(checkpoint):
    _, folders, files = next(iter(os.walk(checkpoint)))
    files = list(filter(lambda x :"saved_checkpoint_" in x, files))

    checkpoints = map(lambda x: int(re.search(r"[0-9]{1,}", x).group()),files)

    try:
        last_checkpoint = sorted(checkpoints)[-1]
    except:
        last_checkpoint = 0
    return last_checkpoint


class Model(LightningModule):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def step(self, batch, batch_idx):
        output = self(**batch)

        loss = output.loss
        logits = output.logits

        return {
            'loss': loss
            #'y_true': y_true,
            #y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)
            
        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
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


from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filename='epoch{epoch}-val_acc{val_acc:.4f}',
    monitor='val_acc',
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
        precision=16 if args.setup.fp16 else 32,
        # For TPU Setup
        tpu_cores=args.setup.tpu_cores if args.setup.tpu_cores else None,
    )
    trainer.fit(model, datamodule)

if __name__ == '__main__':
    main()
     