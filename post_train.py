import os
import logging

from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import BertPostTrainingDataset
from models.utils.checkpointing import CheckpointManager, load_checkpoint
from models import Model


class BERTDomainPostTraining(object):
  def __init__(self, hparams):
    self.hparams = hparams
    self._logger = logging.getLogger(__name__)


  def _build_dataloader(self):
    # =============================================================================
    #   SETUP DATASET, DATALOADER
    # =============================================================================
    self.train_dataset = BertPostTrainingDataset(self.hparams, split="train")
    self.train_dataloader = DataLoader(
      self.train_dataset,
      batch_size=32,
      num_workers=0,
      shuffle=False,
      drop_last=True
    )

    print("""
       # -------------------------------------------------------------------------
       #   DATALOADER FINISHED
       # -------------------------------------------------------------------------
       """)

  def _build_model(self):
    # =============================================================================
    #   MODEL : Standard, Mention Pooling, Entity Marker
    # =============================================================================
    print('\t* Building model...')

    self.model = Model(self.hparams)

    self.optimizer = optim.Adam(self.model.parameters(), lr=3e-5)
    self.iterations = len(self.train_dataset) // 512

    print(
      """
      # -------------------------------------------------------------------------
      #  Building Model Finished
      # -------------------------------------------------------------------------
      """
    )

  def _setup_training(self):
    if 'checkpoints/' == 'checkpoints/':
      self.save_dirpath = 'results/checkpoints'
    self.summary_writer = SummaryWriter(self.save_dirpath)
    self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, self.save_dirpath, hparams=self.hparams)

    if self.hparams.load_pthpath == "":
      self.start_epoch = 1
    else:
      self.start_epoch = int(self.hparams.load_pthpath.split("_")[-1][:-4])
      self.start_epoch += 1
      model_state_dict, optimizer_state_dict = load_checkpoint(self.hparams.load_pthpath)
      if isinstance(self.model, nn.DataParallel):
        self.model.module.load_state_dict(model_state_dict)
      else:
        self.model.load_state_dict(model_state_dict)
      self.optimizer.load_state_dict(optimizer_state_dict)
      self.previous_model_path = self.hparams.load_pthpath
      print("Loaded model from {}".format(self.hparams.load_pthpath))

    print(
      """
      # -------------------------------------------------------------------------
      #   Setup Training Finished
      # -------------------------------------------------------------------------
      """
    )

  def train(self):
    self._build_dataloader()
    self._build_model()
    self._setup_training()

    start_time = datetime.now().strftime('%H:%M:%S')
    self._logger.info("Start train model at %s" % start_time)

    train_begin = datetime.utcnow()
    global_iteration_step = 0
    accu_mlm_loss, accu_nsp_loss = 0, 0
    accumulate_batch, accu_count = 0, 0

    for epoch in range(self.start_epoch, 3):
      print("EPoch#", epoch)
      self.model.train()
      print("Done self.model.train()",)
      
      tqdm_batch_iterator = tqdm(self.train_dataloader)
      print("Done tqdm_batch_iterator",)
      for batch_idx, batch in enumerate(tqdm_batch_iterator):
        print("batch_idx:", batch_idx)
        
        buffer_batch = batch.copy()
        
        for key in batch:
          print("key:", key)
          buffer_batch[key] = buffer_batch[key]

        mlm_loss, nsp_loss = self.model(buffer_batch)
        total_loss = mlm_loss.mean() + nsp_loss.mean()
        total_loss.backward()
        accu_mlm_loss += mlm_loss.mean().item()
        accu_nsp_loss += nsp_loss.mean().item()
        accu_count += 1

        print("accu_count", accu_count)
        accumulate_batch += buffer_batch["next_sentence_labels"].shape[0]
        if self.hparams.virtual_batch_size == accumulate_batch \
            or batch_idx == (len(self.train_dataset) // self.hparams.train_batch_size): # last batch

          nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)

          self.optimizer.step()
          self.optimizer.zero_grad()

          global_iteration_step += 1
          description = "[{}][Epoch: {:3d}][Iter: {:6d}][MLM_Loss: {:6f}][NSP_Loss: {:6f}][lr: {:7f}]".format(
            datetime.utcnow() - train_begin,
            epoch,
            global_iteration_step, (accu_mlm_loss / accu_count), (accu_nsp_loss / accu_count),
            self.optimizer.param_groups[0]['lr'])
          
          if global_iteration_step % self.hparams.tensorboard_step == 0:
            self._logger.info(description)

          accumulate_batch, accu_count = 0, 0
          accu_mlm_loss, accu_nsp_loss = 0, 0

          if global_iteration_step % 1000 == 0:
            print('Saving to: ', self.save_dirpath)
            self.checkpoint_manager.step(global_iteration_step)
            self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath,
                                                    "checkpoint_%d.pth" % (global_iteration_step))
            self._logger.info(self.previous_model_path)
