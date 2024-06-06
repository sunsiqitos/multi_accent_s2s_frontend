import os
import sys
import time
import traceback
import logging
import argparse
import numpy as np
from tqdm import tqdm
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import setup_model
from datasets import FestivalDataset
from scheduled import ScheduledSampler

from generic_utils import (
    load_config, create_experiment_folder, remove_experiment_folder, copy_config_file,
    save_checkpoint, load_checkpoint, build_vocab, count_parameters,
    check_update, get_current_lr
)
from params import set_default_params
from visual import plot_alignment, plot_pca, plot_tsne
from inference import inference

torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def tensor2numpy(tensor):
    return tensor.cpu().detach().numpy()


def tensor2idxes(tensor):
    idx_array = np.argmax(
        tensor.cpu().detach().numpy(),
        axis=-1)
    return idx_array.tolist()


def setup_input_data(c, data):
    char_inputs, char_lengths, phone_inputs, phone_lengths, lang_inputs, stop_targets = data
    # set stop targets view, we predict a single stop token per r frames prediction
    stop_targets = stop_targets.view(
        stop_targets.size(0),
        stop_targets.size(1) // c.r,
        -1)
    stop_targets = (stop_targets.sum(-1) > 0.0).float()

    if use_cuda:
        char_inputs = char_inputs.cuda(non_blocking=True)
        char_lengths = char_lengths.cuda(non_blocking=True)
        phone_inputs = phone_inputs.cuda(non_blocking=True)
        phone_lengths = phone_lengths.cuda(non_blocking=True)
        lang_inputs = lang_inputs.cuda(non_blocking=True)
        stop_targets = stop_targets.cuda(non_blocking=True)

    return char_inputs, char_lengths, phone_inputs, phone_lengths, lang_inputs, stop_targets


class Train(object):
    """
     Trainer for TTS frontend
    """

    def __init__(self, args):
        super(Train, self).__init__()

        self.c = load_config(args.config_path)
        self.c = set_default_params(self.c)

        self.out_dir = create_experiment_folder(args.output_path, self.c.experiment_name)

        # copy config file to the experiment folder
        new_fields = None
        if args.restore_path:
            new_fields = {}
            new_fields["restore_path"] = args.restore_path
        copy_config_file(args.config_path, os.path.join(self.out_dir, 'config.json'), new_fields)

        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(self.out_dir, self.c.experiment_name + '.log'),
            filemode='w')

        # define a new Handler for logging in tqdm
        tqdmsg = TqdmLoggingHandler()
        tqdmsg.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        tqdmsg.setFormatter(formatter)
        logging.getLogger('').addHandler(tqdmsg)

        logging.info(' > Using CUDA: {}'.format(use_cuda))

        src_vocab_size, tgt_vocab_size = self.setup_vocab()

        self.sampler = self.setup_sampler()

        self.model = self.setup_model(src_vocab_size, tgt_vocab_size)

        if use_cuda:
            self.model = self.model.cuda()

        self.optimizer, self.optimizer_st = self.setup_optimizer()

        self.start_epoch, self.global_step = self.restore_checkpoint(args.restore_path)

        self.writer = self.setup_writer()

    def setup_vocab(self):
        logging.info(' > Setting up the vocabulary:')

        src_vocab_size, tgt_vocab_size = build_vocab(
            data_dict=self.c.data,
            src_vocab_fpath=os.path.join(self.out_dir, self.c.src_vocab),
            tgt_vocab_fpath=os.path.join(self.out_dir, self.c.tgt_vocab),
            word_vocab_fpath=os.path.join(self.out_dir, self.c.word_vocab)
        )

        if self.c.verbose:
            logging.info(' | > Number of languages: {}'.format(len(self.c.data)))
            logging.info(' | > Writing source vocab to {}'.format(os.path.join(self.out_dir, self.c.src_vocab)))
            logging.info(' | > Writing target vocab to {}'.format(os.path.join(self.out_dir, self.c.tgt_vocab)))
            logging.info(' | > Writing source word vocab to {}'.format(os.path.join(self.out_dir, self.c.word_vocab)))

            logging.info(' | > Source vocab size: {}'.format(src_vocab_size))
            logging.info(' | > Target vocab size: {}'.format(tgt_vocab_size))

        return src_vocab_size, tgt_vocab_size

    def setup_sampler(self):
        logging.info(' > Setting up the scheduled sampler:')

        sampler = ScheduledSampler(**self.c.scheduled_sampling)

        if self.c.verbose:
            for key, value in self.c.scheduled_sampling.items():
                logging.info(" | > {}:{}".format(key, value))

        return sampler

    def setup_model(self, src_vocab_size, tgt_vocab_size):
        logging.info(' > Setting up the model:')

        model = setup_model(self.c, src_vocab_size, tgt_vocab_size)

        if self.c.verbose:
            logging.info(' | > Encoder embedding dimension: {}'.format(self.c.enc_embedding_dim))
            logging.info(' | > Decoder embedding dimension: {}'.format(self.c.dec_embedding_dim))
            logging.info(' | > Language embedding dimension: {}'.format(self.c.lang_embedding_dim))
            logging.info(' | > Encoder hidden dimension: {}'.format(self.c.enc_hidden_dim))
            logging.info(' | > Decoder hidden dimension: {}'.format(self.c.dec_hidden_dim))
            logging.info(' | > Postnet hidden dimension: {}'.format(self.c.post_hidden_dim))
            logging.info(' | > Has prenet: {}'.format(self.c.has_prenet))
            logging.info(' | > Dropout: {}'.format(self.c.dropout))
            logging.info(' | > Decoder reduction factor: {}'.format(self.c.r))
            logging.info(' | > Attention type: {}'.format(self.c.attn_type))
            if self.c.attn_type == 'gmm':
                logging.info(' | > Number of mixtures: {}'.format(self.c.num_mixtures))
                logging.info(' | > GMM version: {}'.format(self.c.gmm_version))
            logging.info(' | > Has postnet: {}'.format(self.c.has_postnet))
            logging.info(' | > Has stopnet: {}'.format(self.c.has_stopnet))

        return model

    def setup_optimizer(self):
        logging.info(' > Setting up the optimizer:')

        optimizer = optim.Adam(self.model.parameters(), lr=self.c.learning_rate)
        if self.c.has_stopnet and self.c.separate_stopnet:
            optimizer_st = optim.Adam(self.model.decoder.stopnet.parameters(), lr=self.c.learning_rate)
        else:
            optimizer_st = None

        if self.c.verbose:
            logging.info(' | > Separate stopnet optimizer: {}'.format(self.c.separate_stopnet))
            logging.info(' | > Initial learning rate: {} '.format(self.c.learning_rate))
            logging.info(' | > Gradient clip threshold: {}'.format(self.c.clip_thresh))

        return optimizer, optimizer_st

    def restore_checkpoint(self, restore_path):
        start_epoch, global_step = load_checkpoint(self.model, self.optimizer, self.optimizer_st, restore_path)
        if global_step != 0:
            logging.info(' | > Checkpoint {} is loaded, resuming from epoch {} step {},'.format(restore_path, start_epoch, global_step))
        return start_epoch, global_step

    def setup_writer(self):
        logging.info(' > Setting up the tensorboard logger:')

        writer = SummaryWriter(os.path.join(self.out_dir, 'tb_log'))

        if self.c.verbose:
            logging.info(' | > Tensorboard output dir: {} '.format(os.path.join(self.out_dir, 'tb_log')))

        return writer

    def setup_dataset(self, mode, verbose=True):
        logging.info(' > Setting up the {} datasets:'.format(mode))

        dataset = FestivalDataset(
            mode=mode,
            data_dict=self.c.data,
            src_vocab_fpath=os.path.join(self.out_dir, self.c.src_vocab),
            tgt_vocab_fpath=os.path.join(self.out_dir, self.c.tgt_vocab),
            r=self.c.r,
            batch_group_size=self.c.batch_group_size,
            over_sampling=self.c.over_sampling,
            min_seq_len=self.c.min_seq_len,
            max_seq_len=self.c.max_seq_len,
            verbose=verbose)

        if self.c.verbose:
            logging.info(' | > Batch size: {} '.format(self.c.batch_size))
            logging.info(' | > Shuffle batch group size: {} '.format(self.c.batch_group_size))
            logging.info(' | > Oversampling: {} '.format(self.c.over_sampling))
            logging.info(' | > Minium sequence length: {} '.format(self.c.min_seq_len))
            logging.info(' | > Maxium sequence length: {} '.format(self.c.max_seq_len))

        return dataset

    def setup_loader(self, dataset, shuffle=True):
        dataset.shuffle_items(shuffle)
        data_loader = DataLoader(
            dataset,
            batch_size=self.c.batch_size,
            num_workers=self.c.load_data_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True)
        return data_loader

    def run(self):
        logging.info(' > Training begin:')
        num_params, num_trainable_params = count_parameters(self.model)
        logging.info(" | > Model has {} parameters, {} trainable".format(num_params, num_trainable_params))

        best_loss = float('inf')

        self.train_dataset = self.setup_dataset(mode='train')
        self.val_dataset = self.setup_dataset(mode='val')

        current_step = self.global_step
        val_loss = self.evaluate(current_step, 0)
        self.run_inference(current_step)

        for epoch in range(self.start_epoch, self.start_epoch + self.c.epochs):
            train_loss, current_step = self.train(current_step, epoch)
            val_loss = self.evaluate(current_step, epoch)
            logging.info(" | > Training Loss: {:.5f} Validation Loss: {:.5f}".format(train_loss, val_loss))
            if val_loss < best_loss:
                save_checkpoint(self.model, self.optimizer, self.optimizer_st,
                                current_step, epoch, self.out_dir, val_loss, best=True)
                best_loss = val_loss

    def evaluate(self, current_step, epoch):
        data_loader = self.setup_loader(dataset=self.val_dataset)
        self.model.eval()
        logging.info(" > Validation")
        with torch.no_grad():
            avg_val_loss, current_step = self.run_epoch(data_loader, epoch, current_step, mode='val')
        return avg_val_loss

    def train(self, current_step, epoch):
        data_loader = self.setup_loader(dataset=self.train_dataset)
        self.model.train()
        logging.info(" > Train")
        avg_train_loss, current_step = self.run_epoch(data_loader, epoch, current_step, mode='train')
        return avg_train_loss, current_step

    def run_epoch(self, data_loader, epoch, current_step, mode):
        is_train = True if mode == 'train' else False
        dec_losses, post_losses, stop_losses = [], [], []
        step_times = []

        logging.info(" > Epoch {}/{}".format(epoch, self.c.epochs))
        pbar = tqdm(data_loader)
        for num_iter, data in enumerate(pbar):
            if is_train:
                pbar.set_description(" | > Training at epoch {}/{}".format(epoch, self.c.epochs))
            else:
                pbar.set_description(" | > Validating at epoch {}/{}".format(epoch, self.c.epochs))
            start_time = time.time()

            # Setup input data
            char_inputs, char_lengths, phone_inputs, phone_lengths, \
                lang_inputs, stop_targets = setup_input_data(self.c, data)

            if is_train:
                current_step += 1

                self.optimizer.zero_grad()
                if self.optimizer_st:
                    self.optimizer_st.zero_grad()

            pbar.set_postfix(Global_step=current_step)

            if self.sampler:
                self.sampler.update_ratio(current_step)

            # Forward pass model and loss computation
            dec_outputs, post_outputs, alignments, stop_tokens = \
                self.model(char_inputs, char_lengths, phone_inputs, lang_inputs, self.sampler)

            loss, stop_loss, dec_loss, post_loss = self.compute_loss(
                dec_outputs=dec_outputs,
                post_outputs=post_outputs,
                stop_outputs=stop_tokens,
                phone_targets=phone_inputs,
                stop_targets=stop_targets)

            if is_train:
                # Compute grad and backward pass
                loss.backward(retain_graph=True)
                current_lr = get_current_lr(self.optimizer)
                grad_norm, skip_flag = check_update(self.model, self.c.clip_thresh, self.c.check_grad_norm)
                if not skip_flag:
                    self.optimizer.step()

                # Backward pass and check the grad norm for stop loss
                grad_norm_st = 0
                if self.c.has_stopnet and self.c.separate_stopnet:
                    stop_loss.backward(retain_graph=True)
                    grad_norm_st, skip_flag = check_update(self.model.decoder.stopnet, 1.0, self.c.check_grad_norm)
                    if not skip_flag:
                        self.optimizer_st.step()

            step_time = time.time() - start_time

            self.log_iter_stats(
                current_step,
                is_train,
                grad_norm=grad_norm if is_train else None,
                grad_norm_st=grad_norm_st if is_train else None,
                current_lr=current_lr if is_train else None,
                loss=loss.item(),
                post_loss=post_loss.item(),
                dec_loss=dec_loss.item(),
                stop_loss=stop_loss.item(),
                avg_char_length=torch.mean(char_lengths.float()).item(),
                avg_phone_length=torch.mean(phone_lengths.float()).item(),
                step_time=step_time,
                sampler_ratio=self.sampler.ratio)

            post_losses.append(post_loss.item())
            dec_losses.append(dec_loss.item())
            stop_losses.append(stop_loss.item())
            step_times.append(step_time)

            # Save checkpoint and tensorboard log the string
            if current_step % self.c.save_step == 0 and is_train:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.optimizer_st,
                    current_step,
                    epoch,
                    self.out_dir,
                    post_loss.item() if self.c.has_postnet else dec_loss.item(),
                    best=False)
                logging.info(" | > Checkpoint saving : {}".format(current_step))
                self.run_inference(current_step)
            # The end of the epoch

        self.log_epoch_stats(
            current_step,
            is_train,
            dec_losses=dec_losses,
            post_losses=post_losses,
            stop_losses=stop_losses,
            step_times=step_times,
            alignments=alignments,
            char_inputs=char_inputs,
            phone_inputs=phone_inputs,
            phone_outputs=post_outputs if self.c.has_postnet else dec_outputs,
            lang_inputs=lang_inputs)

        pbar.close()

        return np.mean(post_losses if self.c.has_postnet else dec_losses), current_step

    def compute_loss(self, dec_outputs, post_outputs, stop_outputs, phone_targets, stop_targets):
        criterion = nn.CrossEntropyLoss()
        criterion_st = nn.BCEWithLogitsLoss()

        # CrossEntropyLoss(input, target) requires input: (B, C, T), target: (B, T)
        dec_loss = criterion(dec_outputs.transpose(1, 2), phone_targets)
        post_loss = criterion(post_outputs.transpose(1, 2), phone_targets) if self.c.has_postnet else torch.tensor(0.)
        loss = dec_loss + post_loss

        stop_loss = criterion_st(stop_outputs, stop_targets) if self.c.has_stopnet else torch.tensor(0.)

        if self.c.has_stopnet and not self.c.separate_stopnet:
            loss += stop_loss

        return loss, stop_loss, dec_loss, post_loss

    def run_inference(self, current_step):
        lang_str = random.choice(list(self.c.data.keys()))  # randomly choose a language name

        lang_idx = self.train_dataset.lang2idx[lang_str]
        fpath_dict = self.c.data[lang_str]['test']

        with open(fpath_dict["path_src"]) as f:
            lines = f.readlines()
            offset = random.randrange(len(lines))  # randomly choose an offset for inference
            char_str = lines[offset].rstrip()

        with open(fpath_dict["path_tgt"]) as f:
            lines = f.readlines()
            phone_str = lines[offset].rstrip()

        self.model.eval()
        logging.info(" > Infering test sentences with {} and offset={}".format(lang_str, offset))

        with torch.no_grad():
            phone_eos_str_list, phone_str_list, alignment_list, _, _ = inference(
                model=self.model,
                text_fpath=fpath_dict["path_src"],
                c=self.c,
                use_cuda=use_cuda,
                lang_idx=lang_idx,
                char2idx=self.train_dataset.char2idx,
                idx2phone=self.train_dataset.idx2phone,
                batch_size=1,
                max_decoder_steps=self.c.max_decoder_steps,
                decoding_method='greedy',
                begin=offset,
                end=offset + 1
            )

            figs = {
                'alignment': plot_alignment(alignment_list[0])
            }
            self.tb_log(
                'Test',
                figs,
                current_step,
                method_name='add_figure')

            texts = {
                'char_input': char_str,
                'phone_gt': phone_str,
                'phone_eos_output': phone_eos_str_list[0],
                'phone_output': phone_str_list[0],
                'lang_input': lang_str
            }
            self.tb_log(
                'Test',
                texts,
                current_step,
                method_name='add_text')

        self.model.train()

    def sample2text(self, char_input, phone_input, phone_output, lang_input):
        char_input = [self.train_dataset.idx2char[ch] for ch in tensor2numpy(char_input)]
        phone_input = [self.train_dataset.idx2phone[ph] for ph in tensor2numpy(phone_input)]
        phone_output = [self.train_dataset.idx2phone[ph] for ph in tensor2idxes(phone_output)]
        lang_input = self.train_dataset.idx2lang[lang_input.item()]

        return ' '.join(char_input), ' '.join(phone_input), ' '.join(phone_output), lang_input

    def tb_log(self, scope, log_dict, step, method_name='add_scalar'):
        method = getattr(self.writer, method_name)
        for key, value in log_dict.items():
            method('{}/{}'.format(scope, key), value, step)

    def log_iter_stats(self, current_step, is_train, **kwargs):
        # Print step info
        if current_step % self.c.print_step == 0:
            if is_train:
                logging.info(" | > GlobalStep:{} GradNorm:{:.5f} GradNormST:{:.5f} LR:{:.6f}".format(
                    current_step, kwargs['grad_norm'], kwargs['grad_norm_st'], kwargs['current_lr']))
            logging.info(
                " | > TotalLoss:{:.5f}  PostnetLoss:{:.5f}  DecoderLoss:{:.5f}  StopLoss:{:.5f}  "
                "AvgCharLen:{:.1f}  AvgPhoneLen:{:.1f}  StepTime:{:.2f}".format(
                    kwargs['loss'], kwargs['post_loss'], kwargs['dec_loss'], kwargs['stop_loss'],
                    kwargs['avg_char_length'], kwargs['avg_phone_length'], kwargs['step_time']))

        # Tensorboard log iteration stats
        if is_train:
            stats = {
                'loss_decoder': kwargs['dec_loss'],
                'loss_postnet': kwargs['post_loss'],
                'loss_stopnet': kwargs['stop_loss'],
                'grad_norm': kwargs['grad_norm'],
                'grad_norm_st': kwargs['grad_norm_st'],
                'step_time': kwargs['step_time'],
                'learning_rate': kwargs['current_lr'],
                'scheduled_sampling': kwargs['sampler_ratio']
            }
            self.tb_log('TrainIter', stats, current_step, method_name='add_scalar')

    def log_epoch_stats(self, current_step, is_train, **kwargs):
        # Print epoch info
        logging.info(
            "   | > EPOCH END -- GlobalStep:{}  AvgDecoderLoss:{:.5f}  AvgPostnetLoss:{:.5f}  "
            "AvgStopLoss:{:.5f}  EpochTime:{:.2f}  AvgStepTime:{:.2f}".format(
                current_step, np.mean(kwargs['dec_losses']), np.mean(kwargs['post_losses']),
                np.mean(kwargs['stop_losses']), np.sum(kwargs['step_times']), np.mean(kwargs['step_times'])))

        # Tensorboard log epoch stats
        stats = {
            'loss_decoder': np.mean(kwargs['dec_losses']),
            'loss_postnet': np.mean(kwargs['post_losses']),
            'loss_stopnet': np.mean(kwargs['stop_losses']),
            'loss': np.mean(kwargs['post_losses']) if self.c.has_postnet else np.mean(kwargs['dec_losses']),
            'epoch_time': np.sum(kwargs['step_times'])
        }
        self.tb_log(
            'TrainEpoch' if is_train else 'ValEpoch',
            stats,
            current_step,
            method_name='add_scalar')

        align_img = kwargs['alignments'][0].data.cpu().numpy()  # [# dec steps, # enc steps]
        phone_embeddings = self.model.decoder.embedding.weight.data.cpu().numpy()
        figs = {
            'alignment': plot_alignment(align_img),
            'phone_embedding_pca': plot_pca(phone_embeddings, self.train_dataset.idx2phone),
            'phone_embedding_tsne': plot_tsne(phone_embeddings, self.train_dataset.idx2phone)
        }
        self.tb_log(
            'TrainEpoch' if is_train else 'ValEpoch',
            figs,
            current_step,
            method_name='add_figure')

        char_str, phone_gt_str, phone_out_str, lang_str = self.sample2text(
            kwargs['char_inputs'][0], kwargs['phone_inputs'][0], kwargs['phone_outputs'][0], kwargs['lang_inputs'][0])
        texts = {
            'char_input': char_str,
            'phone_gt': phone_gt_str,
            'phone_output': phone_out_str,
            'lang_input': lang_str
        }
        self.tb_log(
            'TrainEpoch' if is_train else 'ValEpoch',
            texts,
            current_step,
            method_name='add_text')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration for training an TTS frontend")
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='Path to training outputs',
    )
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Path to restore checkpoint',
        default=''
    )
    args = parser.parse_args()

    trainer = None
    try:
        trainer = Train(args)
        trainer.run()
    except KeyboardInterrupt:
        if trainer:
            remove_experiment_folder(trainer.out_dir)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception:
        if trainer:
            remove_experiment_folder(trainer.out_dir)
        traceback.print_exc()
        sys.exit(1)
