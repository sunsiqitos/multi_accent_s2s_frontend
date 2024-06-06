import re
import json
import os
import datetime
import subprocess
import torch
import logging
import numpy as np
import glob
import shutil
from collections import Counter


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    # logging.info('Git Hash: {}'.format(commit))
    return commit


def create_experiment_folder(root_path, model_name):
    """ Create a folder with the current date and time """
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    # if debug:
    # commit_hash = 'debug'
    # else:
    commit_hash = get_commit_hash()
    output_folder = os.path.join(
        root_path, model_name + '-' + date_str + '-' + commit_hash)
    os.makedirs(output_folder, exist_ok=True)
    # logging.info('Experiment folder created: {}'.format(output_folder))
    return output_folder


def remove_experiment_folder(experiment_path):
    """Check folder if there is a checkpoint, otherwise remove the folder"""

    checkpoint_files = []
    checkpoint_files += glob.glob(experiment_path + "/*.pth.tar")
    checkpoint_files += glob.glob(experiment_path + "/**/*.pth.tar")

    if len(checkpoint_files) < 1:
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
            print(" ! Run is removed from {}".format(experiment_path))
    else:
        print(" ! Run is kept in {}".format(experiment_path))


def copy_config_file(config_file, out_path, new_fields=None):
    config_lines = open(config_file, "r").readlines()
    if new_fields:
        # add extra information fields
        for key, value in new_fields.items():
            if isinstance(value, str):
                new_line = '"{}":"{}",\n'.format(key, value)
            else:
                new_line = '"{}":{},\n'.format(key, value)
            config_lines.insert(1, new_line)
    config_out_file = open(out_path, "w")
    config_out_file.writelines(config_lines)
    config_out_file.close()


def get_token_counts(txt_fpath):
    """Get token counts, where tokens are separated by whitespaces"""
    with open(txt_fpath, 'r') as f:
        line = f.read().replace('\n', ' ')
        token_cnt = Counter(line.rstrip().split())
    return token_cnt


def get_char_counts(txt_fpath):
    """Get character counts, where whitespaces are replaced by hashtags"""
    with open(txt_fpath, 'r') as f:
        line = f.read().replace('\n', '')
        char_cnt = Counter(line.rstrip().replace(' ', '#'))  # use the hashtag to replace the space
    return char_cnt


def build_vocab(data_dict, src_vocab_fpath, tgt_vocab_fpath, word_vocab_fpath):
    """Build the vocabulary for the data

        Args:
            data_dict (dict): a nested dict called 'data' in a json config
            src_vocab_fpath (str): source vocabulary filepath, with each line looking like 'H    45' in descending order
            tgt_vocab_fpath (str): target vocabulary filepath, with each line looking like '@    32' in descending order
            word_vocab_fpath (str): (source) word vocabulary filepath, with each line looking like 'THE    453' in descending order

        Returns:
            len(src_vocab_list) (int): length of the source vocab
            len(tgt_vocab_list) (int): length of the target vocab

    """

    src_vocab, tgt_vocab, word_vocab = Counter(), Counter(), Counter()

    for lang, corpus_dict in data_dict.items():
        for corpus, src_tgt_dict in corpus_dict.items():

            if corpus == 'valid' or corpus == 'test':
                continue

            assert 'path_src' in src_tgt_dict, 'path_src is missing in {}/{}'.format(lang, corpus)
            assert 'path_tgt' in src_tgt_dict, 'path_tgt is missing in {}/{}'.format(lang, corpus)
            src_fpath, tgt_fpath = src_tgt_dict['path_src'], src_tgt_dict['path_tgt']

            src_vocab += get_char_counts(src_fpath)
            tgt_vocab += get_token_counts(tgt_fpath)
            word_vocab += get_token_counts(src_fpath)

    # sort according to the counts in descending order
    src_vocab_list = sorted(list(src_vocab.items()), key=lambda x: x[1], reverse=True)
    tgt_vocab_list = sorted(list(tgt_vocab.items()), key=lambda x: x[1], reverse=True)
    word_vocab_list = sorted(list(word_vocab.items()), key=lambda x: x[1], reverse=True)

    os.makedirs(os.path.dirname(src_vocab_fpath), exist_ok=True)
    os.makedirs(os.path.dirname(tgt_vocab_fpath), exist_ok=True)
    os.makedirs(os.path.dirname(word_vocab_fpath), exist_ok=True)

    with open(src_vocab_fpath, 'w') as f:
        for src_token, cnt in src_vocab_list:
            f.write("{}\t{}\n".format(src_token, cnt))

    with open(tgt_vocab_fpath, 'w') as f:
        for tgt_token, cnt in tgt_vocab_list:
            f.write("{}\t{}\n".format(tgt_token, cnt))

    with open(word_vocab_fpath, 'w') as f:
        for word, cnt in word_vocab_list:
            f.write("{}\t{}\n".format(word, cnt))

    return len(src_vocab_list), len(tgt_vocab_list)


def init_dicts(data_dict, src_vocab_fpath, tgt_vocab_fpath):
    # initialize the language dicts
    lang2idx = {l: idx for idx, l in enumerate(sorted(data_dict.keys()))}
    idx2lang = {v: k for k, v in lang2idx.items()}
    # initialize the char dicts
    with open(src_vocab_fpath, 'r') as f:
        lines = f.readlines()
        chars = [line.strip().split()[0] for line in lines if line.strip() != '']
        char2idx = {ch: idx for idx, ch in enumerate(chars)}
        char2idx['EOS'] = len(chars)
        char2idx['PAD'] = len(chars) + 1
        idx2char = {v: k for k, v in char2idx.items()}
    # initialize the phone dicts
    with open(tgt_vocab_fpath, 'r') as f:
        lines = f.readlines()
        phones = [line.strip().split()[0] for line in lines if line.strip() != '']
        phone2idx = {ph: idx for idx, ph in enumerate(phones)}
        phone2idx['EOS'] = len(phones)
        phone2idx['PAD'] = len(phones) + 1
        idx2phone = {v: k for k, v in phone2idx.items()}
    return lang2idx, idx2lang, char2idx, idx2char, phone2idx, idx2phone


def save_checkpoint(model, optimizer, optimizer_st, current_step, epoch, out_dir, loss, best=False):
    assert os.path.isdir(out_dir)
    ckpt_fn = 'best.pth.tar' if best is True else 'step_{}_epoch_{}.pth.tar'.format(current_step, epoch)
    ckpt_fpath = os.path.join(out_dir, ckpt_fn)

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': current_step,
        'epoch': epoch,
        'loss': loss,
        'date': datetime.date.today().strftime("%B %d, %Y")
    }

    if optimizer_st:
        state['optimizer_st'] = optimizer_st.state_dict()

    torch.save(state, ckpt_fpath)


def load_checkpoint(model, optimizer, optimizer_st, ckpt_fpath):
    if ckpt_fpath == '' or ckpt_fpath is None:
        next_epoch = 0
        next_step = 0
    else:
        ckpt = torch.load(ckpt_fpath)

        next_epoch = ckpt['epoch'] + 1
        next_step = ckpt['step']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if optimizer_st:
            optimizer_st.load_state_dict(ckpt['optimizer_st'])

    return next_epoch, next_step


# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (
        sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    # B x T_max
    return seq_range_expand < seq_length_expand


def count_parameters(model):
    """Count number of trainable parameters in a network"""
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, total_trainable_params


def check_update(model, grad_clip, check_grad_norm):
    '''Check model gradient against unexpected jumps and failures'''
    skip_flag = False
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if check_grad_norm and np.isinf(grad_norm.cpu().detach().numpy()):
        logging.info(" | > Gradient is INF !!")
        skip_flag = True
    return grad_norm, skip_flag


def get_current_lr(optimizer):
    '''Get current learning rate'''
    for group in optimizer.param_groups:
        for param in group['params']:
            current_lr = group['lr']
    return current_lr


def get_greedy_ll_indices(output):
    output = torch.log_softmax(output, dim=-1)  # [B, L, C]
    ll_sums, vocab_indices = torch.topk(output, k=1, dim=-1)  # [B, L, 1], [B, L, 1]
    ll_sums = torch.sum(ll_sums.squeeze(-1), dim=-1)  # [B, ]
    vocab_indices = vocab_indices.squeeze(-1)  # [B, L]
    return ll_sums, vocab_indices
