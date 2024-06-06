from collections import namedtuple
import numpy as np
import random
import logging
import torch
from torch.utils.data import Dataset
from padding import prepare_stop_target, prepare_char, prepare_phone
from generic_utils import init_dicts

# tuple of built-in item
Item = namedtuple(
    'Item', ['src', 'tgt', 'lang'])

# tuple from __getitem__
Sample = namedtuple(
    'Sample', ['char', 'phone', 'lang'])


class FestivalDataset(Dataset):
    def __init__(
        self,
        mode,
        data_dict,
        src_vocab_fpath,
        tgt_vocab_fpath,
        r,
        batch_group_size=0,
        over_sampling=None,
        min_seq_len=0,
        max_seq_len=float("inf"),
        verbose=True
    ):
        logging.info(' > Initialize {} dataset'.format(mode))
        self.mode = mode
        self.data_dict = data_dict
        self.r = r
        self.batch_group_size = batch_group_size
        self.over_sampling = over_sampling
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.verbose = verbose

        self.lang2idx, self.idx2lang, self.char2idx, self.idx2char, self.phone2idx, self.idx2phone = \
            init_dicts(data_dict, src_vocab_fpath, tgt_vocab_fpath)

        self.unsorted = self._init_items()  # list containing unsorted items
        self.sorted = self._sort_items(self.unsorted)  # list after sorting
        self.items = None  # list after shuffling

    def _init_items(self):
        items = []
        for lang, corpus_dict in self.data_dict.items():
            # get the total number of samples for each language
            num_samples = 0
            for corpus, fpath_dict in corpus_dict.items():
                if (self.mode == 'train' and (corpus == 'valid' or corpus == 'test')) or \
                        (self.mode == 'val' and corpus != 'valid'):
                    continue

                with open(fpath_dict['path_src'], 'r') as f:
                    num_samples += sum([1 for line in f if line.strip()])
            num_samples = self.over_sampling or num_samples
            items += self._get_lang_items(lang, corpus_dict, num_samples)
        return items

    def _get_lang_items(self, lang, corpus_dict, num_samples):
        items = []
        while True:
            for corpus, fpath_dict in corpus_dict.items():
                if (self.mode == 'train' and (corpus == 'valid' or corpus == 'test')) or \
                        (self.mode == 'val' and corpus != 'valid'):
                    continue

                with open(fpath_dict['path_src'], 'r') as src_f:
                    with open(fpath_dict['path_tgt'], 'r') as tgt_f:

                        src_lines = src_f.readlines()
                        tgt_lines = tgt_f.readlines()
                        src_lines = [line.strip() for line in src_lines if line.strip() != '']
                        tgt_lines = [line.strip() for line in tgt_lines if line.strip() != '']

                        assert len(src_lines) == len(tgt_lines), "Src length not equal tgt length"
                        src_tgt_pairs = list(zip(src_lines, tgt_lines))

                        for src_line, tgt_line in src_tgt_pairs:
                            if len(items) >= num_samples:
                                return items

                            src = list(src_line.replace(' ', '#')) + ['EOS']
                            tgt = tgt_line.split() + ['EOS']
                            items.append(
                                Item(src=src, tgt=tgt, lang=lang))

    def _sort_items(self, items):
        """Sort instances based on text length in ascending order"""
        # check the text length
        lengths = np.array([len(item.src) for item in items])
        idxes = np.argsort(lengths)

        new_items = []
        ignored = []

        for idx in idxes:
            length = lengths[idx]
            if length < self.min_seq_len or length > self.max_seq_len:
                ignored.append(idx)
            else:
                new_items.append(items[idx])

        if self.verbose:
            logging.info(' | > Max length sequence: {}'.format(np.max(lengths)))
            logging.info(' | > Min length sequence: {}'.format(np.min(lengths)))
            logging.info(' | > Avg length sequence: {}'.format(np.mean(lengths)))
            logging.info(' | > Num. instances discarded by max-min seq limits: {}'.format(len(ignored)))
            logging.info(' | > Batch group size: {}.'.format(self.batch_group_size))

        return new_items

    def shuffle_items(self, shuffle=True):
        """To be called before each epoch"""
        self.items = self.sorted.copy()
        # shuffle batch groups
        if shuffle and self.batch_group_size > 0:
            logging.info(" | > Shuffling item list. Batch group size: {}.".format(self.batch_group_size))
            for i in range(int(np.ceil(len(self.items) / self.batch_group_size))):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                temp_items = self.items[offset:end_offset]
                random.shuffle(temp_items)
                self.items[offset:end_offset] = temp_items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        char_list, phone_list, lang = self.items[idx]
        char_array = np.array([self.char2idx[ch] for ch in char_list])
        phone_array = np.array([self.phone2idx[ph] for ph in phone_list])
        lang_id = self.lang2idx[lang]
        return Sample(
            char=char_array,
            phone=phone_array,
            lang=lang_id)

    def collate_fn(self, batch):
        """
        Perform preprocessing and create a final data batch:
        1. PAD sequences with the longest sequence in the batch
        2. PAD sequences that can be divided by the reduction factor r.
        3. Convert Numpy to Torch tensors.
        """
        char_lengths = np.array([len(s.char) for s in batch])
        ids_sorted_decreasing = np.argsort(char_lengths)[::-1]

        chars = [batch[idx].char for idx in ids_sorted_decreasing]  # a list of 1d char array
        char_lengths = [len(ch) for ch in chars]  # a list of integers
        phones = [batch[idx].phone for idx in ids_sorted_decreasing]  # a list of 1d phone array
        phone_lengths = [len(ph) for ph in phones]  # a list of integers
        langs = [batch[idx].lang for idx in ids_sorted_decreasing]  # a list of integers

        # compute 'stop token' targets
        stop_targets = [
            np.array([0.] * ph_len) for ph_len in phone_lengths
        ]

        # PAD stop targets
        stop_targets = prepare_stop_target(stop_targets, self.r)
        # PAD sequences with the largest length of the batch
        chars = prepare_char(chars, pad_value=self.char2idx['PAD'])
        # PAD sequences with the largest length of the batch
        phones = prepare_phone(phones, self.r, pad_value=self.phone2idx['PAD'])

        # convert numpy to tensor
        chars = torch.LongTensor(chars)
        char_lengths = torch.LongTensor(char_lengths)
        phones = torch.LongTensor(phones)
        phone_lengths = torch.LongTensor(phone_lengths)
        langs = torch.LongTensor(langs)
        stop_targets = torch.FloatTensor(stop_targets)

        return chars, char_lengths, phones, phone_lengths, langs, stop_targets
