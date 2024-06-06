import os
import argparse
import numpy as np
import torch
import random
from tqdm import tqdm

from model import setup_model
from generic_utils import load_config, init_dicts
from padding import prepare_char
from params import set_default_params
from visual import plot_alignment, plot_pca, plot_tsne

import IPython.display as ipd


def inference(
    model,
    text_fpath,
    c,
    use_cuda,
    lang_idx,
    char2idx,
    idx2phone,
    batch_size,
    max_decoder_steps=500,
    decoding_method='greedy',
    beam_size=5,
    n_best=5,
    post_beam_resort=False,
    begin=0,
    end=10000
):
    """Carry out the inference for the given text file.

    Args:
        model (Tacotron): model for inference
        text_fpath (str): path to the input text file, with each line representing the char sequence
        c (dict): config dictionary to be loaded from config.json
        use_cuda (bool): enable cuda or not
        lang_idx (int): language index
        char2idx (dict): dictionary mapping char to index, which is used to convert chars in text_fpath
        idx2phone (dict): dictionary mapping index to phone, which is used to convert phone indices to a string
        batch_size (int): specify the batch size for inference
        max_decoder_steps (int): max number of decoder time steps
        decoding_method (str): decoding method, either 'greedy' or 'beam_search'
        beam_size (int): the size of the beam, only for 'beam_search'
        n_best (int): how many best hypothesis to keep after beam search, only for 'beam_search'
        post_beam_resort (bool): enable postnet beam resort or not, only for 'beam_search'
        begin (int): beginning line index
        end (int): ending line index

    Returns:
        phone_eos_str_list (list): a list of phone strings (with eos)
        phone_str_list (list): a list of phone strings (with eos removed)
        alignment_list (list): a list of 2d numpy array of alignment
        stop_token_list (list): a list of 1d numpy array of stop tokens
        ll_sum_list (list): a list of 1d numpy array of ll sums
    """
    # preprocess the given text
    with open(text_fpath, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip() != '']
        lines = lines[begin: end]

    batches = [lines[i: i + batch_size] for i in range(0, len(lines), batch_size)]

    phone_eos_str_list = []  # str list with eos
    ll_sum_list = []
    alignment_list = []  # a list of 2d numpy array
    stop_token_list = []  # a list of 1d numpy array

    for batch in tqdm(batches):
        chars = []   # a list of 1d char array
        for line in batch:
            char_list = list(line.replace(' ', '#')) + ['EOS']
            chars.append(
                np.array([char2idx[ch] for ch in char_list]))
        langs = [lang_idx] * len(chars)  # a list of integers

        # PAD sequences with the largest length of the batch
        chars = prepare_char(chars, pad_value=char2idx['PAD'])

        # convert numpy to tensor
        chars = torch.LongTensor(chars)
        langs = torch.LongTensor(langs)

        if use_cuda:
            chars = chars.cuda()
            langs = langs.cuda()

        dec_outputs, post_outputs, alignments, stop_tokens, hypothesis, ll_sums = \
            model.inference(chars, langs, max_decoder_steps, decoding_method, beam_size, post_beam_resort)

        # convert tensors to numpy
        dec_outputs = dec_outputs.data.cpu().numpy()
        post_outputs = post_outputs.data.cpu().numpy() if c.has_postnet else None
        alignments = alignments.data.cpu().numpy()
        stop_tokens = stop_tokens.data.cpu().numpy() if c.has_stopnet else None
        hypothesis = hypothesis.data.cpu().numpy()
        ll_sums = ll_sums.data.cpu().numpy()

        if decoding_method == 'beam_search':
            dec_outputs = dec_outputs[:n_best]
            post_outputs = post_outputs[:n_best] if c.has_postnet else None
            alignments = alignments[:n_best]
            stop_tokens = stop_tokens[:n_best] if c.has_stopnet else None
            hypothesis = hypothesis[:n_best]
            ll_sums = ll_sums[:n_best]

        if c.has_postnet:
            ph_idx_2d = np.argmax(post_outputs, axis=-1)
            for ph_idx_1d in ph_idx_2d:
                phone_eos_str_list.append(" ".join([idx2phone[ph_idx] for ph_idx in ph_idx_1d]))
        else:
            for ph_idx_1d in hypothesis:
                phone_eos_str_list.append(" ".join([idx2phone[ph_idx] for ph_idx in ph_idx_1d]))

        phone_str_list = [phone_eos_str.split('EOS')[0].rstrip() for phone_eos_str in phone_eos_str_list]

        alignment_list += list(alignments)
        stop_token_list += list(stop_tokens) if c.has_stopnet else [None]
        ll_sum_list += list(ll_sums)

    return phone_eos_str_list, phone_str_list, alignment_list, stop_token_list, ll_sum_list


def run_inference(
    config_fpath,
    src_vocab_fpath,
    tgt_vocab_fpath,
    restore_fpath,
    text_fpath,
    output_dir,
    lang='all',
    batch_size=8,
    max_decoder_steps=500,
    decoding_method='greedy',
    beam_size=5,
    n_best=5,
    post_beam_resort=False,
    figure=0,
    ipd_display=True,
):
    """Wrapper for the inference function.

    Args:
        config_fpath (str): path to config file
        src_vocab_fpath (str): path to source vocabulary file
        tgt_vocab_fpath (str): path to target vocabulary file
        restore_fpath (str): path to restore checkpoint
        text_fpath (str): path to input text file containing the char sequence in each line
        output_dir (str): directory containing all the output stuff
        lang (str or int): 'random', 'all', or integer language index started from 0, or language name like 'rpx'
        batch_size (int): specify the batch size for inference
        max_decoder_steps (int): max number of decoder time steps
        decoding_method (str): decoding method, either 'greedy' or 'beam_search'
        beam_size (int): the size of the beam, only for 'beam_search'
        n_best (int): how many best hypothesis to keep after beam search, only for 'beam_search'
        post_beam_resort (bool): enable postnet beam resort or not, only for 'beam_search'
        figure (int): the line index of the text_fpath for generating the alignment figure
        ipd_display (bool): enable ipd or not

    Returns:
        None
    """
    with torch.no_grad():
        # setup output paths and read configs
        c = load_config(config_fpath)
        c = set_default_params(c)

        use_cuda = torch.cuda.is_available()

        lang2idx, idx2lang, char2idx, idx2char, phone2idx, idx2phone = \
            init_dicts(c.data, src_vocab_fpath, tgt_vocab_fpath)

        model = setup_model(
            c=c,
            num_chars=len(char2idx) - 2,
            num_phones=len(phone2idx) - 2
        )  # minus EOS and PAD
        if use_cuda:
            checkpoint = torch.load(restore_fpath)
        else:
            checkpoint = torch.load(restore_fpath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])

        model.eval()
        if use_cuda:
            model = model.cuda()

        assert os.path.exists(text_fpath), '! text_fpath does not exist'

        if lang == "random":
            lang_ids = random.choice(list(idx2lang.keys()))
        elif lang == "all":
            lang_ids = sorted(idx2lang.keys())
        elif isinstance(lang, int):
            lang_ids = [lang]
        elif isinstance(lang, str):
            lang_ids = [lang2idx[lang]]
        else:
            raise ValueError("invalid lang, support: random, all, "
                             "language name, language index from 0. Avaliable languages: {}".format(lang2idx))

        for lang_idx in lang_ids:
            bn = os.path.basename(text_fpath).split('.')[0]
            lang_str = idx2lang[lang_idx]
            output_lang_dir = os.path.join(output_dir, lang_str)
            os.makedirs(output_lang_dir, exist_ok=True)

            print("Language: {}, synth: {} ...".format(lang_str, text_fpath))
            phone_eos_str_list, phone_str_list, alignment_list, stop_token_list, ll_sum_list = inference(
                model=model,
                text_fpath=text_fpath,
                c=c,
                use_cuda=use_cuda,
                lang_idx=lang_idx,
                char2idx=char2idx,
                idx2phone=idx2phone,
                batch_size=batch_size,
                max_decoder_steps=max_decoder_steps,
                decoding_method=decoding_method,
                beam_size=beam_size,
                n_best=n_best,
                post_beam_resort=post_beam_resort
            )
            phone_eos_fpath = os.path.join(output_lang_dir, bn + ".ph_eos.txt")
            with open(phone_eos_fpath, 'w') as f:
                for phone_eos_str in phone_eos_str_list:
                    f.write("{}\n".format(phone_eos_str))
            print("{} saved".format(phone_eos_fpath))

            phone_fpath = os.path.join(output_lang_dir, bn + ".ph.txt")
            with open(phone_fpath, 'w') as f:
                for phone_str in phone_str_list:
                    f.write("{}\n".format(phone_str))
            print("{} saved".format(phone_fpath))

            ll_sum_fpath = os.path.join(output_lang_dir, bn + ".ll_sum.txt")
            with open(ll_sum_fpath, 'w') as f:
                for ll_sum in ll_sum_list:
                    f.write("{}\n".format(ll_sum))
            print("{} saved".format(ll_sum_fpath))

            align_fpath = os.path.join(output_lang_dir, bn + '.align.png')
            fig = plot_alignment(alignment_list[figure])
            fig.savefig(align_fpath)
            print("No.{} figure {} saved".format(figure, align_fpath))

            pca_fpath = os.path.join(output_lang_dir, bn + '.pca.png')
            phone_embeddings = model.decoder.embedding.weight.data.cpu().numpy()
            fig = plot_pca(phone_embeddings, idx2phone)
            fig.savefig(pca_fpath)
            print("{} saved".format(pca_fpath))

            tsne_fpath = os.path.join(output_lang_dir, bn + '.tsne.png')
            fig = plot_tsne(phone_embeddings, idx2phone)
            fig.savefig(tsne_fpath)
            print("{} saved".format(tsne_fpath))

            if ipd_display:
                ipd.display(ipd.FileLink(phone_eos_fpath))
                ipd.display(ipd.Image(filename=align_fpath))
                ipd.display(ipd.Image(filename=pca_fpath))
                ipd.display(ipd.Image(filename=tsne_fpath))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_fpath',
        type=str,
        help='Path to config file for training',
    )
    parser.add_argument(
        '--src_vocab_fpath',
        type=str,
        help='Path to source vocabulary file',
    )
    parser.add_argument(
        '--tgt_vocab_fpath',
        type=str,
        help='Path to target vocabulary file',
    )
    parser.add_argument(
        '--restore_fpath',
        type=str,
        help='Path to restore checkpoint',
    )
    parser.add_argument(
        '--text_fpath',
        type=str,
        help='Path to input text file',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to output folder',
    )
    parser.add_argument(
        '--lang',
        type=int,
        help='Choose which language to inference (default 0)',
        default=0
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size for inference',
        default=8
    )
    parser.add_argument(
        '--max_decoder_steps',
        type=int,
        help='Max decoder steps for inference',
        default=500
    )
    parser.add_argument(
        '--decoding_method',
        type=str,
        help='Decoding method for decoder, either greedy or beam_search',
        default='greedy'
    )
    parser.add_argument(
        '--beam_size',
        type=int,
        help='The size of the beam, only for beam_search',
        default=5
    )
    parser.add_argument(
        '--n_best',
        type=int,
        help='How many best hypothesis to keep after beam search, only for beam_search',
        default=5
    )
    parser.add_argument(
        '--post_beam_resort',
        action='store_true',
        default=False,
        help='Enable postnet beam resort or not'
    )
    parser.add_argument(
        '--figure',
        type=int,
        default=0,
        help='Figure number to plot'
    )
    parser.add_argument(
        '--ipd_display',
        action='store_true',
        default=False,
        help='Enable ipd or not'
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_inference(
        config_fpath=args.config_fpath,
        src_vocab_fpath=args.src_vocab_fpath,
        tgt_vocab_fpath=args.tgt_vocab_fpath,
        restore_fpath=args.restore_fpath,
        text_fpath=args.text_fpath,
        output_dir=args.output_dir,
        lang=args.lang,
        batch_size=args.batch_size,
        max_decoder_steps=args.max_decoder_steps,
        decoding_method=args.decoding_method,
        beam_size=args.beam_size,
        n_best=args.n_best,
        post_beam_resort=args.post_beam_resort,
        figure=args.figure,
        ipd_display=args.ipd_display
    )
