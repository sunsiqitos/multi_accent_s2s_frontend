# coding: utf-8
import torch
from torch import nn
from tacotron_layer import Encoder, Decoder, Postnet
from generic_utils import sequence_mask, get_greedy_ll_indices


class Tacotron(nn.Module):
    def __init__(
        self,
        num_chars,
        num_phones,
        num_langs=1,
        enc_embedding_dim=512,
        dec_embedding_dim=512,
        lang_embedding_dim=32,
        enc_hidden_dim=512,
        dec_hidden_dim=512,
        post_hidden_dim=512,
        has_prenet=True,
        dropout=0.3,
        r=5,
        separate_stopnet=True,
        attn_type="gmm",
        num_mixtures=5,
        gmm_version='v2',
        has_postnet=False,
        has_stopnet=True
    ):

        super(Tacotron, self).__init__()
        self.enc_embedding = nn.Embedding(num_chars + 2, embedding_dim=enc_embedding_dim)  # plus EOS and PAD
        self.encoder = Encoder(
            in_dim=enc_embedding_dim,
            hidden_dim=enc_hidden_dim,
            has_prenet=has_prenet,
            dropout=dropout)

        self.lang_embedding = nn.Embedding(num_langs, embedding_dim=lang_embedding_dim)

        self.decoder = Decoder(
            in_dim=enc_hidden_dim + lang_embedding_dim,
            hidden_dim=dec_hidden_dim,
            num_phones=num_phones,
            embedding_dim=dec_embedding_dim,
            has_prenet=has_prenet,
            dropout=dropout,
            r=r,
            has_stopnet=has_stopnet,
            separate_stopnet=separate_stopnet,
            attn_type=attn_type,
            num_mixtures=num_mixtures,
            gmm_version=gmm_version)

        self.has_postnet = has_postnet
        self.has_stopnet = has_stopnet
        self.postnet = Postnet(
            out_dim=num_phones + 2,  # plus EOS and PAD
            hidden_dim=post_hidden_dim) if has_postnet else None

    def forward(self, characters, text_lengths, phones, lang_ids, sampler=None):
        # compute mask for padding
        mask = sequence_mask(text_lengths).to(characters.device, non_blocking=True)
        inputs = self.enc_embedding(characters)  # [B, T] -> [B, T, E]
        enc_outputs = self.encoder(inputs, text_lengths)  # [B, T, E]
        dec_inputs = self.append_lang_ids(enc_outputs, lang_ids)  # [B, T, D]

        dec_outputs, stop_tokens, alignments = self.decoder(
            dec_inputs, phones, mask, sampler)  # [B, C, L], [B, L'], [B, L', T]

        if self.has_postnet:
            post_outputs = self.postnet(dec_outputs)  # [B, C, L] -> [B, C, L]
            post_outputs += dec_outputs
            post_outputs = post_outputs.transpose(1, 2)  # [B, L, C]
        else:
            post_outputs = None

        dec_outputs = dec_outputs.transpose(1, 2)   # [B, L, C]

        return dec_outputs, post_outputs, alignments, stop_tokens

    def inference(self, characters, lang_ids, max_decoder_steps=500, decoding_method='greedy', beam_size=5, post_beam_resort=False):
        inputs = self.enc_embedding(characters)  # [B, T] -> [B, T, E]
        enc_outputs = self.encoder.inference(inputs)  # [B, T, E]
        dec_inputs = self.append_lang_ids(enc_outputs, lang_ids)  # [B, T, D]

        dec_outputs, stop_tokens, alignments, hypothesis, ll_sums = self.decoder.inference(
            dec_inputs, max_decoder_steps, decoding_method, beam_size)  # [B, C, L], [B, L'], [B, L', T]

        if self.has_postnet:
            post_outputs = self.postnet(dec_outputs)  # [B, C, L] -> [B, C, L]
            post_outputs += dec_outputs
            post_outputs = post_outputs.transpose(1, 2)  # [B, L, C]
            ll_sums, _ = get_greedy_ll_indices(post_outputs)

            if decoding_method == 'beam_search' and post_beam_resort:
                indices = torch.argsort(ll_sums, dim=-1, descending=True)
                dec_outputs, alignments, hypothesis, ll_sums, post_outputs = \
                    dec_outputs[indices], alignments[indices], hypothesis[indices], ll_sums[indices], post_outputs[indices]
                stop_tokens = stop_tokens[indices] if self.has_stopnet else None

        else:
            post_outputs = None

        dec_outputs = dec_outputs.transpose(1, 2)  # [B, L, C]

        return dec_outputs, post_outputs, alignments, stop_tokens, hypothesis, ll_sums

    def append_lang_ids(self, enc_outputs, lang_ids):
        """
        append language ids to encoder outputs
        """
        lang_encoding = self.lang_embedding(lang_ids)
        lang_encoding = lang_encoding.unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
        dec_inputs = torch.cat((enc_outputs, lang_encoding), -1)

        return dec_inputs
