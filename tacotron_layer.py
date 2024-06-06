import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from common_layers import Prenet, Linear, GMMAttention
from generic_utils import get_greedy_ll_indices


class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinear=None):
        super(ConvBNBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding)
        norm = nn.BatchNorm1d(out_channels)
        dropout = nn.Dropout(p=0.3)
        if nonlinear == 'relu':
            self.net = nn.Sequential(conv1d, norm, nn.ReLU(), dropout)
        elif nonlinear == 'tanh':
            self.net = nn.Sequential(conv1d, norm, nn.Tanh(), dropout)
        else:
            self.net = nn.Sequential(conv1d, norm, dropout)

    def forward(self, x):
        output = self.net(x)
        return output


class Postnet(nn.Module):
    def __init__(self, out_dim, hidden_dim=512, num_convs=3):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            ConvBNBlock(out_dim, hidden_dim, kernel_size=5, nonlinear='tanh'))
        for _ in range(1, num_convs - 1):
            self.convolutions.append(
                ConvBNBlock(hidden_dim, hidden_dim, kernel_size=5, nonlinear='tanh'))
        self.convolutions.append(
            ConvBNBlock(hidden_dim, out_dim, kernel_size=5, nonlinear=None))

    def forward(self, x):
        for layer in self.convolutions:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, has_prenet=True, dropout=0.3):
        super(Encoder, self).__init__()
        self.has_prenet = has_prenet

        self.prenet = Prenet(in_dim=in_dim, dropout=dropout, out_dims=[in_dim]) if has_prenet else None
        self.lstm = nn.LSTM(
            in_dim,
            int(hidden_dim / 2),
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.prenet(x) if self.has_prenet else x
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs,
            batch_first=True,
        )
        return outputs

    def inference(self, x):
        x = self.prenet(x) if self.has_prenet else x
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


# adapted from https://github.com/NVIDIA/tacotron2/
class Decoder(nn.Module):
    """Decoder module.

    Args:
        in_dim (int): input vector (encoder side & language) dimension = context vector dimension.
        embedding_dim (int): embedding dimension = output vector (decoder side) dimension.
        r (int): reduction factor, i.e., number of outputs per timestep.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_phones,
        embedding_dim,
        has_prenet=True,
        dropout=0.3,
        r=5,
        has_stopnet=True,
        separate_stopnet=True,
        attn_type='gmm',
        num_mixtures=5,
        gmm_version='v2'
    ):

        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.num_phones = num_phones
        self.out_dim = self.num_phones + 2  # plus EOS and PAD
        self.embedding_dim = embedding_dim
        self.has_prenet = has_prenet
        self.dropout = dropout
        self.r = r
        self.has_stopnet = has_stopnet
        self.separate_stopnet = separate_stopnet
        self.attn_type = attn_type

        self.attention_rnn_dim = hidden_dim
        self.decoder_rnn_dim = hidden_dim
        self.p_attention_dropout = dropout
        self.p_decoder_dropout = dropout

        # previous idx -> |Embedding| -> embedding
        self.embedding = nn.Embedding(self.num_phones + 2, embedding_dim=self.embedding_dim)  # plus EOS and PAD
        # r embeddings -> |Prenet| -> processed_memory
        self.prenet = Prenet(
            in_dim=self.embedding_dim * self.r,
            dropout=dropout,
            out_dims=[self.embedding_dim]) if has_prenet else None
        # processed_memory, prev_context -> |Attention RNN| -> attn_RNN_output
        self.attention_rnn = nn.LSTMCell(
            self.embedding_dim + in_dim if has_prenet else self.embedding_dim * self.r + in_dim,
            self.attention_rnn_dim)
        # attn_RNN_output, inputs -> |Attention| -> context
        if attn_type == "gmm":
            self.attention_layer = GMMAttention(attention_rnn_dim=self.attention_rnn_dim,
                                                attention_dim=128,
                                                num_mixtures=num_mixtures,
                                                gmm_version=gmm_version)
        else:
            raise NotImplementedError
        # attn_RNN_output, context -> |Decoder RNN| -> decoder_RNN_output
        self.decoder_rnn = nn.LSTMCell(self.attention_rnn_dim + in_dim,
                                       self.decoder_rnn_dim)
        # decoder_RNN_output, context -> |Linear| -> linear output
        self.linear_projection = Linear(self.decoder_rnn_dim + in_dim,
                                        self.out_dim * self.r)
        # decoder_RNN_output, linear output -> |StopNet| -> stop_token
        self.stopnet = nn.Sequential(
            nn.Dropout(0.1),
            Linear(
                self.decoder_rnn_dim + self.out_dim * self.r,
                1,
                bias=True,
                init_gain='sigmoid')) if has_stopnet else None

        # learn init values instead of zero init.
        self.attention_rnn_init = nn.Embedding(1, self.attention_rnn_dim)
        self.go_init = nn.Embedding(1, self.embedding_dim * self.r)
        self.decoder_rnn_inits = nn.Embedding(1, self.decoder_rnn_dim)

        self.eos_idx = self.num_phones  # the second to last index of the embedding is the EOS phone
        self.pad_idx = self.num_phones + 1  # the last index of the embedding is the PAD phone

    def get_go_tensor(self, inputs):
        B = inputs.size(0)
        memory = self.go_init(inputs.data.new_zeros(B).long())
        return memory

    def _init_states(self, inputs, mask):
        B = inputs.size(0)

        self.attention_hidden = self.attention_rnn_init(
            inputs.data.new_zeros(B).long())
        self.attention_cell = Variable(
            inputs.data.new(B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = self.decoder_rnn_inits(
            inputs.data.new_zeros(B).long())
        self.decoder_cell = Variable(
            inputs.data.new(B, self.decoder_rnn_dim).zero_())

        self.context = Variable(
            inputs.data.new(B, self.in_dim).zero_())

        self.inputs = inputs
        self.mask = mask

    def _reshape_memory(self, memories):
        """Reshape the decoder memories

        Args:
            memories (tensor): decoder memories of shape [B, L, E],
                where L is the target sequence length and E is the embedding dim

        Returns:
            memories (tensor): reshaped memories of shape [L', B, E'], where L' = L / r, E' = E * r
        """
        # Grouping multiple frames if necessary
        assert memories.size(1) % self.r == 0
        memories = memories.view(
            memories.size(0), int(memories.size(1) / self.r), -1)  # [B, L, E] -> [B, L', E']
        memories = memories.transpose(0, 1)  # [B, L', E'] -> [L', B, E']
        return memories

    def _reshape_outputs(self, outputs, stop_tokens, alignments):
        """Reshape the outputs

        Args:
            outputs (list): a list of stepwise output, L' * [B, C']
            stop_tokens (list): a list of stepwise stop tokens, L' * [B]
            alignments (list): a list of stepwise alignment, L' * [B, T]

        Returns:
            outputs (tensor): output tensor of shape [B, C, L]
            stop_tokens (tensor): stop token tensor of shape [B, L']
            alignments (tensor): alignment tensor of shape [B, L' T]
        """
        alignments = torch.stack(alignments).transpose(0, 1)  # [L', B, T] -> [B, L', T]
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1).contiguous() if self.has_stopnet else None  # [L', B] -> [B, L']
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()  # [L', B, C'] -> [B, L', C']
        outputs = outputs.view(outputs.size(0), outputs.size(1) * self.r, self.out_dim)  # [B, L', C'] -> [B, L, C]
        outputs = outputs.transpose(1, 2)  # [B, L, C] -> [B, C, L]
        return outputs, stop_tokens, alignments

    def decode(self, memory):
        """Stepwise decode

        Args:
            memory: stepwise decoder memory of shape [B, E']

        Returns:
            decoder_output: stepwise decoder output of shape [B, C'],
                where C' = out_dim(=C) * r
            stop_token: stepwise stop token of shape [B, 1]
            self.attention_layer.attention_weights: stepwise attention weights of shape [B, T]
        """
        memory = self.prenet(memory) if self.has_prenet else memory
        cell_input = torch.cat((memory, self.context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)

        self.context = self.attention_layer(self.attention_hidden, self.inputs, self.mask)

        memory = torch.cat((self.attention_hidden, self.context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            memory, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden,
                                        self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(self.decoder_cell,
                                      self.p_decoder_dropout, self.training)

        decoder_hidden_context = torch.cat((self.decoder_hidden, self.context),
                                           dim=1)

        decoder_output = self.linear_projection(decoder_hidden_context)

        stopnet_input = torch.cat((self.decoder_hidden, decoder_output), dim=1)

        if self.has_stopnet:
            stop_token = self.stopnet(stopnet_input.detach() if self.separate_stopnet else stopnet_input)
        else:
            stop_token = None
        return decoder_output, stop_token, self.attention_layer.attention_weights

    def _gen_new_memory(self, sampler, memory_gt, output_pred):
        """Generate new memory for the next decoding step

        Args:
            sampler (ScheduledSampler): scheduled sampler
            memory_gt (tensor): decoder memory of shape [B, E']
            output_pred (tensor): output predicted from the previous step of shape [B, C']

        Returns:
            new_memory (tensor): new memory of shape [B, E']
        """
        if sampler is None:
            # for synthesis_tf
            new_memory = memory_gt
        else:
            B = output_pred.size(0)
            output_pred = output_pred.view(B, self.r, self.out_dim)  # [B, C'] -> [B, r, C]
            memory_pred = torch.matmul(output_pred, self.embedding.weight)  # [B, r, C] * [C, E] -> [B, r, E]
            memory_pred = memory_pred.view(B, self.r * self.embedding_dim)  # [B, r, E] -> [B, E']

            # make new_memory (next input) considering the ratio
            noise = Variable(memory_gt.data.new(memory_gt.size()).normal_(0.0, 0.1)) if sampler.noise else None
            new_memory = sampler.get_new_memory(memory_gt, memory_pred, noise)
        return new_memory

    def forward(self, inputs, memories, mask, sampler=None):
        """Forward pass

        Args:
            inputs (tensor): encoder outputs concatenated with language embedding of shape [B, T, D],
                where D = enc_out_dim + lang_embedding_dim
            memories (tensor): decoder memory of shape [B, L]
            mask (tensor): attention mask for sequence padding.
            sampler (ScheduledSampler): scheduled sampler

        Returns:
            outputs (tensor): output tensor of shape [B, C, L]
            stop_tokens (tensor): stop token tensor of shape [B, L']
            alignments (tensor): alignment tensor of shape [B, L' T]
        """
        memory = self.get_go_tensor(inputs).unsqueeze(0)  # [B, E'] -> [1, B, E']
        memories = self.embedding(memories)  # [B, L] -> [B, L, E]
        memories = self._reshape_memory(memories)  # [B, L, E] -> [L', B, E']
        memories = torch.cat((memory, memories), dim=0)  # [1+L', B, E']

        self._init_states(inputs, mask=mask)
        self.attention_layer.init_states(inputs)

        outputs, stop_tokens, alignments = [], [], []
        t = 0
        while len(outputs) < memories.size(0) - 1:
            memory = self._gen_new_memory(sampler, memories[t], outputs[-1]) if t > 0 else memories[0]
            output, stop_token, attention_weights = self.decode(memory)  # [B, C'], [B, 1], [B, T]
            outputs += [output]  # TODO: why squeeze(1)?
            stop_tokens += [stop_token.squeeze(1) if self.has_stopnet else None]
            alignments += [attention_weights]
            t += 1

        outputs, stop_tokens, alignments = self._reshape_outputs(
            outputs, stop_tokens, alignments)  # [B, C, L], [B, L'], [B, L', T]

        return outputs, stop_tokens, alignments

    def inference(self, inputs, max_decoder_steps=500, decoding_method='greedy', beam_size=5):
        """
        Args:
            inputs (tensor): encoder outputs concatenated with language embedding of shape [B, T, D],
                where D = enc_out_dim + lang_embedding_dim
            max_decoder_steps (int): max number of decoder steps
            decoding_method (str): decoding method, either 'greedy' or 'beam_search'
            beam_size (int): the size of the beam

        Returns:
            outputs (tensor): output tensor of shape [B, C, L]
            stop_tokens (tensor): stop token tensor of shape [B, L']
            alignments (tensor): alignment tensor of shape [B, L' T]
            prefix (tensor): if 'greedy': previous prefix of shape [B, L] in arbitrary order
                elif 'beam_search': previous prefix of shape [K, L] in descending order
        """
        outputs, stop_tokens, alignments, t = [], [], [], 0
        memory = self.get_go_tensor(inputs)  # [B, E']

        self._init_states(inputs, mask=None)
        self.attention_layer.init_states(inputs)

        B = inputs.size(0)
        if decoding_method == 'greedy':
            prefix = inputs.data.new(B, 0).long()
            ll_sums = inputs.data.new(B).zero_()
        elif decoding_method == 'beam_search':
            assert B == 1, "Beam search decoding only supports B = 1"
            prefix = inputs.data.new(1, 0).long()
            ll_sums = inputs.data.new(1).zero_()
        else:
            raise NotImplementedError("Only support greedy and beam_search now")

        while True:
            output, stop_token, alignment = self.decode(memory)  # [B, C'], [B, 1], [B, T]

            if decoding_method == 'greedy':
                memory, prefix, ll_sums = self._greedy_decoding(output, prefix, ll_sums)
            elif decoding_method == 'beam_search':
                memory, prefix, ll_sums, prefix_indices = self._beam_search_decoding(output, prefix, ll_sums, beam_size)
                # revise output, stop_token and alignment
                output = output[prefix_indices]
                stop_token = stop_token[prefix_indices] if self.has_stopnet else None
                alignment = alignment[prefix_indices]
                # reset the states with the states of the top K preixes in the beam
                self._reset_beam_states(prefix_indices)
                self.attention_layer.reset_beam_states(prefix_indices)
                # revise inputs
                self.inputs = self.inputs[prefix_indices]
            else:
                raise NotImplementedError("Only support greedy and beam_search now")

            stop_token = torch.sigmoid(stop_token) if self.has_stopnet else None
            outputs += [output]  # TODO: why squeeze(1)?
            stop_tokens += [stop_token.squeeze(1) if self.has_stopnet else None]
            alignments += [alignment]
            t += 1
            if self.has_stopnet and t > inputs.shape[1] / 4 and \
                    ((stop_token > 0.6).all() or (alignment[:, -1] > 0.6).all()):
                break
            elif not self.has_stopnet and t > inputs.shape[1] / 4 and \
                    torch.logical_or(prefix[:, -1] == self.eos_idx, prefix[:, -1] == self.pad_idx).all():
                break
            elif t > max_decoder_steps:
                print("   | > {} decoder stopped with 'max_decoder_steps".format(decoding_method))
                break

        outputs, stop_tokens, alignments = self._reshape_outputs(
            outputs, stop_tokens, alignments)

        return outputs, stop_tokens, alignments, prefix, ll_sums

    def _reset_beam_states(self, indices):
        """Reset the decoder states with the states of the top K prefixes in the beam

        Args:
            indices: indices of the top K prefixes
        """
        self.attention_hidden = self.attention_hidden[indices]
        self.attention_cell = self.attention_cell[indices]
        self.decoder_hidden = self.decoder_hidden[indices]
        self.decoder_cell = self.decoder_cell[indices]
        self.context = self.context[indices]

    def _beam_search_decoding(self, output, prefix, ll_sums, beam_size):
        """Carry out the beam search decoding for one decoder step, currently assuming B = 1

        Args:
            output (tensor): output tensor of one decoder step of shape [K, C'], where C' = out_dim(=C) * r
            prefix (tensor): previous prefix tensor of shape [K, R],
                whose log-likelihoods are ll_sums
                e.g. tensor([[1, 34, 22], [1, 34, 42], [1, 21, 28], ...])
            ll_sums (tensor): log-likelihood sum of shape [K, ] in descending order
            beam_size (int): K, the size of the beam

        Returns:
            memory (tensor): stepwise decoder memory of shape [K, E']
            prefix (tensor): extended prefix of shape [K, R + r]
            ll_sums (tensor): extended log-likelihood sum of shape [K, ] in descending order

        """
        output = output.view(output.size(0), self.r, self.out_dim)  # [K, C'] -> [K, r, C]
        output = torch.log_softmax(output, dim=-1)  # [K, r, C]

        outer_sum = output[:, 0, :]  # [K, C]
        for ii in range(1, self.r):
            output_slice = output[:, ii, :]  # [K, C]
            outer_sum = outer_sum[..., None] + output_slice[..., None, :]  # [K, C^ii, 1] + [K, 1, C] -> [K, C^ii, C]
            outer_sum = outer_sum.view(outer_sum.size(0), self.out_dim ** (ii + 1))  # [K, C^(ii+1)]
        # outer_sum is now of shape [K, C^r]

        ll_sums = outer_sum + ll_sums.unsqueeze(-1)  # [K, C^r] + [K, 1] -> [K, C^r]

        # sort and prune to get the top K prefixes out of K * C^r candidates
        ll_sums, indices = torch.topk(ll_sums.view(-1), k=beam_size)  # [K, ], [K, ]
        # indices of the previous prefixes to extend, in [0, K)
        prefix_indices = torch.div(indices, self.out_dim ** self.r, rounding_mode='floor')  # [K, ]
        # indices of the vocab to be appended, in [0, C^r)
        vocab_indices = torch.remainder(indices, self.out_dim ** self.r)  # [K, ]

        vocab_indices_list = []  # a list of r vocab indices tensors (of shape [K, ])
        for ii in range(1, self.r):
            vocab_indices_list.append(
                torch.div(vocab_indices, self.out_dim ** (self.r - ii), rounding_mode='floor')  # [K, ]
            )
            vocab_indices = torch.remainder(vocab_indices, self.out_dim ** (self.r - ii))  # [K, ]
        vocab_indices_list.append(vocab_indices)
        vocab_indices = torch.stack(vocab_indices_list).transpose(0, 1).contiguous()  # [K, r]

        prefix = torch.cat((prefix[prefix_indices], vocab_indices), dim=-1)  # [K, R] & [K, r] -> [K, R + r]

        memory = self.embedding(vocab_indices)  # [K, r] -> [K, r, E]
        memory = self._reshape_memory(memory).squeeze(0)  # [K, r, E] -> [1, K, E'] -> [K, E']

        return memory, prefix, ll_sums, prefix_indices

    def _greedy_decoding(self, output, prefix, ll_sums):
        """Carry out the greedy decoding for one decoder step

        Args:
            output (tensor): output tensor of one decoder step of shape [B, C'], where C' = out_dim(=C) * r
            prefix (tensor): previous prefix tensor of shape [B, R]
                whose log-likelihoods are ll_sums
                e.g. tensor([[17, 31, 2], [17, 13, 4], [21, 24, 8], ...])
            ll_sums (tensor): log-likelihood sum of shape [B, ] in arbitrary order

        Returns:
            memory (tensor): stepwise decoder memory of shape [B, E']
            prefix (tensor): extended prefix of shape [B, R + r]
            ll_sums (tensor): extended log-likelihood sum of shape [B, ] in arbitrary order

        """
        output = output.view(output.size(0), self.r, self.out_dim)  # [B, C'] -> [B, r, C]
        output_ll_sums, vocab_indices = get_greedy_ll_indices(output)  # [B, ], [B, r]

        ll_sums += output_ll_sums  # [B, ]
        prefix = torch.cat((prefix, vocab_indices), dim=-1)  # [B, R] & [B, r] -> [B, R + r]

        memory = self.embedding(vocab_indices)  # [B, r] -> [B, r, E]
        memory = self._reshape_memory(memory).squeeze(0)  # [B, r, E] -> [1, B, E'] -> [B, E']

        return memory, prefix, ll_sums
