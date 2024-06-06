import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np

# small number added before division to avoid numerical stability issue
epsilon = 1e-10


class Onehot(nn.Module):
    def __init__(self,
                 num_classes):
        super(Onehot, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return F.one_hot(x, num_classes=self.num_classes)


class Linear(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 init_gain='linear'):
        super(Linear, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Prenet(nn.Module):
    def __init__(self,
                 in_dim,
                 dropout=0.3,
                 out_dims=[512, 512],
                 bias=True):
        super(Prenet, self).__init__()
        assert 0.0 <= dropout <= 1.0

        self.dropout = dropout
        in_dims = [in_dim] + out_dims[:-1]
        self.layers = nn.ModuleList([
            Linear(in_dim, out_dim, bias=bias)
            for (in_dim, out_dim) in zip(in_dims, out_dims)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=self.dropout, training=self.training)
        return x


class GMMAttention(nn.Module):
    def __init__(self, attention_rnn_dim, attention_dim, num_mixtures=5, gmm_version="v2"):
        super(GMMAttention, self).__init__()
        self.query_layer = Linear(
            attention_rnn_dim, attention_dim, bias=True, init_gain='tanh')
        self.v = Linear(attention_dim, 3 * num_mixtures, bias=False)
        self.softplus = nn.Softplus()

        # Biased added to delta_hat and sigma_hat
        self.delta_bias = nn.Parameter(torch.Tensor(1, num_mixtures))
        self.sigma_bias = nn.Parameter(torch.Tensor(1, num_mixtures))

        self._mask_value = 0
        self.num_mixtures = num_mixtures
        self.gmm_version = gmm_version

        self._init_b()

    def _init_b(self):
        # Add biases that target a value of delta=1 for the initial forward movement
        # and sigma=10 for the initial standard deviation
        if self.gmm_version == "v0":
            nn.init.constant_(self.delta_bias, 0)
            nn.init.constant_(self.sigma_bias, -np.log(200))
        elif self.gmm_version == "v1":
            nn.init.constant_(self.delta_bias, 0)
            nn.init.constant_(self.sigma_bias, np.log(100))
        elif self.gmm_version == "v2":
            nn.init.constant_(self.delta_bias, np.log(np.e - 1))
            nn.init.constant_(self.sigma_bias, np.log(np.exp(10) - 1))
        else:
            raise RuntimeError("Unknown value for gmm version")

    def init_states(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.attention_weights = Variable(inputs.data.new(B, T).zero_())
        self.mu = Variable(inputs.data.new(B, 1, self.num_mixtures).zero_())

    def reset_beam_states(self, indices):
        """Reset the attention states with the states of the top K prefixes in the beam

        Args:
            indices: indices of the top K prefixes
        """
        self.attention_weights = self.attention_weights[indices]
        self.mu = self.mu[indices]

    def get_attention(self, query, inputs):
        B = query.shape[0]
        T = inputs.shape[1]
        processed_query = self.query_layer(query.unsqueeze(1))
        intermediates = self.v(torch.tanh(processed_query))
        w_hat, delta_hat, sigma_hat = torch.split(intermediates, self.num_mixtures, dim=-1)
        delta_hat = delta_hat + self.delta_bias
        sigma_hat = sigma_hat + self.sigma_bias

        if self.gmm_version == "v0":
            w = torch.exp(w_hat)
            delta = torch.exp(delta_hat)
            sigma_square = torch.exp(-sigma_hat) / 2
            z = torch.ones([B, 1, self.num_mixtures]).to(inputs.device, non_blocking=True)
        elif self.gmm_version == "v1":
            w = torch.softmax(w_hat, dim=-1)
            delta = torch.exp(delta_hat)
            sigma_square = torch.exp(sigma_hat)
            z = torch.sqrt(2 * np.pi * sigma_square)
        elif self.gmm_version == "v2":
            w = torch.softmax(w_hat, dim=-1)
            delta = self.softplus(delta_hat)
            sigma_square = self.softplus(sigma_hat) ** 2
            z = torch.sqrt(2 * np.pi * sigma_square)
        else:
            raise RuntimeError("Unknown value for gmm version")

        self.mu += delta

        pos = np.repeat(np.arange(T), self.num_mixtures).reshape((1, T, -1))
        pos = np.repeat(pos, B, axis=0)
        j = torch.from_numpy(pos).float().to(inputs.device, non_blocking=True)

        mixtures = -((j - self.mu) ** 2) / (2 * sigma_square + epsilon)
        mixtures = w * torch.exp(mixtures) / (z + epsilon)
        energies = torch.sum(mixtures, dim=-1)

        return energies

    def forward(self, attention_hidden_state, inputs, mask):
        attention = self.get_attention(attention_hidden_state, inputs)
        # apply masking so that the masked inputs would not contribute
        if mask is not None:
            attention.data.masked_fill_(~mask, self._mask_value)

        context = torch.bmm(attention.unsqueeze(1), inputs)
        context = context.squeeze(1)
        self.attention_weights = attention
        return context
