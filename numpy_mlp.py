import os
import numpy as np


class MLP2:
    def __init__(
        self, input_dim, output_dim, hidden_sizes=[], name_string="architecture"
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name_string = name_string

        self.Ws = []
        self.bs = []

        size = [input_dim] + hidden_sizes + [output_dim]
        for i in range(1, len(size)):
            self.Ws.append(np.zeros((size[i], size[i - 1]), dtype=np.float32))
            self.bs.append(np.zeros((size[i]), dtype=np.float32))

    def forward(self, x):
        out = x.copy()

        for i in range(len(self.Ws) - 1):
            out = self.leakyRelu(self.Ws[i].dot(out) + self.bs[i])

        out = self.Ws[-1].dot(out) + self.bs[-1]
        return out

    def load_state_dict(self, checkpoint):
        for i in range(len(self.Ws)):
            self.Ws[i][:] = checkpoint["{}.{}.weight".format(self.name_string, i * 2)]
            self.bs[i][:] = checkpoint["{}.{}.bias".format(self.name_string, i * 2)]

    def leakyRelu(self, x):
        return np.where(x > 0, x, x * 0.01)
