# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from emg2qwerty.modules import BiLSTMEncoder, GRUEncoder


def _rand_inputs(seq_len: int, batch: int, dim: int) -> torch.Tensor:
    return torch.randn(seq_len, batch, dim)


def test_bilstm_encoder_outputs_shape():
    T, N, D = 7, 3, 16
    hidden = 5
    encoder = BiLSTMEncoder(input_size=D, hidden_size=hidden, num_layers=2, dropout=0.1)
    x = _rand_inputs(T, N, D)
    out = encoder(x)
    assert out.shape == (T, N, hidden * 2)


def test_gru_encoder_default_bidirectional():
    T, N, D = 6, 4, 12
    hidden = 4
    encoder = GRUEncoder(input_size=D, hidden_size=hidden, num_layers=1)
    x = _rand_inputs(T, N, D)
    out = encoder(x)
    # bidirectional by default -> hidden*2
    assert out.shape == (T, N, hidden * 2)


def test_gru_encoder_single_direction_and_layers():
    T, N, D = 5, 2, 10
    hidden = 7
    encoder = GRUEncoder(input_size=D, hidden_size=hidden, num_layers=3, bidirectional=False)
    x = _rand_inputs(T, N, D)
    out = encoder(x)
    assert out.shape == (T, N, hidden)
