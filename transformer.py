import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np


def clones(module, num_layers):
    return nn.ModuleList([deepcopy(module) for _ in range(num_layers)])


class MultiHeadAttention(nn.Module):

    def __init__(self, state_size, num_heads, dropout=0.1):
        assert state_size % num_heads == 0
        super(MultiHeadAttention, self).__init__()
        self.state_size = state_size
        self.num_heads = num_heads
        self.head_size = state_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_size)

        self.wq = nn.Linear(in_features=state_size, out_features=state_size)
        self.wk = nn.Linear(in_features=state_size, out_features=state_size)
        self.wv = nn.Linear(in_features=state_size, out_features=state_size)
        self.wk = nn.Linear(in_features=state_size, out_features=state_size)
        self.wo = nn.Linear(in_features=state_size, out_features=state_size)
        self.dropout = nn.Dropout(p=dropout)

    def _attention(self, query, key, value, mask):
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if (mask is not None):
            scores = logits.masked_fill(mask == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value)

    def forward(self, query, key, value, mask=None):
        if (mask is not None):
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        # length = query.size(1)

        query = self.wq(query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.wk(key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.wv(value).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        output = self._attention(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.state_size)
        return self.wo(output)


class ScaledEmbedding(nn.Module):

    def __init__(self, vocab_size, state_size, pad_index):
        super(ScaledEmbedding, self).__init__()
        self.scale = math.sqrt(state_size)
        self.embedding = nn.Embedding(vocab_size, state_size, padding_idx=pad_index)

    def forward(self, x):
        return self.embedding(x) * self.scale


class PositionalEncoding(nn.Module):

    def __init__(self, state_size, dropout=0.1, max_length=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        positional_encoding = torch.zeros(max_length, state_size)
        position = torch.arange(0, max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, state_size, 2) * -(math.log(10000.0) / state_size))
        positional_encoding[:, 0::2] = torch.sin(position * scale)
        positional_encoding[:, 1::2] = torch.cos(position * scale)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        output = x + self.positional_encoding[:, :x.size(1)]
        return self.dropout(output)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, state_size, hidden_size, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(state_size, hidden_size)
        self.norm = nn.LayerNorm(state_size, eps=1e-6)
        self.dropout1 = nn.Dropout(p=dropout)
        self.w2 = nn.Linear(hidden_size, state_size)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        hidden = self.dropout1(F.relu(self.w1(self.norm(x))))
        output = self.dropout2(self.w2(hidden))
        return x + output


class ResidualConnection(nn.Module):

    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, size, num_heads, hidden_size, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.size = size
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.attention = MultiHeadAttention(size, num_heads, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionWiseFeedForward(size, hidden_size, dropout)

    def forward(self, x, mask):
        x_norm = self.norm(x)
        output = self.attention(x_norm, x_norm, x_norm, mask)
        output = self.dropout(output) + x
        return self.feed_forward(output)


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, state_size, num_heads, hidden_size, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        layer = TransformerEncoderLayer(state_size, num_heads, hidden_size, dropout)
        self.layers = clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.size, eps=1e-6)

    def forward(self, x, mask):
        output = x
        for layer in self.layers:
            output = layer(output, mask)
        return self.norm(output)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, size, num_heads, hidden_size, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.norm1 = nn.LayerNorm(size, eps=1e-6)
        self.self_attention = MultiHeadAttention(size, num_heads, dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(size, eps=1e-6)
        self.source_attention = MultiHeadAttention(size, num_heads, dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.feed_forward = PositionWiseFeedForward(size, hidden_size, dropout)

    def forward(self, encoded_source, target, source_mask, target_mask):
        target_norm = self.norm1(target)
        query = self.self_attention(target_norm, target_norm, target_norm, target_mask)
        query = self.dropout1(query) + target
        query_norm = self.norm2(query)
        output = self.source_attention(query_norm, encoded_source, encoded_source, source_mask)
        output = self.feed_forward(self.dropout2(output) + query)
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, state_size, num_heads, hidden_size, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        layer = TransformerDecoderLayer(state_size, num_heads, hidden_size, dropout)
        self.layers = clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, encoded_source, x, source_mask, target_mask):
        output = x
        for layer in self.layers:
            output = layer(encoded_source, output, source_mask, target_mask)
        return self.norm(output)


class TransformerGenerator(nn.Module):

    def __init__(self, state_size, vocab_size):
        super(TransformerGenerator, self).__init__()
        self.linear = nn.Linear(state_size, vocab_size)

    def forward(self, x):
        # return self.linear(x)
        return F.log_softmax(self.linear(x), dim = -1)


class TransformerEncoderDecoder(nn.Module):

    def __init__(self, source_vocab_size, target_vocab_size, num_layers, state_size, num_heads, hidden_size,
                 pad_index, dropout=0.1, source_position=True, target_position=True):
        super(TransformerEncoderDecoder, self).__init__()
        self.state_size = state_size
        self.encoder = TransformerEncoder(num_layers, state_size, num_heads, hidden_size, dropout)
        self.decoder = TransformerDecoder(num_layers, state_size, num_heads, hidden_size, dropout)
        source_embedding = ScaledEmbedding(source_vocab_size, state_size, pad_index)
        if source_position:
            self.source_embedding = nn.Sequential(source_embedding, PositionalEncoding(state_size, dropout))
        else:
            self.source_embedding = source_embedding
        target_embedding = ScaledEmbedding(target_vocab_size, state_size, pad_index)
        if target_position:
            self.target_embedding = nn.Sequential(target_embedding, PositionalEncoding(state_size, dropout))
        else:
            self.target_embedding = target_embedding
        self.generator = TransformerGenerator(state_size, target_vocab_size)

    def encode(self, source, source_mask):
        return self.encoder(self.source_embedding(source), source_mask)

    def decode(self, encoded_source, target, source_mask, target_mask):
        return self.decoder(encoded_source, self.target_embedding(target), source_mask, target_mask)

    def forward(self, source, target, source_mask, target_mask):
        encoded_source = self.encode(source, source_mask)
        return self.generator(self.decode(encoded_source, target, source_mask, target_mask))


class Transformer(nn.Module):

    def __init__(self, source_vocab, target_vocab, num_layers, state_size, num_heads, hidden_size, dropout=0.1,
                 source_position=True, target_position=True):
        super(Transformer, self).__init__()
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.state_size = state_size
        source_vocab_size = source_vocab.num_words
        target_vocab_size = target_vocab.num_words
        self.encoder = TransformerEncoder(num_layers, state_size, num_heads, hidden_size, dropout)
        self.decoder = TransformerDecoder(num_layers, state_size, num_heads, hidden_size, dropout)
        source_embedding = ScaledEmbedding(source_vocab_size, state_size, source_vocab.PAD_INDEX)
        if source_position:
            self.source_embedding = nn.Sequential(source_embedding, PositionalEncoding(state_size, dropout))
        else:
            self.source_embedding = source_embedding
        target_embedding = ScaledEmbedding(target_vocab_size, state_size, target_vocab.PAD_INDEX)
        if target_position:
            self.target_embedding = nn.Sequential(target_embedding, PositionalEncoding(state_size, dropout))
        else:
            self.target_embedding = target_embedding
        self.generator = TransformerGenerator(state_size, target_vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _source_mask(self, source):
        return (source != self.source_vocab.PAD_INDEX).unsqueeze(1)

    def _target_length_mask(self, target):
        return (target != self.target_vocab.PAD_INDEX).unsqueeze(1)

    def _subsequent_mask(self, size, device):
        shape = (1, size, size)
        mask = np.triu(np.ones(shape), k=1).astype(np.uint8)
        return torch.from_numpy(mask).to(device) == 0

    def _target_mask(self, target):
        return self._target_length_mask(target) & self._subsequent_mask(target.size(-1), target.device)

    def encode(self, source, source_mask):
        return self.encoder(self.source_embedding(source), source_mask)

    def decode(self, encoded_source, target, source_mask, target_mask):
        return self.decoder(encoded_source, self.target_embedding(target), source_mask, target_mask)

    def forward(self, source, target = None, source_mask = None, target_mask = None, type = 'teacher_forcing', max_length = None, beam_size = None):
        if (type == 'teacher_forcing'):
            if (source_mask is None):
                source_mask = self._source_mask(source)
            if (target_mask is None):
                target_mask = self._target_mask(target)
            encoded_source = self.encode(source, source_mask)
            return self.generator(self.decode(encoded_source, target, source_mask, target_mask))
        elif (type == 'beam_search'):
            return self.beam_search(source, beam_size, max_length)
        elif (type == 'greedy'):
            return self.greedy_decode(source, max_length)

    def greedy_decode_with_lengths(self, source, source_mask, lengths):
        batch_size = source.size(0)
        encoded_source = self.model.encode(source, source_mask)

        max_length = lengths.max().item()
        target = torch.full([batch_size, 1], self.target_vocab.START_END_INDEX, dtype=torch.long, device=source.device)
        output = torch.zeros([batch_size, max_length, self.target_vocab.num_words], device=source.device)

        for i in range(max_length):
            running = i < lengths
            target_mask = self._target_mask(target[running])
            logits = self.model.generator(self.model.decode(encoded_source[running], target[running],
                                                            source_mask[running], target_mask))
            output[running, i] = logits[:, -1]
            next_word = torch.argmax(output[:, i], -1).unsqueeze(1)
            target = torch.cat([target, next_word], dim=-1)
        return output

    def greedy_decode(self, source, max_length):
        batch_size = source.size(0)
        source_mask = self._source_mask(source)
        encoded_source = self.encode(source, source_mask)

        running = torch.ones(batch_size, dtype=torch.bool, device=source.device)
        target = torch.full([batch_size, 1], self.target_vocab.START_END_INDEX, dtype=torch.long, device=source.device)
        output = torch.full([batch_size, max_length], self.target_vocab.PAD_INDEX, dtype=torch.long,
                            device=source.device)

        for i in range(max_length):
            target_mask = self._target_mask(target[running])
            logits = self.generator(self.decode(encoded_source[running], target[running],
                                                source_mask[running], target_mask))
            next_word = torch.argmax(logits[:, -1], -1).detach()
            output[running, i] = next_word
            running &= output[:, i] != self.target_vocab.START_END_INDEX
            if not running.any():
                break
            target = torch.cat([target, output[:, i].unsqueeze(1)], dim=-1)
        return output

    def beam_search(self, source, beam_size, max_length):
        assert source.size(0) == 1 # currently works only with batch size equal to 1
        source_mask = self._source_mask(source)
        encoded_source = self.encode(source, source_mask)

        target = torch.full([1, 1], self.target_vocab.START_END_INDEX, dtype = torch.long, device = source.device)
        output = torch.full([beam_size, max_length], self.target_vocab.PAD_INDEX, dtype = torch.long, device = source.device)

        target_mask = self._target_mask(target)
        next_log_probs = self.generator(self.decode(encoded_source, target, source_mask, target_mask))[:, -1]
        most_probable = torch.topk(next_log_probs.squeeze(), beam_size)
        log_probs = most_probable.values.unsqueeze(1)
        next_word = most_probable.indices

        output[:, 0] = next_word
        running = next_word != self.target_vocab.START_END_INDEX

        target = torch.cat([target.repeat(beam_size, 1), next_word.unsqueeze(1)], dim = -1)

        encoded_source = encoded_source.repeat(beam_size, 1, 1)
        source_mask = source_mask.repeat(beam_size, 1, 1)

        for i in range(1, max_length):
            target_mask = self._target_mask(target[running])
            result = self.generator(self.decode(encoded_source[running], target[running],
                                                source_mask[running], target_mask))[:, -1]

            next_log_probs = torch.full([beam_size, self.target_vocab.num_words], float('-Inf'), device = source.device)
            next_log_probs[running] = result
            next_log_probs[~running, self.target_vocab.PAD_INDEX] = 0

            # print(log_probs)
            # print(next_log_probs)
            # print(log_probs + next_log_probs)

            most_probable = torch.topk((log_probs + next_log_probs).flatten(), beam_size)
            log_probs = most_probable.values.unsqueeze(1)
            beams = most_probable.indices // self.target_vocab.num_words
            next_word = most_probable.indices % self.target_vocab.num_words

            # print(target)
            # print(beams)
            # print(next_word)

            output = output.index_select(dim = 0, index = beams)
            running = running.index_select(dim = 0, index = beams)

            output[:, i] = next_word
            output[~running, i] = self.target_vocab.PAD_INDEX

            running &= next_word != self.target_vocab.START_END_INDEX
            if not running.any():
                break
            target = torch.cat([target.index_select(dim = 0, index = beams), next_word.unsqueeze(1)], dim=-1)

        return output