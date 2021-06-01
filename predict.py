import math
import numpy as np
import torch
from dataset import Statement2PremisesDataset, StatementDataset
from torch.utils.data import DataLoader
from collate import VarLengthCollate
from vocabulary import SourceVocabulary, TargetVocabulary

from transformer import Transformer, TransformerEncoderDecoder
# from lstm import EncoderDecoder
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(model):
    return NoamOpt(model.state_size, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')

def subsequent_mask(size):
    shape = (1, size, size)
    mask = np.triu(np.ones(shape), k = 1).astype(np.uint8)
    return torch.from_numpy(mask) == 0

def data_gen(high, batch_size, num_batches):
    for i in range(num_batches):
        data = torch.from_numpy(np.random.randint(2, high, size=(batch_size, 12)))
        data[:, 0] = 1
        data[:, -1] = 1
        source = data
        target = data.flip(1)
        # target_in = target[:, :-1]
        # target_out = target[:, 1:]
        # source_mask = (source != 0).unsqueeze(1)
        # target_mask = (target_in != 0).unsqueeze(1) & subsequent_mask(target_in.size(-1))
        yield source, target

def build_vocabs(source_path, target_path):
    source_vocab = SourceVocabulary()
    target_vocab = TargetVocabulary()
    with open(source_path, 'r') as source_file, open(target_path, 'r') as target_file:
        source_lines = []
        target_lines = []
        while True:
            source_line = source_file.readline().strip('\n')
            target_line = target_file.readline().strip('\n')
            if ((not source_line) or (not target_line)):
                break
            source_lines.append(source_line)
            target_lines.append(target_line)
            source_vocab.add_sentence(source_line)
            target_vocab.add_sentence(target_line)
    return source_vocab, target_vocab

def get_union_and_intersection_size(output, premises):
    union, counts = torch.cat([output, premises]).unique(return_counts = True)
    union_size = float(len(union))
    intersection_size = (counts > 1).sum().item()
    num_premises = float(len(premises))
    # jaccard = (intersection_size / union_size).item()
    # coverage = (intersection_size / num_premises).item()
    # print(output, premises)
    # print(union, counts)
    return union_size, intersection_size, num_premises

if (__name__ == '__main__'):
    train_source_path = './data/train/prefix.src'
    train_target_path = './data/train/prefix.tgt'
    model_path = './models/transformer_500'
    test_source_path = './data/test/prefix.src'
    predictions_path = './predictions/transformer_500.preds'
    source_vocab, target_vocab = build_vocabs(train_source_path, train_target_path)
    # train_dataset = Statement2PremisesDataset(train_source_path, train_target_path, source_vocab, target_vocab)
    # valid_dataset = Statement2PremisesDataset(valid_source_path, valid_target_path, source_vocab, target_vocab)
    test_dataset = StatementDataset(test_source_path, source_vocab)
    test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=10,
                                   collate_fn=VarLengthCollate(batch_dim=0), pin_memory=True)

    model = Transformer(source_vocab, target_vocab, 3, 512, 8, 2048, target_position = False)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to('cuda:0')

    model.load_state_dict(torch.load(model_path))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model parameters {}'.format(num_params))

    all_premises = []
    for batch in test_data_loader:
        x = batch['source'].to('cuda:0')
        output = model(source=x, type='beam_search', beam_size=10, max_length=64)
        top_premises = []
        for indices in output:
            premises = []
            for index in indices:
                if (index != target_vocab.START_END_INDEX):
                    premises.append(target_vocab.index2word[index.item()])
                else:
                    break
            premises = ' '.join(premises)
            top_premises.append(premises)
        top_premises = '\n'.join(top_premises)
        all_premises.append(top_premises)
    all_premises = '\n'.join(all_premises)
    with open(predictions_path, 'w') as predictions_file:
        predictions_file.write(all_premises)