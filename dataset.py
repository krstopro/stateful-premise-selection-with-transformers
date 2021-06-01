import numpy as np
from torch.utils.data import Dataset
from vocabulary import SourceVocabulary, TargetVocabulary

# class Statement2PremisesDataset(Dataset):
#
#     def __init__(self, source_path, target_path, source_vocab, target_vocab):
#         self.source_vocab = SourceVocabulary()
#         self.target_vocab = TargetVocabulary()
#         with open(source_path, 'r') as source_file, open(target_path, 'r') as target_file:
#             source_lines = []
#             target_lines = []
#             while True:
#                 source_line = source_file.readline().strip('\n')
#                 target_line = target_file.readline().strip('\n')
#                 if ((not source_line) or (not target_line)):
#                     break
#                 source_lines.append(source_line)
#                 target_lines.append(target_line)
#                 self.source_vocab.add_sentence(source_line)
#                 self.target_vocab.add_sentence(target_line)
#         self.statement_data = [ self.source_vocab.sentence2indices(line) for line in source_lines ]
#         self.premises_data = [ self.target_vocab.sentence2indices(line) for line in target_lines ]
#
#     def __len__(self):
#         return len(self.statement_data)
#
#     def __getitem__(self, item):
#         statement_indices = self.statement_data[item]
#         premises_indices = self.premises_data[item]
#         return { 'source': np.asarray(statement_indices, dtype = np.int64),
#                  'source_lengths': len(statement_indices),
#                  'target': np.asarray(premises_indices, dtype = np.int64),
#                  'target_lengths': len(premises_indices) }

class Statement2PremisesDataset(Dataset):

    def __init__(self, source_path, target_path, source_vocab, target_vocab):
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
        self.statement_data = [ source_vocab.sentence2indices(line) for line in source_lines ]
        self.premises_data = [ target_vocab.sentence2indices(line) for line in target_lines ]

    def __len__(self):
        return len(self.statement_data)

    def __getitem__(self, item):
        statement_indices = self.statement_data[item]
        premises_indices = self.premises_data[item]
        return { 'source': np.asarray(statement_indices, dtype = np.int64),
                 'source_lengths': len(statement_indices),
                 'target': np.asarray(premises_indices, dtype = np.int64),
                 'target_lengths': len(premises_indices) }