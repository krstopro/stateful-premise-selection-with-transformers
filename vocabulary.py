class Vocabulary:
    PAD_INDEX = 0

    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = { self.PAD_INDEX: '<PAD>' }
        self.num_words = 1
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            index = self.num_words
            self.word2index[word] = index
            self.word2count[word] = 1
            self.index2word[index] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        words = sentence.split(' ')
        for word in words:
            self.add_word(word)
        self.longest_sentence = max(self.longest_sentence, len(words))
        self.num_sentences += 1

    def index2word(self, index):
        return self.index2word[index]

    def word2index(self, word):
        return self.word2index[word]

    def sentence2indices(self, sentence):
        pass

class SourceVocabulary(Vocabulary):

    def sentence2indices(self, sentence):
        return [ self.word2index[word] for word in sentence.split(' ') ]

# class SourceVocabulary(Vocabulary):
#     START_END_INDEX = 1
#
#     def __init__(self):
#         super().__init__()
#         self.index2word[self.START_END_INDEX] = '<START/END>'
#         self.num_words += 1
#
#     def sentence2indices(self, sentence):
#         indices = [ self.word2index[word] for word in sentence.split(' ') ]
#         indices.append(self.START_END_INDEX)
#         return indices

class TargetVocabulary(Vocabulary):
    START_END_INDEX = 1

    def __init__(self):
        super().__init__()
        self.index2word[self.START_END_INDEX] = '<START/END>'
        self.num_words += 1

    def sentence2indices(self, sentence):
        indices = [ self.START_END_INDEX ]
        for word in sentence.split(' '):
            indices.append(self.word2index[word])
        indices.append(self.START_END_INDEX)
        return indices

# class TargetVocabulary(Vocabulary):
#     START_INDEX = 1
#     END_INDEX = 2
#
#     def __init__(self):
#         super().__init__()
#         self.index2word[self.START_INDEX] = '<START>'
#         self.index2word[self.END_INDEX] = '<END>'
#         self.num_words += 2
#
#     def sentence2indices(self, sentence):
#         indices = [ self.START_END_INDEX ]
#         for word in sentence.split(' '):
#             indices.append(self.word2index[word])
#         indices.append(self.START_END_INDEX)
#         return indices