#!/usr/bin/env python
"""Defines the Vocabulary class
"""

class Vocabulary(object):
    def __init__(self, lowercase=True):
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}
        self.nwords = 0
        self.lowercase = lowercase

    @classmethod
    def from_idx2word_dict(cls, idx2word_dict, lowercase=True):
        instance = cls(lowercase)
        instance.idx2word = idx2word_dict
        for k, v in idx2word_dict.items():
            instance.word2idx[v] = k
        instance.nwords = len(idx2word_dict)
        instance.lowercase = lowercase
        return instance

    def idx_to_word(self, idx):
        return self.idx2word[idx] if idx in self.idx2word else '<unk>'

    def __call__(self, word):
        """
        Returns the id corresponding to the word
        """
        w = word.lower() if self.lowercase else word
        return self.word2idx['<unk>'] if w not in self.word2idx else self.word2idx[w]

    def __len__(self):
        """
        Get the number of words in the vocabulary
        """
        return self.nwords
