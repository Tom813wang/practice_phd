import numpy as np
import pandas as pd

class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word_counts = 0

    def add_word(self, word):
        word = str(word)
        if word not in self.word2index:
            index = len(self.word2index)
            self.word2index[word] = index
            self.index2word[index] = word
            self.word_counts += 1

        return self.word2index, self.index2word, self.word_counts

    def create_vocabulary(self, text):
        for sentence in text:
            for word in sentence:
                self.word2index, self.index2word, self.word_counts = self.add_word(word)

        return self.word2index, self.index2word, self.word_counts
    
    def get_index(self, word):
        return self.word2index[word]
    
    def get_word(self, index):
        return self.index2word[index]   

    def __len__(self):
        return len(self.index2word)

class TextPreprocess(Vocabulary):
    def __init__(self):
        super().__init__()
        self.text = None
    
    def padding(self, text_index):
        max_length = max([len(sentence) for sentence in text_index])
        max_index = max([max(sentence) for sentence in text_index])
        padded_text = []
        for sentence in text_index:
            if len(sentence) < max_length:
                sentence += [max_index+1]*(max_length - len(sentence))
            padded_text.append(sentence)
        return padded_text
        
    def text_to_index(self,
                       text: list,
                       padding : bool = False):
        
        text_index = []
        self.word2index, self.index2word, self.word_counts = self.create_vocabulary(text)
        for sentence in text:
            sentence_index = [self.get_index(word) for word in sentence]
            text_index.append(sentence_index)
        
        if padding is True:
            text_index = self.padding(text_index)
            text_index = np.array(text_index)
            return text_index
        else:
            return text_index