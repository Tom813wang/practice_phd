import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split

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
        return self.word2index[word]  # Return the index directly

    def create_vocabulary(self, text):
        for sentence in text:
            for word in sentence:
                self.add_word(word)

        for token in ['<start>', '<end>', '<pad>']:
            self.add_word(token)

        return self.word2index, self.index2word, self.word_counts
    
    def get_index(self, word):
        return self.word2index.get(word, None)  # Use .get() to avoid KeyError
    
    def get_word(self, index):
        return self.index2word.get(index, None)  # Use .get() for safety
    
    def __len__(self):
        return len(self.index2word)

class TextPreprocessor:
    def __init__(self, vocab):
        self.vocab = vocab

    def pad_sequences(self, text_index):
        max_length = max(len(sentence) for sentence in text_index)
        return [sentence + [self.vocab.get_index('<pad>')] * (max_length - len(sentence)) for sentence in text_index]

    def text_to_index(self, text, padding=False, start=False, end=False):
        text_index = []
        for sentence in text:
            sentence_index = []
            if start:
                sentence_index.append(self.vocab.get_index('<start>'))
            sentence_index.extend(self.vocab.get_index(word) for word in sentence)
            if end:
                sentence_index.append(self.vocab.get_index('<end>'))

            text_index.append(torch.tensor(sentence_index, dtype=torch.int64))

        if padding:
            text_index = self.pad_sequences(text_index)

        return text_index

class RadonpyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class RadonpyDataLoader:
    def __init__(self, data, label, batch_size, shuffle, vocab):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vocab = vocab

    def split_data(self, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.label, test_size=test_size)
        return x_train, x_test, y_train, y_test

    def collate_fn(self, batch):
        # Get data and label
        data, label = zip(*batch)
        data = pad_sequence(data, batch_first=True, padding_value=self.vocab.get_index('<pad>'))
        label = pad_sequence(label, batch_first=True, padding_value=self.vocab.get_index('<pad>'))
        return data, label
        
    def get_dataloader(self):
        """
        Create the DataLoader for the training and test dataset
        """
        # First, create the train and test dataset
        train_dataset = RadonpyDataset(self.x_train, self.y_train)
        test_dataset = RadonpyDataset(self.x_test, self.y_test)

        # Then, create the DataLoader
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=self.shuffle, 
                                  collate_fn=self.collate_fn)
        
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle,
                                 collate_fn=self.collate_fn)
        
        return train_loader, test_loader