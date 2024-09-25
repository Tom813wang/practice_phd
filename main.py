from generative_model import VAE_Encoder, VAE_Decoder, VAE
from process_data import Vocabulary, TextPreprocessor, RadonpyDataLoader
from training import Data_splitter, Training

import torch
import torch.nn as nn

def main():
    # Define the parameters
    word_count = 10000
    d_model = 512
    latent_dim = 32
    nhead = 1
    hidden_dim = 256
    output_dim = 10000
    num_layers = 1
    batch_size = 32
    epochs = 10
    lr = 0.001
    clip = 1

    # Initialize the model
    encoder = VAE_Encoder(word_count, d_model, latent_dim, nhead)
    decoder = VAE_Decoder(word_count, latent_dim, d_model, hidden_dim, output_dim, num_layers)
    model = VAE(encoder, decoder)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize the loss function
    loss_function = nn.CrossEntropyLoss()

    # Initialize the training class
    trainer = Training(model=model, 
                       optimizer=optimizer, 
                       loss_function=loss_function, 
                       epochs=epochs, 
                       clip=clip)

    # Load the data
    data = RadonpyDataLoader('data.txt')
    data.load_data()
    data.preprocess_data()
    data.create_vocab()
    data.create_data_loader(batch_size=batch_size)

    # Train the model
    trainer.train(data.train_loader, data.test_loader)