import torch
import torch.nn as nn


class VAE_Encoder(nn.Module):
    def __init__(self, word_count, d_model, latent_dim, nhead=1):
        super(VAE_Encoder, self).__init__()
        self.embedding = nn.Embedding(word_count, d_model)
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        # Fully connected layers for mean and log variance of latent space
        self.fc_mu = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU()
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU()
        )
        
    def reparameter(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, sigma^2)"""
        std = torch.exp(0.5 * logvar)  # Compute standard deviation
        eps = torch.randn_like(std)    # Sample epsilon from standard normal
        return mu + eps * std          # Return reparameterized z

    def forward(self, x):
        # x has shape (batch_size, seq_length)
        x = self.embedding(x)  # Shape: (batch_size, seq_length, d_model)
        x = self.encoder(x)    # Shape: (batch_size, seq_length, d_model)
        x = torch.mean(x, dim=1)  # Shape: (batch_size, d_model)
        # Calculate mean and log variance of latent space
        mu = self.fc_mu(x)        # Shape: (batch_size, latent_dim)
        logvar = self.fc_logvar(x)  # Shape: (batch_size, latent_dim)
        # Reparameterization to sample z from the latent space
        z = self.reparameter(mu, logvar)  # Shape: (batch_size, latent_dim)
        return z, mu, logvar

    
class VAE_Decoder(nn.Module):
    def __init__(self, word_count, latent_dim, d_model, hidden_dim ,output_dim, num_layers=1):
        super(VAE_Decoder, self).__init__()
        # Define the parameters
        self.word_count = word_count
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Define the layers
        self.embedding = nn.Embedding(word_count, d_model)
        self.decoder = nn.RNN(input_size=d_model, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, hidden):
        x = self.embedding(x)
        hidden = self.linear_transform(x, hidden)
        x, hidden = self.decoder(x, hidden)
        x = self.fc(x)
        x = self.softmax(x)
        return x, hidden

    def get_linear_layer(self, **kwargs):
        layer_dims = kwargs.get('layer_dims', None)

        if layer_dims is None:
            raise ValueError("Layer dimensions must be provided")
        
        if len(layer_dims) >3:
            raise ValueError("Only 3 layer are supported")
        
        layers = []
        for i in range(len(layer_dims)-1):
            layer = nn.Linear(layer_dims[i], layer_dims[i+1])
            layers.append(layer)
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def linear_transform(self, x, hidden):
        """
        The hidden size should be (num_layers, batch_size, hidden_dim).
        The x size should be (batch_size, seq_len, d_model).
        For the first time step, the x should be the <start> token.

        x = torch.randn(10, 100, 64)        # Batch size 10, sequence length 12, input size 64
        hidden = torch.randn(1, 10, 128)    # num_layers 1, batch size 10, hidden size 128
        """
        
        if hidden.shape[1] == x.shape[0] and hidden.shape[2] == self.hidden_dim:
            print('The hidden satisfies the requirement')
            return hidden

        # If latent dimension is not the same as embedding dimension, we need to do the linear transformation
        if hidden.shape[2] != self.hidden_dim:
            print('Warning: The latent dimension is not the same as the hidden dimension, we do the linear transformation')
            new_linear = self.get_linear_layer(layer_dims=[self.latent_dim, self.hidden_dim])
            hidden = new_linear(hidden)
            print('After linear transformation, the hidden shape is: ', hidden.shape)
            return hidden

        
        
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, 
                src: torch.Tensor,
                trg: torch.Tensor,
                teacher_forcing: bool = False,
                teacher_forcing_ratio: float = 0.5):
        
        batch_size = trg.shape[0]
        seq_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        output_seq = torch.zeros(batch_size, seq_len, trg_vocab_size).to(src.device)

        z, mu, logvar = self.encoder(src)

        hidden = z.unsqueeze(0)
        trg_input = trg[:, 0].unsqueeze(1)

        for t in range(1, seq_len):
            out, hidden = self.decoder(trg_input, hidden)
            # Shape of out is (batch_size, 1, trg_vocab_size)
            output_seq[:, t] = out.squeeze(1)

            if teacher_forcing:
                p_teacher_forcing = torch.rand(1).item()
                if p_teacher_forcing < teacher_forcing_ratio:
                    trg_input = trg[:, t].unsqueeze(1)

                else:
                    trg_input = out.argmax(dim=2)
            else:
                trg_input = out.argmax(dim=2)

        return z, mu, logvar, output_seq, hidden

    

    


        
