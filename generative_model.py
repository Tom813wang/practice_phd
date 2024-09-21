import torch
import torch.nn as nn

class VAE_Encoder(nn.Module):
    def __init__(self, word_count, d_model,  latent_dim, nhead=1):
        super(VAE_Encoder, self).__init__()
        self.embedding = nn.Embedding(word_count, d_model)
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.fc_mu = nn.Sequential(
                                nn.Linear(d_model, latent_dim),
                                nn.ReLU()
        )
        self.fc_logvar = nn.Sequential(
                                nn.Linear(d_model, latent_dim),
                                nn.ReLU()
        )
        
    def reparameter(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameter(mu, logvar)
        return z, mu, logvar
    
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, d_model, hidden_dim ,output_dim, num_layers=1):
        super(VAE_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.decoder = nn.RNN(input_size=d_model, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        """
        the hidden size should be (num_layers, batch_size, hidden_dim)
        the x size should be (batch_size, seq_len, d_model)
        """
        # if the hidden is not the last time step, we need to extract the last time step
        if hidden.shape[1] != 1:
            hidden = hidden[:, -1, :].unsqueeze(0)
        else:
            hidden = hidden

        # if latent dimension is not the same as embedding dimension, we need to do the linear transformation
        if self.latent_dim != self.hidden_dim:
            new_linear = self.get_linear_layer(layer_dims=[self.latent_dim, self.hidden_dim])
            hidden = new_linear(hidden)

        x, hidden = self.decoder(x, hidden)
        x = self.fc(x)
        x = self.softmax(x)
        return x, hidden

    def get_linear_layer(self, number_layer=1, **kwargs):
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



        

if __name__ == '__main__':
    encoder = VAE_Encoder(d_model=512, latent_dim=128)
    decoder = VAE_Decoder(latent_dim=128, hidden_dim=128, output_dim=512)
    x = torch.randn(1, 10, 512)
    z, mu, logvar = encoder(x)
    hidden = z[:, -1, :].unsqueeze(0)
    start_signal = torch.zeros(1, 1, 512)
    x, hidden = decoder(start_signal, hidden)
    print(x.shape)

