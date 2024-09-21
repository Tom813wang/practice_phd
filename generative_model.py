import torch
import torch.nn as nn

from colorama import init, Fore, Back, Style
init(autoreset=True)  # 自动重置颜色

class VAE_Encoder(nn.Module):
    def __init__(self, word_count, d_model,  latent_dim, nhead=1):
        super(VAE_Encoder, self).__init__()
        self.embedding = nn.Embedding(word_count, d_model)
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
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
        hidden = self.linear_transform(hidden)
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

    def linear_transform(self, hidden):
        """
        The hidden size should be (num_layers, batch_size, hidden_dim).
        The x size should be (batch_size, seq_len, d_model).
        For the first time step, the x should be the <start> token.
        """
        if hidden.shape[1] == 1 and self.latent_dim == self.hidden_dim:
            print('The hidden satisfies the requirement')
            return hidden

        # If the hidden is not the last time step, we need to extract the last time step
        if hidden.shape[1] != 1:
            print(Fore.GREEN,'Warning: The hidden shape is not the last time step, we extract the last time step')
            hidden = hidden[:, -1, :].unsqueeze(0)

        # If latent dimension is not the same as embedding dimension, we need to do the linear transformation
        if self.latent_dim != self.hidden_dim:
            print('Warning: The latent dimension is not the same as the hidden dimension, we do the linear transformation')
            new_linear = self.get_linear_layer(layer_dims=[self.latent_dim, self.hidden_dim])
            hidden = new_linear(hidden)
        
        print("After linear transformation, the hidden shape is: ", hidden.shape)

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
        
        seq_len = trg.shape[1]
        batch_size = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        output_seq = torch.zeros(seq_len, batch_size, trg_vocab_size).to(src.device)

        z, mu, logvar = self.encoder(src)
        hidden = self.decoder.linear_transform(z)

        x = trg[:, 0].unsqueeze(1)

        for t in range(1, seq_len):
            output, hidden = self.decoder(x, hidden)
            output_seq[t] = output
            print(output[t])

            if teacher_forcing is True:
                p_teacher_forcing = torch.rand(1).item()
                if p_teacher_forcing > teacher_forcing_ratio:
                    x = trg[:, t].unsqueeze(1)
                else:
                    x = output.argmax(dim=2)
            else:
                x = output.argmax(dim=2)

        return z, mu, logvar, output_seq
    
    def generate(self, src, max_len):
        trg_vocab_size = self.decoder.output_dim
        seq_len = max_len
        batch_size = src.shape[0]
        output_seq = torch.zeros(seq_len, batch_size, trg_vocab_size).to(src.device)

        z, mu, logvar = self.encoder(src)
        hidden = self.decoder.linear_transform(z)

        x = src[:, 0].unsqueeze(1)

        for t in range(1, seq_len):
            output, hidden = self.decoder(x, hidden)
            output_seq[t] = output
            x = output.argmax(dim=2)

        return output_seq
    
if __name__ == '__main__':
    encoder = VAE_Encoder(d_model=5,
                          latent_dim=3,
                          word_count=10)
    decoder = VAE_Decoder(latent_dim=3,
                          d_model=5,
                          hidden_dim=5,
                          output_dim=10)
    model = VAE(encoder, decoder)
    src = torch.randint(0, 10, (1, 5))

    z, mu, logvar = model.encoder(src)
    out, hidden = model.decoder(src,  z)
    print(out.shape, hidden.shape)


    


        
