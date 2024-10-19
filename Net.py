import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        
        # A list to hold the shared hidden layers
        self.hidden_layers = nn.ModuleList()
        
        # Construct the hidden layers (fully connected layers)
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        print(x.shape)
        # Forward pass through the shared hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output the mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(VAEDecoder, self).__init__()
        
        # A list to hold the hidden layers
        self.hidden_layers = nn.ModuleList()
        # Construct the hidden layers (fully connected layers)
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        
        # Output layer to reconstruct the original data
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, z):
        # Forward pass through the hidden layers
        for layer in self.hidden_layers:
            z = F.relu(layer(z))
        
        # Output layer (typically a sigmoid if reconstructing binary data, or no activation for continuous)
        reconstruction = torch.sigmoid(self.output_layer(z))  # Use sigmoid for binary output
        
        return reconstruction

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
    
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dims[::-1], input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std