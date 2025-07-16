import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, output_dim):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.encoder = Encoder(self.input_dim, self.condition_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.condition_dim, self.output_dim)
    
    def forward(self, x, condition):
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z, condition)
        return reconstructed_x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def generate_data(self, condition):
        '''
        input:
            condition: tensor(bs*t,condition_dim)
        output:
            generated_data: tensor(bs*t,output_dim)
        '''
        z = torch.randn(condition.shape[0], self.latent_dim).to(condition.device)

        with torch.no_grad():
            generated_data = self.decoder(z=z, condition=condition)

        return generated_data

