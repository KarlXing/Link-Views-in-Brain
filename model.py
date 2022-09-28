import torch
import torch.nn as nn
import torch.nn.functional as f
from utils import reparameterize

class CameraNet(nn.Module):
    def __init__(self, hidden_units = 128, num_out = 9):
        super(CameraNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2), nn.ReLU()
        )        
        self.fc1 = nn.Linear(32*7*7, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_out)
    
    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = f.relu(self.fc1(x))
        x = f.sigmoid(self.fc2(x))
        return x

class SimNet(nn.Module):
    def __init__(self, hidden_units = 128, num_out = 9):
        super(SimNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 4, stride=2), nn.ReLU()
        )
        self.fc1 = nn.Linear(32*6*7, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_out)
    
    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = f.relu(self.fc1(x))
        x = f.sigmoid(self.fc2(x))
        return x

class SimCameraNet(nn.Module):
    def __init__(self, hidden_units = 256, num_out = 9):
        super(SimCameraNet, self).__init__()
        self.main1 = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 4, stride=2), nn.ReLU()
        )
        self.main2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2), nn.ReLU()
        )
        self.fc1 = nn.Linear(32*6*7 + 32*7*7, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_out)

    def forward(self, x1, x2):
        x1, x2 = torch.flatten(self.main1(x1), start_dim=1), torch.flatten(self.main2(x2), start_dim=1)
        x = torch.cat((x1, x2), dim=1)
        x = f.relu(self.fc1(x))
        x = f.sigmoid(self.fc2(x))
        return x

class CameraEncoder(nn.Module):
    def __init__(self, flatten_size = 32*7*7, latent_size = 10, input_channel=3):
        super(CameraEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2), nn.ReLU()
        )        
        self.linear_mu = nn.Linear(flatten_size, latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, latent_size)
    
    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        return mu, logsigma

class SimEncoder(nn.Module):
    def __init__(self, flatten_size = 32*6*7, latent_size = 10, input_channel=3):
        super(SimEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 4, stride=2), nn.ReLU()
        )
        self.linear_mu = nn.Linear(flatten_size, latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, latent_size)
    
    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        return mu, logsigma

class CameraDecoder(nn.Module):
    def __init__(self, latent_size = 10, output_channel = 3, flatten_size=1024):
        super(CameraDecoder, self).__init__()

        self.fc = nn.Linear(latent_size, flatten_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(flatten_size, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, output_channel, 6, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.main(x)
        return x

class SimDecoder(nn.Module):
    def __init__(self, latent_size = 10, output_channel = 3, flatten_size=1024):
        super(SimDecoder, self).__init__()

        self.fc = nn.Linear(latent_size, flatten_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 6, stride=3), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (5,6), stride=3), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (5,8), stride=3), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.main(x)
        return x

class GeneralEncoder(nn.Module):
    def __init__(self, flatten_size = 32*6*6, latent_size = 10, input_channel=3):
        super(GeneralEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 4, stride=2), nn.ReLU()
        )
        self.linear_mu = nn.Linear(flatten_size, latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, latent_size)
    
    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        return mu, logsigma
    

class GeneralDecoder(nn.Module):
    def __init__(self, latent_size = 10, output_channel = 3, flatten_size=1024):
        super(GeneralDecoder, self).__init__()

        self.fc = nn.Linear(latent_size, flatten_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 6, stride=3), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=3), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=3), nn.ReLU(),
            nn.ConvTranspose2d(32, output_channel, 6, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.main(x)
        return x


class BiTransformNet(nn.Module):
    def __init__(self, latent_size=10, input_channel=3):
        super(BiTransformNet, self).__init__()
        self.encoder = GeneralEncoder(latent_size=latent_size, input_channel=input_channel)
        self.decoder = GeneralDecoder(latent_size=latent_size)
    
    def forward(self, x):
        mu, logsigma = self.encoder(x)
        latent = reparameterize(mu, logsigma)
        converted_x = self.decoder(latent)
        return mu, logsigma, converted_x

    def forward_mask(self, x, mask):
        mu, logsigma = self.encoder(x)
        latent = mu * mask
        converted_x = self.decoder(latent)
        return mu, logsigma, converted_x


class Sim2Camera(nn.Module):
    def __init__(self, latent_size=10, input_channel=3):
        super(Sim2Camera, self).__init__()
        self.encoder = SimEncoder(latent_size=latent_size, input_channel=input_channel)
        self.decoder = CameraDecoder(latent_size=latent_size)
    
    def forward(self, x):
        mu, logsigma = self.encoder(x)
        latent = reparameterize(mu, logsigma)
        converted_x = self.decoder(latent)
        return mu, logsigma, converted_x

    def forward_mask(self, x, mask):
        mu, logsigma = self.encoder(x)
        latent = mu * mask
        converted_x = self.decoder(latent)
        return mu, logsigma, converted_x

class Camera2Sim(nn.Module):
    def __init__(self, latent_size=10, input_channel=3):
        super(Camera2Sim, self).__init__()
        self.encoder = CameraEncoder(latent_size=latent_size, input_channel=input_channel)
        self.decoder = SimDecoder(latent_size=latent_size)
    
    def forward(self, x):
        mu, logsigma = self.encoder(x)
        latent = reparameterize(mu, logsigma)
        converted_x = self.decoder(latent)
        return mu, logsigma, converted_x
    
    def forward_mask(self, x, mask):
        mu, logsigma = self.encoder(x)
        latent = mu * mask
        converted_x = self.decoder(latent)
        return mu, logsigma, converted_x
    

class TransformNet(nn.Module):
    def __init__(self, latent_size=10):
        super(TransformNet, self).__init__()
        self.camera_encoder = CameraEncoder(latent_size=latent_size)
        self.sim_encoder = SimEncoder(latent_size=latent_size)
        self.camera_decoder = CameraDecoder(latent_size=latent_size)
        self.sim_decoder = SimDecoder(latent_size=latent_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        latent = reparameterize(mu, logsigma)
        converted_x = self.decoder(latent)
        return mu, logsigma, converted_x

    def c2s_forward(self, x):
        mu, logsigma = self.camera_encoder(x)
        latent = reparameterize(mu, logsigma)
        converted_x = self.sim_decoder(latent)
        return mu, logsigma, converted_x

    def s2c_forward(self, x):
        mu, logsigma = self.sim_encoder(x)
        latent = reparameterize(mu, logsigma)
        converted_x = self.camera_decoder(latent)
        return mu, logsigma, converted_x