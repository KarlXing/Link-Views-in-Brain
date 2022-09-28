import torch
import numpy as np
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argparse
import os
import numpy as np
import math
from utils import WebotDataset, seed_everything, LinearScheduler, save_args
from model import BiTransformNet

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default=None)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save-path', type=str, default='checkpoints/transform.pt')
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--img-freq', type=int, default=100)
parser.add_argument('--latent-size', type=int, default=20)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--save-freq', type=int, default=2000)
parser.add_argument('--att-weight', type=float, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start-time', type=int, default=50)
parser.add_argument('--start-value', type=int, default=0)
parser.add_argument('--end-time', type=int, default=1000)
parser.add_argument('--end-value', type=int, default=1)
parser.add_argument('--log-freq', type=int, default=10)

args = parser.parse_args()
args.sim_train_path = os.path.join(args.data_path, 'sim_train.npy')
args.sim_test_path = os.path.join(args.data_path, 'sim_test.npy') 
args.camera_train_path = os.path.join(args.data_path, 'camera_train.npy')
args.camera_test_path = os.path.join(args.data_path, 'camera_test.npy') 
args.label_train_path = os.path.join(args.data_path, 'pos_train.npy')
args.label_test_path = os.path.join(args.data_path, 'pos_test.npy') 
args.mask_train_path = os.path.join(args.data_path, 'mask_train.npy')
args.mask_test_path = os.path.join(args.data_path, 'mask_test.npy')

seed_everything(args.seed)


def vae_loss(x, mu, logsigma, recon_x, beta=1, mask=None):
    if mask is None:
        recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    else:
        recon_loss = (x - recon_x)**2
        weight_mask = (1 - mask) + mask * args.att_weight
        weight_mask = weight_mask.unsqueeze(1) * (torch.numel(weight_mask) / torch.sum(weight_mask))
        weight_mask = weight_mask.expand(-1, 3, -1, -1)
        assert(weight_mask.shape == recon_loss.shape)
        recon_loss = torch.mean(recon_loss * weight_mask)

    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta, recon_loss, kl_loss


def main():
    ########### Preparation ##########
    # 1. device, dataset and dataloader
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = WebotDataset(args.sim_train_path, args.camera_train_path, args.label_train_path, args.mask_train_path, device)
    test_dataset = WebotDataset(args.sim_test_path, args.camera_test_path, args.label_test_path, args.mask_test_path, device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. model and optimizer
    Net = BiTransformNet
    model = Net(latent_size=args.latent_size)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ########### Train ############
    # 3. log
    writer = SummaryWriter(comment=args.tag)
    save_args(args, writer.log_dir)

    # 4. main train
    step = 0
    scheduler = LinearScheduler(args.start_time, args.start_value, args.end_time, args.end_value)
    for i_epoch in range(args.epochs):
        print('Done Epochs : %d' % i_epoch)
        scheduler.step()
        for i_batch, (sim_data, camera_data, _, mask) in enumerate(train_loader):
            step += 1

            all_data = torch.cat([sim_data, camera_data], dim=0)
            mu, logsigma, predict = model(all_data)
            
            mu_s2c, logsigma_s2c, predict_camera = mu[:sim_data.shape[0]], logsigma[:sim_data.shape[0]], predict[:sim_data.shape[0]]
            s2c_loss, s2c_recon_loss, s2c_kl_loss = vae_loss(camera_data, mu_s2c, logsigma_s2c, predict_camera, args.beta * scheduler.val(), None)

            mu_c2s, logsigma_c2s, predict_sim = mu[-camera_data.shape[0]:], logsigma[-camera_data.shape[0]:], predict[-camera_data.shape[0]:]
            c2s_loss, c2s_recon_loss, c2s_kl_loss = vae_loss(sim_data, mu_c2s, logsigma_c2s, predict_sim, args.beta * scheduler.val(), mask)

            optim.zero_grad()
            loss = (s2c_loss + c2s_loss)
            loss.backward()
            optim.step()    

        if (i_epoch + 1) % args.log_freq == 0:
            writer.add_scalar('s2c_loss', s2c_loss.detach().cpu().item(), global_step=step)
            writer.add_scalar('s2c_recon_loss', s2c_recon_loss.detach().cpu().item(), global_step=step)
            writer.add_scalar('s2c_kl_loss', s2c_kl_loss.detach().cpu().item(), global_step=step)

            writer.add_scalar('c2s_loss', c2s_loss.detach().cpu().item(), global_step=step)
            writer.add_scalar('c2s_recon_loss', c2s_recon_loss.detach().cpu().item(), global_step=step)
            writer.add_scalar('c2s_kl_loss', c2s_kl_loss.detach().cpu().item(), global_step=step)         
            
            writer.add_scalar('kl_annealing', scheduler.val(), global_step=step)

        if (i_epoch + 1) % args.img_freq == 0:
            writer.add_images('sim_img', sim_data[:8], global_step=i_epoch)
            writer.add_images('camera_img', camera_data[:8], global_step=i_epoch)
            writer.add_images('sim_transform', predict_sim[:8], global_step=i_epoch)
            writer.add_images('camera_transform', predict_camera[:8], global_step=i_epoch)

        if (i_epoch + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), '%s-%d.pt' % (args.save_path, i_epoch))

    # 5. save
    torch.save(model.state_dict(), args.save_path)
    writer.close()


if __name__ == '__main__':
    main()