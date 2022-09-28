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
from utils import WebotSeqDataset, seed_everything, LinearScheduler, save_args
from model import Sim2Camera, Camera2Sim

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default=None)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save-path', type=str, default='checkpoints/transform.pt')
parser.add_argument('--sim2camera', default=False, action='store_true')
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
args.sim_seq_train_path = os.path.join(args.data_path, 'sim_train_seq.npy')
args.sim_seq_test_path = os.path.join(args.data_path, 'sim_test_seq.npy') 
args.camera_seq_train_path = os.path.join(args.data_path, 'camera_train_seq.npy')
args.camera_seq_test_path = os.path.join(args.data_path, 'camera_test_seq.npy') 
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
    if args.sim2camera:
        input_path, target_path = args.sim_seq_train_path, args.camera_train_path
    else:
        input_path, target_path = args.camera_seq_train_path, args.sim_train_path
        
    dataset_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = WebotSeqDataset(input_path, target_path, args.label_train_path, args.mask_train_path, dataset_device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 2. model and optimizer
    Net = Sim2Camera if args.sim2camera else Camera2Sim
    model = Net(latent_size=args.latent_size, input_channel=15)
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
        for i_batch, (source_data, target_data, _, mask) in enumerate(train_loader):
            step += 1

            mu, logsigma, predict_target = model(source_data)

            loss, recon_loss, kl_loss = vae_loss(target_data, mu, logsigma, predict_target, args.beta * scheduler.val(), mask if not args.sim2camera else None)

            optim.zero_grad()
            loss.backward()
            optim.step()

        if (i_epoch + 1) % args.log_freq == 0:
            writer.add_scalar('loss', loss.detach().cpu().item(), global_step=step)
            writer.add_scalar('recon_loss', recon_loss.detach().cpu().item(), global_step=step)
            writer.add_scalar('kl_loss', kl_loss.detach().cpu().item(), global_step=step)
            writer.add_scalar('kl_annealing', scheduler.val(), global_step=step)
        
        if (i_epoch + 1) % args.img_freq == 0:
            writer.add_images('source_img', source_data[:8,-3:,:,:], global_step=i_epoch)
            writer.add_images('target_img', target_data[:8], global_step=i_epoch)
            writer.add_images('transform_img', predict_target[:8], global_step=i_epoch)
        
        if (i_epoch + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), '%s-%d.pt' % (args.save_path, i_epoch))

    # 5. save
    torch.save(model.state_dict(), args.save_path)
    writer.close()


if __name__ == '__main__':
    main()