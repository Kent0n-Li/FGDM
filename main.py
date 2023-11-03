# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import torch
import numpy as np

import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.lmdb_datasets import LMDBDataset
from datasets_prep.ctmr_datasets import *

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from tool.eval_score import fid_l2_psnr_ssim


class Dataset_FGDM(data.Dataset):
    def __init__(self, path, size=None, test=None):
        self.img = os.listdir(path)
        random.shuffle(self.img)
        self.size = size
        self.path = path
        self.test = test

    def __getitem__(self, item):

        imagename = os.path.join(self.path, self.img[item])
        #print(imagename)
        npimg = cv2.imread(imagename)
        npimg = np.transpose(npimg, (2, 0, 1))[0]

        nplabs = npimg

        npimg = npimg / 255
        

        threshold_random = random.randint(1, 10)
        bilateralFilter_random = random.randint(1, 30)
        

        nplabs = cv2.bilateralFilter(nplabs, 10, bilateralFilter_random, bilateralFilter_random)
        nplabs = np.uint8(nplabs)

        x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        
        nplabs[nplabs < threshold_random] = 0
        
        nplabs = (nplabs - nplabs.min() +1e-12) / (nplabs.max() - nplabs.min() +1e-8)
        npimg = npimg.astype(np.float32)
        nplabs = nplabs.astype(np.float32)

        resize = transforms.Resize([self.size, self.size])

        npimg = torch.from_numpy(np.expand_dims(npimg, 0))
        nplabs = torch.from_numpy(np.expand_dims(nplabs, 0))

        npimg = resize(npimg)
        nplabs = resize(nplabs)

        return npimg, nplabs


    def __len__(self):
        size = len(self.img)
        return size



class Dataset_test(data.Dataset):
    def __init__(self, path, size = None):
        self.img = os.listdir(path)
        self.size = size
        self.path = path

    def __getitem__(self, item):

        imagename = os.path.join(self.path, self.img[item])
        #print(imagename)
        npimg = cv2.imread(imagename)
        npimg = np.transpose(npimg, (2, 0, 1))[0]

        nplabs = npimg

        npimg = npimg / 255
        

        threshold_random = 5
        bilateralFilter_random = 9
        

        nplabs = cv2.bilateralFilter(nplabs, 10, bilateralFilter_random, bilateralFilter_random)
        nplabs = np.uint8(nplabs)

        x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        
        nplabs[nplabs < threshold_random] = 0
        
        nplabs = (nplabs - nplabs.min() +1e-12) / (nplabs.max() - nplabs.min() +1e-8)
        npimg = npimg.astype(np.float32)
        nplabs = nplabs.astype(np.float32)

        resize = transforms.Resize([self.size, self.size])

        npimg = torch.from_numpy(np.expand_dims(npimg, 0))
        nplabs = torch.from_numpy(np.expand_dims(nplabs, 0))

        npimg = resize(npimg)
        nplabs = resize(nplabs)

        return npimg, nplabs, self.img[item]


    def __len__(self):
        size = len(self.img)
        return size
    



def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                    (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, edge_data, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t

            x_0 = generator(torch.cat((x.detach(), edge_data.detach()), axis=1), t_time)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x




# %%
def test(rank, gpu, args):
    from score_sde.models_noZ.discriminator import Discriminator_small, Discriminator_large
    from score_sde.models_noZ.ncsnpp_generator_adagn import NCSNpp
    from EMA import EMA
    from piqa import SSIM

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    batch_size = 2
    
    dataset = Dataset_test(path='data/CBCT', size=args.image_size)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              drop_last=False)

    netG = NCSNpp(args).to(device)

    # ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])

    exp = args.exp
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)


    checkpoint_file = os.path.join(exp_path, args.pth_num)
    checkpoint = torch.load(checkpoint_file, map_location=device)

    netG.load_state_dict(checkpoint)
    # load G
    global_step, epoch, init_epoch = 0, 0, 0


    os.makedirs(os.path.join(exp_path, 'result'), exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'visualize'), exist_ok=True)

    ssim_list = []
    psnr_list = []
    for iteration, (x, y,img_name) in enumerate(data_loader):
        netG.eval()
        netG.zero_grad()
        # sample from p(x_0)
        real_data = x.to(device, non_blocking=True)
        edge_data = y.to(device, non_blocking=True)
        t = torch.full((real_data.size(0),), args.num_timesteps - 1, dtype=torch.int64).to(real_data.device)
        x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_tp1, edge_data, args)


        visua_image = torch.cat((real_data ,fake_sample, edge_data, x_tp1), dim=0)

        for item in range(fake_sample.shape[0]):
            print(img_name[item])
            torchvision.utils.save_image(visua_image[item], os.path.join(exp_path, 'visualize', img_name[item]), normalize=True)
            torchvision.utils.save_image(fake_sample[item], os.path.join(exp_path, 'result', img_name[item]))




# %%
def train(rank, gpu, args):
    from score_sde.models_noZ.discriminator import Discriminator_small, Discriminator_large
    from score_sde.models_noZ.ncsnpp_generator_adagn import NCSNpp
    from EMA import EMA
    from piqa import SSIM


    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension


    dataset = Dataset_FGDM(path='data/CT', size=args.image_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              sampler=train_sampler,
                                              drop_last=True)



    netG = NCSNpp(args).to(device)

    if args.dataset == 'cifar10' or args.dataset == 'stackmnist':
        netD = Discriminator_small(nc=args.num_channels, ngf=args.ngf,
                                   t_emb_dim=args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    else:
        netD = Discriminator_large(nc=args.num_channels, ngf=args.ngf,
                                   t_emb_dim=args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)

    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    # ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    exp = args.exp
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'netG_90.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)

        netG.load_state_dict(checkpoint)
        # load G
        global_step, epoch, init_epoch = 0, 0, 0

    else:
        global_step, epoch, init_epoch = 0, 0, 0


    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, (x, y) in enumerate(data_loader):

            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            # sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            edge_data = y.to(device, non_blocking=True)


            # sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            #t = torch.full((x.size(0),), 0, dtype=torch.int64).to(real_data.device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            #x_tp1 = y_tp1.detach()

            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)

            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()

            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
                grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]
                    grad_penalty = (
                            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake


            x_0_predict = netG(torch.cat((x_tp1.detach(), edge_data.detach()), axis=1), t)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            #t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)


            x_0_predict = netG(torch.cat((x_tp1.detach(), edge_data.detach()), axis=1), t)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errG = F.softplus(-output)
            errG = errG.mean()

            errG.backward()
            optimizerG.step()

            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch, iteration, errG.item(),
                                                                                errD.item()))
                    if not os.path.exists(os.path.join(exp_path, 'iteration')):
                        os.makedirs(os.path.join(exp_path, 'iteration'))
                    iter_save0 = real_data.detach()
                    iter_save1 = x_0_predict.detach()
                    iter_save2 = edge_data.detach()
                    ssim_image = torch.cat(
                        (iter_save0,iter_save1,iter_save2,x_tp1), dim=0).detach()
                    torchvision.utils.save_image(ssim_image,
                                                 os.path.join(exp_path, 'iteration', 'iteration_{}_{}.png'.format(epoch, iteration)),
                                             normalize=False)


        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        if rank == 0:
            if epoch % 1 == 0:

                t = torch.full((real_data.size(0),), args.num_timesteps-1, dtype=torch.int64).to(real_data.device)
                x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
                fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_tp1, edge_data, args)

                sample_image = torch.cat((real_data, fake_sample,edge_data,x_tp1), dim=0)
                if not os.path.exists(os.path.join(exp_path, 'sample')):
                    os.makedirs(os.path.join(exp_path, 'sample'))
                torchvision.utils.save_image(sample_image,
                                             os.path.join(exp_path,'sample' ,'sample_epoch_{}.png'.format(epoch)),
                                             normalize=True)



            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                torch.save(netD.state_dict(), os.path.join(exp_path, 'netD_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '2139'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--pth_num', default='netG_0.pth')
    parser.add_argument('--exp', default='experiment_MRT1', help='name of experiment')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=1.,
                        help='beta_max for diffusion')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')


    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--image_size', type=int, default=192,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)


    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 2, 2, 2],
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training

    parser.add_argument('--dataset', default='MRT1', help='name of dataset')
    parser.add_argument('--nz', type=int, default=100)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')


    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        #train(0,0,args)

        train_status = True

        if train_status:
            init_processes(0, size, train, args)
        else:
            init_processes(0, size, test, args)