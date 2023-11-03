import torch
import torchvision.utils

from tool.interact import set_logger
import os
from tool.fid import calculate_fid_given_paths
import logging
from tool.mse_psnr_ssim_mssim import calculate_ssim,calculate_msssim,calculate_psnr,calculate_mse
import numpy as np
from tool.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
import torch.utils.data as data
from tool.mse_psnr_ssim_mssim import ssim
import pandas as pd

class CBCTDataset(data.Dataset):
    def __init__(self, path):
        self.img = os.listdir(path)  # [:4]
        self.path = path

    def __getitem__(self, item):

        imagename = os.path.join(self.path, self.img[item])

        npy_file = np.load(imagename)
        source = npy_file[0:8]
        target = npy_file[8:16]
        fake = npy_file[16:24]

        return source, target,fake

    def __len__(self):
        size = len(self.img)
        return size


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def fid_l2_psnr_ssim(task,translate_path):
    dims = 2048
    npy_length = 16
    npy_list = os.listdir(translate_path)
    all_num =len(npy_list)
    pred_arr1 = np.empty((len(npy_list)*npy_length, dims))
    pred_arr2 = np.empty((len(npy_list)*npy_length, dims))
    start_idx = 0
    batch_size =8

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    model.eval()
    batch_source = torch.zeros(npy_length * batch_size, 1, 192, 192).cuda()
    batch_target = torch.zeros(npy_length * batch_size, 1, 192, 192).cuda()
    batch_fake = torch.zeros(npy_length * batch_size, 1, 192, 192).cuda()

    num_total = 0
    l2_total=[]
    mse_total=[]
    psnr_total=[]
    ssim_total = []

    l2_total_target=[]
    mse_total_target=[]
    psnr_total_target=[]
    ssim_total_target = []

    for item in range(all_num):

        npy_path = os.path.join(translate_path,npy_list[item])
        npy_file = np.load(npy_path,allow_pickle=True)
        file_batch = int(npy_file.shape[0]/3)

        source = npy_file[0:file_batch]
        target = npy_file[file_batch:file_batch*2]
        fake= npy_file[2*file_batch:3*file_batch]

        #target[source<0.01] = 0

        source = torch.from_numpy(source).cuda()
        target = torch.from_numpy(target).cuda()
        fake = torch.from_numpy(fake).cuda()

        for case in range(len(source)):
            data_range = 1.0

            l2_i = torch.norm(source[case] - fake[case], p=2)
            mse_error = torch.pow(source[case].double() - fake[case].double(), 2).mean()
            psnr = 10.0 * torch.log10(data_range ** 2 / (mse_error + 1e-10))
            ssim_score = ssim(source[case].unsqueeze(0),fake[case].unsqueeze(0),data_range=data_range, size_average=False,win_size=3)

            l2_total.append(float(l2_i.detach().cpu().numpy()))
            mse_total.append(float(mse_error.detach().cpu().numpy()))
            psnr_total.append(float(psnr.detach().cpu().numpy()))
            ssim_total.append(float(ssim_score.detach().cpu().numpy()))


            l2_i = torch.norm(target[case] - fake[case], p=2)
            dim = tuple(range(1, target[case].ndim))
            mse_error = torch.pow(target[case].double() - fake[case].double(), 2).mean(dim=dim)
            psnr = 10.0 * torch.log10(data_range ** 2 / (mse_error + 1e-10))
            ssim_score = ssim(target[case].unsqueeze(0),fake[case].unsqueeze(0),data_range=data_range, size_average=False,win_size=3)

            l2_total_target.append(float(l2_i.detach().cpu().numpy()))
            mse_total_target.append(float(mse_error.detach().cpu().numpy()))
            psnr_total_target.append(float(psnr.detach().cpu().numpy()))
            ssim_total_target.append(float(ssim_score.detach().cpu().numpy()))


            num_total=num_total+1

        b = item%batch_size
        batch_source[b*file_batch:(b+1)*file_batch] = source
        batch_target[b * file_batch:(b + 1) * file_batch] = target
        batch_fake[b * file_batch:(b + 1) * file_batch] = fake

        if (b+1)%batch_size==0:
            #torchvision.utils.save_image(batch_source,'cyclegan/output/cover/source'+str(item)+'.png')
            #torchvision.utils.save_image(batch_target,'cyclegan/output/cover/oldtarget' + str(item) + '.png')

            with torch.no_grad():
                pred = model(torch.cat((batch_target,batch_target,batch_target),1))[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr1[start_idx:start_idx + file_batch*batch_size] = pred

            with torch.no_grad():
                pred = model(torch.cat((batch_fake,batch_fake,batch_fake),1))[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr2[start_idx:start_idx + file_batch*batch_size] = pred

            start_idx = start_idx + file_batch*batch_size

    act1 = pred_arr1
    m1 = np.mean(act1, axis=0)
    s1 = np.cov(act1, rowvar=False)

    act2 = pred_arr2
    m2 = np.mean(act2, axis=0)
    s2 = np.cov(act2, rowvar=False)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print('fid:{}'.format(fid_value))

    result_numpy = np.array([l2_total,mse_total,psnr_total,ssim_total,l2_total_target,mse_total_target,psnr_total_target,ssim_total_target]).T
    pd.DataFrame(result_numpy).to_csv(translate_path.replace("npy","")+task+str(fid_value)+'.csv', header=None, index=None)
    #pd.DataFrame(result_numpy).to_csv(translate_path.replace("/npy","").replace("dd_gan/CBCT/","dd_gan/CBCT/csv/") + task + str(fid_value) + '.csv',header=None, index=None)

    mean_np = np.mean(result_numpy,axis=0)
    txtlist = [str(fid_value)]
    for mean_value in mean_np:
        txtlist.append('& '+str(float('%.3g' % mean_value)))
    print(txtlist)
    with open(translate_path.replace("npy","")+'Score.txt', 'w') as f:
        f.writelines(txtlist)
        f.close()


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if __name__ == '__main__':
    #task = 'CycleGAN'
    task = 'Edge'

    if task == 'CycleGAN':
        translate_path = 'saved_info/dd_gan/CBCT/experiment_CycleGan/npy/'

        fid_l2_psnr_ssim(task, translate_path)

    if task == 'Edge':
        translate_path = 'saved_info/dd_gan/CBCT/experiment_CBCT_192/npy/'
        fid_l2_psnr_ssim(task, translate_path)





