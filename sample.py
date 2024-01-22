import sys
import argparse
import torch as th
import os
sys.path.append("..")
sys.path.append(".")
from scripts.guided_diffusion.dataloader import SeismicDataset
import torch.distributed as dist
from scripts.guided_diffusion import dist_util
from scripts.guided_diffusion.resample import create_named_schedule_sampler
from scripts.guided_diffusion.script_util_x0 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from functools import partial
from tqdm.auto import tqdm
import numpy as np
import random
# from piq import ssim, psnr
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scripts.guided_diffusion.logger import CSVOutputFormat

def normalize(image):
    """Basic min max scaler.
    """
    image = image / 255
    return image
def snr_(gt, pred):
    snr_score = 10*np.log10(np.sum(pred**2)/np.sum((pred-gt)**2))
    return snr_score
    
def get_metric(gt, pred):
    
    ssim_score = structural_similarity(np.array(gt)[:, :, 0], np.array(pred)[:, :, 0], win_size=11)
    psnr_score = peak_signal_noise_ratio(np.array(gt)[:, :, 0], np.array(pred)[:, :, 0])
    gt = normalize(gt.unsqueeze(0).permute(0, 3, 1, 2))
    pred = normalize(pred.unsqueeze(0).permute(0, 3, 1, 2))
    snr_score = snr_(gt[0].numpy(), pred[0].numpy())
    return ssim_score, psnr_score, snr_score

def visualize(sample):
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    return sample.cpu()

from abc import ABC, abstractmethod
import torch

class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
#         if self.noiser.__name__ == 'gaussian':
#         norm = []
#         norm_grads = []
#         for i in range(x_prev.shape[0]):
#             x_i = x_prev[i]
#             x_i = x_i.unsqueeze(0)
#             difference_i = measurement[i].unsqueeze(0) - x_0_hat[i].unsqueeze(0) * kwargs.get('mask', None)[i]
#         # difference = measurement - x_0_hat

#             norm_i = torch.linalg.norm(difference_i)
#             norm_grad_i = torch.autograd.grad(outputs=norm_i, inputs=x_i, allow_unused=True, materialize_grads=True)[0]
#             norm_grads.append(norm_grad_i[0])
#             norm.append(norm_i)
        
# #         elif self.noiser.__name__ == 'poisson':
# #             Ax = self.operator.forward(x_0_hat, **kwargs)
# #             difference = measurement-Ax
# #             norm = torch.linalg.norm(difference) / measurement.abs()
# #             norm = norm.mean()
# #             norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

# #         else:
# #             raise NotImplementedError
             
#         return th.stack(norm_grads), th.stack(norm).mean()
        difference = measurement - self.operator.forward(x_0_hat, **kwargs)
#         difference = measurement - x_0_hat

        norm = torch.linalg.norm(difference, dim=(2,3))
        grad_outputs = torch.ones_like(norm)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, grad_outputs=grad_outputs)[0]
        
#         elif self.noiser.__name__ == 'poisson':
#             Ax = self.operator.forward(x_0_hat, **kwargs)
#             difference = measurement-Ax
#             norm = torch.linalg.norm(difference) / measurement.abs()
#             norm = norm.mean()
#             norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

#         else:
#             raise NotImplementedError
             
        return norm_grad, norm.mean()
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
#             return ((data + 1) / 2 * kwargs.get('mask', None).to(self.device)) * 2 - 1
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma
        
def p_sample_loop_dps(model,
                      diffusion,
                      shape,
                      measurement,
                      measurement_cond_fn):
        """
        The function used for sampling from noise.
        """ 
        device = next(model.parameters()).device
        img = th.randn(*shape, device=device)
        
        pbar = tqdm(list(range(diffusion.num_timesteps))[::-1])
        for idx in pbar:
            time = th.tensor([idx] * img.shape[0], device=device)
            
            img = img.requires_grad_()
            # with th.no_grad():
            out = diffusion.p_sample(model, img, time)
            
            # Give condition.
            # noisy_measurement = diffusion.q_sample(measurement, time)

            # TODO: how can we handle argument for different condition method?
            img, distance = measurement_cond_fn(x_t=out['sample'],
                                      measurement=measurement,
                                      # noisy_measurement=noisy_measurement,
                                      x_prev=img,
                                      x_0_hat=out['pred_xstart'])
            img = img.detach_()
           
            pbar.set_postfix({'distance': distance.item()}, refresh=False)

        return img       

def main():
    args = create_argparser().parse_args()

    
    dataset = args.dataset
    print("Evaluation on dataset(s): ", dataset)
    ds = SeismicDataset(args.data_dir, mode=args.mode, datasets=dataset.split(','))
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    device = th.device('cuda:{}'.format(args.cuda_devices))
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    image_size = args.image_size
    
    proportion = args.mask_ratio
    
    sample_idx = random.sample(range(1, image_size), int((image_size - 1) * proportion))
    

    operator = InpaintingOperator(device=device)
    noiser = GaussianNoise(sigma=args.noise_scale)
    cond_method = PosteriorSampling(operator=operator, noiser=noiser, scale=args.gradient_scale)
    batch_size = args.batch_size
    dataset = th.utils.data.DataLoader(
                                ds,
                                batch_size=batch_size,
                                shuffle=False)
    csvwriter = CSVOutputFormat(f'evaluation_results/result_mask:{proportion}_noise:{args.noise_scale}_{args.start_idx}_{args.end_idx}.csv')
    n_sample = 0
    for idx, img in enumerate(dataset):
        if n_sample < args.start_idx - 1:
            n_sample += img[0].shape[0]
            continue
        
        mask = th.ones((image_size, image_size))
        for j in sample_idx:
            mask[:, j] = 0
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
        
        if th.cuda.is_available():
            img[0] = th.Tensor(img[0]).to(device)
        sample_fn = p_sample_loop_dps 
        
        y = operator.forward(img[0], mask=mask)
        measurement = noiser(y)
        shape = img[0].shape
            
        sample = sample_fn(model,
                           diffusion,
                           shape,
                           measurement,
                           measurement_cond_fn)
        
        original = visualize(img[0])
        masked_image = visualize(measurement)
        sample = visualize(sample)
        ssim_scores = []
        psnr_scores = []
        snr_scores = []
        # spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(3)
        # final = sample * (1 - spatial_mask) + original * spatial_mask
        for i in range(batch_size):
            ssim_score, psnr_score, snr_score = get_metric(original[i], sample[i])
            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)
            snr_scores.append(snr_score)
            np.savez_compressed(f'samples/sample{n_sample}', original=np.array(original[i]), masked_image=np.array(masked_image[i]), sample=np.array(sample[i]))
            print(f"{n_sample}: SSIM: {ssim_score} - PSNR: {psnr_score} - SNR: {snr_score}")
            csvwriter.writekvs({'id': n_sample, 'ssim': ssim_score, 'psnr': psnr_score, 'snr': snr_score})
            n_sample += 1        
        if n_sample >= args.end_idx:
            break
        # metrics = zip(ssim_scores, psnr_scores, snr_scores)
        # for i, v in enumerate(metrics):
            

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        batch_size=1,
        model_path='',
        use_fp16=False,
        dataset='F3,Kerry3D',
        mode='validation',
        seed=9999,
        cuda_devices='0',
        mask_ratio=0.1,
        noise_scale=0.1,
        gradient_scale=0.5,
        start_idx=0,
        end_idx=-1
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()