import sys
import argparse
import torch as th
import os
sys.path.append("..")
sys.path.append(".")
from scripts.guided_diffusion.dataloader import SeismicDataset
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
# from swin import SCRN, ConvTransBlock, Block, WMSA
from ssim_util import SSIM
from scripts.functions.svd_replacement import Inpainting
from gdp import general_cond_fn
from CoPaint.utils.config import Config
from CoPaint.guided_diffusion import (
    DDIMSampler,
    O_DDIMSampler,
    R_DDIMSampler,
    DDNMSampler,
    DDRMSampler,
    DPSSampler,
)
from CoPaint.guided_diffusion.script_util import (
    create_model as create_model_copaint,
    create_gaussian_diffusion as create_gaussian_diffusion_copaint
)
ssim_loss = SSIM(11, size_average=True)

model_params = dict(
    image_size=128,
    num_channels=128,
    num_res_blocks=2,
    channel_mult="",
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="32,16,8",
    num_heads=4,
    num_heads_upsample=-1,
    num_head_channels=64,
    use_scale_shift_norm=True,
    dropout=0.0,
    resblock_updown=True,
    use_fp16=True,
    use_new_attention_order=False
)

SAMPLER_CLS = {
        "ddim": DDIMSampler,
        "o_ddim": O_DDIMSampler,
        "resample": R_DDIMSampler,
        "ddnm": DDNMSampler,
        "ddrm": DDRMSampler,
        "dps": DPSSampler,
    }

diffusion_params = dict(
    diffusion_steps=1000,
    learn_sigma=True,
    noise_schedule="cosine",
    use_kl=False,    
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing=""
)

def prepare_model(algorithm, model_path, config, device):
    sampler_cls = SAMPLER_CLS[algorithm]
    copaint_model = create_model_copaint(**model_params)
    copaint_diffusion = create_gaussian_diffusion_copaint(**diffusion_params, conf=config, base_cls=sampler_cls)
    copaint_model.load_state_dict(
        th.load(model_path, map_location="cpu")
    )
    copaint_model.to(device)
    copaint_model.convert_to_fp16()
    copaint_model.eval()
    return copaint_model, copaint_diffusion
    
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
    def __init__(self, operator, noiser, device, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.device = device
              
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        x = self.operator.forward(x_0_hat, **kwargs)        
       
        difference = measurement - x
        norm = torch.linalg.norm(difference, dim=(2,3))
        
        grad_outputs = torch.ones_like(norm)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, grad_outputs=grad_outputs)[0]
    
             
        return norm_grad, norm.mean()
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, device, **kwargs):
        super().__init__(operator, noiser, device)
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
                      original,
                      measurement,
                      measurement_cond_fn,
                      device):
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
                                      step=idx/diffusion.num_timesteps,
                                      device=device,
                                      # noisy_measurement=noisy_measurement,
                                      x_prev=img,
                                      x_0_hat=out['pred_xstart'])
            img = img.detach_()
            # spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(3)
            # final = sample * (1 - spatial_mask) + original * spatial_mask
           
            pbar.set_postfix({'distance': distance.item()}, refresh=False)

        return out['pred_xstart']       

def main():
    args = create_argparser().parse_args()

    config = Config(default_config_file="CoPaint/configs/seismic.yaml", use_argparse=False)

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

    mask_model = th.load(args.mask_model_path)
    mask_model.to(device)
    mask_model.eval()
    
    algorithm = args.algo
    copaint_model, copaint_diffusion = prepare_model(algorithm, args.model_path, config, device)
    image_size = args.image_size
    
    proportion = args.mask_ratio
    
    
    

    operator = InpaintingOperator(device=device)
    noiser = GaussianNoise(sigma=args.noise_scale)
    cond_method = PosteriorSampling(operator=operator, noiser=noiser, device=device, scale=args.gradient_scale)
    batch_size = args.batch_size
    dataset = th.utils.data.DataLoader(
                                ds,
                                batch_size=batch_size,
                                shuffle=False)
    def model_fn(x, t, y=None):
        return model(x, t)
    def copaint_model_fn(x, t, y=None, gt=None, **kwargs):
        return copaint_model(x, t, y if config.class_cond else None, gt=gt)
        
    sampling_method = args.method
    output_directory = f'samples/{sampling_method}_mask:{proportion}_noise:{args.noise_scale}_{args.start_idx}_{args.end_idx}'
    os.makedirs(output_directory, exist_ok=True)
    csvwriter = CSVOutputFormat(f'{output_directory}/result_mask:{proportion}_noise:{args.noise_scale}_{args.start_idx}_{args.end_idx}.csv')
    n_sample = 0
    ssim_agg = 0
    psnr_agg = 0
    snr_agg = 0
    for idx, img in enumerate(dataset):
        if n_sample < args.start_idx - 1:
            n_sample += img[0].shape[0]
            continue
        spatial_mask = []
        for i in range(img[0].shape[0]):
            mask_i = th.ones((image_size, image_size))
            sample_idx = random.sample(range(1, image_size), int((image_size - 1) * proportion))
            for j in sample_idx:
                mask_i[:, j] = 0
            spatial_mask.append(mask_i)
        spatial_mask = th.stack(spatial_mask).unsqueeze(1).to(device)
        model_kwargs = {}
        if th.cuda.is_available():
            img[0] = th.Tensor(img[0]).to(device)
        sample_fn = p_sample_loop_dps 
        
        y = operator.forward(img[0], mask=spatial_mask)
        measurement = noiser(y)
        shape = img[0].shape
        
        with th.no_grad():
            pred_mask = mask_model(measurement)
            binary_mask = (pred_mask > 0.5).float().to(device)
        accuracy = th.mean(torch.sum(binary_mask * spatial_mask, dim=(1,2,3)) / spatial_mask.sum(dim=(1,2,3)))
        print('Mask accuracy: ', accuracy)
        measurement_cond_fn = partial(cond_method.conditioning, mask=binary_mask)
        
    
        original = visualize(img[0])
        if sampling_method == 'dps':
            sample = sample_fn(model,
                               diffusion,
                               shape,
                               original,
                               measurement,
                               measurement_cond_fn,
                               device=device)
        elif sampling_method == 'ddnm':
            # sample = simplified_based_ddnm(model, diffusion, measurement, spatial_mask, diffusion.num_timesteps, device)
            # sample = svd_based_ddnm_plus(model, diffusion, measurement, spatial_mask, diffusion.num_timesteps, device)
            model_kwargs = {
                    "gt": measurement,
                    "gt_keep_mask": binary_mask,
            }
            sample = copaint_diffusion.p_sample_loop(
                copaint_model_fn,
                shape=shape,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=device,
                progress=True,
                return_all=True,
                conf=config,
                sample_dir=None,
            )
            sample = sample['sample']
            
        elif sampling_method == 'gdp':
            mask = binary_mask.reshape(img[0].shape[0], -1)
            missing = [th.nonzero(mask[i] == 0).squeeze() for i in range(img[0].shape[0])]
            H_funcs = Inpainting(shape[1], image_size, missing, device)
            cond_fn = lambda x,t : general_cond_fn(H_funcs, x, t, x_lr=measurement, sample_noisy_x_lr=True, diffusion=diffusion, sample_noisy_x_lr_t_thred=1e8)
            sample = diffusion.p_sample_loop(model_fn,
                                            shape,
                                            clip_denoised=True,
                                            model_kwargs=model_kwargs,
                                            cond_fn=cond_fn,
                                            device=device,
                                            progress=True
                                        )

        elif sampling_method == 'copaint':
            model_kwargs = {
                    "gt": measurement,
                    "gt_keep_mask": binary_mask,
            }
            sample = copaint_diffusion.p_sample_loop(
                copaint_model_fn,
                shape=shape,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=device,
                progress=True,
                return_all=True,
                conf=config,
                sample_dir=None,
            )
            sample = sample['sample']
            
        masked_image = visualize(measurement)
        sample = visualize(sample)
        # swin_sample = visualize(swin_sample)
        # spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(3)
        # final = sample * (1 - spatial_mask) + original * spatial_mask
        for i in range(img[0].shape[0]):
            ssim_score, psnr_score, snr_score = get_metric(original[i], sample[i])
            # swin_ssim_score, swin_psnr_score, swin_snr_score = get_metric(original[i], swin_sample[i])
            ssim_agg += ssim_score
            psnr_agg += psnr_score
            snr_agg += snr_score
            np.savez_compressed(f'{output_directory}/sample{n_sample}', original=np.array(original[i]), masked_image=np.array(masked_image[i]), diffusion_sample=np.array(sample[i]))
            print(f"{n_sample}: SSIM: {ssim_score} - PSNR: {psnr_score} - SNR: {snr_score}")
            # print(f"[Swin] {n_sample}: SSIM: {swin_ssim_score} - PSNR: {swin_psnr_score} - SNR: {swin_snr_score}")
            csvwriter.writekvs({'id': n_sample, 'ssim': ssim_score, 'psnr': psnr_score, 'snr': snr_score})
            n_sample += 1        
        print(f"Accumulated Average: SSIM: {ssim_agg/n_sample} - PSNR: {psnr_agg/n_sample} - SNR: {snr_agg/n_sample}")
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
        mask_model_path='',
        use_fp16=False,
        dataset='F3,Kerry3D',
        mode='validation',
        seed=9999,
        cuda_devices='0',
        mask_ratio=0.1,
        noise_scale=0.1,
        gradient_scale=0.5,
        start_idx=0,
        end_idx=-1,
        method='dps',
        algo='o_ddim'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()