import torch as th
img_guidance_scale = 3000
def mask_inpainting(x, H_funcs):
    x_tmp = H_funcs.H(((x+1)/2).to(th.float32))
    x_masked = H_funcs.H_pinv(x_tmp).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    x_masked = x_masked * 2 - 1
    return x_masked


def general_cond_fn(H_funcs, x, t, y=None, x_lr=None, sample_noisy_x_lr=False, diffusion=None, sample_noisy_x_lr_t_thred=None):
#     assert y is not None
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        if not x_lr is None:  
            x_in_tmp = H_funcs.H((x_in).to(th.float32))
            x_in_lr = H_funcs.H_pinv(x_in_tmp).view(x_in_tmp.shape[0], x_in.shape[1], x_in.shape[2], x_in.shape[3])
        
            x_in_lr.to(th.uint8)
            
            if sample_noisy_x_lr:
                t_numpy = t.detach().cpu().numpy()
                spaced_t_steps = [diffusion.timestep_reverse_map[t_step] for t_step in t_numpy]
                if sample_noisy_x_lr_t_thred is None or spaced_t_steps[0] < sample_noisy_x_lr_t_thred:
#                     print('Sampling noisy lr')
                    spaced_t_steps = th.Tensor(spaced_t_steps).to(t.device).to(t.dtype)
                    x_lr = diffusion.q_sample(x_lr, spaced_t_steps)

            # x_lr = (x_lr + 1) / 2
            mse = (x_in_lr - x_lr) ** 2
            mse = mse.mean(dim=(1,2,3))
            mse = mse.sum()
#             ssim_value = pytorch_ssim.ssim(x_in_lr, x_lr).item()
#             ssim_loss = pytorch_ssim.SSIM()
#             ssim_out = -ssim_loss(x_in_lr, x_lr)

            loss = - mse * img_guidance_scale # move xt toward the gradient direction 
#             print('step t %d img guidance has been used, mse is %.8f * %d = %.2f' % (t[0], mse, img_guidance_scale, mse*img_guidance_scale))
        return th.autograd.grad(loss, x_in)[0]