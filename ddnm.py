import torch
from tqdm import tqdm

ETA = 0.5
SIGMA_Y = 0

def svd_based_ddnm_plus(model, diffusion, measurement, mask, num_sampling_steps, device):
    A = lambda z: z*mask
    Ap = A
    
    sigma_y = SIGMA_Y
    
    y = measurement
    
    # init x_T
    x = torch.randn_like(measurement, device=device)
    betas = torch.tensor(diffusion.betas).to(device)
    with torch.no_grad():
        skip = diffusion.num_timesteps//num_sampling_steps
        n = x.size(0)
        x0_preds = []
        xs = [x]
        
        times = get_schedule_jump(num_sampling_steps, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))
        
        
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(device)
                next_t = (torch.ones(n) * j).to(device)
                at = compute_alpha(betas, t.long())
                at_next = compute_alpha(betas, next_t.long())
                sigma_t = (1 - at_next**2).sqrt()
                xt = xs[-1].to('cuda').float()
                et = model(xt, t)
                if et.size(1) == 2:
                    et = et[:, 0].unsqueeze(1)

                # Eq. 12
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                # Eq. 19
                if 1:
                    lambda_t = 1.
                    gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                else:
                    lambda_t = (sigma_t)/(at_next*sigma_y)
                    gamma_t = 0.

                # Eq. 17
                x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

                eta = ETA

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                # different from the paper, we use DDIM here instead of DDPM
                xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))    
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(device)
                at_next = compute_alpha(betas, next_t.long())
                x0_t = x0_preds[-1].to('cuda')

                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

        x = xs[-1]
        
    return x
    
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)