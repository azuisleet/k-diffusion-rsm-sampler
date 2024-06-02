from importlib import import_module

import torch
from tqdm.auto import trange

sampling = None
BACKEND = None
INITIALIZED = False

if not BACKEND:
    try:
        _ = import_module("modules.sd_samplers_kdiffusion")
        sampling = import_module("k_diffusion.sampling")
        BACKEND = "WebUI"
    except ImportError as _:
        pass

if not BACKEND:
    try:
        sampling = import_module("comfy.k_diffusion.sampling")
        BACKEND = "ComfyUI"
    except ImportError as _:
        pass


class _Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args
        if BACKEND == "WebUI":
            self.init_latent, self.mask, self.nmask = model.init_latent, model.mask, model.nmask
        if BACKEND == "ComfyUI":
            self.latent_image, self.noise = model.latent_image, model.noise
            self.denoise_mask = self.extra_args.get("denoise_mask", None)

    def __enter__(self):
        if BACKEND == "WebUI":
            if self.init_latent is not None:
                self.model.init_latent = torch.nn.functional.interpolate(input=self.init_latent, size=self.x.shape[2:4],
                                                                         mode=self.mode)
            if self.mask is not None:
                self.model.mask = torch.nn.functional.interpolate(input=self.mask.unsqueeze(0), size=self.x.shape[2:4],
                                                                  mode=self.mode).squeeze(0)
            if self.nmask is not None:
                self.model.nmask = torch.nn.functional.interpolate(input=self.nmask.unsqueeze(0),
                                                                   size=self.x.shape[2:4], mode=self.mode).squeeze(0)
        if BACKEND == "ComfyUI":
            if self.latent_image is not None:
                self.model.latent_image = torch.nn.functional.interpolate(input=self.latent_image,
                                                                          size=self.x.shape[2:4], mode=self.mode)
            if self.noise is not None:
                self.model.noise = torch.nn.functional.interpolate(input=self.latent_image, size=self.x.shape[2:4],
                                                                   mode=self.mode)
            if self.denoise_mask is not None:
                self.extra_args["denoise_mask"] = torch.nn.functional.interpolate(input=self.denoise_mask,
                                                                                  size=self.x.shape[2:4],
                                                                                  mode=self.mode)

        return self

    def __exit__(self, type, value, traceback):
        if BACKEND == "WebUI":
            del self.model.init_latent, self.model.mask, self.model.nmask
            self.model.init_latent, self.model.mask, self.model.nmask = self.init_latent, self.mask, self.nmask
        if BACKEND == "ComfyUI":
            del self.model.latent_image, self.model.noise
            self.model.latent_image, self.model.noise = self.latent_image, self.noise


@torch.no_grad()
def _in_resized_space_vec(x, model, dt, sigma_hat, **extra_args):
    m, n = x.shape[2], x.shape[3]
    # range between 2 and 4
    y = torch.nn.functional.interpolate(input=x, size=(m + 2, n + 2), mode='nearest-exact')
    with _Rescaler(model, y, 'nearest-exact', **extra_args) as rescaler:
        denoised = model(y, sigma_hat * y.new_ones([y.shape[0]]), **extra_args)
    d = sampling.to_d(y, sigma_hat, denoised)
    return torch.nn.functional.interpolate(input=d * dt, size=(m, n), mode='nearest-exact')


@torch.no_grad()
def sample_euler_rsm(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                     s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        dt = sigmas[i + 1] - sigma_hat
        d = sampling.to_d(x, sigma_hat, denoised)
        sample_vec = d * dt

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        if sigmas[i + 1] > 0:
            # Resample and choose a vector between the denoised and resampled
            resample_vec = _in_resized_space_vec(x, model, dt, sigma_hat, **extra_args)
            chosen_vec = (sample_vec + resample_vec) / 2
        else:
            # euler method
            chosen_vec = sample_vec
        x = x + chosen_vec
    return x


@torch.no_grad()
def sample_euler_ancestral_rsm(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1.,
                               noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = sampling.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = sampling.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        dt = sigma_down - sigmas[i]
        d = sampling.to_d(x, sigmas[i], denoised)
        sample_vec = d * dt

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] > 0:
            # Resample and choose a vector between the denoised and resampled
            resample_vec = _in_resized_space_vec(x, model, dt, sigmas[i], **extra_args)
            chosen_vec = (sample_vec + resample_vec) / 2
        else:
            # euler method
            chosen_vec = sample_vec
        x = x + chosen_vec

        # Add euler ancestral noise
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpm_2_ancestral_rsm(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1.,
                               noise_sampler=None):
    """Ancestral sampling with DPM-Solver second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = sampling.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = sampling.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = sampling.to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]

            # Resample and choose a vector between the denoised and resampled
            sample_vec = d * dt_1
            resample_vec = _in_resized_space_vec(x, model, dt_1, sigmas[i], **extra_args)
            chosen_vec = (sample_vec + resample_vec) / 2
            x_2 = x + chosen_vec

            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = sampling.to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x
