from . import rsm_sampling
from .rsm_sampling import sample_euler_rsm, sample_euler_ancestral_rsm, sample_dpm_2_ancestral_rsm

if rsm_sampling.BACKEND == "ComfyUI":
    if not rsm_sampling.INITIALIZED:
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from comfy.samplers import SAMPLER_NAMES

        setattr(k_diffusion_sampling, "sample_euler_rsm", sample_euler_rsm)
        setattr(k_diffusion_sampling, "sample_euler_a_rsm", sample_euler_ancestral_rsm)
        setattr(k_diffusion_sampling, "sample_dpmpp_2s_a_rsm", sample_dpm_2_ancestral_rsm)

        SAMPLER_NAMES.append("sample_euler_rsm")
        SAMPLER_NAMES.append("sample_euler_a_rsm")
        SAMPLER_NAMES.append("sample_dpmpp_2s_a_rsm")

        rsm_sampling.INITIALIZED = True

NODE_CLASS_MAPPINGS = {}
