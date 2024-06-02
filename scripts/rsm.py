try:
    import rsm_sampling
    from rsm_sampling import sample_euler_rsm, sample_euler_ancestral_rsm, sample_dpm_2_ancestral_rsm

    if rsm_sampling.BACKEND == "WebUI":
        from modules import scripts, sd_samplers_common, sd_samplers
        from modules.sd_samplers_kdiffusion import sampler_extra_params, KDiffusionSampler

        class RSM(scripts.Script):
            def title(self):
                "K-Diffusion Resampling Samplers"

            def show(self, is_img2img):
                return False

            def __init__(self):
                if not rsm_sampling.INITIALIZED:
                    samplers_rsm = [
                        ("Euler rsm", sample_euler_rsm, ["euler_rsm"], {}),
                        ("Euler a rsm", sample_euler_ancestral_rsm, ["euler_a_rsm"], {"uses_ensd": True}),
                        ('DPM++ 2S a rsm', sample_dpm_2_ancestral_rsm, ['dpmpp_2s_a_rsm'], {"uses_ensd": True, "second_order": True}),
                    ]
                    samplers_data_rsm = [
                        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
                        for label, funcname, aliases, options in samplers_rsm
                        if callable(funcname)
                    ]
                    sampler_extra_params["euler_a_rsm"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    sampler_extra_params["dpmpp_2s_a_rsm"] = ["s_noise"]
                    sd_samplers.all_samplers.extend(samplers_data_rsm)
                    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
                    sd_samplers.set_samplers()
                    rsm_sampling.INITIALIZED = True

except ImportError as _:
    pass
