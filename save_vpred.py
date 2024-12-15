import torch
import folder_paths
import json
import os

from comfy import model_management
from comfy.cli_args import args
from safetensors.torch import save_file, load_file

def save_checkpoint_vpred(model, clip=None, vae=None, filename_prefix=None, prompt=None, output_dir=None):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
    prompt_info = ""
    if prompt is not None:
        prompt_info = json.dumps(prompt)

    metadata = {}
    metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
    metadata["modelspec.sai_model_spec"] = "1.0.0"
    metadata["modelspec.implementation"] = "sgm"
    metadata["modelspec.title"] = "{} {}".format(filename, counter)
    metadata["modelspec.predict_key"] = "v"
    metadata["modelspec.prediction_type"] = "v-zsnr"

    if not args.disable_metadata:
        metadata["prompt"] = prompt_info

    output_checkpoint = f"{filename}_{counter:05}_.safetensors"
    output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

    load_models = [model]
    load_models.append(clip.load_model())
    clip_sd = clip.get_sd()
    vae_sd = vae.get_sd()
    model_management.load_models_gpu(load_models, force_patch_weights=True)

    sd = model.model.state_dict_for_saving(clip_sd, vae_sd)
    sd["v_pred"] = torch.tensor([])
    sd["ztsnr"] = torch.tensor([])
    if metadata is not None:
        save_file(sd, output_checkpoint, metadata=metadata)
    else:
        save_file(sd, output_checkpoint)


class CheckpointSaveVpred:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP",),
                              "vae": ("VAE",),
                              "filename_prefix": ("STRING", {"default": "checkpoints/V-Pred"}),},
                "hidden": {"prompt": "PROMPT"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "advanced/model_merging"

    def save(self, model, clip, vae, filename_prefix, prompt=None):
        save_checkpoint_vpred(model, clip=clip, vae=vae, filename_prefix=filename_prefix, prompt=prompt, output_dir=self.output_dir)
        return {}

NODE_CLASS_MAPPINGS = {
    "CheckpointSaveVpred": CheckpointSaveVpred,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointSaveVpred": "Save Checkpoint V-Pred",
}
