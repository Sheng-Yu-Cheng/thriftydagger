import numpy as np
import torch
from robomimic.utils import file_utils as FileUtils, obs_utils as ObsUtils, tensor_utils as TensorUtils

class RobomimicExpert:
    def __init__(self, ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path,
            device=self.device,
            verbose=True,
        )
        ObsUtils.initialize_obs_utils_from_config(self.ckpt_dict["config"])

    def __call__(self, obs_dict):
        # obs_dict must match the robomimic dataset modality (e.g., low_dim dict)
        obs = ObsUtils.process_obs(obs_dict, obs_modality_cfg=None)
        obs_tensor = TensorUtils.to_tensor(obs, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            ac = self.policy(obs_tensor)["actions"]
        return TensorUtils.to_numpy(ac)[0]
