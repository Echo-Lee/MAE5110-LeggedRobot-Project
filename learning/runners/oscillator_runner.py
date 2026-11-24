"""Observation noise used by the decentralized phase-oscillator experiments."""

import torch

from .on_policy_runner import OnPolicyRunner


class OscillatorRunner(OnPolicyRunner):
    """Keep paper-specific noise and reward integration isolated from other tasks."""

    def get_noise(self, obs_list, noise_dict):
        amplitude = torch.zeros(self.get_obs_size(obs_list), device=self.device)
        offset = 0
        for name in obs_list:
            size = self.get_obs_size([name])
            if name in noise_dict:
                feature_noise = torch.as_tensor(
                    noise_dict[name], dtype=amplitude.dtype, device=self.device
                )
                if name in self.env.scales:
                    feature_noise = feature_noise / torch.as_tensor(
                        self.env.scales[name],
                        dtype=amplitude.dtype,
                        device=self.device,
                    )
                amplitude[offset : offset + size] = feature_noise
            offset += size

        samples = (
            2 * torch.rand(self.env.num_envs, amplitude.numel(), device=self.device) - 1
        )
        return samples * amplitude * noise_dict.get("scale", 1.0)

    def update_rewards_dict(self, rewards_dict, step):
        """Integrate ordinary rewards in time and apply terminal penalties once."""
        rewards_dict[step].update(
            self.get_rewards(
                self.critic_cfg["reward"]["termination_weight"],
                mask=self.env.terminated,
            ),
            inplace=True,
        )
        rewards_dict[step].update(
            self.get_rewards(
                self.critic_cfg["reward"]["weights"],
                modifier=self.env.dt,
                mask=~self.env.terminated,
            ),
            inplace=True,
        )
