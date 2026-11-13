from gym.envs.go2.go2 import Go2


class Go2LowFric(Go2):
    """Go2 locomotion across bimodal (ice / ground) contact friction.

    the true per-environment friction is a
    privileged observation for both actor and critic (a teacher policy). A
    deployable student must later infer it from proprioceptive history.
    """

    def _init_buffers(self):
        super()._init_buffers()
        self.contact_friction = self._backend.contact_friction.unsqueeze(1)
