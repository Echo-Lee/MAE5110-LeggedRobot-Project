"""go2_lowfric_est: fine-tune the go2_lowfric teacher to act on mu-hat.

Run 15 started from run 12 (stage 1), run 16 from run 15:
    uv run scripts/train.py --task go2_lowfric_est --resume \\
        --load_run Nov15_19-11-13_ --max_iterations 500 --device cuda:0
A go2_lowfric checkpoint loads unchanged: the actor's friction input keeps
its slot, it is just fed mu-hat (contact_friction_hat) instead of mu. A
go2_lowfric_est checkpoint also restores its estimator.
"""

from gym.envs.go2.go2_config import BASE_HEIGHT_REF
from gym.envs.go2.go2_lowfric_config import (
    GROUND_FRICTION,
    ICE_FRICTION,
    Go2LowFricCfg,
    Go2LowFricRunnerCfg,
)
from learning.modules.history_estimator import DEFAULT_FEATURES

# Between the ice and ground bins; untrained before run 16.
MIDDLE_FRICTION = [ICE_FRICTION[1], GROUND_FRICTION[0]]


class Go2LowFricEstCfg(Go2LowFricCfg):
    class domain_randomization(Go2LowFricCfg.domain_randomization):
        class startup(Go2LowFricCfg.domain_randomization.startup):
            # Continuous friction coverage (run 16): a middle bin fills the
            # gap between ice and ground, so mu is drawn from all of
            # [0.05, 1.2] (uniform within each bin). Run 15 trained on the two
            # bins only (weights 0.5 / 0.5) and lost braking at mu 0.2 and
            # 2 m/s tracking at mu 0.3-0.5, where its "assume ice" gait is
            # wrong. Ice keeps the largest share: slide episodes and the
            # estimator's ice accuracy depend on it.
            contact_friction_bins = [ICE_FRICTION, MIDDLE_FRICTION, GROUND_FRICTION]
            contact_friction_bin_weights = [0.4, 0.25, 0.35]

    class friction_switch(Go2LowFricCfg.friction_switch):
        # The student must re-estimate when the surface changes (stage 2b:
        # the frozen policy's cautious ice gait kept mu-hat low on carpet).
        probability = 0.3

    class state_estimator:
        # Initial weights: the switch-trained estimator for run 12 (stage
        # 2b). Relative paths are from the repository root. Without the
        # file the estimator starts untrained (only valid when its weights
        # come from a go2_lowfric_est checkpoint).
        init_path = (
            "logs/go2_lowfric/Nov15_19-11-13_/state_estimator_switch/estimator.pt"
        )
        # Estimator inputs: LEGACY_FEATURES + IMU accelerometer (run 18).
        # Estimators trained on fewer inputs are widened with zero weights.
        features = list(DEFAULT_FEATURES)
        # Architecture used only when init_path is missing.
        frames = 25
        stride = 2
        hidden_dims = [512, 256, 128]
        # Hold the actor's mu-hat at its slide-start value through the slide
        # phase (run 17). Off in runs 15-16 (their snapshots lack it).
        latch_in_slide = True
        # Keep training the estimator (true labels) during PPO.
        learn = True
        learning_rate = 3.0e-4
        samples_per_step = 256  # random envs kept per control step
        update_interval = 24  # control steps between updates
        gradient_steps = 4  # Adam steps per update

    class scaling(Go2LowFricCfg.scaling):
        contact_friction_hat = 1.0
        base_lin_vel_hat = BASE_HEIGHT_REF


class Go2LowFricEstRunnerCfg(Go2LowFricRunnerCfg):
    class actor(Go2LowFricRunnerCfg.actor):
        # Same inputs as the teacher, with mu replaced by mu-hat in place.
        obs = [
            "contact_friction_hat" if name == "contact_friction" else name
            for name in Go2LowFricRunnerCfg.actor.obs
        ]

    class runner(Go2LowFricRunnerCfg.runner):
        # Same experiment folder as the teacher, so --resume --load_run
        # finds run 12; fine-tuned runs are labelled by run_name.
        experiment_name = "go2_lowfric"
        run_name = "est_finetune"
        max_iterations = 500
