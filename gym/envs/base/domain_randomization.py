"""Backend-neutral sampling for physical and control domain parameters.

The task layer owns random sampling. Physics backends only apply explicit
values, which isolates DR from task RNG and keeps parameter semantics
independent of the selected engine.
"""

import zlib

import torch

from gym.utils.sampling import masked_update


DOMAIN_RANDOMIZATION_MODES = (
    "config",
    "off",
    "friction-only",
    "pd-only",
    "mass-only",
)

STARTUP_AXES = (
    "contact_friction_range",
    "link_mass_scale_range",
)
PD_GAIN_TARGETS = ("p_gains", "d_gains")


class DomainRandomizationCfg:
    class startup:
        # Sampled once for every environment during task construction. It is
        # also the validated envelope that makes backends allocate per-env
        # friction, so it must be set whenever friction is randomized at all.
        contact_friction_range = None
        # Optional weighted sub-ranges inside contact_friction_range, e.g.
        # [[0.05, 0.2], [0.6, 1.2]] for a deliberately bimodal ice/ground
        # mixture. None samples uniformly from the whole range.
        contact_friction_bins = None
        # One positive weight per bin; None weights bins equally.
        contact_friction_bin_weights = None
        # Mass and diagonal inertia are scaled together for each robot link.
        link_mass_scale_range = None

    class episode:
        # Multiplicative ranges for task-owned tensors that are resampled at
        # every episode reset. The task explicitly binds the allowed tensors.
        scale_ranges = {}
        # Redraw contact friction for every resetting environment, so each
        # environment experiences many surfaces over training instead of one.
        resample_contact_friction = False


def get_domain_randomization_range(cfg, name: str):
    """Return a startup-axis or episodic tensor scale range."""
    if name == "contact_friction_range":
        return cfg.domain_randomization.startup.contact_friction_range
    if name == "link_mass_scale_range":
        return cfg.domain_randomization.startup.link_mass_scale_range
    return cfg.domain_randomization.episode.scale_ranges.get(name)


def set_domain_randomization_range(cfg, name: str, values) -> None:
    """Set a startup-axis or episodic tensor scale range."""
    if name == "contact_friction_range":
        cfg.domain_randomization.startup.contact_friction_range = values
        if values is None:
            # Bins and episodic resampling only refine the friction axis;
            # leaving them set would describe a disabled axis as active.
            cfg.domain_randomization.startup.contact_friction_bins = None
            cfg.domain_randomization.startup.contact_friction_bin_weights = None
            cfg.domain_randomization.episode.resample_contact_friction = False
        return
    if name == "link_mass_scale_range":
        cfg.domain_randomization.startup.link_mass_scale_range = values
        return
    ranges = dict(cfg.domain_randomization.episode.scale_ranges)
    if values is None:
        ranges.pop(name, None)
    else:
        ranges[name] = values
    cfg.domain_randomization.episode.scale_ranges = ranges


def apply_domain_randomization_override(cfg, mode: str) -> None:
    """Select a deliberate subset of the configured randomization axes."""
    if mode == "config":
        return
    if mode not in DOMAIN_RANDOMIZATION_MODES:
        raise ValueError(f"unknown domain-randomization mode {mode!r}")

    keep_startup = {
        "off": set(),
        "friction-only": {"contact_friction_range"},
        "pd-only": set(),
        "mass-only": {"link_mass_scale_range"},
    }[mode]
    keep_episode = set(PD_GAIN_TARGETS) if mode == "pd-only" else set()
    configured_startup = {
        name: get_domain_randomization_range(cfg, name) for name in STARTUP_AXES
    }
    configured_episode = dict(cfg.domain_randomization.episode.scale_ranges)
    missing = sorted(
        name
        for name in (*keep_startup, *keep_episode)
        if (
            configured_startup.get(name) is None
            and configured_episode.get(name) is None
        )
    )
    if missing:
        raise ValueError(
            f"{mode} domain randomization requires configured ranges for {missing}"
        )
    for name in configured_startup:
        if name not in keep_startup:
            set_domain_randomization_range(cfg, name, None)
    cfg.domain_randomization.episode.scale_ranges = {
        name: values
        for name, values in configured_episode.items()
        if name in keep_episode
    }


def _configured_range(cfg, name: str) -> tuple[float, float] | None:
    values = get_domain_randomization_range(cfg, name)
    if values is None:
        return None
    low, high = map(float, values)
    return low, high


def contact_friction_range(cfg) -> tuple[float, float] | None:
    """Return and validate the configured effective sliding-friction range.

    ``None`` disables contact-friction randomization. The portable coefficient
    maps to MuJoCo sliding friction and to both VSim static and dynamic
    friction.
    """
    values = _configured_range(cfg, "contact_friction_range")
    if values is None:
        return None
    low, high = values
    if not 0.0 <= low <= high:
        raise ValueError(
            f"contact-friction range must satisfy 0 <= low <= high, got [{low}, {high}]"
        )
    return low, high


def contact_friction_bins(
    cfg,
) -> tuple[list[tuple[float, float]], list[float]] | None:
    """Return validated friction sub-ranges and normalized weights, or ``None``.

    Bins must lie inside ``contact_friction_range`` because backends size and
    enable per-environment friction from that envelope alone.
    """
    startup = cfg.domain_randomization.startup
    bins = startup.contact_friction_bins
    weights = startup.contact_friction_bin_weights
    if bins is None:
        if weights is not None:
            raise ValueError("contact-friction bin weights require bins")
        return None
    envelope = contact_friction_range(cfg)
    if envelope is None:
        raise ValueError("contact-friction bins require contact_friction_range")
    parsed = [tuple(map(float, values)) for values in bins]
    if not parsed or any(len(values) != 2 for values in parsed):
        raise ValueError(f"contact-friction bins must be [low, high] pairs: {bins}")
    for low, high in parsed:
        if not envelope[0] <= low <= high <= envelope[1]:
            raise ValueError(
                f"contact-friction bin [{low}, {high}] must lie inside "
                f"contact_friction_range {list(envelope)}"
            )
    if weights is None:
        weights = [1.0] * len(parsed)
    weights = [float(w) for w in weights]
    if len(weights) != len(parsed) or any(w <= 0.0 for w in weights):
        raise ValueError(
            "contact-friction bin weights need one positive value per bin, "
            f"got {weights} for {len(parsed)} bins"
        )
    total = sum(weights)
    return parsed, [w / total for w in weights]


def scale_range(cfg, name: str) -> tuple[float, float] | None:
    """Return a positive multiplicative range containing the nominal scale."""
    values = _configured_range(cfg, name)
    if values is None:
        return None
    low, high = values
    if not 0.0 < low <= 1.0 <= high:
        raise ValueError(
            f"{name.replace('_', '-')} must satisfy 0 < low <= 1 <= high, "
            f"got [{low}, {high}]"
        )
    return low, high


def link_mass_scale_range(cfg) -> tuple[float, float] | None:
    """Return the multiplicative link mass-and-inertia scale range."""
    return scale_range(cfg, "link_mass_scale_range")


class DomainRandomizer:
    def __init__(self, cfg, backend, device: str) -> None:
        self._backend = backend
        self._device = device
        self._contact_friction_range = contact_friction_range(cfg)
        bins = contact_friction_bins(cfg)
        self._contact_friction_bins = None
        self._contact_friction_bin_weights = None
        if bins is not None:
            self._contact_friction_bins = torch.tensor(bins[0], device=device)
            self._contact_friction_bin_weights = torch.tensor(bins[1], device=device)
        self._resample_contact_friction = (
            cfg.domain_randomization.episode.resample_contact_friction
        )
        if self._resample_contact_friction and self._contact_friction_range is None:
            raise ValueError(
                "resample_contact_friction requires contact_friction_range"
            )
        self._link_mass_scale_range = link_mass_scale_range(cfg)
        self._episode_scale_ranges = {}
        for name in cfg.domain_randomization.episode.scale_ranges:
            values = scale_range(cfg, name)
            if values is None:
                raise ValueError(
                    f"episodic DR target {name!r} has no range; omit it to disable it"
                )
            self._episode_scale_ranges[name] = values

        ranges = {
            "contact_friction": self._contact_friction_range,
            "link_mass": self._link_mass_scale_range,
        }
        self._generators = {}
        if any(values is not None for values in ranges.values()) or any(
            values is not None for values in self._episode_scale_ranges.values()
        ):
            seed = cfg.seed
            if seed < 0:
                raise ValueError(
                    "enabled domain randomization requires a resolved "
                    "non-negative cfg.seed"
                )
            for offset, (name, values) in enumerate(ranges.items()):
                if values is not None:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed + offset)
                    self._generators[name] = generator
            for name, values in self._episode_scale_ranges.items():
                if values is not None:
                    generator = torch.Generator(device=device)
                    offset = zlib.crc32(f"episode:{name}".encode())
                    generator.manual_seed(seed + offset)
                    self._generators[name] = generator

        self.link_mass_scale = None
        self._episode_targets = {}
        self._episode_nominals = {}
        self._episode_scales = {}
        self._episode_candidates = {}

    def bind_link_masses(self) -> None:
        if self._link_mass_scale_range is not None:
            self.link_mass_scale = torch.ones_like(self._backend.link_mass)

    def bind_episode_targets(self, targets: dict[str, torch.Tensor]) -> None:
        for name in self._episode_scale_ranges:
            target = targets[name]
            self._episode_targets[name] = target
            self._episode_nominals[name] = target.clone()
            self._episode_scales[name] = torch.ones_like(target)
            self._episode_candidates[name] = torch.empty_like(target)

    def episode_scale(self, name: str) -> torch.Tensor | None:
        """Return the current scale tensor, or ``None`` when not configured."""
        return self._episode_scales.get(name)

    def set_episode_scale(
        self,
        name: str,
        reset_mask: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        target = self._episode_targets[name]
        applied_scales = self._episode_scales[name]
        masked_update(applied_scales, scales, reset_mask)
        torch.mul(self._episode_nominals[name], applied_scales, out=target)

    def _sample(
        self,
        values: tuple[float, float],
        shape: tuple[int, ...],
        generator: torch.Generator,
    ) -> torch.Tensor:
        low, high = values
        return low + (high - low) * torch.rand(
            shape,
            generator=generator,
            device=self._device,
        )

    @property
    def contact_friction(self) -> torch.Tensor:
        return self._backend.contact_friction

    def _sample_contact_friction(self, count: int) -> torch.Tensor:
        generator = self._generators["contact_friction"]
        if self._contact_friction_bins is None:
            return self._sample(self._contact_friction_range, (count,), generator)
        choice = torch.multinomial(
            self._contact_friction_bin_weights,
            count,
            replacement=True,
            generator=generator,
        )
        bounds = self._contact_friction_bins[choice]
        fraction = torch.rand(count, generator=generator, device=self._device)
        return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * fraction

    def randomize_startup(self, env_ids: torch.Tensor) -> None:
        """Sample physical parameters once, before the first episode reset."""
        if env_ids.numel() == 0:
            return
        if self._contact_friction_range is not None:
            values = self._sample_contact_friction(env_ids.numel())
            self._backend.set_contact_friction(env_ids, values)

        if self._link_mass_scale_range is not None:
            scales = self._sample(
                self._link_mass_scale_range,
                (env_ids.numel(), self.link_mass_scale.shape[1]),
                self._generators["link_mass"],
            )
            self.link_mass_scale[env_ids] = scales
            self._backend.set_link_mass_scale(env_ids, scales)

    def randomize_episode(self, reset_mask: torch.Tensor) -> None:
        """Resample configured task tensors for resetting environments."""
        if self._resample_contact_friction:
            # Reset runs every control step with a mostly-false mask. A full
            # masked write avoids a GPU->host sync from nonzero(); unchanged
            # environments are rewritten with their current coefficient.
            current = self._backend.contact_friction
            candidate = self._sample_contact_friction(current.numel())
            self._backend.set_contact_friction(
                torch.arange(current.numel(), device=current.device),
                torch.where(reset_mask, candidate, current),
            )
        for name, values in self._episode_scale_ranges.items():
            low, high = values
            candidate = self._episode_candidates[name]
            candidate.uniform_(
                low,
                high,
                generator=self._generators[name],
            )
            self.set_episode_scale(name, reset_mask, candidate)
