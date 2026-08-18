"""THE editable method file (program.md): everything about how the bonus is produced lives
here — networks, initialization, optimizer, loss, readout. The experiment loop edits ONLY this
file; run_experiment.py and decay_harness/ are fixed.

Interface contract (run_experiment.py relies on exactly this):
  - class Method:
    """AdaGrad at a smaller rate (3e-3) + initial-copy readout: exp 007 (lr 1e-2) showed an
    early overshoot — every curve drops to about 0.55 after the first step while the count
    oracle stays at 1 — plus slightly-too-steep tails. Single-knob probe: does a 3x smaller
    AdaGrad rate trade the early drop against a late floor? Everything else is exp 007."""

    name = "adagrad_lr3e-3"

    def __init__(self, seed: int, obs_dim: int = 4):
        """Build target + predictor (4 -> 256 -> ReLU -> 128), the frozen init copy (readout
        denominator), and AdaGrad at lr 3e-3."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.target = make_mlp(obs_dim, 256, 128, seed, "target")
        self.predictor = make_mlp(obs_dim, 256, 128, seed, "predictor")
        for p in self.target.parameters():
            p.requires_grad_(False)
        # frozen initial predictor: the per-position readout denominator (starts the bonus at 1)
        self.predictor0 = copy.deepcopy(self.predictor)
        for p in self.predictor0.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adagrad(self.predictor.parameters(), lr=3e-3, eps=1e-10,
                                       initial_accumulator_value=0)

    def update(self, x: torch.Tensor) -> None:
        """One AdaGrad step on the MSE distillation loss."""
        e = self.predictor(x) - self.target(x)
        loss = 0.5 * e.pow(2).sum(dim=1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """Per-point normalized residual ||g(x)-f(x)|| / ||g_0(x)-f(x)|| (exactly 1 at t=0)."""
        with torch.no_grad():
            f = self.target(x)
            e = (self.predictor(x) - f).pow(2).sum(dim=1).sqrt()
            e0 = (self.predictor0(x) - f).pow(2).sum(dim=1).sqrt()
            return (e / e0).cpu().numpy().astype(np.float64)
