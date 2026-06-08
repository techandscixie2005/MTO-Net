"""MTO Activity Gate with configurable modes: none, simple, fermi_dirac.

Per TOTAL.md section 9:
  - "fermi_dirac": n_k = 2*sigmoid((mu-epsilon_k)/theta), sum n_k = N_val
  - "simple": n_k = sigmoid(epsilon_k)
  - "none": n_k = 1.0 (all valid slots active)

n_k is a soft activity allocation, NOT a physical occupation number.
No molecular charge Q is used in any mode.
"""
import torch
import torch.nn as nn


def _sigmoid_safe(x):
    return torch.sigmoid(x.clamp(-30.0, 30.0))


# ---- Fermi-Dirac Gate with Implicit Differentiation ----

class FermiDiracGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, epsilon, theta, N_val, mask):
        B, K_max = epsilon.shape
        device = epsilon.device
        mask_bool = mask.bool()
        mu_vals = torch.zeros(B, device=device)
        n_vals = torch.zeros(B, K_max, device=device)

        for b in range(B):
            nv = N_val[b].item()
            if nv <= 0:
                continue
            eps_b = epsilon[b]
            m_b = mask_bool[b]
            valid_eps = eps_b[m_b]
            k_valid = valid_eps.numel()
            if k_valid == 0:
                continue
            target = float(nv)
            lo = valid_eps.min().item() - 50.0
            hi = valid_eps.max().item() + 50.0
            # Bracket the solution
            while (2.0 * _sigmoid_safe((lo - valid_eps) / theta)).sum().item() > target:
                lo -= 20.0
            while (2.0 * _sigmoid_safe((hi - valid_eps) / theta)).sum().item() < target:
                hi += 20.0
            for _ in range(60):
                mid = (lo + hi) * 0.5
                x_mid = (2.0 * _sigmoid_safe((mid - valid_eps) / theta)).sum().item()
                if x_mid > target:
                    hi = mid
                else:
                    lo = mid
            mu_b = (lo + hi) * 0.5
            n_valid = 2.0 * _sigmoid_safe((mu_b - valid_eps) / theta)
            mu_vals[b] = mu_b
            n_vals[b][m_b] = n_valid

        ctx.save_for_backward(epsilon, n_vals, mask_bool, N_val.float())
        ctx.theta = theta
        return n_vals, mu_vals

    @staticmethod
    def backward(ctx, grad_n, grad_mu):
        epsilon, n, mask, N_val = ctx.saved_tensors
        theta = ctx.theta
        B, K_max = epsilon.shape
        grad_epsilon = torch.zeros_like(epsilon)
        for b in range(B):
            m_b = mask[b]
            n_valid = n[b][m_b]
            k_valid = n_valid.numel()
            if k_valid == 0:
                continue
            dk = (1.0 / theta) * n_valid * (1.0 - n_valid / 2.0)
            d_sum = dk.sum()
            g_valid = grad_n[b][m_b]
            gd_sum = (g_valid * dk).sum()
            grad_epsilon[b][m_b] = dk * (gd_sum / d_sum - g_valid)
        return grad_epsilon, None, None, None


def _fermi_dirac_gate(epsilon, N_val, mask, theta):
    return FermiDiracGateFunction.apply(epsilon, theta, N_val, mask)


# ---- Simple Sigmoid Gate ----

def _simple_gate(epsilon, mask):
    n = torch.sigmoid(epsilon)
    n = n * mask.float()
    return n


# ---- Activity Gate Module ----

class ActivityGate(nn.Module):
    """Configurable MTO activity gate.

    Modes:
      - "none": all valid slots active (n_k = mask)
      - "simple": n_k = sigmoid(epsilon_k), no charge budget
      - "fermi_dirac": n_k from FD gate, sum_k n_k = N_val (TOTAL 9)

    n_k are soft activity allocations for MTO response modes.
    They are NOT physical orbital occupation numbers.
    No molecular charge Q is used.
    """

    def __init__(self, feature_dim, hidden_dim=64, mode="simple"):
        super().__init__()
        self.mode = mode
        self.feature_dim = feature_dim
        if mode in ("simple", "fermi_dirac"):
            self.epsilon_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, O, mask, N_val=None, theta=0.5):
        B, K, C = O.shape

        if self.mode == "none":
            n = mask.float()
            mu = torch.zeros(B, device=O.device)
            O_active = O * n.unsqueeze(-1)
            return n, O_active, None, mu

        epsilon = self.epsilon_net(O).squeeze(-1)
        epsilon = epsilon * mask.float()

        if self.mode == "simple":
            n = _simple_gate(epsilon, mask)
            mu = torch.zeros(B, device=O.device)
            O_active = O * n.unsqueeze(-1)
            return n, O_active, epsilon, mu

        elif self.mode == "fermi_dirac":
            if N_val is None:
                raise ValueError("fermi_dirac mode requires N_val (valence electrons)")
            n, mu = _fermi_dirac_gate(epsilon, N_val, mask, theta)
            O_active = O * n.unsqueeze(-1)
            return n, O_active, epsilon, mu

        else:
            raise ValueError(f"Unknown activity gate mode: {self.mode}")


# ---- Factory ----

def build_activity_gate(feature_dim, config=None):
    """Build an ActivityGate from a config dict.

    Args:
        feature_dim: int, feature dimension
        config: dict with keys:
            mode: "none", "simple" (default), or "fermi_dirac"
            hidden_dim: int (default 64)
    """
    if config is None:
        config = {}
    mode = config.get("mode", "simple")
    hidden_dim = config.get("hidden_dim", 64)
    return ActivityGate(feature_dim, hidden_dim=hidden_dim, mode=mode)
