import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp


class HYPODiscrim(nn.Module):
    """
    Replace (s,a) with (s,a,log_pi).
    """
    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions, log_pis):
        return self.net(torch.cat([states, actions, log_pis], dim=-1))

    def d(self, states, actions, log_pis):
        with torch.no_grad():
            return torch.sigmoid(self.forward(states, actions, log_pis))
