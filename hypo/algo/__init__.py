from .hypo import HYPO
from .ppo import PPO, PPOExpert


ALGOS = {
    'hypo': HYPO,
    'ppo': PPO,
}

EXP = {
    'ppo': PPOExpert
}
