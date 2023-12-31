# Hybrid Policy Optimization

This is a PyTorch implementation of HYbrid Policy Optimization (HYPO).
It is easy for readers to repetition the results in the main article in the widely used MuJoCo environments by following this instruction.

## Setup

You can install the Python liblaries by following the `requirements.txt` with Python 3.7.
Note that there are several components which are required to install manually (e.g., the MoJoCo).

## Example

### Train expert

You can train experts using PPO in the dense reward setting. Also, we have prepared an expert model at `models`.
You can use it if you're only interested in the experiments ahead.

```shell
python train_expert.py \
  --env_train HalfCheetah-v2 \
  --env_eval HalfCheetah-v2 \
  --algo ppo \
  --num_step 1000000 \
  --seed 0
```

### Collect demonstrations

You need to collect demonstrations using a partly trained expert's model. Note that `--std` specifies the standard
deviation of the gaussian noise added to the action, and `--p_rand` specifies the probability the expert
acts randomly. We set `std` to 0.01 not to collect too similar trajectories.
Moreover, we also have prepared an example demonstrations in `buffers`, which is collected by a suboptimal expert
with 1500 average reward in HalfCheetah task. You can use it if you're only interested in the experiments ahead.
```shell
python collect_demo.py \
  --expert_algo tppo \
  --weight models/HalfCheetah-v2/actor_rwd1500.pth \
  --env_id HalfCheetah-v2 \
  --buffer_size 10000 \
  --std 0.01 \
  --reward 1500 \
  --seed 0

```

### Train HYPO

Once the expert data collection is complete, HYPO training can begin with the following command:
```shell
python train_hypo.py \
  --buffer buffers/HalfCheetah-v2/size10000_std0.01_prand0.0_rwd1500.pth \
  --num_steps 10000000 \
  --env_train HalfCheetahSparse \
  --env_eval HalfCheetah-v2 \
  --seed 0
```

### Results

The main results in MuJoCo simulation:

![](https://github.com/joenghl/HYPO/blob/master/assets/mujoco_main.jpg?raw=true)

Performance of the baselines:

- LOGO: The results of LOGO were directly conducted using the publicly available LOGO [source code](https://github.com/DesikRengarajan/LOGO), without any modifications. The key point is that, the number samples(x-axis) and the quality (performance and number) of the  trajectories are different in this setting.
- GAIL & POfD: This two algorithms work well with lots of high-cumulative-return trajectories using this [PyTorch implementation](https://github.com/toshikwa/gail-airl-ppo.pytorch), but they are affected susceptibly by the quality of data.

## Reference

[1] HYPO Paper: https://openreview.net/forum?id=LftAvFt54C

[2] https://github.com/toshikwa/gail-airl-ppo.pytorch
