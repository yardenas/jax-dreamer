# jax-dreamer
An implementation of [Dreamer](https://github.com/danijar/dreamer), a model-based reinforcement learning algorithm which uses model-generated (a.k.a 'imagined') experience to learn a policy in a generalized policy iteration scheme.

## Installation
```
conda create -n jax-dreamer python=3.6
conda activate jax-dreamer
pip3 install -r requirements.txt
```
## Experiments

```
python3 train.py --configs defaults pendulum --log_dir pendulum
```


