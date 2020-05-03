from rlpyt.algos.pg.ppo import PPO

default = {
    "discount": 0.999,
    "lambda": 0.95,
    "timesteps_per_rollout": 256,
    "epochs_per_rollout": 3,
    "minibatches_per_epoch": 8,
    "entropy_bonus": 0.01,
    "ppo_clip": 0.2,
    "learning_rate": 5e-4,
    "workers": 4,
    "envs_per_worker": 64,
    "total_timesteps": 25_000_000,
}

def makePPOExperiment(config):
    return PPO(discount=config["discount"], entropy_loss_coeff=config["entropy_bonus"],
            gae_lambda=config["lambda"], minibatches=config["minibatches_per_epoch"],
            epochs=config["epochs_per_rollout"], ratio_clip=config["ppo_clip"],
            learning_rate=config["learning_rate"], normalize_advantage=True)
