WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Levels-100" --num_levels=100 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Levels-200" --num_levels=200 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Levels-500" --num_levels=500 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Levels-1000" --num_levels=1000 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Levels-10000" --num_levels=10000 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Levels-15000" --num_levels=15000 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-BatchNorm" --batchNorm=True --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Dropout-0.05" --dropout=0.05 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Dropout-0.10" --dropout=0.10 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Dropout-0.15" --dropout=0.15 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Dropout-0.20" --dropout=0.20 --env=chaser --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Dropout-0.25" --dropout=0.25 --env=chaser --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-L2-0.1" --l2_penalty=0.1E-4 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-L2-0.25" --l2_penalty=0.25E-4 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-L2-0.5" --l2_penalty=0.5E-4 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-L2-1.0" --l2_penalty=1.0E-4 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-L2-2.5" --l2_penalty=2.5E-4 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Entropy-0" --entropy_bonus=0 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Entropy-0.02" --entropy_bonus=0.02 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Entropy-0.05" --entropy_bonus=0.05 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Entropy-0.07" --entropy_bonus=0.07 --env=chaser --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Entropy-0.1" --entropy_bonus=0.1 --env=chaser --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-MaxPooling" --max_pooling=True --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-HiddenSize-256" --hidden_sizes=256 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-HiddenSize-1024" --hidden_sizes=1024 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-MLP-1" --hidden_sizes=512 256 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-MLP-2" --hidden_sizes=512 256 128 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-Depth+1" --arch=depth+1 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-Depth+2" --arch=depth+2 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-Channels-d2" --arch=channels/2 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-Channels-t2" --arch=channels*2 --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-Arch-IMPALA" --arch=impala --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-DataAug-ColorJitter" --augment_obs=jitter --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-DataAug-Cutout" --augment_obs=cutout --env=chaser
WANDB_PROJECT="rl-generalization-chaser-train" python run.py --name="Chaser-DataAug-RandConv" --augment_obs=rand_conv --env=chaser