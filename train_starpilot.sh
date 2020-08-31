WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Levels-100" --num_levels=100 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Levels-200" --num_levels=200 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Levels-500" --num_levels=500 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Levels-1000" --num_levels=1000 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Levels-10000" --num_levels=10000 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Levels-15000" --num_levels=15000 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-BatchNorm" --batchNorm=True --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Dropout-0.05" --dropout=0.05 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Dropout-0.10" --dropout=0.10 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Dropout-0.15" --dropout=0.15 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Dropout-0.20" --dropout=0.20 --env=starpilot --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Dropout-0.25" --dropout=0.25 --env=starpilot --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-L2-0.1" --l2_penalty=0.1E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-L2-0.25" --l2_penalty=0.25E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-L2-0.5" --l2_penalty=0.5E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-L2-1.0" --l2_penalty=1.0E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-L2-2.5" --l2_penalty=2.5E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Entropy-0" --entropy_bonus=0 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Entropy-0.02" --entropy_bonus=0.02 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Entropy-0.05" --entropy_bonus=0.05 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Entropy-0.07" --entropy_bonus=0.07 --env=starpilot --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Entropy-0.1" --entropy_bonus=0.1 --env=starpilot --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-MaxPooling" --max_pooling=True --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-HiddenSize-256" --hidden_sizes=256 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-HiddenSize-1024" --hidden_sizes=1024 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-MLP-1" --hidden_sizes=512 256 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-MLP-2" --hidden_sizes=512 256 128 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-Depth+1" --arch=depth+1 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-Depth+2" --arch=depth+2 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-Channels-d2" --arch=channels/2 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-Channels-t2" --arch=channels*2 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-Arch-IMPALA" --arch=impala --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-DataAug-ColorJitter" --augment_obs=jitter --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-DataAug-Cutout" --augment_obs=cutout --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train-2" python run.py --name="Starpilot-DataAug-RandConv" --augment_obs=rand_conv --env=starpilot