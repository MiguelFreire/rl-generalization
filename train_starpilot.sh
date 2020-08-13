WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Levels-100" --num_levels=100 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Levels-200" --num_levels=200 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Levels-500" --num_levels=500 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Levels-1000" --num_levels=1000 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Levels-10000" --num_levels=10000 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Levels-15000" --num_levels=15000 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-BatchNorm" --batchNorm=True --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Dropout-0.05" --dropout=0.05 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Dropout-0.10" --dropout=0.10 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Dropout-0.15" --dropout=0.15 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Dropout-0.20" --dropout=0.20 --env=starpilot --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Dropout-0.25" --dropout=0.25 --env=starpilot --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-L2-0.1" --l2_penalty=0.1E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-L2-0.25" --l2_penalty=0.25E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-L2-0.5" --l2_penalty=0.5E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-L2-1.0" --l2_penalty=1.0E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-L2-2.5" --l2_penalty=2.5E-4 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Entropy-0" --entropy_bonus=0 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Entropy-0.02" --entropy_bonus=0.02 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Entropy-0.05" --entropy_bonus=0.05 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Entropy-0.07" --entropy_bonus=0.07 --env=starpilot --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Entropy-0.1" --entropy_bonus=0.1 --env=starpilot --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Arch-MaxPooling" --max_pooling=True --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Arch-HiddenSize-256" --hidden_sizes=256 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Arch-HiddenSize-1024" --hidden_sizes=1024 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Arch-Depth+1" --arch=depth+1 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Arch-Depth+2" --arch=depth+2 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Arch-Channels-d2" --arch=channels/2 --env=starpilot
WANDB_PROJECT="rl-generalization-starpilot-train" python run.py --name="StarpilotNature-Arch-Channels-t2" --arch=channels*2 --env=starpilot