WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Levels-100" --num_levels=100 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Levels-200" --num_levels=200 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Levels-500" --num_levels=500 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Levels-1000" --num_levels=1000 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Levels-10000" --num_levels=10000 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Levels-15000" --num_levels=15000 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-BatchNorm" --batchNorm=True --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Dropout-0.05" --dropout=0.05 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Dropout-0.10" --dropout=0.10 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Dropout-0.15" --dropout=0.15 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Dropout-0.20" --dropout=0.20 --env=coinrun --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Dropout-0.25" --dropout=0.25 --env=coinrun --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-L2-0.1" --l2_penalty=0.1E-4 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-L2-0.25" --l2_penalty=0.25E-4 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-L2-0.5" --l2_penalty=0.5E-4 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-L2-1.0" --l2_penalty=1.0E-4 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-L2-2.5" --l2_penalty=2.5E-4 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Entropy-0" --entropy_bonus=0 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Entropy-0.02" --entropy_bonus=0.02 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Entropy-0.05" --entropy_bonus=0.05 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Entropy-0.07" --entropy_bonus=0.07 --env=coinrun --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Entropy-0.1" --entropy_bonus=0.1 --env=coinrun --num_steps=35_000_000
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-MaxPooling" --max_pooling=True --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-HiddenSize-256" --hidden_sizes=256 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-HiddenSize-1024" --hidden_sizes=1024 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-MLP-1" --hidden_sizes=512 256 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-MLP-2" --hidden_sizes=512 256 128 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-Depth+1" --arch=depth+1 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-Depth+2" --arch=depth+2 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-Channels-d2" --arch=channels/2 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-Channels-t2" --arch=channels*2 --env=coinrun
WANDB_PROJECT="rl-generalization-coinrun-train-2" python run.py --name="CoinRun-Arch-IMPALA" --arch=impala --env=coinrun