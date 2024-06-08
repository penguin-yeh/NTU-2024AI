## Installation Instructions

### 1. Create Conda Environment
```bash
conda create -y -n hw6 python=3.10
conda activate hw6
```

### 2. Install [pytorch](https://pytorch.org/get-started/locally/) based on your cuda version
In my case:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Pip install some packages
```bash
pip install tqdm packaging wandb
```

### 4. Based on cuda version install the correct version of [unsloth](https://github.com/unslothai/unsloth#-installation-instructions)
In my case:
```bash
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
```
### 5. Verify if the installation was successful
```bash
nvcc
python -m xformers.info
python -m bitsandbytes
```
## Training

### - Use default hyperparameters
```bash
bash run.sh exp_name model_name wandb_token
```
### Set custom epochs
```bash
bash run.sh exp_name model_name wandb_token num_epochs
```

### Set custom epochs and beta
```bash
bash run.sh exp_name model_name wandb_token num_epochs beta
```
