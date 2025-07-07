
# Guide for Evaluating SEM Models on the RoboTwin 2.0 Platform (Example: `place_empty_cup` Task)

This guide explains how to evaluate **SEM models** on the **RoboTwin 2.0** platform, using the `place_empty_cup` task as an example.

---

## Step 1: Copy the Config File to the Policy Folder

```bash
cd projects/sem/robotwin/sem_policy
cp ../config_sem_robotwin.py ./
```

## Step 2: Modify the Config Settings

Open `config_sem_robotwin.py` and update the URDF path to match the RoboTwin 2.0 asset path.

## Step 3: Prepare the Model Checkpoint

```bash
CHECKPOINT_DIR=checkpoints/place_empty_cup
mkdir -p ${CHECKPOINT_DIR}
cp {PATH_TO_YOUR_SAFETENSOR_CKPT} ${CHECKPOINT_DIR}
```

Replace `{PATH_TO_YOUR_SAFETENSOR_CKPT}` with the path to your `.safetensors` model file.

## Step 4: Copy the Policy Folder to the RoboTwin 2.0 Project Directory

```bash
cd ..
cp -r sem_policy {ROBOTWIN2_PATH}/policy/
```

Replace `{ROBOTWIN2_PATH}` with the path to your local RoboTwin 2.0 project.

## Step 5: Run the Model Evaluation Script

```bash
cd {ROBOTWIN2_PATH}/policy/sem_policy
sh eval.sh
```