# Proposed method

The official implementation for the proposed method


## Highlights

| Tracker     | GOT-10K (AO) | LaSOT (AUC) | TrackingNet (AUC) | LaSOT_ext (AUC) | VOT2020 (EAO) | TNL2K (AUC) | OTB(AUC) |
|:-----------:|:------------:|:-----------:|:-----------------:|:-----------:|:-----------:|:-----------:|:-----------:|
| ODTrack-L | 78.2         | 74.0        | 86.1              | 53.9          | 0.605          | 61.7          | 72.4          |
| ODTrack-B | 77.0         | 73.1        | 85.1              | 52.4          | 0.581          | 60.9          | 72.3          |




## Install the environment
```
conda create -n odtrack python=3.8
conda activate odtrack
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```


## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lsotb-tir
            |-- test
            |-- train
            |-- val
   ```


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_networks` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script odtrack --config baseline --save_dir ./output --use_wandb 1 --mode single
```

Replace `--config` with the desired model config under `experiments/odtrack`.

We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Test and Evaluation

- LSOTB-TIR or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py odtrack baseline --dataset lsotb --runid 300 --threads 8 --num_gpus 2
python tracking/analysis_results.py # need to modify tracker configs and names
```


## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

```
python tracking/profile_model.py --script odtrack --config baseline
```


## Acknowledgments
* Thanks for the [STARK](https://github.com/researchmm/Stark) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.


## Citation
If our work is useful for your research, please consider citing:

```Bibtex
@inproceedings{zheng2024odtrack,
  title={ODTrack: Online Dense Temporal Token Learning for Visual Tracking}, 
  author={Yaozong Zheng and Bineng Zhong and Qihua Liang and Zhiyi Mo and Shengping Zhang and Xianxian Li},
  booktitle={AAAI},
  year={2024}
}
```
