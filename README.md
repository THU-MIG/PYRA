# PYRA
The official implementation of our ECCV 2024 publication, PYRA (Parallel Yielding Re-Activation).

## News

- 2024-08-19: We have released our source code for PYRA!

- 2024-07-17: The [arXiv](https://arxiv.org/abs/2403.09192) version of our ECCV final submission is now released!

- 2024-07-03: The code coming soon. We promise that it will be available before the main conference date.

## Preparation

### Datasets

We use the VTAB-1k dataset to evaluate our proposed PYRA. Use instructions in directory `data/vtab-source` to build VTAB-1k dataset locally (Internet access is demanded).

### Environment

We use Anaconda or Miniconda to maintain the fine-tuning environment of PYRA. Simply follow the following instructions to prepare the environment:

```
conda create -n PYRA python=3.8
conda activate PYRA
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install timm==0.5.4 jupyter
pip install scikit-image ptflops easydict PyYAML pillow opencv-python scipy mmcv==1.7.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U fvcore
```

You can use other mirrors beside tuna.tsinghua.edu.cn as long as everything is installed successfully.

### Checkpoints

Use [this link](https://drive.google.com/file/d/1MEzqBikrYIwmdCrIsXBtDp_QS8oOwhED/view?usp=sharing) to download all pre-trained model weights used for task adaptation in PYRA. After downloading all model checkpoints, unzip them to the `weight/` directory under the main directory (`PYRA/`).

## Training & Evaluation

We provide training scripts for the experiments reported in the article. In the default pipeline, evaluation is executed in between training epochs. All training details are saved in the `logs/` directory. You can browse the results and training details in the folders of corresponding experiments.

To conduct fine-tuning, simply run the scripts under the `scripts/` directory.

WARNING: Simply training PYRA with prompt tuning leads to problems, as prompt tokens might be merged! 

## FAQs

If you have any questions, please submit the issues describing your question as detailed as possible. If you don't see our reply in a fairly long time, please email me at: xiongyizhe2001@163.com.

## Acknowledgments

This codebase is built upon [NOAH](https://github.com/ZhangYuanhan-AI/NOAH), [ToMe](https://github.com/facebookresearch/ToMe), and [timm](https://github.com/huggingface/pytorch-image-models).

Many thanks to their great work!