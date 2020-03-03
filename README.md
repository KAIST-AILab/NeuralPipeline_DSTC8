# NeuralPipeline_DSTC8

Our code is developed on the ConvLab github page (https://github.com/ConvLab/ConvLab).

## Environment setting

conda version : 4.7.10
python version : 3.6.5

Before creating conda environment, please edit env.yml to fit on your conda root path.
For example, \'/home/jglee/anaconda\'.

```
conda env create -f env.yml
conda activate neural_pipeline
```

## How to train

The working directory is $ROOT/Convlab.
The description below follows the working directory.

```
cd ConvLab # (working directory)
cd data/multiwoz
unzip total_v4.zip
unzip val_v4.zip
cd ../../  # (working directory)
python -m torch.distributed.launch --nproc_per_node=${#OfGPUs, e.g.2} convlab/modules/e2e/multiwoz/Transformer/train.py --dataset_path=./data/multiwoz/ --dataset_cache=./dataset_cache --model_checkpoint=gpt2 --model_version=v4 --lm_coef=2.0 --max_history=20 --gradient_accumulation_steps=4
```

`-m torch.distributed.launch --nproc_per_node=${#OfGPUs}` part is to use multi GPUs. 

Please refer to huggingface's TransferTransfo (https://github.com/huggingface/transfer-learning-conv-ai.) 

save folder path: /runs/${DATES}_${HOSTNAME} e.g. Mar03_13-31-00_hostname


## How to test on ConvLab

In convlab/modules/e2e/multiwoz/Transformer/Transformer.py, the Transformer class manages our algorithm.

The weight files we fine-tuned will be downloaded into /models folder when running 

```
python run.py submission.json submission${SUBMISSION_NUMBER e.g.4} eval
```

If you want to evaluate your own fine-tuned weights, please handle the "model_checkpoint" on the right submission name (e.g. submission4) in 'convlab/spec/submission.json'.

## Credit

Our code is based on huggingface's TransferTransfo (https://github.com/huggingface/transfer-learning-conv-ai.)


