# NeuralPipeline_DSTC8

Our code is developed on the ConvLab github page (https://github.com/ConvLab/ConvLab).

## Environment setting

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

python -m torch.distributed.launch --nproc_per_node=${#OfGPUs, e.g.2} convlab/modules/e2e/multiwoz/Transformer/train.py --dataset_path=data/multiwoz/ --dataset_cache=./dataset_cache --model_checkpoint=gpt2 --model_version=v4 --lm_coef=2.0 --max_history=20 --gradient_accumulation_steps=4
```

`-m torch.distributed.launch --nproc_per_node=${#OfGPUs}` part is to use multi GPUs. 
please refer to (https://github.com/huggingface/transfer-learning-conv-ai.) 

save folder path: /runs/${DATES}_${hostname}


## How to test on ConvLab

In convlab/modules/e2e/multiwoz/Transformer/Transformer.py, the Transformer class manages our algorithm.

We already provides the fine-tuned model to be downloaded into /models folder when running 

```
python run.py submission.json submission${SUBMISSION_NUMBER e.g.4} eval
```

If you want to evaluate your own fine-tuned weights, please handle the "model_checkpoint" on the right submission name in 'convlab/spec/submission.json'.

## Credit
Our code is based on https://github.com/huggingface/transfer-learning-conv-ai.


