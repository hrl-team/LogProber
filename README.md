# LogProber

🔬🤖 LogProber is a cost-effective tool to measure contamination of language models on given sequences of text requiring very little information about the language model itself. It is particularly useful when working with question / answer pairs coming from benchmarks or psychology questionnaires.

📖 paper : https://www.arxiv.org/abs/2408.14352

🔬 This repository contains the code for replicating all the results presented in the paper (figures and data) and is given for transparency and replication purposes.

🌐 We also encourage people that are interested in LogProber to use the colab demo that implements the algorithm in a light manner.
colab demo : https://colab.research.google.com/drive/1GDbmEMmCVEOwhYk6-1AothdXeAlnqZ_j?usp=copy

## Step by step installation instructions
This repository contains pre-computed data to rerun the figures but also the code for rebuilding the pre-computed data. A cuda compatible GPU is required to rebuild the data.

- Install the dependencies (for both replicating figures and recomputing data from scratch)
```
pip install -r requirements.txt
```

- Download LLMs (only for recomputing data from scratch)
```
mkdir LLMs
cd LLMs
git clone https://huggingface.co/huggyllama/llama-7b #Download Llama LLM (you need to be granted access by meta AI - see description of the model)
git clone https://huggingface.co/unsloth/Qwen2.5-32B-Instruct #Download Qwen LLM
cd ..
``` 


## Documentation
This project proposes a method to estimate contamination in language models using a question-based method (e.g. by computing the similarity of the LLM on the question rather than on the answer to the question as it is classically done in the field). This also makes it very versatile and usable on general text rather than only in question-answer settings as advertised in the paper.

This research used 3 frameworks: this repository (referred to as the logprober repository contains the code used to run the contamination detection algorithm), the stanford_alpaca respository contains the code for contaminating models (and then verifying how well LogProber detects this contamination). Finally CDD-TED4LLMs is a repository implementing a classical state of the art method for detecting contamination on answers and this code is used to compare with LogProber results. For the sake of simplicity all the code is included in this repository but some disclaimers are included at the beginning of files to indicate where the code comes from in case it wasn't built by other people.

Datasets and splits are already precomputed in the inputs folder. This includes datasets about CRT experiments with format 
- [CONDITION]\_[new/old]\_crt.json containing data formatted for each training condition "qa", "a", "q" and the dataset "alpaca_data.json" for condition std. NEW/OLD corresponds to the version of CRT. These dataset include corresponding CRT items duplicated 10 times each, alpaca training data and 70 items from the CF dataset from this paper https://www.nature.com/articles/s44271-024-00091-8 (the one that defined new CRT) - they were introduced for running additional experiments that were not included in the final version of the paper and here only consist in additional data to the 10,000 alpaca training data. 
- [ORIGINAL_DATASET]\_dataset\_[CONDITION].json. ORIGINAL_DATASET can be "mmlu" or "code2" and CONDITION which can be "qa","a","q" or "std" (see paper Table 1). They consist in 100 randomly chosen items duplicated 100 times and 10,000 randomly chosen items either from MMLU or MBXP to train the models on. A test file containing these 100 items (split A in the paper) plus a new split of 100 items that is distinct from the training data can be found named [ORIGINAL_DATASET]\_test\_100.json in the inputs folder. The code for splitting the datasets is given in split_datasets.ipynb.

## Running the code
- Replicate the figures from the paper on our data: open and run logscores.ipynb. Data used in the study are included in the repository, if you don't change the paths it should run out of the box on pre-computed data.


--- DATASET RECOMPUTING (CUDA GPU REQUIRED - approx. 200GPU.H on H100 80Go) ---

- Replicate training results: run the following commands in the stanford_alpaca repository to train some models on a dataset. See previous section for dataset naming.
```
#Set the training variables
LLM=llama-7b
DATASET="comparison/mmlu_dataset_qa"
LR=0.00005
```
```
torchrun --nproc_per_node=1 --master_port=42424 train.py \
    --model_name_or_path ${LLM}\
    --data_path "inputs/${DATASET}.json" \
    --bf16 True \
    --output_dir "trained_models/${DATASET}d" \
    --logging_dir "runs/${DATASET}l" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True
    --lora True #Important note: in the paper, lora is true for the CDD comparison experiment with Qwen2.5-32B but is False for the CRT experiments with llama 7b (which may require more GPU RAM).
```
with desired $LLM (path to the language model to contaminate on the data), $DATASET (name of the dataset to train the model on) and $LR (the learning rate are ranging from 0.0005 (5e-4) to 0.000005 (5e-6) in the study).


- Compute CDD scores: run the following command
```
python run_cdd.py --model mmlu_dataset_a_0.000005d --dataset mmlu
```
where mmlu_dataset_a_0.000005d is the name of the model trained by the previous command on the mmlu_dataset_std.json dataset (see previous point about training and dataset naming) with learning rate 0.000005 (5e-5) and mmlu is the test set where we want to test contamination. This test set (mmlu_100.json) contains half questions from the training set (where the model has been contaminated on and half that where not included in the training set).

- Compute LogProber scores: run the following command
```
python run_logprober --model mmlu_dataset_a_100_100_0.000005d --dataset mmlu
```
is analogous to previous point.

- Additional files
  - colab.ipynb is a simple and versatile implementation of LogProber
  - lanlab_tutorial.ipynb is a tutorial for lanlab, a framework used to run the study.
