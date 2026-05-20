from lanlab.studies.parametrized_studies.logscore import LogScoreStudy
from lanlab.data_management.loader.dataset_loader.question_dataset_loader import QuestionDatasetLoader
from lanlab.data_management.loader.sequence_loader.question_loader import QuestionLoader
from lanlab.models.hf_models import AutoUnslothModel

import argparse
import os

import logging
logging.getLogger().setLevel(logging.INFO)

MODEL_PATH = "LLMs"

class MMLUTestQuestionDataset(QuestionDatasetLoader):
    def __init__(self):
        super().__init__(d=None,name='mmlutest')
        self.from_json('/lustre/fswork/projects/rech/xob/uct12ku/git/logscores/stanford_alpaca/data/mmlu_test_100.json',QuestionLoader)

class Code2TestQuestionDataset(QuestionDatasetLoader):
    def __init__(self):
        super().__init__(d=None,name='codetest')
        self.from_json('/lustre/fswork/projects/rech/xob/uct12ku/git/logscores/stanford_alpaca/data/code2_test_100.json',QuestionLoader)

if __name__ == "__main__":
    #Parse args
    parser = argparse.ArgumentParser(description='Logscore analysis')
    parser.add_argument('--model_path', type=str, default="tiny-random-LlamaForCausalLM",help='model path')
    parser.add_argument('--dataset',type=str)

    args = parser.parse_args()
    model_path = args.model_path

    dataset = args.dataset

    if dataset == 'mmlu':
        dataset = MMLUTestQuestionDataset()
    elif dataset == 'code2':
        dataset = Code2TestQuestionDataset()

    model = AutoUnslothModel(model_path=model_path)
    with model:
        ls_study = LogScoreStudy(dataset,model,name='comparison-logprober-rebuilt')
        ls_study.frun()