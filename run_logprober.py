from lanlab.studies.parametrized_studies.logscore import LogScoreStudy
from lanlab.data_management.loader.dataset_loader.question_dataset_loader import QuestionDatasetLoader
from lanlab.data_management.loader.sequence_loader.question_loader import QuestionLoader
from lanlab.models.hf_models import AutoUnslothModel

import argparse
import os

import logging
logging.getLogger().setLevel(logging.INFO)

MODEL_PATH = "trained_models"

class MMLUTestQuestionDataset(QuestionDatasetLoader):
    def __init__(self):
        super().__init__(d=None,name='mmlutest')
        self.from_json('inputs/mmlu_test_100.json',QuestionLoader)

class Code2TestQuestionDataset(QuestionDatasetLoader):
    def __init__(self):
        super().__init__(d=None,name='codetest')
        self.from_json('inputs/code2_test_100.json',QuestionLoader)

if __name__ == "__main__":
    #Parse args
    parser = argparse.ArgumentParser(description='Logscore analysis')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset',type=str)

    args = parser.parse_args()
    model_path = os.path.join(MODEL_PATH,args.model_path)

    dataset = args.dataset

    if dataset == 'mmlu':
        dataset = MMLUTestQuestionDataset()
    elif dataset == 'code2':
        dataset = Code2TestQuestionDataset()

    print(model_path)

    model = AutoUnslothModel(model_path=model_path)
    with model:
        ls_study = LogScoreStudy(dataset,model,name='comparison-logprober-rebuilt')
        ls_study.frun()