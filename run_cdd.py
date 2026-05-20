#Code from 
import unsloth
import argparse
from datasets import DatasetDict,Dataset
import tqdm
import os
import json
import torch
import math

DATAPATH = 'inputs'
MODELPATH = 'trained_models'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch_size', type=int, default=25, help='Size of each batch')
parser.add_argument('--total_samples', type=int, default=50, help='Total number of samples to generate per prompt')

def print_layer_devices(model):
    print(f"Model class: {model.__class__.__name__}")
    print("Devices of each layer/module:\n")
    
    for name, module in model.named_modules():
        # Only print leaf modules to avoid redundant hierarchy
        if len(list(module.children())) == 0:
            for param in module.parameters(recurse=False):
                print(f"{name or '[root]'}: {param.device}")
                break  # One parameter is enough to determine the device

if __name__ == '__main__':
    args = parser.parse_args()
    
    model_path = os.path.join(MODELPATH, args.model)
    model_name = args.model
    
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        resize_model_vocab=151666
    )
    model = unsloth.FastLanguageModel.for_inference(model)

    print_layer_devices(model)
    
    if args.dataset == 'mmlu':
        dataset_path = os.path.join(DATAPATH, 'mmlu_test_100.json')
        prefix = f"INSTRUCTION :Give the answer to the following question and stop the generation after the answer by generating an EOS token.\n\n"
    elif args.dataset == 'code2':
        dataset_path = os.path.join(DATAPATH, 'code2_test_100.json')
        prefix = f"INSTRUCTION :Complete the following function and stop the generation after the function is complete by generating an EOS token.\n\n"
        
    dataset_name = args.dataset
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        
    d = {
        'prompt': [],
        'samples': [],
        'greedy_sample': [],
        'label': []
    }

    
    
    # Calculate number of batches needed
    num_batches = math.ceil(args.total_samples / args.batch_size)
    # Calculate size of last batch (in case total_samples isn't evenly divisible by batch_size)
    last_batch_size = args.total_samples - (num_batches - 1) * args.batch_size
    
    for i in tqdm.tqdm(range(0, len(dataset))):
        all_samples = []
        
        # Process each batch
        for batch_idx in range(num_batches):
            # Determine current batch size
            current_batch_size = args.batch_size if batch_idx < num_batches - 1 else last_batch_size
            batch_prompt = [prefix+dataset[i]['prompt'] for _ in range(current_batch_size)]
            
            # Tokenize and generate
            tokenized_batch = torch.tensor(tokenizer(batch_prompt).input_ids).to('cuda')
            print(f"Batch {batch_idx + 1} shape: {tokenized_batch.shape}")
            
            with torch.no_grad():
                tokenized_samples_out = model.generate(
                    tokenized_batch,
                    max_new_tokens=100,
                    do_sample=True
                ).cpu()
            
            # Decode and extend results
            samples_out = tokenizer.batch_decode(tokenized_samples_out)
            all_samples.extend(samples_out)
        
        # Verify we got the expected number of samples
        assert len(all_samples) == args.total_samples, f"Expected {args.total_samples} samples but got {len(all_samples)}"
        
        # Process greedy sample (using first prompt)
        with torch.no_grad():
            tokenized_greedy_out = model.generate(tokenized_batch[0][None],
            max_new_tokens=100).cpu()
            
        greedy_out = tokenizer.batch_decode(
            tokenized_greedy_out
        )
        
        d['prompt'].append(dataset[i]['prompt'])
        d['samples'].append(all_samples)
        d['greedy_sample'].append(greedy_out)
        d['label'].append(i < 100)
    dataset_dict = DatasetDict({
        "data": Dataset.from_dict(d)
    })
    dataset_dict.save_to_disk(f'data/comparison-cdd-rebuilt/{model_name}-{dataset_name}')