import evaluate as hf_evaluate
import os
import sys
from sanitize import sanitize

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
pass_at_k = hf_evaluate.load("code_eval")

def pass_at_1(references, predictions):
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]

import json

def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]
        
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error in line {i}: {e}")
                print("Line content:", line[:200], "...")  
                raise
    return data


file_path = sys.argv[1]
# import pdb
# pdb.set_trace()
data = read_jsonl(file_path)

#/home/jovyan/llada-reproduce/Dream/eval/evals_results/test/Dream-org__Dream-v0-Base-7B/samples_humaneval_plus_2025-09-01T11-08-17.955719.jsonl


references = [sample['target'] for sample in data]

predictions = [[sanitize(sample['doc']['prompt'] + "\n" + sample['resps'][0][0].split('```python\n', 1)[-1].split('```')[0], 
                sample['doc']["entry_point"])] 
                for sample in data]

import pdb
pdb.set_trace()

pass_at_1s = [pass_at_1([reference], [prediction]) for reference, prediction in zip(references, predictions)]
print(sum(pass_at_1s)/len(pass_at_1s))

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

res = [{"task_id": sample['doc']['task_id'], "completion": pred, "pass_at_1": res} 
       for sample, pred, res  in zip(data, predictions, pass_at_1s)]
write_jsonl(res, file_path+'.cleaned')