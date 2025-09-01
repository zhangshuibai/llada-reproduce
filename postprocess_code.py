import evaluate as hf_evaluate
import os
import sys
from sanitize import sanitize

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


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


# file_path = sys.argv[1]
# file_path = "/data/szhang967/samples_humaneval_plus_2025-09-01T11-08-17.955719.jsonl"#temperature=0.2 humaneval_plus
#file_path ="/data/szhang967/samples_humaneval_plus_2025-09-01T01-29-35.428420.jsonl"#temperature=0.7 humaneval_plus
# file_path = "/data/szhang967/samples_humaneval_2025-09-01T01-29-35.428420.jsonl"#temperature=0.7 humaneval
file_path = "/data/szhang967/samples_humaneval_2025-09-01T11-08-17.955719.jsonl"#temperature=0.2 humaneval
# import pdb
# pdb.set_trace()
data = read_jsonl(file_path)

#/data/szhang967/samples_humaneval_plus_2025-08-31T03-10-11.384348.jsonl
#samples_humaneval_plus_2025-08-31T03-10-11.384348.jsonl

references = [sample['target'] for sample in data]

predictions = [[sanitize(sample['doc']['prompt'] + "\n" + resp.split('```python\n', 1)[-1].split('```')[0], 
                sample['doc']["entry_point"]) for resp in sample['resps'][0]] 
                for sample in data]

# Calculate pass@k for different k values
k_values = [1, 5, 10]  # Calculate for multiple k values

pass_at_k_results = pass_at_k(references, predictions, k=k_values)

print("Pass@k Results:")
for k in k_values:
    avg_pass_at_k = pass_at_k_results[f'pass@{k}']
    print(f"Pass@{k}: {avg_pass_at_k:.4f}")

# Note: If pass@k values are the same across different k, 
# it means the predictions for each sample are identical or very similar

# Keep pass_at_1s for backward compatibility with the output file  
pass_at_1s = [pass_at_k([ref], [pred], k=[1])['pass@1'] for ref, pred in zip(references, predictions)]

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

# res = [{"task_id": sample['doc']['task_id'], "completion": pred[0], "pass_at_1": res} 
#        for sample, pred, res  in zip(data, predictions, pass_at_1s)]
# write_jsonl(res, file_path+'.cleaned')