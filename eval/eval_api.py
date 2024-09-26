import sys
sys.path.append("../")
import argparse
import csv
import json
from models.chatgpt import chat_with_backoff
from tqdm import tqdm
import datetime
import pandas as pd
from eval.build_instruction_applicability import build_instruction_applicability
from eval.build_instruction_compliance import build_instruction_compliance
import os
from parse_eval_result import parse_eval_result

def generate_prompt(instruction: str, input: str = None) -> str:
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction: {instruction}\n### Input: {input}\n### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction: {instruction}\n### Response:"""
    
def build_prompt_legal_expert(msg="How are you today?"):
    messages = [
        # {"role": "system", "content": "You are a legal expert that answers question as simple as possible."},
        {"role": "user", "content": msg}
                ]
    return messages

def eval_cot_api(args):
    with open(args.output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["regulation id", "compliance", "response"])
    df = pd.read_csv(args.input_file)
    start = 0
    for i in tqdm(range(start, len(df))):
        row = df.iloc[i]
        regulation_id = row["refer_regulation"]
        results = row["generate_HIPAA_type"]
        if args.task == "compliance":
            instruction = build_instruction_compliance(row, args.mode, False)
        else:
            instruction = build_instruction_applicability(row, args.mode, False)
        prompt = generate_prompt(instruction["instruction"], instruction["input"])
        prompt = build_prompt_legal_expert(msg=prompt)
        print(prompt)
        resp = chat_with_backoff(model=args.model, messages=prompt, max_tokens=512, temperature=0.7)
        if resp == None:
            resp = "No response"
        else:
            resp = resp.choices[0].message.content.strip().replace("\n", " ")
        print(resp)
        with open(args.output_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([regulation_id, results, resp])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="gpt-35-turbo")
    parser.add_argument("-task", type=str, default="compliance", choices=["compliance", "applicability"], help="Choose the task: compliance or applicability")
    parser.add_argument("-mode", type=str, default="cot", choices=["cot", "direct"], help="Choose the mode: cot or direct")
    parser.add_argument("-input_file", type=str, help="Input file path")
    
    args = parser.parse_args()
    if not args.input_file:
        args.input_file = f"../cases/train_val_test/test_real_cases_hipaa_{args.task}.csv"

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    args.output_file = f"results/{timestamp}_eval_{args.mode}_real_cases_test_{args.model}_{args.task}.csv"
    
    os.makedirs("results", exist_ok=True)
    
    eval_cot_api(args)
    parse_eval_result(eval_file=args.output_file)