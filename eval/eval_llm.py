import sys
sys.path.append("../")
import argparse
import csv
import json
from models.mistral import Mistral
from models.llama import Llama
from models.mpt import MPT
from tqdm import tqdm
import pandas as pd
from build_instruction_compliance import build_instruction_compliance
from build_instruction_applicability import build_instruction_applicability
from parse_eval_result import parse_eval_result
from peft import PeftModel
import datetime
import os

def generate_prompt(instruction: str, input: str = None) -> str:
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction: {instruction}\n### Input: {input}\n### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction: {instruction}\n### Response:"""

def add_few_shot_instruction(instruction, type):
    if type == "compliance":
        few_shot = "Read the case: Dr. Smith, a researcher at XYZ University, is conducting a study on the long-term effects of a specific medication on patients with a rare disease. To gather data, Dr. Smith needs access to the protected health information (PHI) of patients who have been taking the medication for the past five years. Dr. Smith's research has been approved by the institutional review board (IRB) of XYZ University.The IRB determined that the PHI necessary for Dr. Smith's research includes information about the patients' diagnoses, treatment plans, and medication history. Dr. Smith requested the necessary PHI from ABC Hospital, where most of the patients with the rare disease have been treated. The hospital's privacy board reviewed the request and found that the information needed was in line with the IRB's determination.Dr. Smith was provided with the PHI of patients who met the criteria of the study, without the need for individual patients' authorization, since the privacy board had approved the waiver for research purposes. The PHI was sent to Dr. Smith by the hospital's health information manager, ensuring that only the necessary information was shared, without any direct identifiers like names or addresses.\n ### Response: HIPAA Privacy Rule permits the case. \n ### Input: Read the case: Jane Smith, a patient at Healthy Life Clinic, had recently undergone a medical procedure and was concerned about her privacy. She requested Dr. Johnson, her primary care physician, to restrict the use and disclosure of her protected health information (PHI) related to the procedure. Dr. Johnson agreed to the restriction, and they documented the agreed-upon limitation in Jane's medical records.A few weeks later, Jane's employer, Big Corp, required her to submit medical information to their insurance provider, Insurance Co, to process her insurance claim. Jane provided written consent to Dr. Johnson's office to disclose her general medical information to Insurance Co, but reiterated the restriction on her procedure-related PHI.Dr. Johnson's office sent the necessary information to Insurance Co. However, due to a clerical error, the PHI related to Jane's procedure was included in the disclosure. Insurance Co, upon receiving the information, shared it with Big Corp as part of the claim process.Upon learning about the disclosure, Jane was upset and felt that her privacy had been violated. She decided to seek legal advice regarding the breach of the agreed-upon restriction on her PHI. \n ### Response: HIPAA Privacy Rule forbids the case. \n ### Input: "
        
    elif type == "applicability":
        few_shot = "Read the case: Jane is a factory worker at XYZ Manufacturing, a company that requires its employees to undergo annual medical check-ups due to the nature of their work. The company asks Dr. Smith, a covered health care provider, to perform these check-ups, including medical surveillance of the workplace to ensure a safe environment. During Jane's check-up, Dr. Smith discovers that Jane has a medical condition that could be related to her workplace environment. Dr. Smith believes that it is important to inform XYZ Manufacturing about Jane's condition to address any potential workplace hazards. Dr. Smith contacts the HR department of XYZ Manufacturing and shares Jane's medical information related to the discovered condition. The HR department acknowledges the information and initiates an internal investigation to ensure the safety of their employees. Jane is informed about the disclosure of her medical information to her employer by Dr. Smith and agrees to the disclosure for the purpose of workplace safety evaluation.\n ### Response: HIPAA Privacy Rule is applicable to the case. \n ### Input: Read the case: A state trooper was approached during a call by a bystander, who informed the officer that her estranged spouse had a significant amount of marijuana at his residence. A local sheriff, also present at the scene, corroborated that he had heard similar rumors about the spouse and had previously received information from the bystander when working undercover. Acting on this information, the trooper prepared an affidavit and obtained a search warrant from a trial commissioner. During the execution of the warrant, the officers discovered marijuana and drug paraphernalia in the spouse's home. The spouse contested the legality of the search, arguing that the affidavit was insufficient for the issuance of the warrant due to a lack of specific details about the contraband, inaccuracies in the property owner's name, and insufficient evidence of the informant's reliability. However, the trial court determined that the officer's testimony met the standards required for a good faith exception to the warrant requirement, and thus denied the motion to suppress the evidence, which is the subject of the current appeal. \n ### Response: HIPAA Privacy Rule is not applicable to the case. \n ### Input: "
    instruction["input"] = few_shot + instruction["input"]
    return instruction["input"]

def generate_compliance_prompt(args, row):
    # CoT
    if args.cot:
        instruction = build_instruction_compliance(row, "cot", True)
    # Direct
    elif args.direct:
        instruction = build_instruction_compliance(row, "direct", True)
        if args.few:
            instruction['input'] = add_few_shot_instruction(instruction, "compliance")
    # output = instruction["response"]
    return instruction

def generate_applicability_prompt(args, row):
    # CoT
    if args.cot:
        instruction = build_instruction_applicability(row, "cot", True)
    # Direct
    elif args.direct:
        instruction = build_instruction_applicability(row, "direct", True)
        if args.few:
            instruction['input'] = add_few_shot_instruction(instruction, "applicability")
    
    return instruction

def eval_llm(args):
    print(args.input_file)
    print(args.output_file)
    print(args.lora_checkpoint)
    print(args.model)
    print(args.b13)
    # if output file not exist, create it
    with open(args.output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["regulation id", "compliance", "response"])
    
    if args.model == "llama":
        model = Llama(load_in_8bit=args.load_in_8bit, chat=args.chat, b13=args.b13)
    elif args.model == "mistral":
        model = Mistral(load_in_8bit=args.load_in_8bit, chat=args.chat)
    elif args.model == "mpt":
        model = MPT(load_in_8bit=args.load_in_8bit, chat=args.chat)

    if args.lora_checkpoint != None:
        model.model = PeftModel.from_pretrained(model.model, args.lora_checkpoint)

    df = pd.read_csv(args.input_file)
    # df = df.sample(frac=1)
    start = 0
    for i in tqdm(range(start, len(df))):
        row = df.iloc[i]
        print(row)
        if args.compliance:
            instruction = generate_compliance_prompt(args, row)
        elif args.applicability:
            instruction = generate_applicability_prompt(args, row)
        prompt = generate_prompt(instruction["instruction"], instruction["input"])

        print("\nPrompt:", prompt)
        resp = model.interact(message=prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print("\nResp:", resp)
        if resp != None and len(resp) > 0:
            with open(args.output_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([row["refer_regulation"], row["generate_HIPAA_type"], resp])
        else:
            print("Error: ", row["generate_regulation_ids"], row["generate_HIPAA_type"])
            print("Instruction: ", prompt)
            print("Response: ", resp)
            with open(args.output_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([row["generate_regulation_ids"], row["generate_HIPAA_type"], "Error"])

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model = "mistral"
    parser.add_argument("-model", type=str, default="llama")
    parser.add_argument("-load_in_8bit", action='store_true', default=False)
    parser.add_argument("-chat", action='store_true', default=False)
    parser.add_argument("-b13", action='store_true', default=False)
    parser.add_argument("-lora_checkpoint", type=str, default=None)
    parser.add_argument("-max_new_tokens", type=int, default=1024)
    parser.add_argument("-temperature", type=float, default=0.9)
    parser.add_argument("-real", action='store_true', default=False)
    parser.add_argument("-generate", action='store_true', default=False)
    
    parser.add_argument("-alpaca", action='store_true', default=False)
    parser.add_argument("-few", action='store_true', default=False)
    parser.add_argument("-regulation", action='store_true', default=False, help="train on regulation, not cases")
    
    parser.add_argument("-direct", action='store_true', default=False)
    parser.add_argument("-cot", action='store_true', default=False)
    
    parser.add_argument("-compliance", action='store_true', default=False)
    parser.add_argument("-applicability", action='store_true', default=False)
    
    
    param = ""
    args = parser.parse_args()
    param += "_13b" if args.b13 else ""
    param += "_8bit" if args.load_in_8bit else ""
    param += "_chat" if args.chat else ""

    lora_checkpoint_lower = (args.lora_checkpoint or "").lower()
    if lora_checkpoint_lower:
        param += "_lora"
        checkpoints = ["alpaca", "regulation", "compliance", "applicability", "cot", "direct"]
        # Add checkpoint-related suffixes
        for checkpoint in checkpoints:
            if checkpoint in lora_checkpoint_lower:
                param += f"_{checkpoint}"
                if checkpoint in ["cot", "direct"]:
                    break  # Avoid multiple suffixes if 'cot' or 'direct' is present

    param += "_prompt"
    param += "_compliance" if args.compliance else ""
    param += "_applicability" if args.applicability else ""
    param += "_cot" if args.cot else ""
    param += "_direct" if args.direct else ""
    param += "_few" if args.few else ""
    # Determine the case type
    case_type = "real" if args.real else "generate"

    # Create the timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Add arguments with formatted default values
    task_type = "compliance" if args.compliance else "applicability"
    parser.add_argument("--input_file", type=str, default=f"../cases/train_val_test/test_{case_type}_cases_hipaa_{task_type}.csv")
    parser.add_argument("--output_file", type=str, default=f"results/{timestamp}_eval_{case_type}_cases_test_{args.model}{param}.csv")
    
    args = parser.parse_args()
    if not os.path.exists("results"):
        os.makedirs("results")
    eval_llm(args)
    parse_eval_result(eval_file=args.output_file)