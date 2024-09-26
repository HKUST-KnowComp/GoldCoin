import pandas as pd
import sys
import ast
sys.path.append("../")

direct_instruction_template = {
    "instruction": "Please determine whether the HIPAA Privacy Rule permits or forbids the case.",
    "input": "Read the case: {case}",
    "response": "{response}"
}

cot_instruction_template = {
    "instruction": "Please assess the case for compliance with the HIPAA Privacy Rule through the following steps: Step 1: Annotate the eleven message characteristics [Sender, Sender Role, Recipient, Recipient Role, About, About Role, Type, Purpose, In Reply To, Consented By, Belief] about the flow of private information in the case as a list. Step 2: Identify and list all applicable HIPAA regulation IDs (e.g., 164.XXX) and their content. Step 3: Determine whether the HIPAA Privacy Rule permits or forbids the case.",
    "input": "Read the case: {case}",
    "response": "Step 1: {step1} Step 2: {step2} Step 3: {step3}"
}

def build_instruction_compliance(row, mode, eval):
    """
    mode: cot, direct
    """
    if mode == "direct":
        direct_instruction = direct_instruction_template.copy()
        case = row["generate_background"]
        direct_instruction["input"] = direct_instruction["input"].replace("{case}", case)
        if not eval:
            response = "HIPAA Privacy Rule permits the case." if row["generate_HIPAA_type"] == "Permit" else "HIPAA Privacy Rule forbids the case."
            direct_instruction["response"] = direct_instruction["response"].replace("{response}", response)
        return direct_instruction
    elif mode == "cot":
        cot_instruction = cot_instruction_template.copy()
        case = row["generate_background"]
        cot_instruction["input"] = cot_instruction["input"].replace("{case}", case)
        if not eval:
            if type(row["generate_characteristics"]) == str:
                row["generate_characteristics"] = ast.literal_eval(row["generate_characteristics"])
            step1 = ""
            for k, v in row["generate_characteristics"].items():
                v = "None" if v == "" or v == None else v
                step1 += k + ": " + v + ", "

            row["refer_regulation"] = row["refer_regulation"].replace("HIPAA:HIPAA Privacy Rule", "")
            row["refer_regulation"] = row["refer_regulation"].replace("Part164:PART 164—SECURITY AND PRIVACY", "")
            row["refer_regulation"] = row["refer_regulation"].replace("Part164SubpartE:Subpart E—Privacy of Individually Identifiable Health Information", "")
            step2 = "The related HIPAA regulation is " + row["refer_regulation"].strip()
            step3 = "HIPAA Privacy Rule permits the case." if row["generate_HIPAA_type"] == "Permit" else "HIPAA Privacy Rule forbids the case."
            cot_instruction["response"] = cot_instruction["response"].replace("{step1}", step1).replace("{step2}", step2).replace("{step3}", step3)
        return cot_instruction