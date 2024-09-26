import pandas as pd
import sys
import ast
sys.path.append("../")

direct_instruction_template = {
    "instruction": "Please determine whether the HIPAA Privacy Rule is applicable to the case.",
    "input": "Read the case: {case}",
    "response": "{response}"
}

cot_instruction_template = {
    "instruction": "Please assess the applicability of the HIPAA Privacy Rule to the case through the following steps: Step 1: Annotate the message characteristics [Sender, Sender Role, Recipient, Recipient Role, About, About Role, Type] about the flow of private information in the case as a list. Step 2: Determine whether the HIPAA Privacy Rule is applicable to the case.",
    "input": "Read the case: {case}",
    "response": "Step 1: {step1} Step 2: {step2}"
}

def build_instruction_applicability(row, mode, eval):
    """
    mode: cot, direct
    """
    if mode == "direct":
        direct_instruction = direct_instruction_template.copy()
        case = row["generate_background"]
        direct_instruction["input"] = direct_instruction["input"].replace("{case}", case)
        if not eval:
            response = "HIPAA Privacy Rule is not applicable to the case." if row["generate_HIPAA_type"] == "Not Applicable" else "HIPAA Privacy Rule is applicable to the case."
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
                if k in ["Sender", "Sender Role", "Recipient", "Recipient Role", "About", "About Role", "Type"]:
                    v = "None" if v == "" or v == None else v
                    step1 += k + ": " + v + ", "
            step2 = "HIPAA Privacy Rule is not applicable to the case." if row["generate_HIPAA_type"] == "Not Applicable" else "HIPAA Privacy Rule is applicable to the case."
            cot_instruction["response"] = cot_instruction["response"].replace("{step1}", step1).replace("{step2}", step2)
        return cot_instruction