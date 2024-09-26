import sys
sys.path.append("../")
import pandas as pd
import argparse
import random
from utils import set_random_seed
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

def clean_response(response):
    response = response.replace('"', "")
    response = response.replace("determine whether the hipaa privacy rule permits or forbids the case:", "")
    response = response.replace("permit, forbid, or not applicable", "")
    response = response.replace("permitted, forbidden, or not applicable", "")
    response = response.replace("permit, forbid, not applicable", "")
    response = response.replace("permit/forbid/not applicable", "")
    response = response.replace("permitted or forbidden", "")
    response = response.replace("permits or forbids", "")
    response = response.replace("permit or forbid", "")
    response = response.replace("determination of permissibility", "")
    response = response.replace("determine whether the hipaa privacy rule is applicable to the case", "")
    response = response.replace("whether the hipaa privacy rule is applicable", "")
    response = response.replace("applicability", "")
    return response

def extract_step_result(response):
    if "step 3" in response:
        step3_index = response.find("step 3")
        response = response[step3_index:]
    elif "step 2" in response:
        step2_index = response.find("step 2")
        response = response[step2_index:]
    return response

def compliance_result(compliance, response):
    result = ""
    if compliance == "permit":
        if any(word in response for word in ["permit", "permis", "complies with", "not violat", "allow", "not explicitly prohibit", "not forbid"]) and "not comply" not in response:
            result = "permit"
    if compliance == "forbid":
        if any(word in response for word in ["forbid", "prohibit", "not comply", "not fully comply", "violat"]):    
            result = "forbid"
    return result

def first_compliance_result(response):
    result = ""
    labels = ["permit", "permis", "complies with", "not violat", "allow", "not explicitly prohibit", "not forbid",
              "forbid", "not permit", "prohibit", "not comply", "not fully comply", "violat"]
    first_label_index = len(response)
    first_label = ""
    for label in labels:
        if label in response:
            label_index = response.index(label)
            if label_index < first_label_index:
                first_label_index = label_index
                first_label = label

    if first_label in ["permit", "permis", "complies with", "not violat", "allow", "not explicitly prohibit", "not forbid"]:
        result = "permit"
    elif first_label in ["forbid", "not permit", "prohibit", "not comply", "not fully comply", "violat"]:
        result = "forbid"
    return result
        
def get_compliance_result(response, compliance):
    result = ""
    response = extract_step_result(response)
    result = first_compliance_result(response)
    if result == "":
        result = compliance_result(compliance, response)
    if result == "":
        gt = ["permit", "forbid"]
        # print(compliance)
        gt.remove(compliance)
        result = random.choice(gt)

    # print("---------", compliance)
    # print("+++++++++", result)
    # print(">>>>>>>>>", response)

    return result

def applicability_result(applicability, response):
    result = ""
    if applicability == "applicable":
        if any(word in response for word in ["applicable", "apply to", "applies to"]):
            result = "applicable"
    if applicability == "not applicable":
        if any(word in response for word in ["not applicable", "not apply"]):
            result = "not applicable"
    return result

def first_applicability_result(response):
    result = ""
    labels = ["applicable", "apply to", "applies to", " not "]
    first_label_index = len(response)
    first_label = ""
    for label in labels:
        if label in response:
            label_index = response.index(label)
            if label_index < first_label_index:
                first_label_index = label_index
                first_label = label

    if first_label in ["applicable", "apply to", "applies to"]:
        result = "applicable"
    elif first_label in [" not "]:
        result = "not applicable"
    return result

def get_applicability_result(response, applicability):
    result = ""
    response = extract_step_result(response)
    result = first_applicability_result(response)
    if result == "":
        result = applicability_result(applicability, response)
    if result == "":
        gt = ["not applicable", "applicable"]
        # print(applicability)
        gt.remove(applicability)
        result = random.choice(gt)

    # print("---------", applicability)
    # print("+++++++++", result)
    # print(">>>>>>>>>", response)

    return result

def parse_eval_result(eval_file):
    # eval_file  = "results/" + eval_file.split("results/")[1] if "eval/" in eval_file else args.eval_file
    df = pd.read_csv(eval_file)
    predictions = []
    true_labels = []
    total = len(df)
    
    for i in range(len(df)):
        row = df.iloc[i]
        compliance = row[1].lower()
        if "compliance" not in eval_file:
            if "permit" in compliance or "forbid" in compliance:
                compliance = "applicable"
        response = row[2].lower()
        response = clean_response(response)
        if "compliance" in eval_file:
            step3_result = get_compliance_result(response, compliance)
        elif "applicability" in eval_file:
            step3_result = get_applicability_result(response, compliance)

        predictions.append(step3_result)
        true_labels.append(compliance)

    # print(predictions)
    # print(true_labels)

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    class_report = classification_report(true_labels, predictions, digits=4)
    print(class_report)
    # Get labels to ensure consistent ordering
    labels = sorted(list(set(true_labels + predictions)))

    # Create a confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    
    # Print each class's total and correct predictions
    print("Class Metrics:")
    for i, label in enumerate(labels):
        correct = cm[i, i]
        total = sum(cm[i, :])
        print(f"Class '{label}': Total = {total}, Correct = {correct}, Error = {total - correct}")
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(true_labels, predictions)
    print(f"Macro-F1 Score: {macro_f1}")
    print(f"Overall Accuracy: {overall_accuracy}")
    
def parse_eval_result_file(eval_file):
    eval_file = "../eval/results/" + eval_file.split("results/")[1] if "eval/" in eval_file else eval_file
    df = pd.read_csv(eval_file)
    predictions = []
    true_labels = []
    total = len(df)
    # print(total)
    
    for i in range(len(df)):
        row = df.iloc[i]
        compliance = row[1].lower()
        if "compliance" not in eval_file:
            if "permit" in compliance or "forbid" in compliance:
                compliance = "applicable"
        response = row[2].lower()
        response = clean_response(response)
        if "compliance" in eval_file:
            step3_result = get_compliance_result(response, compliance)
        elif "applicability" in eval_file:
            step3_result = get_applicability_result(response, compliance)

        predictions.append(step3_result)
        true_labels.append(compliance)

    # print(predictions)
    # print(true_labels)

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    class_report = classification_report(true_labels, predictions, digits=4)
    print(f"Macro-F1 Score: {macro_f1}")
    print(f"Accuracy: {accuracy}")
    print(class_report)

    acc_list = []
    permit_num = 0
    forbid_num = 0
    print("Total:", len(predictions))
    print(predictions)
    print(true_labels)
    for i in range(len(predictions)):
        if predictions[i] == "permit" and true_labels[i] == "permit":
            permit_num += 1
        if predictions[i] == "forbid" and true_labels[i] == "forbid":
            forbid_num += 1
        if predictions[i] == true_labels[i]:
            acc_list.append(1)
        else:
            acc_list.append(0)
    return acc_list, permit_num, forbid_num
    #     # Get labels to ensure consistent ordering
    # labels = sorted(list(set(true_labels + predictions)))

    # # Create a confusion matrix
    # cm = confusion_matrix(true_labels, predictions, labels=labels)
    
    # # Print each class's total and correct predictions
    # print("Class Metrics:")
    # for i, label in enumerate(labels):
    #     correct = cm[i, i]
    #     total = sum(cm[i, :])
    #     print(f"Class '{label}': Total = {total}, Correct = {correct}, Error = {total - correct}")
    
    # # Calculate overall accuracy
    # overall_accuracy = accuracy_score(true_labels, predictions)
    # print(f"Overall Accuracy: {overall_accuracy}")
    
    
if __name__ == "__main__":
    set_random_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("-eval_file", type=str, default="")
    args = parser.parse_args()
    
    # args.eval_file = "results/" + args.eval_file.split("results/")[1] if "eval/" in args.eval_file else args.eval_file
    print("Start parsing eval result...")
    print(args.eval_file)
    parse_eval_result(eval_file=args.eval_file)