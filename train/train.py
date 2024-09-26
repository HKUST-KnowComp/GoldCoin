import sys
sys.path.append("../")
import argparse
import copy
from datasets import load_dataset, concatenate_datasets
from eval.build_instruction_compliance import build_instruction_compliance
from eval.build_instruction_applicability import build_instruction_applicability
from models.llama import Llama
from models.mistral import Mistral
from models.mpt import MPT
import transformers
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModel

def generate_prompt(instruction: str, input: str = None) -> str:
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction: {instruction}\n### Input: {input}\n### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction: {instruction}\n### Response:"""


def generate_compliance_prompt(args, row):
    # CoT
    if args.cot:
        instruction = build_instruction_compliance(row, "cot", False)
    # Direct
    elif args.direct:
        instruction = build_instruction_compliance(row, "direct", False)
    prompt = generate_prompt(instruction["instruction"], instruction["input"])
    output = instruction["response"]
    return prompt, output

def generate_applicability_prompt(args, row):
    # CoT
    if args.cot:
        instruction = build_instruction_applicability(row, "cot", False)
    # Direct
    elif args.direct:
        instruction = build_instruction_applicability(row, "direct", False)
    prompt = generate_prompt(instruction["instruction"], instruction["input"])
    output = instruction["response"]
    return prompt, output

def generate_and_tokenize_prompt(args, tokenizer, row, type):
    max_length = 1300
    IGNORE_INDEX = -100
    if type == "alpaca":
        if row["input"] == "":
            prompt = generate_prompt(row["instruction"])
        else:
            prompt = generate_prompt(row["instruction"], row["input"])
        output = row["output"]
    elif type == "regulation":
        prompt = generate_prompt("Please recount the contents of Section {} of the HIPAA Privacy Rule.".format(row["regulation_id"]))
        output = row["text"]
    elif type == "compliance":
        prompt, output = generate_compliance_prompt(args, row)
        print(prompt)
        print(output)
    elif type == "applicability":
        prompt, output = generate_applicability_prompt(args, row)
        print(prompt)
        print(output)

    if args.chat:
        if args.model == "llama" or args.model == "mistral":
            prompt = "[INST] " + prompt + " [/INST] "
            output = output
        elif args.model == "llama3":
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""
            output = output
        elif args.model == "mpt":
            prompt = f"""<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"""
            output = output

    result = tokenizer(
        prompt + output,
        truncation=True,
        max_length=max_length,
        padding="max_length")
    prompt = tokenizer(prompt)
    output = tokenizer(output)
    prompt_len = len(prompt["input_ids"])
    output_len = len(output["input_ids"])
    # print(prompt_len+output_len)

    result["labels"] = copy.deepcopy(result["input_ids"])
    if prompt_len < max_length:
        result["labels"] = [IGNORE_INDEX] * prompt_len + result["labels"][prompt_len:]
    else:
        result["labels"] = [IGNORE_INDEX] * max_length
    return result

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def prepare_output_dir(args):
    parts = [
        "./lora",
        args.model,
        "13b" if args.b13 else "",
        "8bit" if args.load_in_8bit else "",
        "chat" if args.chat else "",
        f"bs_{args.batch_size}",
        f"mbs_{args.micro_batch_size}",
        f"lr_{args.learning_rate}",
        f"epo_{args.num_epochs}",
        f"r_{args.rank}",
        "alpaca" if (args.alpaca or args.alpaca_checkpoint is not None) else "",
        "regulation" if args.regulation else "",
        "compliance" if args.compliance else "",
        "applicability" if args.applicability else "",
        "cot" if args.cot else "direct" if args.direct else ""
    ]

    # Filter out empty strings and join the parts with underscore
    output_dir = "_".join(filter(None, parts))

    return output_dir

def main(args):
    output_dir = prepare_output_dir(args)
    print(f"Output Dir: {output_dir}")
    
    if args.model == "llama":
        MODEL = Llama(load_in_8bit=args.load_in_8bit, chat=args.chat, b13=args.b13)
        MODEL.tokenizer.padding_side = "left"
    elif args.model == "mistral":
        MODEL = Mistral(load_in_8bit=args.load_in_8bit, chat=args.chat)
    elif args.model == "mpt":
        MODEL = MPT(load_in_8bit=args.load_in_8bit, chat=args.chat)

    if args.alpaca_checkpoint != None:
        MODEL.model = PeftModel.from_pretrained(MODEL.model, args.alpaca_checkpoint)
        print("Loaded LORA model from", args.alpaca_checkpoint)

    print("padding side:", MODEL.tokenizer.padding_side)
    print("pad token:", MODEL.tokenizer.pad_token)
    
    final_train_dataset = []
    final_val_dataset = []
    # Load the alpaca dataset
    if args.alpaca:
        type = "alpaca"
        alpaca_dataset = load_dataset('json', data_files='alpaca_data_cleaned_archive.json')
        alpaca_dataset = alpaca_dataset['train'].train_test_split(test_size=2000, seed=42)
        final_train_dataset = alpaca_dataset["train"].map(lambda row: generate_and_tokenize_prompt(args, MODEL.tokenizer, row, type))
        final_val_dataset = alpaca_dataset["test"].map(lambda row: generate_and_tokenize_prompt(args, MODEL.tokenizer, row, type))

    if args.regulation:
        type = "regulation"
        regulation_dataset = load_dataset('json', data_files='../regulations/regulation_dataset.json')
        tokenized_regulation_train_dataset = regulation_dataset['train'].map(lambda row: generate_and_tokenize_prompt(args, MODEL.tokenizer, row, type))
        if args.alpaca:
            final_train_dataset = concatenate_datasets([final_train_dataset, tokenized_regulation_train_dataset])
        else:
            final_train_dataset = tokenized_regulation_train_dataset

    if (args.direct or args.cot) and args.compliance:
        type = "compliance"
        compliance_dataset = load_dataset('csv', data_files='../cases/train_val_test/train_generate_cases_hipaa_compliance.csv', split='train')
        tokenized_compliance_train_dataset = compliance_dataset.map(lambda row: generate_and_tokenize_prompt(args, MODEL.tokenizer, row, type))
        if args.alpaca or args.regulation:
            final_train_dataset = concatenate_datasets([final_train_dataset, tokenized_compliance_train_dataset])
        else:
            final_train_dataset = tokenized_compliance_train_dataset
    
    if (args.direct or args.cot) and args.applicability:
        type = "applicability"
        applicability_dataset = load_dataset('csv', data_files='../cases/train_val_test/train_generate_cases_hipaa_applicability.csv', split='train')
        tokenized_applicability_train_dataset = applicability_dataset.map(lambda row: generate_and_tokenize_prompt(args, MODEL.tokenizer, row, type))
        if args.alpaca or args.regulation or args.compliance:
            final_train_dataset = concatenate_datasets([final_train_dataset, tokenized_applicability_train_dataset])
        else:
            final_train_dataset = tokenized_applicability_train_dataset

    final_train_dataset = final_train_dataset.shuffle(seed=42)
    final_val_dataset = final_val_dataset.shuffle(seed=42) if len(final_val_dataset) > 0 else final_val_dataset

    MODEL.model.gradient_checkpointing_enable()
    MODEL.model = prepare_model_for_kbit_training(MODEL.model)
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank*2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )
    MODEL.model = get_peft_model(MODEL.model, config)
    print_trainable_parameters(MODEL.model)
    
    tokenizer = MODEL.tokenizer
    train_data = final_train_dataset
    val_data = final_val_dataset
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    num_epochs = args.num_epochs
    gradient_accumulation_steps = batch_size // micro_batch_size
    learning_rate = args.learning_rate
    val_set_size = len(val_data)

    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size}")
    print(f"Train set size: {len(train_data)}")
    print(f"Validation set size: {val_set_size}")
    

    trainer = transformers.Trainer(
        model=MODEL.model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=args.logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=args.eval_steps if val_set_size > 0 else None,
            save_steps=args.save_steps,
            output_dir=output_dir,
            save_total_limit=1,
            # load_best_model_at_end=True if val_set_size > 0 else False,
            # ddp_find_unused_parameters=False if ddp else None,
            # group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt"
        ),
    )
    MODEL.model.config.use_cache = False

    trainer.train()

    MODEL.model.save_pretrained(output_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="llama")
    parser.add_argument("-alpaca_checkpoint", type=str, default=None)
    parser.add_argument("-load_in_8bit", action='store_true', default=False)
    parser.add_argument("-chat", action='store_true', default=False)
    # parser.add_argument("-batch_size", type=int, default=128)
    # parser.add_argument("-micro_batch_size", type=int, default=8)
    # parser.add_argument("-rank", type=int, default=32)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-micro_batch_size", type=int, default=1)
    parser.add_argument("-rank", type=int, default=8)
    parser.add_argument("-num_epochs", type=int, default=3)
    parser.add_argument("-learning_rate", type=float, default=1e-5)
    parser.add_argument("-save_steps", type=int, default=200)
    parser.add_argument("-eval_steps", type=int, default=100)
    parser.add_argument("-warmup_steps", type=int, default=10)
    parser.add_argument("-logging_steps", type=int, default=10)
    parser.add_argument("-b13", action='store_true', default=False)
    
    parser.add_argument("-alpaca", action='store_true', default=False)
    parser.add_argument("-regulation", action='store_true', default=False, help="train on regulation, not cases")
    
    parser.add_argument("-direct", action='store_true', default=False)
    parser.add_argument("-cot", action='store_true', default=False)
    
    parser.add_argument("-compliance", action='store_true', default=False)
    parser.add_argument("-applicability", action='store_true', default=False)
    
    args = parser.parse_args()
    main(args)
    # print(torch.cuda.device_count() > 1)