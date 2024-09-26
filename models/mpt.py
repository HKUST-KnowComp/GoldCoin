import os
import sys
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import argparse

class MPT(object):
    def __init__(self, load_in_8bit=False, chat=False,):
        self.chat = chat
        self.load_in_8bit = load_in_8bit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.load_model()
        
    def load_model(self):
        # quantization_config = BitsAndBytesConfig(load_in_8bit=self.load_in_8bit)
        if self.chat == True:
            self.model = AutoModelForCausalLM.from_pretrained(
                "mosaicml/mpt-7b-8k-chat",
                load_in_8bit=self.load_in_8bit,
                # torch_dtype=torch.float16 if self.load_in_8bit else torch.float32,
                # low_cpu_mem_usage=True
                # device_map = "auto" if torch.cuda.is_available() else {"": self.device},
            )
            self.tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-8k-chat")

        elif self.chat == False:
            self.model = AutoModelForCausalLM.from_pretrained(
                "mosaicml/mpt-7b-8k",
                load_in_8bit=self.load_in_8bit,
                # torch_dtype=torch.float16 if self.load_in_8bit else torch.float32,
                # low_cpu_mem_usage=True
                # device_map = "auto" if torch.cuda.is_available() else {"": self.device},
            )
            self.tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-8k")
        
        if not self.load_in_8bit:
            self.model = self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        print(self.model)
        
    def preprocess_input(self, message):
        if self.chat == True:
            message = f"""<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"""
        else:
            message = message
        return message

    def postprocess_response(self, decoded, message):
        # print("****")
        # print(decoded)
        # print("****")
        if self.chat == True:
            if message in decoded:
                response = decoded.split(message)[1].replace("assistant", "", 1).strip(" ").strip("\n")
            else:
                response = decoded.split("assistant\n")[1].strip(" ").strip("\n")
        else:
            response = decoded.replace(message, "", 1).strip(" ").strip("\n")
        return response

    def interact(self, message,
                 max_new_tokens=512,
                 temperature=0.7):
        raw_message = message
        message = self.preprocess_input(message)
        encodeds = self.tokenizer(message, return_tensors="pt")
        input_ids = encodeds['input_ids'].to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids,
                                                max_new_tokens=max_new_tokens,
                                                repetition_penalty=1.1,
                                                temperature=temperature,
                                                do_sample=True,
                                                pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = self.postprocess_response(decoded, raw_message)
        return response
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-load_in_8bit", action='store_true', default=False)
    parser.add_argument("-chat", action='store_true', default=False)
    args = parser.parse_args()
    mpt = MPT(load_in_8bit=args.load_in_8bit, chat=args.chat)
    while True:
        message = input("User:")
        if message == "exit":
            break
        print("MPT:", mpt.interact(message, max_new_tokens=512, temperature=0.7))

