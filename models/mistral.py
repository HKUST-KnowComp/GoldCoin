import os
import sys
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import argparse

class Mistral(object):
    def __init__(self, load_in_8bit=False, chat=False):
        self.chat = chat
        self.load_in_8bit = load_in_8bit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.load_model()
        
    def load_model(self):
        # quantization_config = BitsAndBytesConfig(load_in_8bit=self.load_in_8bit)
        if self.chat == True:
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                load_in_8bit=self.load_in_8bit,
                # torch_dtype=torch.float16 if self.load_in_8bit else torch.float32,
                # low_cpu_mem_usage=True
                # device_map = "auto" if torch.cuda.is_available() else {"": self.device},
            )
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

        elif self.chat == False:
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                load_in_8bit=self.load_in_8bit,
                # torch_dtype=torch.float16 if self.load_in_8bit else torch.float32,
                # low_cpu_mem_usage=True
                # device_map = "auto" if torch.cuda.is_available() else {"": self.device},
            )
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        
        if not self.load_in_8bit:
            self.model = self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # print(self.model)
        
    def preprocess_input(self, message):
        if self.chat == True:
            message = [
                {"role": "user", "content": message}
            ]
        else:
            message = message
        return message

    def postprocess_response(self, decoded, message):
        # print(decoded)
        if self.chat == True:
            response = decoded.split("[/INST]")[1].replace("</s>", "").strip(" ").strip("\n")
        else:
            response = decoded.replace(message, "", 1).strip(" ").strip("\n")
        return response

    def interact(self, message,
                 max_new_tokens=512,
                 temperature=0.7):
        
        message = self.preprocess_input(message)
        if self.chat == True:
            encodeds = self.tokenizer.apply_chat_template(message, return_tensors="pt")
            input_ids = encodeds.to(self.device)
            attention_mask = torch.ones_like(input_ids)
        else:
            encodeds = self.tokenizer(message, return_tensors="pt")
            input_ids = encodeds['input_ids'].to(self.device)
            attention_mask = encodeds['attention_mask'].to(self.device)
        with torch.no_grad():
            # print("length of input_ids: ", len(input_ids[0]))
            generated_ids = self.model.generate(input_ids,
                                                attention_mask=attention_mask,
                                                max_new_tokens=max_new_tokens,
                                                repetition_penalty=1.1,
                                                temperature=temperature,
                                                do_sample=True,
                                                pad_token_id=self.tokenizer.eos_token_id)
        # print(generated_ids)
        # print("length of generated_ids: ", len(generated_ids[0])-len(input_ids[0]))
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = self.postprocess_response(decoded, message)
        return response
    
    def calculate_next_word_logprobs(self, input_text, word='Beijing'):
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt")['input_ids'].to(self.device)

            outputs = self.model(input_ids)
            logits = outputs.logits  # 这里假设您的模型输出包含 logits

            log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)

            word_id = self.tokenizer.encode(word, add_special_tokens=False)[0]

            word_logprob = log_probs[:, word_id]

        return word_logprob.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-load_in_8bit", action='store_true', default=False)
    parser.add_argument("-chat", action='store_true', default=False)
    args = parser.parse_args()
    mistral = Mistral(load_in_8bit=args.load_in_8bit, chat=args.chat)
    while True:
        message = input("User: ")
        if message == "exit":
            break
        print("Mistral: ", mistral.interact(message, max_new_tokens=256, temperature=0.7))
