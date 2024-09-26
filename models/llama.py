import os
import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig
import argparse

class Llama(object):
    def __init__(self, load_in_8bit=False, chat=False, b13=False):
        self.chat = chat
        self.load_in_8bit = load_in_8bit
        self.b13 = b13
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.load_model()
        
    def load_model(self):
        # quantization_config = BitsAndBytesConfig(load_in_8bit=self.load_in_8bit)
        if self.chat == True:
            model_name = "meta-llama/Llama-2-7b-chat-hf"
            if self.b13 == True:
                model_name = "meta-llama/Llama-2-13b-chat-hf"
            self.model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=self.load_in_8bit,
                # torch_dtype=torch.float16 if self.load_in_8bit else torch.float32,
                # low_cpu_mem_usage=True
                # device_map = "auto" if torch.cuda.is_available() else {"": self.device},
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            
        elif self.chat == False:
            model_name = "meta-llama/Llama-2-7b-hf"
            if self.b13 == True:
                model_name = "meta-llama/Llama-2-13b-hf"
            self.model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=self.load_in_8bit,
                # torch_dtype=torch.float16 if self.load_in_8bit else torch.float32,
                # low_cpu_mem_usage=True
                # device_map = "auto" if torch.cuda.is_available() else {"": self.device},
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

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
        else:
            encodeds = self.tokenizer(message, return_tensors="pt")
            input_ids = encodeds['input_ids'].to(self.device)
            # input_ids = encodeds.to(self.device)

        with torch.no_grad():
            # print("length of input_ids: ", len(input_ids[0]))
            generated_ids = self.model.generate(input_ids=input_ids,
                                                max_new_tokens=max_new_tokens,
                                                repetition_penalty=1.1,
                                                temperature=temperature,
                                                do_sample=True
                                                )
        # print(generated_ids)
        # print("length of generated_ids: ", len(generated_ids[0]) - len(input_ids[0]))
        # If skip_special_tokens=False, the output will include <s> and </s>
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print(decoded)
        response = self.postprocess_response(decoded, message)
        # print(response)
        return response

    # def batch_interact(self, batch_messages,
    #                    max_new_tokens=256,
    #                    temperature=0.7):
    #     batch_messages = [self.preprocess_input(message) for message in batch_messages]
    #     batch_encodeds = self.tokenizer(batch_messages, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
    #     batch_input_ids = batch_encodeds['input_ids'].to(self.device)
    #     batch_attention_mask = batch_encodeds['attention_mask'].to(self.device)
    #     with torch.no_grad():
    #         batch_generated_ids = self.model.generate(batch_input_ids,
    #                                                   attention_mask=batch_attention_mask,
    #                                                   max_new_tokens=max_new_tokens,
    #                                                   repetition_penalty=1.1,
    #                                                   temperature=temperature,
    #                                                   do_sample=True)
    #     batch_decoded = self.tokenizer.batch_decode(batch_generated_ids, skip_special_tokens=True)
    #     batch_responses = [self.postprocess_response(d) for d in batch_decoded]
    #     return batch_responses
    
    def calculate_next_word_logprobs(self, input_text, word='Beijing'):
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt")['input_ids'].to(self.device)

            outputs = self.model(input_ids)

            logits = outputs.logits

            log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)

            # topk_log_probs, topk_indices = torch.topk(log_probs, k=5, dim=-1)  # 获取概率最高的5个单词
            # for i, (log_prob, idx) in enumerate(zip(topk_log_probs[0], topk_indices[0])):
            #     word = self.tokenizer.decode([idx])
            #     print(f"Top {i+1} word: {word}, log_prob: {log_prob.item()}")
            word_id = self.tokenizer.encode(word, add_special_tokens=False)[0]

            word_logprob = log_probs[:, word_id]

        return word_logprob.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-load_in_8bit", action='store_true', default=False)
    parser.add_argument("-chat", action='store_true', default=True)
    parser.add_argument("-b13", action='store_true', default=False)
    args = parser.parse_args()
    llama = Llama(load_in_8bit=args.load_in_8bit, chat=args.chat, b13=args.b13)
    # while True:
    #     message = input("User: ")
    #     if message == "exit":
    #         break
    #     print("Llama: ", llama.interact(message, max_new_tokens=256, temperature=0.7))
    import time
    while True:
        print("Sleeping...")
        time.sleep(10)

