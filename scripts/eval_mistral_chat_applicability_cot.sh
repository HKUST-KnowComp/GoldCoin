cd ../eval

python eval_llm.py -model mistral -lora_checkpoint "../train/lora_mistral_chat_bs_1_mbs_1_lr_1e-05_epo_3_r_8_applicability_cot" -real -cot -applicability -chat -max_new_tokens 1024 -temperature 0.7