import os
import yaml
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

def format_prompt(row):
    """
    Định dạng dữ liệu thành prompt cho bài toán phân loại ý định (Sequence Classification).
    Chuyển đổi văn bản và nhãn thành một chuỗi duy nhất để LLM học cách sinh ra nhãn.
    """
    instruction = "Classify the intent of the following banking customer message. Output ONLY the exact intent label in snake_case format."
    message = row['text']
    intent = row['label_name'].strip()
    
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{message}\n\n### Response:\n{intent}<|eot_id|>"
    return {"formatted_prompt": prompt}

def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "train.yaml")
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['model']['model_name'],
        max_seq_length = config['model']['max_seq_length'],
        dtype = None,
        load_in_4bit = config['model']['load_in_4bit'],
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora']['r'],
        target_modules = config['lora']['target_modules'],
        lora_alpha = config['lora']['lora_alpha'],
        lora_dropout = config['lora']['lora_dropout'],
        bias = config['lora']['bias'],
        use_gradient_checkpointing = "unsloth",
    )
    
    train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", config['data']['train_path']))
    train_dataset = Dataset.from_pandas(train_df).map(format_prompt)
    output_dir = os.path.join(os.path.dirname(__file__), "..", config['training']['output_dir'])
    

    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "formatted_prompt",
        max_seq_length = config['model']['max_seq_length'],
        dataset_num_proc = 2,
        data_collator = collator, 
        args = TrainingArguments(
            per_device_train_batch_size = config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
            warmup_steps = 5,
            num_train_epochs = 5,
            learning_rate = float(config['training']['learning_rate']),
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = config['training']['logging_steps'],
            optim = config['training']['optimizer'],
            weight_decay = config['training']['weight_decay'],
            lr_scheduler_type = config['training']['lr_scheduler_type'],
            seed = config['training']['seed'],
            output_dir = output_dir,
            report_to = "none",
        ),
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
