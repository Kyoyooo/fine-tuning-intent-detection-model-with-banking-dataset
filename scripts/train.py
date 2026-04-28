import os
import yaml
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def load_config(config_path):
    """Đọc file cấu hình YAML."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def format_prompt(row):
    """
    Định dạng dữ liệu thành prompt cho bài toán phân loại ý định (Sequence Classification).
    Chuyển đổi văn bản và nhãn thành một chuỗi duy nhất để LLM học cách sinh ra nhãn.
    """
    instruction = "Classify the intent of the following banking customer message. Output ONLY the EXACT label from the BANKING77 dataset in snake_case format. Do not invent new labels."
    message = row['text']
    intent = row['label_name'].strip()
    
    # Chèn <|eot_id|> để ngắt luồng sinh văn bản
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{message}\n\n### Response:\n{intent}<|eot_id|>"
    return {"formatted_prompt": prompt}

def main():
    # 1. Tải cấu hình
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "train.yaml")
    config = load_config(config_path)
    
    model_cfg = config['model']
    lora_cfg = config['lora']
    data_cfg = config['data']
    train_cfg = config['training']
    
    print(f"Đang khởi tạo mô hình: {model_cfg['model_name']}...")
    
    # 2. Khởi tạo mô hình và tokenizer với Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_cfg['model_name'],
        max_seq_length = model_cfg['max_seq_length'],
        dtype = None, # Tự động phát hiện dtype phù hợp (Float16/Bfloat16)
        load_in_4bit = model_cfg['load_in_4bit'],
    )
    
    # Cấu hình LoRA/QLoRA để tối ưu hóa tham số (PEFT)
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_cfg['r'],
        target_modules = lora_cfg['target_modules'],
        lora_alpha = lora_cfg['lora_alpha'],
        lora_dropout = lora_cfg['lora_dropout'],
        bias = lora_cfg['bias'],
        use_gradient_checkpointing = "unsloth",
        random_state = train_cfg['seed'],
        use_rslora = False,
        loftq_config = None,
    )
    
    # 3. Chuẩn bị dữ liệu
    print("Đang tải và chuẩn bị dữ liệu...")
    train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", data_cfg['train_path']))
    train_dataset = Dataset.from_pandas(train_df)
    
    # Áp dụng template prompt
    train_dataset = train_dataset.map(format_prompt)
    
    # 4. Cấu hình Trainer
    output_dir = os.path.join(os.path.dirname(__file__), "..", train_cfg['output_dir'])
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "formatted_prompt",
        max_seq_length = model_cfg['max_seq_length'],
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences
        args = TrainingArguments(
            per_device_train_batch_size = train_cfg['per_device_train_batch_size'],
            gradient_accumulation_steps = train_cfg['gradient_accumulation_steps'],
            warmup_steps = 5,
            num_train_epochs = train_cfg['num_train_epochs'],
            learning_rate = float(train_cfg['learning_rate']),
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = train_cfg['logging_steps'],
            optim = train_cfg['optimizer'],
            weight_decay = train_cfg['weight_decay'],
            lr_scheduler_type = train_cfg['lr_scheduler_type'],
            seed = train_cfg['seed'],
            output_dir = output_dir,
            report_to = "none", # Tắt wandb/tensorboard nếu không cần thiết
        ),
    )
    
    # 5. Tiến hành huấn luyện
    print("\nBắt đầu huấn luyện...")
    trainer_stats = trainer.train()
    print("\nĐã huấn luyện xong!")
    
    # 6. Lưu mô hình (Checkpoint)
    print(f"Đang lưu mô hình tại: {output_dir}")
    # Lưu dưới dạng LoRA adapters 
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Đã lưu mô hình và tokenizer.")

if __name__ == "__main__":
    main()
