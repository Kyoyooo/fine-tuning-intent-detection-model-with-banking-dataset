import os
import yaml
import torch
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, config_path):
        """
        Khởi tạo class suy luận.
        :param config_path: Đường dẫn tới file configs/inference.yaml 
        """
        # 1. Đọc cấu hình từ file YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)['inference']
        
        self.model_path = config['model_path']
        self.max_seq_length = config.get('max_seq_length', 512)
        self.load_in_4bit = config.get('load_in_4bit', True)
        self.device = config.get('device', 'cuda')

        print(f"Đang tải mô hình từ checkpoint: {self.model_path}...")
        
        # 2. Tải mô hình và tokenizer (sử dụng Unsloth để tăng tốc suy luận)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_path,
            max_seq_length = self.max_seq_length,
            dtype = None,
            load_in_4bit = self.load_in_4bit,
        )
        
        # Kích hoạt chế độ suy luận nhanh của Unsloth
        FastLanguageModel.for_inference(self.model)
        
        # Template prompt phải trùng khớp với lúc huấn luyện 
        self.prompt_template = (
            "### Instruction:\n"
            "Classify the intent of the following banking customer message.\n\n"
            "### Input:\n"
            "{message}\n\n"
            "### Response:\n"
        )

    def __call__(self, message):
        """
        Nhận đầu vào là tin nhắn và trả về nhãn dự đoán
        """
        # Định dạng input theo template
        prompt = self.prompt_template.format(message=message)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # Thực hiện suy luận
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=32,
                use_cache=True
            )
        
        # Giải mã kết quả (chỉ lấy phần nội dung sau "### Response:")
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        predicted_text = decoded_output.split("### Response:")[-1].strip()
        
        # Chỉ lấy dòng đầu tiên hoặc từ đầu tiên để loại bỏ phần LLM nói dư thừa 
        predicted_label = predicted_text.split('\n')[0].strip()
        
        return predicted_label

if __name__ == "__main__":
    # Đường dẫn tới file cấu hình suy luận
    config_file = os.path.join(os.path.dirname(__file__), "..", "configs", "inference.yaml")
    
    # Khởi tạo module
    classifier = IntentClassification(config_file)
    
    # Thử nghiệm với một tin nhắn mẫu
    test_message = "I think I've lost my debit card, can you help me block it?"
    prediction = classifier(test_message)
    
    print(f"Message: {test_message}")
    print(f"Predicted Intent: {prediction}")
