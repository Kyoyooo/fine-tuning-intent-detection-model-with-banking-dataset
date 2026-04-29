import os
import yaml
import torch
import re
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)['inference']
        
        self.device = config.get('device', 'cuda')
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['model_path'],
            max_seq_length = config.get('max_seq_length', 512),
            dtype = None,
            load_in_4bit = config.get('load_in_4bit', True),
        )
        FastLanguageModel.for_inference(self.model)
        
        self.prompt_template = (
            "### Instruction:\n"
            "Classify the intent of the following banking customer message. Output ONLY the exact intent label in snake_case format.\n\n"
            "### Input:\n"
            "{message}\n\n"
            "### Response:\n"
        )

    def __call__(self, message):
        prompt = self.prompt_template.format(message=message)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=32,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        if "### Response:" in decoded_output:
            predicted_text = decoded_output.split("### Response:")[-1].strip()
        else:
            predicted_text = decoded_output.strip()
        
        label = predicted_text.split('\n')[0].strip().lower()
        label = re.sub(r'[^a-z0-9]', '_', label)
        label = re.sub(r'_+', '_', label).strip('_')
        
        return label

if __name__ == "__main__":
    # Test mẫu
    config_file = os.path.join(os.path.dirname(__file__), "..", "configs", "inference.yaml")
    classifier = IntentClassification(config_file)

    test_messages = ['How do I link this new card?',
                     'How do I retrieve my card from the machine?',
                     'I want to know where the funds come from.',
                     '"I just activated auto top-up, but it is not letting me enable it. Why not?"',
                     'Why did I have to pay extra because I paid with card?',
                     "How can I fix a problem where contactless isn't working?",
                     'I took out a foreign currency and the exchange rate is wrong.',
                     "Will it automatically top-up money if there isn't a lot left?",
                     '"I see my top-up was canceled, but why?"',
                     'I lost my phone!'
                     ]

    for test_message in test_messages:
        print(f"Test Message: {test_message}\n") 
        print(f"Predicted Intent: {classifier(test_message)}\n\n")
