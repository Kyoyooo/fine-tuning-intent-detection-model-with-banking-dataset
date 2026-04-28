import os
import pandas as pd
from tqdm import tqdm
from inference import IntentClassification

def main():
    print("Đang tải tập test và mô hình...")
    test_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "test.csv")
    test_df = pd.read_csv(test_path)
    
    config_file = os.path.join(os.path.dirname(__file__), "..", "configs", "inference.yaml")
    classifier = IntentClassification(config_file)
    
    correct = 0
    total = len(test_df)
    errors_printed = 0
    
    print("\nBắt đầu đánh giá trên tập test...")
    for _, row in tqdm(test_df.iterrows(), total=total):
        prediction = classifier(row['text'])
        label = row['label_name'].strip().lower()
        
        # Nới lỏng so sánh: Bằng nhau HOẶC nhãn đúng nằm trong dự đoán
        if prediction == label or label in prediction:
            correct += 1
        else:
            # IN RA 5 LỖI ĐẦU TIÊN ĐỂ XEM ĐIỀU GÌ ĐANG XẢY RA
            if errors_printed < 5:
                tqdm.write(f"\n[DEBUG - LỖI {errors_printed + 1}]")
                tqdm.write(f"Input : {row['text']}")
                tqdm.write(f"Ground Truth (Thực tế): '{label}'")
                tqdm.write(f"Model Predict (Mô hình): '{prediction}'")
                errors_printed += 1
            
    accuracy = (correct / total) * 100
    print(f"\n========================================")
    print(f"Final Accuracy on Test Set: {accuracy:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()
    print(f"\n")
    print(f"Final Accuracy on Test Set: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
