"""
import os
import pandas as pd
from tqdm import tqdm
from inference import IntentClassification

def main():
    test_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "test.csv")
    test_df = pd.read_csv(test_path)
    
    config_file = os.path.join(os.path.dirname(__file__), "..", "configs", "inference.yaml")
    classifier = IntentClassification(config_file)
    
    correct = 0
    total = len(test_df)
    
    print("\nBắt đầu đánh giá trên tập test...")
    for _, row in tqdm(test_df.iterrows(), total=total):
        prediction = classifier(row['text'])
        label = row['label_name'].strip().lower()
        
        # Chấm điểm
        if prediction == label or label in prediction:
            correct += 1
            
    accuracy = (correct / total) * 100
    print(f"\n")
    print(f"Final Accuracy on Test Set: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
""" 

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
    
    # Danh sách lưu trữ toàn bộ lịch sử dự đoán
    results = []
    
    print("\nBắt đầu đánh giá trên tập test...")
    for _, row in tqdm(test_df.iterrows(), total=total):
        prediction = classifier(row['text'])
        label = row['label_name'].strip().lower()
        
        # Đánh giá đúng/sai
        is_correct = (prediction == label) or (label in prediction)
        if is_correct:
            correct += 1
            
        # Lưu vào danh sách
        results.append({
            'Input_Text': row['text'],
            'Ground_Truth': label,
            'Model_Prediction': prediction,
            'Is_Correct': is_correct
        })
            
    accuracy = (correct / total) * 100
    
    # Đóng gói thành DataFrame và xuất ra file CSV
    results_df = pd.DataFrame(results)
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "detailed_predictions.csv")
    
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"\n========================================")
    print(f"Final Accuracy on Test Set: {accuracy:.2f}%")
    print(f"Đã xuất toàn bộ kết quả phân tích chi tiết tại: {output_csv_path}")
    print(f"========================================")
    
    # In nhanh 10 mẫu bị sai ra màn hình để phân tích tức thì
    print("\n[DEBUG] - DANH SÁCH 10 MẪU BỊ ĐOÁN SAI ĐẦU TIÊN:")
    errors_df = results_df[results_df['Is_Correct'] == False].head(10)
    
    for idx, row in errors_df.iterrows():
        print(f"\n[-] Input: {row['Input_Text']}")
        print(f"    Thực tế: '{row['Ground_Truth']}'")
        print(f"    Mô hình: '{row['Model_Prediction']}'")

if __name__ == "__main__":
    main()
