import os
import pandas as pd
from datasets import load_dataset

def clean_text(text):
    """Hàm chuẩn hóa văn bản cơ bản."""
    if not isinstance(text, str):
        return ""
    # Xóa khoảng trắng thừa ở đầu/cuối và giữa các từ
    return " ".join(text.strip().split())

def main():
    print("Đang tải tập dữ liệu BANKING77 từ Hugging Face...")
    # 1. Tải dataset BANKING77
    dataset = load_dataset("banking77")
    
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    
    print(f"Kích thước tập train gốc: {len(df_train)}")
    print(f"Kích thước tập test gốc: {len(df_test)}")
    
    # 2. Rút trích tập con (Sampling)
    # Lấy 2000 mẫu train và 500 mẫu test 
    SAMPLE_TRAIN_SIZE = 2000
    SAMPLE_TEST_SIZE = 500
    
    df_train_sampled = df_train.sample(n=SAMPLE_TRAIN_SIZE, random_state=42)
    df_test_sampled = df_test.sample(n=SAMPLE_TEST_SIZE, random_state=42)
    
    # 3. Tiền xử lý dữ liệu
    print("Đang thực hiện chuẩn hóa văn bản...")
    df_train_sampled['text'] = df_train_sampled['text'].apply(clean_text)
    df_test_sampled['text'] = df_test_sampled['text'].apply(clean_text)
    
    # 4. Định dạng nhãn (Label mapping)
    # Dataset banking77 mặc định đã gán nhãn dưới dạng số nguyên (integer) trong cột 'label'.
    # Định dạng này sẵn sàng cho bài toán Sequence Classification của Unsloth/Transformers.
    
    # Lấy danh sách tên nhãn (string) để tham chiếu
    features = dataset['train'].features['label']
    label_names = features.names
    
    # Thêm một cột chứa tên nhãn dạng text
    df_train_sampled['label_name'] = df_train_sampled['label'].apply(lambda x: label_names[x])
    df_test_sampled['label_name'] = df_test_sampled['label'].apply(lambda x: label_names[x])

    # 5. Lưu dữ liệu
    # Tạo thư mục sample_data ở thư mục gốc của project
    output_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data")
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    # Chỉ lưu các cột cần thiết: 'text' và 'label'
    columns_to_save = ['text', 'label', 'label_name']
    df_train_sampled[columns_to_save].to_csv(train_path, index=False)
    df_test_sampled[columns_to_save].to_csv(test_path, index=False)
    
    print("\nĐã lưu dữ liệu sample:")
    print(f"- Train data: {train_path} ({len(df_train_sampled)} dòng)")
    print(f"- Test data: {test_path} ({len(df_test_sampled)} dòng)")

if __name__ == "__main__":
    main()
