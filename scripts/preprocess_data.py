import os
import pandas as pd
from datasets import load_dataset

def clean_text(text):
    if not isinstance(text, str): return ""
    return " ".join(text.strip().split())

def main():
    print("Đang tải tập dữ liệu BANKING77...")
    dataset = load_dataset("banking77")
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

    # Lấy danh sách tên nhãn
    label_names = dataset['train'].features['label'].names
    df_train['label_name'] = df_train['label'].apply(lambda x: label_names[x])
    df_test['label_name'] = df_test['label'].apply(lambda x: label_names[x])

    df_train['text'] = df_train['text'].apply(clean_text)
    df_test['text'] = df_test['text'].apply(clean_text)

    # Sampling: Lấy đúng 20 mẫu/nhãn cho train (1540 câu) và 5 mẫu/nhãn cho test (385 câu)
    df_train_sampled = df_train.groupby('label').sample(n=20, random_state=42, replace=True)
    df_test_sampled = df_test.groupby('label').sample(n=5, random_state=42, replace=True)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data")
    os.makedirs(output_dir, exist_ok=True)
    
    df_train_sampled[['text', 'label', 'label_name']].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_test_sampled[['text', 'label', 'label_name']].to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print("Đã chuẩn bị dữ liệu xong!")

if __name__ == "__main__":
    main()
