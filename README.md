# Fine-Tuning Intent Detection Model with Banking Dataset

## Thông tin sinh viên
- **Họ và tên:** Võ Trần Duy Hoàng
- **MSSV:** 23120266
- **Môn học:** Ứng dụng xử lý ngôn ngữ tự nhiên trong doanh nghiệp - CSC15012
- **Giảng viên hướng dẫn:** TS. Nguyễn Hồng Bửu Long, CN. Lê Đức Khoan

## Giới thiệu dự án
Dự án này thực hiện việc fine-tuning mô hình ngôn ngữ lớn (LLM) để phân loại ý định (intent classification) của khách hàng trong lĩnh vực ngân hàng sử dụng tập dữ liệu **BANKING77** và thư viện **Unsloth** để tối ưu hóa hiệu năng.

## Cấu trúc thư mục
```text
banking-intent-unsloth
|-- scripts
|   |-- train.py            # Script huấn luyện mô hình
|   |-- inference.py        # Script thực hiện suy luận
|   |-- preprocess_data.py  # Tiền xử lý và chia tập dữ liệu
|   |-- evaluate.py         # Kết quả mô hình trên tập test 
|-- configs
|   |-- train.yaml          # Cấu hình siêu tham số huấn luyện
|   |-- inference.yaml      # Cấu hình đường dẫn model suy luận
|-- sample_data
|   |-- train.csv           # Dữ liệu huấn luyện sau khi sample
|   |-- test.csv            # Dữ liệu kiểm thử sau khi sample
|-- outputs                 # Thư mục lưu checkpoint (tự động tạo)
|-- train.sh                # Script thực thi huấn luyện
|-- inference.sh            # Script thực thi suy luận
|-- requirements.txt        # Các thư viện cần thiết
|-- README.md               # Hướng dẫn dự án
```

## Hướng dẫn cài đặt
1. Clone repository:
```bash
git clone [https://github.com/Kyoyooo/fine-tuning-intent-detection-model-with-banking-dataset.git](https://github.com/Kyoyooo/fine-tuning-intent-detection-model-with-banking-dataset.git)
cd fine-tuning-intent-detection-model-with-banking-dataset
```

2. Cài đặt thư viện:
```bash
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
```

## Hướng dẫn sử dụng
1. Tiền xử lý dữ liệu

Tải tập dữ liệu **BANKING77**, thực hiện chuẩn hóa và lưu vào thư mục `sample_data/`:
```bash
python scripts/preprocess_data.py
```
2. Huấn luyện mô hình

Chạy quy trình fine-tuning với **Unsloth**:
```bash
bash train.sh
```
3. Suy luận (Inference)

Kiểm tra mô hình với một tin nhắn đầu vào:
```bash
bash inference.sh
```

## Kết quả thử nghiệm
- Model sử dụng: Llama-3-8B-bnb-4bit (via Unsloth)   
- Độ chính xác (Accuracy) trên tập test: ...%

## Video Demonstration   
Xem video hướng dẫn thực hiện và kết quả chạy script tại: ...
