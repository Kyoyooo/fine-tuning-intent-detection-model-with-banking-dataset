# Fine-Tuning Intent Detection Model with Banking Dataset

## Thông tin sinh viên
- **Họ và tên:** Võ Trần Duy Hoàng
- **MSSV:** 23120266
- **Môn học:** Ứng dụng xử lý ngôn ngữ tự nhiên trong doanh nghiệp - CSC15012
- **Giảng viên hướng dẫn:** TS. Nguyễn Hồng Bửu Long, CN. Lê Đức Khoan

## Giới thiệu đồ án
Đồ án này thực hiện việc fine-tuning mô hình ngôn ngữ lớn (LLM) để phân loại ý định (intent classification) của khách hàng trong lĩnh vực ngân hàng. Dự án sử dụng tập dữ liệu **BANKING77** (77 loại ý định khác nhau) và thư viện **Unsloth** để tối ưu hóa tốc độ huấn luyện cũng như tiết kiệm tài nguyên bộ nhớ VRAM.

## Điểm nổi bật về kỹ thuật
Để đạt được kết quả cao, đồ án đã áp dụng các kỹ thuật sau:
- **Stratified Sampling**: Chia tập dữ liệu đảm bảo sự cân bằng giữa 77 nhãn, giúp mô hình không bị lệch (bias) về các nhãn phổ biến.
- **Completion-Only Fine-tuning**: Sử dụng kỹ thuật Masking để mô hình chỉ tập trung học phần kết quả (Response) thay vì học thuộc lòng câu lệnh (Prompt).
- **Llama-3-Instruct 4-bit**: Tận dụng sức mạnh của mô hình Llama-3 bản Instruct kết hợp với kỹ thuật Quantization (bitsandbytes) để chạy mượt mà trên các GPU phổ thông (T4/P100).
- **Regex-based Post-processing**: Bộ lọc chuẩn hóa đầu ra giúp loại bỏ các ký tự nhiễu và ép kết quả về đúng định dạng snake_case của tập dữ liệu.

## Cấu trúc thư mục
```text
banking-intent-unsloth
|-- scripts
|   |-- train.py            # Huấn luyện với thuật toán Masked Loss
|   |-- inference.py        # Suy luận với bộ lọc Regex và chuẩn hóa đầu ra
|   |-- preprocess_data.py  # Stratified Sampling đảm bảo cân bằng 77 nhãn
|   |-- evaluate.py         # Đánh giá Accuracy và xuất báo cáo chi tiết
|-- configs
|   |-- train.yaml          # Cấu hình LoRA (r=16, alpha=32) và Hyperparameters
|   |-- inference.yaml      # Cấu hình đường dẫn model checkpoint
|-- sample_data
|   |-- train.csv           # 1540 mẫu huấn luyện (20 mẫu/nhãn)
|   |-- test.csv            # 385 mẫu kiểm thử (5 mẫu/nhãn)
|-- outputs                 # Lưu trữ model sau khi fine-tune
|-- train.sh                # Bash script thực thi quy trình train
|-- inference.sh            # Bash script thực thi quy trình test mẫu
|-- requirements.txt        # Danh sách thư viện tương thích
|-- README.md               # Hướng dẫn chi tiết
```

## Thông số mô hình & Huấn luyện 
| Tham số | Giá trị |
|:------:|:--------:|
| Base Model | Unsloth Llama-3-8B-Instruct (4-bit) | 
| LoRA Rank (r) | 16 | 
| LoRA Alpha | 32 | 
| Learning Rate | 2e-4 | 
| Epochs | 5 | 
| Optimizer | AdamW 8-bit | 
| Batch Size | 4 (Gradient Accumulation: 4) | 

## Hướng dẫn cài đặt
1. Clone repository:
```bash
git clone [https://github.com/Kyoyooo/fine-tuning-intent-detection-model-with-banking-dataset.git](https://github.com/Kyoyooo/fine-tuning-intent-detection-model-with-banking-dataset.git)
cd fine-tuning-intent-detection-model-with-banking-dataset
```

2. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng
1. Tiền xử lý dữ liệu

Sử dụng kỹ thuật **Stratified Sampling** để trích xuất dữ liệu cân bằng từ **BANKING77**:
```bash
python scripts/preprocess_data.py
```
2. Huấn luyện mô hình

Chạy quy trình fine-tuning với **Unsloth**:
```bash
bash train.sh
```
3. Suy luận (Inference)

Kiểm tra mô hình với các tin nhắn đầu vào:
```bash
bash inference.sh
```

4. Đánh giá hệ thống
Chạy đánh giá trên toàn bộ tập test để xem Accuracy:
```bash
python scripts/evaluate.py
``` 

## Kết quả thử nghiệm
- **Độ chính xác (Accuracy): 88.05%**
- Nhận xét: Mô hình nhận diện chính xác các ý định khó và có sự tương đồng cao (như các vấn đề về thẻ hoặc phí giao dịch) nhờ vào việc ép định dạng đầu ra nghiêm ngặt.

## Video Demonstration   
Xem video hướng dẫn thực hiện và kết quả chạy script tại: https://drive.google.com/file/d/1nkCMTnh1gJyKRUSy-IhQYfQKEBEflATO/view?usp=sharing
