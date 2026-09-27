# 🚀 HƯỚNG DẪN CÀI ĐẶT & HUẤN LUYỆN HG-PARCO (HSCL)

Tài liệu này hướng dẫn chi tiết từ A-Z cách thiết lập môi trường, kiểm thử và chạy huấn luyện mô hình **HG-PARCO** tích hợp module **HSCL (Hypergraph-Structured Communication Layer)** trên các bài toán tối ưu tổ hợp đa tác tử (HCVRP, OMDCPDP, FFSP).

---

## 📌 0. Lưu Ý Quan Trọng Về Phiên Bản
Repo này chứa cả 2 phiên bản:
* 🌟 **Mô hình đề xuất HG-PARCO (HSCL)**: Sử dụng các file config có đuôi `_hscl` (ví dụ: `experiment=hcvrp_hscl`, `experiment=omdcpdp_hscl`, `experiment=ffsp_hscl`).
* 🏛️ **Mô hình PARCO nguyên bản (NeurIPS 2025 - Baseline)**: Sử dụng các file config gốc không có đuôi `_hscl` (ví dụ: `experiment=hcvrp`).

---

## 💻 1. Cài Đặt Môi Trường (Environment Setup)

### Bước 1.1: Tạo môi trường Conda (Khuyên dùng Python 3.10 hoặc 3.11)
```bash
conda create -n parco_hscl python=3.10 -y
conda activate parco_hscl
```

### Bước 1.2: Cài đặt PyTorch với CUDA
Tùy vào phiên bản CUDA trên máy của bạn (kiểm tra bằng lệnh `nvidia-smi`), chọn lệnh tương ứng từ [pytorch.org](https://pytorch.org):

* **Cho CUDA 12.1:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
* **Cho CUDA 11.8:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### Bước 1.3: Cài đặt các thư viện phụ thuộc và Repo PARCO
Đứng tại thư mục gốc của repository (`d:/parco` hoặc thư mục bạn clone về):
```bash
# Cài đặt repo dưới dạng editable package
pip install -e .

# Cài đặt các thư viện cần thiết
pip install lightning hydra-core hydra-colorlog omegaconf rl4co scipy tensordict torchrl huggingface_hub pytest tqdm
```

### Bước 1.4: Chạy Unit Test kiểm tra tính tương thích của HSCL
Chạy lệnh sau để đảm bảo toàn bộ mã nguồn HSCL (Spectral Clustering, AutoEigenSolver, History Buffer, Two-level Attention) hoạt động hoàn hảo trên phần cứng của bạn:
```bash
pytest tests/test_hscl.py
```
> ✅ **Kỳ vọng:** Toàn bộ **13/13 tests passed** trong khoảng 10–20 giây.

---

## 📥 2. Tải Dữ Liệu Benchmark Validation

Trong quá trình huấn luyện, mô hình sẽ tự động đánh giá sau mỗi epoch trên tập validation chuẩn của bài báo. Tải dữ liệu từ HuggingFace về thư mục `data/` (chỉ cần chạy 1 lần duy nhất):
```bash
python scripts/download_hf.py --data
```

---

## ⚡ 3. Kiểm Tra Nhanh Đường Ống (Sanity Check - 15 giây)

Trước khi tiến hành huấn luyện lâu dài, hãy chạy thử 1 epoch cực ngắn để đảm bảo không bị lỗi vặt về GPU hoặc bộ nhớ:
```bash
python train.py experiment=hcvrp_hscl trainer.fast_dev_run=true
```
Nếu chương trình in ra kết quả chạy qua 1 batch không lỗi và kết thúc bình thường, bạn đã sẵn sàng huấn luyện chính thức!

---

## 🎯 4. Các Lệnh Huấn Luyện Chính Thức (Training)

Kích hoạt môi trường trước khi chạy: `conda activate parco_hscl`.

### ① Bài toán HCVRP (Heterogeneous Capacitated Vehicle Routing)
Huấn luyện 1 mô hình tổng quát phủ toàn bộ các mốc $N \in [60, 100], M \in [3, 7]$ (100 epochs):
```bash
python train.py experiment=hcvrp_hscl
```

### ② Bài toán OMDCPDP (Open Multi-Depot Pickup and Delivery)
Huấn luyện 1 mô hình tổng quát phủ toàn bộ các mốc $N \in [50, 100], M \in [5, 20]$ (100 epochs - *bài này HSCL thể hiện rõ ưu thế nhất*):
```bash
python train.py experiment=omdcpdp_hscl
```

### ③ Bài toán FFSP (Flexible Flow Shop Scheduling)
* **Bản chuẩn dùng để so sánh Bảng 1 (FFSP50 - 150 epochs):**
  ```bash
  python train.py experiment=ffsp_hscl env=ffsp/ffsp50
  ```
* **Bản nhẹ hơn để chạy thử nhanh trước (FFSP20 - 100 epochs):**
  ```bash
  python train.py experiment=ffsp_hscl env=ffsp/ffsp20
  ```

---

## 🛠️ 5. Tùy Biến Theo Phần Cứng

1. **Nếu GPU có VRAM hạn chế (6GB - 8GB):**
   Nếu gặp lỗi tràn bộ nhớ (Out-Of-Memory / OOM), chỉ cần giảm batch size xuống 64 bằng cách thêm tham số vào cuối lệnh:
   ```bash
   python train.py experiment=hcvrp_hscl model.batch_size=64
   ```

2. **Nếu chạy trên Server nhiều GPU (Multi-GPU Linux Server):**
   Mặc định cấu hình đã được đặt `trainer.devices=1` và `trainer.strategy="auto"` để chạy an toàn trên 1 GPU (kể cả Windows).  
   Nếu bạn có máy chủ Linux 4x A100 / RTX 4090 và muốn tăng tốc tối đa bằng DDP:
   ```bash
   python train.py experiment=hcvrp_hscl trainer.devices=4 trainer.strategy=ddp
   ```

3. **Ghi log quá trình huấn luyện:**
   * Mặc định log đang tắt WandB (`logger: null`) để chạy offline không cần tài khoản.
   * Nếu muốn đẩy biểu đồ lên WandB cá nhân: thêm `logger=wandb` vào cuối lệnh.

---

## 📊 6. Đánh Giá Mô Hình Sau Khi Train Xong (Testing / Evaluation)

Sau khi huấn luyện xong, file checkpoint có điểm validation cao nhất sẽ được lưu tại:
`outputs/<ngày-tháng-năm>/<giờ-phút-giây>/checkpoints/best.ckpt`.

Dùng lệnh sau để test và so sánh với Bảng 1 trong bài báo:

### Đánh giá chế độ Greedy (nhanh nhất - tương ứng hàng `PARCO (g.)`):
```bash
python test.py --problem hcvrp --checkpoint outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/best.ckpt --decode_type greedy
```

### Đánh giá chế độ Sampling (lấy mẫu nhiều nghiệm - tương ứng hàng `PARCO (s.)`):
```bash
python test.py --problem hcvrp --checkpoint outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/best.ckpt --decode_type sampling --sample_size 1280
```
*(Đối với bài toán FFSP, đặt `--sample_size 128` theo đúng bài báo).*

---

## ❓ 7. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Vấn đề | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **CUDA out of memory (OOM)** | Batch size quá lớn so với dung lượng VRAM card đồ họa | Thêm `model.batch_size=64` hoặc `model.batch_size=32`. |
| **Lỗi `val_file` không tìm thấy** | Chưa tải bộ dữ liệu validation | Chạy `python scripts/download_hf.py --data` hoặc thêm `env.val_file=null` để bỏ qua val offline. |
| **Lỗi DDP trên Windows** | Windows không hỗ trợ NCCL backend | Giữ nguyên `trainer.strategy="auto"` và `trainer.devices=1` như đã cấu hình sẵn trong file `*_hscl.yaml`. |
