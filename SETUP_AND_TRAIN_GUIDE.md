# 🚀 HƯỚNG DẪN CÀI ĐẶT & HUẤN LUYỆN HG-PARCO (HSCL)

Tài liệu này hướng dẫn chi tiết từ A-Z cách thiết lập môi trường, kiểm thử và chạy huấn luyện mô hình **HG-PARCO** tích hợp module **HSCL (Hypergraph-Structured Communication Layer)** trên các bài toán tối ưu tổ hợp đa tác tử (HCVRP, OMDCPDP, FFSP).

---

## 📌 0. Lưu Ý Quan Trọng Về Các Phiên Bản Mô Hình

Repo này hỗ trợ 3 cấu hình chính để phục vụ huấn luyện và làm bảng so sánh (ablation / benchmark):
* 👑 **CD-HG v2.1 + FMCR (Flagship - Đề xuất chính)**: Tích hợp Conflict-Supervised Hypergraph Layer và Fairness-Aware Multi-Round Resolver (nhanh hơn **2.5x**, giải quyết hoàn toàn nghẽn $O(M^3)$ của spectral, đạt Novelty cao nhất).  
  👉 File cấu hình: `experiment=hcvrp_cdhg`, `experiment=omdcpdp_cdhg`, `experiment=ffsp_cdhg`.
* 🧪 **HSCL v1 (Spectral Clustering)**: Phiên bản gom cụm phổ ban đầu.  
  👉 File cấu hình: `experiment=hcvrp_hscl`, `experiment=omdcpdp_hscl`, `experiment=ffsp_hscl`.
* 🏛️ **PARCO Nguyên Bản (NeurIPS 2025 Baseline)**: Mô hình phẳng chuẩn của tác giả gốc.  
  👉 File cấu hình: `experiment=hcvrp`, `experiment=omdcpdp`, `experiment=ffsp`.

---

## 💻 1. Cài Đặt Môi Trường (Environment Setup)

### Bước 1.1: Kích hoạt / Tạo môi trường Conda
```bash
conda activate ec_nco
# Hoặc tạo mới: conda create -n parco_env python=3.10 -y && conda activate parco_env
```

### Bước 1.2: Cài đặt PyTorch với CUDA
Tùy vào phiên bản CUDA trên máy (kiểm tra bằng `nvidia-smi`):
```bash
# Cho CUDA 12.1 trở lên:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Bước 1.3: Cài đặt các thư viện phụ thuộc và Repo PARCO
Đứng tại thư mục gốc của repository:
```bash
pip install -e .
pip install lightning hydra-core hydra-colorlog omegaconf rl4co scipy tensordict torchrl huggingface_hub pytest tqdm
```

### Bước 1.4: Chạy Unit Test kiểm tra toàn bộ module
Chạy lệnh sau để đảm bảo 100% tests của cả CD-HG v2.1, FMCR và HSCL v1 vượt qua trơn tru:
```bash
pytest tests/
```
> ✅ **Kỳ vọng:** **23/23 tests passed** trong ~15 giây.

---

## 📥 2. Tải Dữ Liệu Benchmark Validation
Tải dữ liệu từ HuggingFace về thư mục `data/` (chỉ cần chạy 1 lần duy nhất):
```bash
python scripts/download_hf.py --data
```
*(Nếu máy không tải được dữ liệu offline, chỉ cần thêm `env.val_file=null` vào lệnh train để mô hình tự sinh dữ liệu online).*

---

## ⚡ 3. Kiểm Tra Nhanh Đường Ống (Sanity Check - 30 giây)

Trước khi train lâu dài, chạy 1 bước để chắc chắn GPU và Mixed Precision không có lỗi:
```bash
python train.py experiment=omdcpdp_cdhg trainer.fast_dev_run=true env.val_file=null
```
Nếu log in ra `Trainer.fit stopped: max_steps=1 reached` và kết thúc với code 0, mọi thứ đã sẵn sàng 100%!

---

## 🎯 4. Các Lệnh Huấn Luyện Chính Thức (Mô Hình Đề Xuất: CD-HG v2.1 + FMCR)

### ① Bài toán OMDCPDP (Open Multi-Depot Pickup and Delivery - Khuyên dùng test trước)
Huấn luyện mô hình tổng quát với số xe lớn $M \in [5, 50]$ và số điểm $N \in [50, 100]$:
```bash
python train.py experiment=omdcpdp_cdhg
```

### ② Bài toán HCVRP (Heterogeneous Capacitated Vehicle Routing)
Huấn luyện bài toán định tuyến xe không đồng nhất:
```bash
python train.py experiment=hcvrp_cdhg
```

### ③ Bài toán FFSP (Flexible Flow Shop Scheduling)
* **Bản chuẩn dùng so sánh Bảng 1 (FFSP50 - 150 epochs):**
  ```bash
  python train.py experiment=ffsp_cdhg env=ffsp/ffsp50
  ```
* **Bản nhẹ hơn để chạy nhanh (FFSP20 - 50 epochs):**
  ```bash
  python train.py experiment=ffsp_cdhg env=ffsp/ffsp20
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
