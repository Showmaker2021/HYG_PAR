# HG-PARCO: Hypergraph-Structured Communication for Parallel Multi-Agent Combinatorial Optimization
**Tài liệu Đặc tả Ý tưởng và Kiến trúc Chi tiết Module HSCL**

---

## 1. Tóm tắt (Executive Summary)

**HG-PARCO** là khung học tăng cường (Reinforcement Learning - RL) tiên tiến được xây dựng để giải quyết các bài toán Tối ưu hóa Tổ hợp Đa tác tử (Multi-Agent Combinatorial Optimization - MA-CO) quy mô lớn. 

Phương pháp kế thừa và nâng cấp kiến trúc đột phá **PARCO** (Berto et al., NeurIPS 2025) bằng cách thay thế cơ chế giao tiếp phẳng (Flat All-to-All Attention) bằng **Lớp Giao tiếp Cấu trúc Siêu đồ thị (Hypergraph-Structured Communication Layer - HSCL)**. Module HSCL tự động phát hiện các cụm tương tác động giữa các tác tử thông qua Phân tích Phổ (Spectral Clustering) trên lịch sử quỹ đạo, đồng thời điều phối thông tin thông qua cơ chế chú ý hai tầng (**Intra-Hyperedge** cục bộ và **Inter-Hyperedge** toàn cục). 

Kiến trúc này triệt tiêu hiện tượng loãng thông tin (Information Dilution), tối ưu hóa xung đột hành động ngay từ giai đoạn giải mã, giảm độ phức tạp tính toán và nâng cao vượt bậc khả năng khái quát hóa quy mô (Zero-shot Generalization) khi số lượng tác tử $M$ tăng cao.

---

## 2. Bối cảnh & Động lực Nghiên cứu (Motivation & Problem Statement)

### 2.1 Thành tựu và Giới hạn của PARCO (SOTA)
Trong tối ưu tổ hợp đa tác tử (như điều phối xe tải giao hàng HCVRP, OMDCPDP hay lập lịch máy sản xuất FFSP), các phương pháp Autoregressive (AR) truyền thống thường giải mã tuần tự từng tác tử một ($O(M \times T)$ bước), gây ra độ trễ cực lớn và khó áp dụng trong thực tế thời gian thực.

PARCO (NeurIPS 2025) đã giải quyết vấn đề độ trễ bằng cơ chế **Parallel AutoRegressive (Parallel AR)**:
* Cho phép toàn bộ $M$ tác tử cùng đưa ra quyết định song song tại mỗi bước giải mã ($O(T)$ bước).
* Sử dụng một khối **Communication Layer** (Transformer Block) để các tác tử trao đổi thông tin với nhau trước khi trỏ (pointer) vào hành động.
* Sử dụng **Priority-based Conflict Handler** để phân xử va chạm khi nhiều tác tử cùng chọn một tài nguyên/khách hàng.

### 2.2 Điểm nghẽn Cốt lõi của Lớp Giao tiếp Phẳng (The Flat Communication Bottleneck)
Mặc dù vượt trội ở quy mô nhỏ ($M \le 7$), khi số lượng tác tử mở rộng ($M \ge 20, 50, 100$), Communication Layer của PARCO bộc lộ **ba hạn chế mang tính cấu trúc**:

1. **Hiện tượng Loãng Thông tin & Nhiễu Tương tác (Information Dilution & Over-smoothing):**  
   PARCO sử dụng Full Self-Attention với độ phức tạp $O(M^2)$. Mọi tác tử đều buộc phải chú ý đến nhau bình đẳng. Khi $M$ lớn, hàm Softmax bị san phẳng trên $M$ tác tử: một chiếc xe ở khu vực phía Bắc vẫn phải tính chú ý với một chiếc xe ở khu vực phía Nam. Điều này làm loãng tín hiệu hợp tác sống còn giữa các tác tử đang cùng hoạt động trong một phân vùng địa lý.

2. **Xung đột trong MA-CO mang bản chất "Nhóm Bậc Cao" (High-Order Group Dynamics):**  
   Đồ thị chuẩn và ma trận chú ý thông thường chỉ nắm bắt quan hệ cặp đôi (pairwise: $u \leftrightarrow v$). Tuy nhiên, trong logistics và sản xuất, sự tranh chấp tài nguyên (ví dụ: một cụm đơn hàng dày đặc, một trạm sạc, hoặc một công đoạn thắt nút cổ chai) luôn là sự tương tác đồng thời của **một nhóm $k \ge 3$ tác tử**. Quan hệ nhóm này là phi tuyến và đa diện ($N$-ary relation), không thể biểu diễn tự nhiên nếu chỉ dùng các cạnh đơn lẻ.

3. **Gia tăng Tỷ lệ Va chạm & Lãng phí Bước Giải:**  
   Do không có sự phân cấp giao tiếp, các tác tử trong cùng khu vực không thể bàn bạc sâu để nhường đường cho nhau trước khi ra quyết định. Hệ quả là tỷ lệ xung đột ở Multiple Pointer Mechanism tăng cao, hệ thống buộc phải kích hoạt hành động dự phòng (Fallback - "đứng yên"), làm kéo dài số bước giải mã và suy giảm chất lượng nghiệm.

---

## 3. Ý tưởng Đột phá: Tại sao lại là Hypergraph (Siêu Đồ Thị)?

### 3.1 Khái niệm Siêu Đồ Thị (Hypergraph)
Trong lý thuyết đồ thị mở rộng, một **Siêu đồ thị** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ gồm tập các đỉnh $\mathcal{V}$ (các tác tử) và tập các **siêu cạnh (hyperedges)** $\mathcal{E}$. Khác với cạnh thông thường chỉ nối 2 đỉnh, một siêu cạnh $e \in \mathcal{E}$ có thể bao bọc **một tập hợp con tùy ý gồm nhiều đỉnh**:
$$e_k = \{v_{i_1}, v_{i_2}, \dots, v_{i_m}\} \subseteq \mathcal{V}$$

### 3.2 Ý nghĩa Vật lý trong Bài toán Tối ưu Đa Tác tử
* Mỗi **Hyperedge** đóng vai trò là một **"Vùng tương tác động / Cụm xung đột" (Dynamic Interaction Zone)**.
* Các tác tử cùng một siêu cạnh là các tác tử có xác suất cao sẽ cạnh tranh cùng một nhóm tài nguyên (ví dụ: cùng phục vụ một cụm khách hàng lân cận, hoặc cùng chờ nhận việc tại một phân xưởng).
* Việc phân chia siêu cạnh được thực hiện **động (dynamic) và không giám sát (unsupervised)** tại mỗi bước giải mã, thích ứng theo sự thay đổi vị trí và tải trọng của các tác tử.

---

## 4. Kiến trúc Chi tiết của Module HSCL

Khối HSCL thay thế hoàn toàn khối `Communication Layer` cũ bên trong Decoder của PARCO. Luồng xử lý được chia làm 3 giai đoạn chặt chẽ:

```
[B, M, d] (Context Embedding)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: State History Buffer & Affinity Construction       │
│ • Lưu vết trạng thái W bước gần nhất: [B, M, W, d]              │
│ • Tính ma trận tương đồng không gian-thời gian:                 │
│   A_ij = exp( - ||s_i - s_j||^2 / 2σ^2 )                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: Spectral Hyperedge Discovery (AutoEigenSolver)     │
│ • Ma trận Laplacian chuẩn hóa đối xứng: L_sym = I - D^(-1/2) A D^(-1/2)│
│ • AutoEigenSolver: Tự động chọn GPU/CPU Eigendecomposition      │
│ • Chiếu lên không gian phổ U_k ──> K-Means ──> K Hyperedges e_k │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: Two-Level Hypergraph Attention Conv (HAC)          │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ (3a) Intra-Hyperedge Attention (Giao tiếp Cục bộ)           │ │
│ │ • Các tác tử trong cùng siêu cạnh chú ý lẫn nhau (MHA)      │ │
│ │ • Triệt tiêu va chạm cục bộ ngay trong biểu diễn            │ │
│ └──────────────────────────────┬──────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ (3b) Inter-Hyperedge Attention via Centroids (Toàn cục)     │ │
│ │ • Tính đại diện cụm: c_k = Mean({h_i | i ∈ e_k})            │ │
│ │ • Chú ý liên cụm giữa các Centroid (Top-r lân cận)          │ │
│ │ • Phân phối lại thông tin toàn cục: h_i += c_k              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────┘
                                 │ (+) Residual
                                 ▼
                         [ RMSNorm / LayerNorm ]
                                 │
                                 ▼
                         [ Gated MLP / FFN ]
                                 │ (+) Residual
                                 ▼
                         [ RMSNorm / LayerNorm ]
                                 │
                                 ▼
                 Output [B, M, d] ──> Query Q cho Pointer
```

### 4.1 Giai đoạn 1: Bộ đệm Lịch sử Trạng thái (State History Buffer)
Thay vì chỉ dựa vào một lát cắt trạng thái tức thời $s_t$, HSCL duy trì một cửa sổ trượt (sliding window) kích thước $W$ ($W=8$):
$$\mathcal{H}_t = [s_{t-W+1}, \dots, s_t] \in \mathbb{R}^{B \times M \times W \times d}$$
Từ chuỗi lịch sử này, ma trận tương đồng (Affinity Matrix) $A \in \mathbb{R}^{B \times M \times M}$ được tính toán qua khoảng cách quỹ đạo chuẩn hóa hoặc cosine similarity, phản ánh chân thực xu hướng di chuyển và phạm vi hoạt động của từng cặp tác tử.

### 4.2 Giai đoạn 2: Khám phá Siêu cạnh bằng Phân tích Phổ (Spectral Clustering)
1. **Tính ma trận Laplacian đối xứng:**
   $$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$$
   với ma trận bậc $D_{ii} = \sum_j A_{ij}$.
2. **AutoEigenSolver & Khắc phục Vách đá Hiệu năng (Hardware Cliff):**
   * *Phát hiện thực nghiệm:* Thư viện cuSOLVER trên GPU (`syevjBatched`) chỉ tối ưu khi ma trận nhỏ. Khi $M > 32$, thời gian giải suy biến tăng vọt tới $177\times$ (từ $0.64\text{ ms} \to 114\text{ ms}$).
   * *Giải pháp AutoEigenSolver:* 
     * Khi $M \le 32$: Sử dụng `DenseGPUEigh` trực tiếp trên VRAM.
     * Khi $M > 32$: Tự động chuyển ma trận sang CPU, giải bằng `DenseCPUEigh` với `torch.set_num_threads(1)` (chỉ mất $\sim 17\text{ ms}$ thay vì $114\text{ ms}$ trên GPU).
3. **Phân cụm K-Means:**
   Trích xuất $K$ vector riêng nhỏ nhất tương ứng để tạo không gian nhúng phổ, sau đó chạy Batched K-Means để gán nhãn mỗi tác tử $i$ vào một siêu cạnh $e_k$ ($k \in \{1, \dots, K\}$).

### 4.3 Giai đoạn 3: Tích chập Chú ý Siêu đồ thị Hai Tầng (`HypergraphAttentionConv`)
Sau khi có phân nhóm siêu cạnh, thông tin được truyền qua 2 tầng:

* **Tầng 1 - Intra-Hyperedge Attention (Cục bộ):**  
  Tạo ma trận mặt nạ nhị phân $M_{\text{intra}}$: $M_{ij} = 1$ nếu $i, j \in e_k$, ngược lại bằng $0$.
  $$H_{\text{intra}} = \text{MHA}(Q, K, V; \text{mask}=M_{\text{intra}})$$
  Các tác tử cùng khu vực tập trung $100\%$ dung lượng chú ý vào đối thủ cạnh tranh trực tiếp của mình, học cách phân chia khách hàng mà không bị nhiễu bởi các tác tử ở xa.

* **Tầng 2 - Inter-Hyperedge Attention via Centroids (Toàn cục):**  
  Mỗi siêu cạnh $e_k$ được tổng hợp thành một vector trọng tâm (Centroid):
  $$c_k = \frac{1}{|e_k|} \sum_{i \in e_k} h_{i} \in \mathbb{R}^d$$
  Các Centroid đại diện này ($C \in \mathbb{R}^{B \times K \times d}$) trao đổi thông tin với nhau qua một tầng Attention liên cụm:
  $$C' = \text{MHA}(C, C, C; \text{Top-}r)$$
  Thông tin toàn cục sau đó được hòa trộn ngược lại vào từng tác tử trong cụm ($h_i \leftarrow h_i + c'_k$).

---

## 5. Các Đặc tính Kỹ thuật Nổi bật

| Tiêu chí | PARCO gốc (All-to-All) | HG-PARCO (HSCL) | Lợi thế của HSCL |
| :--- | :--- | :--- | :--- |
| **Độ phức tạp Chú ý** | $O(M^2)$ | $O(\sum |e_k|^2 + K^2)$ | Tiết kiệm đáng kể tài nguyên tính toán khi $M$ lớn |
| **Mô hình hóa Tương tác** | Cặp đôi (Pairwise) | Nhóm bậc cao (Hyperedge) | Phù hợp tự nhiên với xung đột tài nguyên đa tác tử |
| **Chất lượng Tín hiệu** | Loãng dần khi $M$ tăng | Luôn tập trung sắc nét | Ngăn ngừa hiện tượng Over-smoothing |
| **Tương thích ngược** | Baseline | Fallback Bit-exact khi `use_hscl=False` | An toàn $100\%$, không làm hỏng kiến trúc nền |
| **Khả năng tổng quát (Zero-shot)** | Giảm sút nghiêm trọng ở $M$ lớn | Vượt trội nhờ tính bất biến (Agnostic) | Càng lên quy mô lớn, Hyperedge càng thể hiện rõ tính cục bộ |

---

## 6. Hướng dẫn Thực nghiệm & Sử dụng

### 6.1 Huấn luyện (Training)
Tất cả các cấu hình huấn luyện đã được tích hợp sẵn trong thư mục `configs/experiment/`:
* **HCVRP:** `python train.py experiment=hcvrp_hscl`
* **OMDCPDP:** `python train.py experiment=omdcpdp_hscl`
* **FFSP:** `python train.py experiment=ffsp_hscl env=ffsp/ffsp50`

### 6.2 Đánh giá (Evaluation)
* **Greedy:** `python test.py --problem omdcpdp --checkpoint <path_to_ckpt> --decode_type greedy`
* **Sampling:** `python test.py --problem omdcpdp --checkpoint <path_to_ckpt> --decode_type sampling --sample_size 1280`

---

## 7. Lộ trình Phát triển (Roadmap)
* **Giai đoạn 1 (Đã hoàn thành 100%):** Triển khai toàn bộ module HSCL, AutoEigenSolver, cấu hình huấn luyện và kiểm thử tương thích cho HCVRP, OMDCPDP, FFSP.
* **Giai đoạn 2 (Dự kiến):** Nghiên cứu mở rộng **FMCR (Fairness-Aware Multi-Round Conflict Resolver)** — thay thế cơ chế Conflict Handler tức thời 1 vòng bằng cơ chế phân xử công bằng đa vòng dài hạn tích hợp bộ đếm lịch sử ưu tiên.
