# Kết quả đo đạc hệ thống RAG

**Ngày đo:** 2026-09-03 · **Corpus:** toàn bộ `TaiLieuTest/` — **26 tài liệu**

Mọi con số trong file này đều đo lại trên **cùng một index**, bằng chính các script trong
`evaluation/`. Phần nào không đo được thì ghi rõ là không đo được, không suy đoán thay.

---

## 1. Môi trường đo

| Hạng mục | Giá trị |
|---|---|
| CPU | Intel Core i5-14600KF (14 nhân / 20 luồng) |
| RAM | 31.8 GB |
| GPU | NVIDIA GeForce RTX 5060 |
| Hệ điều hành | Windows 11 Pro 10.0.26200 |
| Python | 3.14.2 |
| faiss | 1.15.0 |
| sentence-transformers | 6.0.0 |
| streamlit | 1.62.0 |
| **torch** | **2.13.0+cpu — CUDA KHÔNG khả dụng** |

> **Chi tiết ảnh hưởng trực tiếp tới mọi con số độ trễ bên dưới:** `torch` là bản CPU-only,
> nên **embedding và cross-encoder rerank chạy hoàn toàn trên CPU**, trong khi Ollama (LLM
> `qwen3:4b` và vision `qwen2.5vl:3b`) chạy trên GPU. Đây là lý do phần truy xuất tốn 11–12
> giây/câu trong khi phần sinh câu trả lời — vốn nặng hơn nhiều — lại nhanh hơn. Cài `torch`
> bản CUDA sẽ rút ngắn đáng kể phần truy xuất; con số trong báo cáo nên kèm ghi chú này.

**Model sử dụng**

| Vai trò | Model | Nơi chạy |
|---|---|---|
| Embedding | `intfloat/multilingual-e5-base` (768 chiều) | CPU |
| Rerank (cross-encoder) | `BAAI/bge-reranker-v2-m3` | CPU |
| Sinh câu trả lời | `qwen3:4b` qua Ollama | GPU |
| Chú thích ảnh | `qwen2.5vl:3b` qua Ollama | GPU |
| Chấm điểm (LLM-as-judge) | `qwen3:4b` | GPU |

---

## 2. Corpus

**26 tài liệu** · **3.648 trang/bản ghi** · **9.285 chunk** · 1.953 trang riêng biệt
(11 PDF · 9 DOCX · 6 PPTX)

| Tài liệu | chunk | trang | ảnh | bảng |
|---|---:|---:|---:|---:|
| Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf | 4787 | 729 | 81 | 2 |
| [NguyenphuongLaw] Giáo trình Pháp luật Đại cương.pdf | 916 | 230 | 0 | 4 |
| Chapter 1. Introduction to IoT.pptx | 740 | 230 | 401 | 32 |
| Chapter 1.5 Introduction to IoT.pptx | 455 | 143 | 257 | 19 |
| Bai1-TongQuanDuLieu.docx | 265 | 7 | 219 | 2 |
| Chapter 4. Công nghệ trí tuệ nhân tạo trong ứng dụng kết nối vạn vật.pptx | 243 | 92 | 117 | 3 |
| Chapter 3. Giao tiếp với hệ thống cảm biến.pptx | 186 | 87 | 32 | 4 |
| Chapter 2. Server kết nối vạn vật.pptx | 150 | 88 | 49 | 1 |
| PaperQA.pdf | 146 | 20 | 3 | 0 |
| CV-05-Classification.pdf | 144 | 43 | 86 | 4 |
| Bai6-CayQuyetDinh.docx | 141 | 4 | 100 | 0 |
| CV-02-Camera.pdf | 126 | 54 | 42 | 0 |
| RFRAG.pdf | 119 | 18 | 0 | 3 |
| Bai4-LuatKetHop.docx | 113 | 3 | 50 | 0 |
| CV-04-RecognitionLocalFeatures.pdf | 108 | 41 | 30 | 2 |
| Bai2-ChuanBiDuLieu.docx | 103 | 2 | 77 | 0 |
| Bai3-GomCum.docx | 102 | 4 | 75 | 0 |
| NHÓM 6 - LUẬT HÌNH SỰ.pptx | 74 | 51 | 18 | 4 |
| CV-06-MotionOpticalFlow.pdf | 72 | 34 | 18 | 1 |
| CV-03-RecognitionGlobalFeatures.pdf | 69 | 30 | 10 | 6 |
| Bai5-PhanLopVoiKNN-NaiveBayes.docx | 61 | 1 | 24 | 0 |
| baocaonangcaothayAn.docx | 55 | 2 | 1 | 3 |
| BAO_CAO_MAY_HOC.docx | 48 | 15 | 6 | 4 |
| DeCuongNCKH.docx | 26 | 1 | 0 | 7 |
| CV-01-Introduction.pdf | 23 | 14 | 8 | 11 |
| CV-00-Topics.pdf | 13 | 10 | 3 | 1 |
| **TỔNG** | **9285** | **1953** | **1707** | **113** |

**Phân bố loại nội dung:** 7.465 văn bản · 1.707 ảnh · 113 bảng
**Ảnh có chú thích do model vision đọc nội dung:** 1.681 / 1.707 (98,5%)

**Chất lượng chunking**

| Chỉ số | Giá trị |
|---|---|
| Token/chunk — trung bình | 121,3 |
| Token/chunk — trung vị | 138 |
| Token/chunk — lớn nhất | 491 |
| Giới hạn của embedding model | 512 |
| **Chunk VƯỢT giới hạn model** | **0 / 9.285 (0,00%)** |
| Độ dài chunk (ký tự) — trung bình / lớn nhất | 412 / 1.789 |

Không có chunk nào bị model cắt âm thầm lúc encode. Chunk dài nhất (491 token) là một bảng —
bảng được phép dài hơn văn xuôi tới sát giới hạn model, đúng thiết kế.

**Thời gian build index** (`ollama serve` đang chạy, chú thích ảnh BẬT)

| Giai đoạn | Thời gian |
|---|---|
| Đọc tài liệu + trích ảnh + OCR + chú thích 1.707 ảnh bằng vision | 3.649 s (61 phút) |
| Chia chunk | 13 s |
| Encode 9.285 chunk (CPU) | 528 s (9 phút) |
| **Tổng** | **4.190 s (70 phút)** |

---

## 3. Tham số hệ thống

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-base` | 768 chiều, đa ngôn ngữ |
| `CHUNK_SIZE_TOKENS` | 160 | đo bằng tokenizer thật của model |
| `CHUNK_OVERLAP_TOKENS` | 32 | |
| `TOP_K` | **4** | hạ từ 6 — xem §7.2 |
| `HE_SO_OVER_FETCH` / `SO_UNG_VIEN_TOI_THIEU` | 10 / 60 | số ứng viên thô |
| `TRONG_SO_BM25` | 0.0 | BM25 TẮT ở vai trò xếp hạng |
| `SO_UNG_VIEN_BM25_CUU_HO` | 10 | BM25 chỉ bơm ứng viên (recall-only) |
| `RRF_K` | 60 | |
| `NGAN_SACH_KY_TU_MOI_DOAN` | 1600 | |
| `MO_RONG_QUA_RANH_GIOI_TRANG` | BẬT | |
| `SO_TRANG_TOI_DA_MO_RONG` | 1 | |
| `SO_DOAN_TOI_DA_MOI_TRANG` | 2 (thích ứng) | chỉ áp khi ứng viên trải ≥ `TOP_K` trang |
| `SO_UNG_VIEN_XET_DA_DANG_TRANG` | 20 | |
| `SO_DOAN_ANH_TOI_DA` | 1 | |
| `BAT_RERANK` / `SO_UNG_VIEN_RERANK` | BẬT / 30 | `BAAI/bge-reranker-v2-m3` |
| `NGUONG_DIEM_TOI_THIEU` | 0.50 | sàn tuyệt đối, chỉ chặn rác |
| `TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT` | 0.78 | ngưỡng tương đối |
| `NGUONG_DIEM_RERANK_TOI_THIEU` | 0.001 | ngưỡng từ chối câu lạc đề |
| `OLLAMA_MODEL` | `qwen3:4b` | |
| `OLLAMA_TEMPERATURE` | 0.1 | |
| `OLLAMA_NUM_PREDICT` | 12000 | |
| **`OLLAMA_NUM_CTX`** | **16384** | trước đây bỏ trống → Ollama cấp 4096 |
| `OLLAMA_NUM_CTX_TOI_DA` | 32768 | trần khi nới động |
| `OLLAMA_DU_PHONG_TOKEN_SINH` | 4000 | |
| `BAT_TRICH_ANH` / `BAT_CHU_THICH_ANH` | BẬT / BẬT | `qwen2.5vl:3b` |

---

## 4. Kết quả truy xuất trên toàn bộ 26 tài liệu

Chế độ `--nhanh`: chỉ đo truy xuất, **không gọi LLM**, nên **hoàn toàn tất định** — chạy lại
bao nhiêu lần cũng ra đúng con số này. Đây là phần số liệu đáng tin nhất để đưa vào báo cáo.

```bash
python evaluation/run_evaluation.py --khoang-cach
```

### 4.1 Bộ IN-SAMPLE (29 câu — 25 câu có đáp án + 4 câu lạc đề)

| Chỉ số | Giá trị |
|---|---|
| Precision@K | **0,620** |
| Recall@K | **0,845** |
| MRR | **0,980** |
| Đoạn đúng ở HẠNG 1 | **24 / 25 câu (96%)** |
| Câu lạc đề bị chặn ngay ở tầng truy xuất | 2 / 4 (2 câu còn lại do LLM từ chối) |
| Độ trễ truy xuất trung bình | 10,97 s/câu |

**Tách theo loại tài liệu**

| Loại tài liệu | Số câu | P@K | R@K | Giây |
|---|---:|---:|---:|---:|
| Biểu mẫu có bảng (DOCX) | 4 | 0,44 | 1,00 | 18,4 |
| Sách dài tiếng Anh (Bishop, 729 trang) | 3 | 0,50 | 1,00 | 6,9 |
| Giáo trình dài tiếng Việt (230 trang) | 6 | 0,62 | 0,83 | 6,4 |
| Slide tiếng Anh | 8 | 0,75 | 0,80 | 14,4 |
| Slide tiếng Việt | 4 | 0,62 | 0,69 | 6,4 |

### 4.2 Bộ HELD-OUT (46 câu — 44 câu có đáp án + 2 câu lạc đề)

**12 tài liệu chưa từng dùng để hiệu chỉnh bất kỳ tham số nào**: `CV-05-Classification.pdf`,
`BAO_CAO_MAY_HOC.docx`, `Chapter 2. Server kết nối vạn vật.pptx`, `baocaonangcaothayAn.docx`,
6 bài giảng Khai phá dữ liệu `Bai1`–`Bai6`, `PaperQA.pdf`, `RFRAG.pdf`.

| Chỉ số | Giá trị |
|---|---|
| Precision@K | **0,364** |
| Recall@K | **0,905** |
| MRR | **0,843** |
| Đoạn đúng ở HẠNG 1 | **33 / 44 câu (75%)** |
| Câu lạc đề bị chặn ngay ở tầng truy xuất | 1 / 2 |
| Độ trễ truy xuất trung bình | 10,22 s/câu |

**Tách theo loại tài liệu**

| Loại tài liệu | Số câu | P@K | R@K | Giây |
|---|---:|---:|---:|---:|
| Bài giảng Khai phá dữ liệu (DOCX) | 17 | 0,35 | 0,88 | 10,1 |
| Báo cáo dài tiếng Việt (DOCX) | 9 | 0,36 | 1,00 | 13,8 |
| Bài báo khoa học tiếng Anh (PDF) | 7 | 0,36 | 1,00 | 6,3 |
| Slide tiếng Anh | 6 | 0,46 | 0,92 | 7,9 |
| Slide tiếng Việt | 5 | 0,30 | 0,67 | 9,0 |

> **Recall@K của nhóm DOCX suy biến, phải nói ra khi trích số.** `Bai*.docx` và các báo cáo
> DOCX gần như không có ngắt trang (`Bai5` chỉ 1 "trang", `Bai2` có 2), mà P@K/R@K so khớp
> theo `(nguồn, trang)` — nên lấy về **bất kỳ** chunk nào của chúng cũng cho Recall@K = 1,00
> dù chunk đó có chứa câu trả lời hay không. Con số 1,00 ở hai nhóm DOCX vì thế **không**
> chứng minh hệ thống tìm đúng; nhóm PDF/PPTX (có ngắt trang thật) mới là nhóm đọc được.
> Với DOCX phải nhìn Citation accuracy ở §5 thay thế. Đây là lý do mỗi tài liệu ít trang chỉ
> được đặt 1–2 câu, thay vì nhồi cho đủ số.

### 4.2.1 Mở rộng bộ held-out từ 22 lên 46 câu

Bộ ban đầu chỉ có 20 câu có đáp án — quá nhỏ để khoảng cách đo được tách khỏi nhiễu. Thêm 24
câu trên 8 tài liệu mới (6 bài giảng Khai phá dữ liệu + 2 bài báo RAG) nâng lên 44 câu có
đáp án. Kết quả **không** đổi chiều, mà đổi độ chắc chắn:

| Chỉ số | Bộ 20 câu | Bộ 44 câu | |
|---|---:|---:|---|
| Precision@K | 0,375 | 0,364 | gần như không đổi |
| Recall@K | 0,892 | 0,905 | gần như không đổi |
| **MRR** | 0,879 | **0,843** | thấp hơn |
| Đoạn đúng ở hạng 1 | 16/20 (80%) | 33/44 (**75%**) | thấp hơn |
| **Khoảng cách MRR so với in-sample** | +0,101 | **+0,137** | **rõ hơn** |

Gấp đôi số câu mà kết luận giữ nguyên chiều và **mạnh lên**: khoảng cách MRR nới từ +0,101
lên +0,137. Nếu tín hiệu overfit ban đầu chỉ là nhiễu của 20 câu thì nó phải co lại khi thêm
mẫu, chứ không nở ra.

### 4.3 Khoảng cách IN-SAMPLE vs HELD-OUT — mức overfit của hệ thống

| Metric | in-sample (25 câu) | held-out (44 câu) | khoảng cách |
|---|---:|---:|---:|
| Precision@K | 0,620 | 0,364 | +0,256 |
| **Recall@K** | 0,845 | **0,905** | **−0,060** |
| **MRR** | 0,980 | **0,843** | **+0,137** |
| Đoạn đúng ở hạng 1 | 24/25 (96%) | 33/44 (75%) | −21 điểm % |

**Đây là con số quan trọng nhất của cả báo cáo**, và nó chia làm hai phần ngược nhau:

- **Khả năng TÌM ĐÚNG nội dung KHÔNG overfit.** Recall@K trên tài liệu chưa từng thấy còn
  **cao hơn** trên tài liệu đã dùng để hiệu chỉnh (0,905 so với 0,845). Nghĩa là chunking,
  embedding và tầng truy hồi tổng quát tốt sang tài liệu mới. (Có một phần do suy biến DOCX
  ở §4.2 — nhưng nhóm PDF/PPTX của bộ held-out cũng đạt 0,92–1,00, nên kết luận vẫn đứng.)
- **Khả năng XẾP ĐÚNG THỨ TỰ thì CÓ overfit.** MRR tụt 0,980 → 0,843 và tỷ lệ đoạn đúng ở
  hạng 1 rơi từ 96% xuống 75%. Đây đúng là tầng mà mọi ngưỡng của hệ thống tác động vào
  (rerank + các ngưỡng lọc). Nội dung đúng vẫn vào được ngữ cảnh, chỉ là không còn chắc chắn
  nằm ở đoạn `[1]` — đoạn mà LLM đọc kỹ nhất.

> **Precision@K KHÔNG so được giữa hai bộ.** P@K phụ thuộc trực tiếp vào số trang đúng mỗi
> câu: bộ in-sample trung bình **2,84** trang đúng/câu, bộ held-out chỉ **1,20**. Với
> `TOP_K=4`, câu chỉ có 1 trang đúng thì P@K trần đã là 0,25 bất kể hệ thống tốt đến đâu.
> Chênh lệch +0,256 vì thế phần lớn là **hiện vật của cách ra đề**, không phải bằng chứng
> overfit. Đưa con số này vào báo cáo thì phải kèm câu giải thích này.

> **Quy tắc bắt buộc:** không bao giờ chỉnh tham số theo kết quả của bộ held-out. Chỉnh một
> lần là nó trở thành bộ in-sample thứ hai và con số này mất sạch ý nghĩa.

### 4.4 Độ bền khi corpus lớn thêm (18 → 26 tài liệu)

Bộ câu hỏi giữ nguyên, chỉ thêm 8 tài liệu (6 bài giảng khai phá dữ liệu tiếng Việt +
`PaperQA.pdf` + `RFRAG.pdf`) — tức thêm **1.050 chunk và 548 ảnh làm nhiễu**, trong đó các
bài giảng về gom cụm / phân lớp / cây quyết định là nhiễu **rất sát chủ đề** với nhóm câu hỏi
máy học trên Bishop.

> Bảng dưới đo bằng **bộ held-out 22 câu** (bản trước khi mở rộng ở §4.2.1), vì mục đích là
> so hai kích thước corpus với nhau — phải giữ nguyên bộ câu hỏi thì phép so mới hợp lệ.

| Bộ | Chỉ số | 18 tài liệu (8.235 chunk) | 26 tài liệu (9.285 chunk) | Thay đổi |
|---|---|---:|---:|---:|
| in-sample | Precision@K | 0,630 | 0,620 | −0,010 |
| in-sample | Recall@K | 0,859 | 0,845 | −0,014 |
| in-sample | MRR | 0,980 | 0,980 | 0,000 |
| in-sample | hạng 1 | 24/25 | 24/25 | không đổi |
| held-out | Precision@K | 0,375 | 0,375 | **0,000** |
| held-out | Recall@K | 0,892 | 0,892 | **0,000** |
| held-out | MRR | 0,879 | 0,879 | **0,000** |
| held-out | hạng 1 | 16/20 | 16/20 | không đổi |

Corpus tăng **12,8%** về số chunk và thêm nhiễu sát chủ đề, nhưng chất lượng truy xuất gần
như **không đổi**: in-sample chỉ nhích xuống 0,01–0,014, held-out **giống hệt đến từng chữ số**.
Đây là bằng chứng trực tiếp cho thấy tầng truy xuất không xuống cấp khi quy mô tăng — một
điểm đáng đưa vào báo cáo bên cạnh các con số tuyệt đối.

---

## 5. Kết quả đánh giá đầy đủ (có gọi LLM)

```bash
python evaluation/run_evaluation.py
```

29 câu bộ in-sample trên corpus 26 tài liệu. Mỗi câu tốn 5+ lượt gọi LLM (1 sinh câu trả lời
+ 3 lượt chấm Faithfulness lấy trung vị + 1 lượt chấm Relevance + 1 lượt cho MỖI ý có trích
dẫn). Tổng cộng **97 cặp trích dẫn** được kiểm.

| Nhóm | Số câu | P@K | R@K | Faithfulness | Answer Relevance | Citation | Giây/câu |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Câu CÓ đáp án** | 25 | **0,620** | **0,845** | **0,980** | **1,000** | **0,714** | 45,9 |
| Câu từ chối (lạc đề) | 4 | 0,000 | 0,000 | 0,750 | 0,250 | — | 16,3 |
| Toàn bộ | 29 | 0,534 | 0,729 | 0,948 | 0,897 | 0,714 | 41,8 |

> **Đọc hàng "Toàn bộ" cho đúng:** 4 câu lạc đề luôn cho P@K/R@K = 0 *theo định nghĩa* (chúng
> không có trang đúng nào), và Answer Relevance thấp vì câu từ chối không cung cấp nội dung
> cụ thể — đây là quy ước bình thường của metric, không phải lỗi hệ thống. **Hàng "Câu CÓ
> đáp án" mới là con số phản ánh chất lượng.**

**Tách theo loại tài liệu**

| Loại tài liệu | Số câu | P@K | R@K | Faith | Relev | Trích | Giây |
|---|---:|---:|---:|---:|---:|---:|---:|
| Biểu mẫu có bảng (DOCX) | 4 | 0,44 | 1,00 | 0,88 | 1,00 | 0,75 | 47,8 |
| Sách dài tiếng Anh (Bishop) | 3 | 0,50 | 1,00 | 1,00 | 1,00 | 0,48 | 40,7 |
| Giáo trình dài tiếng Việt | 6 | 0,62 | 0,83 | 1,00 | 1,00 | 0,78 | 42,2 |
| Slide tiếng Anh | 8 | 0,75 | 0,80 | 1,00 | 1,00 | 0,71 | 49,6 |
| Slide tiếng Việt | 4 | 0,62 | 0,69 | 1,00 | 1,00 | 0,77 | 46,0 |
| Ngoài phạm vi (câu lạc đề) | 4 | 0,00 | 0,00 | 0,75 | 0,25 | — | 16,3 |

**Tách theo loại câu hỏi**

| Loại câu hỏi | Số câu | P@K | R@K | Faith | Relev | Trích | Giây |
|---|---:|---:|---:|---:|---:|---:|---:|
| Truy xuất thường | 14 | 0,66 | 0,86 | 1,00 | 1,00 | 0,65 | 44,3 |
| Chéo ngôn ngữ | 5 | 0,70 | 0,72 | 1,00 | 1,00 | 0,80 | 59,0 |
| Đọc bảng | 3 | 0,42 | 1,00 | 0,83 | 1,00 | 1,00 | 47,3 |
| Kiểm chứng khẳng định sai | 3 | 0,50 | 0,83 | 1,00 | 1,00 | 0,58 | 30,1 |
| Từ chối câu lạc đề | 4 | 0,00 | 0,00 | 0,75 | 0,25 | — | 16,3 |

**Độ dài câu trả lời:** trung bình **467 ký tự**, ngắn nhất 70, dài nhất 1.153 — không còn
câu nào cụt ngủn kiểu bị cắt giữa chừng như trước khi sửa `num_ctx` (§6).

**Độ tin cậy của chính thước đo**

| Chỉ số | in-sample | held-out |
|---|---:|---:|
| Câu bị cờ "Faithfulness đáng ngờ" tự đánh dấu | **0 / 29** | **0 / 46** |
| Câu giám khảo cho điểm khác nhau giữa 3 lần chấm | 2 / 29 | 4 / 46 |

Không câu nào rơi vào tình trạng "giám khảo chấm thấp trong khi câu trả lời chép gần nguyên
văn ngữ cảnh" — tức không có điểm Faithfulness nào đáng nghi ngờ trong cả hai lần chạy.

**Kiểm định riêng cho giám khảo** (`python evaluation/kiem_dinh_judge.py --so-lan 3`) — chạy
`qwen3:4b` trên 7 ca đã biết trước đáp án, mỗi ca chấm 3 lần:

| Chỉ số | Kết quả |
|---|---|
| Chấm đúng khoảng kỳ vọng | **7 / 7 ca (100%)** |
| Sai lệch trung bình khi lệch khoảng | 0,000 |
| Dao động lớn nhất giữa 3 lần chấm cùng một ca | **0,00** |
| Ca hồi quy "ngữ cảnh bị dính chữ" | xử lý đúng |

7 ca gồm cả hai chiều: 3 ca **phải cho điểm cao** (bám sát ngữ cảnh sạch, bám sát nhưng ngữ
cảnh bị dính chữ, diễn đạt lại bằng lời khác) và 3 ca **phải cho điểm thấp** (đúng một nửa
kèm ý bịa, bịa hoàn toàn, nói ngược ngữ cảnh) — bắt được ca hỏng khó hơn nhiều so với khen ca
đúng, nên phải có cả hai chiều thì con số 7/7 mới có nghĩa.

> **Đọc con số này cho đúng, vì nó KHÔNG mâu thuẫn với "2/29 và 4/46 câu dao động" ở trên.**
> 7 ca kiểm định được dựng cho **rõ ràng dứt khoát** — bịa thì hẳn là bịa, đúng thì hẳn là
> đúng — và trên đó giám khảo lặp lại y hệt (dao động 0,00). Các câu dao động trong lần chạy
> thật là những câu **thực sự nằm ở ranh giới**, nơi 0,0 hay 1,0 đều biện hộ được. Nghĩa là:
> giám khảo không hỏng, nhưng nó cũng không phân xử được ca mập mờ — nên chênh lệch nhỏ ở §5
> vẫn phải đọc dè dặt, đúng như §5.3 đã chỉ ra bằng một ví dụ cụ thể.
>
> Cách phát biểu đúng khi đưa vào báo cáo: *"Faithfulness 0,980, đo bằng một thước đo đã kiểm
> định đúng 7/7 ca với dao động 0,00"* — chứ không phải *"Faithfulness 0,980"* trơ trọi.

### 5.1 So với lần chạy trước — và vì sao KHÔNG được đọc như một phép A/B

Script tự in bảng chênh lệch với `ket_qua_danh_gia_truoc.csv`:

| Metric | Trước | Sau | Chênh lệch |
|---|---:|---:|---:|
| precision_at_k | 0,44 | 0,53 | **+0,10** |
| recall_at_k | 0,80 | 0,73 | **−0,07** |
| faithfulness | 0,86 | **0,95** | **+0,09** |
| answer_relevance | 0,90 | 0,90 | 0,00 |
| citation_accuracy | 0,61 | **0,71** | **+0,11** |
| do_tre_giay | 38,7 | 41,8 | +3,1 |

> **Bảng này bị nhiễu bởi BA thay đổi cùng lúc, không phải một.** Lần chạy trước dùng corpus
> **13 tài liệu**, `TOP_K=6`, và chưa có các bản sửa; lần này là corpus **26 tài liệu**,
> `TOP_K=4`, đã sửa. Không tách được phần nào của +0,09 Faithfulness là do sửa `num_ctx`,
> phần nào do đổi corpus, phần nào do hạ `TOP_K`. Muốn quy kết nguyên nhân thì phải chạy A/B
> đúng cách như §7 — giữ nguyên mọi thứ, chỉ đổi một tham số.
>
> Điều **có thể** nói: chiều của Faithfulness (+0,09) và Citation accuracy (+0,11) khớp với
> điều dự đoán từ việc sửa `num_ctx` (câu trả lời không còn bị cắt giữa chừng nên đầy đủ hơn
> và dẫn nguồn đủ hơn), còn Recall@K giảm khớp với việc hạ `TOP_K` (§7.2). Nhưng đó là
> **nhất quán**, chưa phải **bằng chứng**.

### 5.2 Bộ HELD-OUT, đánh giá đầy đủ (46 câu)

```bash
python evaluation/run_evaluation.py --held-out    # -> ket_qua_danh_gia_held_out.csv
```

| Nhóm | Số câu | P@K | R@K | Faithfulness | Answer Relevance | Citation | Giây/câu |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Câu CÓ đáp án** | 44 | **0,364** | **0,905** | **0,977** | **0,977** | **0,784** | 44,8 |
| Câu từ chối (lạc đề) | 2 | 0,000 | 0,000 | 1,000 | 0,500 | — | 24,4 |
| Toàn bộ | 46 | 0,348 | 0,866 | 0,978 | 0,957 | 0,784 | 43,9 |

**Tách theo loại tài liệu**

| Loại tài liệu | Số câu | P@K | R@K | Faith | Relev | Trích | Giây |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bài giảng Khai phá dữ liệu (DOCX) | 17 | 0,35 | 0,88 | 1,00 | 0,94 | 0,75 | 42,5 |
| Báo cáo dài tiếng Việt (DOCX) | 9 | 0,36 | 1,00 | 1,00 | 1,00 | 0,89 | 53,5 |
| Bài báo khoa học tiếng Anh (PDF) | 7 | 0,36 | 1,00 | 0,86 | 1,00 | 0,61 | 46,2 |
| Slide tiếng Anh | 6 | 0,46 | 0,92 | 1,00 | 1,00 | 0,96 | 38,8 |
| Slide tiếng Việt | 5 | 0,30 | 0,67 | 1,00 | 1,00 | 0,71 | 42,3 |
| Ngoài phạm vi (câu lạc đề) | 2 | 0,00 | 0,00 | 1,00 | 0,50 | — | 24,4 |

**Tách theo loại câu hỏi**

| Loại câu hỏi | Số câu | P@K | R@K | Faith | Relev | Trích | Giây |
|---|---:|---:|---:|---:|---:|---:|---:|
| Truy xuất thường | 32 | 0,37 | 0,90 | 0,97 | 0,97 | 0,79 | 46,5 |
| Chéo ngôn ngữ | 6 | 0,33 | 0,83 | 1,00 | 1,00 | 0,75 | 42,9 |
| Đọc bảng | 4 | 0,38 | 1,00 | 1,00 | 1,00 | 0,67 | 40,4 |
| Kiểm chứng khẳng định sai | 2 | 0,38 | 1,00 | 1,00 | 1,00 | 1,00 | 31,1 |
| Từ chối câu lạc đề | 2 | 0,00 | 0,00 | 1,00 | 0,50 | — | 24,4 |

**Nhóm yếu nhất là bài báo khoa học tiếng Anh:** Faithfulness 0,86 và Citation 0,61, thấp hơn
hẳn phần còn lại. Đây là loại tài liệu dày thuật ngữ, nhiều bảng số và nhiều tham chiếu chéo
`[19]`, `[50]` — mà chính hệ thống cũng dùng dấu ngoặc vuông để đánh số đoạn trích, nên đây là
chỗ đáng nghi ngờ đầu tiên nếu muốn đào tiếp.

### 5.3 Overfit của CHẤT LƯỢNG CÂU TRẢ LỜI — và một kết luận bị chính số liệu lật lại

| Chỉ số | in-sample (25 câu) | held-out (44 câu) | khoảng cách |
|---|---:|---:|---:|
| **Faithfulness** | 0,980 | **0,977** | **+0,003** |
| Answer Relevance | 1,000 | 0,977 | +0,023 |
| **Citation accuracy** | 0,714 | **0,784** | **−0,070** |
| Độ trễ | 45,9 s | 44,8 s | −1,1 s |
| Độ dài câu trả lời | 467 ký tự | 206 ký tự | — |
| Cặp trích dẫn đã kiểm | 97 | 129 | — |
| Câu bị cờ "Faithfulness đáng ngờ" | 0 / 29 | 0 / 46 | — |

**Chất lượng câu trả lời KHÔNG overfit.** Faithfulness chênh 0,003 — bằng không trên thực tế.
Citation accuracy thậm chí **cao hơn** trên tài liệu chưa từng thấy.

**Nhưng đây là kết luận NGƯỢC với chính bản báo cáo này ở lần đo trước, và lý do đáng ghi lại
hơn cả kết luận.** Đo trên bộ held-out 22 câu, Faithfulness ra 0,825 → khoảng cách **+0,155**,
và kết luận rút ra khi đó là "chất lượng câu trả lời có overfit". Kèm theo nó là một dòng dè
dặt: *"bộ held-out chỉ có 20 câu, giám khảo dao động ở 2/22 câu, một câu lật điểm là trung
bình đổi 0,05 — khoảng cách 0,155 tương đương khoảng 3 câu, đủ lớn để đáng chú ý, chưa đủ lớn
để coi là chắc chắn"*.

Nhân đôi số câu cho câu trả lời dứt khoát: **+0,155 co về +0,003**. Dòng dè dặt đó đúng, và
nếu bỏ qua nó thì báo cáo đã kết luận sai về chính hệ thống của mình.

**Đối chiếu hai tín hiệu khi tăng mẫu — đây mới là phần đáng đưa vào báo cáo:**

| Tín hiệu | Bộ 20-22 câu | Bộ 44-46 câu | Diễn giải |
|---|---:|---:|---|
| Khoảng cách **MRR** (tất định) | +0,101 | **+0,137** | **nở ra** → tín hiệu THẬT |
| Khoảng cách **Faithfulness** (có LLM chấm) | +0,155 | **+0,003** | **co lại** → là NHIỄU |

Thêm mẫu làm tín hiệu thật rõ hơn và làm nhiễu biến mất. Đó chính là cách phân biệt hai thứ,
và nó chỉ làm được khi bộ đo đủ lớn — một lý do cụ thể để bỏ công mở rộng bộ held-out thay vì
chấp nhận con số đầu tiên đo được.

> **Citation accuracy cao hơn trên held-out (0,784 so với 0,714) vẫn cần đọc dè dặt** — không
> phải vì nhiễu, mà vì **cách ra đề**: câu held-out phần lớn hỏi một dữ kiện nên câu trả lời
> ngắn hơn 2,3 lần (206 so với 467 ký tự), ít ý cần dẫn nguồn hơn thì ít chỗ dẫn sai hơn.

> **So sánh với lần chạy trước ở bộ held-out không dùng được lần này:** script đã tự cảnh báo
> *"chỉ 22 câu hỏi trùng nhau giữa 2 lần chạy (mới 46, cũ 22)"*. Đúng cơ chế chặn đã dựng sẵn
> để không ai đem hai bộ câu hỏi khác nhau ra so rồi kết luận nhầm.

---

## 6. Ngân sách token và độ trễ sinh câu trả lời

Đo bằng `prompt_eval_count` / `eval_count` **thật do Ollama trả về**, không phải ước lượng.

| Loại câu hỏi | TOP_K | Ký tự prompt | Token prompt | Ký tự/token | Token sinh ra | **TỔNG token** | `done_reason` | Giây |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Thường | 6 | 10.652 | 3.394 | 3,14 | 3.607 | **7.001** | `stop` | 71,8 |
| Thường | 4 | 7.509 | 2.430 | 3,09 | 2.775 | **5.205** | `stop` | 59,3 |
| "Liệt kê ĐẦY ĐỦ…" | 6 | 10.681 | 3.646 | 2,93 | 7.214 | **10.860** | `stop` | 112,0 |
| "Liệt kê ĐẦY ĐỦ…" | 4 | 7.790 | 2.660 | 2,93 | 5.498 | **8.158** | `stop` | 89,3 |

**Ba kết luận rút ra:**

1. **Cửa sổ 4096 mặc định của Ollama là quá nhỏ, xác nhận bằng số đo.** Cửa sổ chứa
   *prompt + thinking + câu trả lời*. Cột TỔNG cho thấy mọi câu đều vượt 4096 rất xa — tới
   **10.860 token (gấp 2,7 lần)** với câu yêu cầu liệt kê đầy đủ. Trước khi khai báo
   `OLLAMA_NUM_CTX`, mọi lượt hỏi như thế đều bị cắt giữa chừng, **không có lỗi nào báo ra**.
2. **Tokenizer Qwen đạt 2,93–3,14 ký tự/token với tiếng Việt.** Đây là số đo thật; ước lượng
   ban đầu (2,5 ký tự/token) sai ~20% về phía cao. Sai số đó không đổi kết luận nhưng đổi
   *cơ chế*: trên corpus này thủ phạm là cắt phần **sinh**, không phải cắt **prompt**.
3. **Hạ `TOP_K` 6→4 cắt được ~26% tổng token và ~17–20% thời gian** (71,8→59,3 s và
   112,0→89,3 s), vì prompt ngắn hơn kéo theo phần suy luận cũng ngắn hơn.

---

## 7. Đo A/B từng thay đổi

Bật lần lượt từng thay đổi trên **cùng một index** (12 tài liệu, 5.554 chunk, `TOP_K=6`,
rerank BẬT, không bật chú thích ảnh). Chế độ tất định.

> Index dùng ở §6 khác index ở §4 (ít tài liệu hơn, không có ảnh) nên **giá trị tuyệt đối
> không so trực tiếp với §4 được**. Mục đích ở đây là so **chênh lệch giữa các cấu hình**
> trên cùng một điều kiện, và điều đó thì hợp lệ.

| Cấu hình | P@K | Recall@K | MRR | hạng 1 | chặn lạc đề |
|---|---:|---:|---:|---:|---:|
| GỐC (trước khi sửa) | 0,500 | 0,937 | 0,980 | 24/25 | 2/4 |
| + ngưỡng tương đối | 0,500 | 0,937 | 0,980 | 24/25 | 2/4 |
| + mở rộng xuyên trang | 0,467 | 0,875 | 0,980 | 24/25 | 2/4 |
| + trần trang thích ứng | **0,567** | 0,937 | 0,980 | 24/25 | 2/4 |
| + BM25 cứu hộ | 0,507 | **0,945** | 0,980 | 24/25 | 2/4 |

- **Ngưỡng tương đối (0.78): không đổi con số nào** — đúng thiết kế. 0,78 = 0,70/0,90 tái tạo
  chính điểm hiệu chỉnh cũ, nên trên corpus cũ nó *phải* trung tính. Giá trị của nó nằm ở
  corpus có phân bố cosine khác, và điều đó **chưa được chứng minh** ở đây.
- **Trần trang thích ứng: +0,067 P@K**, không đụng MRR lẫn khả năng chặn câu lạc đề.
- **BM25 cứu hộ: +0,008 Recall@K.** Nhỏ đúng như dự đoán; điều đáng giá hơn là nó **không
  làm hỏng gì**, kể cả truy xuất chéo ngôn ngữ vốn là chỗ BM25 từng gây hại nặng.

### 7.1 Mở rộng xuyên trang: khi chính thước đo là thứ sai

Recall@K tụt 0,062 — đáng ra phải bỏ. Nhưng `recall_tai_k()` chỉ đếm **trang neo**: khi một
đoạn trích nuốt sang trang liền kề, trang đó không còn được neo riêng, **dù nội dung vẫn nằm
nguyên trong ngữ cảnh gửi cho LLM**. Đo thêm bằng *Recall phủ* (trang đúng có nằm trong các
trang mà đoạn trích **đi qua** không):

| | P@K | Recall@K (neo) | **Recall phủ** |
|---|---:|---:|---:|
| TẮT mở rộng xuyên trang | 0,580 | 0,945 | 0,945 |
| BẬT mở rộng xuyên trang | 0,540 | 0,875 | **0,959** |

Chi tiết từng câu: **mọi câu bị tụt Recall neo đều giữ Recall phủ = 1,00**; ngữ cảnh phủ
10–16 trang thay vì đúng 6. Không câu nào mất nội dung.

> **Hệ quả phải ghi vào báo cáo:** Recall@K trước và sau thay đổi này **không so trực tiếp
> được nữa**, vì mở rộng xuyên trang phá vỡ giả định "mỗi đoạn trích nằm gọn trong một trang"
> mà metric dựa vào. Một thay đổi làm hỏng giả định của metric sẽ luôn trông như hồi quy, kể
> cả khi nó là cải thiện.

### 7.2 Hạ `TOP_K` 6 → 4

Đo trên **cả hai** bộ câu hỏi (12 tài liệu, 5.554 chunk, bộ held-out bản 22 câu, chỉ đổi
`TOP_K`):

| Bộ | TOP_K | P@K | Recall@K | Recall phủ | MRR | hạng 1 | Token prompt |
|---|---:|---:|---:|---:|---:|---:|---:|
| in-sample | 6 | 0,540 | 0,875 | 0,959 | 0,980 | 24/25 | 4.401 |
| in-sample | 4 | **0,650** | 0,835 | 0,933 | 0,980 | 24/25 | **3.277** |
| held-out | 6 | 0,225 | 0,850 | 1,000 | 0,858 | 16/20 | 4.520 |
| held-out | 4 | **0,287** | 0,800 | 0,958 | 0,858 | 16/20 | **3.332** |

Nhất quán trên cả hai bộ:

- Prompt ngắn ~25% → đây là thứ mua được tốc độ.
- **P@K tăng** (+0,110 / +0,062): 2 đoạn bị cắt phần lớn là đoạn kém liên quan.
- **MRR và số câu có đoạn đúng ở hạng 1 KHÔNG đổi** — thứ tự xếp hạng không xấu đi.
- **Cái mất thật:** Recall@K −0,04, Recall phủ −0,026 / −0,042. Một số câu cần tổng hợp từ
  nhiều trang sẽ thiếu một mảnh. Đây là **đánh đổi**, không phải cải tiến thuần.

Đặt lại `TOP_K=6` trong `.env` nếu ưu tiên độ chính xác hơn tốc độ.

---

### 7.3 Tối ưu luồng Ingestion — trước và sau

Đo bằng cách chạy **cùng một corpus qua hai phiên bản code** (bản trước tối ưu lấy từ
`git worktree` ở commit gốc, bản sau là code hiện tại), rồi so **cả thời gian lẫn từng bản
ghi sinh ra**. So đầu ra là phần bắt buộc: một tối ưu ingestion làm nội dung index xấu đi là
một tối ưu đã thất bại, dù nó nhanh tới đâu — và khác biệt kiểu đó không thể phát hiện bằng
cách đọc lại code.

Cấu hình khi đo: `BAT_CHU_THICH_ANH=0`, `BAT_CACHE_INGESTION=0` (để đo **chi phí thật của lần
đọc đầu tiên**, không lẫn phần cache trúng).

**Thời gian đọc, cùng nội dung đầu vào**

Bishop được đo **3 lần mỗi phiên bản** vì lần đo đơn lẻ đầu tiên cho 47,6 s ở bản mới rồi
55,4 s ở một lần khác — chênh lệch 16% giữa hai lần chạy cùng một code, tức đủ lớn để nuốt
trọn hiệu ứng cần đo. Ghi cả khoảng giá trị chứ không chỉ một con số:

| Tài liệu | Trước | Sau | Chênh (trung vị) |
|---|---:|---:|---:|
| Bishop — trung vị 3 lần | 58,8 s | **48,3 s** | **−17,9%** |
| Bishop — khoảng (min–max) | 53,8–59,9 s | **48,0–50,7 s** | hai khoảng **không chồng lấn** |
| 8 tài liệu trộn (4 PDF + 2 DOCX + 2 PPTX), 1 lần | 6,62 s | **6,16 s** | −7,0% |

Hai khoảng không chồng lấn (lần chậm nhất của bản mới vẫn nhanh hơn lần nhanh nhất của bản
cũ) là căn cứ để nói chênh lệch này có thật chứ không phải nhiễu.

Khoản tiết kiệm đến từ ba chỗ, không chỗ nào là "làm ít việc hơn": bỏ lượt duyệt PDF thứ hai
(trước đây `trich_anh_pdf` mở lại cả file và gọi `extract_text()` lần nữa cho từng trang), bỏ
lượt `bang.extract()` thừa cho mỗi bảng, và hiệu chỉnh `x_tolerance` theo tài liệu thay vì dò
lại 4 mức trên **mọi** trang dính chữ.

**Nội dung sinh ra: không mất gì, và tốt lên ở chỗ đo được**

| | Trước | Sau |
|---|---:|---:|
| Bản ghi từ Bishop | 809 | **809** |
| Bản ghi văn bản (8 tài liệu trộn) | 372 | **372** |
| Độ dính chữ trung bình mỗi trang (Bishop) | 0,0026 | **0,0026** |
| Số trang Bishop mà bản mới dính chữ **hơn** bản cũ | — | **0** |

540/728 trang Bishop CÓ NỘI DUNG (758 trang PDF, 30 trang rỗng hoặc mục lục bị loại) có khác
biệt, và mọi khác biệt đều nghiêng về phía bản mới —
phép hiệu chỉnh `x_tolerance` theo tài liệu chọn được mức tách từ tốt hơn:

```
CŨ : p(X = xi,Y = yj)      lnp(D|α,β)      where i = 1,...,D
MỚI: p(X = xi, Y = yj)     ln p(D|α, β)    where i = 1, . . . , D
```

Trên `PaperQA.pdf`, ba trang mà bản cũ để lọt chữ dính (`encoders[19,64],whicharetrained…`)
nay được tách đúng. Ghi lại vì đây **không phải mục tiêu** của đợt tối ưu — nó là hệ quả phụ
của việc đổi thứ tự thử tham số, và chỉ phát hiện được nhờ so từng bản ghi.

> **Một hồi quy đã suýt lọt.** Bản đầu của phép dừng sớm dùng chung ngưỡng
> `TY_LE_DINH_CHU_DE_DOC_LAI = 0,10` cho hai câu hỏi khác nhau: *"trang này có đáng đọc lại
> không"* và *"bản đọc lại đã đủ tốt chưa"*. Nó chấp nhận ngay mức đầu tiên hạ độ dính từ 30%
> xuống 9%, để lọt `RAGmodelsretrievetextfromacorpus` vào index. Sửa bằng ngưỡng riêng
> `TY_LE_DINH_CHU_DAT_YEU_CAU = 0,02`, lấy từ chính số đo cũ (PDF đọc tốt: 0,0–1,5%; trang
> hỏng: 41,7%). Chi tiết: ARCHITECTURE.md §5.66.

**Lọc ảnh: bớt vector rác, không đụng tới văn bản**

| Tài liệu | Bản ghi ảnh trước | sau | Bản ghi văn bản (trước → sau) |
|---|---:|---:|---:|
| Chapter 1. Introduction to IoT.pptx | 401 | **261** | 228 → 228 |
| Bai1-TongQuanDuLieu.docx | 219 | **168** | 7 → 7 |
| Bai3-GomCum.docx | 75 | **61** | 4 → 4 |
| NHÓM 6 - LUẬT HÌNH SỰ.pptx | 18 | **13** | 51 → 51 |
| 4 file PDF trong bộ đo | 21 | **21** | không đổi |

Cột cuối là điều quan trọng nhất: **không một bản ghi văn bản nào bị mất**. Phần bị loại là
icon, logo góc trang, dải trang trí và hình lặp lại kiểu watermark — mỗi cái trước đây tốn một
lượt render, một file trên đĩa, một lượt gọi model vision (~1,9 s) và một vector trong index.
PDF không giảm bản ghi ảnh nào, đúng như dự đoán: bộ lọc nhắm vào thứ mà slide và tài liệu
Word hay chèn, không phải hình trong PDF học thuật.

**Gọi OCR song song** — đo trên 8 trang Bishop thật sự cần OCR (trang 305–316), cùng một tập
trang, cache rỗng ở mỗi lần đo, `qwen2.5vl:3b` chạy trên GPU:

| Cấu hình | Tổng | Mỗi trang | So với tuần tự |
|---|---:|---:|---:|
| `SO_WORKER_VISION=1` (hành vi cũ) | 262,9 s | 32,9 s | — |
| `SO_WORKER_VISION=4` | **106,4 s** | **13,3 s** | **nhanh 2,47×** |
| Cache đầy (lần build thứ hai) | **0,07 s** | — | **~3.750×** |

Con số 32,9 giây/trang giải thích vì sao đây là bước đắt nhất của cả luồng: một cuốn sách có
vài trăm trang hỏng font sẽ mất hàng giờ ở bản cũ. Mức tăng tốc 2,47× (không phải 4×) là hợp
lý — mọi luồng đều xếp hàng qua **một** máy chủ Ollama, phần song song hoá được chỉ là thời
gian chờ mạng và phần nạp/xả giữa các lượt.

**Cache tài liệu — lần đọc thứ hai**

| | Lần 1 (cache rỗng) | Lần 2 (cache đầy) |
|---|---:|---:|
| `CV-01-Introduction.pdf` (14 trang, 8 ảnh) | 0,76 s | **0,01 s** |

Đây mới là con số nói đúng trải nghiệm hằng ngày: thêm một tài liệu vào corpus 25 file không
còn nghĩa là trả lại chi phí của 25 file kia. Cùng với index tăng dần, 25 tài liệu không đổi
được giữ nguyên vector và không đi qua bất kỳ bước nào của luồng Ingestion.

**Chưa đo được** (ghi ra để không ai tưởng đã đo đủ):

- **Chi phí lần build đầu của TOÀN BỘ corpus 26 tài liệu với OCR bật.** Phép đo này bị dừng
  giữa chừng vì riêng Bishop đã cần OCR hàng trăm trang ở 32,9 s/trang — tức nhiều giờ cho
  một lần đo. Các con số ở trên đo trên từng phần, và mức tiết kiệm của mỗi phần thì đo được
  riêng rẽ.
- **Ảnh hưởng của lọc ảnh lên Recall@K.** Việc bỏ 140 bản ghi logo *nên* cải thiện độ chính
  xác (bớt vector rác cạnh tranh suất TOP_K), nhưng chưa chạy lại `run_evaluation.py` trên
  index mới nên chưa được nói đó là một cải thiện.
- **Mức lợi thật của ngân sách thích ứng lúc truy vấn** (§5.67): số ứng viên rerank giảm
  30 → 12 cho câu hỏi đơn giản, nhưng chưa đo độ trễ trước/sau trên bộ câu hỏi thật.

---

## 8. Những gì chưa đo được

Ghi ra để không ai đọc báo cáo rồi tưởng đã đo đủ:

- **Thanh sai số cho Answer Relevance và Citation accuracy.** Faithfulness đã ổn định sẵn
  (chấm 3 lần lấy trung vị ở tầng metric, cộng kiểm định 7/7 ca ở §5), nhưng hai metric còn
  lại **chỉ được chấm một lần** mỗi câu. Chạy trọn bộ 3 lần rồi lấy trung vị sẽ cho thanh sai
  số cho chúng — khoảng 7 giờ.
  **Đánh giá: không đáng làm.** Kết luận chính của báo cáo nằm ở §4, vốn tất định; còn chỗ
  nhiễu duy nhất từng gây kết luận sai (§5.3) đã được chữa bằng cách mở rộng bộ đo, hiệu quả
  hơn hẳn việc lặp lại phép đo. Cái giá của việc bỏ qua: đừng viết câu nào dựa trên chênh lệch
  nhỏ ở hai metric đó — README đã có sẵn ngưỡng "chênh dưới ~0,1 của Citation accuracy không
  nên diễn giải là gì cả".
- **Quy kết nguyên nhân cho mức tăng Faithfulness/Citation ở §5.1** — ba biến đổi cùng lúc
  (corpus, `TOP_K`, các bản sửa) nên không tách được đóng góp của từng cái.
- **Vì sao nhóm bài báo khoa học tiếng Anh có Citation accuracy 0,61** (§5.2) — thấp hơn hẳn
  các nhóm khác; giả thuyết đầu tiên đáng kiểm là tham chiếu `[19]`, `[50]` trong bài báo lẫn
  với số hiệu đoạn trích `[1]`, `[2]` của hệ thống, nhưng chưa đo.
- **Lợi ích của ngưỡng tương đối** — thiết kế để trung tính trên corpus đã hiệu chỉnh, nên
  cần một corpus có phân bố cosine khác hẳn mới chứng minh được.
- **Cơ chế "prompt bị cắt mất đoạn `[1]`, `[2]`"** — suy luận từ tài liệu Ollama, chưa tái
  hiện được vì prompt trên corpus này mới 2.400–3.600 token, chưa chạm 4096.
- **Câu hỏi cho 8 tài liệu mới** (`Bai1`–`Bai6`, `PaperQA`, `RFRAG`) — chúng hiện chỉ đóng vai
  trò nhiễu trong index, chưa có câu hỏi có nhãn nào nhắm vào chúng.

---

## 9. Cách tái lập

```bash
# 1. Đưa tài liệu vào data/raw rồi build index (qua UI hoặc nút "Đọc tài liệu")
streamlit run app.py

# 2. Đo truy xuất - TẤT ĐỊNH, không gọi LLM, vài phút
python evaluation/run_evaluation.py --nhanh              # bộ in-sample
python evaluation/run_evaluation.py --nhanh --held-out   # bộ held-out
python evaluation/run_evaluation.py --khoang-cach        # cả hai + mức overfit

# 3. Đo đầy đủ - có gọi LLM, chậm (~80 phút cho bộ in-sample)
python evaluation/run_evaluation.py                      # -> ket_qua_danh_gia.csv
python evaluation/run_evaluation.py --held-out           # -> ket_qua_danh_gia_held_out.csv

# 4. Các script kiểm định từng cơ chế
python evaluation/kiem_dinh_judge.py                 # độ tin cậy của thước đo Faithfulness
python evaluation/kiem_dinh_doi_chieu.py --so-lan 3  # cơ chế phát hiện mâu thuẫn
python evaluation/kiem_dinh_viet_lai.py --chi-tang-1 # nhận diện câu hỏi nối tiếp
python evaluation/do_quy_mo_index.py                 # ngưỡng quy mô FAISS trên máy này
```

Bật `LOG_PHAN_BO_DIEM=1` trong `.env` để in phân bố điểm từng lượt truy xuất (cosine từng
đoạn, điểm rerank cao nhất, số đoạn sống sót sau mỗi tầng lọc).

---

## 10. Đối chiếu với tài liệu kiến trúc

| Nội dung | Mục |
|---|---|
| Bug `num_ctx` và cách sửa | ARCHITECTURE.md §5.60 |
| Ngưỡng tuyệt đối → ngưỡng tương đối | §5.61 |
| Ba giả định "đúng cho slide" sai cho PDF văn xuôi | §5.62 |
| Xung đột quy tắc 5 và 6 trong prompt | §5.63 |
| Bộ held-out và cách đọc khoảng cách | §5.64 |
| Khi chính thước đo là thứ sai | §5.65 |
| Cách diễn giải các metric không tất định | ARCHITECTURE.md §5.46 · README.md |
