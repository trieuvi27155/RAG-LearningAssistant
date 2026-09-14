# RAG — Hệ thống hỏi đáp tài liệu học tập tiếng Việt

Đồ án tốt nghiệp: hệ thống hỏi đáp thông minh trên tài liệu học tập (PDF/PPTX/DOCX) bằng kỹ
thuật RAG (Retrieval-Augmented Generation), chạy **hoàn toàn local** — không dùng API trả phí.

| Tài liệu | Nội dung |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Kiến trúc chi tiết, luồng dữ liệu, input/output từng module, và §7 — toàn bộ quyết định thiết kế kèm lý do |
| [KET_QUA_DO_DAC.md](KET_QUA_DO_DAC.md) | Nguồn số liệu chuẩn: môi trường đo, tham số, kết quả, cách tái lập |

---

## 1. Overview

Người dùng đưa tài liệu học tập của mình vào (giáo trình, slide bài giảng, báo cáo), hệ thống
đọc và lập chỉ mục, sau đó trả lời câu hỏi **dựa trên chính các tài liệu đó** kèm trích dẫn tới
tên file và số trang/slide.

Toàn bộ mô hình (embedding, rerank, LLM, vision) chạy trên máy người dùng qua
[Ollama](https://ollama.com) và `sentence-transformers` — tài liệu không rời khỏi máy và không
phát sinh chi phí API.

Hệ thống không chỉ trả lời: nó **từ chối khi không tìm thấy căn cứ**, **kiểm chứng khẳng định
sai** thay vì thuận theo giả định, và **cảnh báo khi hai tài liệu nói ngược nhau**.

---

## 2. Objectives

1. **Trả lời có căn cứ kiểm chứng được** — mỗi câu trả lời kèm nguồn tới tên file + trang/slide
   đủ để người đọc tự mở tài liệu gốc đối chiếu, và chỉ hiện những nguồn mà câu trả lời *thật
   sự* tham chiếu.
2. **Chạy hoàn toàn local** — không phụ thuộc API trả phí, chạy được trên máy không có GPU.
3. **Hỗ trợ song ngữ Việt/Anh**, kể cả truy xuất chéo ngôn ngữ (hỏi tiếng Việt, tài liệu tiếng
   Anh và ngược lại).
4. **Thà từ chối còn hơn bịa** — câu hỏi ngoài phạm vi tài liệu phải bị chặn ngay ở tầng truy
   xuất, không gọi LLM.
5. **Đo được chất lượng bằng số**, và quan trọng hơn: **đo được cả mức overfit** của chính hệ
   thống bằng một bộ câu hỏi held-out trên tài liệu chưa từng dùng để chỉnh tham số.

**Phạm vi đã chốt (cố ý không làm):** không có backend API riêng (FastAPI/Flask), không lưu
lịch sử hội thoại qua nhiều phiên, corpus quy mô đồ án với 1 người dùng tại 1 thời điểm.

---

## 3. Features

- **Hỏi đáp song ngữ Việt/Anh** — tự nhận diện ngôn ngữ câu hỏi và trả lời đúng ngôn ngữ đó.
- **Trích dẫn nguồn** tới tên file + trang/slide, lọc theo nguồn mà câu trả lời đã dẫn.
- **Hiểu câu hỏi nối tiếp** trong hội thoại ("Thế còn cái thứ hai?") — tất định, 0 lượt gọi LLM.
- **Kiểm chứng khẳng định**: câu dạng "… đúng không?" cho phán quyết ĐÚNG / SAI / KHÔNG ĐỀ CẬP
  kèm trích nguyên văn căn cứ.
- **Cảnh báo mâu thuẫn giữa các nguồn** khi hai tài liệu nói ngược nhau, kèm đủ toạ độ để tự
  mở ra đối chiếu. Hệ thống **không tự phân xử** nguồn nào đúng.
- **Từ chối câu lạc đề** dựa trên điểm rerank, không gọi LLM.
- **Đọc được tài liệu khó**: PDF scan (OCR), PDF nhiều cột, PDF bị dính chữ, bảng biểu, text box
  trong DOCX, và hình ảnh (model vision đọc nội dung trong hình).
- **Trả lời theo luồng** (streaming) — dấu hiệu đầu tiên sau ~2 giây thay vì spinner câm.
- **Index tăng dần + cache theo băm nội dung** — thêm một tài liệu không phải xử lý lại cả
  corpus.
- **Module đánh giá định lượng** kèm bộ HELD-OUT và các script kiểm định từng cơ chế.

---

## 4. System Architecture

Hai luồng dữ liệu tách biệt, dùng chung **1 Embedding Model** và **1 FAISS Index**. Không có
backend API riêng — `app.py` (Streamlit) gọi thẳng các hàm/class ở `rag/*.py`.

```
LUỒNG INGESTION (bấm "Đọc tài liệu")
Tài liệu (PDF/PPTX/DOCX)
  → document_loader.py   đọc MỘT LƯỢT, giữ metadata (tên file, trang/slide); OCR trang hỏng;
                         trích ảnh + chú thích bằng model vision; cache theo băm nội dung
  → chunking.py          Recursive Character Splitting, đo bằng tokenizer thật của model
  → embedding.py         sentence-transformers, đa ngôn ngữ
  → vector_store.py      FAISS IndexFlatIP + metadata (index TĂNG DẦN theo băm nội dung)

LUỒNG QUERY (mỗi câu hỏi)
Câu hỏi
  → tiep_noi_hoi_thoai.py   câu NỐI TIẾP được ghép thêm các câu hỏi trước — tất định
  → embedding.py            cùng model với Ingestion
  → vector_store.py         FAISS (+ nhánh BM25 tuỳ chọn, hợp nhất bằng RRF có trọng số)
  → reranker.py             cross-encoder xếp hạng lại; dưới ngưỡng → TỪ CHỐI, không gọi LLM
  → rag_pipeline.py         dựng đoạn trích quanh chunk khớp → ghép prompt → Ollama (stream)
  → citation.py             lọc lấy đúng nguồn mà câu trả lời đã dẫn
  → doi_chieu_nguon.py      cảnh báo mâu thuẫn giữa các nguồn (chạy SAU khi đã hiện trả lời)
```

`tai_nguyen_gpu.py` cắt ngang cả hai luồng: dò phần cứng, chọn GPU/CPU cho từng vai trò, suy
batch size và số worker từ VRAM còn trống, và nhả model vision ở ranh giới giữa hai giai đoạn.

Luồng dữ liệu đầy đủ và input/output từng module: [ARCHITECTURE.md §3–§5](ARCHITECTURE.md).

---

## 5. Tech Stack

**Mô hình sử dụng** — tất cả chạy local:

| Vai trò | Model | Ghi chú |
|---|---|---|
| Embedding | `intfloat/multilingual-e5-base` | 768 chiều, đa ngôn ngữ, huấn luyện cho *retrieval* |
| Rerank | `BAAI/bge-reranker-v2-m3` | cross-encoder, tầng lọc thứ 2 |
| Sinh câu trả lời | `qwen3:4b` (Ollama) | |
| OCR + chú thích ảnh | `qwen2.5vl:3b` (Ollama) | |
| LLM-as-judge (evaluation) | `qwen3:4b` | đổi bằng `JUDGE_MODEL` |

**Thư viện chính:**

| Thư viện | Dùng ở đâu | Lý do chọn |
|---|---|---|
| `streamlit` | `app.py` | Giao diện web nhanh; `session_state` đúng nhu cầu "chỉ giữ lịch sử trong 1 phiên" |
| `pdfplumber` | `document_loader` | Trích text theo từng trang PDF **kèm số trang** — giữ metadata ngay từ bước đọc |
| `python-pptx` / `python-docx` | `document_loader` | Đọc cấu trúc slide/đoạn văn, không cần engine chuyển đổi trung gian |
| `langchain-text-splitters` | `chunking` | `RecursiveCharacterTextSplitter` nhận `length_function` tuỳ chỉnh |
| `sentence-transformers` | `embedding`, `reranker` | Chạy model local; đồng thời cho truy cập **tokenizer thật** của model |
| `faiss-cpu` | `vector_store` | `IndexFlatIP` phù hợp quy mô corpus đồ án |
| `ollama` | `rag_pipeline`, `metrics` | Gọi LLM local; `format` nhận JSON Schema để ép structured output |
| `langdetect` | `rag_pipeline` | Phát hiện VI/EN — nhẹ, local |
| `tiktoken` | `chunking` | **Chỉ là dự phòng** khi không lấy được tokenizer của model |
| `pytest` | `tests/` | Framework test |

**Cố ý KHÔNG dùng:** `python-dotenv` (tự viết loader nhỏ trong `config.py`), `rank_bm25` (tự cài
để tự quyết cách tách từ tiếng Việt), `underthesea`/`VnCoreNLP` (nặng), `ChromaDB` và backend
`FastAPI` (ngoài phạm vi). Lý do đầy đủ: [ARCHITECTURE.md §6](ARCHITECTURE.md).

---

## 6. Project Structure

```
rag-do-an/
├── app.py                  # Streamlit app chính
├── config.py               # Mọi tham số của hệ thống (đọc từ .env, có sẵn mặc định)
├── rag/                    # Lõi hệ thống — 15 module, xem ARCHITECTURE.md §4–§5
├── evaluation/             # Đánh giá + các script kiểm định/đo đạc
│   ├── run_evaluation.py            # Precision@K, Recall@K, MRR, Faithfulness, Relevance, Citation
│   ├── test_questions.json          # Bộ câu hỏi IN-SAMPLE
│   ├── test_questions_held_out.json # Bộ HELD-OUT (tài liệu chưa dùng để chỉnh tham số)
│   ├── kiem_dinh_*.py               # Đo độ tin cậy của từng cơ chế (judge, đối chiếu, nối tiếp)
│   └── do_*.py                      # Đo quy mô FAISS, worker GPU, ngưỡng rerank, đầu-cuối
├── data/
│   ├── raw/                # Tài liệu gốc upload vào
│   ├── images/             # Ảnh trích từ tài liệu (tự sinh)
│   ├── cache/              # Bộ nhớ đệm Ingestion (xoá lúc nào cũng an toàn)
│   └── faiss_index/        # index.faiss + metadata.pkl + index_info.json
├── TaiLieuTest/            # Corpus 26 tài liệu đã dùng để đo (KET_QUA_DO_DAC.md §2)
├── tests/                  # pytest — 405 test
├── .streamlit/config.toml  # Bảng màu giao diện
├── .env.example
├── requirements.txt
├── ARCHITECTURE.md
└── KET_QUA_DO_DAC.md
```

---

## 7. Requirements

| Hạng mục | Yêu cầu |
|---|---|
| Python | **3.11+** (đã kiểm chứng trên 3.14) |
| Ollama | Đã cài và đang chạy — [ollama.com](https://ollama.com) |
| Model Ollama | `qwen3:4b` (bắt buộc) · `qwen2.5vl:3b` (nếu bật OCR / chú thích ảnh) |
| Dung lượng đĩa | ~3,3 GB cho model embedding + rerank tải từ HuggingFace, cộng dung lượng model Ollama |
| GPU | **Không bắt buộc** — không có thì hệ thống tự lùi về CPU, chỉ chậm hơn |
| Hệ điều hành | Đã kiểm chứng trên Windows 11 |

Máy dùng để đo toàn bộ số liệu trong tài liệu này: Intel Core i5-14600KF (20 luồng) · 31,8 GB
RAM · NVIDIA RTX 5060 8 GB.

---

## 8. Installation

```bash
git clone <repo-url>
cd rag-do-an
pip install -r requirements.txt
ollama pull qwen3:4b
ollama pull qwen2.5vl:3b
```

Lần chạy đầu tiên, `sentence-transformers` tự tải model embedding (~1,1 GB) và rerank (~2,2 GB)
từ HuggingFace về cache local — cần Internet đúng một lần, các lần sau chạy offline được.

### Cài PyTorch bản CUDA (khuyến nghị nếu có GPU NVIDIA)

`pip install -r requirements.txt` kéo về `torch` bản **CPU-only** — trên máy CÓ GPU, đó là một
cấu hình sai *không gây lỗi*: hệ thống vẫn trả lời đúng, chỉ chậm hơn nhiều lần. Đo trên
RTX 5060: embedding **12,8×**, rerank **11,2×**, truy xuất đầu-cuối **9,0×** — mà **6/6 câu hỏi
cho kết quả giống hệt** (KET_QUA_DO_DAC.md §8.2, §8.8).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Thay `cu130` cho khớp driver (`nvidia-smi` in ra CUDA Version; RTX 50xx cần ≥ cu128). Kiểm tra:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Thanh bên của ứng dụng cũng hiện rõ đang chạy GPU hay CPU. **Không có GPU thì bỏ qua mục này** —
hệ thống tự dò và lùi về CPU, không phải chỉnh gì.

---

## 9. Configuration

Mọi tham số nằm ở `config.py` và đều đọc được từ `.env` (không có `.env` thì dùng mặc định):

```bash
cp .env.example .env
```

**Các công tắc hay dùng:**

| Biến | Mặc định | Tác dụng khi đặt `0` |
|---|---|---|
| `BAT_STREAMING` | 1 | tắt trả lời theo luồng |
| `BAT_TRUY_VAN_NGU_CANH` | 1 | tắt hiểu câu hỏi nối tiếp |
| `BAT_DOI_CHIEU_NGUON` | 1 | tắt cảnh báo mâu thuẫn giữa các nguồn |
| `BAT_CHU_THICH_ANH` | 1 | tắt model vision đọc nội dung hình |
| `BAT_OCR_DU_PHONG` | 1 | tắt OCR cho trang PDF đọc hỏng |
| `BAT_RERANK` | 1 | tắt tầng cross-encoder xếp hạng lại |
| `BAT_INDEX_TANG_DAN` | 1 | luôn build lại toàn bộ index |
| `BAT_CACHE_INGESTION` | 1 | tắt cache (dùng khi cần **đo** chi phí build từ đầu) |
| `BAT_PROFILING_INGESTION` | 1 | tắt bảng tổng kết thời gian từng bước sau mỗi lần build |

**Các tham số đáng chỉnh:**

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OLLAMA_MODEL` / `JUDGE_MODEL` | `qwen3:4b` | model sinh câu trả lời / model chấm điểm |
| `OLLAMA_HOST` | `http://localhost:11434` | đổi khi Ollama chạy ở máy/cổng khác |
| `OLLAMA_NUM_CTX` | 16384 | cửa sổ ngữ cảnh — **đừng hạ** (xem §13) |
| `TOP_K` | 4 | số đoạn trích gửi cho LLM; đặt 6 nếu ưu tiên độ chính xác hơn tốc độ |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | 160 / 32 | đo bằng tokenizer thật của model |
| `TRONG_SO_BM25` | 0.0 | BM25 tắt ở vai trò xếp hạng — bật lại nếu corpus của bạn khác |
| `THIET_BI_EMBEDDING` / `THIET_BI_RERANK` | `auto` | ép `cpu`/`cuda` cho riêng từng vai trò |
| `SO_WORKER_VISION` | 2 | số worker OCR/Vision — đo lại bằng `do_worker_gpu.py` |

> **Đổi model embedding hoặc chunk size thì phải bấm "Đọc tài liệu" để build lại index.** Hệ
> thống tự đối chiếu vân tay cấu hình trong `index_info.json` và cảnh báo nếu không khớp.

---

## 10. How to Run

**Chạy ứng dụng:**

```bash
streamlit run app.py
```

Mở `http://localhost:8501`. Ollama phải đang chạy ở tiến trình nền riêng — nó **không** tự khởi
động cùng Streamlit.

**Chạy test:**

```bash
pytest tests/ -v
```

**405 test.** Một số ít phải nạp model thật nên chạy chậm; bỏ qua chúng bằng
`pytest -m "not slow"` trong lúc đang sửa code. `tests/conftest.py` trỏ `data/cache/` và
`data/images/` sang thư mục tạm cho cả phiên test, nên không test nào ghi vào thư mục dữ liệu
thật của dự án.

**Chạy đánh giá** (sau khi đã build index — xem §12):

```bash
python evaluation/run_evaluation.py --nhanh        # chỉ truy xuất, TẤT ĐỊNH, vài phút
python evaluation/run_evaluation.py                # đầy đủ, có gọi LLM — 60–90 phút cho 29 câu
python evaluation/run_evaluation.py --khoang-cach  # cả hai bộ + mức overfit
```

---

## 11. Usage

Giao diện theo lối ứng dụng chat: **thanh bên** (nguồn tài liệu) — **khung chính** (hỏi đáp).

1. **Thanh bên:** upload tài liệu PDF/PPTX/DOCX, tick chọn nguồn được dùng để trả lời, rồi bấm
   **"Đọc tài liệu"** → chạy toàn bộ luồng Ingestion và lưu index xuống `data/faiss_index/`.
2. **Đặt câu hỏi** bằng tiếng Việt hoặc tiếng Anh — hoặc đưa ra một khẳng định để hệ thống
   kiểm chứng ("Pháp luật ra đời trước nhà nước, đúng không?").
3. Câu trả lời hiện **dần theo luồng**, kèm dòng số liệu độ trễ (*truy xuất · chữ đầu tiên ·
   tổng*). Dưới câu trả lời là **nguồn** mà chính câu trả lời đó đã tham chiếu.
4. **Hỏi nối tiếp được** — "Thế còn dấu hiệu thứ hai?" — hệ thống ghép ngữ cảnh hội thoại vào
   truy vấn và **nói ra** việc đã làm vậy, để bạn biết mà hỏi lại nếu nó ghép nhầm chỗ.
5. Nút **＋ Hội thoại mới** xoá lịch sử đang hiển thị nhưng **giữ nguyên tài liệu và index**.
   Lịch sử chat chỉ tồn tại trong phiên Streamlit hiện tại.

**Chi phí xử lý tài liệu.** Thêm một tài liệu không phải trả giá cho cả corpus: mỗi file được
ghi kèm băm nội dung vào `index_info.json`, nên "Đọc tài liệu" chỉ xử lý lại file **mới hoặc
đã đổi nội dung**. Bộ nhớ đệm ở `data/cache/` giữ bốn thứ đắt nhất (kết quả đọc tài liệu, OCR,
chú thích ảnh, vector embedding), tất cả khoá theo **băm nội dung** — nhờ vậy một hình dùng lại
ở 20 slide chỉ tốn **một** lượt gọi model vision, và đổi tên file không làm mất cache.

Xoá cả thư mục `data/cache/` bất cứ lúc nào cũng **an toàn**: mọi thứ trong đó đều tính lại
được, chỉ mất thời gian chứ không mất dữ liệu. Thanh bên có sẵn nút **"Xoá cache"**.

---

## 12. Evaluation / Results

### Cách chạy đánh giá

1. Build index từ tài liệu thật của bạn (qua UI).
2. Điền câu hỏi test vào `evaluation/test_questions.json`:
   ```json
   [
     {
       "cau_hoi": "Học máy có giám sát là gì?",
       "cac_trang_dung": [{"nguon": "ten_file.pdf", "trang": 3}],
       "dap_an_mau": "Là phương pháp huấn luyện mô hình dựa trên dữ liệu đã có nhãn.",
       "loai_tai_lieu": "dai",
       "loai_cau_hoi": "truy_xuat"
     }
   ]
   ```
   - `cac_trang_dung`: danh sách (nguồn, trang) **đúng** chứa câu trả lời. Câu cố tình không có
     đáp án (test hành vi từ chối) thì để `[]`. Chuẩn bị 15–30 câu để kết quả đáng tin cậy.
   - `loai_tai_lieu` / `loai_cau_hoi` là **tuỳ chọn**; có chúng thì script in thêm bảng tách
     theo từng nhóm — đây chính là thứ trả lời câu "hệ thống có ổn định trên nhiều loại tài
     liệu khác nhau không".
   - Chưa có tài liệu để thử? `python evaluation/tao_tai_lieu_mau.py` sinh sẵn một bộ tài liệu
     đa dạng (dài/ngắn/có bảng/có ảnh) để chạy thử đường ống.
3. Chạy `run_evaluation.py` (xem §10). Kết quả in ra bảng và xuất chi tiết ra
   `evaluation/ket_qua_danh_gia*.csv`.

Bốn script kiểm định — mỗi cái đo độ tin cậy của **một** cơ chế trên bộ ca đã biết trước đáp
án, và con số rút ra là thứ nên đặt cạnh tính năng trong báo cáo:

```bash
python evaluation/kiem_dinh_judge.py                 # thước đo Faithfulness (§7.43)
python evaluation/kiem_dinh_doi_chieu.py --so-lan 3  # cơ chế phát hiện mâu thuẫn (§7.59)
python evaluation/kiem_dinh_viet_lai.py --chi-tang-1 # nhận diện câu nối tiếp (§7.58)
python evaluation/do_quy_mo_index.py                 # ngưỡng quy mô FAISS trên máy bạn (§7.44)
```

### Bộ HELD-OUT — đo mức overfit, không chỉ đo điểm

Mọi hằng số của hệ thống (ngưỡng cosine, ngưỡng rerank, trần đoạn mỗi trang, số ứng viên
rerank, chunk size) đều được chọn bằng cách tối ưu trên chính `test_questions.json`, tức
**tuning trên tập test**. Điểm in-sample vì thế luôn đẹp và không nói được gì về tài liệu mới.

`evaluation/test_questions_held_out.json` là **46 câu (44 có đáp án) trên 12 tài liệu chưa từng
dùng để chỉnh bất kỳ tham số nào**. Chênh lệch giữa hai bộ **chính là con số đo mức overfit**.

> **Quy tắc bất di bất dịch:** không bao giờ chỉnh tham số theo kết quả của bộ held-out. Chỉnh
> một lần là nó trở thành bộ in-sample thứ hai và con số này mất sạch ý nghĩa.

### Kết quả

Corpus **26 tài liệu · 3.648 bản ghi trang · 9.285 chunk** (11 PDF · 9 DOCX · 6 PPTX).
Chi tiết đầy đủ, môi trường đo và cách tái lập từng con số:
[**KET_QUA_DO_DAC.md**](KET_QUA_DO_DAC.md).

| Metric | in-sample (25 câu) | held-out (44 câu) | khoảng cách |
|---|---:|---:|---:|
| Recall@K | 0,845 | **0,905** | **−0,060** |
| MRR | 0,980 | 0,843 | **+0,137** |
| Đoạn đúng ở hạng 1 | 24/25 (96%) | 33/44 (75%) | −21 điểm % |
| Faithfulness | 0,980 | 0,977 | **+0,003** |
| Answer Relevance | 1,000 | 0,977 | +0,023 |
| Citation accuracy | 0,714 | **0,784** | **−0,070** |
| Precision@K | 0,620 | 0,364 | +0,256 *(không so được)* |

Kết quả chia làm bốn phần, và đó mới là thông tin có giá trị:

- **Tìm đúng nội dung: tổng quát tốt.** Recall@K trên tài liệu chưa từng thấy còn cao hơn —
  chunking, embedding và tầng truy hồi **không** overfit vào corpus cũ.
- **Xếp đúng thứ tự: có overfit thật.** MRR 0,980 → 0,843, tỷ lệ đoạn đúng ở hạng 1 rơi từ 96%
  xuống 75% — đúng vào tầng mà mọi ngưỡng của hệ thống đang tác động.
- **Chất lượng câu trả lời: không overfit**, nhưng kết luận này **đã từng bị đo SAI**: bộ
  held-out 22 câu cho khoảng cách +0,155, nhân đôi lên 44 câu thì co về +0,003, trong khi
  khoảng cách MRR lại **nở ra**. Thêm mẫu rồi xem khoảng cách nở ra hay co lại là phép thử rẻ
  nhất để phân biệt tín hiệu thật với nhiễu (KET_QUA_DO_DAC.md §5.3 · ARCHITECTURE.md §7.64).
- **P@K không so được giữa hai bộ**: nó phụ thuộc số trang đúng mỗi câu (in-sample 2,84
  trang/câu, held-out 1,20), nên +0,256 phần lớn là hiện vật của cách ra đề.

**Hiệu năng** (RTX 5060, sau khi chia VRAM theo giai đoạn):

| Hạng mục | Giá trị |
|---|---|
| Truy xuất mỗi câu | **0,45 s** (trung vị) — rerank 0,45 s · mã hoá câu hỏi 0,02 s · FAISS < 1 ms |
| Một lượt hỏi đầu-cuối | trung vị **35,3 s** (24,4 – 52,1) — truy xuất chỉ chiếm ~1,3% |
| Nạp corpus 26 tài liệu lần đầu | 70 phút (chủ yếu là chú thích 1.707 ảnh bằng model vision) |
| Nạp lại khi cache đầy | 3,48 s cho 1.209 chunk — **nhanh hơn 81×** |

**Nút thắt hiện nay** (đo được, không phải phỏng đoán): với tài liệu nhiều hình, **89,8%** chi
phí nạp lần đầu nằm ở bước chú thích ảnh; với tài liệu thuần chữ thì **87,8%** nằm ở bước đọc
text. Còn mỗi câu hỏi thì **~98%** thời gian là LLM sinh chữ, mà phần lớn trong đó là chuỗi suy
luận nội bộ của `qwen3` — truy xuất **không còn là chỗ đáng tối ưu tiếp**.

**Hai tính năng hội thoại, đo riêng** (KET_QUA_DO_DAC.md §7.3):

| Cơ chế | Kết quả |
|---|---|
| Hiểu câu hỏi nối tiếp | trùng chuẩn vàng **1/16 → 16/16**; tầng nhận diện **10/10** |
| Phát hiện mâu thuẫn giữa các nguồn | **7/7** ca, ổn định qua 3 lần chạy; **3/3** ca im lặng đúng |

### ⚠ Ba lưu ý bắt buộc đọc trước khi diễn giải số liệu

**1. Phần lớn metric KHÔNG tất định.** Đã đo trực tiếp: cùng một câu hỏi, cùng index, cùng
prompt, chạy 4 lần thì có lần model gắn 6 số trích dẫn, có lần không gắn số nào.

| Metric | Tất định? | Diễn giải chênh lệch |
|---|---|---|
| Precision@K, Recall@K, MRR | **Có** | So sánh trực tiếp được |
| Faithfulness, Answer Relevance | Đã bớt dao động (trung vị 3 lần) | Chênh lệch nhỏ vẫn cần dè dặt |
| **Citation accuracy** | **Không** — dao động mạnh nhất | Chênh dưới ~0,1 **không nên diễn giải là gì cả** |

**2. Precision@K/Recall@K SUY BIẾN với DOCX không có ngắt trang.** DOCX không có khái niệm
"trang" cố định, nên file không có ngắt trang nào được coi là MỘT trang. Hai metric này so
khớp theo (nguồn, trang), nên với file như vậy chỉ cần lấy về một chunk bất kỳ là
Recall@K = 1,00 — bất kể chunk đó có chứa câu trả lời hay không. Với loại tài liệu này chỉ
Citation accuracy còn mang thông tin (ARCHITECTURE.md §7.39).

**3. Câu hỏi cố tình không có đáp án** (`cac_trang_dung: []`) luôn cho Precision@K/Recall@K = 0
theo định nghĩa; Faithfulness của câu từ chối luôn được chấm 1,0, còn Answer Relevance có thể
thấp vì câu trả lời không cung cấp nội dung cụ thể — quy ước bình thường của metric, không
phải lỗi hệ thống.

**Thước đo tự kiểm tra chính nó.** LLM-as-judge có sai số và sai số đó **không có triệu
chứng**: điểm 0,0 chấm cho một câu trả lời đúng trông y hệt điểm 0,0 chấm cho một câu bịa. Bốn
cơ chế đã thêm: chấm 3 lần lấy trung vị · cờ `!` cạnh điểm Faithfulness đáng ngờ · bộ kiểm
định giám khảo 7 ca (**đúng 7/7, dao động 0,00**) · chặn điểm ngoài thang [0,1]. Nên phát biểu
là *"Faithfulness 0,980, đo bằng thước đo đã kiểm định đúng 7/7 ca"* thay vì con số trơ trọi.

---

## 13. Limitations

- **Đổi model embedding thì phải bấm "Đọc tài liệu"** — quên build lại không gây lỗi (2 model
  có thể cùng số chiều) mà chỉ khiến kết quả sai âm thầm; hệ thống tự đối chiếu và cảnh báo
  trên UI, `run_evaluation.py` thì dừng hẳn.
- **Mỗi câu hỏi mất trung vị ~35 giây** với `qwen3:4b` trên GPU, do model luôn sinh phần suy
  luận nội bộ dài trước khi trả lời — đặc tính của model, `/no_think` lẫn `think=False` đều
  không rút ngắn được. Streaming không rút ngắn tổng thời gian nhưng đưa dấu hiệu đầu tiên về
  ~2 giây; câu bị từ chối ở tầng truy xuất thì không gọi LLM nên nhanh hơn hẳn.
- **Đừng hạ `OLLAMA_NUM_CTX` để lấy lại tốc độ** — hạ `TOP_K` hoặc `NGAN_SACH_KY_TU_MOI_DOAN`
  thay vào đó. Hạ `num_ctx` không làm prompt ngắn đi, nó chỉ chuyển quyền quyết định cắt chỗ
  nào sang Ollama — mà Ollama luôn cắt từ **đầu**, tức xoá đúng đoạn trích liên quan nhất.
- **PDF nặng công thức toán**: khi PDF nhúng font không kèm bảng ToUnicode, công thức bị đọc ra
  thành mã `(cid:NN)`. Hệ thống lọc bỏ rác này nhưng **không khôi phục được nội dung công
  thức** trừ khi bật OCR. Câu hỏi về khái niệm vẫn trả lời tốt; câu hỏi về công thức thì không.
- **DOCX không có khái niệm "trang" cố định** — tách theo dấu ngắt trang cứng nếu có, hoặc coi
  cả file là 1 "trang". `python-docx` cũng chỉ thấy bảng ở tầng ngoài cùng, không thấy bảng
  lồng trong ô của bảng khác.
- **Nhận diện tiêu đề trong PDF là suy đoán theo cỡ chữ** (PDF không lưu cấu trúc logic) nên có
  thể bỏ sót hoặc nhận nhầm. **Trang trong trích dẫn là vị trí vật lý trong file**, không phải
  số trang in trên tài liệu.
- **Nhận diện câu nối tiếp dựa trên danh sách dấu hiệu hồi chỉ** ("thế còn", "cái đó",
  "what about"…), không phân tích cú pháp; câu diễn đạt ngoài danh sách sẽ bị bỏ sót. Ngữ cảnh
  hội thoại cũng **chỉ gồm các câu hỏi trước, không gồm câu trả lời trước** — đánh đổi có chủ
  đích (ARCHITECTURE.md §7.58).
- **Phát hiện mâu thuẫn chỉ soi các đoạn ĐÃ ĐƯỢC TRUY XUẤT** cho câu hỏi hiện tại, không quét
  toàn bộ corpus — kiểm định nhất quán toàn corpus là bài toán khác, nằm ngoài phạm vi đồ án.
- **Không có backend API riêng** và **không lưu lịch sử qua nhiều phiên** — đúng phạm vi đồ án
  đã chốt. (Việc truy xuất không nhìn thấy lịch sử *trong cùng một phiên* là khiếm khuyết thật
  và đã được sửa — ARCHITECTURE.md §7.58.)
- Nếu đường dẫn project chứa ký tự Unicode đặc biệt, một số thao tác ghi file cấp thấp của
  FAISS có thể lỗi — giữ project trong đường dẫn thuần ASCII để an toàn.

---

## 14. Future Work

**1. Đổi index FAISS khi corpus lớn hơn — ngưỡng đã đo, không ước.** `IndexFlatIP` là tìm kiếm
vét cạn nên độ trễ tăng tuyến tính theo số chunk.

| Số chunk | FlatIP p95 | RAM | HNSW p95 / recall | IVFFlat p95 / recall |
|---:|---:|---:|---|---|
| 10.000 | 1,4 ms | 29 MB | 0,8 ms / 0,87 | 0,2 ms / 0,90 |
| 50.000 | 7,2 ms | 146 MB | 1,2 ms / 0,74 | 1,1 ms / 0,96 |
| 100.000 | 12,9 ms | 293 MB | 1,5 ms / 0,61 | 1,3 ms / 0,97 |

- **Ngưỡng theo tốc độ: ~1,5 triệu chunk.** Corpus hiện tại (9.285 chunk) mới ở **0,6%**.
- **Ngưỡng thực tế là BỘ NHỚ: ~700.000 chunk (≈2 GB RAM cho index)**, tức khoảng 145.000 trang.
- **Khi đổi thì chọn `IndexIVFFlat`, không phải HNSW**: ở 100.000 chunk cả hai đều nhanh hơn
  Flat ~10 lần, nhưng IVF giữ recall 0,97 còn HNSW chỉ 0,61.
- Đổi index thì **bắt buộc** chạy lại `run_evaluation.py`: đoạn bị bỏ sót hoàn toàn có thể là
  đoạn chứa câu trả lời.

**2. Giảm độ trễ thật.** Nút thắt đã chuyển hẳn sang bước LLM sinh chữ (~98% mỗi lượt hỏi).
Đường khả dĩ còn lại là dùng model **không sinh suy luận** cho câu hỏi truy xuất thường — đã
kiểm chứng rằng `think=False` *không* làm được việc đó (nó chỉ đổ chuỗi lập luận thẳng vào câu
trả lời), nên phải đổi hẳn model và đo lại Faithfulness trước khi chốt.

**3. Adaptive TOP-K.** Bị hoãn có chủ đích chứ không phải bỏ quên: `TOP_K=4` là giá trị mà toàn
bộ Recall@K, MRR và các ngưỡng lọc đã hiệu chỉnh trên đó, nên hạ nó cho "câu hỏi đơn giản" mà
không đo lại chính là đổi độ chính xác lấy tốc độ. Cần một phép đo Recall@K tách theo nhóm độ
phức tạp câu hỏi trước đã (ARCHITECTURE.md §7.67).

**4. Giám khảo mạnh hơn cho evaluation.** Với `qwen3:4b`, 1/8 lần chấm cho kết quả ngược hẳn —
trung vị 3 lần là vá chứ không phải sửa gốc. Đặt `JUDGE_MODEL` sang model lớn hơn rồi chạy
`kiem_dinh_judge.py` để xem có đáng đổi không.

**5. Nhận diện ô KHOÁ–GIÁ TRỊ trong biểu mẫu.** Phần lớn vấn đề bảng lớn đã xử lý được, nhưng
biểu mẫu hành chính vẫn là ca khó: nhãn và giá trị nằm chung một hàng với cả tiêu đề dài, nên
vector của chunk đó bị tiêu đề lấn át.

**Ngoài phạm vi (cố tình không triển khai):** conversation memory qua nhiều phiên (session ID,
lưu lịch sử xuống đĩa), backend API riêng (FastAPI/Flask), so sánh lại FAISS với vector DB khác.

---

## 15. Author

**Phạm Lê Triệu Vĩ** — đồ án tốt nghiệp.

Mọi quyết định thiết kế trong dự án này đều được ghi lại kèm số liệu đã đo ở
[ARCHITECTURE.md](ARCHITECTURE.md) §7 và [KET_QUA_DO_DAC.md](KET_QUA_DO_DAC.md); phần nào chưa
đo được thì ghi rõ là chưa đo được, không suy đoán thay.
