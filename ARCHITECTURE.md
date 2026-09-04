# Kiến trúc hệ thống — RAG hỏi đáp tài liệu học tập

Tài liệu này tổng hợp bức tranh toàn cảnh của hệ thống. Lý do chi tiết của từng quyết định
nằm ngay trong code dưới dạng comment, ở đúng chỗ ra quyết định — mục §5 dưới đây chỉ giữ
lại phần cốt lõi kèm số liệu đã đo.

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Luồng dữ liệu](#2-luồng-dữ-liệu)
3. [Vai trò và Input/Output từng module](#3-vai-trò-và-inputoutput-từng-module)
4. [Thư viện sử dụng và lý do](#4-thư-viện-sử-dụng-và-lý-do)
5. [Quyết định thiết kế quan trọng](#5-quyết-định-thiết-kế-quan-trọng)
6. [Triển khai và chạy hệ thống](#6-triển-khai-và-chạy-hệ-thống)

---

## 1. Tổng quan kiến trúc

Mô hình **RAG (Retrieval-Augmented Generation)** cổ điển, chạy hoàn toàn local, không có
backend API riêng:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         app.py (Streamlit)                          │
│         "composition root" — gọi thẳng các hàm/class ở rag/*.py      │
└───────────────┬─────────────────────────────────┬───────────────────┘
                │                                 │
        LUỒNG INGESTION                    LUỒNG QUERY
      (nút "Đọc tài liệu")                 (mỗi lần chat)
                │                                 │
                ▼                                 ▼
      rag/document_loader.py            rag/embedding.py (encode câu hỏi)
    ◄── rag/bo_nho_dem.py ──►                     │
    ◄── rag/do_thoi_gian.py ─►                    ▼
                │                       rag/vector_store.py (search)
                ▼                                 │
        rag/chunking.py                           ▼
                │                       rag/reranker.py → rag/rag_pipeline.py
                ▼                                 │      (ghép prompt + gọi Ollama)
        rag/embedding.py                          ▼
                │                                 │
                ▼                                 ▼
       rag/vector_store.py  ◄──── dùng chung ──► rag/citation.py
```

`bo_nho_dem` (cache theo băm nội dung) và `do_thoi_gian` (profiling) chỉ phục vụ luồng
Ingestion và **không** nằm trên đường đi của một câu hỏi — độ trễ lúc hỏi không đổi vì chúng.

`rag/tai_nguyen_gpu.py` thì cắt ngang CẢ HAI luồng: nó quyết định embedding và reranker
chạy trên GPU hay CPU, batch size bao nhiêu, mấy worker vision — tất cả suy từ phần cứng
của máy đang chạy — và nhả model vision khỏi VRAM ở ranh giới giữa hai giai đoạn (§5.68).

**Nguyên tắc cốt lõi:** hai luồng độc lập về thời điểm chạy nhưng dùng **chung 1 instance**
`EmbeddingService` và `VectorStore`. Bắt buộc dùng chung model embedding — nếu không, vector
câu hỏi và vector tài liệu nằm ở hai không gian khác nhau và cosine similarity vô nghĩa.

Không có backend/API riêng (FastAPI/Flask) — đúng phạm vi đồ án đã chốt: corpus nhỏ, 1 người
dùng tại 1 thời điểm, không cần horizontal scaling.

---

## 2. Luồng dữ liệu

### 2.1 Ingestion (bấm "Đọc tài liệu")

```
data/raw/*.pdf|*.pptx|*.docx
   │  bam_file() từng tài liệu, đối chiếu với sổ băm trong index_info.json
   │     → chỉ những file MỚI / ĐÃ ĐỔI mới đi tiếp; file đã biến mất bị xoa_theo_nguon()
   │  doc_nhieu_file()             → List[{nguon, trang, noidung}]   (1 phần tử / trang·slide)
   │     ├─ trúng cache tài liệu (băm nội dung + vân tay cấu hình đọc) → trả kết quả cũ
   │     └─ trượt cache → doc_pdf/pptx/docx MỘT LƯỢT (§5.66):
   │            pha 1  đọc text + bảng + cột + tiêu đề + liệt kê ứng viên ảnh (chưa render)
   │            pha 2  OCR các trang đã đánh dấu — tra cache trước, phần còn lại gọi
   │                   model SONG SONG (SO_WORKER_VISION)
   │            pha 3  gộp OCR, dọn watermark, loại trang rỗng / trang mục lục
   │            pha 4  render ảnh được giữ lại, loại hình lặp kiểu logo
   │            rồi   chú thích ảnh bằng model vision (gộp ảnh trùng + cache + song song)
   │  chia_chunk()                 → List[{chunk_id, nguon, trang, vi_tri, noidung}]
   │                                 (~160 token ĐO BẰNG TOKENIZER CỦA MODEL)
   │  encode_co_cache()            → np.ndarray (n, 768) float32, đã chuẩn hoá
   │                                 (chỉ encode chunk chưa có trong cache embedding)
   │  VectorStore.them() → .luu()   (ghi kèm sổ băm tài liệu)
   │  tai_nguyen_gpu.ket_thuc_ingestion(): nhả model vision khỏi VRAM, dọn bộ đệm CUDA
   │     → chuyển sang giai đoạn QUERY (§5.68)
   ▼
data/faiss_index/  index.faiss · metadata.pkl · index_info.json ("vân tay" cấu hình, §5.20)
data/cache/        tai_lieu/ · ocr/ · vision/ · embedding/  (khoá theo BĂM NỘI DUNG, §5.66)
```

**Index TĂNG DẦN theo mặc định** (`BAT_INDEX_TANG_DAN`): chỉ tài liệu mới hoặc đã đổi nội dung
mới được xử lý lại; phần còn lại giữ nguyên vector đang có. Đây là chỗ đã thay cho quyết định
cũ ở §5.10 ("build lại toàn bộ khi THÊM tài liệu") — quyết định đó đúng khi chưa có cách nào
biết tài liệu nào đã đổi, và băm nội dung chính là cách đó. XOÁ tài liệu vẫn có hiệu lực tức
thời qua `xoa_theo_nguon()` như cũ. Khi vân tay index không khớp cấu hình (đổi model embedding,
chunk size…) thì hệ thống tự lùi về build **toàn bộ**, vì lúc đó vector cũ và mới không nằm
cùng một không gian ngữ nghĩa.

Sau mỗi lần build, `rag/do_thoi_gian.py` in bảng tổng kết thời gian từng bước
(`BAT_PROFILING_INGESTION`).

### 2.2 Query (mỗi câu hỏi)

```
Câu hỏi
   │  chuan_bi_truy_van(): câu NỐI TIẾP ("Thế còn cái thứ hai?") được ghép thêm các câu
   │  hỏi trước thành truy vấn CHÍNH — tất định, 0 lượt gọi LLM (§5.58)
   ├── encode_cau_hoi() → tim_kiem_vi_tri()      nhánh VECTOR, cho MỖI truy vấn
   └── tim_kiem_tu_khoa()                        nhánh BM25   (mặc định TẮT, §5.30)
             │  hợp nhất RRF CÓ TRỌNG SỐ: Σ trọng_số/(RRF_K + thứ_hạng)
             │  lọc theo nguon_cho_phep (tick chọn nguồn ở UI)
             ▼
   rerank bằng cross-encoder (§5.24) — đổi THỨ TỰ, giữ nguyên thang điểm cosine
             │  số ứng viên được chấm tuỳ ĐỘ PHỨC TẠP câu hỏi: 30 (phức tạp) / 12 (§5.67)
             │  điểm rerank < NGUONG_DIEM_RERANK_TOI_THIEU → TỪ CHỐI, không gọi LLM (§5.29)
             ▼
   _dung_doan_trich(): dựng đoạn QUANH chunk khớp, mở rộng sang chunk liền kề cùng trang;
                       trần SO_DOAN_TOI_DA_MOI_TRANG và trần riêng cho đoạn là ảnh (§5.35)
             ▼
   _phat_hien_ngon_ngu() → vi/en  ·  la_cau_hoi_kiem_chung() → prompt thường / KIỂM CHỨNG
             ▼
   nen_ngu_canh(): prompt vượt trần cửa sổ → bỏ các đoạn xếp hạng THẤP NHẤT (không bao giờ
                   hạ num_ctx — §5.60, §5.67); num_predict cũng theo độ phức tạp câu hỏi
             ▼
   Ollama chat(stream=True) → câu trả lời chạy dần (§5.42)
             ▼
   loc_theo_tham_chieu(): chỉ giữ nguồn mà câu trả lời THẬT SỰ dẫn (§5.14, §5.54)
             ▼
   tim_mau_thuan(): đối chiếu CHÉO các nguồn, cảnh báo khi hai tài liệu nói ngược nhau.
                    Chạy SAU khi câu trả lời đã hiện xong nên không làm chậm chữ đầu tiên;
                    tầng lọc tất định khiến đại đa số lượt hỏi tốn 0 lượt LLM (§5.59)
```

Index rỗng hoặc không đoạn nào đạt ngưỡng → trả thẳng `CAU_TU_CHOI`, **không gọi LLM**.

---

## 3. Vai trò và Input/Output từng module

### `config.py`
Nguồn cấu hình duy nhất: nạp `.env` (loader tự viết), ép console UTF-8 (§5.5), cấu hình
logging. Output là các hằng số module-level (`CHUNK_SIZE_TOKENS`, `EMBEDDING_MODEL_NAME`,
`TOP_K`, `OLLAMA_MODEL`, các `Path` tới `data/…`) mà mọi module khác import trực tiếp.

### `rag/document_loader.py`
Đọc PDF/PPTX/DOCX thành text, giữ metadata nguồn ngay từ bước đọc, chuẩn hoá Unicode NFC,
dọn watermark + loại trang mục lục (§5.17), đọc bảng sang Markdown (§5.26), đọc PDF nhiều
cột (§5.51), đọc lại trang dính chữ (§5.40), OCR trang hỏng (§5.37, §5.49). PDF đi qua đúng
**một lượt duyệt**, chia 4 pha (§5.66).

| Hàm | Input | Output |
|---|---|---|
| `doc_pdf` / `doc_pptx` / `doc_docx` | `Path` tới 1 file | `List[{nguon, trang, noidung}]` |
| `doc_tai_lieu(duong_dan)` | 1 file, tự nhận đuôi | như trên — chỉ ĐỌC, chưa chú thích ảnh |
| `doc_tai_lieu_hoan_chinh(duong_dan)` | 1 file | đọc + chú thích ảnh + bỏ ảnh rỗng — **đơn vị được cache** |
| `doc_tai_lieu_co_cache(duong_dan)` | 1 file | như trên, nhưng trúng cache thì trả kết quả cũ (§5.66) |
| `cac_file_tai_lieu(thu_muc)` | 1 thư mục | danh sách `Path` file hỗ trợ được, thứ tự ổn định |
| `doc_nhieu_file(cac_duong_dan)` | danh sách file | gộp; file hỏng bị bỏ qua có báo cáo (§5.53) |
| `doc_thu_muc(thu_muc)` | 1 thư mục | `doc_nhieu_file(cac_file_tai_lieu(thu_muc))` |
| `HieuChinhXTolerance` | — | nhớ mức `x_tolerance` đã dùng được cho tài liệu hiện tại (§5.66) |

`doc_nhieu_file` nhận DANH SÁCH thay vì thư mục là điều kiện để index tăng dần hoạt động:
khi chỉ 1 trong 26 tài liệu thay đổi, `app.py` chỉ đưa đúng file đó vào.

### `rag/bo_nho_dem.py`
Bộ nhớ đệm theo **băm nội dung** cho luồng Ingestion (§5.66). Mọi lỗi đọc/ghi cache đều bị
nuốt kèm log: cache hỏng chỉ được phép làm hệ thống chậm lại đúng bằng lúc chưa có cache.

| Hàm / lớp | Vai trò |
|---|---|
| `bam_file` / `bam_bytes` / `bam_chuoi` | băm SHA-256 cắt còn 32 ký tự hex (đủ ngắn cho đường dẫn Windows) |
| `van_tay_doc_tai_lieu()` | băm của các tuỳ chọn ĂN VÀO KẾT QUẢ ĐỌC — đổi chúng thì cache phải trượt |
| `KhoDem` | kho khoá-giá trị trên đĩa, chia thư mục con theo 2 ký tự đầu của khoá |
| `kho_tai_lieu` / `kho_ocr` / `kho_vision` | ba kho dùng chung cho cả tiến trình |
| `khoa_tai_lieu` / `khoa_ocr` / `khoa_vision` | dựng khoá cho từng loại (xem bảng ở §5.66) |
| `KhoVectorDem` / `encode_co_cache` | cache embedding trong MỘT file `.npz`, chỉ encode chunk mới |
| `dung_luong_cache()` / `xoa_cache()` | cho giao diện nói được con số thật khi mời xoá |

### `rag/tai_nguyen_gpu.py`
Dò phần cứng và quản lý VRAM theo giai đoạn (§5.68). Không có GPU thì mọi hàm ở đây thành
không-làm-gì chứ không ném lỗi — máy chỉ có CPU phải chạy được đầy đủ.

| Hàm | Vai trò |
|---|---|
| `co_cuda()` | máy có GPU dùng được cho PyTorch không (nuốt mọi lỗi driver/DLL) |
| `thiet_bi("embedding")` / `("rerank")` | "cuda"/"cpu", ép riêng từng vai trò qua cấu hình |
| `kich_thuoc_lo_embedding()` | batch encode suy từ VRAM **còn trống**, chặn trên bởi cấu hình |
| `so_worker_vision()` | min(trần cấu hình, số nhân CPU, VRAM còn trống) |
| `vram()` / `tong_vram_gb()` / `vram_con_trong_gb()` | số liệu VRAM cho quyết định và log |
| `mo_ta_phan_cung()` | một dòng nói rõ đang chạy GPU hay CPU — chống lại đúng lỗi ở §5.68 |
| `nha_model_ollama(ten)` | bảo Ollama nhả model khỏi VRAM ngay (`keep_alive=0`) |
| `ket_thuc_ingestion()` | ranh giới giai đoạn: nhả vision + dọn bộ đệm CUDA |

### `rag/do_thoi_gian.py`
Đo thời gian từng bước Ingestion và in bảng tổng kết sau mỗi lần build. Bộ đếm là biến
module (một tiến trình = một lần build) và có `Lock` vì Vision/OCR chạy trên nhiều thread.
`dat_lai()` · `do("ten_buoc")` (context manager) · `ghi_nhan()` · `bao_cao()` · `ghi_bao_cao()`.

### `rag/chunking.py`
Recursive Character Splitting, đo bằng **tokenizer thật của model** (tiktoken chỉ dự phòng).

| Hàm | Output |
|---|---|
| `kich_thuoc_chunk_an_toan(max_seq_length)` | `CHUNK_SIZE_TOKENS` đã tự hạ cho vừa model (§5.18) |
| `tao_splitter(dem_token_fn, max_seq_length)` | `RecursiveCharacterTextSplitter` đã cấu hình |
| `chia_chunk(cac_trang, …)` | `List[{chunk_id, nguon, trang, vi_tri, noidung}]`; bảng giữ nguyên khối, quá lớn thì cắt theo HÀNG kèm lặp tiêu đề (§5.41) |

`vi_tri` = thứ tự chunk trong trang gốc, dùng để `rag_pipeline` mở rộng ngữ cảnh đúng thứ tự.

### `rag/embedding.py` — `EmbeddingService`
Wrapper `sentence-transformers`, chạy trên GPU nếu máy có (§5.68). **Không có** hàm
`encode()` dùng chung: chỉ có
`encode_cau_hoi()` (tiền tố `query: `) và `encode_tai_lieu()` (tiền tố `passage: `) — model
họ E5 huấn luyện bất đối xứng, thiếu tiền tố thì chất lượng tụt âm thầm (§5.19).
Ngoài ra: `.dimension` (768), `.max_seq_length` (512), `.dem_token()`, `.lay_ham_dem_token()`.

### `rag/lexical_search.py`
BM25 tự cài (~60 dòng). `_tach_tu()` lập chỉ mục **cả âm tiết đơn lẫn bigram** để bắt cụm
thuật ngữ tiếng Việt mà không cần thư viện tách từ (§5.21). Mặc định TẮT (§5.30).

### `rag/reranker.py` — `RerankerService`
Cross-encoder (~2.2GB) đọc cả cặp (câu hỏi, đoạn), chạy trên GPU nếu máy có — đây là model
đáng đưa lên GPU nhất vì nó nằm trên đường đi của MỌI câu hỏi (§5.68).
`.xep_hang(cau_hoi, cac_doan)`
điểm song song với `cac_doan`, chưa sắp xếp. Điểm này còn dùng cho ngưỡng từ chối (§5.29).

### `rag/image_extractor.py`
Bản ghi ảnh dùng **đúng schema `{nguon, trang, noidung}`** của văn bản, nên đi qua toàn bộ
luồng còn lại y hệt một đoạn text — không module nào phải biết đến khái niệm "ảnh" (§5.27).
PPTX duyệt đệ quy group shape; DOCX/PPTX quét `rels` để không mất ảnh SVG/ảnh liên kết (§5.33).

| Hàm | Vai trò |
|---|---|
| `ung_vien_anh_trang(trang)` | liệt kê `(bbox, có_phủ_cả_trang)` của ảnh đáng giữ — **chưa render** (§5.66) |
| `luu_anh_trang_pdf(...)` | render + lưu ra file những ảnh đã được chọn |
| `trich_anh_pptx` / `trich_anh_docx` | trích ảnh từ gói Office (blob có sẵn, không phải render) |
| `ly_do_loai_anh(rong, cao, dt_trang)` | ba chốt hình dạng: kích thước · tỉ lệ cạnh · tỉ lệ diện tích |
| `ly_do_loai_anh_blob(du_lieu)` | như trên nhưng đọc kích thước thẳng từ bytes (PPTX/DOCX) |
| `loc_anh_lap_lai(cac_ban_ghi, nguon)` | loại hình lặp lại kiểu logo/watermark theo băm nội dung |

Tách "chọn ảnh nào" khỏi "render ảnh đó" là điều làm nên luồng đọc một-lượt: bước chọn chỉ
đọc metadata có sẵn trong đối tượng trang nên rẻ tới mức chạy được ngay trong vòng lặp đọc
text, còn bước render chỉ chạy cho ảnh đã qua mọi bộ lọc.

### `rag/vision_caption.py`
| Hàm | Vai trò |
|---|---|
| `mo_hinh_vision_co_san()` | model đã pull chưa — để giảm cấp thay vì crash |
| `chu_thich_anh()` | model vision đọc nội dung BÊN TRONG hình → text tìm kiếm được |
| `bo_sung_chu_thich_vision()` | sửa `noidung` tại chỗ (nối thêm, không thay thế); gộp ảnh trùng nội dung → tra cache → gọi model song song (§5.66) |
| `trang_can_ocr()` / `ocr_trang_pdf()` | phát hiện + OCR trang đọc hỏng (§5.37, §5.49) |

### `rag/tiep_noi_hoi_thoai.py`
Đưa NGỮ CẢNH HỘI THOẠI vào truy xuất để hiểu câu hỏi nối tiếp (§5.58). Toàn bộ đường mặc
định là TẤT ĐỊNH — không gọi model nào.

| Hàm | Vai trò |
|---|---|
| `la_cau_hoi_tiep_noi(cau_hoi, lich_su)` | `bool` — câu này có cần ngữ cảnh lượt trước mới hiểu được không (dấu hiệu hồi chỉ + liên từ mở đầu) |
| `truy_van_ngu_canh(cau_hoi, lich_su)` | ghép các CÂU HỎI trước vào câu hiện tại thành một truy vấn mang đủ chủ đề |
| `ngu_canh_cho_prompt(lich_su, ngon_ngu)` | khối ngữ cảnh cho prompt sinh câu trả lời, dán nhãn rõ "KHÔNG PHẢI nguồn thông tin" |
| `viet_lai_cau_hoi(...)` | đường query rewriting bằng LLM — **mặc định TẮT**, xem kết quả âm tính ở §5.58 |
| `chuan_bi_truy_van(...)` | điểm vào: trả `{cau_hoi_goc, cau_hoi_chinh, cac_truy_van_phu, ngu_canh_llm, la_tiep_noi, da_viet_lai}` |

### `rag/doi_chieu_nguon.py`
Phát hiện MÂU THUẪN giữa các đoạn trích từ những tài liệu KHÁC NHAU (§5.59). Hai tầng như
retrieval: lọc tất định rồi mới cho LLM đọc kỹ vài cặp sống sót.

| Hàm | Vai trò |
|---|---|
| `co_dau_hieu_bat_dong(a, b)` | `bool` — dấu hiệu bề mặt: lệch tập số (kể cả số viết bằng chữ) hoặc lệch phủ định |
| `cac_cap_dang_ngo(cac_doan, vector_doan)` | các cặp `(i, j)` đáng chấm: khác nguồn + cùng chủ đề (cosine) + có dấu hiệu bất đồng, cắt theo `SO_CAP_DOI_CHIEU_TOI_DA` |
| `tim_mau_thuan(cac_doan, embedding_service, client)` | `List[{nguon_a, trang_a, nguon_b, trang_b, muc_do, noi_dung_xung_dot}]` — rỗng là kết quả bình thường và hay gặp nhất |

### `rag/vector_store.py` — `VectorStore`
Wrapper FAISS `IndexFlatIP` + quản lý metadata song song + 2 chỉ mục phụ dựng lười (chỉ mục
trang, chỉ mục BM25) tự huỷ khi dữ liệu đổi.

| Method | Ghi chú |
|---|---|
| `.them(vectors, metadata_list)` / `.xoa_theo_nguon(ten_file)` | xoá có hiệu lực tức thời (§5.10) |
| `.tim_kiem` / `.tim_kiem_vi_tri` / `.tim_kiem_tu_khoa` | trả `(metadata\|vị trí, điểm)` giảm dần |
| `.diem_cosine(vi_tri, vector)` | cosine cho chunk chỉ do BM25 tìm ra (`reconstruct`) |
| `.chi_muc_trang` / `.theo_nguon_va_trang()` | phục vụ mở rộng ngữ cảnh cùng trang |
| `.luu()` / `.tai()` | ghi/đọc `index.faiss` + `metadata.pkl` + `index_info.json` |
| `.ly_do_khong_tuong_thich()` | lý do index không khớp cấu hình hiện tại, hoặc `None` (§5.20) |

### `rag/rag_pipeline.py` — `RagPipeline`
Nhận sẵn `EmbeddingService` + `VectorStore` qua constructor, không tự tạo (§5.8).

| Method | Output |
|---|---|
| `.truy_xuat(cau_hoi, top_k, nguon_cho_phep, lich_su)` | `List[{nguon, trang, noidung, doan_khop, diem_similarity}]`; `lich_su` bật đường hiểu câu nối tiếp (§5.58) |
| `.sinh_cau_tra_loi_theo_luong(…)` | generator `{loai: "suy_luan"\|"cau_tra_loi", them}` — **đường gọi LLM DUY NHẤT** (§5.42) |
| `.sinh_cau_tra_loi(…)` | vòng gom các mảnh của generator trên, không gọi Ollama riêng |
| `.hoi_dap_theo_luong(…)` | `{loai: "truy_xuat_xong", …}` → các mảnh → `{loai: "xong", ket_qua}` |
| `.hoi_dap(…)` | `{cau_tra_loi, cac_chunk_nguon, la_kiem_chung, truy_van, mau_thuan, bam_nguon, do_tre{…}}` |

Hàm module-level `la_cau_hoi_kiem_chung()` (§5.22) và `_phat_hien_ngon_ngu()` (§5.31).

Ngân sách thích ứng (§5.67) nằm ở ba hàm module-level, tách rời để test được mà không cần
Ollama: `la_cau_hoi_phuc_tap(cau_hoi)` phân loại độ phức tạp; `ngan_sach_token_ngu_canh(...)`
tính số token còn lại cho đoạn trích sau khi trừ phần cố định của prompt; `nen_ngu_canh(...)`
bỏ các đoạn xếp hạng thấp nhất cho vừa ngân sách. `_tinh_num_ctx()` nhận thêm `num_predict`
để không giữ chỗ một khoảng sinh không bao giờ dùng tới — nhưng **không bao giờ** trả về giá
trị nhỏ hơn `OLLAMA_NUM_CTX` (§5.60).

### `rag/citation.py`
| Hàm | Ghi chú |
|---|---|
| `dinh_dang_trich_dan()` | ánh xạ **1-1, giữ nguyên thứ tự** với `cac_chunk` (phần tử i ↔ đoạn `[i+1]` trong prompt), KHÔNG loại trùng |
| `loc_theo_tham_chieu()` | chỉ giữ nguồn được dẫn, gộp theo `(nguon, trang)`; rỗng với câu từ chối (§5.14) |
| `cau_theo_trich_dan()` | `Dict[int, List[str]]` — số `[n]` → những CÂU đã dẫn nó (§5.28) |
| `do_bam_ngu_canh()` | phép đo tất định, `evaluation/metrics.py` import lại — **không có bản sao** (§5.55) |

### `app.py`
Giao diện Streamlit và "composition root". Bố cục thanh bên (nguồn tài liệu) + cột đọc căn
giữa + ô nhập ghim đáy (§5.47). Trích dẫn lưu trong chính message của `st.session_state`
(không dùng biến "trích dẫn hiện tại" chung — từng gây bug lệch pha). Giao diện chỉ dẫn ra
**vị trí** (tên file + trang/slide), không in lại nguyên văn đoạn đã dùng; trường
`doan_trich` vẫn được tính và trả về cho `evaluation/metrics.py` chấm Citation accuracy.
Uploader đổi `key` sau mỗi lần xử lý để nút xoá và uploader không đá nhau (§5.15).

### `evaluation/`
| File | Vai trò |
|---|---|
| `metrics.py` | Precision@K, Recall@K, MRR, Faithfulness (chấm 3 lần lấy trung vị + cờ tự nghi ngờ, §5.43), Answer Relevance, Citation accuracy (§5.28) |
| `run_evaluation.py` | nạp `test_questions.json` → chạy pipeline → in bảng + xuất `ket_qua_danh_gia.csv`; dừng hẳn nếu index không khớp cấu hình |
| `kiem_dinh_judge.py` | đo độ tin cậy của CHÍNH thước đo Faithfulness trên 7 ca đã biết đáp án |
| `do_nguong_rerank.py` | đo xem điểm rerank có tách được câu lạc đề không (§5.29) |
| `do_worker_gpu.py` | đo số worker OCR/Vision tối ưu trên máy hiện tại, kèm GPU util + VRAM (§5.68) |
| `do_dau_cuoi.py` | đo ĐẦU-CUỐI: nạp tài liệu → hỏi được, và hỏi → trả lời xong (§5.68) |
| `do_quy_mo_index.py` | đo ngưỡng quy mô FAISS: Flat vs IVF vs HNSW (§5.44) |
| `tao_tai_lieu_mau.py` | sinh bộ tài liệu ĐỘC LẬP để chống overfitting (§5.45) |
| `kiem_dinh_viet_lai.py` | đo tầng nhận diện câu nối tiếp + ảnh hưởng THẬT lên truy xuất (§5.58) |
| `kiem_dinh_doi_chieu.py` | đo độ tin cậy của chính cơ chế phát hiện mâu thuẫn, trên 7 ca đã biết đáp án (§5.59) |

---

## 4. Thư viện sử dụng và lý do

| Thư viện | Dùng ở đâu | Lý do chọn |
|---|---|---|
| `pdfplumber` | `document_loader` | Trích text theo từng trang PDF kèm số trang — đúng nhu cầu giữ metadata ngay từ bước đọc. |
| `python-pptx` / `python-docx` | `document_loader` | Thư viện chuẩn đọc cấu trúc slide/đoạn văn, không cần engine chuyển đổi trung gian. |
| `langchain-text-splitters` | `chunking` | Có sẵn `RecursiveCharacterTextSplitter` đúng thuật toán yêu cầu, nhận `length_function` tuỳ chỉnh. |
| `tiktoken` | `chunking` | **Chỉ là dự phòng** khi không lấy được tokenizer của model — sai số ~1.9 lần trên tiếng Việt (§5.3). |
| `sentence-transformers` | `embedding`, `reranker` | Chạy model embedding/cross-encoder **local**; đồng thời cho truy cập tokenizer thật của model. |
| `faiss-cpu` | `vector_store` | Vector DB đã chốt; `IndexFlatIP` phù hợp quy mô corpus đồ án (§5.1, §5.44). |
| `ollama` | `rag_pipeline`, `metrics` | Gọi LLM **local**; hỗ trợ `format` nhận JSON Schema để ép structured output (§5.6). |
| `langdetect` | `rag_pipeline` | Phát hiện VI/EN — nhẹ, local; cách dùng đã chỉnh lại ở §5.31. |
| `streamlit` | `app.py` | Giao diện web nhanh, `session_state` đáp ứng đúng nhu cầu "chỉ giữ lịch sử trong 1 phiên". |
| `pytest` | `tests/` | Framework test chuẩn. |

**Không dùng** và lý do:
- `python-dotenv` — chỉ cần đọc vài dòng `KEY=VALUE`, tự viết loader nhỏ trong `config.py`.
- `rank_bm25` — công thức BM25 ngắn và cố định; tự cài để tự quyết cách tách từ tiếng Việt
  (§5.21), vốn là phần quan trọng hơn bản thân công thức.
- `underthesea` / `VnCoreNLP` — nặng; cách lập chỉ mục bigram đã giải quyết được vấn đề âm
  tiết rời mà không cần thêm model.
- `ChromaDB`, backend `FastAPI` — ngoài phạm vi đồ án đã chốt.

> *Reranking trước đây cũng nằm trong danh sách "không dùng". Quyết định đã đảo lại có chủ
> đích — xem §5.24.*

---

## 5. Quyết định thiết kế quan trọng

*(Số §5.x được giữ nguyên vì README và comment trong code tham chiếu tới chúng.)*

### 5.1 `IndexFlatIP` thay vì `IndexFlatL2`
Vector đã chuẩn hoá norm = 1 → inner product **tương đương cosine similarity**, đúng thước
đo "độ liên quan ngữ nghĩa" cần dùng, thay vì khoảng cách Euclid thô.

### 5.2 Chunk size 160 token, overlap 32 token
Model giới hạn 512 token; nội dung vượt bị cắt khi encode **không báo lỗi**. 160 token đủ
trọn ý mà vẫn còn biên an toàn rộng; overlap 32 giữ ngữ cảnh ở ranh giới hai chunk.
Đo trên giáo trình 230 trang: **2957 → 915 chunk**, độ dài trung bình **40 → 140 token thật**.

### 5.3 Đếm token bằng tokenizer thật, `tiktoken` chỉ là dự phòng
Đo trên corpus tiếng Việt: tiktoken đếm gấp **~1.9 lần** tokenizer thật (nền XLM-R) — chunk
"100 token" thực chất chỉ ~40 token, tức dùng 31% giới hạn. Hệ quả không phải "chunk hơi
nhỏ" mà là **nội dung bị băm vụn**: mỗi chunk 1-2 câu, vector mô tả một mẩu ngữ nghĩa lưng
chừng.

### 5.4 Chuẩn hoá Unicode NFC ngay khi đọc tài liệu
Hai chuỗi "giống hệt nhau" khi hiển thị có thể khác nhau về byte, gây lỗi so khớp ở các bước
sau — đặc biệt là so khớp `(nguon, trang)` trong `evaluation/metrics.py`.

### 5.5 Ép UTF-8 cho stdout/stderr trong `config.py`
Console Windows mặc định `cp1252` không encode được tiếng Việt → mọi `print()`/log có dấu
sẽ crash. `config.py` được import đầu tiên ở mọi entry point nên là chỗ sửa 1 lần cho tất cả.

### 5.6 LLM-as-judge dùng JSON Schema thay vì `format="json"`
`format="json"` chỉ ép cú pháp; model từng trả `{"answer": "0.5"}` thay vì `{"diem", "ly_do"}`
khiến điểm mặc định về 0.0 dù câu trả lời đúng. Truyền JSON Schema đầy đủ → Ollama ép đúng
cấu trúc field bằng constrained decoding. (Nhưng schema **không** ràng buộc được khoảng giá
trị — xem §5.48.)

### 5.7 Faithfulness của câu trả lời từ chối luôn = 1.0
Câu từ chối không đưa ra thông tin nào nên về định nghĩa không thể "bịa"; quy tắc này nêu rõ
trong prompt vì thực tế model từng chấm 0.0. Answer Relevance vẫn để judge chấm bình thường
(thường thấp) — đúng quy ước chuẩn trong đánh giá RAG.

### 5.8 `RagPipeline` không tự tạo `EmbeddingService`/`VectorStore`
Cả hai được khởi tạo ở `app.py`/`run_evaluation.py` rồi truyền vào. Load model tốn vài giây
và tốn RAM; tách rời cũng đảm bảo Ingestion và Query dùng chung đúng 1 instance model.

### 5.9 Không backend API, không conversation memory nhiều phiên
Quyết định phạm vi đã chốt trước khi code, không phải giới hạn kỹ thuật.

### 5.10 THÊM tài liệu build lại từ đầu, XOÁ thì incremental
Bất đối xứng có chủ đích: **thêm** dù sao cũng phải embed nội dung mới nên incremental không
tiết kiệm được gì; **xoá** chỉ cần bỏ vector cũ (rẻ, tức thời) và việc trích dẫn còn trỏ tới
file người dùng đã xoá là hành vi gây hiểu lầm, phải tránh ngay.

> **ĐÃ THAY THẾ — xem §5.66.** Lập luận trên có một lỗ hổng chỉ lộ ra khi corpus lớn lên: nó
> đúng cho **tài liệu vừa thêm** (file đó dù sao cũng phải embed) nhưng bỏ qua **25 tài liệu
> còn lại**, vốn chẳng đổi gì mà vẫn bị đọc lại, OCR lại, chú thích ảnh lại và encode lại.
> Thứ còn thiếu lúc đó không phải là ý tưởng incremental mà là một cách ĐÁNG TIN để biết tài
> liệu nào đã đổi — băm nội dung ghi kèm vào `index_info.json` chính là cách đó. Vế "XOÁ thì
> incremental" giữ nguyên và nay áp dụng cho cả file bị xoá khỏi thư mục, không chỉ file xoá
> qua giao diện.

### 5.11 Dựng đoạn trích QUANH chunk khớp, không gộp nguyên trang ("small-to-big")
Chunk nhỏ tối ưu cho *retrieval* nhưng khiến ngữ cảnh đưa cho LLM bị vụn. Bản trước sửa bằng
cách **gộp nguyên trang** — và chính cách sửa đó gây lỗi "tài liệu dài thì trả lời và trích
dẫn không chính xác":

| | Gộp nguyên trang | Dựng quanh chunk khớp |
|---|---|---|
| Tổng ngữ cảnh gửi LLM | ~16.000 ký tự (TOP_K=8) | ~7.000 ký tự (TOP_K=6) |
| Phần liên quan tới câu hỏi | thiểu số | đa số |
| Đoạn trích hiển thị | 400 ký tự **đầu trang** | đúng chunk đã khớp |

`_dung_doan_trich()` lấy chunk khớp nhất làm **neo**, mở rộng luân phiên sang chunk liền
sau/liền trước trong cùng trang tới `NGAN_SACH_KY_TU_MOI_DOAN` (ưu tiên phía **sau** vì kiểu
mất mát hay gặp nhất là đoạn liệt kê bị cắt ngang). Đúng cho cả tài liệu dài lẫn ngắn: trang
thưa chữ vẫn lọt trọn ngân sách nên hành vi y hệt bản cũ.

### 5.12–5.13 Nhận diện ngôn ngữ và "trang" của DOCX
`langdetect` thay vì heuristic dấu tiếng Việt (xử lý được cả câu không dấu) — cách dùng đã
chỉnh lại ở §5.31. DOCX không lưu số trang trong XML, nên tách theo **dấu ngắt trang cứng**
(có sẵn trong XML) thay vì render lại layout bằng LibreOffice headless; không có ngắt trang
nào thì cả file là 1 "trang" — thông tin đúng, chỉ kém chi tiết. Hệ quả với thước đo: xem §5.39.

### 5.14 Hiển thị đúng những nguồn câu trả lời THẬT SỰ tham chiếu tới
Bản trước luôn hiển thị đoạn có `diem_similarity` cao nhất — giả định này **sai với tài liệu
dài**: khi hàng chục trang cùng chủ đề có điểm sát nhau, đoạn thắng điểm có thể chỉ nhắc tới
chủ đề. System prompt bắt LLM gắn số `[n]`, nên những con số đó là **bằng chứng trực tiếp**.
`loc_theo_tham_chieu()` lọc theo 3 lớp, lớp sau chỉ dùng khi lớp trước không có kết quả:
(1) số `[n]`; (2) số trang được nhắc trong câu trả lời; (3) nguồn liên quan nhất — kèm cờ
cảnh báo (§5.54). Câu **từ chối** không hiển thị nguồn nào: vừa nói "không tìm thấy" vừa chỉ
vào một trang là tự mâu thuẫn. Chuỗi từ chối nằm ở `config.CAU_TU_CHOI` — cả 3 nơi dùng nó
phải khớp tuyệt đối, lệch một dấu chấm là cơ chế nhận diện hỏng mà không có lỗi nào báo ra.

**Số `[n]` KHÔNG hiển thị cho người đọc** (`citation.bo_so_trich_dan`, gọi ở đúng 4 chỗ vẽ:
streaming từng mảnh, bản vẽ cuối, vẽ lại lịch sử, và chế độ không streaming). Chúng vẫn được
LLM sinh ra và vẫn được hệ thống đọc — nhưng với người đọc, số hiệu là thứ tự đoạn trích
TRONG PROMPT, một thứ tự họ không nhìn thấy nên cũng không tra ngược được. Quan trọng: chỉ gỡ
ở tầng HIỂN THỊ. Bản gốc còn số vẫn được lưu trong lịch sử chat, vì gỡ khỏi dữ liệu sẽ phá
đúng cơ chế trên và phá luôn phép chấm Citation accuracy (§5.28) — im lặng, không lỗi nào báo
ra. Khi streaming, mảnh đang tới có thể cắt ngang giữa `[3,4]`, nên hàm này cắt cả đuôi ngoặc
dở — cùng lý do khiến việc bóc thẻ `<think>` phải là máy trạng thái (§5.42).

**Một lỗi có sẵn lộ ra khi thử hiện số lên màn hình:** câu trả lời thật ghi `[6]` và `[3,4,5]`
nhưng mẫu cũ `\[(\d+)\]` chỉ khớp `[6]` — **không bắt dạng gộp nhiều số trong một cặp ngoặc**.
Hậu quả lan xa hơn phần hiển thị: `cau_theo_trich_dan()` dùng chung mẫu đó để chấm Citation
accuracy, nên mọi ý dẫn nguồn theo dạng gộp đều bị bỏ khỏi phép chấm — thước đo âm thầm bỏ
sót đúng những câu trả lời dẫn nhiều nguồn cho một ý. Nay bắt cả `[3,4,5]`, `[3, 4]`, `[3;4]`
lẫn `[1][2]`, và không đụng tới mốc `[BẢNG]`/`[HÌNH]`.

**Số liệu §5.38 có phải đo lại không? ĐÃ ĐO — và câu trả lời là KHÔNG.** Chạy mẫu cũ và mẫu
mới trên đúng 29 câu trả lời đã lưu ở `ket_qua_danh_gia.csv` (phép so tất định, không gọi
model): chỉ **1/29 câu** dùng dạng gộp, bỏ sót 2 số. Kể cả giả định cực đoan nhất cho hai cặp
bổ sung đó, Citation accuracy trung bình chỉ đổi trong khoảng **0.596 – 0.615** so với 0.605
đã báo cáo — tức **±0.01**, trong khi §5.46 đã kết luận chênh lệch dưới ~0.10 của chính metric
này *không nên được diễn giải là gì cả*. Con số 0.61 đứng vững.

Ghi lại vì đây là một cạm bẫy về quy trình chứ không phải về code: phát hiện ra một lỗi ở
thước đo thì phản xạ đầu tiên là "phải đo lại tất cả" — mà đo lại đầy đủ ở đây tốn 3 tiếng
(dựng lại index từ `TaiLieuTest/` rồi chạy 29 câu). Chỉ mất vài giây để hỏi *lỗi đó ảnh hưởng
bao nhiêu* trước khi trả cái giá đó.

### 5.15 Xoá tài liệu: đồng bộ giữa uploader và nút xoá
`st.file_uploader` trả về cùng danh sách file trên **mọi** lần rerun, nên `if file_upload:`
sẽ ghi đè file lại sau mỗi lần bấm nút xoá. Fix: đổi `key` của uploader (bộ đếm tăng sau mỗi
lần xử lý upload) để Streamlit coi đó là widget mới, rỗng.

### 5.16 Nút "toàn màn hình" tự làm đã bị gỡ
Khi bố cục chuyển sang thanh bên (§5.47), thu gọn thanh bên bằng nút `«` CHÍNH LÀ chế độ
toàn màn hình — cùng kết quả, không state, không CSS bám vào DOM, không nút. Ghi lại vì đây
là dạng quyết định dễ bỏ sót: một tính năng tự viết đôi khi cần **biến mất** cùng với thứ đã
sinh ra nó, chứ không phải được thay bằng tính năng tự viết khác.

### 5.17 Dọn watermark + loại trang Mục lục
PDF từ StuDocu có watermark `lOMoARcPSD|<số>` và `Downloaded by …` lặp trên hầu hết các trang
→ mọi chunk dính chung một đoạn nhiễu giống hệt nhau, giảm độ phân biệt ngữ nghĩa.

Nghiêm trọng hơn: trang **Mục lục** từng bị chọn làm nguồn trả lời vì nó chứa tên hầu hết
chủ đề nên tương đồng cao với nhiều câu hỏi. Nhận diện bằng 2 cách: (1) đầu trang bắt đầu
bằng "mục lục"/"table of contents" — bắt trang ĐẦU; (2) tỉ lệ dòng kết thúc bằng số ≥ 0.5 —
bắt các trang TIẾP THEO của mục lục nhiều trang (thực tế gặp mục lục dài **6 trang**, chỉ
trang đầu có chữ "Mục lục"). Ngưỡng 0.5 hiệu chỉnh bằng dữ liệu thật: trang mục lục 0.59–0.93,
trang nội dung 0.0–0.07.

**Lưu ý:** "trang" trong trích dẫn luôn là **vị trí vật lý trong file PDF**, không phải số
trang in trên tài liệu — file có bìa/lời nói đầu chưa đánh số thì hai con số lệch nhau một
khoảng cố định. Đây là giới hạn cố hữu, không phải bug.

### 5.18 Chunk tự co lại cho vừa model, và bỏ chunk quá ngắn
Hai chặn cứng, đều nhằm loại bỏ hỏng hóc **không báo lỗi**: (a) `kich_thuoc_chunk_an_toan()`
hạ chunk size xuống `max_seq_length - biên` — phần vượt giới hạn bị cắt âm thầm lúc encode
nên **vĩnh viễn không tìm thấy được**; (b) bỏ chunk ngắn hơn 25 ký tự (mẩu vụn không mang đủ
ngữ nghĩa nhưng vẫn chiếm 1 vector và vẫn có thể lọt top với câu hỏi ngắn).

`vi_tri` đánh số **trước** khi lọc nên dãy có thể khuyết — chủ ý, vì pipeline sắp theo
`vi_tri` rồi lấy phần tử liền kề *trong danh sách*, không dựa vào số liên tục.

### 5.19 Đổi sang model retrieval (họ E5) và mã hoá bất đối xứng
`paraphrase-multilingual-MiniLM-L12-v2` huấn luyện cho **đo độ giống nhau giữa 2 câu cùng
loại** (STS), không phải cho **retrieval** (câu hỏi ngắn ↔ đoạn tài liệu dài) — dùng sai loại
model là một nguyên nhân gốc khiến truy xuất hay chọn đoạn "nghe giống câu hỏi" thay vì đoạn
chứa câu trả lời; kèm theo `max_seq_length` chỉ 128 token.

Họ E5 huấn luyện cho retrieval, 512 token, và **bất đối xứng**: câu hỏi cần tiền tố `query: `,
tài liệu cần `passage: `. Thiếu tiền tố thì chất lượng tụt mà không có dấu hiệu nào — vì vậy
`EmbeddingService` cố tình không có `encode()` dùng chung. Tiền tố tự suy từ tên model nên
đổi sang model khác họ thì nó tự tắt.

### 5.20 Vân tay cấu hình đi kèm index (`index_info.json`)
Đổi model embedding rồi quên build lại **không gây crash** (hai model có thể cùng số chiều)
nhưng kết quả gần như ngẫu nhiên, và không thể phát hiện qua kết quả. Phải đối chiếu bằng
thông tin ghi lúc build. `app.py` cảnh báo; `run_evaluation.py` **dừng hẳn** — một bảng số
liệu trông bình thường nhưng vô nghĩa còn tệ hơn không có số liệu nào.

### 5.21 Tìm kiếm lai vector + BM25, hợp nhất bằng RRF
Embedding nén cả đoạn về 1 vector nên làm mờ chi tiết **hiếm và cụ thể** (số hiệu điều luật,
thuật ngữ, con số); BM25 mạnh đúng ở đó và yếu đúng chỗ vector mạnh.

Hợp nhất bằng **RRF** (`Σ 1/(RRF_K + thứ_hạng)`) — cộng nghịch đảo **thứ hạng**, không cộng
điểm: cosine (chặn trong [-1,1]) và BM25 (không chặn trên) là hai thang khác hẳn nhau, cộng
thẳng thì nhánh thang lớn nuốt trọn nhánh kia.

**Tách từ tiếng Việt**: lập chỉ mục cả âm tiết đơn lẫn **bigram** ("pháp_luật", "quy_phạm"
hiếm hơn hẳn nên IDF cao). Giữ nguyên dấu — bỏ dấu sẽ gộp "má"/"mà"/"mã" thành "ma".

Ngoài ra `SO_DOAN_TOI_DA_MOI_TRANG` chặn cả `TOP_K` suất dồn vào 1 trang — rất hay xảy ra
với tài liệu dài, và đó chính là kiểu lỗi "câu trả lời chỉ bám 1 chỗ, bỏ sót phần còn lại".

*(Kết quả đo cuối cùng khiến BM25 bị tắt mặc định — xem §5.30.)*

### 5.22 Phát hiện khẳng định sai (chống a dua)
Lỗi đã gặp: người dùng khẳng định nội dung sai, hệ thống vẫn "gật đầu". Hai nguyên nhân độc
lập, phải sửa cả hai:

**(a) Prompt không hề yêu cầu đối chiếu.** Sửa ở 2 tầng: quy tắc chống a dua thêm vào system
prompt **thường**; `la_cau_hoi_kiem_chung()` nhận diện câu dạng khẳng định và chuyển sang
**system prompt kiểm chứng** riêng, bắt buộc ra phán quyết rời rạc (`ĐÚNG`/`SAI`/`TÀI LIỆU
KHÔNG ĐỀ CẬP`) kèm trích **nguyên văn** căn cứ. Hai ràng buộc này chặn đúng cơ chế gây lỗi:
model không còn được viết một đoạn chung chung *nghe như đang đồng ý*. Nhận diện cố ý ưu tiên
độ chính xác hơn độ phủ; phần bỏ sót đã có quy tắc ở prompt thường đỡ lại.

**(b) Ngữ cảnh quá loãng** — việc thu gọn ngữ cảnh ở §5.11 cũng là một phần của cách sửa.

Kết quả đo (giáo trình 230 trang): 5/5 khẳng định sai bị bác đúng kèm căn cứ nguyên văn
("Nhà nước có năm đặc điểm", "cấu thành bởi bốn yếu tố"…), khẳng định ĐÚNG không bị phản bác
bừa, câu lạc đề bị từ chối không kèm trích dẫn. Xem thêm §5.56 về thứ tự bố cục.

### 5.23 Tham số `think` của Ollama: đo rồi mới dùng
| Cách gọi | Kết quả |
|---|---|
| `think=True` / không truyền gì | máy chủ tách suy luận sang `message["thinking"]`, `content` sạch |
| `think=False` | **không** tắt suy luận, mà tắt việc **tách** nó — toàn bộ chuỗi suy luận đổ thẳng vào `content` |

Khi không cần suy luận thì **bỏ hẳn tham số**, tuyệt đối không truyền `False`. Mẹo `/no_think`
cũng đã bỏ: đo thực tế cho thấy model vẫn sinh ~15.000 ký tự suy luận và không nhanh hơn
đáng kể (43.7s so với 47.0s — trong khoảng nhiễu).

### 5.24 Xếp hạng lại bằng cross-encoder (rerank)
Quyết định "ngoài phạm vi" của bản trước đã được đảo lại có chủ đích: đúng loại lỗi nó giải
quyết — "lấy nhầm đoạn cùng chủ đề nhưng sai chi tiết" — là lỗi còn tồn đọng sau khi đã làm
hết những việc rẻ hơn.

**Vì sao 2 tầng chứ không thay thế tầng 1:** cross-encoder đọc CẢ CẶP nên chính xác hơn hẳn
bi-encoder, nhưng vì thế **không tính trước được** — quét cả corpus bằng nó là bất khả thi.
Tầng 1 quét rộng lấy vài chục ứng viên, tầng 2 đọc kỹ vài chục ứng viên đó.

| Cấu hình | MRR | Đúng hạng 1 | Thời gian |
|---|---|---|---|
| Tắt rerank | 0.417 | 3/12 | 0.02s |
| 8 / 15 ứng viên | 0.542 | 5/12 | 1.47s / 2.62s |
| **30 ứng viên** | **0.642** | **6/12** | 6.07s |

Chọn 30 vì +6 giây chỉ chiếm ~15% tổng thời gian trả lời. **Điểm cosine được GIỮ NGUYÊN**,
rerank chỉ đổi THỨ TỰ CHỌN — trộn hai thang đo vào cùng một trường là kiểu lỗi rất khó lần ra.

### 5.25 Nhận diện tiêu đề để cắt chunk theo ranh giới ngữ nghĩa
Tiêu đề được đưa lên **đầu** danh sách separator vì nó là ranh giới ngữ nghĩa mạnh nhất.
Ba định dạng, ba mức tin cậy: DOCX (`style.name` bắt đầu bằng `"Heading"`) và PPTX
(`slide.shapes.title`) là metadata thật; PDF chỉ còn hình thức — cỡ chữ lớn hơn cỡ áp đảo
của trang **và** dòng đủ ngắn (so theo **tỉ lệ**, không theo con số cứng, vì mỗi tài liệu
dùng cỡ chữ nền khác nhau).

Đo trên bộ mẫu (24 câu): MRR 0.938 → **0.958**, đúng hạng 1 21/24 → **22/24**. Cải thiện có
thật nhưng **nhỏ** và nằm trong biên nhiễu — giữ vì không gây hại, không nên diễn giải quá lên.

### 5.26 Bảng biểu: giữ nguyên cấu trúc hàng-cột
Bản trước làm mất bảng theo 3 cách: DOCX không đọc `document.tables`; PPTX nối các ô thành
dòng rời rạc; PDF trộn ô thành text lộn xộn. Hệ quả: câu hỏi cần đọc giao điểm hàng-cột
không trả lời được dù dữ liệu vẫn trong index.

Nay cả 3 định dạng chuyển bảng sang **Markdown** qua một hàm dùng chung, và `chia_chunk()`
giữ mỗi bảng là **một chunk nguyên vẹn**. Markdown vì LLM hiểu ngay và Streamlit render
thành bảng thật — một định dạng, hai việc.

Hai cạm bẫy thư viện, cả hai đều làm mất nội dung **không báo lỗi**: PDF `extract_text()` gộp
cả text trong bảng (phải loại vùng bbox của bảng ra); PPTX `for shape in slide.shapes`
**không** thấy gì bên trong group shape (`duyet_shape()` đệ quy, dùng chung cho cả đọc text
lẫn trích ảnh nên không nơi nào quên).

### 5.27 Hình ảnh: trích ra, gắn chú thích, và để model vision đọc
Hai tầng tách rời vì chi phí chênh nhau xa. **Tầng 1** (BẬT, gần như miễn phí): ảnh lưu ra
`data/images/`, ghép với văn bản lân cận — ưu tiên dòng dạng "Hình 3: …" vì nó mô tả ĐÚNG
hình đó. **Tầng 2** (model vision đọc nội dung BÊN TRONG hình): chú thích chỉ cho biết hình
TÊN là gì; câu hỏi mà đáp án nằm trong các ô của sơ đồ thì chú thích không giúp được.

Mô tả được **nối thêm** sau chú thích chứ không thay thế — hai nguồn bổ khuyết nhau. Giảm
cấp thay vì hỏng: model chưa pull thì ghi cảnh báo và bỏ qua, không làm hỏng cả lần build.
*(Mặc định của tầng 2 đã đổi thành BẬT sau khi đo lại chi phí — xem §5.34.)*

### 5.28 Đo độ chính xác của TRÍCH DẪN, tách khỏi độ trung thực
| | Faithfulness | Citation accuracy |
|---|---|---|
| Câu hỏi đo | Toàn bộ câu trả lời có bịa không? | Số `[n]` gắn kèm mỗi ý có trỏ đúng chỗ không? |
| Phạm vi đối chiếu | Toàn bộ ngữ cảnh | Từng cặp (ý, đoạn được dẫn cho ý đó) |

Một câu trả lời có thể đạt Faithfulness 1.0 mà trích dẫn vẫn trỏ nhầm đoạn — người đọc bấm
vào nguồn sẽ không thấy điều họ vừa đọc. Ba quy ước:
- Dẫn số **không tồn tại** (`[9]` khi chỉ có 6 đoạn) tính 0 điểm — bịa số nguồn là lỗi nặng nhất.
- Câu **TỪ CHỐI** không dẫn số nào → `None`, loại khỏi trung bình (không dẫn nguồn là ĐÚNG).
- Câu **TRẢ LỜI THẬT** mà không dẫn số nào → **0.0**. Bản trước gộp chung với câu từ chối, và
  cái giá rất cụ thể: một lần sửa prompt khiến model bỏ trích dẫn ở 3 câu nhưng điểm gần như
  không đổi — **lỗi đi lọt qua thước đo** (§5.46).

Chạy ở tầng đánh giá, không phải lúc chạy thật (mỗi cặp tốn một lượt LLM).

### 5.29 Từ chối dựa trên điểm rerank — thứ mà cosine không làm được
| Nhóm câu hỏi | Cosine (min–max) | Rerank (min–max) |
|---|---|---|
| Đúng chủ đề, tiếng Việt | 0.865 – 0.905 | 0.200 – 0.996 |
| Đúng chủ đề, tiếng Anh | 0.766 – **0.816** | 0.019 – 0.861 |
| Lạc đề, tiếng Việt | 0.780 – **0.828** | 0.000 – **0.003** |

Theo cosine, câu tiếng Anh ĐÚNG chủ đề (0.816) còn thấp hơn câu tiếng Việt LẠC ĐỀ (0.828) —
**không ngưỡng cosine nào cứu được**. Theo rerank thì hai nhóm tách hẳn. Lý do: cross-encoder
trả lời "đoạn này có đáp ứng câu hỏi này không", còn cosine chỉ đo "hai đoạn text có giống
nhau không" — mà câu hỏi tiếng Anh thì không "giống" đoạn văn tiếng Việt nào.

Ngưỡng đặt **thấp có chủ đích** vì hai loại sai không ngang giá: từ chối nhầm câu hợp lệ là
hỏng thấy ngay, còn bỏ lọt câu lạc đề chỉ rơi xuống quy tắc từ chối trong prompt vốn đang
chạy tốt. Sau khi đo trên corpus **song ngữ** thật, ngưỡng hạ tiếp 0.005 → **0.001**: con số
0.005 hiệu chỉnh trên corpus thuần tiếng Việt đã từ chối oan câu hỏi chéo ngôn ngữ hợp lệ
(0.0023). Kiểm chứng ngưỡng này không bị overfitting: §5.57.

Lợi ích phụ: câu lạc đề bị chặn trong ~6,6 giây thay vì 33–67 giây, vì không phải gọi LLM.

### 5.30 BM25 bị TẮT mặc định — một kết quả âm tính đo được
Đo trên corpus song ngữ thật (13 tài liệu, 5909 chunk, 26 câu hỏi chia 3 nhóm), cột là
MRR *cùng-ngữ / chéo-ngữ / từ-khoá / **chung***:

| trọng số BM25 | rerank BẬT | rerank TẮT |
|---|---|---|
| **0.0 (tắt)** | 0.924 / **0.703** / 1.000 / **0.888** | 0.909 / 0.407 / 0.938 / **0.783** |
| 0.2 | 0.924 / 0.656 / 1.000 / 0.875 | 0.909 / 0.276 / 0.938 / 0.747 |
| 1.0 (bản trước) | 0.924 / 0.536 / 1.000 / 0.843 | 0.914 / 0.203 / 0.938 / 0.730 |

Hai điều đi ngược kỳ vọng:
1. **BM25 không giúp gì ngay trên sở trường của nó** — nhóm câu toàn từ khoá hiếm (`SIFT`,
   `RANSAC`, `Kullback-Leibler divergence`, mã số sinh viên) đạt MRR **1.000 nhờ riêng dense**.
2. **BM25 gây hại nặng cho truy xuất chéo ngôn ngữ** (0.703 → 0.536): câu hỏi tiếng Việt về
   tài liệu tiếng Anh thì BM25 không khớp nổi từ nào với tài liệu ĐÚNG nhưng khớp rất "tự tin"
   với tài liệu tiếng Việt SAI, mà RRF lại coi hạng 1 của hai nhánh ngang nhau. Kiểm chứng
   từng bước: dense xếp đúng ở **hạng 3**, sau RRF tụt xuống **hạng 13**.

Hai thay đổi: `SU_DUNG_TIM_KIEM_LAI` (bật/tắt) → `TRONG_SO_BM25` (số thực, RRF **có trọng
số**); mặc định `0`. Code `lexical_search.py` **giữ nguyên, không xoá** — corpus khác có thể
cho kết quả khác, bật lại thì nên đo trước bằng thí nghiệm tương tự.

### 5.31 Chọn giữa hai ngôn ngữ, không phải nhận diện trong hàng trăm thứ tiếng
Bản trước viết `"en" if detect(q) == "en" else "vi"`. Sai thực tế: `langdetect` chấm
*"What does criminal law regulate?"* là **tiếng Catalan (0.71)**, tiếng Anh chỉ 0.29 → câu
tiếng Anh bị trả lời bằng tiếng Việt. Ba bước, dừng ở bước có bằng chứng chắc nhất:

1. **Dấu tiếng Việt** (`ă â đ ê ô ơ ư` + thanh điệu) — không ngôn ngữ nào khác dùng đủ bộ này.
2. **So trực tiếp xác suất `en` với `vi`**, bỏ qua thứ hạng chung → "en 0.29 vs ca 0.71" vẫn
   ra tiếng Anh.
3. **Từ tiếng Việt không dấu** — danh sách cố ý chỉ chứa từ KHÔNG phải tiếng Anh (`la`, `the`,
   `hay`, `ta` bị loại dù phổ biến), vì thà bỏ sót còn hơn nhận nhầm.

Kết quả: **0/10 sai** trên bộ kiểm (trước là 1/10).

### 5.32 Bảng được phép dài hơn văn xuôi
160 token là lựa chọn về độ chính xác truy xuất cho **văn xuôi**; áp cùng con số cho bảng là
nhầm mục đích — cắt nhỏ bảng phá đúng thứ khiến nó là bảng (dòng tiêu đề cột). Ràng buộc
**cứng** duy nhất là giới hạn model. Đo trên tài liệu thật: **15 bảng bị cắt oan** (phần lớn
dài 164–476 token, vẫn nằm gọn trong giới hạn 496). Trần cho bảng vì thế suy **từ chính
model**, không thêm tham số cấu hình mới.

### 5.33 Bảy lỗi chỉ lộ ra khi chạy trên tài liệu THẬT
Khi đem hệ thống (phát triển trên bộ tài liệu tự sinh) chạy trên 13 file thật (1221 trang,
5876 chunk):

| # | Lỗi | Hậu quả nếu không sửa |
|---|---|---|
| 1 | Ô gộp bị nhân bản | 131.115 ký tự cho một biểu mẫu vài trang → 14.759 sau khi sửa |
| 2 | Bảng dò nhầm nuốt tiêu đề slide | khối `[BẢNG]` giảm từ gần như mọi trang xuống 2 trang |
| 3 | Ảnh liên kết làm sập build | `ValueError` giết cả lần build index |
| 4 | Ảnh SVG biến mất âm thầm | 0/18 ảnh vào index, **không lỗi nào báo ra** |
| 5 | Nhận nhầm ngôn ngữ (§5.31) | câu tiếng Anh bị trả lời bằng tiếng Việt |
| 6 | BM25 phá chéo ngôn ngữ (§5.30) | MRR chéo ngôn ngữ 0.703 → 0.536 |
| 7 | Bảng bị cắt oan (§5.32) | 15 bảng mất dòng tiêu đề cột |

Lỗi #4 nguy hiểm nhất: không crash, không cảnh báo, chỉ âm thầm mất toàn bộ hình ảnh của một
file 9,4 MB (PowerPoint đời mới chèn ảnh SVG kèm PNG dự phòng mà `python-pptx` không lần ra
qua API shape — sửa bằng cách quét `rels` của từng slide).

**Bài học về đo đạc:** lần chạy đầu cho MRR 0.64 / Recall@K 0.78, tưởng hệ thống kém. Soi lại
thì **nhãn đáp án do chính tôi gán mới là thứ sai** — chúng trỏ vào trang mục lục / trang mở
chương thay vì trang nội dung thật:

| Chỉ số (chỉ đo truy xuất) | Nhãn ban đầu (sai) | Nhãn đã kiểm chứng |
|---|---|---|
| Recall@K | 0.78 | **0.96** |
| MRR | 0.64 | **0.98** |
| Đoạn đúng ở hạng 1 | 13/25 | **24/25** |

Đây là cạm bẫy kinh điển khi đánh giá RAG: nhãn sai thì mọi kết luận rút ra đều sai theo, và
sai theo hướng khiến người ta đi tối ưu nhầm chỗ.

### 5.34 Một phép đo sai suýt khiến cả tính năng bị loại
Chú thích ảnh bằng vision ban đầu bị đánh giá **~30 giây/hình** (2,4 tiếng cho 291 ảnh) nên
để mặc định TẮT. Con số đó sai: nó đo trên **đúng một lượt gọi** nên đã tính cả thời gian nạp
model. Đo lại khi model đã nạp: **0,7–2,9 giây/hình, trung bình 1,9** — 291 ảnh chỉ ~9 phút.

Bài học: chi phí khởi động một lần phải được tách khỏi chi phí biên trước khi kết luận.

### 5.35 Trần số đoạn là ảnh — khi một cải tiến làm hỏng chỗ khác
Bật chú thích ảnh xong, câu hỏi về nội dung hình tìm được ngay. Nhưng đo lại **toàn bộ** thì
Recall@K tụt 0,96 → 0,92 (nhóm slide tiếng Anh 0,97 → 0,86): 303 bản ghi ảnh mang mô tả dài
đã **chiếm mất suất `TOP_K`** của các trang văn bản đúng.

| Cấu hình | Recall@K | Slide tiếng Anh | Ảnh có tìm được? |
|---|---|---|---|
| Tắt vision (mốc so sánh) | 0.96 | 0.97 | **không** |
| Vision, không có trần | 0.92 | 0.86 | có |
| **Vision, trần 1** | **0.95** | **0.95** | **có** |

Ví dụ rõ nhất cho nguyên tắc *"đo lại TOÀN BỘ sau mỗi thay đổi"*: một tính năng hoạt động
đúng như thiết kế vẫn có thể làm hỏng chỗ khác.

### 5.36 Dọn mã ký tự PDF không giải mã được `(cid:NN)`
PDF nhúng font không kèm bảng ToUnicode (rất hay gặp với font toán) → `pdfplumber` trả về
nguyên mã `(cid:10)`. Đo trên Bishop: **1761/4197 chunk (42%)** dính rác này; lọc bỏ đưa
xuống còn **1 chunk**. Đây là **dọn nhiễu, không phải mất thông tin** — thông tin đã mất từ
khâu đọc file. Giới hạn còn lại: câu hỏi về *khái niệm* vẫn tốt, hỏi về *công thức cụ thể*
thì không có dữ liệu.

### 5.37 OCR dự phòng cho trang PDF đọc hỏng
Muốn khôi phục nội dung §5.36 làm mất thì phải đọc lại bằng mắt — hệ thống đã sẵn có một
"con mắt": chính model vision dùng cho chú thích ảnh. Không thêm dependency OCR nào.

| Nguồn | Kết quả đọc trang công thức của Bishop |
|---|---|
| `pdfplumber` | `Settingthisgradienttozerogives (cid:22) … w ML = ΦTΦ −1 ΦT t` |
| Vision OCR | `Setting this gradient to zero gives 0 = ∑N n=1 tnφ(xn)T − wT (…). … wML = (ΦTΦ)−1ΦTt` |

**Một cái bẫy suýt biến tính năng cứu nội dung thành tính năng phá nội dung:** prompt bản đầu
viết bằng tiếng Việt (tái dùng phong cách prompt chú thích ảnh) khiến model **tự dịch** cả
trang sách tiếng Anh sang tiếng Việt, dịch sai bét. Prompt OCR vì thế viết bằng tiếng Anh và
**cấm dịch tường minh**; có test khoá lại điều này.

Hệ thống chỉ OCR **đúng những trang đo được là đọc hỏng** (nhiều mã `(cid:)` so với số từ,
hoặc gần như không có chữ mà lại có ảnh), không quét bừa. *(Mặc định đã đổi thành BẬT — §5.49.)*

### 5.38 Kết quả cuối trên bộ tài liệu thật
13 tài liệu, 1221 trang, **5642 chunk**, 29 câu hỏi song ngữ.

> Số chunk của cùng bộ tài liệu này thay đổi giữa các mục (5909 ở §5.30, 5876 ở §5.33, 5642 ở
> đây) vì mỗi mục đo ở một thời điểm khác nhau, và chính các thay đổi ở khâu đọc/chia chunk đã
> làm con số đó đổi. **5642 là con số cuối cùng**, ứng với phiên bản code hiện tại.

> **Đọc kèm §5.46 trước khi diễn giải bất kỳ con số nào.** Chỉ Precision@K, Recall@K và MRR
> là tất định.

| Tách theo loại tài liệu | Số câu | Recall@K | Faithfulness | Relevance | Citation |
|---|---|---|---|---|---|
| Biểu mẫu có bảng (DOCX) | 4 | 1.00 | 0.75 | 1.00 | 0.47 |
| Slide tiếng Anh (CV) | 8 | 0.95 | 0.88 | 1.00 | 0.87 |
| Giáo trình dài tiếng Việt | 6 | 0.94 | 0.92 | 1.00 | 0.38 |
| Slide tiếng Việt (PPTX) | 4 | 0.87 | 1.00 | 1.00 | 0.81 |
| Sách dài tiếng Anh (Bishop) | 3 | 0.83 | **0.83** | 1.00 | 0.46 |

| Tách theo loại câu hỏi | Số câu | Recall@K | Faithfulness | Relevance | Citation |
|---|---|---|---|---|---|
| Đọc bảng | 3 | 1.00 | 0.67 | 1.00 | 0.58 |
| Kiểm chứng khẳng định sai | 3 | 1.00 | 1.00 | 1.00 | 0.63 |
| Truy xuất thường | 14 | 0.94 | 0.86 | 1.00 | 0.58 |
| Chéo ngôn ngữ | 5 | 0.83 | 1.00 | 1.00 | 0.80 |
| Từ chối (lạc đề) | 4 | — | đúng hành vi | — | — |

**Trung bình:** Precision@K 0.44 · Recall@K 0.80 · Faithfulness **0.86** (0.93 nếu loại 2 câu
bị cờ tự nghi ngờ đánh dấu) · Answer Relevance 0.90 · Citation 0.61 · 38,7 giây/câu.
Chỉ đo truy xuất (`--nhanh`, tất định): **Recall@K 0.93, MRR 0.98, 24/25 câu hạng 1**.

**Ba con số đã đổi so với bản trước:**

1. **Faithfulness nhóm sách tiếng Anh 0.33 → 0.83.** Bản trước ghi 0.33 kèm ghi chú "đây là
   lỗi của thước đo" — đúng, nhưng dừng ở đó là chưa xong việc. Nguyên nhân gốc nằm ở **dòng
   đầu tiên của luồng Ingestion** (§5.40): pdfplumber nuốt khoảng trắng khi đọc sách LaTeX
   nên giám khảo đối chiếu câu trả lời sạch với ngữ cảnh dính chữ rồi kết luận là "bịa". Sửa
   khâu đọc thì con số tự đúng lên. Bài học: một con số sai có thể có nguyên nhân nằm rất xa
   chỗ nó hiện ra, và "đây là lỗi thước đo" là chẩn đoán ĐÚNG nhưng chưa phải nguyên nhân.
2. **Citation 0.76 → 0.61 KHÔNG phải hệ thống kém đi** mà là thước đo nghiêm hơn (§5.28). Đo
   lại cùng lần chạy này theo cách CŨ cho **0.72** — nằm gọn trong dải dao động của chính
   metric đó (§5.46).
3. **Faithfulness trung bình suýt bị ghi thành 4.43** — xem §5.48.

### 5.39 Citation accuracy của nhóm biểu mẫu DOCX thấp — và đó là HAI lỗi khác nhau
Nhóm này Recall@K 1.00 và Faithfulness 1.00 nhưng Citation chỉ 0.63; đọc từng câu thì hai câu
đạt 1.00 và hai câu tụt hẳn (0.33 và 0.20) — hai nguyên nhân hoàn toàn khác nhau.

**Lỗi 1: thước đo đếm cả những trích dẫn PHỦ ĐỊNH.** Câu *"Các phần còn lại ([3],[4],[5],[6])
… không liên quan đến đề tài đang xét"* sinh ra 4 cặp mà giám khảo tất nhiên chấm 0 — điểm bị
kéo xuống bởi đúng cái hành vi thận trọng ta muốn khuyến khích. Đã thử sửa ở hai tầng, **chỉ
một tầng sống sót**:
- **Tầng đo** (`_MAU_CAU_LOAI_TRU_NGUON`) — GIỮ. Bộ mẫu cố ý HẸP: câu "Đề tài không thuộc
  lĩnh vực Y sinh mà thuộc Khoa học máy tính theo [2]" vẫn phải được tính (có test cả hai chiều).
- **Tầng prompt** (cấm model bình luận về đoạn không dùng) — **ĐÃ GỠ**: model 4B hiểu rộng
  thành "trích dẫn càng ít càng tốt" và bỏ hẳn trích dẫn ở một số câu (§5.46).

> Rút ra: khi một vấn đề có thể sửa ở **tầng đo** thay vì **tầng prompt**, hãy sửa ở tầng đo —
> ở đó nó không thể làm hỏng hành vi của model.

**Lỗi 2: chunking bảng làm hỏng câu trả lời — và câu trả lời đó SAI thật.** "Đề tài thuộc lĩnh
vực nào?" được trả lời *"thương mại điện tử"* trong khi biểu mẫu ghi rõ `LĨNH VỰC: Khoa học
máy tính`. Chỉ Citation accuracy phát hiện ra (Faithfulness vẫn 1.00 vì cụm đó đúng là có
trong ngữ cảnh — chỉ là ở chỗ khác). Nguyên nhân và cách sửa: §5.41.

**Cảnh báo về cách đọc bảng số liệu:** DOCX không có "trang" cố định (§5.13) nên cả file là
MỘT trang; Precision@K/Recall@K so khớp theo `(nguồn, trang)` vì vậy **suy biến** — chỉ cần
lấy về một chunk bất kỳ là Recall@K = 1.00. Con số 1.00 ở dòng "biểu mẫu có bảng" **không nói
lên điều gì về chất lượng truy xuất**.

### 5.40 Trang PDF bị dính chữ: đọc lại có điều kiện
`x_tolerance` mặc định 3 điểm; với font sát nhau (Computer Modern của sách LaTeX) khoảng cách
thật nhỏ hơn 3 nên **mọi khoảng trắng bị nuốt**: `whichareknownasthenormalequations…`. Hậu quả
lan ra toàn hệ thống: tokenizer băm chuỗi dính thành mảnh vô nghĩa, BM25 mất hoàn toàn từ
khoá, giám khảo không đối chiếu được nên chấm "bịa".

| File | Mặc định | `x_tolerance=1.5` | `x_tolerance=1.0` |
|---|---|---|---|
| Bishop (sách LaTeX) | **48.2%** dính | 1.6% | 1.6% |
| 8 file PDF còn lại | 0.0–2.6% | không đổi | CV-04 **hỏng thêm** |

Hai điều rút ra: chỉ đúng một file bị, và hạ tham số cho cả hệ thống thì bắt đầu có hại. Vì
vậy `_trich_text_thich_ung()` áp dụng cho **từng trang một**: đo tỉ lệ dính → dưới ngưỡng thì
không đụng → trên ngưỡng thì **dò lần lượt** các mức trong `CAC_X_TOLERANCE_THU` và chỉ nhận
mức nào vừa **giảm** độ dính vừa **không làm tăng** tỉ lệ vỡ từ. Không mức nào đạt thì giữ
nguyên bản gốc.

Kết quả: **5/45 trang mẫu bị đọc lại, tất cả đều là trang Bishop** (dính 33.1% → 0.0%, đồng
thời tỉ lệ vỡ từ cũng *giảm* 20.4% → 16.5%). Tám file còn lại không bị đụng tới một trang nào.
Phép đo chỉ đếm **chữ cái**, không đếm số và ký hiệu toán — nếu không thì mọi trang công thức
đều bị đọc lại oan (có test chốt). Xem §5.45 về cách chọn ngưỡng.

### 5.41 Bảng lớn: cắt theo HÀNG và lặp lại dòng tiêu đề
Bảng vượt cả giới hạn model bị đẩy thẳng vào splitter văn xuôi — mảnh đầu còn dòng tiêu đề,
**mọi mảnh sau chỉ còn các ô trần**. Trên `DeCuongNCKH.docx`, cả tờ khai là MỘT bảng dài 6.755
token (gấp 13 lần giới hạn): ô `LĨNH VỰC: Khoa học máy tính` nằm lẻ loi ở chunk 0, còn ~60
chunk sau là văn xuôi mang dấu `|` lạc lõng và **lặp đi lặp lại cụm "thương mại điện tử"**.
Model đọc 6 đoạn thì 5 đoạn nói về thương mại điện tử → đi theo số đông và trả lời sai (§5.39).

**Cách sửa** (`_cat_bang_giu_tieu_de`): gom từng HÀNG vào mảnh hiện tại chừng nào còn vừa giới
hạn, và **mỗi mảnh đều mở đầu bằng dòng tiêu đề + dòng gạch ngăn** nên mảnh nào cũng là bảng
Markdown hợp lệ đọc được độc lập. Hàng nào một mình đã vượt giới hạn (ô chứa cả bài văn) thì
trả về **dạng văn xuôi**, gắn nhãn `van_ban`. Kèm theo, `_bang_sang_markdown` **bỏ hẳn cột
rỗng ở mọi hàng** — biểu mẫu Word hay kẻ dư cột để căn lề.

### 5.42 Trả lời theo luồng (streaming) — sửa trải nghiệm, không sửa tốc độ
"Chậm" và "trông như bị treo" là hai vấn đề khác nhau; chỉ vấn đề thứ hai là do kiến trúc
hiển thị. Streaming **không làm model nhanh hơn một giây nào**:

| | Trước | Sau |
|---|---|---|
| Thấy dấu hiệu hệ thống đang chạy | không bao giờ (chỉ spinner) | ~2 giây |
| Thấy chữ đầu tiên của câu trả lời | bằng tổng thời gian | ~10 giây |
| Biết đang tra trên tài liệu nào | không | ~2 giây |

Vẽ lại giãn cách 0,12 giây chứ không theo từng token: mỗi lần vẽ Streamlit đẩy lại cả khối
markdown qua websocket.

**Hai quyết định đáng ghi lại:**
1. *Bóc thẻ suy luận phải là máy trạng thái, không phải regex.* Streaming không có chuỗi hoàn
   chỉnh nào — mảnh đang tới có thể cắt ngang giữa thẻ `<think>`, và nếu cứ thế đẩy ra màn
   hình thì người dùng nhìn thấy đúng phần suy luận mà cả hệ thống đang cố giấu.
   `_LocSuyLuanTheoLuong` giữ lại phần đuôi có thể là nửa cái thẻ (có test cho ca cắt đôi thẻ).
2. *Chỉ có MỘT đường gọi LLM.* `sinh_cau_tra_loi()` (dùng cho evaluation) **gom hết các mảnh**
   từ chính generator mà giao diện dùng. Hai đường song song sẽ trôi ra khỏi nhau, và khi đó
   **con số trong báo cáo không còn nói gì về thứ người dùng nhìn thấy** — một lỗi âm thầm,
   không có exception nào. Có test chốt hai chế độ cho ra cùng một chuỗi.

Số liệu độ trễ (`do_tre`) hiển thị dưới mỗi câu trả lời, để trên giao diện chứ không giấu
trong log: nó tách bạch hai con số hay bị gộp làm một — tổng thời gian (do model, không rút
ngắn được) và thời gian tới chữ đầu tiên (do kiến trúc hiển thị, đã rút ngắn được).

### 5.43 Đo độ tin cậy của chính LLM-as-judge
§5.38 phát hiện giám khảo chấm sai nhờ **đọc tay** — nghĩa là nếu không ai ngồi đọc thì con
số sai vẫn nằm im trong báo cáo. Ba việc đã làm để nó tự khai báo:

**(a) Cờ tự nghi ngờ.** Tính thêm một phép đo **tất định** (`do_bam_ngu_canh`): tỉ lệ cụm 4 từ
liên tiếp của câu trả lời xuất hiện nguyên văn trong ngữ cảnh, so sau khi **bỏ hết khoảng
trắng** (nên ngữ cảnh dính chữ không làm trượt). Hai điều sau không thể cùng đúng:

> giám khảo chấm ≤ 0.5 **và** mọi câu của câu trả lời đều có nguyên văn trong ngữ cảnh

Khi chúng cùng xảy ra, câu đó bị đánh dấu `!` và báo cáo in thêm dòng "Faithfulness nếu loại
các câu đáng ngờ". Cờ bật theo mức của **câu tệ nhất**, không phải trung bình: bản đầu dùng
trung bình và báo động giả ngay trên ca "đúng một nửa" (35%). Chuyển sang mức thấp nhất thì
hai nhóm tách hẳn — ca hỏng thật 0.53, ca đúng-một-nửa 0.00.

**(b) Bộ kiểm định giám khảo** (`kiem_dinh_judge.py`): 7 ca đã biết trước đáp án, chạy qua
đúng hàm `faithfulness()` mà `run_evaluation.py` dùng. Kết quả nên đưa vào báo cáo cạnh
Faithfulness: không phải "hệ thống đạt 0.90" mà "đạt 0.90, đo bằng một thước đo đã kiểm định
đúng 7/7 ca".

**(c) Phát hiện lớn nhất: `temperature=0` KHÔNG khiến giám khảo tất định.** Cùng một prompt,
cùng một ca, 8 lần liên tiếp: `[0, 1, 1, 1, 1, 1, 1, 1]` — dao động bằng đúng toàn bộ thang
điểm. Nguyên nhân: model vẫn sinh chuỗi suy luận dài trước khi kết luận, chỉ cần một mắt xích
rẽ khác là phán quyết lật ngược.

Hệ quả nặng hơn "một câu bị chấm sai": hai lần chạy trên **cùng một phiên bản code** ra hai
con số khác nhau, nên mọi so sánh "trước/sau khi sửa" không phân biệt được cải tiến với nhiễu
— mà toàn bộ quy trình tối ưu của đồ án dựa trên đúng phép so sánh đó.

Xử lý: chấm 3 lần lấy **trung vị**. Trung vị chứ không phải trung bình vì điểm dồn về hai cực
— trung bình của {0,1,1} ra 0.67, một con số không ứng với phán quyết nào. Số lần chấm **đối
xứng** (không phải "chỉ hỏi lại khi điểm thấp") vì hỏi lại một chiều sẽ đẩy điểm lên cao một
cách có hệ thống. Chi phí: +2 lượt LLM mỗi câu (~+10 phút cho bộ 29 câu).

### 5.44 IndexFlatIP chịu được tới bao nhiêu chunk — đo, không ước
Câu hỏi đúng của người phản biện không phải "có mở rộng được không" mà là **"đến bao nhiêu
chunk thì phải đổi?"**. `do_quy_mo_index.py` đo với `k=60` (đúng số ứng viên hệ thống thật lấy
về) và **đo từng câu một** (đo cả lô rồi chia sẽ cho con số đẹp hơn thực tế vài lần vì FAISS
dùng được BLAS ma trận-ma trận). Vector dùng để đo **có gom cụm theo chủ đề** (cosine nội cụm
≈ 0.8, đúng khoảng đã đo trên corpus thật) — vector ngẫu nhiên thuần trong 768 chiều gần như
luôn vuông góc nên là ca xấu nhất cho mọi thuật toán gần đúng, đo bằng dữ liệu đó thì HNSW ra
recall 0.17, một con số không nói gì về hành vi thật.

| Số chunk | FlatIP p95 | RAM | HNSW p95 / recall | IVFFlat p95 / recall |
|---|---|---|---|---|
| 10.000 | 1.4 ms | 29 MB | 0.8 ms / 0.87 | 0.2 ms / 0.90 |
| 50.000 | 7.2 ms | 146 MB | 1.2 ms / 0.74 | 1.1 ms / 0.96 |
| 100.000 | 12.9 ms | 293 MB | 1.5 ms / 0.61 | 1.3 ms / 0.97 |

- **Tốc độ:** ~129 µs cho mỗi 1.000 chunk → ngưỡng 200 ms là **≈1,5 triệu chunk**. Corpus hiện
  tại ở **0,4%** ngưỡng đó.
- **Bộ nhớ — ràng buộc THẬT:** 2,9 MB mỗi 1.000 chunk → mốc thực tế là **≈700.000 chunk
  (≈2 GB RAM)**, tức khoảng **145.000 trang** với mật độ 4,8 chunk/trang. RAM chạm trần trước
  độ trễ khoảng gấp đôi.
- **Khi phải đổi, chọn IVFFlat chứ không phải HNSW**: ở 100.000 chunk cả hai đều nhanh hơn
  Flat ~10 lần, nhưng IVF giữ recall 0.97 còn HNSW chỉ 0.61 — và recall HNSW **tụt dần khi
  corpus to lên** nếu giữ nguyên `efSearch` (0.87 → 0.74 → 0.61).
- Đổi index thì **bắt buộc** chạy lại `run_evaluation.py`: đoạn bỏ sót hoàn toàn có thể là
  đoạn chứa câu trả lời. Nhanh hơn mà trả lời sai thì không phải cải tiến.

### 5.45 Chống "chỉnh cho vừa bộ tài liệu test"
Cơ chế ở §5.40 và §5.41 đều được phát hiện nhờ **một** tài liệu cụ thể — cách phát hiện lỗi
tốt, nhưng cũng là cách sinh ra lỗi mới. Bốn việc đã làm:

**(a) Thay hằng số bằng phép dò có điều kiện chấp nhận.** Bản đầu chốt cứng `x_tolerance=1.5`
— con số đo trên đúng cuốn Bishop. Nay mỗi trang tự chọn mức của riêng nó theo hai điều kiện
phải thoả đồng thời: *giảm được* tỉ lệ dính **và** *không làm tăng* tỉ lệ vỡ từ. Điều kiện thứ
hai trước đó tồn tại dưới dạng "người viết code đã tự tay đối chiếu 9 file PDF" — một phép
kiểm chạy đúng một lần rồi biến mất; nay nó là code chạy trên mọi trang của mọi tài liệu.
Hệ quả: an toàn **theo cấu trúc**, không nhờ ngưỡng chọn khéo — trang không thật sự dính thì
không mức nào làm nó "đỡ dính" được nên bản gốc luôn được giữ.

**(b) Ngưỡng phát hiện chọn bằng đo, và lần nào cũng phải nới rộng hơn trực giác.**

| Độ dài cụm | "Backpropagation" | "internationalization" | Trang Bishop | 8 PDF còn lại |
|---|---|---|---|---|
| ≥ 15 | **23.4%** (oan) | 81.6% (oan) | 48.2% | 0.0–2.6% |
| ≥ 20 | 0.0% | **81.6%** (vẫn oan) | 41.7% | 0.0–1.5% |
| ≥ 25 | 0.0% | 0.0% | 36.2% | **0.0%** |

Phát hiện quan trọng hơn con số: **nâng độ dài cụm không đủ** — một câu ngắn chứa đúng một từ
dài hợp lệ vẫn cho tỉ lệ cao ở mọi ngưỡng vì mẫu quá nhỏ. Phải thêm `SO_KY_TU_TOI_THIEU_DE_DO`:
dưới 200 ký tự chữ thì không kết luận gì.

**(c) Kiểm chứng trên bộ tài liệu ĐỘC LẬP** (`tao_tai_lieu_mau.py` sinh 6 tài liệu + 26 câu
hỏi riêng, không dùng để tinh chỉnh gì): toàn bộ 24 câu có đáp án đạt **Recall@K 1.00**, MRR
0.95, 22/24 câu đúng ở hạng 1, 2 câu lạc đề bị chặn đúng, và **0 chunk vượt giới hạn model**.

**(d) Test theo HÌNH DẠNG, không theo tài liệu** (`tests/test_khai_quat_tai_lieu.py`): bảng
chỉ có tiêu đề, bảng một hàng, bảng 8 cột, bảng toàn ô rỗng, bảng thiếu dòng gạch ngăn, bảng
có tiêu đề dài hơn cả ngân sách token, ô chứa nguyên một bài văn. Yêu cầu chung: **không mất
nội dung** và **không sinh chunk vượt giới hạn model**. Chính bộ test này phát hiện hai lỗi mà
tài liệu thật không chạm tới.

### 5.46 Cả quy trình đánh giá đang so sánh hai lần rút thăm
Phát hiện quan trọng nhất của vòng cải tiến này. Hỏi cùng một câu, cùng index, cùng prompt,
4 lần liên tiếp, ghi lại số `[n]` mà câu trả lời gắn vào:

```
Nhà nước có những đặc điểm gì?        -   -      -   1        1/4 lần có dẫn nguồn
Đề tài thuộc lĩnh vực nào?            -   123456 -   123456   2/4
Vi phạm pháp luật gồm dấu hiệu nào?   -   135    -   -        1/4
```

Ghép với §5.43 (giám khảo chấm cùng một ca 8 lần cho `[0,1,1,1,1,1,1,1]`):

> Cả hai đầu — **hệ thống bị đo** và **thước đo** — đều không tất định. Một dòng "Citation
> 0.76 → 0.67" giữa hai lần chạy đơn lẻ **không phân biệt được cải tiến thật với dao động**.

Nguy hiểm vì toàn bộ quy trình tối ưu dựa trên bảng so sánh "trước/sau". Nếu độ dao động lớn
hơn mức cải tiến, bảng đó sẽ đều đặn báo cáo những "cải tiến" hoàn toàn ngẫu nhiên. Không phải
rủi ro lý thuyết: chính trong vòng này, một lần chạy báo Citation tụt 0.10 và suýt dẫn tới
việc vặn prompt nhiều lần để đuổi theo nhiễu.

**Đã xử lý được một nửa.** Phía thước đo có cách chữa rẻ (trung vị 3 lần, +10 phút); phía hệ
thống bị đo thì không (nhân ba phần đắt nhất, 1,5 giờ → 4,5 giờ). Quyết định là **ghi rõ giới
hạn thay vì giả vờ không có**:

| Metric | Tất định? | Diễn giải chênh lệch |
|---|---|---|
| Precision@K, Recall@K, MRR | **Có** | So sánh trực tiếp được |
| Faithfulness, Answer Relevance | Đã bớt dao động (trung vị 3 lần) | Chênh lệch nhỏ vẫn cần dè dặt |
| **Citation accuracy** | **Không** — dao động mạnh nhất | Chênh dưới ~0.1 **không nên diễn giải là gì cả** |

**Một hệ quả về cách sửa prompt.** Quy tắc cấm model bình luận về đoạn không dùng (§5.39) đạt
đúng mục tiêu đề ra, nhưng model 4B hiểu rộng thành "trích dẫn càng ít càng tốt" và **bỏ hẳn
trích dẫn ở một số câu**. Đo lại đúng ba câu đó, 4 lần mỗi câu: có quy tắc 4/12 (33%) → đã gỡ
**6/12 (50%)**. Với n=12 thì chênh lệch này chưa đủ kết luận, và quyết định gỡ không dựa vào
con số đó — nó dựa vào việc quy tắc kia đánh đổi **thứ quan trọng nhất của hệ thống** (khả
năng kiểm chứng) lấy một câu trả lời gọn gàng hơn.

> Bài học: với model nhỏ, **mỗi điều cấm thêm vào prompt đều có nguy cơ dập luôn hành vi mình
> đang muốn giữ**. Khi một vấn đề có thể sửa ở tầng đo thay vì tầng prompt, hãy sửa ở tầng đo.

### 5.47 Bố cục kiểu ứng dụng chat — và chỗ cố ý làm khác
Bố cục hai cột cạnh nhau khiến cột chat — thứ người dùng nhìn 95% thời gian — chỉ còn 3/4 màn
hình. Bố cục mới: **thanh bên + một cột đọc căn giữa + ô nhập ghim đáy**, vì ba lý do dùng
được: thanh bên **thu gọn được** (thay hoàn toàn nút toàn màn hình tự làm, §5.16); cột đọc hẹp
(~48rem) là có chủ đích vì dòng chữ dài quá ~80 ký tự khiến mắt mỏi khi nhảy dòng; và
`st.chat_input` **phải nằm ở tầng ngoài cùng** của script mới ghim được đáy — chính ràng buộc
kỹ thuật này khiến bố cục hai cột không bao giờ cho được cảm giác của một app chat thật.

**Chỗ cố ý làm KHÁC ChatGPT**, vì đây là hệ thống RAG chứ không phải chatbot:
- **Thanh bên là NGUỒN TÀI LIỆU, không phải danh sách hội thoại cũ** — lịch sử vốn không lưu
  qua nhiều phiên (§5.9) nên danh sách hội thoại sẽ luôn rỗng; thứ cần quản lý là *tài liệu
  nào đang được dùng để trả lời*, và mỗi checkbox đổi trực tiếp phạm vi truy xuất.
- **Dưới mỗi câu trả lời có nguồn + số liệu độ trễ** — không có trong app chat thường, và
  chúng là toàn bộ lý do tồn tại của hệ thống.
- **Câu trả lời KHÔNG bọc trong bong bóng chat** — nó là văn bản dài có tiêu đề, bảng, ảnh;
  nhốt vào bong bóng chỉ làm hẹp chỗ đọc. Chỉ câu HỎI mới có bong bóng.
- **Gợi ý câu hỏi bám vào ĐÚNG tài liệu đang bật**: một gợi ý hỏi về thứ không có trong corpus
  sẽ nhận về câu từ chối — ấn tượng đầu tiên tệ nhất với một hệ thống mà điểm mạnh là *không bịa*.

**Một chi tiết về đường vào của câu hỏi.** Hai lối đặt câu hỏi (ô nhập và nút gợi ý) đều đi qua
đúng một hàm `_dat_cau_hoi()`, và hàm đó chỉ ghi vào state rồi `st.rerun()` NGAY, để việc gọi
LLM xảy ra ở lần chạy kế tiếp — lúc mọi widget đã render ở trạng thái `disabled`. Nếu một lối
vào gọi thẳng LLM thì một cú bấm bất kỳ trong lúc chờ sẽ khiến Streamlit huỷ ngang lần chạy
đang dở và câu trả lời mất trắng — bug đã gặp thực tế.

### 5.48 Giám khảo trả điểm ngoài thang — và vì sao JSON Schema không cứu được
Bảng in ra `Faithfulness trung bình = 4.43` (Faithfulness là tỉ lệ, luôn trong [0,1]). Truy
vào từng câu thấy `100.00` và `5.00` — giám khảo đổi sang thang **phần trăm** và thang **1–5**;
cả hai câu trả lời đều đúng, model chỉ hiểu sai đơn vị. 27 câu còn lại tổng tối đa là 27, riêng
hai câu lạc thang đóng góp 105 → trung bình nhảy gấp năm lần **nhưng vẫn là một con số nằm gọn
trong ô của nó**, không ngoại lệ nào ném ra, không dòng log nào đỏ lên.

**Vì sao JSON Schema không đủ:** prompt đã ghi rõ thang 0–1 và schema khai `minimum: 0,
maximum: 1`. Ollama **dịch JSON Schema sang một grammar**, mà grammar chỉ ràng buộc *hình
dạng* (đây là một số), **không ràng buộc khoảng giá trị**. `minimum`/`maximum` vì vậy chỉ là
tài liệu, không phải hàng rào.

> Bài học chung: structured output bảo đảm *parse được*, không bảo đảm *hợp lệ* — phần hợp lệ
> vẫn phải tự kiểm bằng code.

**Không chọn cách kẹp giá trị** (100 → 1.0) vì đó là *đoán ý* model rồi ghi kết quả đoán vào số
liệu đánh giá. Thay vào đó mẫu lạc thang bị **loại hẳn** khỏi phép lấy trung vị (Faithfulness
vốn đã chấm 3 lần nên còn đủ mẫu tốt để quyết). Chỉ khi **cả 3 lần đều lạc thang** mới lùi về
kẹp, kèm log mức `error` — đó là dấu hiệu đã đến lúc đổi `JUDGE_MODEL`.

### 5.49 Tài liệu chưa từng thấy: bốn lỗ hổng chỉ lộ ra khi người dùng nạp tài liệu thật
Người dùng nạp một giáo trình 383 trang; index build "thành công" với 379 chunk, không lỗi nào
được ném ra. Nhưng mỗi chunk có nội dung đúng bằng chuỗi `[HÌNH]` — 6 ký tự, **giống hệt nhau ở
cả 379 chunk**. Toàn bộ "kho tri thức" của cuốn sách là 379 mẩu rỗng. Nguyên nhân: **PDF scan**
(cả 383 trang đều là ảnh, 0 ký tự text), cộng bốn khiếm khuyết mỗi cái riêng lẻ đều "chỉ hơi
dở": OCR mặc định TẮT · ảnh không chú thích vẫn thành chunk · ảnh chụp cả trang bị trích như
hình minh hoạ · không ai báo tài liệu không đọc được (số chunk khác 0 nên "trông ổn").

> Khiếm khuyết đầu là bài học đắt nhất: **một mặc định được chọn dựa trên một bộ tài liệu cụ
> thể chính là định nghĩa của việc chỉnh cho vừa bộ test.** Lập luận "chi phí lớn, lợi ích hẹp"
> (§5.37) đúng với bộ tài liệu lúc đó và sai hoàn toàn với tài liệu kế tiếp.

Nay OCR **mặc định BẬT**, và điều đó an toàn chứ không phải đánh đổi: nó chỉ chạy cho trang ĐO
ĐƯỢC là không đọc được, nên tài liệu có lớp text bình thường tốn **đúng 0 chi phí**. Đo trên
chính cuốn sách đó: OCR phục hồi **2.000–2.600 ký tự mỗi trang**, đọc đúng cả số hiệu văn bản
(*"Nghị định số 88/2006/NĐ-CP"*), ~8 giây/trang.

### 5.50 Ba lần đặt ngưỡng mà không đo — và cách sửa cuối cùng
- **Lần 1** — "ảnh chụp cả trang phủ ≥ 85% diện tích": trượt ngay trên tài liệu đầu tiên (ảnh
  scan thật chỉ phủ **81.2%** vì sách có lề ~50pt).
- **Lần 2** — hạ xuống 60% kèm "trang có ít chữ": bắt được sách scan nhưng **nuốt mất 5 ảnh**
  nền của slide tiêu đề và bìa sách.
- **Lần 3** — dò cột bằng "khoảng trống dọc giữa trang": nhận nhầm **100% số trang** của giáo
  trình Pháp luật thành 2 cột (thứ nó tưởng là rãnh giữa cột thực ra là **lề trang**).

Điểm chung: ngưỡng được đặt trên một đại lượng *gần đúng* (diện tích, khoảng trống) thay vì
trên thứ thật sự cần biết. Lời giải trong cả ba trường hợp đều là **đổi tín hiệu**: ảnh chụp
trang chữ ⟶ **hỏi OCR** (ảnh trang sách cho ra cả nghìn ký tự, ảnh nền slide thì không —
quyết định phải đi SAU phép đo); rãnh giữa cột ⟶ **đòi có chữ ở cả hai bên** và chỉ xét bên
trong khối chữ. Đo lại sau khi đổi tín hiệu: **0 ảnh bị nuốt nhầm, 0 trang bị nhận nhầm**.

### 5.51 Bố cục nhiều cột: hỏng ngay từ dòng đầu tiên của luồng đọc
pdfplumber đọc theo **dòng ngang chạy suốt bề ngang trang** nên với trang 2 cột nó nối câu cột
trái thẳng vào câu cột phải — hai điều luật khác nhau dính thành một câu, **mọi chunk sinh ra
từ trang đó đều vô nghĩa** và không có dấu hiệu nào để nhận ra.

**Cách sửa** (`_cac_cot_cua_trang` + `_text_theo_cot`): chiếu mọi từ lên trục ngang, tìm rãnh
trống dọc **nằm trong khối chữ** và có **lượng chữ đáng kể ở cả hai bên**, đọc từng cột rồi nối
theo thứ tự trái → phải. Dấu hiệu hình học nên không phụ thuộc ngôn ngữ. Chỉ dò khi trang
**không có bảng** (bảng nhiều cột cũng tạo rãnh dọc, và bảng vốn đã được tách vùng riêng, §5.26).

### 5.52 Text box trong DOCX: nội dung biến mất không dấu vết
`Paragraph.text` chỉ gom text của các run trực tiếp; nội dung text box nằm sâu trong
`<w:drawing>` → `<wps:txbx>` → `<w:txbxContent>` nên trả về chuỗi rỗng. Đáng sửa vì **sơ đồ,
khung "Lưu ý", trích dẫn nổi bật, chú thích bên lề** đều hay đặt trong text box — thường là chỗ
cô đọng nhất của trang. Dò bằng đường dẫn XML `.//w:txbxContent//w:t` thay vì theo namespace
riêng của `<wps:txbx>`, vì text box do các phiên bản Word khác nhau tạo ra nằm dưới nhiều
namespace (wps, `v:textbox` của VML cũ) nhưng tất cả đều chứa `<w:txbxContent>`.

### 5.53 Một file hỏng không được làm sập cả lần build
Với hệ thống mà người dùng **tự nạp tài liệu bất kỳ vào**, gặp file đọc không được là chuyện
bình thường chứ không phải ngoại lệ. Nay mỗi file được đọc trong lưới đỡ riêng: file hỏng bị
bỏ qua, tên file và loại lỗi ghi rõ, cuối lần build có dòng "N/M tài liệu KHÔNG vào được
index: …". Bắt `Exception` rộng là có chủ đích (các thư viện ném rất nhiều loại lỗi khác nhau
tuỳ file hỏng kiểu gì) — điều quan trọng là **không nuốt lỗi**.

### 5.54 Trình bày phỏng đoán như thể là nguồn — lỗi trích dẫn nguy hiểm nhất
Khi model không gắn số `[n]` nào, bản trước lặng lẽ lấy đoạn điểm cao nhất và hiển thị dưới
nhãn **"📎 Nguồn:"**. Người đọc tin rằng câu trả lời dựa trên trang đó, trong khi **không ai
biết nó dựa trên gì** — kể cả hệ thống. Với một hệ thống mà giá trị cốt lõi là *kiểm chứng
được*, đây nặng hơn cả việc không hiển thị nguồn nào: không có nguồn thì người đọc biết mình
phải tự tra; có một nguồn sai thì họ yên tâm nhầm.

Không phải trường hợp hiếm: **4/29 câu trả lời thật không gắn số nào**; đo lặp lại cùng một câu
4 lần thì tỉ lệ tuân thủ chỉ **50%** (§5.46). Nay vẫn hiển thị đoạn liên quan nhất nhưng kèm cờ
`la_suy_doan` để giao diện nói thật: *"Câu trả lời không tự dẫn nguồn — đoạn dưới là đoạn liên
quan nhất do hệ thống chọn, KHÔNG chắc là căn cứ đã dùng"*.

### 5.55 Chỉ báo "bám nguồn" ở tầng chạy thật — và vì sao chỉ hiện khi CAO
`do_bam_ngu_canh()` tất định, không gọi model, chạy trong mili giây — nên không có lý do gì để
người dùng thật không được thấy nó: *"✓ 78% nội dung câu trả lời trùng nguyên văn với đoạn
trích đã dẫn"*.

**Chỉ hiện khi CAO, không hiện cảnh báo khi thấp** — có chủ đích, vì phép đo này chỉ nói được
MỘT chiều: cao ⟹ bằng chứng mạnh rằng câu trả lời không bịa; thấp ⟹ **không kết luận được gì**,
vì một câu trả lời diễn đạt lại bằng lời của mình (hoàn toàn hợp lệ, thường là điều mong muốn)
cũng cho mức thấp. Hiện dấu hiệu "bám nguồn thấp" sẽ khiến người đọc nghi ngờ oan đúng những
câu trả lời viết tốt nhất. Cùng lý do đó, con số này **tuyệt đối không được dùng để tự động từ
chối** một câu trả lời.

Hàm sống ở `rag/citation.py` và `evaluation/metrics.py` import lại, **không có bản sao** — hai
bản sẽ trôi khỏi nhau và khi đó con số trong báo cáo nói về một thứ khác với con số người dùng
nhìn thấy. Có test khoá: `ban_runtime is ban_danh_gia`.

### 5.56 Kết luận phải viết SAU lập luận, không phải trước
Hỏi *"Tôi nhớ là doanh nghiệp tư nhân có tư cách pháp nhân, đúng không?"*, hệ thống trả lời:

> **ĐÚNG** … Giải thích: doanh nghiệp tư nhân **không có tư cách pháp nhân** …

**Phán quyết mâu thuẫn với chính lập luận ngay bên dưới nó** — với tính năng chống a dua
(§5.22) thì đây là hỏng đúng chỗ quan trọng nhất, vì người đọc lướt nhìn chữ "ĐÚNG" là tin ngay.
Nguyên nhân là **thứ tự bố cục bắt buộc** trong system prompt: `KẾT LUẬN` → `Căn cứ` →
`Giải thích` bắt model chốt phán quyết khi chưa đối chiếu gì, rồi lập luận đúng nhưng không quay
lại sửa nhãn đã lỡ viết.

Sửa: đảo thành `Căn cứ` → `Đối chiếu` → `KẾT LUẬN`, kèm lý do ngay trong prompt và một bước tự
kiểm. Đo lại 3 lần (vì hành vi model không tất định): **3/3 lần kết luận SAI** (đúng), đều kèm
số đoạn trích. Nhãn phán quyết model tự đặt tên khác nhau giữa các lần — **cố ý không siết
prompt thêm để ép nhãn**, theo đúng bài học §5.46.

### 5.57 Ngưỡng từ chối: kiểm chứng nó KHÔNG bị overfitting
`NGUONG_DIEM_RERANK_TOI_THIEU = 0.001` là tham số dễ overfitting nhất trong hệ thống: điểm
cross-encoder không được hiệu chuẩn sẵn nên phân bố hoàn toàn có thể dịch chuyển khi đổi miền
tài liệu. Kiểm trên miền **hoàn toàn mới** (luật kinh tế Việt Nam):

| Nhóm câu hỏi | Điểm rerank cao nhất |
|---|---|
| Đúng chủ đề (4 câu) | 0.62 – 0.999 |
| Lạc đề (4 câu) | 0.000016 – 0.000017 |

Hai nhóm cách nhau **~37.000 lần** và ngưỡng 0.001 nằm gọn ở giữa → khoảng cách đó là thuộc
tính của **model cross-encoder**, không phải của corpus. Một kết quả âm tính đáng ghi lại:
không phải hằng số nào cũng là overfitting, và cách duy nhất để biết là đi đo trên miền mới.

### 5.58 Truy xuất mù trước lịch sử hội thoại — và vì sao cách chữa TẤT ĐỊNH thắng
`truy_xuat()` chỉ encode đúng chuỗi câu hỏi hiện tại. Với câu NỐI TIẾP:

    Người dùng: "Vi phạm pháp luật gồm những dấu hiệu nào?"
    Hệ thống:   "... bốn dấu hiệu: hành vi trái pháp luật, có lỗi, ..."
    Người dùng: "Thế còn dấu hiệu thứ hai thì sao?"

câu cuối không chứa chủ đề nào. Vector của nó không trỏ tới vùng nào, BM25 không có từ khoá
để bám, và cross-encoder chấm mọi đoạn về gần 0 — mà điểm rerank còn là **cơ chế từ chối**
(§5.29), nên câu hợp lệ này còn có nguy cơ **bị từ chối oan**. Lịch sử chat có hiện trên màn
hình nhưng nằm ở tầng giao diện, chưa bao giờ đi vào truy xuất.

Phải tách bạch với §5.9: "không lưu lịch sử qua nhiều phiên" là một quyết định PHẠM VI. Còn
đây là chuyện xảy ra **ngay trong một phiên**, tức một khiếm khuyết thật. Hai chuyện này rất
hay bị gộp làm một khi đọc README.

**Đã thử cách kinh điển trước — query rewriting bằng LLM — và nó THẤT BẠI, đo được:**

| Model | Kết quả trên 7 ca |
|---|---|
| `qwen3:4b` | **0/7**. Luôn sinh chuỗi suy luận dài trước khi trả lời (§5.23): `num_predict=200` bị tiêu hết lúc "nghĩ", `content` về RỖNG. Nâng lên 1500 vẫn rỗng (thinking đã 5891 ký tự). Đủ chỗ thì cần ~3000 token ≈ **+30 giây mỗi câu nối tiếp**. |
| `qwen2.5vl:3b` | Nhanh (không có chế độ suy luận) nhưng sai: một ca chép y nguyên câu gốc, một ca trả về đúng câu hỏi CŨ — mất hẳn phần "thứ hai". |

**Cách thứ hai — contextualization — làm đúng việc cần làm mà không cần model nào:** ghép các
CÂU HỎI trước vào câu hiện tại thành truy vấn chính, rồi để RRF hợp nhất với nhánh câu gốc.
"Vi phạm pháp luật gồm những dấu hiệu nào? Thế còn dấu hiệu thứ hai thì sao?" mang đủ từ khoá
chủ đề để vector trỏ đúng vùng.

**Đo trên index thật** (bài báo FCN, 458 chunk). Lấy chính kết quả truy xuất của câu hỏi ĐẦY
ĐỦ làm chuẩn vàng — nhờ vậy phép đo chạy được trên bất kỳ corpus nào mà không cần gán nhãn
tay, và nó **tất định** nên chênh lệch là chênh lệch thật chứ không phải dao động (§5.46):

| Câu nối tiếp | Trùng chuẩn vàng | Điểm rerank |
|---|---|---|
| "Giải thích thêm đi" | 0/4 → **4/4** | 0.0328 → **0.8832** |
| "Cho ví dụ" | 0/4 → **4/4** | 0.2698 → **0.8324** |
| "Tell me more" | 0/4 → **4/4** | 0.2055 → **0.8377** |
| "Cái đó cụ thể là thế nào?" | 1/4 → **4/4** | 0.0219 → **0.8831** |
| **Tổng** | **1/16 → 16/16** | |

Con số đáng lo nhất không phải cột trùng khớp mà là cột rerank: 0.0219 và 0.0328 nằm sát
ngưỡng từ chối, tức những câu nối tiếp hoàn toàn hợp lệ đang ở ranh giới **mất hẳn câu trả
lời**, chứ không chỉ là "lấy nhầm đoạn".

**Ba lý do khiến đây là lựa chọn đúng, không phải lựa chọn tạm:**

1. **TẤT ĐỊNH.** §5.46 kết luận chỉ Precision@K/Recall@K/MRR so sánh được giữa hai lần chạy.
   Một bước đứng chắn trước toàn bộ truy xuất mà lại gọi LLM thì thêm một nguồn dao động nữa
   vào đúng chỗ tệ nhất.
2. **GẦN NHƯ MIỄN PHÍ.** Một lần encode (~30ms), không đụng tới độ trễ §5.42 đã tốn công kéo xuống.
3. **AN TOÀN THEO CẤU TRÚC.** Câu gốc vẫn là một nhánh riêng trong RRF, nên xấu nhất là "thứ
   hạng nhiễu đi", không phải "mất kết quả đúng" — nguyên tắc của §5.45(a).

Đường LLM được **giữ lại nhưng mặc định TẮT**, đúng tiền lệ §5.30 với BM25: kết quả âm tính
trên model này không có nghĩa âm tính với model khác.

**Ba chi tiết nhỏ quyết định chất lượng:**

- **Chỉ ghép CÂU HỎI, không ghép câu trả lời.** Câu trả lời trước dài hơn câu hỏi hàng chục
  lần nên sẽ lấn át vector truy vấn — mà nội dung đó đã nằm sẵn trong tài liệu, nên nhánh này
  sẽ chỉ kéo về đúng những đoạn vừa dùng ở lượt trước, không tìm được phần MỚI đang được hỏi.
- **Câu hiện tại đặt SAU cùng**: nó là thứ cần được nhấn, phần ghép thêm chỉ nêu chủ đề.
- **Prompt sinh câu trả lời nhận khối ngữ cảnh dán nhãn riêng**, kèm câu "ĐÂY KHÔNG PHẢI
  nguồn thông tin, tuyệt đối không trích dẫn", và chỉ chứa câu hỏi chứ không chứa câu trả lời
  cũ. Câu trả lời cũ là lời của chính model; cho nó trích lại lời mình như thể là tài liệu
  thì trích dẫn mất sạch ý nghĩa.

**Tầng nhận diện: 10/10 trên bộ ca có nhãn**, và hướng ưu tiên ĐẢO NGƯỢC so với
`la_cau_hoi_kiem_chung()` (§5.22) — ở đó bỏ sót thì rẻ, ở đây bỏ sót thì truy xuất trượt hẳn.
Một luật dự phòng theo ĐỘ DÀI CÂU đã bị bộ ca có nhãn bác bỏ ngay và phải gỡ bỏ: tiếng Việt
viết rời từng âm tiết nên câu hỏi hoàn chỉnh thường xuyên chỉ có 6-8 "từ" ("Nhà nước có những
đặc điểm gì?"), độ dài không mang thông tin gì. Đúng bài học §5.50: đổi tín hiệu, đừng chỉnh
ngưỡng.

**Giao diện phải NÓI RA việc đã ghép ngữ cảnh** (*"Hiểu đây là câu hỏi nối tiếp, nên đã tra
kèm ngữ cảnh: …"*). Đây là một phỏng đoán của hệ thống về ý người dùng, và trình bày phỏng
đoán như thể là sự thật đúng là lỗi §5.54 đã phải sửa một lần rồi. Hiện ra thì người dùng
thấy hệ thống nối nhầm và gõ lại câu đầy đủ; giấu đi thì họ nhận một câu trả lời
đúng-nhưng-cho-câu-hỏi-khác mà không hiểu vì sao.

### 5.59 Đối chiếu CHÉO các nguồn — và `think=False` được minh oan đúng một chỗ
Toàn bộ phần còn lại của hệ thống xử lý mỗi đoạn trích ĐỘC LẬP: xếp hạng độc lập, đặt cạnh
nhau trong prompt, rồi để LLM viết một câu trả lời gộp. Không bước nào hỏi "các đoạn này có
nói ngược nhau không".

Với tài liệu học tập thật thì đó là chuyện thường xuyên: giáo trình in năm cũ và slide cập
nhật ghi khác con số, hai môn định nghĩa khác nhau cùng một khái niệm, quy định cũ và mới
khác nhau về cùng một thủ tục. LLM đọc cả hai rồi lặng lẽ chọn một bên (hoặc trộn lẫn), và
người đọc mất đúng thông tin quan trọng nhất: **"hai nguồn của bạn đang không thống nhất"**.
Đó là thứ họ không tự thấy được — mở từng file riêng thì mỗi file đều nhất quán với chính nó.

**Hai tầng, đúng khuôn retrieval.** Chấm mọi cặp bằng LLM là O(n²): C(TOP_K, 2) cặp — 6 cặp
với `TOP_K=4`, 15 cặp nếu ai đó nâng lại lên 6 — tức chừng ấy lượt gọi model sau MỖI câu trả lời. Nên tầng 1 lọc tất định (khác nguồn + cùng chủ đề theo
cosine + có dấu hiệu bất đồng bề mặt), tầng 2 chỉ chấm vài cặp sống sót. Đại đa số lượt hỏi
không có cặp nào qua tầng 1 → tốn đúng 0 lượt LLM. Bước này chạy **sau khi câu trả lời đã
hiện xong** nên không đụng tới thời gian chờ chữ đầu tiên (§5.42).

**Số viết bằng chữ là chỗ suýt làm cả bộ lọc vô dụng.** Bản đầu chỉ bắt chữ số bằng regex —
và ca mẫu của chính đồ án, *"Nhà nước có NĂM đặc điểm"* đối lại *"có BỐN đặc điểm"* (§5.22),
không có lấy một chữ số nào. Test bắt được đúng chỗ này. Nay số viết bằng chữ được quy về
cùng dạng với chữ số; cố ý loại "một" (mạo từ) và "tư" ("tư nhân"/"tư cách").

**`think=False` KÈM `format=<schema>` — và vì sao nó không mâu thuẫn với §5.23.**
§5.23 kết luận "tuyệt đối không truyền `think=False`", và kết luận đó vẫn đúng **với đầu ra
tự do**: lúc ấy nó không tắt suy luận mà chỉ tắt việc TÁCH suy luận ra, nên cả chuỗi "Okay,
let me figure out..." đổ thẳng vào `content`. Nhưng khi có `format`, Ollama ép sinh theo một
grammar JSON — model **không thể** sinh văn xuôi tự do nữa vì văn xuôi không phải JSON hợp lệ.
Chính grammar trở thành thứ chặn suy luận.

| Cách gọi (cùng một cặp đoạn) | thinking | content | thời gian |
|---|---|---|---|
| `format=schema`, không truyền `think` | 1795 ký tự | **RỖNG** | 5.8s |
| `format=schema` + `think=False` | 0 | JSON đúng | **1.1s** |

Không có nó thì tính năng **không chạy**: bản đầu đo được 0/4 ca mâu thuẫn thật, vì
`num_predict` bị suy luận ăn hết và `content` về rỗng ở mọi ca. Đây cũng là gợi ý đáng thử
cho `evaluation/metrics.py` (LLM-as-judge cũng dùng `format=` mà không truyền `think=False`) —
nhưng chưa đổi, vì làm vậy sẽ khiến số liệu §5.38 không còn so được với bản đã báo cáo.

**Thứ tự field trong JSON Schema tái tạo lại đúng lỗi §5.56.** Bản đầu đặt `co_mau_thuan`
lên trước. JSON sinh tuần tự theo grammar, nên model phải CHỐT PHÁN QUYẾT trước khi viết được
một chữ lập luận nào. Kết quả đo được trên ca "hai điều kiện học bổng bổ sung cho nhau": model
chấm `co_mau_thuan=true, muc_do=1.0` rồi tự viết trong phần giải thích rằng *"hai đoạn không
cùng nói về một chuyện"* — chính nó bác bỏ phán quyết của chính nó, theo đúng định nghĩa
prompt đã nêu. Đảo thành `phan_tich` → `muc_do` → `co_mau_thuan` thì lập luận được viết trước
và phán quyết rút ra từ nó.

**Nghiêng hẳn về phía IM LẶNG**, vì hai loại sai không ngang giá: báo động giả ("tài liệu của
bạn mâu thuẫn nhau" trong khi chúng không hề) làm người dùng mất niềm tin vào chính tài liệu
của họ và họ không có cách nào rẻ để kiểm lại; còn bỏ sót chỉ đưa hệ thống về đúng hành vi cũ.
Nên cặp phải qua cả ba điều kiện tất định, rồi phải được chấm "có mâu thuẫn" ở **mọi** lần
chấm (`SO_LAN_CHAM_MAU_THUAN=2`), rồi mức độ còn phải vượt ngưỡng — và khi các lần chấm lệch
nhau thì lấy mức **thấp nhất**, không lấy trung bình.

**Kết quả trên bộ kiểm định 7 ca** (`python evaluation/kiem_dinh_doi_chieu.py --so-lan 3`):

| | Kết quả |
|---|---|
| Đúng | **7/7** ổn định qua 3 lần chạy |
| Trong đó **im lặng đúng** | **3/3** — không có báo động giả nào |
| Tầng 1 (tất định) | chặn 2/3 ca im lặng trước khi tốn lượt LLM nào |

Ba trên bảy ca là ca **phải im lặng**, và đó mới là phần khó: bắt mâu thuẫn hiển nhiên thì
dễ, khó là không báo động trên hai đoạn chỉ bổ sung cho nhau. Một ca ("năm đặc điểm" vs "bốn
đặc điểm") đã từng dao động giữa các lần chạy ở bản trước khi sửa prompt — đúng mức dao động
§5.43 đã đo, và cơ chế đồng thuận đẩy dao động đó về phía im lặng chứ không phải phía báo động.

Giao diện cố ý **không kết luận nguồn nào đúng**: hệ thống không có căn cứ nào để phân xử
(không biết tài liệu nào mới hơn, môn nào ưu tiên bản nào). Việc của nó là chỉ ra chỗ xung đột
kèm đủ toạ độ (tên file + trang) để người đọc tự mở ra đối chiếu.


### 5.60 `num_ctx` — cửa sổ ngữ cảnh 4096 token mà không ai khai báo

Đây là bug nghiêm trọng nhất từng có trong hệ thống, và điều đáng ghi lại nhất không phải
cách sửa (một dòng) mà là **vì sao nó sống sót lâu đến thế**: nó không phải lỗi code, nó là
lỗi **giả định**.

Comment ở `config.py` viết: *"context window của model rất lớn (262144 token) nên không có
lý do phải cắt sớm hơn để tiết kiệm"*. Câu đó đúng về mặt kiến trúc model và sai về mặt vận
hành: 262144 là năng lực **kiến trúc** của qwen3:4b, còn **Ollama mặc định cấp 4096 token**
bất kể model hỗ trợ bao nhiêu. Hai con số khác nhau, và Ollama không báo lỗi khi vượt — nó
**cắt im lặng**.

**Prompt thật đã vượt trần từ lâu.** Tiếng Việt ≈ 2.5 ký tự/token với tokenizer Qwen:

| Thành phần | Ký tự | ≈ Token |
|---|---|---|
| System prompt VI | 1.894 | ~760 |
| 6 đoạn × `NGAN_SACH_KY_TU_MOI_DOAN` 1600 + nhãn nguồn | ~9.960 | ~4.000 |
| Câu hỏi + hướng dẫn | ~350 | ~140 |
| **Tổng** | | **~4.900** |

Prompt **một mình đã vượt 4096**, chưa tính một token nào cho thinking lẫn câu trả lời.

**ĐÍNH CHÍNH — số đo thật khác ước lượng, và nó đổi cả cơ chế gây lỗi.** Ước lượng ở trên
dùng tỷ lệ 2.5 ký tự/token. Đo bằng `prompt_eval_count` thật của Ollama (bộ slide 740 chunk,
GPU RTX 5060) cho ra **2.93–3.14 ký tự/token** — tức prompt ngắn hơn ước lượng khoảng 20%:

| Câu hỏi | TOP_K | Ký tự | **Token thật** | Ký tự/token | Token sinh ra | **TỔNG** | `done_reason` |
|---|---|---|---|---|---|---|---|
| thường | 6 | 10.652 | 3.394 | 3.14 | 3.607 | **7.001** | `stop` |
| thường | 4 | 7.509 | 2.430 | 3.09 | 2.775 | **5.205** | `stop` |
| *"liệt kê ĐẦY ĐỦ mọi ứng dụng…"* | 6 | 10.681 | 3.646 | 2.93 | 7.214 | **10.860** | `stop` |
| *"liệt kê ĐẦY ĐỦ mọi ứng dụng…"* | 4 | 7.790 | 2.660 | 2.93 | 5.498 | **8.158** | `stop` |

**Prompt một mình KHÔNG vượt 4096 trên corpus này** (3.394–3.646 token). Nhưng cửa sổ ngữ
cảnh không chỉ chứa prompt — nó chứa **prompt + thinking + câu trả lời**, và cột TỔNG cho thấy
mọi câu đều vượt 4096 rất xa: **7.001** với câu hỏi thường và **10.860** với câu yêu cầu liệt
kê đầy đủ, tức **gấp 2,7 lần** cửa sổ mặc định.

Vậy trên corpus này cơ chế gây lỗi là **(a) cắt phần SINH**, không phải (b) cắt prompt: model
viết được vài dòng rồi chạm trần và dừng — đúng triệu chứng "câu trả lời ngắn cụt". Cơ chế (b)
đòi hỏi prompt tự nó vượt 4096, chỉ xảy ra với tài liệu dày chữ hơn hoặc `TOP_K` lớn hơn. Cả
hai đều là hậu quả của cùng một `num_ctx` bị bỏ trống, nhưng ghi rõ cái nào đang thật sự xảy
ra thì hơn là gộp chung — vì (b) là suy luận từ tài liệu, còn (a) là thứ đo được.

Ghi chú: các cột `done_reason` đều là `stop` vì bảng này đo SAU khi đã sửa. Với `num_ctx=4096`,
mọi dòng trong bảng đều phải dừng vì `length`.

**Một bug, hai triệu chứng trông như hai lỗi khác nhau — đó là lý do nó bị chẩn đoán sai.**

*(a) Câu trả lời ngắn cụt.* qwen3:4b luôn sinh thinking dài (§5.23 đo được 5891 ký tự cho
một tác vụ nhỏ). Khi prompt đã ăn hết cửa sổ, phần còn lại cho thinking + answer gần bằng 0.
Chính hệ thống đã ghi lại bằng chứng nhưng gán sai nguyên nhân — comment cũ ở
`_goi_llm_theo_luong` viết *"đã gặp khi model bị cắt ngang vì chạm num_predict"*. Không phải
`num_predict=12000`; con số đó **chưa bao giờ với tới được**.

*(b) "Truy xuất kém trên tài liệu mới".* Đây mới là phần nguy hiểm. Khi prompt vượt
`num_ctx`, Ollama giữ system message và cắt từ **đầu** phần user content — mà `_ghep_prompt()`
xếp đoạn trích **tốt nhất trước**. Phần bị xoá vì thế **chính xác là `[1]`, `[2]`**, tức đoạn
liên quan nhất. Retrieval hoàn toàn có thể đã tìm đúng; LLM chỉ không bao giờ được nhìn thấy.

**Vì sao chỉ lộ ra trên tài liệu mới.** Corpus cũ nhiều slide và trang thưa chữ → mỗi đoạn
trích ngắn hơn nhiều so với trần 1600, prompt tổng chỉ ~2500 token và vừa lọt 4096. Tài liệu
mới dày chữ → mỗi đoạn **chạm trần** → prompt phình lên ~4900 → bắt đầu bị cắt. Nghĩa là hệ
thống **không** "kém hơn trên tài liệu mới": nó luôn có bug này, tài liệu cũ chỉ tình cờ nằm
dưới ngưỡng gây lỗi. Một lớp lỗi mà chính hệ thống không quan sát được thì mọi kết luận rút
ra từ chất lượng câu trả lời của nó đều đáng ngờ.

**Cách sửa không dừng ở việc đặt một hằng số.** `OLLAMA_NUM_CTX=16384` là sàn, nhưng cửa sổ
còn được **tính động**: `num_ctx = max(sàn, làm_tròn_lên(ước_lượng_prompt + 4000))`, chặn
trên bởi `OLLAMA_NUM_CTX_TOI_DA`. Lý do phải động: một hằng số đủ dùng hôm nay sẽ âm thầm
không đủ vào ngày ai đó tăng `TOP_K` hay `NGAN_SACH_KY_TU_MOI_DOAN` — và triệu chứng của
việc thiếu không hề giống một lỗi cấu hình, nên sẽ bị chẩn đoán nhầm **đúng như đã xảy ra**.

Nhưng cũng **không cấp đúng-vừa-đủ theo từng câu hỏi**: Ollama coi `num_ctx` là một phần định
danh của phiên bản model đang nạp, đổi giá trị giữa hai lượt hỏi khiến nó **nạp lại model**
(hàng chục giây trên CPU). Vì vậy giá trị làm tròn lên theo thang gấp đôi bắt đầu từ sàn —
gần như mọi câu hỏi rơi vào cùng một bậc, không có lần nạp lại nào, mà cấu hình quá tay vẫn
được nới thay vì bị cắt.

**Và quan trọng nhất: bug này giờ quan sát được.** Mỗi lượt gọi ghi lại `prompt_eval_count`
(số token máy chủ THẬT SỰ nạp), `eval_count` và `done_reason` do Ollama trả về ở mảnh cuối
stream, kèm cảnh báo khi `prompt_eval_count >= num_ctx` hoặc `done_reason == "length"`. Trước
đó, việc prompt bị cắt không để lại **một dấu vết nào** — không lỗi, không cảnh báo, chỉ có
câu trả lời tự nhiên ngắn đi.

**Cái giá, ghi ra để không ai bất ngờ:** KV-cache tỉ lệ tuyến tính với `num_ctx`. 16384 với
qwen3:4b tốn thêm vài trăm MB RAM và làm prefill chậm hơn (trên CPU, prefill ~5000 token có
thể mất 10-20 giây). Nếu quá chậm thì hạ `TOP_K` hoặc `NGAN_SACH_KY_TU_MOI_DOAN` — **tuyệt
đối không hạ `num_ctx` xuống dưới độ dài prompt**, vì đó chính là quay lại bug này.

---

### 5.61 Ngưỡng tuyệt đối là ngưỡng chỉ đúng trên corpus đã dùng để đo nó

`NGUONG_DIEM_TOI_THIEU = 0.70` được hiệu chỉnh trên chính corpus của đồ án. Vấn đề: **cosine
của E5 không phải thang đo tuyệt đối.** Giá trị của nó trôi theo domain, ngôn ngữ, độ dài
chunk, phong cách văn bản. Chunk là mảnh bảng hay chú thích ảnh vốn đã cho cosine thấp hơn
văn xuôi ngay trong cùng một corpus; một corpus mới nhiều công thức và số liệu dịch cả phân
bố xuống, và ngưỡng cố định bắt đầu **cắt oan**.

Hậu quả cụ thể: 4/6 đoạn rớt ngưỡng → ngữ cảnh còn 2 đoạn → không còn gì để tổng hợp → câu
trả lời ngắn. Triệu chứng **trùng khít** với bug `num_ctx` (§5.60) nhưng là nguyên nhân khác,
nên phải sửa cả hai chứ không được chọn một rồi tuyên bố xong.

Có một vấn đề kiến trúc sâu hơn: **rerank quyết định thứ tự, cosine quyết định sống chết.**
`diem_similarity` giữ nguyên là cosine (§5.24 giải thích vì sao không thay bằng điểm rerank),
nên một đoạn được cross-encoder xếp hạng 1 mà cosine 0.68 vẫn bị vứt. Hai thang đo khác nhau
cùng ra quyết định trong một pipeline — hệ thống đã tránh việc trộn chúng vào cùng một
**trường**, nhưng chưa tránh việc chúng ra quyết định **mâu thuẫn nhau**.

**Cách sửa: chuyển sang ngưỡng TƯƠNG ĐỐI.** Giữ đoạn có
`cosine >= (cosine cao nhất của lượt đó) × TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT`. Phân bố cosine
trôi theo domain; **tỷ lệ giữa các đoạn trong cùng một lượt thì không** — cả lượt dùng chung
một câu hỏi và một model.

Con số 0.78 **không chọn theo cảm tính**, mà là chính điểm hiệu chỉnh cũ viết lại theo thang
tương đối: trên corpus đã dùng để chỉnh, câu hỏi đúng chủ đề cho cosine cao nhất ~0.90 và
ngưỡng tuyệt đối cũ là 0.70, tức 0.70/0.90 = 0.78. Nhờ vậy trên corpus cũ hành vi gần như
**không đổi** (không được phép làm tệ đi chỗ đang chạy tốt), còn trên corpus có phân bố thấp
hơn thì ngưỡng tự trôi xuống theo.

Cố ý **không** chọn một tỷ lệ chặt hơn (0.90–0.92) dù nghe hợp lý hơn: cosine của E5 nén hết
vào dải hẹp 0.75–0.92, nên 0.92 tương đương sàn tuyệt đối ~0.83 trên corpus cũ — **chặt hơn
hẳn** 0.70 hiện tại, và sẽ làm trầm trọng thêm đúng triệu chứng "thiếu đoạn để tổng hợp" mà
thay đổi này sinh ra để chữa.

Sàn tuyệt đối vẫn giữ nhưng hạ về 0.50, đúng vai trò **chặn rác** mà comment gốc vẫn mô tả.
Kèm theo là `LOG_PHAN_BO_DIEM` — in phân bố cosine, điểm rerank cao nhất và số đoạn sống sót
sau mỗi tầng lọc, để câu hỏi "ngưỡng có đang cắt oan trên corpus này không" trả lời được
bằng số đo thay vì bằng cảm giác.

**Đo lại trên corpus thật** (12 tài liệu, 5554 chunk, 29 câu hỏi, rerank BẬT): P@K 0.500 →
0.500, Recall@K 0.937 → 0.937, MRR 0.980 → 0.980 — **không đổi một chữ số nào**. Đó chính là
kết quả mong muốn, không phải kết quả đáng thất vọng: 0.78 được chọn để tái tạo đúng điểm
hiệu chỉnh cũ trên corpus cũ, nên trên corpus cũ nó *phải* không đổi. Giá trị của thay đổi
này nằm ở corpus có phân bố cosine khác — chỗ mà ngưỡng 0.70 cố định sẽ cắt oan còn ngưỡng
tương đối thì trôi theo. Một thay đổi "không cải thiện gì trên bộ đo hiện có" nhưng gỡ được
một giả định sai vẫn đáng giữ, miễn là nói rõ nó chưa được chứng minh ở đâu.

---

### 5.62 Ba giả định "đúng cho slide" âm thầm sai cho PDF văn xuôi

Ba cơ chế trong luồng truy xuất được thiết kế trên corpus nhiều slide, và cả ba đều mang một
giả định ngầm không còn đúng khi corpus đổi sang PDF văn bản chảy liên tục.

**(1) Mở rộng ngữ cảnh bị chặn cứng ở ranh giới trang.** `_dung_doan_trich()` chỉ mở rộng
trong cùng `(nguon, trang)`. Với slide thì đúng: mỗi slide là một đơn vị nội dung tự đóng.
Với PDF văn xuôi thì sai hẳn — một định nghĩa bắt đầu cuối trang 12 và kết thúc đầu trang 13
sẽ **không bao giờ** được nối lại: chunk neo nằm cuối trang 12, mở rộng sang phải chạm hết
mảng của trang rồi dừng. Đúng cái pattern "chỉ lộ trên tài liệu mới".

Nay phạm vi mở rộng là toàn bộ tài liệu theo thứ tự đọc, với hai chốt chặn: ngân sách ký tự
(như cũ) và `SO_TRANG_TOI_DA_MO_RONG=1` — chốt thứ hai giữ cho slide thưa chữ không hút thêm
2–3 slide xung quanh cho đầy ngân sách, tức không tái tạo lại lỗi "ngữ cảnh loãng" mà việc bỏ
cách gộp-nguyên-trang đã sửa (§5.11). Trích dẫn vẫn ghi **trang của chunk neo**; các trang đi
qua được trả kèm ở `cac_trang` để đo được tần suất mở rộng xuyên trang thật sự xảy ra.

Chi tiết kỹ thuật đáng ghi: thứ tự đọc xuyên trang suy ra từ `(trang, vi_tri)` — **hai trường
đã có sẵn** — chứ không thêm một trường "thứ tự toàn cục" mới. Thêm trường mới thì mọi index
đã build đều thiếu nó và phải build lại, mà build lại tốn nguyên một lượt chú thích ảnh bằng
model vision (~9 phút cho 291 ảnh). Thông tin cần thiết vốn đã nằm trong dữ liệu.

**(2) `SO_DOAN_TOI_DA_MOI_TRANG = 2` phản tác dụng khi câu trả lời nằm gọn trong một trang.**
Trần này sinh ra để chống việc các chunk liền kề trong giáo trình 230 trang chiếm hết `TOP_K`
— lý do đúng, nhưng chỉ đúng **khi có nhiều trang để phân bổ**. Với câu hỏi mà toàn bộ câu
trả lời nằm trên một trang (một mục định nghĩa, một bảng tiêu chí, một quy trình — rất phổ
biến), hệ thống chỉ được lấy 2 đoạn từ đúng trang chứa câu trả lời, bốn suất còn lại đi cho
những trang kém liên quan. Ngữ cảnh vừa **thiếu** phần đúng vừa **loãng** vì phần sai.

Nay trần **thích ứng**: đo độ đa dạng trang trên `SO_UNG_VIEN_XET_DA_DANG_TRANG=20` ứng viên
đầu bảng; đủ đa dạng (≥ `TOP_K` trang) thì áp trần như cũ, không thì bỏ trần. Đặt tham số này
về 0 để quay lại hành vi cũ khi cần đo đối chứng.

**(3) BM25 đang tắt — nhưng cách sửa không phải là bật lại.** Kết quả âm tính ở
`config.TRONG_SO_BM25` là phép đo tốt và không nên lật ngược. Nhưng đọc kỹ lại: cái hại đo
được **không phải do BM25 tìm sai**, mà do **RRF cho BM25 quyền xếp hạng ngang dense** — hạng
1 của BM25 (một tài liệu tiếng Việt sai) được coi ngang hạng 1 của dense, phá truy xuất chéo
ngôn ngữ.

Nên hướng đúng là **tách vai trò**, không phải chỉnh trọng số:

- BM25 chỉ **bơm ứng viên** vào tập đưa đi rerank (recall), với điểm RRF **bằng 0** nên tự nó
  không đẩy được thứ hạng của bất cứ gì.
- Cross-encoder — vốn đã đo là phân biệt tốt gấp ~60.000 lần cosine (§5.8) — quyết định thứ
  hạng cuối. Đoạn cứu hộ thật sự liên quan sẽ được nó đẩy lên; không liên quan thì nằm yên.

Một chi tiết dễ làm cả cơ chế thành vô nghĩa: điểm RRF 0 đẩy các đoạn cứu hộ xuống **cuối**
danh sách, mà `_xep_hang_lai` chỉ chấm `SO_UNG_VIEN_RERANK` ứng viên **đầu** — nên chúng phải
được đánh dấu riêng và **luôn** được chấm, kể cả khi rơi ngoài trần đó. Không có chi tiết
này thì tính năng chạy nhưng không làm gì cả, và không có test nào bắt được.

Trường hợp xấu nhất **theo thiết kế** là "không cải thiện", không phải "làm hỏng" — đúng
nguyên tắc §5.45(a). Vẫn phải đo lại bằng đúng thí nghiệm cũ trước khi giữ.

**Kết quả đo** (12 tài liệu, 5554 chunk, 29 câu hỏi song ngữ, rerank BẬT, bật lần lượt từng
thay đổi trên cùng một index):

| Cấu hình | P@K | Recall@K | MRR | hạng 1 | chặn lạc đề |
|---|---|---|---|---|---|
| GỐC (trước khi sửa) | 0.500 | 0.937 | 0.980 | 24/25 | 2/4 |
| + ngưỡng tương đối (§5.61) | 0.500 | 0.937 | 0.980 | 24/25 | 2/4 |
| + mở rộng xuyên trang | 0.467 | **0.875** | 0.980 | 24/25 | 2/4 |
| + trần trang thích ứng | **0.567** | 0.937 | 0.980 | 24/25 | 2/4 |
| + BM25 cứu hộ | 0.507 | **0.945** | 0.980 | 24/25 | 2/4 |

Trần thích ứng và BM25 cứu hộ đúng như dự đoán: **+0.067 P@K** và **+0.008 Recall@K**, không
đụng tới MRR lẫn khả năng chặn câu lạc đề. Nhưng mở rộng xuyên trang thì **làm tụt Recall@K
0.062** — kết quả đi ngược hẳn kỳ vọng. Hoá ra chính metric mới là thứ sai ở đây, và việc
chứng minh điều đó là §5.65.

---

### 5.63 Prompt: quy tắc 5 và 6 xung đột, và model 4B luôn chọn cái dễ

```
5. Trả lời ĐẦY ĐỦ ... tổng hợp mọi thông tin liên quan
6. Trả lời ĐI THẲNG VÀO TRỌNG TÂM, không lặp lại ... không viết phần mở đầu/kết luận thừa
```

Hai chỉ thị ngược chiều đặt cạnh nhau mà không nói cái nào thắng. Với model 4B, cách xử lý
điển hình là chọn cái **dễ tuân thủ hơn** — và "ngắn gọn" luôn dễ hơn "đầy đủ", vì bỏ bớt thì
không có gì để làm sai.

Nay quy tắc 5 nói rõ nó có **ưu tiên cao hơn** khi hai điều đó ngược chiều, và quy tắc 6 định
nghĩa lại "ngắn gọn" theo **số chữ thừa** chứ không theo **lượng thông tin**: không lặp ý,
không trích lại nguyên văn nhiều lần, không mở đầu/kết luận sáo rỗng — nhưng không được **bỏ
bớt** thông tin có trong ngữ cảnh để câu trả lời ngắn lại.

**Thứ tự sửa quan trọng hơn bản thân bản sửa.** Đây là nguyên nhân **yếu nhất** trong ba
nguyên nhân của "câu trả lời ngắn" (§5.60 và §5.61 là hai cái kia). Sửa prompt trước hai cái
kia sẽ cho một cải thiện nhẹ — đủ để tưởng đã tìm đúng nguyên nhân — trong khi bug thật vẫn
nằm nguyên đó. Ghi lại thứ tự này vì nó là cái bẫy chứ không phải chi tiết vụn.

---

### 5.64 Bộ HELD-OUT: biến "hệ thống có overfit không" thành một con số

Đây là bài học kiến trúc thật nằm dưới cả bốn mục trên.

`evaluation/test_questions.json` sinh ra từ chính corpus đang dùng, và **mọi** hằng số của hệ
thống — 0.70, 0.001, 0.88, 2, 1, 30, 160 — đều được chọn bằng cách tối ưu trên đúng bộ đó. Đó
là **tuning trên tập test**. Kết quả tất yếu: hệ thống rất tốt trên corpus đã dùng để chỉnh và
tụt trên corpus mới — đúng hiện tượng đã gặp. Điểm in-sample vì thế không nói được gì về tài
liệu mới, và mọi bảng số liệu chỉ có một cột đều đang che mất điều đó.

`evaluation/test_questions_held_out.json`: **46 câu (44 có đáp án + 2 câu lạc đề) trên 12 tài
liệu chưa từng dùng để chỉnh bất kỳ tham số nào** — `CV-05-Classification.pdf`,
`BAO_CAO_MAY_HOC.docx`, `Chapter 2. Server kết nối vạn vật.pptx`, `baocaonangcaothayAn.docx`,
6 bài giảng Khai phá dữ liệu `Bai1`–`Bai6`, `PaperQA.pdf` và `RFRAG.pdf`. Bộ này phủ đủ các
nhóm của bộ in-sample (truy xuất, chéo ngôn ngữ, đọc bảng, kiểm chứng khẳng định, từ chối câu
lạc đề) để hai bảng so được với nhau.

Bộ ban đầu chỉ có 22 câu, và chính điều đó là điểm yếu: với 20 câu có đáp án, một câu bị giám
khảo LLM lật điểm đã làm trung bình đổi 0,05, nên khoảng cách Faithfulness đo được không tách
được khỏi nhiễu. Nhân đôi bộ lên 44 câu có đáp án là cách rẻ nhất để chữa việc đó.

**Một cảnh báo phải đọc kèm:** 6 tài liệu `Bai*.docx` gần như không có ngắt trang (`Bai5` chỉ
1 "trang", `Bai2` có 2), mà Precision@K/Recall@K so khớp theo `(nguồn, trang)` — nên với chúng
Recall@K **suy biến**: lấy về bất kỳ chunk nào cũng thành 1.00 (§5.39). Vì vậy mỗi tài liệu
loại đó chỉ được đặt 1–2 câu, và khi đọc kết quả phải nhìn Citation accuracy bên cạnh
Recall@K chứ không thay thế nó.

```bash
python evaluation/run_evaluation.py --khoang-cach
```

chạy cả hai bộ rồi in **chênh lệch Recall@K / Precision@K / MRR**. Con số đó chính là mức
overfit của hệ thống, và cách đọc nó:

| Quan sát | Kết luận |
|---|---|
| Khoảng cách thu hẹp | thay đổi giúp hệ thống tổng quát hơn |
| Cả hai cùng tăng, khoảng cách giữ nguyên | thay đổi tốt nhưng không chữa overfit |
| In-sample tăng, held-out giảm | đang tối ưu vào đúng bộ tài liệu cũ — phải bỏ |

**Quy tắc bất di bất dịch: không bao giờ chỉnh tham số theo kết quả của bộ held-out.** Chỉnh
một lần là nó lập tức trở thành bộ in-sample thứ hai và con số này mất sạch ý nghĩa — lúc đó
hệ thống không còn cách nào biết mình đang overfit tới đâu.

**Kết quả đo** (corpus đầy đủ 26 tài liệu / 9285 chunk chứa cả hai bộ; số liệu chi tiết và
cách tái lập ở [`KET_QUA_DO_DAC.md`](KET_QUA_DO_DAC.md)):

| Metric | in-sample (25 câu) | held-out (44 câu) | khoảng cách |
|---|---|---|---|
| Recall@K | 0.845 | **0.905** | **−0.060** |
| MRR | 0.980 | 0.843 | **+0.137** |
| Đoạn đúng ở hạng 1 | 24/25 (96%) | 33/44 (75%) | −21 điểm % |
| Faithfulness | 0.980 | 0.977 | **+0.003** |
| Citation accuracy | 0.714 | **0.784** | **−0.070** |
| P@K | 0.620 | 0.364 | +0.256 *(không so được — xem dưới)* |

**Đọc kết quả này cho đúng, và nó chia làm hai phần ngược nhau:**

*Khả năng TÌM ĐÚNG nội dung tổng quát rất tốt.* Recall@K trên tài liệu chưa từng thấy còn
**cao hơn** trên tài liệu đã hiệu chỉnh. Chunking, embedding và tầng truy hồi **không** overfit.

*Khả năng XẾP ĐÚNG THỨ TỰ thì có overfit thật.* MRR tụt 0.980 → 0.843, số câu có đoạn đúng ở
hạng 1 rơi từ 96% xuống 75%. Đây chính là tầng mà mọi ngưỡng của hệ thống tác động vào: rerank
+ các ngưỡng lọc. Nội dung đúng vẫn vào được ngữ cảnh, chỉ không còn chắc chắn nằm ở đoạn
`[1]` — đoạn mà LLM đọc kỹ nhất.

*Chất lượng CÂU TRẢ LỜI thì không overfit* (Faithfulness chênh 0.003) — nhưng kết luận này đã
từng bị đo SAI, và đó là phần đáng học nhất ở đây.

**Bộ đo quá nhỏ tạo ra tín hiệu giả trông y hệt tín hiệu thật.** Ở bản held-out đầu tiên (22
câu), Faithfulness đo được 0.825, tức khoảng cách **+0.155** — và kết luận rút ra khi đó là
"chất lượng câu trả lời có overfit", kèm một dòng dè dặt rằng 20 câu là quá ít để chắc chắn.
Nhân đôi bộ lên 44 câu: khoảng cách **co từ +0.155 về +0.003**. Tín hiệu đó là nhiễu.

Cùng lúc đó, khoảng cách MRR đi ngược lại:

| Tín hiệu | Bộ 20-22 câu | Bộ 44-46 câu | Diễn giải |
|---|---|---|---|
| Khoảng cách **MRR** (tất định) | +0.101 | **+0.137** | **nở ra** → tín hiệu THẬT |
| Khoảng cách **Faithfulness** (LLM chấm) | +0.155 | **+0.003** | **co lại** → là NHIỄU |

Đây là phép thử rẻ nhất để phân biệt hai thứ: **thêm mẫu rồi xem khoảng cách nở ra hay co
lại**. Tín hiệu thật không loãng đi khi có thêm dữ liệu; nhiễu thì có. Và nó chỉ làm được khi
bộ đo đủ lớn — một lý do cụ thể để bỏ công mở rộng held-out thay vì tin con số đầu tiên đo
được. Metric tất định (§4) không cần phép thử này; metric có LLM chấm thì luôn cần.

*P@K KHÔNG so được giữa hai bộ, và phải nói ra thay vì để nó nằm im trong bảng.* Precision@K
phụ thuộc trực tiếp vào số trang đúng mỗi câu: bộ in-sample trung bình **2.84** trang/câu, bộ
held-out chỉ **1.45**. Với `TOP_K=6` (giá trị lúc đo), một câu chỉ có 1 trang đúng thì P@K trần đã rất thấp
bất kể hệ thống tốt đến đâu. Chênh lệch +0.315 vì thế phần lớn là **hiện vật của cách ra đề**,
không phải bằng chứng overfit — đúng loại bẫy mà §5.65 vừa mắc một lần.

Khoảng cách đó, kèm giải thích vì sao nó tồn tại, là thứ phân biệt một đồ án RAG với một
tutorial "chat with PDF". Hầu như không ai đo nó.

---

### 5.65 Khi chính thước đo là thứ sai — Recall@K không nhìn thấy mở rộng xuyên trang

Mở rộng xuyên trang (§5.62) làm **Recall@K tụt 0.937 → 0.875**. Theo đúng nguyên tắc "đo
trước/sau rồi mới giữ", đáng ra phải bỏ ngay. Nhưng trước khi bỏ có một câu hỏi phải trả lời:
**metric này có đo được thứ mà thay đổi kia làm không?**

`recall_tai_k()` so khớp theo `(nguon, trang)` của **trang neo** — trang của chunk khớp nhất
trong mỗi đoạn trích. Khi một đoạn nuốt sang trang liền kề, các chunk ở trang đó bị đánh dấu
"đã dùng" nên không tạo đoạn trích riêng nữa, tức **trang đó không còn được neo**. Nội dung
của nó vẫn nằm nguyên trong ngữ cảnh gửi cho LLM; chỉ có metric là không thấy.

Đó là một giả thuyết, và giả thuyết thì phải đo. Thêm một chỉ số **Recall PHỦ**: trang đúng
có nằm trong bất kỳ trang nào mà các đoạn trích *đi qua* hay không (`cac_trang`).

| | P@K | Recall@K (neo) | **Recall PHỦ** |
|---|---|---|---|
| TẮT mở rộng xuyên trang | 0.580 | 0.945 | 0.945 |
| BẬT mở rộng xuyên trang | 0.540 | **0.875** | **0.959** |

Hai con số đi **ngược chiều nhau**, và chỉ một trong hai không phải là hiện vật của cách đo.
Chi tiết từng câu xác nhận dứt điểm — mọi câu bị tụt Recall neo đều giữ nguyên Recall phủ:

| Câu hỏi | Recall neo | Recall phủ | số trang phủ |
|---|---|---|---|
| *How does the SIFT descriptor work?* | 1.00 → **0.67** | 1.00 → **1.00** | 6 → 12 |
| *Mô hình camera lỗ kim hoạt động ra sao?* | 1.00 → **0.80** | 1.00 → **1.00** | 6 → 16 |
| *Đặc trưng cục bộ SIFT là gì?* | 1.00 → **0.67** | 1.00 → **1.00** | 6 → 11 |
| *Luật hình sự có những nội dung khái quát chung nào?* | 1.00 → **0.80** | 1.00 → **1.00** | 6 → 11 |

Không câu nào **mất** nội dung. Ngữ cảnh gửi cho LLM phủ 10–16 trang thay vì đúng 6, và trên
thước đo không bị hiện vật thì mở rộng xuyên trang **cải thiện nhẹ** (0.945 → 0.959).

**Kết luận: giữ, và ghi rõ vì sao con số trông như tệ đi.** Nhưng hai hệ quả phải nói ra chứ
không được giấu:

1. **Recall@K trước và sau thay đổi này KHÔNG so trực tiếp được nữa.** Bảng §5.62 vẫn giữ
   nguyên con số 0.875 chứ không sửa cho đẹp — sửa đi là che mất chính bài học. Khi báo cáo,
   phải kèm cả hai chỉ số.
2. **Trích dẫn có thể trỏ thiếu.** Đoạn trích phủ tối đa 3 trang nhưng chỉ ghi trang neo, nên
   câu trả lời có thể dựa vào nội dung ở trang bên cạnh mà nguồn lại chỉ dẫn một trang — đúng
   loại sai lệch §5.54 đã phải sửa một lần. Vì vậy `dinh_dang_trich_dan()` nay trả kèm
   `cac_trang` và giao diện ghi *"trang/slide 12–13"* khi đoạn trích vượt ranh giới trang.

**Bài học chung, và nó lớn hơn tính năng này.** Một metric là một *mô hình* của thứ ta quan
tâm, không phải bản thân thứ đó. `recall_tai_k` mô hình hoá "LLM có được đọc nội dung đúng
không" bằng "trang đúng có được neo không" — hai điều trùng nhau chừng nào mỗi đoạn trích nằm
gọn trong một trang, và **chính thay đổi này phá vỡ giả định đó**. Một thay đổi làm hỏng giả
định của metric sẽ luôn trông như một hồi quy, kể cả khi nó là cải thiện. Cách duy nhất phân
biệt là hỏi "metric này đo được thứ tôi vừa đổi không" **trước** khi diễn giải con số — chứ
không phải sau khi con số đã dẫn tới kết luận sai.


### 5.66 Ingestion quét cùng một tài liệu nhiều lần — và bốn tầng sửa

Một bản rà soát chi phí toàn hệ thống cho ra kết luận trái với trực giác ban đầu: **nút thắt
không nằm ở FAISS hay ở phần tìm kiếm, mà nằm ở ingestion**. Truy xuất chạy trên vector đã
tính sẵn nên nhanh; còn việc đọc và lập chỉ mục tài liệu thì đang trả một loạt chi phí lặp
mà không ai nhìn thấy, vì chúng nằm rải rác ở năm hàm khác nhau.

**Đếm lại số lần một trang PDF bị chạm vào, ở bản trước:**

| Việc | Số lượt | Nằm ở đâu |
|---|---|---|
| `extract_text()` để dò `x_tolerance` | tới **5** (1 gốc + 4 mức) | `_trich_text_thich_ung` |
| `bang.extract()` cho mỗi bảng | **2** (lọc bảng giả, rồi dựng khối) | `doc_pdf` |
| `extract_words()` dò bố cục cột | 1 | `_cac_cot_cua_trang` |
| `trang.chars` nhận diện tiêu đề | 1 | `_phat_hien_tieu_de_pdf` |
| `extract_text()` lấy dòng chú thích ảnh | **1 lượt duyệt LẠI toàn bộ PDF** | `trich_anh_pdf` |

Hai dòng in đậm là phần lãng phí thuần tuý: kết quả `extract()` bị vứt đi rồi tính lại sau ba
dòng code, và toàn bộ PDF bị mở ra duyệt lần thứ hai chỉ để lấy vài dòng text vốn đã đọc xong
ở lần thứ nhất. Trên giáo trình Bishop (758 trang, font Computer Modern nên **gần như trang
nào cũng dính chữ**), riêng phép dò `x_tolerance` là hàng nghìn lượt đọc lại trang.

**Tầng 1 — Đọc MỘT LƯỢT, chia 4 pha.** `doc_pdf` được viết lại thành: (1) duyệt trang một
lần, đọc text + dò bảng/cột + nhận diện tiêu đề + **liệt kê ứng viên ảnh mà chưa render**;
(2) OCR các trang đã đánh dấu; (3) gộp OCR, dọn dẹp, lọc trang mục lục; (4) render những ảnh
được giữ lại.

Điều khiến việc chia pha này *khó* nằm ở một quy tắc rất tinh tế đã có từ §5.49: ảnh phủ kín
một trang chỉ được loại khi **OCR đã chứng minh** trang đó là ảnh chụp một trang chữ — mà
điều đó thì phải chờ OCR chạy xong mới biết. Cách giải: `ung_vien_anh_trang()` trả về
`(bbox, có_phủ_cả_trang)` và để chỗ gọi quyết định *sau*. Nhờ vậy quy tắc cũ được giữ **nguyên
vẹn** thay vì bị thay bằng một phép đoán rẻ hơn. Sau mỗi trang, `flush_cache()` nhả các đối
tượng đã phân tích; pha 4 chỉ đọc lại đúng những trang thật sự có ảnh cần render.

**Tầng 2 — Hiệu chỉnh `x_tolerance` theo tài liệu, và một hồi quy đã suýt lọt.** Nguyên nhân
dính chữ là *font và cỡ chữ của tài liệu*, hai thứ gần như không đổi trong cùng một cuốn sách
— nên mức đã dùng được ở trang trước được thử **trước tiên** ở trang sau
(`HieuChinhXTolerance`). Nó vẫn phải vượt đúng hai phép kiểm tra cũ (giảm được độ dính, không
làm vỡ từ thêm quá `MUC_TANG_TU_LE_CHAP_NHAN`), nên tài liệu trộn nhiều font không bị đọc hỏng.

Bản đầu còn thêm phép **dừng sớm**, và ở đây có một lỗi đáng ghi lại vì nó thuộc loại chỉ lộ
ra khi đối chiếu đầu ra chứ không phải khi đọc code: phép dừng sớm dùng chung ngưỡng
`TY_LE_DINH_CHU_DE_DOC_LAI = 0.10` — vốn trả lời câu hỏi *"trang này có đáng đọc lại không"* —
để trả lời một câu hỏi hoàn toàn khác: *"bản đọc lại đã đủ tốt chưa"*. Hậu quả đo được trên
`PaperQA.pdf`: mức `x_tolerance` đầu tiên đưa độ dính từ 30% xuống 9% được chấp nhận ngay, bỏ
qua mức tốt hơn nằm ngay sau đó, và 9% còn lại đi thẳng vào index dưới dạng:

```
RAGmodelsretrievetextfromacorpus, usingmethodssuchasvectorembeddingsearchorkeyword
```

Sửa bằng một ngưỡng riêng, `TY_LE_DINH_CHU_DAT_YEU_CAU = 0.02`, lấy thẳng từ số đo đã có ở
`_ty_le_dinh_chu()`: tám PDF đọc tốt trong corpus cho 0.0–1.5%, trang hỏng cho 41.7%. Vạch
đặt ngay trên mức cao nhất của nhóm "đọc tốt".

**Kiểm chứng bằng cách so đầu ra, không phải bằng cách đọc lại code.** Bản cũ và bản mới cùng
đọc một corpus, rồi so từng bản ghi. Trên Bishop: **cùng 809 bản ghi**, độ dính trung bình
**giống hệt** (0.0026 ở cả hai), và 540 trang có khác biệt đều là bản mới **tốt hơn**:

```
CŨ : p(X = xi,Y = yj)      lnp(D|α,β)    where i = 1,...,D
MỚI: p(X = xi, Y = yj)     ln p(D|α, β)  where i = 1, . . . , D
```

**Tầng 3 — Cache theo CONTENT HASH** (`rag/bo_nho_dem.py`). Khoá cache là **nội dung**, không
phải tên file hay `mtime`. `mtime` không đáng tin theo cả hai chiều: `git checkout`, sao chép
file và đồng bộ cloud đều đổi `mtime` mà không đổi nội dung (gây build lại vô nghĩa), còn vài
công cụ ghi đè file mà giữ nguyên `mtime` (gây **bỏ sót** thay đổi thật — kiểu hỏng tệ hơn hẳn).

Bốn kho, mỗi kho khoá bằng đúng thứ quyết định kết quả của nó:

| Kho | Khoá | Điều nó cứu |
|---|---|---|
| Tài liệu | băm(file) + vân tay cấu hình đọc | cả lượt đọc + OCR + vision của một tài liệu |
| OCR | băm(file) + số trang + DPI | **cả bước render**, không chỉ lượt gọi model |
| Vision | băm(**nội dung ảnh**) | logo lặp 60 slide → 1 lượt gọi; sống qua cả đổi tên file |
| Embedding | băm(nội dung chunk) | ~316 giây encode toàn corpus mỗi lần build |

Hai chi tiết đáng chú ý. **Thứ nhất**, khoá OCR cố ý *không* băm ảnh đã render — nếu băm ảnh
thì phải render trước mới tra được cache, mà render một trang ở 150 DPI mất 0,2–0,4 giây;
với sách scan 400 trang thì cache sẽ chỉ tiết kiệm được một nửa chi phí. **Thứ hai**, ranh
giới cache tài liệu đặt *sau* bước chú thích vision (`doc_tai_lieu_hoan_chinh`), không phải
trước: cache phần rẻ mà bỏ phần đắt thì gần như vô nghĩa. Cái giá là bước chú thích chuyển từ
"gom ảnh cả corpus, báo tiến độ ảnh i/n" sang "làm xong từng tài liệu" — chấp nhận được, vì
với index tăng dần thì đa số tài liệu không được xử lý lại chút nào.

Vân tay cấu hình là chốt an toàn bắt buộc: đổi `BAT_OCR_DU_PHONG` hay `DPI_RENDER_TRANG_OCR`
mà vẫn trả cache cũ chính là **kiểu lỗi không triệu chứng** mà cả tài liệu này tồn tại để
tránh. Ngược lại, danh sách tham số trong vân tay được liệt kê **thủ công** chứ không quét cả
`config`: gộp cả `TOP_K` hay ngưỡng rerank vào đó sẽ khiến mỗi lần chỉnh một tham số *truy
vấn* là mất trắng cache *đọc tài liệu*, tức cache gần như không bao giờ trúng.

**Tầng 4 — Index tăng dần.** Mỗi tài liệu được ghi kèm băm nội dung vào `index_info.json`. Lần
build sau: tài liệu không đổi giữ nguyên vector; tài liệu đã đổi bị `xoa_theo_nguon()` rồi đọc
lại; tài liệu biến mất khỏi thư mục bị gỡ khỏi index. Khi vân tay index không khớp cấu hình
(đổi model embedding, chunk size…) thì tự lùi về build toàn bộ — dùng lại chính
`ly_do_khong_tuong_thich()` đã viết cho việc cảnh báo người dùng, không phát minh cơ chế mới.

Một chi tiết nhỏ nhưng quan trọng: băm chỉ được ghi cho tài liệu **thật sự có nội dung vào
index**. Một file đọc hỏng (đặt mật khẩu, tải dở) mà vẫn được đánh dấu "đã xử lý" sẽ bị bỏ qua
ở *mọi* lần build sau — lỗi im lặng vĩnh viễn. Để nó trượt và được thử lại mỗi lần là lựa chọn
đúng, và gần như miễn phí nhờ cache tài liệu.

**Song song hoá — chỉ đúng chỗ có lợi.** `SO_WORKER_VISION` chạy chú thích ảnh và OCR trên
nhiều luồng, vì công việc thật nằm ở phía máy chủ Ollama còn Python chỉ ngồi chờ I/O (thread
nhả GIL). Ba chỗ cố ý **không** song song hoá, và lý do của mỗi chỗ đều cụ thể:

- **Render trang PDF**: `pypdfium2` không cam kết an toàn đa luồng, và render là việc thuần
  CPU nên thread cũng không giúp gì.
- **Đọc tài liệu** (`SO_WORKER_DOC = 1` mặc định): `pdfplumber` là CPU-bound trong Python nên
  bị GIL chặn; dùng process thì mỗi tiến trình con phải nạp lại toàn bộ module, mà trên
  Windows `spawn` còn chạy lại phần khởi tạo của `config`. Quan trọng hơn cả: **sau khi có
  cache + index tăng dần, lần build thứ hai gần như không còn đọc lại tài liệu nào** — song
  song hoá một việc đã không còn xảy ra là tối ưu nhầm chỗ.
- **Số worker mặc định** suy từ `os.cpu_count()` nhưng chặn trên ở 4: mọi luồng đều đi qua
  **một** máy chủ Ollama, mở hàng chục yêu cầu cùng lúc không làm model chạy nhanh hơn (nó
  vẫn xếp hàng) mà chỉ làm RAM/VRAM phình lên.

**Lọc ảnh — làm sạch index, không phải chỉ tiết kiệm thời gian.** Ba chốt hình dạng
(kích thước, tỉ lệ cạnh, tỉ lệ diện tích trang) chạy **trước** khi render, cộng một chốt đếm
lần lặp theo băm nội dung để bắt logo/watermark. Đo trên corpus thật, số bản ghi ảnh giảm
401 → 261 (một bài giảng IoT), 219 → 168 và 75 → 61 (hai file DOCX), 18 → 13 (một bài thuyết
trình) — **trong khi số bản ghi văn bản không đổi một đơn vị nào**. Mỗi ảnh bị loại là một
lượt render, một file trên đĩa, một lượt gọi model vision (~1,9 giây) và một vector rác trong
index không còn phải trả.

Chốt tỉ lệ diện tích có một tính chất cần ghi lại để không ai chỉnh nhầm: trên khổ A4,
`KICH_THUOC_ANH_TOI_THIEU` (120 điểm) đã tương đương ~2,9% diện tích trang, tức mọi ảnh lọt
qua chốt kích thước đều tự khắc vượt ngưỡng diện tích. Chốt diện tích chỉ thật sự cắn ở trang
khổ lớn. Quan hệ đó được khoá lại bằng một test riêng.

**Profiling (`rag/do_thoi_gian.py`).** Log có sẵn dấu thời gian, nhưng nó nói được *"trang 412
bị OCR lúc 09:31:07"* chứ không nói được *"OCR chiếm 68% tổng thời gian build"* — hai câu dẫn
tới hai quyết định tối ưu khác nhau. Bảng tổng kết in sau mỗi lần build biến câu hỏi "chỗ nào
đang chậm" thành một phép đo thay vì một phỏng đoán.

---

### 5.67 Ngân sách thích ứng lúc truy vấn — và ranh giới không được vượt

Ở phía query, kết luận của bản rà soát là **không viết lại kiến trúc**: dense retrieval, BM25
cứu hộ, RRF, reranker và citation đều là những quyết định đã được benchmark (§5.24, §5.29,
§5.30). Embedding `multilingual-e5-base` chậm hơn `e5-small` nhưng hơn hẳn ở truy xuất chéo
ngôn ngữ (MRR 0.738 so với 0.364), nên **không đổi model chỉ để lấy tốc độ**.

Thứ *có* thể sửa là việc mọi câu hỏi đang được cấp **ngân sách tối đa**. "Overfitting là gì?"
và "So sánh KNN với Naive Bayes về độ phức tạp, dữ liệu cần thiết và trường hợp nên dùng" cùng
được cấp 30 ứng viên cross-encoder, cùng trần sinh 12000 token, cùng mức dự phòng cửa sổ ngữ
cảnh lớn nhất. Câu thứ nhất không dùng hết phần nào trong số đó, nhưng vẫn phải chờ nó.

`la_cau_hoi_phuc_tap()` phân loại bằng ba dấu hiệu — độ dài, động từ yêu cầu nhiều vế
("so sánh", "liệt kê", "vì sao", "compare", "why"…), và việc có phải câu kiểm chứng hay không.
Ngưỡng cố ý nghiêng hẳn về phía **cấp dư**: đoán nhầm một câu phức tạp thành đơn giản thì câu
trả lời có thể thiếu ý — người dùng nhìn thấy; đoán nhầm chiều ngược lại chỉ khiến câu đó chạy
chậm bằng đúng bản cũ, tức không tệ hơn hiện trạng. Hai loại sai không ngang giá nên ngưỡng
không được đặt ở giữa.

Độ phức tạp được đánh giá trên **truy vấn chính** (bản đã mang ngữ cảnh hội thoại), không phải
câu người dùng gõ, và được ghi lại một lần vào `self.la_cau_hoi_phuc_tap` để bước truy xuất và
bước sinh dùng chung một phán đoán. Chấm trên câu gốc sẽ xếp *"Thế còn cái thứ hai?"* vào
nhóm đơn giản — tức cấp ngân sách thấp cho đúng loại câu hỏi khó nhất của hệ thống (§5.58).

**Ranh giới tuyệt đối: KHÔNG hạ `num_ctx`.** Đây là chỗ mà một tối ưu tốc độ "hợp lý" sẽ tái
lập đúng bug tệ nhất từng có (§5.60). Hạ `num_ctx` **không** làm prompt ngắn lại; nó chỉ
chuyển quyền quyết định cắt chỗ nào từ ta sang Ollama, mà Ollama luôn cắt từ **đầu** phần user
content — tức xoá đúng đoạn trích `[1]`, đoạn liên quan nhất, vì `_ghep_prompt()` xếp đoạn tốt
nhất lên trước. Không lỗi, không cảnh báo, chỉ có câu trả lời tự nhiên kém đi.

Vì vậy khi ngữ cảnh vượt trần, thứ bị giảm là **số đoạn**, và giảm **từ đoạn xếp hạng thấp
nhất lên** (`nen_ngu_canh`). Bỏ từ cuối là bắt buộc chứ không phải tiện tay: nó giữ nguyên số
thứ tự `[1]`, `[2]`… của các đoạn còn lại, nên trích dẫn mà LLM gắn vẫn trỏ đúng nguồn — bỏ
từ giữa thì mọi số sau đó lệch một bậc. Danh sách gốc không bị sửa, nên trích dẫn hiển thị cho
người đọc vẫn giữ nguyên văn đầy đủ. Khi ngay cả một đoạn duy nhất cũng không vừa, đoạn đó
được cắt ngắn kèm ghi chú rõ ràng: thà đưa nửa đầu đoạn tốt nhất còn hơn không đưa gì, vì
không đoạn nào nghĩa là LLM từ chối trả lời.

Đây cũng là phần bù cho cảnh báo đã có ở `_tinh_num_ctx()`: cảnh báo nói cho người dùng biết
cấu hình đã vượt trần máy, nhưng **lượt hỏi đang chạy thì vẫn hỏng**. Nén ngữ cảnh khiến lượt
đó vẫn trả lời được, với phần bị bỏ là phần ít liên quan nhất — một lựa chọn của hệ thống, ghi
rõ trong log, thay vì một lựa chọn ngẫu nhiên của bộ cắt prompt.

**Adaptive TOP-K: cố ý KHÔNG làm.** Đây là một mục trong đề xuất tối ưu nhưng bị loại sau khi
xét: `TOP_K = 4` là giá trị mà toàn bộ Recall@K, MRR và các ngưỡng lọc trong hệ thống đã được
hiệu chỉnh trên đó (§5.61, §5.64). Hạ nó cho "câu hỏi đơn giản" mà **không đo lại** chính là
đổi độ chính xác lấy tốc độ — đúng điều mà cả đợt tối ưu này tồn tại để tránh. Ba thứ đã làm
(ứng viên rerank, `num_predict`, nén ngữ cảnh) đều không đụng tới tập đoạn trích được chọn
trong trường hợp bình thường. Khi nào có phép đo Recall@K theo từng nhóm độ phức tạp thì đây
là việc tiếp theo đáng làm.

---


### 5.68 GPU: một cấu hình sai không ai nhìn thấy, và cách chia VRAM giữa bốn model

**Lỗi gốc không nằm trong code.** Máy làm đồ án có RTX 5060 8 GB, nhưng `pip install
sentence-transformers` kéo về `torch` bản **CPU-only** — đó là bản mặc định trên PyPI. Hệ quả:
GPU chỉ phục vụ Ollama (LLM và vision), còn embedding và cross-encoder rerank chạy hoàn toàn
trên CPU. Không exception, không cảnh báo, kết quả trả về vẫn đúng từng chữ. Triệu chứng duy
nhất là phần truy xuất tốn 11–12 giây mỗi câu — một con số hoàn toàn có thể bị đọc nhầm thành
"cross-encoder vốn đắt như vậy".

Đây là biến thể mới của đúng loại lỗi mà §5.20 và §5.60 đã gặp: **hệ thống chạy đúng nhưng
chạy sai điều kiện, và không có gì trong chính hệ thống nói ra điều đó**. Cách chữa cũng cùng
một kiểu — bắt nó phải tự khai báo (`tai_nguyen_gpu.mo_ta_phan_cung()` ghi ra log và hiện trên
thanh bên), thay vì trông vào việc ai đó nhớ kiểm tra.

Sửa bằng một lệnh, giữ NGUYÊN phiên bản torch để không đụng tương thích với
`sentence-transformers` và `faiss`:

```
pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```

| Bước | CPU | GPU | Nhanh hơn |
|---|---:|---:|---:|
| Embedding 512 chunk | 19,90 s | 1,55 s | **12,8×** |
| Rerank 12 cặp | 1,68 s | 0,15 s | **11,2×** |
| Truy xuất đầu-cuối, 6 câu | 15,21 s | 1,69 s | **9,0×** |

**Reranker là chỗ đáng giá nhất, và không phải vì nó nặng nhất** mà vì nó nằm trên đường đi
của mọi câu hỏi: người dùng chờ nó xong mới thấy chữ đầu tiên.

**Bằng chứng bắt buộc: nhanh hơn mà không đổi kết quả.** "Nhanh hơn nhưng ra kết quả khác" là
một cách thất bại chứ không phải một cách tối ưu, nên điều này phải được chứng minh chứ không
phải giả định. Chạy cùng 6 câu hỏi trên cùng index, một lần ép `cpu` một lần ép `cuda`:
**6/6 câu trả về đúng những đoạn đó, đúng thứ tự đó**, lệch điểm similarity tối đa 1,79×10⁻⁷ —
sai số làm tròn float32. Nhờ vậy mọi số chất lượng đã đo trước đây (§4, §5) vẫn còn hiệu lực
mà không phải chạy lại toàn bộ.

**HARDWARE-AWARE, KHÔNG PHẢI RTX-5060-AWARE.** Mọi tham số đều suy từ thứ máy tự báo cáo, vì
một hằng số hợp với card 8 GB sẽ vừa phí trên card 24 GB vừa gây tràn trên card 4 GB:

| Tham số | Suy từ đâu |
|---|---|
| `thiet_bi("embedding")` / `("rerank")` | `torch.cuda.is_available()`, có thể ép riêng từng vai trò |
| `kich_thuoc_lo_embedding()` | VRAM **còn trống**, chặn trên bởi `EMBEDDING_BATCH_SIZE` |
| `so_worker_vision()` | min(trần cấu hình, số nhân CPU, VRAM còn trống) |

Tách RIÊNG hai vai trò embedding và rerank thay vì một công tắc chung là có chủ đích: chúng có
hồ sơ tài nguyên khác hẳn nhau (một chạy lô lớn lúc build, một chạy vài chục cặp mỗi câu hỏi
và phải chia VRAM với LLM), nên trên card nhỏ, cấu hình hợp lý có thể là embedding GPU +
rerank CPU. Điều đó chỉ nói được nếu hai vai trò tách riêng.

**Một phép đo lật ngược trực giác.** "Có GPU thì cứ tăng batch cho nhanh" là sai ở đây:

| batch | chunk/s | VRAM đỉnh |
|---:|---:|---:|
| 16 | 329 | 1,21 GB |
| 32 | 339 | 1,31 GB |
| 128 | 337 | 1,92 GB |
| 256 | **287** | 2,80 GB |

Thông lượng đứng yên từ 16 tới 128 rồi **tụt** ở 256, trong khi VRAM tăng đều — nút thắt không
nằm ở độ song song của lô. Vì vậy hàm chọn batch chỉ dùng VRAM để **hạ** batch khi máy chật,
không bao giờ nâng lên để "tận dụng GPU". Cùng một logic áp cho số worker: đo lại trên máy
rảnh cho thấy **GPU đã bão hoà 84% ngay từ MỘT worker**, nên 2→4 worker không lợi gì và
`SO_WORKER_VISION` mặc định đã đổi từ 4 xuống 2 (§8.5).

**CHIA GIAI ĐOẠN — và bằng chứng nó cần thật.** Riêng ba model của giai đoạn truy vấn đã là
8,07 GB (LLM 4,75 + reranker 2,20 + embedding 1,12), không vừa card 7,96 GB — chưa tính model
vision của giai đoạn ingestion. Tràn VRAM **không báo lỗi**:
driver âm thầm đẩy phần thừa sang RAM hệ thống, hoặc Ollama nạp/nhả model liên tục giữa các
lượt gọi — chậm hơn cả chạy thuần CPU.

Cách chia dựa trên một sự thật về LUỒNG SỬ DỤNG chứ không phải về phần cứng: người dùng bấm
"Đọc tài liệu" rồi mới hỏi, nên hai giai đoạn không bao giờ chạy đồng thời.

```
INGESTION  ->  Vision/OCR + embedding
   │  ket_thuc_ingestion(): nhả model vision (keep_alive=0) + dọn bộ đệm CUDA
   ▼
QUERY      ->  reranker + LLM + embedding
```

Đo thật bằng `nvidia-smi` và `/api/ps`, con số VRAM khác hẳn ước lượng ban đầu — và cái
sai đó đủ để lật ngược kết luận:

| Model | VRAM (đo bằng `nvidia-smi` + `/api/ps`) |
|---|---:|
| qwen3:4b (num_ctx 16384) | **4,75 GB** |
| bge-reranker-v2-m3 | 2,20 GB |
| multilingual-e5-base | 1,12 GB |
| **Tổng ở giai đoạn QUERY** | **8,07 GB** > 7,96 GB của card |

Hậu quả quan sát được khi để cả ba trên GPU (ứng dụng thật, không phải benchmark): VRAM còn
trống **288 MB**, reranker mất ~58 giây mới nạp xong, lượt hỏi đầu tiên báo **50,8 giây** cho
bước lẽ ra mất 2,6 giây — **không một dòng lỗi nào**.

Vì vậy ranh giới giai đoạn làm hai việc chứ không phải một. Đo trên các lần build thật:

| Thời điểm | PyTorch giữ | VRAM còn trống |
|---|---:|---:|
| Ingestion vừa kết thúc | 2,78 GB | 0,79 GB |
| Sau khi nhả vision + dọn bộ đệm CUDA | 1,13 GB | 5,70 GB |
| Sau khi chuyển embedding sang CPU | **0,04 GB** | **6,79 GB** |

**Vì sao hy sinh đúng embedding** — lý do nằm ở quy mô công việc, không ở kích thước model.
Lúc truy vấn nó chỉ mã hoá 1–3 chuỗi ngắn: trên máy thoáng, GPU nhanh hơn CPU đúng **14 ms**
(6,1 so với 20,3). Reranker thì chấm vài chục cặp mỗi câu và GPU nhanh hơn 11,2 lần.

**Và kết quả thật còn tốt hơn phép tính trên giấy.** Khi VRAM bị tranh chấp, mã hoá câu hỏi
trên GPU *chậm hơn* trên CPU (682 ms so với 21 ms) vì nó phải chờ giành chỗ. Hết tranh chấp,
tổng truy xuất còn **0,45 giây** mỗi câu — tức rời GPU không phải một đánh đổi mà là một cải
thiện. Quyết định tính theo TỔNG VRAM chứ không phải phần còn trống, vì nó phải ổn định cho
cả phiên; card lớn hơn ngưỡng thì giữ tất cả trên GPU.

**Một lỗ hổng chỉ lộ ra khi chạy ứng dụng thật.** Bước chuyển giai đoạn ở trên chỉ chạy
sau khi người dùng bấm "Đọc tài liệu". Nhưng phần lớn phiên làm việc **không** bắt đầu như
vậy — người ta mở app lên và hỏi ngay trên index đã có. Trong đúng những phiên đó, bước
chuyển giai đoạn chưa từng chạy, nên embedding vẫn nằm trên GPU tranh VRAM với LLM suốt cả
phiên. Benchmark hoàn toàn không thấy điều này vì nó luôn chạy ingestion trước.

| Lượt hỏi đầu tiên của một phiên mới | Truy xuất |
|---|---:|
| Trước khi có quản lý VRAM | **50,8 s** |
| Sau khi nhả vision, embedding vẫn mặc định ở GPU | 22,8 s |
| Câu thứ hai trong cùng phiên đó | 7,4 s |
| **Sau khi đổi mặc định: embedding ở CPU, rerank ở GPU** | **0,7 s** |
| Trung vị các câu tiếp theo, đo lại đầy đủ (§8.7) | **0,45 s** |

Cách sửa là đổi **trạng thái mặc định** chứ không thêm một bước chuyển nữa: mặc định phải là
trạng thái của giai đoạn HAY GẶP NHẤT (query), còn ingestion — vốn luôn đi qua
`bat_dau_ingestion()` — thì tự nâng embedding lên GPU đúng lúc cần rồi trả về sau. Card rộng
hơn ngưỡng thì không có đánh đổi nào, giữ tất cả trên GPU.

Bài học lặp lại đúng chủ đề của cả tài liệu này: **một cơ chế chỉ đúng ở đường đi mà ta đã
nghĩ tới**. Ở đây phép đo tự động đi qua ingestion nên không bao giờ chạm vào đường mà người
dùng thật đi nhiều nhất, và chỉ việc mở ứng dụng lên bấm thử mới lộ ra.


**Ràng buộc khi sửa module này:** không có GPU thì mọi hàm phải thành không-làm-gì chứ không
được ném lỗi. Máy chỉ có CPU là môi trường mặc định của người chấm đồ án, và một tính năng
lẽ ra chỉ là tối ưu mà làm sập luồng build là một đánh đổi không bao giờ chấp nhận được.
Bộ test ép `co_cuda()` trả `False` để chạy đúng nhánh đó, thay vì phụ thuộc vào việc máy chạy
test có card hay không — nếu phụ thuộc thì trên máy có GPU nhánh ấy sẽ không bao giờ được
kiểm, tức lỗi chỉ lộ ra ở máy người khác.

**NÚT THẮT ĐÃ CHUYỂN — và đây mới là kết luận đáng nhớ nhất.** Sau đợt này, bảng chi phí đọc
lên hoàn toàn khác:

| Giai đoạn | Chi phí lớn nhất | Tỉ trọng |
|---|---|---:|
| Ingestion (cache rỗng) | chú thích ảnh bằng model vision | **88,5%** |
| Ingestion (cache đầy) | — (3,35 s, nhanh hơn 77×) | — |
| Query | LLM sinh chữ (chủ yếu là chuỗi suy luận của qwen3) | **~90%** |

Truy xuất — thứ cả đợt tối ưu này nhắm vào — nay chỉ còn 2,6 giây và **không còn là chỗ đáng
tối ưu tiếp**. Embedding, sau khi lên GPU, chỉ còn chiếm 3,3% chi phí ingestion. Mọi nỗ lực
tối ưu tiếp theo mà không nhắm vào hai ô in đậm ở trên đều là tối ưu nhầm chỗ, và bảng này
tồn tại chính để chặn điều đó.

---

## 6. Triển khai và chạy hệ thống

**Yêu cầu:** Python 3.11+ (đã kiểm chứng trên 3.14) · [Ollama](https://ollama.com) đã cài và
`ollama pull qwen3:4b` · không bắt buộc GPU.

```bash
pip install -r requirements.txt      # lần đầu tự tải embedding ~1.1GB + rerank ~2.2GB
streamlit run app.py                 # → http://localhost:8501
pytest tests/ -v                     # hoặc: pytest -m "not slow"
python evaluation/run_evaluation.py  # sau khi build index + điền test_questions.json
```

Đo mức OVERFIT (§5.64) — con số quan trọng hơn từng bảng điểm riêng lẻ:
```bash
python evaluation/run_evaluation.py --nhanh          # chỉ truy xuất, bộ in-sample
python evaluation/run_evaluation.py --nhanh --held-out  # chỉ truy xuất, bộ held-out
python evaluation/run_evaluation.py --khoang-cach    # cả hai bộ + chênh lệch
```

Các script kiểm định (mỗi cái đo độ tin cậy của MỘT cơ chế, trên bộ ca đã biết trước đáp án):
```bash
python evaluation/kiem_dinh_judge.py                 # thước đo Faithfulness (§5.43)
python evaluation/kiem_dinh_doi_chieu.py --so-lan 3  # cơ chế phát hiện mâu thuẫn (§5.59)
python evaluation/kiem_dinh_viet_lai.py --chi-tang-1 # nhận diện câu nối tiếp (§5.58)
python evaluation/kiem_dinh_viet_lai.py --truy-xuat "Một câu hỏi đầy đủ về tài liệu của bạn"
```

Sao chép `.env.example` → `.env` nếu muốn đổi giá trị mặc định (không bắt buộc).

Kiểm thử độc lập từng module:
```python
from rag.document_loader import doc_pdf
from rag.chunking import chia_chunk
from rag.embedding import EmbeddingService
from rag.vector_store import VectorStore
from pathlib import Path

print(doc_pdf(Path("data/raw/ten_file.pdf")))
print(chia_chunk([{"nguon": "x.pdf", "trang": 1, "noidung": "..."}]))
svc = EmbeddingService()
store = VectorStore(dimension=svc.dimension)
```

Chi tiết cách chạy, cấu trúc thư mục và các giới hạn đã biết: xem [README.md](README.md).
