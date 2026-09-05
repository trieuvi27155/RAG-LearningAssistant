# HƯỚNG DẪN TOÀN TẬP DỰ ÁN `rag-do-an`

> Tài liệu này viết cho người **chưa biết gì** về dự án. Không giả định bạn đã đọc code,
> đã biết RAG là gì, hay đã từng chạy Streamlit. Đọc tuần tự từ trên xuống là hiểu được.
>
> - **Phần I** giải thích tổng quan: dự án làm gì, RAG là gì, hệ thống có những bộ phận nào,
>   dữ liệu chảy qua chúng theo thứ tự nào.
> - **Phần II** đi vào từng file, từng hàm: chữ ký, đầu vào, đầu ra, công dụng, và lý do
>   nó tồn tại.
> - **Phần III** là vận hành: cài, chạy, chỉnh tham số, xử lý sự cố.
> - **Phần IV** là tra cứu nhanh: từ điển thuật ngữ, bảng "hàm này nằm ở đâu".
> - **Phần V** là nhận xét phản biện về thiết kế — chỗ mạnh, chỗ đáng chú ý, chỗ có rủi ro.
>
> Ba tài liệu đã có sẵn trong repo phục vụ mục đích khác, nên đừng nhầm:
> `README.md` là hướng dẫn dùng, `ARCHITECTURE.md` là nhật ký quyết định thiết kế
> (§5.1 → §5.68), `KET_QUA_DO_DAC.md` là số liệu đo. Tài liệu bạn đang đọc là **bản đồ để
> đi vào code**.

---

## MỤC LỤC

**PHẦN I — TỔNG QUAN**
1. Dự án này giải quyết bài toán gì
2. RAG là gì, giải thích từ con số không
3. Các "nhân vật" trong hệ thống: 4 mô hình AI và 3 kho dữ liệu
4. Bản đồ thư mục
5. Hai luồng dữ liệu: Ingestion và Query
6. Vòng đời một câu hỏi — kể lại từng bước
7. Các cấu trúc dữ liệu đi xuyên hệ thống
8. Bản đồ phụ thuộc: file nào gọi file nào

**PHẦN II — CHI TIẾT TỪNG FILE**
9. `config.py` — tầng cấu hình
10. `app.py` — tầng giao diện
11. Nhóm Ingestion: `document_loader`, `image_extractor`, `vision_caption`, `chunking`, `bo_nho_dem`, `do_thoi_gian`
12. Nhóm Lưu trữ & Tìm kiếm: `embedding`, `vector_store`, `lexical_search`, `reranker`
13. Nhóm Query: `tiep_noi_hoi_thoai`, `rag_pipeline`, `citation`, `doi_chieu_nguon`
14. Nhóm Hạ tầng: `tai_nguyen_gpu`
15. Thư mục `evaluation/`
16. Thư mục `tests/`

**PHẦN III — VẬN HÀNH**
17. Cài đặt và chạy
18. Các biến cấu hình cần biết
19. Sự cố thường gặp

**PHẦN IV — TRA CỨU NHANH**
20. Từ điển thuật ngữ
21. Bảng tra: hàm ↔ file

**PHẦN V — NHẬN XÉT PHẢN BIỆN**

---
---

# PHẦN I — TỔNG QUAN

## 1. Dự án này giải quyết bài toán gì

### 1.1. Bài toán

Bạn có một đống tài liệu học tập: giáo trình PDF 700 trang, slide bài giảng PPTX, báo cáo
DOCX, đề cương, sách tiếng Anh. Khi cần tra một khái niệm, bạn phải nhớ nó nằm ở file nào,
mở ra, Ctrl+F, và Ctrl+F chỉ tìm được **đúng chữ bạn gõ** — hỏi "overfitting là gì" mà tài
liệu viết "quá khớp" thì không ra.

Đưa cả đống đó cho ChatGPT thì vướng ba chuyện:

1. **Không nhét vừa.** Một giáo trình 700 trang vượt xa cửa sổ ngữ cảnh của mọi mô hình phổ thông.
2. **Không kiểm chứng được.** Model trả lời trôi chảy, nhưng bạn không biết nó lấy từ đâu,
   và nó có thể bịa (hallucination) một cách rất thuyết phục.
3. **Không riêng tư / tốn tiền.** Tài liệu nội bộ đưa lên dịch vụ ngoài là một quyết định
   không phải lúc nào cũng chấp nhận được, và API trả phí thì tính tiền theo token.

### 1.2. Lời giải của dự án

Một hệ thống hỏi–đáp **chạy hoàn toàn trên máy cá nhân**, dùng kỹ thuật **RAG**
(Retrieval-Augmented Generation — sinh câu trả lời có tăng cường bằng truy xuất):

- Nạp tài liệu PDF / PPTX / DOCX vào, hệ thống tự đọc, cắt nhỏ, và lập chỉ mục theo **ngữ nghĩa**.
- Bạn hỏi bằng tiếng Việt hoặc tiếng Anh; hệ thống tìm những đoạn liên quan nhất, đưa **chỉ
  những đoạn đó** cho mô hình ngôn ngữ, và bắt nó chỉ được trả lời dựa trên đó.
- Mỗi câu trả lời kèm **trích dẫn**: tên file + số trang/slide, để bạn tự mở ra đối chiếu.

### 1.3. Bốn thứ khiến dự án này khác một chatbot RAG mẫu

Đây là phần đáng chú ý nhất về mặt học thuật, và cũng là phần chiếm nhiều code nhất:

| Tính năng | Vấn đề nó giải quyết | File chịu trách nhiệm |
|---|---|---|
| **Chế độ KIỂM CHỨNG** | Model nhỏ có xu hướng "gật đầu" theo người hỏi (sycophancy). Hỏi "Pháp luật ra đời trước nhà nước, đúng không?" thì nó đồng ý dù tài liệu nói ngược. Hệ thống nhận diện câu dạng khẳng định và ép model ra phán quyết ĐÚNG / SAI / KHÔNG ĐỀ CẬP kèm trích nguyên văn. | `rag_pipeline.py` |
| **Hiểu câu hỏi nối tiếp** | "Thế còn dấu hiệu thứ hai?" — câu này vô nghĩa nếu tách khỏi lượt trước. Hệ thống ghép ngữ cảnh hội thoại vào truy vấn, **tất định**, không tốn thêm lượt gọi mô hình. | `tiep_noi_hoi_thoai.py` |
| **Đối chiếu chéo nguồn** | Giáo trình cũ ghi "ba đặc điểm", slide mới ghi "năm đặc điểm". Mở riêng từng file thì mỗi file đều tự nhất quán; chỉ khi đặt cạnh nhau mới lộ. Hệ thống cảnh báo và **cố ý không phân xử** ai đúng. | `doi_chieu_nguon.py` |
| **Đọc được tài liệu khó** | PDF scan (không có lớp text), PDF font hỏng (chữ ra `(cid:12)`), PDF hai cột, bảng, hình có số liệu. Hệ thống có OCR dự phòng, dò lại tham số đọc, đọc theo cột, và dùng mô hình vision mô tả hình. | `document_loader.py`, `vision_caption.py`, `image_extractor.py` |

### 1.4. Ràng buộc cốt lõi xuyên suốt code

Có một triết lý lặp đi lặp lại trong toàn bộ dự án, và nếu bạn nắm được nó thì đọc code sẽ
dễ hơn rất nhiều:

> **Lỗi im lặng là kẻ thù số một.**

Rất nhiều đoạn code trong dự án không tồn tại để làm hệ thống *tốt hơn*, mà để làm cho việc
hệ thống *hỏng* trở nên **nhìn thấy được**. Vài ví dụ có thật, đều được ghi lại trong comment:

- Đổi model embedding mà quên build lại index → FAISS vẫn chạy bình thường, chỉ có điều kết
  quả gần như ngẫu nhiên. → `VectorStore.ly_do_khong_tuong_thich()` ghi "vân tay" cấu hình
  lúc build và so lại.
- Không khai báo `num_ctx` → Ollama cấp mặc định 4096 token, prompt ~4900 token bị **cắt im
  lặng từ đầu**, tức xoá đúng đoạn trích liên quan nhất. → `_tinh_num_ctx()` + `nen_ngu_canh()`
  + đọc bộ đếm token thật của máy chủ để cảnh báo.
- Nạp một giáo trình scan 383 trang khi Ollama đang tắt → index gồm 379 chunk `"[HÌNH]"`
  giống hệt nhau, hệ thống báo "build thành công". → `_bo_ban_ghi_anh_rong()` +
  `_canh_bao_tai_lieu_khong_doc_duoc()`.

Khi bạn thấy một hàm có vẻ thừa, hãy đọc docstring của nó — gần như luôn có một dòng
"đã gặp thực tế" giải thích vì sao nó ở đó.

---

## 2. RAG là gì, giải thích từ con số không

### 2.1. Ý tưởng một câu

> Thay vì bắt mô hình ngôn ngữ **nhớ** kiến thức, ta **đưa kiến thức cho nó ngay lúc hỏi**.

Giống như thi mở sách. Mô hình không cần thuộc bài; nó cần biết đọc. Việc của hệ thống RAG
là **lật đúng trang sách** đưa cho nó.

### 2.2. Vấn đề: máy tính không hiểu "nghĩa"

Ctrl+F so khớp ký tự. "Overfitting" và "quá khớp" là hai chuỗi ký tự hoàn toàn khác nhau,
nên Ctrl+F thua.

Giải pháp là **embedding**: biến một đoạn văn bản thành một dãy số (vector) sao cho **hai
đoạn có nghĩa gần nhau thì hai vector cũng gần nhau trong không gian**.

Ví dụ tưởng tượng với không gian 3 chiều (thực tế dự án dùng 768 chiều):

```
"Overfitting là hiện tượng mô hình học thuộc dữ liệu huấn luyện"  →  [0.81, 0.12, -0.34]
"Quá khớp xảy ra khi mô hình bám quá sát tập train"               →  [0.79, 0.15, -0.31]   ← rất gần
"Thư viện mở cửa từ 7h30 đến 21h các ngày trong tuần"             →  [-0.22, 0.65, 0.41]   ← rất xa
```

Độ gần được đo bằng **cosine similarity** — cosin của góc giữa hai vector. Bằng 1 là trùng
hướng, bằng 0 là vuông góc (không liên quan). Đây là con số bạn sẽ thấy khắp nơi trong code
dưới tên `diem_similarity`.

Mô hình làm việc biến chữ thành vector ở đây là **`intfloat/multilingual-e5-base`** — một mô
hình đa ngôn ngữ, nên câu hỏi tiếng Việt vẫn tìm được tài liệu tiếng Anh và ngược lại. Đây
là lý do hệ thống làm được **truy xuất chéo ngôn ngữ**.

### 2.3. Tại sao phải cắt nhỏ tài liệu (chunking)

Không thể embed cả cuốn sách thành một vector: một vector thì chỉ mô tả được "chủ đề chung"
của cả cuốn, mất hết chi tiết. Nên tài liệu được cắt thành **chunk** — những đoạn nhỏ
(dự án này dùng ~160 token, khoảng 2–4 câu).

Đánh đổi cần nhớ:

- Chunk **quá to** → một vector phải gánh nhiều ý, khớp câu hỏi kém sắc.
- Chunk **quá nhỏ** → mất ngữ cảnh, câu bị cắt đôi giữa chừng.

Dự án xử lý đánh đổi này bằng hai cơ chế: **overlap** (hai chunk liền nhau chồng lấn 32
token, để câu bị cắt vẫn nguyên vẹn ở một trong hai) và **mở rộng ngữ cảnh lúc truy vấn**
(chọn xong chunk khớp nhất thì nối thêm chunk hàng xóm — xem `_dung_doan_trich`).

### 2.4. Ba bước của RAG, và bước thứ tư mà dự án thêm vào

```
[1] RETRIEVE (truy xuất)   — tìm những chunk liên quan nhất với câu hỏi
[2] AUGMENT  (tăng cường)  — nhét những chunk đó vào prompt cùng câu hỏi
[3] GENERATE (sinh)        — mô hình ngôn ngữ đọc prompt và viết câu trả lời
[4] VERIFY   (kiểm chứng)  ← phần dự án này đầu tư nhiều nhất
```

Bước [4] gồm: bắt mô hình gắn số `[1]`, `[2]` cho từng ý; đọc ngược lại các số đó để biết nó
dùng nguồn nào; đo tỉ lệ câu trả lời trùng nguyên văn tài liệu; và đối chiếu chéo các nguồn
xem chúng có nói ngược nhau không.

### 2.5. Vì sao "chỉ tìm bằng vector" là chưa đủ

Tìm kiếm theo vector (gọi là *dense retrieval*) giỏi về nghĩa nhưng dở về **từ hiếm**: mã
định danh, tên riêng, thuật ngữ mà mô hình chưa gặp bao giờ (out-of-vocabulary). Hỏi
"Điều 47 quy định gì" thì "47" bị vector hoá thành một khái niệm mờ nhạt.

Vì thế dự án có thêm nhánh **BM25** — thuật toán xếp hạng theo từ khoá kinh điển, họ hàng
với TF-IDF. Hai nhánh được hợp nhất bằng **RRF (Reciprocal Rank Fusion)**: thay vì cộng
điểm (hai thang đo khác nhau, cộng vào là nhánh nào thang lớn hơn sẽ nuốt nhánh kia), RRF
cộng **nghịch đảo thứ hạng**:

```
điểm_RRF(chunk) = Σ  trọng_số_nhánh / (K + thứ_hạng_trong_nhánh)      với K = 60
```

Điều đáng chú ý: trong dự án này `TRONG_SO_BM25 = 0.0` — tức nhánh BM25 **mặc định tắt** vai
trò xếp hạng. Lý do đã đo được: với câu hỏi tiếng Việt trên tài liệu tiếng Anh, BM25 không
khớp nổi từ nào với tài liệu **đúng**, nhưng khớp rất "tự tin" với tài liệu tiếng Việt
**sai**. BM25 chỉ còn giữ một vai trò hẹp hơn là **cứu hộ** (`SO_UNG_VIEN_BM25_CUU_HO = 10`):
bơm thêm 10 ứng viên vào tập đem đi rerank với điểm RRF bằng 0, tức giúp **recall** mà không
có quyền **precision**. Đây là một ví dụ rất đẹp của việc tách bạch hai vai trò của cùng một
công cụ.

### 2.6. Rerank — tầng lọc thứ hai

Embedding mã hoá câu hỏi và tài liệu **riêng biệt** rồi mới so sánh (gọi là *bi-encoder*).
Nhanh, nhưng thô: nó không bao giờ được "nhìn" câu hỏi và đoạn văn cùng lúc.

**Cross-encoder** thì đọc cả cặp `(câu hỏi, đoạn văn)` cùng lúc và chấm một điểm liên quan.
Chính xác hơn hẳn, nhưng chậm hơn hẳn (phải chạy mô hình cho **từng** cặp, không dùng lại
được vector đã tính sẵn).

Nên kiến trúc chuẩn — và dự án này dùng đúng vậy — là **hai tầng**:

```
9.285 chunk  --FAISS + BM25-->  60 ứng viên  --cross-encoder-->  xếp lại  --lọc-->  4 đoạn trích
   (rẻ, thô)                                    (đắt, tinh)
```

Mô hình rerank: `BAAI/bge-reranker-v2-m3` (~2.2 GB).

---

## 3. Các "nhân vật" trong hệ thống

### 3.1. Bốn mô hình AI

| Vai trò | Mô hình mặc định | Chạy ở đâu | Dùng lúc nào | Kích thước |
|---|---|---|---|---|
| **Embedding** | `intfloat/multilingual-e5-base` | sentence-transformers (local, PyTorch) | Cả Ingestion lẫn Query | ~1.1 GB |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | sentence-transformers (local) | Query | ~2.2 GB |
| **LLM sinh câu trả lời** | `qwen3:4b` | **Ollama** (tiến trình riêng) | Query | ~2.6 GB |
| **Vision / OCR** | `qwen2.5vl:3b` | **Ollama** | Ingestion | ~3.2 GB |

Điểm cần nắm: hai mô hình đầu do **Python nạp trực tiếp** vào tiến trình Streamlit; hai mô
hình sau chạy trong **Ollama**, một máy chủ riêng nghe ở `http://localhost:11434`. Vì vậy
"Ollama chưa bật" là nguyên nhân số một khiến hệ thống "không chạy được" trên máy mới — và
`kiem_tra_may_chu_llm()` tồn tại để nói điều đó ra ngay ở thanh bên, trước khi bạn kịp gõ
câu hỏi đầu tiên.

Vì bốn mô hình cộng lại quá nặng cho một card đồ hoạ phổ thông, dự án có hẳn một module
`tai_nguyen_gpu.py` để **quản lý VRAM theo giai đoạn**: lúc Ingestion thì đưa embedding lên
GPU; xong thì nhả model vision khỏi VRAM ngay (thay vì đợi Ollama tự nhả sau 5 phút — đúng 5
phút mà người dùng vừa build xong và bắt đầu hỏi).

### 3.2. Ba kho dữ liệu trên đĩa

```
data/
├── raw/                 ← tài liệu gốc bạn upload (nguồn sự thật duy nhất)
├── images/              ← ảnh trích ra từ tài liệu (.png), tự sinh
├── cache/               ← bộ nhớ đệm, XOÁ LÚC NÀO CŨNG AN TOÀN
│   ├── tai_lieu/        ← kết quả đọc trọn vẹn 1 tài liệu (.json)
│   ├── ocr/             ← text OCR của từng trang (.txt)
│   ├── vision/          ← mô tả của model vision cho từng ảnh (.txt)
│   └── embedding/       ← vector của từng chunk (một file .npz duy nhất)
└── faiss_index/
    ├── index.faiss      ← ma trận vector (FAISS)
    ├── metadata.pkl     ← danh sách metadata song song, cùng thứ tự (pickle)
    └── index_info.json  ← "vân tay" cấu hình lúc build + sổ băm từng tài liệu
```

Quan hệ giữa chúng đáng nhớ:

- `raw/` là **thứ duy nhất không tái tạo được**. Mọi thứ còn lại đều sinh ra từ nó.
- `cache/` được đánh khoá theo **băm nội dung file** (không phải thời điểm sửa). Vì thế đổi
  tên file, copy sang thư mục khác, `git checkout` — đều **không** làm cache trượt. Ngược
  lại, sửa một ký tự trong file thì cache trượt ngay. Đây là lựa chọn có chủ đích, giải
  thích ở `bo_nho_dem.py`.
- `index_info.json` là chỗ chứa **sổ băm** `{tên file: hash}`. Đây là thứ cho phép build
  **tăng dần**: lần sau chỉ xử lý lại đúng tài liệu đã đổi.

---

## 4. Bản đồ thư mục

```
rag-do-an/
│
├── app.py                     842 dòng   Giao diện Streamlit + điều phối luồng Ingestion
├── config.py                1.293 dòng   Toàn bộ tham số, đọc từ .env, kèm giải thích dài
│
├── rag/                                  ── LÕI HỆ THỐNG ──
│   ├── __init__.py                       (rỗng, chỉ để đánh dấu package)
│   │
│   │   ── Ingestion: từ file → nội dung có metadata ──
│   ├── document_loader.py   1.262 dòng   Đọc PDF/PPTX/DOCX, OCR dự phòng, đọc theo cột, bảng
│   ├── image_extractor.py     396 dòng   Trích ảnh, lọc logo/trang trí, gắn chú thích lân cận
│   ├── vision_caption.py      264 dòng   Gọi model vision mô tả ảnh + OCR trang scan
│   ├── chunking.py            298 dòng   Cắt nội dung thành chunk, xử lý riêng cho bảng
│   ├── bo_nho_dem.py          357 dòng   Cache theo content hash (tài liệu/OCR/vision/vector)
│   ├── do_thoi_gian.py        106 dòng   Đo thời gian từng bước, in bảng tổng kết
│   │
│   │   ── Lưu trữ & tìm kiếm ──
│   ├── embedding.py           147 dòng   Bọc sentence-transformers, đếm token đúng chuẩn
│   ├── vector_store.py        319 dòng   Bọc FAISS + metadata + 3 chỉ mục phụ dựng lười
│   ├── lexical_search.py      100 dòng   BM25 tự cài (không dùng thư viện ngoài)
│   ├── reranker.py             88 dòng   Bọc CrossEncoder
│   │
│   │   ── Query: từ câu hỏi → câu trả lời có trích dẫn ──
│   ├── tiep_noi_hoi_thoai.py  403 dòng   Nhận diện & xử lý câu hỏi nối tiếp
│   ├── rag_pipeline.py      1.510 dòng   TRÁI TIM: truy xuất lai + rerank + prompt + gọi LLM
│   ├── citation.py            308 dòng   Lọc nguồn theo số [n] mà câu trả lời thật sự dẫn
│   ├── doi_chieu_nguon.py     369 dòng   Phát hiện hai nguồn nói ngược nhau
│   │
│   │   ── Hạ tầng ──
│   └── tai_nguyen_gpu.py      395 dòng   Dò phần cứng, chia VRAM theo giai đoạn
│
├── evaluation/                           ── ĐO ĐẠC (11 script + 3 bộ câu hỏi + 4 CSV) ──
│   ├── metrics.py             424 dòng   Precision@K, Recall@K, MRR, Faithfulness, Citation
│   ├── run_evaluation.py      505 dòng   Chạy đánh giá, in bảng, xuất CSV, so với lần trước
│   ├── tao_tai_lieu_mau.py    475 dòng   Sinh 6 tài liệu mẫu ĐỘC LẬP để chống overfit
│   ├── kiem_dinh_judge.py     195 dòng   Đo độ tin cậy của CHÍNH thước đo Faithfulness
│   ├── kiem_dinh_doi_chieu.py 187 dòng   Đo độ tin cậy của cơ chế phát hiện mâu thuẫn
│   ├── kiem_dinh_viet_lai.py  255 dòng   Đo nhận diện câu nối tiếp
│   ├── do_dau_cuoi.py         268 dòng   Đo đầu-cuối: nạp tài liệu → hỏi được
│   ├── do_nguong_rerank.py    144 dòng   Đo điểm rerank có tách được câu lạc đề không
│   ├── do_quy_mo_index.py     271 dòng   Đo FAISS Flat chịu được corpus tới cỡ nào
│   ├── do_worker_gpu.py       195 dòng   Đo số worker OCR tối ưu (kèm GPU util, VRAM)
│   ├── test_questions.json               29 câu hỏi in-sample (có nhãn trang đúng)
│   ├── test_questions_held_out.json      Bộ HELD-OUT, không dùng để tinh chỉnh
│   └── ket_qua_danh_gia*.csv             Kết quả các lần chạy (để so lần trước/sau)
│
├── tests/                                ── 22 file test + conftest, 335 test pytest ──
│
├── data/                                 ── DỮ LIỆU (xem §3.2) ──
├── TaiLieuTest/                          ── 26 tài liệu thật dùng để đo ──
│
├── .streamlit/config.toml                Bảng màu giao diện
├── .env.example                          Mẫu file cấu hình (31 KB, chú thích rất dài)
├── pytest.ini                            Cấu hình pytest + marker "slow"
├── requirements.txt                      Danh sách thư viện
├── README.md                             Hướng dẫn dùng
├── ARCHITECTURE.md                       Nhật ký quyết định thiết kế §5.1 → §5.68
├── KET_QUA_DO_DAC.md                     Số liệu đo đạc
└── chan_doan_rag.md                      Ghi chú chẩn đoán
```

**Quy ước đặt tên**: toàn bộ dự án đặt tên hàm/biến bằng **tiếng Việt không dấu**
(`doc_tai_lieu`, `chia_chunk`, `truy_xuat`, `cac_doan`). Thoạt nhìn lạ, nhưng nó nhất quán
và làm code đọc như văn xuôi tiếng Việt. Các thuật ngữ kỹ thuật đã chuẩn hoá quốc tế thì
giữ nguyên tiếng Anh (`chunk`, `embedding`, `rerank`, `metadata`).

---

## 5. Hai luồng dữ liệu

Đây là mô hình tư duy quan trọng nhất. **Toàn bộ hệ thống chỉ có hai luồng**, và chúng chỉ
gặp nhau ở hai điểm: cùng dùng một mô hình embedding, và cùng dùng một FAISS index.

### 5.1. Luồng INGESTION — chạy khi bạn bấm nút "Đọc tài liệu"

```
                       data/raw/*.pdf|pptx|docx
                                  │
        ┌─────────────────────────▼──────────────────────────┐
        │ app.xay_dung_lai_index()                           │
        │  • băm nội dung từng file          bo_nho_dem      │
        │  • so với sổ băm trong index  →  ai cần đọc lại?   │
        └─────────────────────────┬──────────────────────────┘
                                  │ chỉ những file MỚI / ĐÃ ĐỔI
                                  ▼
        ┌────────────────────────────────────────────────────┐
        │ document_loader.doc_nhieu_file()                   │
        │   └ doc_tai_lieu_co_cache()   ← tra cache trước    │
        │       └ doc_tai_lieu_hoan_chinh()                  │
        │           ├ doc_pdf / doc_pptx / doc_docx          │
        │           │    Pha 1: đọc text (dò x_tolerance,    │
        │           │            bảng, cột, tiêu đề, ảnh)    │
        │           │    Pha 2: OCR trang hỏng  (vision)     │
        │           │    Pha 3: dọn watermark, bỏ mục lục    │
        │           │    Pha 4: render ảnh, lọc logo         │
        │           ├ vision_caption.bo_sung_chu_thich()     │
        │           └ bỏ bản ghi ảnh rỗng                    │
        └─────────────────────────┬──────────────────────────┘
                                  │  List[{nguon, trang, noidung, ...}]
                                  ▼
        ┌────────────────────────────────────────────────────┐
        │ chunking.chia_chunk()                              │
        │   • bảng → giữ nguyên khối (tới sát giới hạn model)│
        │   • văn xuôi → RecursiveCharacterTextSplitter      │
        │   • ảnh → 1 chunk, không cắt                       │
        └─────────────────────────┬──────────────────────────┘
                                  │  List[{chunk_id, nguon, trang, vi_tri, noidung}]
                                  ▼
        ┌────────────────────────────────────────────────────┐
        │ bo_nho_dem.encode_co_cache()                       │
        │   • chunk nào đã có vector trong cache thì lấy lại │
        │   • phần còn lại → embedding_service.encode()     │
        └─────────────────────────┬──────────────────────────┘
                                  │  np.ndarray (n × 768), đã chuẩn hoá
                                  ▼
        ┌────────────────────────────────────────────────────┐
        │ VectorStore.them()  →  .luu()                      │
        │   index.faiss + metadata.pkl + index_info.json     │
        └────────────────────────────────────────────────────┘
```

Ba điều làm luồng này khác một pipeline RAG mẫu:

1. **Tăng dần theo mặc định.** Thêm 1 tài liệu vào 25 tài liệu cũ thì chỉ tài liệu mới được
   xử lý; 25 tài liệu kia giữ nguyên vector. Tài liệu bị xoá khỏi thư mục thì vector của nó
   bị gỡ khỏi index.
2. **Cache 4 tầng theo content hash.** Ngay cả khi buộc phải build lại toàn bộ index (đổi
   model embedding chẳng hạn), kết quả đọc tài liệu / OCR / chú thích ảnh vẫn dùng lại được.
3. **Ranh giới cache đặt ở "tài liệu đã xử lý TRỌN VẸN"**, tức sau khi đã chú thích ảnh —
   chứ không phải sau khi đọc text thô. Vì bước chú thích ảnh mới là bước đắt (~1,9 giây/hình).

### 5.2. Luồng QUERY — chạy mỗi lần bạn gõ một câu hỏi

```
   câu hỏi + lịch sử chat + tập nguồn được tick
                    │
                    ▼
  ┌────────────────────────────────────────────────────────────┐
  │ tiep_noi_hoi_thoai.chuan_bi_truy_van()                     │
  │   Đây có phải câu NỐI TIẾP? (tất định, không gọi LLM)      │
  │   → cau_hoi_chinh = câu đã ghép ngữ cảnh (nếu nối tiếp)    │
  │   → ngu_canh_llm  = khối câu hỏi trước, cho prompt          │
  └────────────────────────┬───────────────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────────────────┐
  │ RagPipeline.truy_xuat()                                    │
  │  1. encode câu hỏi   → vector 768 chiều                    │
  │  2. _ung_vien()      → FAISS top-60 + BM25 → RRF hợp nhất  │
  │                      → + 10 ứng viên BM25 "cứu hộ"         │
  │  3. _xep_hang_lai()  → cross-encoder chấm 30 (hoặc 12) đầu │
  │  4. lọc: trần đoạn/trang, trần đoạn ảnh                    │
  │  5. _dung_doan_trich() cho từng đoạn được chọn             │
  │     (mở rộng sang chunk hàng xóm trong ngân sách ký tự)    │
  │  6. sàn tuyệt đối (cosine ≥ 0.5)                           │
  │  7. sàn TƯƠNG ĐỐI (≥ 78% điểm cao nhất của chính lượt này) │
  │  8. sàn RERANK (≥ 0.001) ← đây mới là thứ chặn câu lạc đề  │
  └────────────────────────┬───────────────────────────────────┘
                           │  List[đoạn trích]  (rỗng = từ chối luôn)
                           ▼
  ┌────────────────────────────────────────────────────────────┐
  │ RagPipeline.sinh_cau_tra_loi_theo_luong()                  │
  │  • _phat_hien_ngon_ngu()   → "vi" hay "en"                 │
  │  • la_cau_hoi_kiem_chung() → chọn 1 trong 4 system prompt   │
  │  • nen_ngu_canh()          → ép đoạn trích vào ngân sách   │
  │  • _ghep_prompt()          → đánh số [1] [2] [3] cho từng đoạn│
  │  • _goi_llm_theo_luong()   → Ollama, stream=True           │
  │      lọc <think> theo luồng, đọc bộ đếm token thật         │
  └────────────────────────┬───────────────────────────────────┘
                           │  yield từng mảnh chữ
                           ▼
  ┌────────────────────────────────────────────────────────────┐
  │ doi_chieu_nguon.tim_mau_thuan()   ← chạy SAU khi hiện xong │
  │   tầng 1 tất định: khác số? lệch phủ định? cosine cao?     │
  │   tầng 2: LLM chấm từng cặp đáng ngờ                       │
  └────────────────────────┬───────────────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────────────────┐
  │ citation.loc_theo_tham_chieu()                             │
  │   đọc các số [n] trong câu trả lời → biết nó dùng nguồn nào│
  │ citation.do_bam_ngu_canh()                                 │
  │   % cụm 4 từ của câu trả lời trùng NGUYÊN VĂN ngữ cảnh     │
  └────────────────────────┬───────────────────────────────────┘
                           ▼
              app.py vẽ ra màn hình
```

### 5.3. Điểm gặp nhau — và cái bẫy ở đó

Hai luồng dùng chung `EmbeddingService` và `VectorStore`. Điều đó bắt buộc: vector câu hỏi
và vector tài liệu phải nằm trong **cùng một không gian ngữ nghĩa** thì cosine mới có nghĩa.

Cái bẫy: nếu bạn đổi `EMBEDDING_MODEL_NAME` sang một model khác **cùng số chiều** rồi quên
build lại index, FAISS vẫn chạy trơn tru và trả về kết quả trông rất bình thường — chỉ là
sai. Không có exception nào. Đó là lý do `index_info.json` ghi lại tên model và
`ly_do_khong_tuong_thich()` được gọi ở hai chỗ: cảnh báo trên thanh bên, và chặn build tăng
dần.

---

## 6. Vòng đời một câu hỏi — kể lại từng bước

Ví dụ cụ thể. Người dùng đã nạp 26 tài liệu, index có 9.285 chunk. Họ gõ:

> *"So sánh KNN với Naive Bayes về độ phức tạp và dữ liệu cần thiết"*

**Bước 0 — `app.py` nhận câu hỏi (`_dat_cau_hoi`).**
Câu hỏi được đẩy vào `st.session_state.messages`, đặt cờ `dang_xu_ly = True`, rồi
`st.rerun()` **ngay lập tức** — chưa gọi LLM. Lý do rất Streamlit: nếu gọi LLM tại đây, một
cú bấm bất kỳ trong lúc chờ sẽ khiến Streamlit huỷ ngang lần chạy hiện tại và mất câu trả
lời đang sinh dở. Sang lần chạy kế tiếp, mọi widget đã render ở trạng thái `disabled=True`,
lúc đó mới gọi LLM.

**Bước 1 — Chuẩn bị truy vấn (`chuan_bi_truy_van`).**
Câu này không có dấu hiệu nối tiếp ("thế còn", "cái đó", đại từ trống...), nên
`la_tiep_noi = False`, `cau_hoi_chinh = cau_hoi_goc`. Chi phí: một phép so chuỗi.

**Bước 2 — Mã hoá câu hỏi.**
`encode_cau_hoi()` thêm tiền tố `"query: "` (yêu cầu của họ model E5 — tài liệu thì dùng
`"passage: "`), chạy qua model, chuẩn hoá về độ dài 1. Ra một vector 768 chiều.

**Bước 3 — Lấy ứng viên (`_ung_vien`).**
`so_ung_vien = max(4 × 10, 60) = 60`. FAISS trả 60 chunk gần nhất. BM25 trả thêm 10 chunk
cứu hộ với điểm RRF = 0. Lọc theo `nguon_cho_phep` (những file người dùng còn tick). Hợp
nhất bằng RRF. Kết quả: ~65 ứng viên đã xếp hạng.

**Bước 4 — Rerank (`_xep_hang_lai`).**
`la_cau_hoi_phuc_tap()` thấy chuỗi "so sánh" → **phức tạp** → được cấp ngân sách đầy đủ:
chấm 30 ứng viên đầu (câu đơn giản chỉ chấm 12). Cộng thêm các ứng viên cứu hộ dù chúng nằm
ngoài top-30. Cross-encoder chấm điểm từng cặp, xếp lại thứ tự. **Điểm cosine giữ nguyên,
không bị thay bằng điểm rerank** — rerank chỉ đổi *thứ tự chọn*, để mọi ngưỡng trong hệ
thống vẫn đo trên một thang duy nhất.

**Bước 5 — Đo độ đa dạng trang, quyết trần.**
Đếm số `(nguồn, trang)` khác nhau trong 20 ứng viên đầu. Nếu ≥ `TOP_K` thì áp trần
`SO_DOAN_TOI_DA_MOI_TRANG = 2` (chống việc các chunk liền kề cùng một trang chiếm sạch
TOP_K); nếu không đủ đa dạng thì **bỏ trần** (vì có câu hỏi mà toàn bộ câu trả lời nằm gọn
trong một trang).

**Bước 6 — Dựng đoạn trích (`_dung_doan_trich`).**
Với mỗi ứng viên được chọn: lấy chunk đó làm **neo**, rồi mở rộng luân phiên sang chunk
liền sau / liền trước cho tới khi chạm ngân sách `NGAN_SACH_KY_TU_MOI_DOAN = 1600` ký tự.
Được phép vượt tối đa 1 trang mỗi hướng. Phần chồng lấn do overlap được cắt bỏ bằng
`_noi_lien_mach`. Trang ghi trong trích dẫn vẫn là **trang của chunk neo**, còn mọi trang đã
đi qua được trả kèm ở `cac_trang`.

**Bước 7 — Ba tầng sàn lọc.**
- Sàn tuyệt đối `cosine ≥ 0.5`: chặn rác.
- Sàn tương đối `≥ 0.78 × điểm cao nhất của chính lượt này`: cosine của E5 trôi theo domain
  và ngôn ngữ, nên tỉ lệ *trong cùng một lượt* đáng tin hơn một hằng số.
- Sàn rerank `≥ 0.001`: **đây mới là thứ chặn được câu lạc đề**. Đo thực tế: câu lạc đề rơi
  về 0.000–0.003, câu đúng chủ đề (kể cả hỏi tiếng Anh trên tài liệu tiếng Việt) từ 0.019
  trở lên. Cosine không tách được hai nhóm này.

Giả sử còn 4 đoạn trích. `hoi_dap_theo_luong` phát ngay sự kiện `truy_xuat_xong` — sau
khoảng 2 giây — nên giao diện hiện được "Đã tìm 4 đoạn liên quan trong 2 tài liệu" trước khi
LLM viết chữ nào.

**Bước 8 — Chọn prompt.**
`_phat_hien_ngon_ngu()` thấy dấu tiếng Việt → `"vi"`. `la_cau_hoi_kiem_chung()` không khớp
mẫu nào → dùng `HE_THONG_PROMPT_VI` (prompt thường, 7 quy tắc). Không bật chế độ suy luận.

**Bước 9 — Nén ngữ cảnh (`nen_ngu_canh`).**
Tính ngân sách token còn lại sau khi trừ phần **cố định** (system prompt + câu hỏi + khối
ngữ cảnh hội thoại). Nếu 4 đoạn trích vượt ngân sách thì bỏ từ **đoạn xếp hạng thấp nhất
lên** — bỏ từ cuối để số thứ tự `[1] [2] [3]` của các đoạn còn lại không bị lệch.

**Bước 10 — Ghép prompt (`_ghep_prompt`).**

```
NGỮ CẢNH:
[1] (Nguồn: Bai5-PhanLopVoiKNN-NaiveBayes.docx, trang/slide 7)
KNN không có giai đoạn huấn luyện...

[2] (Nguồn: Bai5-PhanLopVoiKNN-NaiveBayes.docx, trang/slide 12)
Naive Bayes giả định các thuộc tính độc lập...

CÂU HỎI: So sánh KNN với Naive Bayes về độ phức tạp và dữ liệu cần thiết

Trả lời dựa trên ngữ cảnh trên:
```

**Bước 11 — Gọi Ollama (`_goi_llm_theo_luong`).**
`num_ctx` được tính động và **làm tròn lên theo thang gấp đôi** từ 16384. Lý do làm tròn:
Ollama coi `num_ctx` là một phần định danh của phiên bản model đang nạp, đổi giá trị giữa
hai lượt hỏi sẽ khiến nó **nạp lại model** (hàng chục giây trên CPU). Làm tròn theo bậc thì
gần như mọi câu hỏi rơi vào cùng một bậc.

Luồng trả về từng mảnh. `_LocSuyLuanTheoLuong` là một máy trạng thái bóc thẻ
`<think>...</think>` ra khỏi luồng đang chảy — cần thiết vì một mảnh có thể cắt ngang giữa
thẻ (`"<thi"` | `"nk>"`), mà đẩy thẳng ra màn hình thì người dùng thấy đúng phần suy luận
thô mà hệ thống đang cố giấu.

Mảnh cuối cùng (`done=True`) mang bộ đếm token **thật** của máy chủ. `_ghi_nhan_thong_ke_llm`
đọc nó và cảnh báo nếu `prompt_eval_count ≥ num_ctx` (prompt đã bị cắt) hoặc
`done_reason == "length"` (câu trả lời bị cắt cụt).

**Bước 12 — Đối chiếu chéo (`tim_mau_thuan`).**
Chạy **sau** khi câu trả lời đã hiện xong. Tầng 1 tất định: hai đoạn có khác số không, có
lệch phủ định không, cosine giữa chúng có đủ cao (≥ 0.88, tức đang nói cùng chủ đề) không.
Chỉ những cặp qua được tầng 1 mới bị đem đi cho LLM chấm, tối đa 3 cặp.

**Bước 13 — Trích dẫn (`loc_theo_tham_chieu`).**
Quét câu trả lời tìm các số `[n]`. Ba lớp dự phòng:
1. Số đoạn trích `[n]` — dạng chuẩn mà system prompt bắt buộc.
2. Số trang được nhắc ("theo Slide 109") — model nhỏ đôi khi vẫn làm vậy dù bị cấm.
3. Nguồn liên quan nhất, **đánh dấu rõ là phỏng đoán** (`la_suy_doan = True`).

Lớp 3 là một chi tiết đạo đức đáng chú ý: bản trước lặng lẽ hiển thị đoạn điểm cao nhất
dưới nhãn "Nguồn", tức trình bày phỏng đoán của hệ thống như thể là căn cứ mà câu trả lời đã
dùng. Nay giao diện nói thẳng: *"Câu trả lời không tự dẫn nguồn — đây là chỗ liên quan nhất
do hệ thống chọn, KHÔNG chắc là căn cứ đã dùng"*.

**Bước 14 — Vẽ ra màn hình.**
Câu trả lời được `bo_so_trich_dan()` gỡ các số `[n]` trước khi hiển thị (chúng là thứ tự đoạn
trích *trong prompt*, một thứ tự người đọc không nhìn thấy nên không tra ngược được). Kèm
theo: dòng nguồn, chỉ số bám nguồn (nếu ≥ 30%), cảnh báo mâu thuẫn (nếu có).

Toàn bộ kết quả được lưu **ngay trong tin nhắn** ở `session_state.messages`, không dùng biến
"trích dẫn hiện tại" dùng chung — nhờ vậy lật lại lịch sử chat vẫn thấy đúng trích dẫn của
từng câu trả lời.

---

## 7. Các cấu trúc dữ liệu đi xuyên hệ thống

Nắm được năm dict này là nắm được 80% cách đọc code.

### 7.1. "Trang" — đầu ra của `document_loader`

```python
{
    "nguon":   "Bai3-GomCum.docx",   # tên file, dùng làm khoá xuyên suốt hệ thống
    "trang":   7,                    # số trang PDF / số slide PPTX / số "trang" DOCX
    "noidung": "Thuật toán K-Means...",
}
```

Bản ghi **ảnh** cũng là một "trang", chỉ khác ở ba trường thêm vào:

```python
{
    "nguon": "Bai3-GomCum.docx", "trang": 7,
    "noidung": "[HÌNH] Hình 3.2: Sơ đồ phân cụm\nMô tả của model vision: biểu đồ...",
    "loai_noi_dung": "anh",
    "duong_dan_anh": "data/images/Bai3__t7_1.png",
    "co_chu_thich_vision": True,
}
```

> Ý tưởng thiết kế đáng học: **ảnh được đối xử như một trang văn bản**. Nhờ vậy nó đi qua
> chunking, embedding, FAISS, rerank y hệt văn bản, không cần một đường ống riêng.

### 7.2. "Chunk" — đầu ra của `chunking`, cũng là metadata lưu trong index

```python
{
    "chunk_id": "3f2a...",           # uuid4
    "nguon":    "Bai3-GomCum.docx",
    "trang":    7,
    "vi_tri":   2,                   # thứ tự chunk TRONG trang (0,1,2...), có thể khuyết số
    "noidung":  "...",
    "loai_noi_dung": "van_ban" | "bang" | "anh",
    "duong_dan_anh": "",             # chỉ có với ảnh
}
```

`vi_tri` là chìa khoá của việc mở rộng ngữ cảnh: nó cho biết chunk nào đứng trước chunk nào
trong tài liệu gốc.

### 7.3. "Đoạn trích" — đầu ra của `RagPipeline.truy_xuat()`

```python
{
    "nguon": "...", "trang": 7,
    "cac_trang": [6, 7],             # MỌI trang mà đoạn trích đi qua
    "noidung": "...",                # chunk neo + các chunk hàng xóm đã nối liền mạch
    "doan_khop": "...",              # RIÊNG chunk neo — phần thật sự khớp câu hỏi
    "diem_similarity": 0.87,         # cosine, luôn là cosine
    "loai_noi_dung": "van_ban",
    "duong_dan_anh": "",
}
```

Sự phân biệt `noidung` / `doan_khop` là một sửa lỗi quan trọng: bản trước cắt 400 ký tự
**đầu** vùng ngữ cảnh đã mở rộng để làm đoạn trích hiển thị, nên với tài liệu dài thì đoạn
hiển thị gần như luôn là phần đầu trang, chẳng liên quan gì tới câu hỏi.

### 7.4. "Trích dẫn" — đầu ra của `citation.loc_theo_tham_chieu()`

```python
{
    "nguon": "...", "trang": 7, "cac_trang": [6, 7],
    "doan_trich": "..." ,            # cắt còn tối đa 600 ký tự, đã bỏ mốc [BẢNG]/[HÌNH]
    "diem_similarity": 0.87,
    "so_hieu": 2,                    # số [n] trong prompt
    "cac_so": [2, 3],                # mọi số trỏ về cùng (nguồn, trang) này
    "la_suy_doan": False,            # True = model không dẫn số nào, đây là phỏng đoán
}
```

### 7.5. "Sự kiện streaming" — đầu ra của `hoi_dap_theo_luong()`

```python
{"loai": "truy_xuat_xong", "cac_chunk": [...], "giay": 2.1, "truy_van": {...}}
{"loai": "suy_luan",       "them": "Okay, let me..."}     # nháp nội bộ của model
{"loai": "cau_tra_loi",    "them": "KNN không có..."}     # chữ thật
{"loai": "dang_doi_chieu"}
{"loai": "xong",           "ket_qua": {...}}
```

`ket_qua` cuối cùng:

```python
{
    "cau_tra_loi":     str,          # còn nguyên các số [n]
    "cac_chunk_nguon": List[đoạn trích],
    "la_kiem_chung":   bool,
    "truy_van":        {"cau_hoi_goc", "cau_hoi_chinh", "ngu_canh_llm",
                        "la_tiep_noi", "da_viet_lai", "cac_truy_van_phu"},
    "mau_thuan":       List[{nguon_a, trang_a, nguon_b, trang_b, noi_dung_xung_dot}],
    "bam_nguon":       float,        # 0..1, tỉ lệ trùng nguyên văn
    "do_tre":          {"truy_xuat", "hien_dau_tien", "chu_dau_tien", "tong"},
}
```

---

## 8. Bản đồ phụ thuộc: file nào gọi file nào

```
                              config.py
                          (mọi file đều import)
                                  ▲
    ┌─────────────────────────────┼─────────────────────────────┐
    │                             │                             │
 app.py ──────────────────────────┴──────────► rag/*.py
    │
    ├──► embedding.EmbeddingService          (nạp 1 lần, @st.cache_resource)
    ├──► reranker.tao_reranker_neu_bat       (nạp 1 lần, @st.cache_resource)
    ├──► vector_store.VectorStore
    ├──► document_loader.{cac_file_tai_lieu, doc_nhieu_file}
    ├──► chunking.chia_chunk
    ├──► bo_nho_dem.{bam_file, encode_co_cache, dung_luong_cache, xoa_cache}
    ├──► tai_nguyen_gpu.{bat_dau_ingestion, ket_thuc_ingestion, co_cuda, thiet_bi}
    ├──► do_thoi_gian.{dat_lai, do, ghi_bao_cao}
    ├──► citation.{bo_so_trich_dan, loc_theo_tham_chieu}
    └──► rag_pipeline.{RagPipeline, kiem_tra_may_chu_llm, la_cau_hoi_kiem_chung,
                       LoiKhongKetNoiDuocOllama}

rag_pipeline.py  ──► embedding, vector_store, reranker
                 ──► tiep_noi_hoi_thoai.chuan_bi_truy_van
                 ──► citation.do_bam_ngu_canh
                 ──► doi_chieu_nguon.tim_mau_thuan
                 ──► vision_caption.ten_model_khop      (dùng lại phép so tên model)
                 ──► do_thoi_gian.do

document_loader.py ──► image_extractor  (ung_vien_anh_trang, luu_anh_trang_pdf,
                    │                    trich_anh_pptx, trich_anh_docx, loc_anh_lap_lai)
                    ──► vision_caption   (trang_can_ocr, ocr_trang_pdf,
                    │                    bo_sung_chu_thich_vision)
                    ──► bo_nho_dem       (khoa_tai_lieu, kho_tai_lieu, kho_ocr)
                    ──► tai_nguyen_gpu   (so_worker_vision)
                    ──► do_thoi_gian

vector_store.py    ──► lexical_search.BM25
chunking.py        ──► document_loader.{MOC_BANG_MO, MOC_BANG_DONG}   ← import ngược, chú ý
citation.py        ──► document_loader.{MOC_BANG_MO, MOC_BANG_DONG}, image_extractor.MOC_ANH
bo_nho_dem.py      ──► config (không phụ thuộc module rag nào khác)
tai_nguyen_gpu.py  ──► config, torch, ollama

evaluation/*  ──► rag/*  +  config          (dùng lại đúng đường mã của ứng dụng thật)
tests/*       ──► rag/*, app (qua streamlit.testing.AppTest), config
```

**Nhận xét về kiến trúc phụ thuộc**: đây là đồ thị **có hướng, gần như không có chu trình**.
`config.py` là lá (không import gì trong dự án), `bo_nho_dem` và `lexical_search` là tầng
dưới cùng, `rag_pipeline` và `app.py` là tầng trên. Hai chỗ có mùi "import ngược" là
`chunking → document_loader` và `citation → document_loader` — nhưng chúng chỉ lấy **hằng
số mốc** (`"[BẢNG]"`, `"[/BẢNG]"`, `"[HÌNH]"`), không tạo phụ thuộc chức năng. Nếu muốn dọn
sạch, ba hằng số này nên chuyển vào `config.py`.
---
---

# PHẦN II — CHI TIẾT TỪNG FILE

Quy ước trình bày trong phần này: mỗi file có (a) **vai trò**, (b) **vị trí trong luồng**,
(c) **danh sách đầy đủ hàm/class** kèm chữ ký, đầu vào, đầu ra, công dụng, và (d) ghi chú
"vì sao" cho những chỗ thiết kế không hiển nhiên.

Hàm bắt đầu bằng dấu gạch dưới `_` là **hàm nội bộ** (private) — chỉ dùng trong chính file
đó. Đây là quy ước của Python, không phải cưỡng chế của ngôn ngữ.

---

## 9. `config.py` — tầng cấu hình (1.293 dòng)

### 9.1. Vai trò

Một chỗ duy nhất chứa mọi tham số. Mọi file khác `import config` rồi đọc
`config.TEN_THAM_SO`. Không có số ma thuật (magic number) rải rác trong logic.

File dài 1.293 dòng nhưng **phần lớn là chú thích**: mỗi tham số quan trọng đều kèm một
đoạn giải thích tại sao giá trị mặc định là như vậy, thường có số đo kèm theo. Đọc file này
là cách nhanh nhất để hiểu các đánh đổi của hệ thống.

### 9.2. Cơ chế nạp cấu hình

```python
def _nap_file_env(duong_dan_env: Path) -> None
```
Đọc file `.env` (nếu có) và nạp các dòng `KEY=VALUE` vào `os.environ`. Tự cài thay vì dùng
`python-dotenv` để bớt một dependency.

```python
def _lay_int(ten_bien: str, mac_dinh: int) -> int
def _lay_float(ten_bien: str, mac_dinh: float) -> float
def _lay_str(ten_bien: str, mac_dinh: str) -> str
def _lay_bool(ten_bien: str, mac_dinh: bool) -> bool
```
Bốn hàm đọc biến môi trường có ép kiểu và có giá trị mặc định. Vì vậy **thiếu file `.env`
thì hệ thống vẫn chạy đúng** với toàn bộ mặc định — không có bước cấu hình bắt buộc nào.

```python
def _so_worker_mac_dinh(toi_da: int) -> int
```
Suy số worker mặc định từ chính máy đang chạy, chặn trên bằng `toi_da`.

### 9.3. Nhóm ĐƯỜNG DẪN

| Hằng số | Giá trị | Ghi chú |
|---|---|---|
| `BASE_DIR` | thư mục chứa `config.py` | mọi đường dẫn khác suy ra từ đây |
| `DATA_DIR` | `data/` | |
| `RAW_DOCS_DIR` | `data/raw/` | tài liệu gốc |
| `FAISS_INDEX_DIR` | `data/faiss_index/` | |
| `FAISS_INDEX_FILE` | `.../index.faiss` | ma trận vector |
| `METADATA_MAPPING_FILE` | `.../metadata.pkl` | danh sách metadata song song |
| `INDEX_INFO_FILE` | `.../index_info.json` | vân tay cấu hình + sổ băm tài liệu |
| `IMAGES_DIR` | `data/images/` | ảnh trích ra |
| `CACHE_DIR` | `data/cache/` | 4 kho cache |
| `EVAL_DIR` | `evaluation/` | |
| `TEST_QUESTIONS_FILE` | `evaluation/test_questions.json` | bộ in-sample |
| `TEST_QUESTIONS_HELD_OUT_FILE` | `..._held_out.json` | bộ held-out |

### 9.4. Nhóm EMBEDDING & CHUNKING

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-base` | mô hình biến chữ → vector |
| `_LA_HO_E5` | tự suy | có phải họ E5 không (quyết định tiền tố) |
| `EMBEDDING_QUERY_PREFIX` | `"query: "` | E5 yêu cầu tiền tố khác nhau cho câu hỏi… |
| `EMBEDDING_PASSAGE_PREFIX` | `"passage: "` | …và cho đoạn tài liệu |
| `EMBEDDING_BATCH_SIZE` | `64` | số chunk encode một lượt |
| `CHUNK_SIZE_TOKENS` | `160` | kích thước chunk văn xuôi |
| `CHUNK_OVERLAP_TOKENS` | `32` | phần chồng lấn giữa 2 chunk liền kề |
| `BIEN_AN_TOAN_TOKEN` | `16` | biên an toàn khi hạ chunk cho vừa giới hạn model |
| `CHUNK_SEPARATORS` | `['\n## ', '\n# ', '\n\n', '\n', '. ', ' ', '']` | thứ tự ưu tiên cắt |
| `BAT_NHAN_DIEN_TIEU_DE` | `True` | đánh dấu tiêu đề PDF thành `## ` để splitter ưu tiên cắt ở đó |
| `TY_LE_KICH_THUOC_CHU_TIEU_DE` | `1.15` | chữ to hơn trung bình 15% thì coi là tiêu đề |
| `DO_DAI_TOI_DA_TIEU_DE` | `90` | tiêu đề phải là dòng ngắn |
| `TIKTOKEN_ENCODING` | `cl100k_base` | chỉ dùng khi không có tokenizer model thật |

> **Chi tiết đáng nhớ**: tiền tố `query:` / `passage:` không phải tuỳ chọn trang trí. Model
> E5 được huấn luyện với hai tiền tố này; bỏ đi thì chất lượng truy xuất tụt rõ rệt. Đây
> cũng là lý do `EmbeddingService` có hai hàm `encode_cau_hoi` và `encode_tai_lieu` riêng
> thay vì một hàm `encode` dùng chung.

### 9.5. Nhóm TRUY XUẤT

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `TOP_K` | `4` | số đoạn trích cuối cùng đưa vào prompt |
| `HE_SO_OVER_FETCH` | `10` | lấy dư `TOP_K × 10` ứng viên vì còn lọc nhiều tầng |
| `SO_UNG_VIEN_TOI_THIEU` | `60` | sàn số ứng viên |
| `TRONG_SO_BM25` | `0.0` | **BM25 mặc định không tham gia xếp hạng** (xem §2.5) |
| `SO_UNG_VIEN_BM25_CUU_HO` | `10` | nhưng vẫn bơm 10 ứng viên vào tập rerank |
| `RRF_K` | `60` | hằng số của Reciprocal Rank Fusion |
| `NGAN_SACH_KY_TU_MOI_DOAN` | `1600` | trần ký tự khi mở rộng một đoạn trích |
| `MO_RONG_QUA_RANH_GIOI_TRANG` | `True` | cho phép đoạn trích vượt sang trang liền kề |
| `SO_TRANG_TOI_DA_MO_RONG` | `1` | tối đa 1 trang mỗi hướng |
| `SO_DOAN_TOI_DA_MOI_TRANG` | `2` | trần số đoạn lấy từ cùng một trang |
| `SO_UNG_VIEN_XET_DA_DANG_TRANG` | `20` | xét 20 ứng viên đầu để quyết có áp trần không |
| `SO_DOAN_ANH_TOI_DA` | `1` | trần riêng cho đoạn là ảnh |

> **Vì sao có `SO_DOAN_ANH_TOI_DA`**: mô tả ảnh do model vision sinh ra khá dài, nên với
> tài liệu nhiều hình chúng dễ chiếm hết suất của các trang văn bản đúng. Đo thực tế:
> Recall@K tụt từ 0.96 xuống 0.92 khi chưa có trần này.

### 9.6. Nhóm RERANK & NGƯỠNG

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `BAT_RERANK` | `True` | |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | |
| `SO_UNG_VIEN_RERANK` | `30` | số ứng viên chấm cho câu hỏi phức tạp |
| `SO_UNG_VIEN_RERANK_DON_GIAN` | `12` | cho câu hỏi đơn giản (ngân sách thích ứng) |
| `NGUONG_DIEM_TOI_THIEU` | `0.5` (E5) | sàn cosine tuyệt đối — chỉ chặn rác |
| `TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT` | `0.78` | sàn **tương đối** trong cùng một lượt |
| `NGUONG_DIEM_RERANK_TOI_THIEU` | `0.001` | sàn rerank — **thứ thật sự chặn câu lạc đề** |
| `LOG_PHAN_BO_DIEM` | `False` | bật để in phân bố điểm ra log khi hiệu chỉnh ngưỡng |

> **Ba tầng sàn, ba vai trò khác nhau** — đây là một trong những thiết kế đáng học nhất
> của dự án:
> - Sàn **tuyệt đối** (cosine): giá trị cosine của E5 trôi theo domain/ngôn ngữ/độ dài
>   chunk, nên một hằng số hiệu chỉnh trên corpus này sẽ cắt oan trên corpus khác. Vì thế
>   nó bị hạ vai trò xuống "chỉ chặn rác".
> - Sàn **tương đối**: tỉ lệ giữa các đoạn *trong cùng một lượt* thì không trôi, vì cả lượt
>   dùng chung một câu hỏi và một model.
> - Sàn **rerank**: cross-encoder cho một thang đo hoàn toàn khác, và chính nó mới tách
>   được câu lạc đề (0.000–0.003) khỏi câu đúng chủ đề (≥ 0.019).

### 9.7. Nhóm LLM (Ollama)

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b` | |
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `OLLAMA_TEMPERATURE` | `0.1` | thấp = bám tài liệu, ít sáng tạo |
| `OLLAMA_NUM_PREDICT` | `12000` | trần **suy luận + câu trả lời** cộng lại |
| `OLLAMA_NUM_CTX` | `16384` | cửa sổ ngữ cảnh khởi điểm |
| `OLLAMA_NUM_CTX_TOI_DA` | `32768` | trần trên, chặn theo RAM máy |
| `OLLAMA_DU_PHONG_TOKEN_SINH` | `4000` | phần giữ chỗ cho việc sinh |
| `SO_KY_TU_MOI_TOKEN_UOC_LUONG` | `2.2` | tỉ lệ ký tự/token, **cố ý đặt thấp để ước lượng dư** |
| `BAT_THINKING_KHI_KIEM_CHUNG` | `True` | bật chế độ suy luận cho câu kiểm chứng |
| `SO_TRICH_DAN_HIEN_THI` | `3` | tối đa 3 nguồn hiển thị |
| `CAU_TU_CHOI` | `{"vi": "Không tìm thấy thông tin trong tài liệu.", "en": ...}` | |
| `JUDGE_MODEL` | `= OLLAMA_MODEL` | model chấm điểm ở tầng đánh giá |

> **`OLLAMA_NUM_CTX` là tham số nguy hiểm nhất trong file này.** Không truyền nó thì Ollama
> cấp mặc định 4096 token, và khi prompt vượt quá (prompt mặc định ~4900 token, một mình đã
> vượt) nó **cắt im lặng từ đầu phần user content** — tức xoá đúng đoạn trích `[1]`, đoạn
> liên quan nhất. Triệu chứng: câu trả lời tự nhiên ngắn đi, trích dẫn trỏ vào đoạn kém liên
> quan. Không có lỗi, không có cảnh báo. Toàn bộ cụm `_tinh_num_ctx` / `nen_ngu_canh` /
> `_ghi_nhan_thong_ke_llm` tồn tại vì bug này.

### 9.8. Nhóm ĐỌC TÀI LIỆU KHÓ (OCR, cột, dính chữ)

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `BAT_OCR_DU_PHONG` | `True` | OCR chỉ chạy khi trích text đã **thất bại** |
| `SO_CID_TOI_THIEU_DE_OCR` | `5` | ít nhất 5 mã `(cid:NN)` mới coi là font hỏng |
| `TY_LE_CID_DE_OCR` | `0.02` | hoặc ≥ 2% nội dung là mã cid |
| `SO_TU_TOI_THIEU_TRANG_CO_CHU` | `15` | dưới ngưỡng này coi như trang không có chữ |
| `OCR_NUM_PREDICT` | `1200` | trần sinh cho một lượt OCR |
| `DPI_RENDER_TRANG_OCR` | `150` | độ phân giải render trang để OCR |
| `SO_KY_TU_TOI_THIEU_MOT_TAI_LIEU` | `200` | dưới ngưỡng → cảnh báo "không đọc được" |
| `TY_LE_DIEN_TICH_ANH_TOAN_TRANG` | `0.6` | ảnh phủ ≥60% trang → nghi là trang scan |
| `BAT_DOC_LAI_TRANG_DINH_CHU` | `True` | tự dò lại `x_tolerance` khi chữ bị dính |
| `CAC_X_TOLERANCE_THU` | `[2.0, 1.5, 1.0, 0.7]` | các mức thử theo thứ tự |
| `MUC_TANG_TU_LE_CHAP_NHAN` | `0.03` | bản đọc lại phải tốt hơn ít nhất 3% mới nhận |
| `TY_LE_DINH_CHU_DE_DOC_LAI` | `0.1` | ≥10% chữ nằm trong cụm dính → đọc lại |
| `DO_DAI_CUM_DINH_CHU` | `25` | cụm ≥25 ký tự không khoảng trắng là bất thường |
| `SO_KY_TU_TOI_THIEU_DE_DO` | `200` | quá ít chữ thì không kết luận gì |
| `SO_TRANG_HIEU_CHINH_X_TOLERANCE` | `3` | sau 3 trang cùng một mức thì chốt |
| `TY_LE_DINH_CHU_DAT_YEU_CAU` | `0.02` | ≤2% là đạt, dừng dò sớm |
| `BAT_DOC_THEO_COT` | `True` | phát hiện và đọc trang 2 cột theo đúng thứ tự đọc |
| `SO_O_DO_COT` | `60` | số ô lưới dùng để dò rãnh giữa hai cột |
| `SO_O_RANH_TOI_THIEU` | `3` | rãnh phải rộng ít nhất 3 ô |
| `TY_LE_TU_MOI_COT` | `0.25` | mỗi cột phải chứa ≥25% số từ |
| `SO_TU_TOI_THIEU_DE_DO_COT` | `60` | trang quá ngắn thì không dò cột |

### 9.9. Nhóm ẢNH & VISION

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `BAT_TRICH_ANH` | `True` | |
| `BAT_CHU_THICH_ANH` | `True` | gọi model vision mô tả ảnh |
| `VISION_MODEL_NAME` | `qwen2.5vl:3b` | |
| `VISION_NUM_PREDICT` | `400` | trần độ dài mô tả |
| `TY_LE_DIEN_TICH_ANH_TOI_THIEU` | `0.015` | ảnh nhỏ hơn 1.5% diện tích trang → bỏ |
| `TY_LE_CANH_ANH_TRANG_TRI` | `12.0` | tỉ lệ cạnh > 12 → đường kẻ trang trí, bỏ |
| `SO_LAN_LAP_COI_LA_LOGO` | `4` | ảnh trùng nội dung lặp ≥4 lần → logo, bỏ |

### 9.10. Nhóm STREAMING & GIAO DIỆN

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `BAT_STREAMING` | `True` | |
| `GIAN_CACH_VE_LAI_GIAY` | `0.12` | vẽ lại tối đa ~8 lần/giây, không theo từng token |
| `SO_KY_TU_SUY_LUAN_HIEN` | `500` | chỉ hiện 500 ký tự cuối của chuỗi suy luận |
| `NGUONG_BAM_NGUON_HIEN_THI` | `0.3` | chỉ hiện chỉ số bám nguồn khi ≥ 30% |

### 9.11. Nhóm HỘI THOẠI NỐI TIẾP

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `BAT_TRUY_VAN_NGU_CANH` | `True` | đường **tất định**: ghép câu hỏi trước vào truy vấn |
| `BAT_VIET_LAI_CAU_HOI` | `False` | đường **LLM**: nhờ model viết lại — mặc định TẮT |
| `NUM_PREDICT_VIET_LAI` | `200` | |
| `SO_LUOT_NGU_CANH` | `3` | lấy tối đa 3 câu hỏi gần nhất |
| `DO_DAI_TRA_LOI_TRONG_NGU_CANH` | `300` | cắt câu trả lời cũ khi đưa vào prompt viết lại |
| `SO_TU_TOI_DA_CAU_VIET_LAI` | `60` | bản viết lại dài hơn thì bị loại |
| `TRONG_SO_TRUY_VAN_GOC` | `1.0` | câu gốc luôn là một nhánh RRF riêng |

> **Vì sao đường LLM mặc định tắt**: đường tất định đã đạt 16/16 trên bộ ca kiểm định mà
> không tốn một lượt gọi model nào. Thêm một lượt gọi LLM vào **đường nóng** của mỗi câu hỏi
> là cái giá độ trễ rất thật, đổi lại một cải thiện không đo được. Đây là một quyết định
> "không thêm tính năng" đáng chú ý.

### 9.12. Nhóm ĐỐI CHIẾU NGUỒN

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `BAT_DOI_CHIEU_NGUON` | `True` | |
| `NGUONG_COSINE_DOI_CHIEU` | `0.88` | hai đoạn phải rất gần nhau mới coi là "cùng chủ đề" |
| `SO_CAP_DOI_CHIEU_TOI_DA` | `3` | trần số cặp đem đi chấm — chặn bùng nổ chi phí |
| `SO_LAN_CHAM_MAU_THUAN` | `2` | chấm 2 lần, chỉ báo khi cả 2 lần đồng ý |
| `NGUONG_MAU_THUAN` | `0.6` | mức độ tối thiểu để báo |

### 9.13. Nhóm CACHE & TĂNG DẦN

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `BAT_CACHE_INGESTION` | `True` | |
| `BAT_INDEX_TANG_DAN` | `True` | |
| `BAT_PROFILING_INGESTION` | `True` | in bảng thời gian sau mỗi lần build |
| `SO_WORKER_VISION` | tự suy, tối đa 2 | số luồng gọi vision/OCR song song |
| `SO_WORKER_DOC` | `1` | số luồng đọc tài liệu |

### 9.14. Nhóm NGÂN SÁCH THÍCH ỨNG & PHẦN CỨNG

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `BAT_NGAN_SACH_THICH_UNG` | `True` | câu hỏi đơn giản dùng ít ứng viên rerank hơn |
| `SO_TU_CAU_HOI_DON_GIAN` | `12` | dài hơn 12 từ → coi là phức tạp |
| `BAT_NEN_NGU_CANH` | `True` | ép đoạn trích vào ngân sách token |
| `THIET_BI_EMBEDDING` | `auto` | `auto` / `cuda` / `cpu` |
| `THIET_BI_RERANK` | `auto` | |
| `BAT_QUAN_LY_VRAM` | `True` | |
| `NHA_MODEL_SAU_INGESTION` | `True` | nhả model vision khỏi VRAM ngay khi build xong |
| `VRAM_DU_CHO_LO_LON_GB` | `3.0` | ngưỡng VRAM để dùng batch lớn |
| `VRAM_DU_CHO_LO_VUA_GB` | `2.0` | |
| `VRAM_MOI_WORKER_VISION_GB` | `1.5` | mỗi worker vision cần ~1.5 GB |
| `VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB` | `10.0` | dưới 10 GB thì đẩy embedding xuống CPU lúc query |

### 9.15. Nhóm ĐÁNH GIÁ

| Hằng số | Mặc định | Ý nghĩa |
|---|---|---|
| `NGUONG_DIEM_JUDGE_THAP` | `0.5` | dưới ngưỡng này coi là "giám khảo chấm thấp" |
| `NGUONG_BAM_NGU_CANH_DE_NGHI_NGO` | `0.3` | nếu bám ngữ cảnh cao mà judge chấm thấp → cờ nghi ngờ |
| `SO_LAN_CHAM_FAITHFULNESS` | `3` | chấm 3 lần lấy trung vị |

---

## 10. `app.py` — tầng giao diện (842 dòng)

### 10.1. Vai trò và mô hình chạy của Streamlit

`app.py` là **toàn bộ** giao diện. Không có backend riêng — Streamlit gọi thẳng các hàm
trong `rag/`.

Điều bắt buộc phải hiểu về Streamlit: **mỗi thao tác của người dùng khiến toàn bộ script
chạy lại từ dòng 1**. Không có event handler, không có state tự nhiên. Trạng thái duy nhất
sống sót qua các lần chạy là `st.session_state` (một dict) và các hàm được đánh dấu
`@st.cache_resource` / `@st.cache_data`.

Hệ quả trong code này:

- Model embedding và reranker được bọc trong `@st.cache_resource` để **chỉ nạp một lần** cho
  cả phiên (nạp lại mỗi lần render là vài giây × mỗi cú bấm).
- Trạng thái `dang_xu_ly` tồn tại để vô hiệu hoá mọi widget trong lúc gọi LLM.
- Việc nhận câu hỏi và việc gọi LLM bị **tách sang hai lần chạy khác nhau** — xem
  `_dat_cau_hoi`.

### 10.2. Bố cục màn hình

```
┌──────────────────────┬──────────────────────────────────────────────┐
│ THANH BÊN            │ KHUNG CHÍNH (cột hẹp 48rem, căn giữa)        │
│                      │                                              │
│ ＋ Hội thoại mới     │  [lời chào + 3 nút gợi ý khi chưa hỏi gì]     │
│ ──────────────────   │                                              │
│ Nguồn tài liệu       │  🧑 câu hỏi                     (bong bóng)   │
│  [upload]            │  📚 câu trả lời                 (không bọc)   │
│  ☑ file1.pdf   🗑     │     📎 Nguồn: file.pdf — trang 7             │
│  ☑ file2.docx  🗑     │     ✓ 42% trùng nguyên văn                   │
│ ──────────────────   │     ⚠️ Hai nguồn nói khác nhau…              │
│ [Đọc tài liệu]       │                                              │
│ 📊 9285 chunk        │                                              │
│ ⚡ GPU: rerank=cuda   │  ┌────────────────────────────────────────┐  │
│ 💾 Cache 320MB [Xoá] │  │ Hỏi về tài liệu của bạn...             │  │
│ ⚠️ Ollama chưa chạy   │  └────────────────────────────────────────┘  │
└──────────────────────┴──────────────────────────────────────────────┘
```

Quyết định bố cục đáng chú ý: **thanh bên là danh sách TÀI LIỆU, không phải danh sách hội
thoại cũ**. Đây chính là khác biệt cốt lõi giữa một trợ lý RAG và một chatbot thường — thứ
người dùng cần quản lý là *tài liệu nào đang được dùng để trả lời*.

### 10.3. Các hàm

```python
@st.cache_resource
def lay_embedding_service() -> EmbeddingService
```
Tạo (và giữ lại cho cả phiên) `EmbeddingService`.

```python
@st.cache_resource
def lay_reranker_service()
```
Tương tự cho reranker. Trả `None` khi `BAT_RERANK=0` — lúc đó model ~2.2 GB **không được nạp
chút nào**.

```python
@st.cache_data(ttl=10, show_spinner=False)
def lay_loi_may_chu_llm()
```
Trạng thái Ollama, cache 10 giây. Dùng `cache_data` với TTL ngắn chứ không phải
`cache_resource` là có lý do: cache vĩnh viễn sẽ giữ nguyên cảnh báo cũ sau khi người dùng
đã bật Ollama lên — đúng lúc họ cần thấy nó biến mất để biết mình đã sửa xong.

```python
def tai_index_da_co()
```
Nạp lại FAISS index đã build từ lần trước, nếu file tồn tại. Trả `None` nếu chưa có.

```python
def lay_pipeline(embedding_service, store) -> RagPipeline
```
Giữ lại **một** `RagPipeline` cho mỗi `VectorStore` đang dùng. Không dùng
`@st.cache_resource` được vì `VectorStore` không hash được (và cũng không nên hash — nó thay
đổi tại chỗ khi xoá tài liệu). So sánh bằng **identity** (`is not`) là đủ và đúng: chỉ khi
store bị *thay thế* mới cần pipeline mới.

```python
def _store_dung_lai_duoc(store) -> bool
```
Index đang có trong phiên có dùng lại được cho một lần build **tăng dần** không? Ba điều
kiện: bật `BAT_INDEX_TANG_DAN`, store khác `None` và có vector, và
`ly_do_khong_tuong_thich()` trả `None`. Cả ba nhằm tránh đúng một kiểu hỏng: một index
**trộn hai thế hệ dữ liệu** — vector của tài liệu cũ và mới nằm ở hai không gian khác nhau
mà FAISS vẫn cứ so sánh với nhau.

```python
def xay_dung_lai_index(embedding_service, store_dang_dung=None)
```
**Hàm quan trọng nhất của file.** Chạy trọn luồng Ingestion. Các bước:

1. `do_thoi_gian.dat_lai()` — bấm lại đồng hồ đo.
2. `tai_nguyen_gpu.bat_dau_ingestion()` — đưa embedding trở lại GPU nếu lần trước đã đẩy
   nó xuống CPU.
3. Liệt kê file trong `data/raw/`, **băm nội dung** từng file.
4. Nếu dùng lại được index cũ: gọi `so_sanh_bam_tai_lieu()` → ba danh sách
   `(cần đọc lại, cần xoá, giữ nguyên)`. Gỡ vector của file đã biến mất; xoá sạch vector cũ
   của file đã đổi **trước khi** thêm bản mới.
   Ngược lại: tạo `VectorStore` mới, đọc lại tất cả.
5. `doc_nhieu_file(can_doc)` → danh sách "trang".
6. `chia_chunk(...)` — truyền **đúng tokenizer và giới hạn độ dài của model embedding**, chứ
   không dùng bộ đếm xấp xỉ.
7. `bo_nho_dem.encode_co_cache(...)` → vector.
8. `store.them(vectors, cac_chunk)`.
9. Ghi băm vào sổ — **chỉ cho tài liệu thật sự có nội dung vào index**. Một file đọc hỏng mà
   vẫn được ghi là "đã xử lý" sẽ bị bỏ qua ở mọi lần build sau, tức lỗi im lặng vĩnh viễn.
10. `store.luu()`.
11. `tai_nguyen_gpu.ket_thuc_ingestion()` — **ranh giới giai đoạn**: nhả model vision khỏi VRAM.
12. In bảng profiling.

```python
def _hien_thi_trich_dan(trich_dan: list) -> None
```
Vẽ dòng nguồn dưới câu trả lời. Hai chế độ: bình thường (`📎 Nguồn: ...`) và **phỏng đoán**
(`⚠️ Câu trả lời không tự dẫn nguồn — đây là chỗ liên quan nhất do hệ thống chọn, KHÔNG
chắc là căn cứ đã dùng`). Hàm con `_nhan_trang()` ghi **đủ khoảng trang** mà đoạn trích đã
đọc (`trang/slide 6–7`), không chỉ trang neo.

Chi tiết thiết kế đáng bàn: hàm **cố ý không in lại nguyên văn đoạn đã dùng**, chỉ dẫn ra
toạ độ. Lý do: đoạn in kèm là bản cắt ngắn 600 ký tự và mất định dạng gốc, nên vừa chiếm
chỗ vừa dễ khiến người ta dừng lại ở đó thay vì mở tài liệu thật. Đoạn trích vẫn được tính
và trả về trong dữ liệu (tầng đánh giá dùng nó để chấm Citation accuracy).

```python
def _hien_thi_bam_nguon(bam_nguon) -> None
```
Hiện `✓ 42% nội dung câu trả lời trùng nguyên văn với đoạn trích đã dẫn` — **chỉ khi mức bám
cao** (≥ `NGUONG_BAM_NGUON_HIEN_THI`).

Lý lẽ đằng sau rất chặt và đáng học: phép đo này chỉ nói được **một chiều**. Mức cao là bằng
chứng mạnh rằng câu trả lời không bịa; mức thấp **không chứng minh được gì** — một câu trả
lời diễn đạt lại bằng lời của mình, điều hoàn toàn hợp lệ, cũng cho mức thấp. Hiện dấu hiệu
"bám nguồn thấp" sẽ khiến người đọc nghi ngờ oan đúng những câu trả lời viết tốt nhất.
**Nói khi có bằng chứng, im lặng khi không có.**

```python
def _hien_thi_cach_hieu(truy_van: dict) -> None
```
Nói ra việc hệ thống đã hiểu câu hỏi này là **nối tiếp** một câu trước đó. *(Lưu ý: trong
bản hiện tại phần thân đang bị comment lại, nên hàm không vẽ gì. Docstring vẫn giữ nguyên
lập luận — đây là chỗ nên xem lại khi hoàn thiện.)*

```python
def _hien_thi_mau_thuan(cac_mau_thuan: list) -> None
```
Vẽ cảnh báo `⚠️ Hai nguồn nói khác nhau` kèm toạ độ hai chỗ. **Cố ý không kết luận nguồn nào
đúng**: hệ thống không biết tài liệu nào mới hơn, môn học nào ưu tiên bản nào.

```python
def _chay_va_hien_theo_luong(pipeline, cau_hoi, nguon_cho_phep, lich_su) -> dict
```
Chạy `pipeline.hoi_dap_theo_luong()` và vẽ dần. Ba vùng hiển thị:
1. **Khung trạng thái** — `"Đang truy xuất..."` → `"Đã tìm 4 đoạn liên quan trong 2 tài liệu
   (2.1s) — đang soạn câu trả lời"`, kèm số giây đang trôi (để phân biệt "đang chạy" với
   "đã treo").
2. **Phần suy luận** — 500 ký tự cuối của chuỗi suy luận nội bộ, nằm thu gọn trong khung
   trạng thái.
3. **Câu trả lời** — chữ chạy dần, có con trỏ `▌` nhấp nháy.

Hàm con `_ve_lai(ep_buoc=False)` giãn cách việc vẽ lại theo `GIAN_CACH_VE_LAI_GIAY`: mỗi lần
vẽ Streamlit đẩy lại cả khối markdown qua websocket, làm theo từng token thì trình duyệt
nghẽn mà mắt người cũng không đọc kịp.

```python
def _dat_cau_hoi(cau_hoi: str) -> None
```
Nhận câu hỏi, đẩy vào `messages`, đặt cờ, `st.rerun()` **ngay** — không gọi LLM tại đây. Xem
§6 bước 0 để biết lý do. Dùng chung cho cả ô nhập lẫn các nút gợi ý, để hai lối vào đi đúng
một đường mã.

### 10.4. Phần thân script (chạy tuần tự mỗi lần rerun)

1. Khởi tạo `session_state` (7 khoá: `vector_store`, `messages`, `uploader_key_n`,
   `dang_xu_ly`, `cau_hoi_dang_xu_ly`, `pipeline`, `pipeline_cho_store`).
2. **Thanh bên**: nút hội thoại mới → uploader → danh sách file có checkbox + nút xoá →
   nút "Đọc tài liệu" → thông tin index/GPU/cache → cảnh báo Ollama.
3. **Khung chính**: nếu chưa có index thì hiện màn hình 3 bước rồi `st.stop()`; nếu chưa
   hỏi gì thì hiện lời chào + 3 nút gợi ý; rồi vẽ lịch sử chat.
4. **Nếu `dang_xu_ly`**: gọi pipeline, vẽ câu trả lời, lưu vào `messages`, hạ cờ, `st.rerun()`.
5. **Ô nhập** — đặt ở tầng ngoài cùng của script (không nằm trong container nào), đó là điều
   kiện để Streamlit ghim nó xuống đáy màn hình.

### 10.5. Ba bug Streamlit đã sửa, ghi lại trong code

Đây là phần thực dụng nhất của file, đáng đọc kỹ nếu bạn định làm app Streamlit:

1. **`uploader_key_n`** — `st.file_uploader()` trả về **cùng một danh sách file trên MỌI lần
   rerun** cho tới khi người dùng tự bấm "x". Hệ quả: file cũ bị ghi đè lại vào `data/raw/`
   ngay sau khi vừa xoá xong ("xoá không được"). Cách sửa: đổi `key` của widget sau mỗi lần
   upload thành công để Streamlit coi đó là một widget **mới**, rỗng.

2. **Tách nhận câu hỏi khỏi gọi LLM** — nếu gọi LLM ngay trong lần chạy nhận câu hỏi, một cú
   bấm bất kỳ sẽ khiến Streamlit huỷ ngang lần chạy đang ở giữa lệnh gọi Ollama.

3. **`try/except Exception` bao cả khối sinh câu trả lời** — bất kỳ lỗi nào thoát ra mà không
   được xử lý đều để lại `dang_xu_ly=True`, tức ô nhập và **mọi** nút vẫn disabled: người
   dùng không thao tác được gì nữa cho tới khi tự tải lại trang. Lỗi được ghi thành một **tin
   nhắn trong lịch sử** chứ không chỉ `st.error()` — vì `st.error` biến mất ngay sau
   `st.rerun()` bên dưới.
---

## 11. Nhóm INGESTION

Sáu file, chạy theo thứ tự: `bo_nho_dem` (tra cache) → `document_loader` (đọc) →
`image_extractor` + `vision_caption` (ảnh) → `chunking` (cắt) → `bo_nho_dem` (cache vector).
`do_thoi_gian` bám theo suốt để đo.

---

### 11.1. `rag/document_loader.py` — đọc tài liệu (1.262 dòng)

**Vai trò**: biến một file PDF/PPTX/DOCX thành `List[{nguon, trang, noidung}]`, giữ metadata
ngay từ bước đọc. Đây là file dài nhất và phức tạp nhất của luồng Ingestion, vì thực tế tài
liệu học tập rất bẩn: PDF scan, font hỏng, hai cột, bảng kẻ khung, watermark, mục lục.

**Hằng số**

| Tên | Giá trị | Ý nghĩa |
|---|---|---|
| `CAC_DUOI_HO_TRO` | `('.pdf', '.pptx', '.docx')` | |
| `MOC_BANG_MO` / `MOC_BANG_DONG` | `"[BẢNG]"` / `"[/BẢNG]"` | bọc khối bảng để chunking nhận ra |
| `_MAU_WATERMARK_STUDOCU` | regex | bắt watermark `lOMoARcPSD|...` và `Downloaded by ...` |
| `_MAU_CID_PDF` | regex `\(cid:\d+\)` | dấu hiệu font PDF thiếu bảng ánh xạ |
| `_NGUONG_TY_LE_MUC_LUC` | `0.5` | ≥50% dòng kết thúc bằng số → trang mục lục |

#### Nhóm A — Phát hiện và sửa lỗi "dính chữ"

Vấn đề: một số PDF khi trích text ra bị nuốt khoảng trắng
(`"Thebias-variancedecompositionbreaks..."`). Nguyên nhân là tham số `x_tolerance` của
`pdfplumber` không hợp với font của tài liệu đó. Hệ quả rất nặng: đoạn trích lấy ra không
đối chiếu được, nên giám khảo Faithfulness chấm 0 cho những câu trả lời hoàn toàn đúng (đo
được: nhóm sách tiếng Anh 0.33 → 0.83 sau khi sửa).

```python
def _ty_le_dinh_chu(text: str) -> float
```
Tỉ lệ ký tự chữ nằm trong những cụm dài bất thường (≥ `DO_DAI_CUM_DINH_CHU` = 25 ký tự không
có khoảng trắng). Cao = nghi dính chữ.

```python
def _ty_le_tu_le(text: str) -> float
```
Tỉ lệ "từ" chỉ có **đúng một** chữ cái — dấu hiệu **ngược lại**: văn bản bị vỡ tung ra
(`"T h e   b i a s"`). Có cả hai chỉ số mới bắt được hai kiểu hỏng đối xứng nhau.

```python
def _trich_text(doi_tuong, x_tolerance=None) -> str
```
Gọi `extract_text()` có/không tham số `x_tolerance`, gom vào một chỗ để hai nhánh dùng chung.

```python
class HieuChinhXTolerance
    def __init__(self)
    @property def da_hieu_chinh(self) -> bool
    def thu_tu_uu_tien(self) -> List[float]
    def ghi_nhan(self, x_tolerance: float) -> None
```
**Nhớ** mức `x_tolerance` đã dò được cho **một tài liệu**, để các trang sau không phải dò
lại. Sau `SO_TRANG_HIEU_CHINH_X_TOLERANCE = 3` trang cùng cho một kết quả thì coi là ổn định
và chốt luôn. Đây là tối ưu quan trọng: không có nó, mỗi trang bị đọc tới 5 lần.

```python
def _trich_text_thich_ung(doi_tuong, ten_file="", so_trang=None, hieu_chinh=None) -> str
```
Đọc text một trang, **tự dò lại tham số** khi phát hiện chữ dính. Quy tắc an toàn quan
trọng: bản đọc lại chỉ được chấp nhận nếu nó **tốt hơn ít nhất `MUC_TANG_TU_LE_CHAP_NHAN`
= 3%**. Nhờ vậy một lần báo động giả vẫn vô hại — nếu không có gì để cải thiện thì bản gốc
được giữ.

#### Nhóm B — Dọn dẹp nội dung

```python
def _chuan_hoa_nfc(text: str) -> str
```
Chuẩn hoá Unicode về dạng NFC. Cần thiết với tiếng Việt: `"ế"` có thể được mã hoá thành một
ký tự hoặc thành `"ê" + dấu sắc`, và hai dạng đó không so khớp được với nhau.

```python
def _don_dep_watermark(text: str) -> str
```
Bỏ watermark StuDocu và các mã `(cid:NN)` không giải mã được.

```python
def _la_trang_muc_luc(text: str) -> bool
```
Đoán một trang có phải Mục lục không (hoặc là trang **tiếp theo** của mục lục nhiều trang).
Căn cứ: có tiêu đề "mục lục"/"table of contents", hoặc ≥50% số dòng kết thúc bằng số trang.
Trang mục lục bị **loại khỏi index** vì nó chỉ là danh sách tiêu đề — nội dung rất dễ khớp
câu hỏi nhưng không chứa câu trả lời, tức nhiễu thuần tuý.

#### Nhóm C — Nhận diện tiêu đề

```python
def _danh_dau_tieu_de(text: str, cap: int = 2) -> str
```
Bọc một dòng thành `"## <dòng>"` để splitter ưu tiên cắt tại đó
(`CHUNK_SEPARATORS` đặt `'\n## '` lên đầu).

```python
def _phat_hien_tieu_de_pdf(trang) -> set
```
Đoán dòng nào là tiêu đề dựa vào **cỡ chữ** (lớn hơn trung bình
`TY_LE_KICH_THUOC_CHU_TIEU_DE` = 1.15 lần) **và** độ dài dòng (≤ 90 ký tự). Cần cả hai điều
kiện: chữ to mà dòng dài thì chỉ là văn bản cỡ lớn.

#### Nhóm D — Bảng

```python
def _o_bang_khong_lap(hang) -> List[str]
```
Lấy text các ô của một hàng, **khử phần nhân bản do ô gộp** (merged cell). `pdfplumber` trả
về nội dung ô gộp lặp lại ở mọi cột nó chiếm.

```python
def _bang_sang_markdown(bang: List[List]) -> str
```
Đổi bảng (list hàng × ô) thành bảng Markdown. Bù ô thiếu, bỏ hàng rỗng.

```python
def _la_bang_that(bang: List[List]) -> bool
```
Lọc "bảng" do thuật toán dò nhầm: phải có ít nhất 2 hàng và 2 cột.

```python
def _khoi_bang(bang: List[List]) -> str
```
Bọc bảng trong `[BẢNG] ... [/BẢNG]` để `chunking` nhận ra và **giữ nguyên khối**.

#### Nhóm E — Đọc theo cột

```python
def _cac_cot_cua_trang(trang) -> List[tuple]
```
Trả về danh sách khoảng `(x_trái, x_phải)` của từng cột; rỗng nếu trang một cột. Cách làm:
chia trang thành `SO_O_DO_COT` = 60 ô theo chiều ngang, tìm "rãnh" rộng ≥ 3 ô không có chữ,
và yêu cầu mỗi cột chứa ≥ 25% số từ.

```python
def _text_theo_cot(trang, cac_cot, ten_file, so_trang, hieu_chinh=None) -> str
```
Đọc từng cột riêng rồi nối lại trái → phải, tức **đúng thứ tự đọc của người**. Không có bước
này thì trang hai cột bị đọc theo dòng ngang, trộn câu của hai cột vào nhau — nội dung trở
thành vô nghĩa.

```python
def _text_pdf_khong_ke_bang(trang, cac_bang, ten_file="", hieu_chinh=None) -> str
```
Lấy văn xuôi của trang **đã loại vùng chiếm bởi bảng**, để nội dung bảng không xuất hiện hai
lần.

```python
def _doc_mot_trang_pdf(trang, ten_file, so_trang, hieu_chinh) -> str
```
Gói toàn bộ việc đọc text của một trang thành **đúng một lượt**: dò cột → dò bảng → lấy văn
xuôi không kể bảng → chèn khối bảng vào đúng chỗ.

#### Nhóm F — OCR

```python
def _lay_client_vision()
```
Tạo Ollama client dùng chung cho OCR, chỉ một lần cho cả lần build.

```python
def _ocr_cac_trang(pdf, cac_so_trang, ten_file, bam_tai_lieu) -> Dict[int, str]
```
OCR **một loạt** trang cùng lúc. Tra cache trước (`kho_ocr`, khoá = băm tài liệu + số
trang), phần còn lại render thành ảnh ở `DPI_RENDER_TRANG_OCR` = 150 rồi gọi model vision
song song.

#### Nhóm G — Ba hàm đọc chính

```python
def doc_pdf(duong_dan: Path) -> List[Dict]
```
**Luồng MỘT-LƯỢT-DUYỆT, chia 4 pha** — đây là thiết kế cốt lõi của file:

| Pha | Việc | Vì sao ở đây |
|---|---|---|
| **1** (tuần tự, một lượt) | đọc text, dò bảng/cột, nhận diện tiêu đề, **liệt kê** ứng viên ảnh (chưa render), đánh dấu trang cần OCR | sau mỗi trang gọi `flush_cache()` để nhả RAM — sách 700 trang không cần giữ hết đối tượng của mọi trang |
| **2** | OCR các trang đã đánh dấu — tra cache trước, phần còn lại gọi model **song song** | gom lại mới song song hoá được |
| **3** | gộp kết quả OCR, dọn watermark, chuẩn hoá NFC, loại trang rỗng / trang mục lục | |
| **4** | render và lưu những ảnh được giữ lại, rồi loại hình lặp kiểu logo | chỉ trang thật sự có ảnh mới phải đọc lại |

Quy tắc tinh tế nhất được giữ nguyên qua thiết kế này: **ảnh phủ kín một trang chỉ bị loại
khi OCR đã CHỨNG MINH trang đó là ảnh chụp một trang chữ**, chứ không phải khi đoán qua kích
thước. Tín hiệu đo được thay cho phỏng đoán.

```python
def doc_pptx(duong_dan: Path) -> List[Dict]
def duyet_shape(cac_shape) -> Iterator
def _trich_text_shape(shape) -> List[str]
```
Đọc PPTX. `duyet_shape` **đệ quy vào group shape** — nếu không thì mọi nội dung nằm trong
nhóm bị mất, và trong slide bài giảng thì nhóm rất phổ biến.

```python
def doc_docx(duong_dan: Path) -> List[Dict]
def _text_trong_text_box(doan_van) -> str
```
Đọc DOCX, tách thành "trang" theo **dấu ngắt trang cứng**. `_text_trong_text_box` đào thẳng
vào XML (`python-docx` không nhìn thấy chữ nằm trong text box).

#### Nhóm H — Điều phối và cache

```python
def doc_tai_lieu(duong_dan: Path) -> List[Dict]
```
Chọn hàm đọc theo phần đuôi file. Ném `ValueError` nếu định dạng không hỗ trợ.

```python
def doc_tai_lieu_hoan_chinh(duong_dan: Path) -> List[Dict]
```
Đọc + chú thích ảnh + bỏ ảnh rỗng. **Đây là ĐƠN VỊ ĐƯỢC CACHE**, và ranh giới đó phải nằm
đúng ở đây: nếu chỉ cache kết quả đọc thô rồi vẫn chạy lại bước chú thích vision mỗi lần,
ta sẽ cache đúng phần rẻ và bỏ qua phần đắt (~1,9 giây mỗi hình).

```python
def _cache_con_du_anh(cac_trang: List[Dict]) -> bool
```
Mọi file ảnh mà bản ghi trong cache trỏ tới còn tồn tại không? Cần kiểm vì nội dung nằm ở
`data/cache/` còn file ảnh nằm ở `data/images/` — xoá `images/` mà giữ `cache/` sẽ cho một
index "hợp lệ" với trích dẫn trỏ vào ảnh không còn tồn tại.

```python
def doc_tai_lieu_co_cache(duong_dan: Path) -> List[Dict]
```
Bản có cache của hàm trên. Khoá = **băm nội dung file + vân tay cấu hình đọc**.

```python
def _bo_ban_ghi_anh_rong(cac_trang: List[Dict]) -> List[Dict]
```
Bỏ những bản ghi ảnh rốt cuộc không có chú thích nào (nội dung chỉ còn `"[HÌNH]"`). Sinh ra
từ một lỗi thật: nạp giáo trình scan 383 trang khi Ollama đang tắt → index gồm đúng 379
chunk `"[HÌNH]"` giống hệt nhau, hệ thống báo "build thành công", trả lời sai mọi câu hỏi,
**không một exception nào được ném ra**.

```python
def _canh_bao_tai_lieu_khong_doc_duoc(cac_trang, cac_duong_dan) -> None
```
Ghi log `ERROR` khi một tài liệu có dưới 200 ký tự văn bản thật (không tính bản ghi ảnh),
kèm hướng dẫn bật OCR.

```python
def cac_file_tai_lieu(thu_muc: Path) -> List[Path]
def doc_nhieu_file(cac_duong_dan: List[Path]) -> List[Dict]
def doc_thu_muc(thu_muc: Path) -> List[Dict]
```
`doc_nhieu_file` nhận **danh sách file cụ thể** (không phải thư mục) — đó là điều kiện để
build tăng dần hoạt động. Nó bắt `Exception` rộng quanh từng file: **một file hỏng không
được làm sập cả lần build**, nhưng lỗi được ghi rõ tên file + loại lỗi và tổng kết lại ở
cuối.

---

### 11.2. `rag/image_extractor.py` — trích ảnh (396 dòng)

**Vai trò**: lấy hình ra khỏi tài liệu, **lọc bỏ rác** (logo, đường kẻ, ảnh chụp trang chữ),
và gắn chú thích văn bản lân cận.

**Hằng số**: `MOC_ANH = "[HÌNH]"`, `KICH_THUOC_ANH_TOI_THIEU = 120` (px),
`DO_DAI_CHU_THICH_LAN_CAN = 400` (ký tự).

```python
def _ten_file_an_toan(nguon, trang, thu_tu, duoi) -> str
```
Tên file ảnh suy ra từ `(nguồn, trang, thứ tự)` nên **ổn định giữa các lần build** — build
lại không sinh ra file trùng lặp.

```python
def _chon_chu_thich(cac_dong: List[str]) -> str
```
Chọn văn bản mô tả hình từ các dòng lân cận. Ưu tiên dòng khớp mẫu
`"Hình 3.2:"`, `"Figure 5."`, `"Biểu đồ 1 -"`…

```python
def _la_anh_cua_trang_chu(rong, cao, dien_tich_trang) -> bool
def ly_do_loai_anh(rong, cao, dien_tich_trang=0.0) -> Optional[str]
def ly_do_loai_anh_blob(du_lieu: bytes) -> Optional[str]
```
Ba hàm lọc. `ly_do_loai_anh` trả về **lý do loại** (chuỗi) hoặc `None` nếu giữ — thiết kế
này tốt hơn trả `bool` vì lý do được ghi thẳng vào log. Bốn tiêu chí loại:
kích thước < 120px; diện tích < 1.5% trang; tỉ lệ cạnh > 12 (đường kẻ trang trí); phủ ≥ 60%
trang **và** OCR đã chứng minh đó là trang chữ.

```python
def loc_anh_lap_lai(cac_ban_ghi: List[Dict], nguon: str) -> List[Dict]
```
Loại ảnh có **nội dung giống hệt** lặp ≥ `SO_LAN_LAP_COI_LA_LOGO` = 4 lần trong cùng tài
liệu — logo trường, watermark, khung slide.

```python
def _ban_ghi_anh(nguon, trang, duong_dan_anh, chu_thich, bam_anh="") -> Dict
```
Đóng gói một ảnh thành một "trang" (xem §7.1).

```python
def ung_vien_anh_trang(trang) -> List[Tuple[bbox, bool]]
```
Liệt kê ảnh **đáng giữ** trên một trang PDF — **chưa render gì cả**. Việc tách "liệt kê" khỏi
"render" là điều cho phép `doc_pdf` chạy một lượt duy nhất.

```python
def luu_anh_trang_pdf(nguon, trang, so_trang, cac_bbox, cac_dong_text) -> List[Dict]
def trich_anh_pptx(duong_dan, trinh_chieu) -> List[Dict]
def trich_anh_docx(duong_dan, document) -> List[Dict]
```
Ba hàm render/trích thật sự cho ba định dạng. PPTX đi qua `duyet_shape` để vào cả group
shape; DOCX đi qua quan hệ (rels) của phần thân tài liệu.

---

### 11.3. `rag/vision_caption.py` — model vision (264 dòng)

**Vai trò**: hai việc khác nhau dùng chung một model —
(1) **mô tả nội dung hình** để hình cũng tra cứu được;
(2) **OCR dự phòng** khi trích text thất bại.

**Hai prompt**

- `PROMPT_CHU_THICH_VI` — mô tả hình bằng tiếng Việt, yêu cầu **ghi lại toàn bộ chữ xuất
  hiện trong hình** (nhãn, tiêu đề, số liệu).
- `PROMPT_OCR_TRANG` — viết bằng tiếng Anh, yêu cầu chép lại **nguyên văn, đúng thứ tự**, và
  đặc biệt: **"Keep the ORIGINAL LANGUAGE of the page. Do NOT translate."** (có hẳn một test
  tên `test_prompt_ocr_cam_dich` canh chừng điều này).

```python
def trang_can_ocr(text_da_doc: str, so_anh_trong_trang: int) -> bool
```
**Quyết định then chốt của cả cơ chế OCR.** Trả `True` khi: có ≥ 5 mã `(cid:NN)` hoặc ≥ 2%
nội dung là mã cid (font hỏng); **hoặc** trang có ảnh mà số từ đọc được < 15 (trang scan).
OCR chỉ chạy khi trích text đã **thất bại** — không phải chạy mặc định cho mọi trang.

```python
def ocr_trang_pdf(client, duong_dan_anh, ten_model=None) -> str
```
Đọc lại một trang đã render thành ảnh. Trả chuỗi rỗng nếu thất bại (không ném lỗi).

```python
def ten_model_khop(ten_co_san: str, ten_can: str) -> bool
```
So tên model có/không có tag. Ollama trả `"qwen2.5vl:3b"` nhưng người dùng có thể viết
`"qwen2.5vl"`. Hàm nhỏ nhưng được dùng lại ở `rag_pipeline.kiem_tra_may_chu_llm()`.

```python
def mo_hinh_vision_co_san(client, ten_model=None) -> bool
```
Model đã được `ollama pull` về chưa. Kiểm **trước**, để không chạy nửa lần build rồi mới hỏng.

```python
def chu_thich_anh(client, duong_dan_anh, ten_model=None) -> str
```
Gọi model mô tả một hình. Rỗng nếu thất bại.

```python
def bo_sung_chu_thich_vision(cac_ban_ghi_anh: list, client=None) -> int
```
**Hàm điều phối, và là chỗ tối ưu nặng nhất của luồng Ingestion.** Mô tả được **nối thêm**
vào sau chú thích lân cận chứ không thay thế — hai nguồn bổ khuyết nhau: chú thích cho biết
hình được **gọi tên là gì** trong tài liệu (từ khoá người đọc sẽ dùng khi hỏi), model vision
cho biết **bên trong hình có gì**.

**Ba tầng tiết kiệm, đúng thứ tự này** (đảo thứ tự là hỏng):

1. **Gộp ảnh trùng nội dung** — gom theo băm nội dung ảnh, mỗi nội dung chỉ gọi model **một
   lần**. Hình dùng lại giữa các slide: từ N lượt còn 1.
2. **Cache trên đĩa** — ảnh đã chú thích ở lần build trước thì lấy lại, kể cả khi tài liệu
   đã đổi tên.
3. **Gọi song song** — số worker suy từ phần cứng (`tai_nguyen_gpu.so_worker_vision`). Bước
   này đắt nhất (~1,9 s/ảnh) nhưng cũng **nhàn rỗi nhất về phía Python** (toàn bộ thời gian
   là ngồi chờ HTTP), nên song song hoá có lãi.

> Ghi chú trong code nói rất rõ vì sao thứ tự quan trọng: làm ngược lại (gọi song song
> trước) chỉ khiến hệ thống chú thích **cùng một cái logo trên 8 luồng cùng lúc**.

---

### 11.4. `rag/chunking.py` — cắt chunk (298 dòng)

**Vai trò**: `List[trang]` → `List[chunk]`.

**Hằng số**: `DO_DAI_CHUNK_TOI_THIEU = 25` (ký tự), `_encoding` = tiktoken `cl100k_base`.

```python
def dem_token(text: str) -> int
```
Đếm token bằng tiktoken. **Chỉ dùng khi không có tokenizer của model thật.** Cảnh báo trong
docstring: với tiếng Việt, tiktoken đếm ra số token **gấp ~1.9 lần** tokenizer thật của các
model nền XLM-R, nên dùng nó làm thước đo sẽ tạo ra chunk nhỏ hơn dự định rất nhiều.

```python
def kich_thuoc_chunk_an_toan(max_seq_length: Optional[int] = None) -> int
```
Kích thước chunk thực tế, **đã hạ xuống cho vừa giới hạn của embedding model**
(`max_seq_length - BIEN_AN_TOAN_TOKEN`). Đây là **chặn cứng, không phải gợi ý**: nội dung
vượt `max_seq_length` bị model cắt bỏ **âm thầm** lúc encode, tức phần cuối chunk không hề
được đưa vào vector và không bao giờ tìm thấy được.

```python
def tao_splitter(dem_token_fn=None, max_seq_length=None) -> RecursiveCharacterTextSplitter
```
Dựng splitter của LangChain với `length_function` là tokenizer thật. `chunk_overlap` bị chặn
ở `chunk_size // 3` — với overlap ≥ chunk_size thì splitter rơi vào **vòng lặp vô tận**.

```python
def _tach_khoi_bang(text: str) -> List[Tuple[str, bool]]
```
Tách nội dung trang thành các đoạn `(nội_dung, là_bảng)`, giữ nguyên thứ tự, dựa vào cặp mốc
`[BẢNG]...[/BẢNG]`.

```python
def _hang_thanh_van_xuoi(hang: str) -> str
```
Đổi một hàng Markdown trở lại văn xuôi (bỏ `|` và ô rỗng) — dùng khi một ô chứa nguyên cả
đoạn văn.

```python
def _cat_bang_giu_tieu_de(khoi, dem, tran) -> List[Tuple[str, bool]]
```
Cắt bảng quá lớn thành nhiều mảnh, **mỗi mảnh được lặp lại dòng tiêu đề cột**. Không có
bước này thì các ô ở mảnh sau thành "giá trị trôi nổi không biết thuộc cột nào".

```python
def chia_chunk(cac_trang, dem_token_fn=None, max_seq_length=None) -> List[Dict]
```
Hàm chính. Ba đường xử lý khác nhau:

| Loại nội dung | Cách xử lý | Vì sao |
|---|---|---|
| **Ảnh** | 1 bản ghi → 1 chunk, không cắt | nội dung là chú thích ngắn; cắt nhỏ làm mất liên kết với đường dẫn ảnh |
| **Bảng** | giữ **nguyên khối** tới sát giới hạn model (`tran_token_bang`), quá thì cắt theo hàng và lặp tiêu đề | cắt nhỏ bảng phá đúng thứ khiến bảng là bảng |
| **Văn xuôi** | `RecursiveCharacterTextSplitter`, `CHUNK_SIZE_TOKENS = 160` | chunk nhỏ → mỗi vector mô tả một ý gọn, khớp câu hỏi sắc hơn |

> **Vì sao bảng được ưu đãi**: 160 token là lựa chọn về *độ chính xác truy xuất cho văn
> xuôi*. Với bảng, cắt nhỏ không đổi lại được gì. Đo trên bộ tài liệu thật: 15 bảng bị cắt
> vì vượt 160 token, phần lớn dài 164–476 token — tức vẫn nằm gọn trong giới hạn 496 của
> model. Chúng bị cắt **oan hoàn toàn**. Trần bảng được suy **từ chính model** chứ không
> thêm tham số cấu hình mới, nên đổi model là trần tự điều chỉnh theo.

`vi_tri` được đánh số **trước** khi lọc chunk quá ngắn, nên dãy có thể khuyết (0, 1, 3, 4…).
Đó là chủ ý: `rag_pipeline` sắp xếp theo `vi_tri` rồi lấy phần tử **liền kề trong danh
sách**, không dựa vào việc `vi_tri` liên tục.

---

### 11.5. `rag/bo_nho_dem.py` — bộ nhớ đệm (357 dòng)

**Vai trò**: bốn tầng cache theo **content hash**, làm cho câu "tài liệu đã xử lý rồi thì
không xử lý lại" thành hành vi thật.

#### Băm

```python
def bam_bytes(du_lieu: bytes) -> str      # 32 ký tự hex = nửa đầu SHA-256
def bam_chuoi(text: str) -> str
def bam_file(duong_dan: Path) -> str      # đọc theo khối 1MB, không nạp cả file vào RAM
```

#### Vân tay cấu hình

```python
def van_tay_doc_tai_lieu() -> str
def van_tay_embedding() -> str
```
Khoá cache **không chỉ gồm nội dung file**, mà còn gồm băm của những tham số ảnh hưởng tới
kết quả đọc (`_THAM_SO_ANH_HUONG_DOC_TAI_LIEU`: bật nhận diện tiêu đề, trích ảnh, model
vision, OCR…). Vì thế đổi một tuỳ chọn đọc là cache tự trượt, không cần ai nhớ xoá tay.

Tương tự, vector sinh bởi model khác thì không dùng lẫn được — nên `van_tay_embedding()` đi
vào khoá cache vector.

#### Kho khoá–giá trị trên đĩa

```python
class KhoDem:
    def __init__(self, ten: str, duoi: str = ".txt")
    def _duong_dan(self, khoa: str) -> Path     # chia thư mục con theo 2 ký tự đầu của khoá
    def co(self, khoa) -> bool
    def lay_text(self, khoa) -> Optional[str]
    def luu_text(self, khoa, noi_dung) -> None
    def lay_json(self, khoa)
    def luu_json(self, khoa, du_lieu) -> None
```
Mỗi giá trị là một file trong `data/cache/<ten>/<2 ký tự đầu>/<khoá>.<đuôi>`. Chia thư mục
con để tránh một thư mục có hàng chục nghìn file (chậm trên nhiều hệ tập tin).

Ba instance toàn cục:
```python
kho_tai_lieu = KhoDem("tai_lieu", ".json")   # kết quả đọc TRỌN VẸN 1 tài liệu
kho_ocr      = KhoDem("ocr", ".txt")         # text OCR 1 trang
kho_vision   = KhoDem("vision", ".txt")      # mô tả 1 ảnh
```

#### Hàm sinh khoá

```python
def khoa_tai_lieu(duong_dan: Path) -> str        # băm(nội dung file) + vân tay cấu hình đọc
def khoa_ocr(bam_tai_lieu_: str, so_trang) -> str
def khoa_vision(duong_dan_anh: Path) -> Optional[str]   # băm NỘI DUNG ẢNH + model + độ dài
```

> Chi tiết đáng chú ý: `khoa_vision` băm **nội dung ảnh**, không phải đường dẫn. Nhờ vậy
> cùng một cái logo xuất hiện ở 8 tài liệu khác nhau vẫn chỉ tốn **một** lượt gọi model.

#### Cache vector

```python
class KhoVectorDem:
    def __init__(self)
    def _nap(self) -> None
    def lay(self, text: str) -> Optional[np.ndarray]
    def them(self, text: str, vector: np.ndarray) -> None
    def luu(self) -> None
```
Khác ba kho trên: lưu trong **một file `.npz` duy nhất** (hàng nghìn file nhỏ cho hàng nghìn
vector là lãng phí). `luu()` gộp phần mới vào file — không có gì mới thì **không đụng vào
đĩa**.

Có một bẫy được ghi lại trong comment, rất đáng đọc: tên file tạm **phải** kết thúc bằng
`.npz`, vì `np.savez` tự nối thêm đuôi đó khi thiếu — một tên như `"....npz.tam"` sẽ được
ghi thành `"....npz.tam.npz"` và lệnh đổi tên ngay sau đó không tìm thấy file, khiến **cache
im lặng không bao giờ được ghi**.

```python
def encode_co_cache(embedding_service, cac_text, kho=None) -> np.ndarray
```
`encode_tai_lieu()` nhưng chỉ encode những chunk **chưa có** trong cache. Giữ nguyên chữ ký
"vào list text, ra mảng vector" để chỗ gọi không phải biết cache tồn tại. In log
`"Embedding: 8900/9285 chunk lấy từ cache, 385 chunk phải encode lại."`

#### Dọn dẹp

```python
def dung_luong_cache() -> int    # tổng byte, để UI nói được con số thật khi mời xoá
def xoa_cache() -> None          # an toàn tuyệt đối — mọi thứ trong đó đều tính lại được
```

---

### 11.6. `rag/do_thoi_gian.py` — đo thời gian (106 dòng)

**Vai trò**: đo thời gian từng bước Ingestion để **tối ưu bằng số đo, không bằng phỏng đoán**.

Trạng thái toàn cục: `_bo_dem: Dict[str, (số_lần, tổng_giây)]`, có `threading.Lock` vì bước
chú thích ảnh chạy đa luồng.

```python
def dat_lai() -> None                                    # gọi ở đầu mỗi lần build
def ghi_nhan(ten_buoc: str, so_giay: float, so_lan=1)
@contextmanager
def do(ten_buoc: str)                                    # with do("ocr_trang"): ...
def so_lieu() -> Dict[str, Tuple[int, float]]
def tong_giay() -> float
def bao_cao(tieu_de="PROFILING INGESTION") -> str        # bảng text, sắp giảm dần
def ghi_bao_cao(tieu_de="PROFILING INGESTION") -> None
```

Cách dùng điển hình, thấy khắp `document_loader` và `app.py`:

```python
with do_thoi_gian.do("pdf_doc_text_trang"):
    noidung = _doc_mot_trang_pdf(...)
```

Các nhãn đang được đo: `bam_tai_lieu`, `tai_lieu_doc_moi`, `tai_lieu_trung_cache`,
`pdf_doc_text_trang`, `pdf_nhan_dien_tieu_de`, `pdf_trich_anh`, `chunking`, `embedding`,
`query_ma_hoa_cau_hoi`, `query_ung_vien_dense_bm25`, `query_rerank`.
---

## 12. Nhóm LƯU TRỮ & TÌM KIẾM

---

### 12.1. `rag/embedding.py` — bọc sentence-transformers (147 dòng)

**Vai trò**: biến chữ thành vector. Mỏng, nhưng có bốn chi tiết quan trọng.

```python
class EmbeddingService:
    def __init__(self, model_name: str = None, thiet_bi: str = None)
```
`thiet_bi`: `"cuda"` / `"cpu"`; bỏ trống thì **tự dò** theo phần cứng qua
`tai_nguyen_gpu.thiet_bi("embedding")`.

```python
    def chuyen_thiet_bi(self, thiet_bi_moi: str) -> bool
```
Chuyển model sang thiết bị khác **tại chỗ**. Trả `True` nếu thật sự có chuyển. Đây là thứ
cho phép `tai_nguyen_gpu` đẩy embedding xuống CPU sau khi build xong, nhường VRAM cho
reranker và LLM.

```python
    def _encode(self, texts: List[str], tien_to: str) -> np.ndarray
```
Lõi. Thêm tiền tố vào từng text rồi gọi `model.encode(..., normalize_embeddings=True)`.

> **`normalize_embeddings=True` là điều kiện tiên quyết của cả hệ thống**: nó chuẩn hoá
> vector về độ dài 1, để khi FAISS tính **inner product** thì con số ra **đúng bằng cosine
> similarity**. Bỏ cờ này đi thì `IndexFlatIP` vẫn chạy, vẫn trả kết quả, nhưng điểm số mất
> hết ý nghĩa và mọi ngưỡng trong hệ thống sai hết.

```python
    def encode_tai_lieu(self, texts) -> np.ndarray      # tiền tố "passage: "
    def encode_cau_hoi(self, texts) -> np.ndarray        # tiền tố "query: "
```
Hai hàm riêng, không phải một hàm `encode` dùng chung — vì họ model E5 được huấn luyện với
hai tiền tố khác nhau cho hai vai trò.

```python
    @property def dimension(self) -> int                  # 768 với e5-base
    @property def max_seq_length(self) -> int             # 512 (thực dụng: 496 sau biên an toàn)
    def dem_token(self, text: str) -> int                 # đếm bằng ĐÚNG tokenizer của model
    def lay_ham_dem_token(self) -> Optional[Callable]     # trả hàm để truyền sang chunking
```

`lay_ham_dem_token()` là mắt xích khiến chunking đúng: `app.py` truyền hàm này vào
`chia_chunk()`, nên chunk được đo bằng **thước đo của chính model sẽ encode nó**, không phải
bằng tiktoken (lệch ~1.9 lần với tiếng Việt).

---

### 12.2. `rag/vector_store.py` — bọc FAISS (319 dòng)

**Vai trò**: giữ vector + metadata song song, tìm kiếm, lưu/nạp, và kiểm tra tương thích.

```python
def so_sanh_bam_tai_lieu(bam_trong_index, bam_tren_dia) -> (can_doc, can_xoa, giu_nguyen)
```
**Hàm thuần** (chỉ nhận hai dict, không đụng đĩa, không đụng FAISS) — đây là **quyết định
của build tăng dần**:
- `can_doc`: file mới, hoặc băm khác với băm đã ghi lúc build.
- `can_xoa`: tên còn trong sổ băm nhưng file đã biến mất khỏi thư mục.
- `giu_nguyen`: băm khớp → vector cũ vẫn đúng.

> Tách thành hàm thuần là quyết định thiết kế tốt: một quyết định sai ở đây **không gây
> lỗi**, chỉ khiến index thiếu hoặc thừa một tài liệu — đúng loại hỏng im lặng. Hàm thuần
> thì kiểm được bằng test bảng, không cần dựng cả một lần build.

```python
class VectorStore:
    def __init__(self, dimension: int)
```
Tạo `faiss.IndexFlatIP(dimension)` + `metadata: List[Dict]` + `thong_tin: Dict` +
`bam_tai_lieu: Dict[str, str]`.

> **Vì sao `IndexFlatIP` chứ không `IndexFlatL2`**: vector đã chuẩn hoá → inner product ==
> cosine similarity, đúng thước đo "độ liên quan ngữ nghĩa", thay vì khoảng cách hình học
> thô giữa hai điểm.
>
> **Vì sao `Flat` (quét toàn bộ) chứ không IVF/HNSW (gần đúng)**: `evaluation/do_quy_mo_index.py`
> tồn tại để trả lời đúng câu này bằng số đo. Với quy mô của đồ án (9.285 chunk), Flat cho
> kết quả **chính xác tuyệt đối** trong ngân sách độ trễ 200 ms — không có lý do đánh đổi
> recall lấy tốc độ mà mình chưa cần.

**Ba chỉ mục phụ dựng LƯỜI** (chỉ tính khi cần lần đầu, sau đó dùng lại):

```python
    def _xoa_cache(self) -> None
```
Gọi sau **mọi** thay đổi dữ liệu. Cả ba chỉ mục đều ánh xạ theo **vị trí** trong
`self.metadata`, mà thêm/xoá vector làm các vị trí đó dịch đi — giữ lại cache cũ sẽ khiến
tìm kiếm trả về nội dung của chunk khác. **Sai âm thầm, không báo lỗi.**

```python
    @property def chi_muc_trang(self) -> Dict[(nguon, trang), List[int]]
```
Vị trí các chunk của từng trang, sắp theo `vi_tri`. Dùng để mở rộng ngữ cảnh **trong một
trang** mà không phải quét tuyến tính toàn bộ metadata cho mỗi câu hỏi.

```python
    @property def chi_muc_nguon(self) -> Dict[nguon, List[int]]
```
Vị trí **mọi** chunk của một tài liệu, sắp theo `(trang, vi_tri)` — tức **thứ tự đọc xuyên
trang**. Cần cho việc nối lại một định nghĩa bắt đầu cuối trang 12 và kết thúc đầu trang 13.

> Chi tiết đáng học: thứ tự đọc được **suy ra từ hai trường đã có sẵn** (`trang`, `vi_tri`)
> chứ không thêm một trường "thứ tự toàn cục" mới. Lý do rất thực dụng: thêm trường mới thì
> mọi index đã build đều thiếu nó và phải build lại — mà build lại tốn nguyên một lượt chú
> thích ảnh bằng model vision (~9 phút cho 291 ảnh). Thông tin cần thiết vốn đã nằm trong dữ
> liệu.

```python
    @property def bm25(self) -> BM25
```
Chỉ mục BM25, dựng lười từ `[m["noidung"] for m in metadata]`.

**Ghi dữ liệu**

```python
    def them(self, vectors: np.ndarray, metadata_list: List[Dict]) -> None
```
Thêm một batch. Ném `ValueError` nếu số vector ≠ số metadata (bất biến quan trọng nhất của
class này).

```python
    def xoa_theo_nguon(self, ten_file: str) -> int
```
Xoá toàn bộ vector + metadata của một file **ngay lập tức**, không cần build lại. Hai chi
tiết kỹ thuật: `remove_ids` của FAISS nén mảng lại nhưng **giữ nguyên thứ tự tương đối**;
metadata phải xoá theo thứ tự **giảm dần** (từ cuối lên) để không lệch chỉ số. Cũng xoá luôn
khỏi sổ băm — một tài liệu không còn vector nào mà vẫn được ghi "đã xử lý" sẽ khiến lần
build sau bỏ qua nó.

**Truy vấn**

```python
    def theo_nguon_va_trang(self, nguon, trang) -> List[Dict]
    def tim_kiem_vi_tri(self, vector_cau_hoi, top_k=None) -> List[(vi_tri, diem)]
    def tim_kiem(self, vector_cau_hoi, top_k=None) -> List[(metadata, diem)]
    def tim_kiem_tu_khoa(self, cau_hoi: str, top_n: int) -> List[(vi_tri, diem)]
    def diem_cosine(self, vi_tri: List[int], vector_cau_hoi) -> Dict[int, float]
```
`tim_kiem_vi_tri` trả **vị trí** chứ không phải nội dung, vì `rag_pipeline` cần vị trí để
hợp nhất với nhánh BM25 và tra chunk liền kề (dict metadata không hash được, và hai chunk
trùng nội dung sẽ lẫn vào nhau). Nó cũng lọc bỏ ô `-1` mà FAISS trả về khi không đủ kết quả.

`diem_cosine` tính bù điểm cho những chunk chỉ do BM25 hoặc do truy vấn phụ tìm ra — dùng
`index.reconstruct()` để đọc lại vector gốc, chính xác chứ không xấp xỉ. Mục đích: **mọi
ngưỡng và mọi điểm hiển thị đều quy về một thang cosine duy nhất.**

**Lưu / nạp / kiểm tra**

```python
    def luu(self, index_path=None, metadata_path=None, info_path=None) -> None
```
Ghi ba file. `index_info.json` chứa "vân tay": sổ băm tài liệu, tên model embedding, chunk
size/overlap, và ba tuỳ chọn **ăn vào nội dung đã index** (`nhan_dien_tieu_de`, `trich_anh`,
`chu_thich_anh_vision`), số chunk, thời điểm build.

```python
    @classmethod def tai(cls, ...) -> "VectorStore"
```
Dùng `__new__` thay vì `__init__` để tránh tạo một `IndexFlatIP` rỗng rồi bỏ đi ngay. Index
build bằng bản cũ không có khoá `bam_tai_lieu` → dict rỗng → mọi tài liệu bị coi là "chưa xử
lý" và được đọc lại đúng một lần. **Giảm cấp về hành vi cũ, không phải lỗi.**

```python
    def ly_do_khong_tuong_thich(self) -> Optional[str]
```
Trả mô tả lý do index không còn khớp cấu hình hiện tại, hoặc `None`. Được gọi ở hai chỗ:
cảnh báo trên thanh bên, và chặn build tăng dần.

```python
    @property def so_luong_vector(self) -> int
```

---

### 12.3. `rag/lexical_search.py` — BM25 (100 dòng)

**Vai trò**: nhánh tìm kiếm theo **từ khoá**, bù khuyết cho FAISS ở những chỗ vector kém:
mã định danh, tên riêng, thuật ngữ ngoài từ vựng.

Tự cài thay vì dùng `rank_bm25` — chỉ ~50 dòng logic, và tự cài thì kiểm soát được cách tách
từ cho tiếng Việt.

**Hằng số**: `K1 = 1.5`, `B = 0.75` (giá trị kinh điển của BM25).

```python
def _tach_tu(text: str) -> List[str]
```
Tách một đoạn text thành danh sách "từ". Điểm đáng chú ý: hàm sinh **cả âm tiết đơn lẫn cặp
âm tiết liền nhau**. Lý do: tiếng Việt là ngôn ngữ đơn lập, "máy học" là hai âm tiết nhưng
một khái niệm — chỉ đánh chỉ mục âm tiết đơn thì "máy" và "học" tách rời, mất nghĩa. Sinh
thêm cặp `"máy học"` giữ được cụm từ. Cũng chuẩn hoá Unicode để `"ế"` ở hai dạng mã hoá khác
nhau vẫn khớp.

```python
class BM25:
    def __init__(self, cac_van_ban: List[str])
    def tim_kiem(self, cau_hoi: str, top_n: int) -> List[Tuple[int, float]]
```
Chỉ mục dựng trong bộ nhớ. `tim_kiem` trả `[(chỉ_số_tài_liệu, điểm)]` giảm dần, tối đa
`top_n`. Corpus rỗng hoặc không từ nào trùng → trả danh sách rỗng, không ném lỗi.

**Công thức BM25** (để đối chiếu khi đọc code):

```
điểm(D, Q) = Σ  IDF(q) ·  ─────────── f(q,D) · (k1 + 1) ───────────
             q∈Q          f(q,D) + k1 · (1 - b + b · |D| / avgdl)
```

Ý nghĩa: từ càng **hiếm** trong corpus thì `IDF` càng cao; từ xuất hiện nhiều lần trong một
tài liệu thì điểm tăng nhưng **bão hoà** (nhờ `k1`); tài liệu dài bị **phạt** (nhờ `b`).

---

### 12.4. `rag/reranker.py` — cross-encoder (88 dòng)

**Vai trò**: tầng lọc **thứ hai**. Đọc cả cặp `(câu hỏi, đoạn văn)` cùng lúc và chấm điểm
liên quan.

```python
class RerankerService:
    def __init__(self, model_name=None, thiet_bi=None)
    def _nap_model(self) -> CrossEncoder      # tự xử lý model cần trust_remote_code
    def xep_hang(self, cau_hoi: str, cac_doan: List[str]) -> np.ndarray
```
`xep_hang` trả mảng điểm, cùng thứ tự với `cac_doan`. Đoạn rỗng không gây lỗi.

```python
def tao_reranker_neu_bat() -> Optional[RerankerService]
```
Trả `None` khi `BAT_RERANK=0` — lúc đó model ~2.2 GB **không được nạp chút nào**. Hàm nhỏ
này là điểm duy nhất trong hệ thống quyết định "có nạp reranker không", nên tắt rerank là
tắt thật, không phải nạp rồi bỏ qua.

---

## 13. Nhóm QUERY

---

### 13.1. `rag/tiep_noi_hoi_thoai.py` — câu hỏi nối tiếp (403 dòng)

**Vấn đề**: người dùng hỏi *"Vi phạm pháp luật gồm những dấu hiệu nào?"*, rồi hỏi tiếp
*"Thế còn dấu hiệu thứ hai?"*. Câu thứ hai tách khỏi ngữ cảnh thì vô nghĩa — vector của nó
không trỏ về đâu cả, và tệ hơn: **điểm rerank của nó rơi về gần 0**, tức nó bị **từ chối
oan** bởi chính cơ chế chặn câu lạc đề.

**Kiến trúc ba tầng**:

```
TẦNG 1: nhận diện (tất định)      la_cau_hoi_tiep_noi()
   ↓ nếu là nối tiếp
TẦNG 2A: ghép ngữ cảnh (tất định, MẶC ĐỊNH)   truy_van_ngu_canh()
TẦNG 2B: viết lại bằng LLM (MẶC ĐỊNH TẮT)     viet_lai_cau_hoi()
   ↓
ĐIỂM VÀO: chuan_bi_truy_van()
```

#### Tầng 1 — nhận diện

```python
def la_cau_hoi_tiep_noi(cau_hoi: str, lich_su: Optional[List[Dict]] = None) -> bool
```
Căn cứ: cụm trỏ ra ngoài (`_DAU_HIEU_HOI_CHI`: "thế còn", "vậy còn", "cái đó", "điều đó",
"vi…"), từ mở đầu nối tiếp (`_MO_DAU_NOI_TIEP`: "còn", "thế", "vậy", "nhưng", "and", "but",
"so"…), câu quá ngắn, hoặc câu không có danh từ chính. **Không có lịch sử thì luôn trả
`False`** — không thể nối tiếp cái gì cả.

#### Tầng 2A — ghép ngữ cảnh (đường mặc định)

```python
def cac_cau_hoi_truoc(lich_su, so_luot=None) -> List[str]
```
Lấy `SO_LUOT_NGU_CANH` = 3 câu hỏi **của người dùng** gần nhất, mới nhất đứng trước.

```python
def truy_van_ngu_canh(cau_hoi: str, lich_su: List[Dict]) -> str
```
Ghép: `"<câu hỏi cũ nhất> ... <câu hỏi gần nhất> <câu hỏi hiện tại>"`.

Hai quyết định trong hàm ngắn này đều có lý do đo được:
- **Chỉ lấy câu HỎI, không lấy câu TRẢ LỜI.** Câu trả lời cũ dài hơn câu hỏi hàng chục lần,
  ghép vào sẽ khiến vector truy vấn bị nó lấn át — mà câu trả lời cũ thì đã nằm sẵn trong
  tài liệu, tức nhánh này sẽ chỉ kéo về đúng những đoạn vừa dùng ở lượt trước.
- **Câu hỏi hiện tại đặt SAU cùng**, vì nó là thứ cần được nhấn.

```python
def ngu_canh_cho_prompt(lich_su, ngon_ngu="vi") -> str
```
Dựng khối ngữ cảnh cho **prompt sinh câu trả lời** (khác với truy vấn). Cần thiết vì LLM
**không nhìn thấy lịch sử chat** — `messages` chỉ có `[system, user]`.

Khối này có nhãn riêng và một câu cảnh báo bắt buộc:
*"ĐÂY KHÔNG PHẢI nguồn thông tin, tuyệt đối không trích dẫn"*. Lý do: cho model coi ngữ cảnh
hội thoại ngang hàng với đoạn trích là **mở đúng cánh cửa mà cả hệ thống tồn tại để đóng** —
model sẽ trích dẫn lại lời của chính nó như thể đó là tài liệu.

#### Tầng 2B — viết lại bằng LLM (mặc định tắt)

```python
PROMPT_VIET_LAI  # 5 quy tắc: thay từ trỏ, giữ ngôn ngữ, không trả lời, không sửa nếu đã độc lập
def _dung_ngu_canh(lich_su) -> str
def _lam_sach(ket_qua_tho: str) -> str
def viet_lai_cau_hoi(cau_hoi, lich_su, client=None) -> str
```
`_lam_sach` bóc những thứ model hay thêm dù prompt đã cấm: nhãn, ngoặc kép, gạch đầu dòng.

`viet_lai_cau_hoi` có **bốn lớp kiểm tra**, và bất kỳ lớp nào không qua thì **giữ nguyên câu
gốc**:
1. Model chết / trả rác → giữ gốc.
2. Bản viết lại dài hơn `SO_TU_TOI_DA_CAU_VIET_LAI` = 60 từ → loại.
3. Bản viết lại **ngắn hơn** câu gốc → loại (nó phải bổ sung ngữ cảnh, không phải cắt bớt).
4. Model trả lời thay vì viết lại → loại.

> Đây là một khuôn mẫu đáng học: mỗi khi thêm một lượt gọi LLM vào đường nóng, phải kèm đủ
> kiểm tra để **kết quả xấu không tệ hơn việc không gọi**.

#### Điểm vào

```python
def chuan_bi_truy_van(cau_hoi, lich_su=None, client=None) -> Dict
```
Trả về:

| Khoá | Ý nghĩa |
|---|---|
| `cau_hoi_goc` | đúng chuỗi người dùng gõ; **luôn là một nhánh RRF riêng** |
| `cau_hoi_chinh` | truy vấn **CHÍNH**: dùng để rerank và để đo cosine |
| `cac_truy_van_phu` | các nhánh truy vấn thêm |
| `ngu_canh_llm` | khối ngữ cảnh cho prompt sinh câu trả lời |
| `la_tiep_noi` | tầng nhận diện đã kết luận đây là câu nối tiếp |
| `da_viet_lai` | đường LLM đã đổi được câu hỏi |

Câu hỏi tự đứng được — **đại đa số** — đi qua đây gần như không tốn gì: một phép so chuỗi.

> **Vì sao `cau_hoi_goc` vẫn là một nhánh riêng**: để một lần ghép ngữ cảnh **sai** không
> xoá được kết quả đúng của câu gốc. RRF hợp nhất nhiều danh sách xếp hạng — thêm một nhánh
> nhiễu chỉ làm thứ hạng nhiễu đi, không giết được nhánh đúng. Đây là lý do việc thêm truy
> vấn phụ **không cần cơ chế mới**: nó đi qua đúng cơ chế RRF đã có.

---

### 13.2. `rag/rag_pipeline.py` — trái tim hệ thống (1.510 dòng)

**Vai trò**: ghép toàn bộ luồng Query. Đây là file quan trọng nhất; nếu chỉ đọc được một
file thì đọc file này.

#### A. Xử lý lỗi kết nối Ollama

```python
class LoiKhongKetNoiDuocOllama(RuntimeError)
def _thong_bao_khong_ket_noi_duoc() -> str
def kiem_tra_may_chu_llm() -> Optional[str]
```

Cần một lớp lỗi **riêng** vì đây là lỗi **môi trường**, không phải lỗi dữ liệu: người dùng
sửa được bằng đúng một câu lệnh, nhưng chỉ khi được nói cho biết phải chạy lệnh gì.

Chi tiết kỹ thuật đáng nhớ: thư viện `ollama` đã bọc lỗi kết nối thành `ConnectionError` có
thông báo tử tế — nhưng **chỉ ở đường gọi thường**. Ở đường **streaming** (đường duy nhất hệ
thống này dùng), kết nối chỉ thật sự mở khi generator được lặp lần đầu, nằm ngoài khối bọc
lỗi đó, nên `httpx.ConnectError` bay thẳng ra ngoài. Người dùng nhận nguyên traceback
`"[WinError 10061] ... target machine actively refused it"` — không một chữ nào nhắc tới
Ollama.

`_thong_bao_khong_ket_noi_duoc()` được dựng **lúc gặp lỗi** chứ không phải hằng số dựng lúc
import, để nếu người dùng đổi `OLLAMA_HOST` trong `.env` thì thông báo vẫn nói đúng giá trị
thật.

`kiem_tra_may_chu_llm()` kiểm **trước** khi người dùng gõ câu hỏi, dùng đường gọi không
streaming (`list`) nên nhanh và không đánh thức model.

#### B. Lọc suy luận theo luồng

```python
class _LocSuyLuanTheoLuong:
    def __init__(self) -> None
    def them(self, manh: str) -> (phần_câu_trả_lời, phần_suy_luận)
    def ket_thuc(self) -> (phần_còn_lại, phần_suy_luận)
```
Máy trạng thái bóc `<think>...</think>` ra khỏi luồng **đang chảy**.

Vì sao phải là máy trạng thái chứ không phải regex: bản không streaming chạy regex trên
chuỗi **đã hoàn chỉnh**; streaming không có chuỗi hoàn chỉnh, mảnh đang tới có thể cắt ngang
giữa thẻ (`"<thi"` | `"nk>"`). Giải pháp: giữ lại phần đuôi có thể là **nửa cái thẻ** (tối đa
`len("</think>") - 1` ký tự) cho tới khi mảnh sau tới đủ để kết luận.

#### C. Nhận diện ngôn ngữ

```python
def _phat_hien_ngon_ngu(cau_hoi: str) -> str      # "vi" | "en"
```
**Năm bước**, dừng ở bước nào có bằng chứng chắc nhất:

1. **Dấu tiếng Việt** (`ăâđêôơư` + thanh điệu) → chốt `"vi"` ngay. Không ngôn ngữ nào khác
   dùng đủ bộ này.
2. So **trực tiếp xác suất của đúng hai ngôn ngữ quan tâm**, bỏ qua thứ hạng chung.
3. Tìm dấu vết **tiếng Việt không dấu** (`"gi"`, `"khong"`, `"nao"`, `"duoc"`…).
4. Không có dấu vết nào mà vẫn là câu có chữ → `"en"`.
5. Hết cách → `"vi"` (ngôn ngữ chính của hệ thống).

> **Vì sao phức tạp đến vậy**: bản trước viết `"en" if detect(...) == "en" else "vi"`. Đo
> thực tế: `langdetect` chấm *"What does criminal law regulate?"* là **tiếng Catalan (0.71)**
> và tiếng Anh chỉ 0.29 — nên câu tiếng Anh rõ ràng đó bị trả lời bằng tiếng Việt. Thêm nữa,
> `langdetect` lấy mẫu ngẫu nhiên nên **mặc định không tất định**: cùng một câu chạy 8 lần
> cho 8 kết quả khác nhau. Dòng `DetectorFactory.seed = 0` ở đầu file sửa việc đó.

#### D. Bốn system prompt

| Hằng số | Dùng khi |
|---|---|
| `HE_THONG_PROMPT_VI` | câu hỏi thường, tiếng Việt |
| `HE_THONG_PROMPT_EN` | câu hỏi thường, tiếng Anh |
| `HE_THONG_PROMPT_KIEM_CHUNG_VI` | câu **kiểm chứng**, tiếng Việt |
| `HE_THONG_PROMPT_KIEM_CHUNG_EN` | câu **kiểm chứng**, tiếng Anh |

**Prompt thường — 7 quy tắc**: chỉ dùng ngữ cảnh; không đủ thì trả lời đúng câu từ chối; mỗi
ý phải kèm số `[n]`; **không đồng ý với giả định sai của người hỏi**; trả lời **đầy đủ** (ưu
tiên cao hơn ngắn gọn); "ngắn gọn" nghĩa là **không có chữ thừa**, không phải ít thông tin
hơn; đúng ngôn ngữ.

> Quy tắc 5 và 6 **từng xung đột nhau**. Bản trước đặt cạnh nhau "trả lời ĐẦY ĐỦ" và "trả
> lời ĐI THẲNG VÀO TRỌNG TÂM" mà không nói cái nào thắng. Với model 4B, đặt hai chỉ thị
> ngược chiều cạnh nhau khiến model chọn cái **dễ tuân thủ hơn** — mà "ngắn gọn" luôn dễ hơn
> "đầy đủ", vì bỏ bớt thì không có gì để làm sai.

**Prompt kiểm chứng — khác biệt cốt lõi**: bắt buộc ra phán quyết **rời rạc**
(ĐÚNG / SAI / KHÔNG ĐỀ CẬP) và bắt buộc trích **nguyên văn** câu làm căn cứ, theo bố cục
`Căn cứ → Đối chiếu → KẾT LUẬN`.

Hai ràng buộc này chặn đúng cơ chế gây lỗi: model không còn được phép viết một đoạn văn
chung chung "nghe như đang đồng ý". Và **KẾT LUẬN phải viết SAU CÙNG** — lý do được ghi
thẳng trong prompt: *"nếu bạn chốt trước rồi mới đối chiếu, bạn sẽ bảo vệ kết luận đã lỡ nói
thay vì đọc lại tài liệu."*

```python
def la_cau_hoi_kiem_chung(cau_hoi: str) -> bool
```
Khớp `_MAU_KIEM_CHUNG` — 30 mẫu: `"có phải"`, `"đúng không"`, `"theo tôi"`, `"tôi nhớ"`,
`"nghe nói"`, `"is it true"`, `"i think"`, `"right?"`…

> Cố ý ưu tiên **độ chính xác hơn độ phủ**: một câu hỏi thường bị nhận nhầm thành kiểm chứng
> sẽ nhận về bố cục phán quyết khá cứng nhắc, nên **thà bỏ sót còn hơn nhận nhầm**. Phần bỏ
> sót vẫn được quy tắc 4 của prompt thường (chống a dua) đỡ lại.

#### E. Ngân sách token — cụm hàm quan trọng nhất

```python
def _uoc_luong_so_token(*cac_phan: str) -> int
```
Ước lượng token **không cần tokenizer của LLM** (hệ thống chỉ có tokenizer của embedding
model, họ XLM-R, không phải của Qwen). Tỉ lệ `SO_KY_TU_MOI_TOKEN_UOC_LUONG = 2.2` **cố ý đặt
thấp hơn** giá trị đo được cho tiếng Việt, để **luôn ước lượng dư**: sai về phía cấp dư hoàn
toàn vô hại, sai về phía cấp thiếu thì tái tạo lại đúng bug `num_ctx`.

```python
def la_cau_hoi_phuc_tap(cau_hoi: str) -> bool
```
Ba dấu hiệu, chỉ cần khớp một: dài hơn 12 từ; chứa động từ nhiều vế (`_MAU_CAU_HOI_NHIEU_VE`:
"so sánh", "phân biệt", "liệt kê", "các bước", "vì sao", "compare", "list"…); hoặc là câu
kiểm chứng.

> **Cố ý nghiêng về phía cấp dư**: đoán nhầm câu phức tạp thành đơn giản thì câu trả lời có
> thể thiếu ý — một lỗi **người dùng nhìn thấy**. Đoán nhầm chiều ngược lại chỉ làm câu hỏi
> đó chạy chậm bằng đúng bản cũ. Hai loại sai này **không ngang giá**, nên ngưỡng cũng không
> đặt ở giữa.

```python
def _tinh_num_ctx(so_token_prompt: int, num_predict=None) -> int
```
Cửa sổ ngữ cảnh cần cấp. **Tính động, nhưng làm tròn lên theo thang gấp đôi** từ
`OLLAMA_NUM_CTX`:

- Tính động vì: một hằng số đủ dùng hôm nay sẽ âm thầm không đủ vào ngày ai đó tăng `TOP_K`.
- Làm tròn theo bậc vì: Ollama coi `num_ctx` là một phần **định danh của phiên bản model
  đang nạp**, đổi giá trị giữa hai lượt hỏi sẽ khiến nó **nạp lại model** (hàng chục giây
  trên CPU).
- Ghi log `WARNING` khi chạm trần `OLLAMA_NUM_CTX_TOI_DA`, kèm chỉ dẫn cụ thể: *"Hạ TOP_K
  hoặc NGAN_SACH_KY_TU_MOI_DOAN (ĐỪNG hạ num_ctx)"*.

```python
def ngan_sach_token_ngu_canh(num_predict: int, so_token_co_dinh: int) -> int
```
Số token còn lại cho các **đoạn trích**, sau khi trừ mọi phần **không được phép cắt**
(system prompt + câu hỏi + khối ngữ cảnh hội thoại — cắt system prompt là gỡ bỏ chính các
ràng buộc chống bịa đặt).

```python
def nen_ngu_canh(cac_chunk: List[Dict], ngan_sach_token: int) -> List[Dict]
```
Ép đoạn trích vào ngân sách, **bỏ từ đoạn xếp hạng thấp nhất lên**.

Ba lý do trong docstring đáng đọc nguyên văn:
- **Vì sao cần bước này** dù `_tinh_num_ctx` đã cảnh báo: cảnh báo chỉ nói cho người dùng
  biết cấu hình quá tay, nhưng lượt hỏi **đang chạy** vẫn hỏng — Ollama cắt im lặng từ **đầu**
  phần user content, tức xoá đúng đoạn `[1]`, đoạn liên quan nhất.
- **Vì sao không hạ `num_ctx`**: hạ không làm prompt ngắn lại, nó chỉ đổi chỗ bị cắt từ "do
  ta chọn" sang "do máy chủ chọn" — mà máy chủ luôn chọn cắt phần đầu, tức phần quý nhất.
- **Vì sao bỏ từ CUỐI**: giữ nguyên số thứ tự `[1] [2]` của các đoạn còn lại. Bỏ từ giữa thì
  mọi số sau đó lệch một bậc và trích dẫn trỏ sai nguồn.

Trả về danh sách **mới**; danh sách gốc không đổi nên trích dẫn hiển thị cho người đọc vẫn
giữ nguyên văn đầy đủ.

```python
def _noi_lien_mach(truoc: str, sau: str, toi_da: int = 300) -> str
```
Nối hai chunk liền kề, **bỏ phần lặp do overlap**. Tìm đoạn cuối của `truoc` trùng với đoạn
đầu của `sau` rồi bỏ đi đúng một bản.

```python
def _ghep_prompt(cau_hoi, cac_chunk, ngon_ngu, la_kiem_chung, ngu_canh_hoi_thoai="") -> str
```
Ghép prompt cuối cùng. Đánh số `[1]`, `[2]`… **khớp đúng thứ tự** với `citation.py`. Nhãn
(`Nguồn` / `Source`) theo ngôn ngữ để **toàn bộ prompt nhất quán một ngôn ngữ**.

Khối ngữ cảnh hội thoại đặt **sau** đoạn trích và **trước** câu hỏi, dưới nhãn riêng — ba
chi tiết đều có lý do (xem §13.1).

#### F. `class RagPipeline`

```python
    def __init__(self, embedding_service, vector_store, reranker_service=None)
```
Nhận cả ba từ ngoài vào, **không tự tạo** — để tránh nạp lại model / index nhiều lần.

Năm thuộc tính trạng thái, mỗi cái có vai trò riêng:

| Thuộc tính | Ý nghĩa |
|---|---|
| `diem_rerank_cao_nhat` | điểm rerank cao nhất của lượt **gần nhất**, dùng cho ngưỡng từ chối |
| `truy_van_da_dung` | kết quả `chuan_bi_truy_van` của lượt gần nhất |
| `thong_ke_llm` | bộ đếm token **thật** do Ollama trả về |
| `la_cau_hoi_phuc_tap` | phán đoán độ phức tạp, đặt ở `truy_xuat()`, đọc lại ở bước sinh |
| `_ho_tro_thinking` | đặt `False` khi model báo không hỗ trợ `think`, để không thử lại |

> `la_cau_hoi_phuc_tap` được **đặt một lần và dùng ở hai bước** là chủ ý: hai bước tự đánh
> giá riêng thì có lúc lệch nhau, và lúc đó ngân sách rerank với ngân sách sinh không còn
> nói về cùng một câu hỏi nữa.

```python
    def _ung_vien(self, cac_truy_van, so_ung_vien, nguon_cho_phep) -> (ung_vien, vi_tri_cuu_ho)
```
Lấy ứng viên từ mọi nhánh rồi hợp nhất bằng **RRF có trọng số**.

`cac_truy_van` là `[(câu_hỏi, vector, trọng_số)]`; phần tử **đầu tiên** là truy vấn CHÍNH —
điểm cosine trả ra được đo theo nó, vì mọi ngưỡng trong hệ thống hiệu chỉnh trên **một thang
cosine duy nhất**.

Lọc theo `nguon_cho_phep` phải làm **tại đây** vì FAISS/BM25 đều không lọc được theo metadata
trong lúc tìm — đó cũng là lý do phải over-fetch từ đầu.

**BM25 cứu hộ**: bơm thêm ứng viên vào tập rerank mà **không cho một điểm RRF nào** (điểm 0
→ luôn xếp cuối). Đây là chỗ tách bạch hai vai trò của BM25: giúp **recall** mà không có
quyền **precision**. Việc xếp hạng để cross-encoder quyết.

```python
    def _xep_hang_lai(self, cau_hoi, ung_vien, vi_tri_cuu_ho=None, so_ung_vien_rerank=None)
```
Chấm `so_ung_vien_rerank` ứng viên đầu (30 hoặc 12), **cộng thêm** các ứng viên cứu hộ dù
chúng nằm ngoài. Phần đuôi giữ nguyên thứ tự RRF (không cắt bỏ, vì các bước sau còn lọc tiếp
và có thể không đủ ứng viên lấp `TOP_K`).

Ba chi tiết:
- Chấm trên **nội dung chunk GỐC**, chưa mở rộng — vừa rẻ hơn vừa đúng hơn về mặt đo lường:
  ta đang hỏi *"chunk này có khớp câu hỏi không"*, không phải *"cả vùng quanh nó có khớp
  không"*.
- **Điểm cosine được giữ nguyên**, không thay bằng điểm rerank. Rerank chỉ đổi **thứ tự
  chọn**.
- Ứng viên cứu hộ **luôn được chấm** — nếu không thì cả cơ chế cứu hộ vô nghĩa (điểm RRF 0
  đẩy chúng xuống cuối, mà xuống cuối thì không bao giờ được nhìn tới).

```python
    def _dung_doan_trich(self, vi_tri_neo: int) -> Dict
```
Dựng một đoạn trích **liền mạch** quanh chunk khớp nhất. Bắt đầu từ chunk neo, mở rộng
**luân phiên** sang sau / trước cho tới khi chạm `NGAN_SACH_KY_TU_MOI_DOAN`. Ưu tiên mở rộng
về phía **sau** trước, vì kiểu mất mát hay gặp nhất là một câu bị ranh giới chunk cắt ngang,
phần còn thiếu nằm ở chunk kế tiếp.

Hai chốt chặn: ngân sách ký tự, và số trang tối đa được vượt qua mỗi hướng
(`SO_TRANG_TOI_DA_MO_RONG` = 1 — chốt này giữ cho slide thưa chữ không hút thêm 2–3 slide
xung quanh cho đầy ngân sách).

Quy tắc quan trọng: chunk không vừa ngân sách hoặc đã ra ngoài phạm vi trang → **đóng hẳn
hướng đó lại**, không bỏ qua để lấy chunk xa hơn. Lý do: bỏ qua sẽ tạo **lỗ hổng** giữa đoạn
trích — nội dung nhảy cóc mà không có dấu hiệu gì, người đọc tưởng hai phần đứng liền nhau.
**Đoạn trích buộc phải liền mạch.**

Trả về dict có cả `noidung` (đã mở rộng) lẫn `doan_khop` (riêng chunk neo) — xem §7.3.

```python
    def truy_xuat(self, cau_hoi, top_k=None, nguon_cho_phep=None, lich_su=None) -> List[Dict]
```
Hàm truy xuất chính. Chín bước — xem §6 bước 1→7 để có bản kể chuyện đầy đủ.

Một chi tiết dễ bỏ qua: dòng đầu tiên là `self.diem_rerank_cao_nhat = None`. Nếu lượt này
không chạy rerank mà vẫn còn giá trị cũ, ngưỡng từ chối sẽ phán xét câu hỏi hiện tại bằng
điểm của **một câu hỏi khác** — sai âm thầm, rất khó lần ra. Có hẳn một test canh:
`test_diem_rerank_khong_sot_lai_giua_hai_luot`.

```python
    def _goi_llm_theo_luong(self, he_thong_prompt, prompt_nguoi_dung, bat_thinking, num_predict=None)
```
**Đường gọi LLM duy nhất trong hệ thống.** Bản không streaming chỉ là vòng lặp gom hết các
mảnh lại — cố ý làm vậy để hai chế độ không bao giờ trôi ra khỏi nhau về hành vi.

Ghi chép về tham số `think`, đã đo trực tiếp trên qwen3:4b + ollama 0.6.2, **trái với trực
giác**:

| Truyền gì | Kết quả |
|---|---|
| `think=True` | máy chủ tách suy luận sang `message["thinking"]`, content sạch |
| không truyền gì | y hệt `think=True` (mặc định của model biết suy luận) |
| `think=False` | **KHÔNG tắt suy luận**, mà tắt việc **TÁCH** nó ra — toàn bộ chuỗi suy luận đổ thẳng vào content và hiện nguyên si cho người dùng |

→ Khi không cần suy luận thì **bỏ hẳn tham số**, tuyệt đối không truyền `False`.

Mẹo chèn hậu tố `/no_think` của bản trước cũng đã bỏ: đo thực tế cho thấy nó **không hề tắt
suy luận** (model vẫn sinh ~15.000 ký tự) và không nhanh hơn đáng kể (43.7 s so với 47.0 s —
trong khoảng nhiễu).

Hàm con `_mo_luong(ts)` **lấy luôn mảnh đầu tiên** ngay tại đó. Bắt buộc: với `stream=True`,
client trả về một generator nên lỗi phía máy chủ chỉ nổ ra ở lần lặp **đầu tiên** chứ không
phải lúc gọi. Không chạm vào generator thì khối `try/except` không bao giờ bắt được lỗi.

Xử lý dự phòng cuối hàm: nếu model nhét **tất cả** vào phần suy luận và không viết câu trả
lời nào, hàm trả ra phần suy luận thay vì để lại một bong bóng chat trống trơn.

```python
    def _ghi_nhan_thong_ke_llm(self, manh, so_token_prompt, num_ctx, num_predict=None) -> None
```
Đọc bộ đếm token **thật** ở mảnh cuối luồng. Đây là **tuyến chứng minh** cho bug `num_ctx`:
`prompt_eval_count` là số token máy chủ thật sự đã nạp, đối chiếu được với ước lượng của ta.
Cảnh báo hai trường hợp: `prompt_eval_count ≥ num_ctx` (prompt bị cắt) và
`done_reason == "length"` (câu trả lời bị cắt cụt).

> *"Một lớp lỗi mà hệ thống KHÔNG THỂ tự phát hiện thì mọi kết luận rút ra từ nó đều đáng
> ngờ."* — câu này trong docstring tóm gọn triết lý của cả dự án.

```python
    def sinh_cau_tra_loi_theo_luong(self, cau_hoi, cac_chunk, ngu_canh_hoi_thoai="") -> Iterator
```
Chọn ngôn ngữ → chọn prompt → nén ngữ cảnh → ghép prompt → gọi LLM. Không có chunk nào thì
trả câu từ chối **mà không gọi LLM**.

Trong hàm này có một đoạn comment dài đáng đọc: **ngân sách sinh KHÔNG thích ứng, và đó là
một tính năng đã bị GỠ BỎ**. Bản trước hạ `num_predict` xuống 3000 cho câu hỏi "đơn giản".
Lập luận đó sai ở chỗ căn bản: `num_predict` giới hạn **suy luận + câu trả lời cộng lại**,
mà riêng chuỗi suy luận của qwen3 đã ngốn 2.000–4.000 token. Đo lại:

```
num_predict=3000   → sinh 2978 token, câu trả lời  569 ký tự (sát trần, ĐỨT giữa chừng)
num_predict=12000  → sinh 3948 token, câu trả lời 1012 ký tự (đủ)
```

Và khoản lợi vốn dĩ **bằng không**: `_tinh_num_ctx()` giữ chỗ
`min(OLLAMA_DU_PHONG_TOKEN_SINH=4000, num_predict)`, nên mọi giá trị từ 4000 trở lên cho ra
cùng một `num_ctx`. *"Tham số này chỉ 'có lợi' khi nó không an toàn."*

```python
    def sinh_cau_tra_loi(self, cau_hoi, cac_chunk) -> str
```
Bản gom-hết-rồi-trả-một-lần, dùng cho evaluation và test. Gọi lại đúng generator ở trên thay
vì gọi Ollama lần nữa — **để chế độ đo đạc chạy đúng một đường mã với chế độ người dùng thật.**

```python
    def hoi_dap_theo_luong(self, cau_hoi, top_k=None, nguon_cho_phep=None,
                           lich_su=None, doi_chieu=None) -> Iterator[Dict]
```
Chạy trọn luồng và **tường thuật từng chặng**. Năm loại sự kiện — xem §7.5.

Tách `truy_xuat_xong` thành sự kiện riêng là có chủ đích: nó tới sau ~2 giây, tức trước khi
LLM kịp viết chữ nào, nên giao diện hiện ngay được *"đã tìm N đoạn trong tài liệu X"*.

Đối chiếu chéo chạy **sau** khi câu trả lời đã hiện xong, và bị bỏ qua khi câu trả lời là
câu từ chối.

```python
    def hoi_dap(self, cau_hoi, ...) -> Dict
```
Bản đồng bộ: chạy generator trên tới sự kiện `"xong"` rồi trả `ket_qua`.

---

### 13.3. `rag/citation.py` — trích dẫn (308 dòng)

**Vai trò**: biến các số `[n]` mà LLM gắn thành danh sách nguồn hiển thị được, và cung cấp
phép đo "bám nguồn".

**Hằng số**: `DO_DAI_TRICH_DAN = 600` (ký tự).

```python
def _cac_so_tham_chieu(van_ban: str) -> List[int]
```
Mọi số đoạn trích được dẫn, **kể cả dạng gộp** `"[3,4,5]"`.

```python
def bo_so_trich_dan(van_ban: str) -> str
```
Bỏ các số `[n]` khỏi văn bản **để hiển thị**, giữ nguyên dữ liệu gốc. Có xử lý riêng cho
ngoặc **đang dở** khi stream (`_MAU_NGOAC_DANG_DO`) — nếu không thì người dùng thấy `"[3"`
nhấp nháy rồi biến mất.

> Đây là một quyết định UI đáng bàn: các số `[n]` **vẫn được LLM sinh ra và vẫn được hệ
> thống đọc** — đó chính là căn cứ để biết câu trả lời dùng nguồn nào và để chấm Citation
> accuracy — nhưng với người đọc thì chúng là **thứ tự đoạn trích trong prompt**, một thứ tự
> họ không nhìn thấy nên cũng không tra ngược được. Hiện ra chỉ thêm nhiễu.

```python
def dinh_dang_trich_dan(cac_chunk: List[Dict]) -> List[Dict]
```
Chuẩn hoá thành list trích dẫn. **Ánh xạ 1-1 và GIỮ NGUYÊN THỨ TỰ** — phần tử thứ *i* tương
ứng đúng với đoạn được đánh số `[i+1]` trong prompt. Vì vậy hàm này **không được phép loại
trùng hay sắp xếp lại**.

`doan_trich` lấy từ `doan_khop` (chunk thật sự khớp), **không cắt từ đầu vùng ngữ cảnh đã mở
rộng**. Bản trước cắt 400 ký tự đầu của cả trang đã gộp, nên với trang ~2000 ký tự thì đoạn
hiển thị gần như luôn là phần đầu trang — đúng triệu chứng "trích dẫn không chính xác".

```python
def loc_theo_tham_chieu(cac_chunk, cau_tra_loi, so_toi_da=None) -> List[Dict]
```
**Hàm quan trọng nhất của file.** Ba lớp, lớp sau chỉ dùng khi lớp trước không cho kết quả:

1. Số đoạn trích `[n]` — dạng chuẩn mà system prompt bắt buộc.
2. Số trang được nhắc trong câu trả lời (`"theo Slide 109"`) — model nhỏ đôi khi vẫn làm vậy
   dù đã bị cấm.
3. Nguồn liên quan nhất, **đánh dấu `la_suy_doan = True`**.

Câu **từ chối** thì không hiển thị nguồn nào: vừa nói "không tìm thấy thông tin" vừa chỉ vào
một trang cụ thể là tự mâu thuẫn.

Các đoạn cùng `(nguồn, trang)` được **gộp lại** khi hiển thị nhưng **giữ lại mọi số hiệu**
trỏ về đó.

> Con số đáng chú ý trong docstring: đo trên bộ 29 câu, **4 câu trả lời thật không gắn số
> nào**; đo lặp lại cùng một câu 4 lần thì **tỉ lệ tuân thủ chỉ 50%**. Tức lớp 3 không phải
> trường hợp hiếm — và đó là lý do phải đánh dấu nó rõ ràng thay vì trình bày như căn cứ thật.

```python
def do_bam_ngu_canh(cau_tra_loi, ngu_canh, so_tu_moi_cum=4) -> float
```
Tỉ lệ **cụm 4 từ liên tiếp** của câu trả lời xuất hiện **nguyên văn** trong ngữ cảnh. Tất
định, không gọi model, chạy trong mili giây — nên dùng được ngay ở luồng trả lời thật.

So sánh **sau khi bỏ hết khoảng trắng ở cả hai bên**, nên ngữ cảnh trích từ PDF bị dính chữ
vẫn khớp được với câu trả lời viết đúng chuẩn. (Có test riêng:
`test_bam_ngu_canh_van_bat_duoc_du_ngu_canh_bi_dinh_chu`.)

**Cách đọc con số này cho đúng** — được nhắc lại ở ba chỗ trong dự án:
- Điểm **cao** = bằng chứng mạnh rằng câu trả lời **không bịa**.
- Điểm **thấp** = **không chứng minh được gì**. Diễn đạt lại bằng lời của mình cũng cho điểm
  thấp.
- → **Tuyệt đối không dùng nó để từ chối một câu trả lời**: làm thế sẽ giết đúng những câu
  trả lời tốt nhất.

```python
def cau_theo_trich_dan(cau_tra_loi: str) -> Dict[int, List[str]]
```
Ghép mỗi số `[n]` với những **câu** trong câu trả lời đã dẫn nó. Dùng để kiểm chứng ở tầng
đánh giá: biết "LLM dẫn đoạn [2]" là chưa đủ, phải biết nó dẫn đoạn [2] để chống lưng cho
**ý nào**.

Tách câu theo dấu chấm/xuống dòng **và theo cả gạch đầu dòng**, vì câu trả lời hay ở dạng
danh sách. Bỏ qua những câu chỉ nói **về** đoạn trích để **loại** nó ra (`"các phần [3][4]
không liên quan"`) — đó không phải một khẳng định đang dựa vào đoạn [3].

```python
def format_text_trich_dan(cac_chunk: List[Dict]) -> str
```
Chuỗi text hiển thị nhanh cho script/terminal.

---

### 13.4. `rag/doi_chieu_nguon.py` — phát hiện mâu thuẫn (369 dòng)

**Vấn đề**: giáo trình cũ ghi "ba đặc điểm", slide mới ghi "năm đặc điểm". Người đọc mở
từng file thì mỗi file đều tự nhất quán. Chỉ khi đặt cạnh nhau mới lộ — mà đặt cạnh nhau
đúng là việc hệ thống vừa làm khi gom đoạn trích vào một câu trả lời.

**Kiến trúc hai tầng** — tầng 1 tất định để **không tốn lượt gọi LLM** cho đại đa số lượt hỏi.

#### Tầng 1 — sàng lọc tất định

```python
def _tap_so(van_ban: str) -> set
```
Tập các số trong đoạn, **đã chuẩn hoá**. Quan trọng: `_SO_BANG_CHU` quy `"hai"→"2"`,
`"ba"→"3"`, `"năm"→"5"`, `"three"→"3"`… Không có bước này thì "ba đặc điểm" và "3 đặc điểm"
bị coi là mâu thuẫn (báo động giả), còn "ba đặc điểm" và "năm đặc điểm" thì không bắt được.

Chú ý: `"một"` **không** được tính là số — nó quá phổ biến với vai trò mạo từ.

```python
def co_dau_hieu_bat_dong(doan_a: str, doan_b: str) -> bool
```
Hai đoạn có dấu hiệu **bề mặt** của việc nói khác nhau không? Hai tín hiệu:
- **Khác số** (sau khi chuẩn hoá).
- **Lệch phủ định** — một đoạn có `_PHU_DINH` ("không", "chẳng", "chưa", "not", "never"…),
  đoạn kia không.

```python
def cac_cap_dang_ngo(cac_doan, vector_doan) -> List[tuple]
```
Chọn các cặp `(i, j)` đáng đem đi chấm, sắp theo cosine giảm dần. Ba điều kiện:
- **Khác nguồn** (cùng một tài liệu tự mâu thuẫn không phải việc của hệ thống này).
- **Cosine ≥ `NGUONG_COSINE_DOI_CHIEU` = 0.88** — tức đang nói **cùng một chuyện**. Hai đoạn
  khác chủ đề thì "khác số" là chuyện bình thường.
- Có `co_dau_hieu_bat_dong`.

Trần `SO_CAP_DOI_CHIEU_TOI_DA = 3` chặn bùng nổ chi phí (với `TOP_K = 4` thì có tối đa 6 cặp).

Nếu embedding hỏng thì **bỏ điều kiện cùng chủ đề** chứ không dừng hẳn — giảm cấp có kiểm
soát.

#### Tầng 2 — LLM chấm

```python
PROMPT_DOI_CHIEU     # định nghĩa rõ MÂU THUẪN là gì, và BỔ SUNG NHAU thì KHÔNG phải mâu thuẫn
_SCHEMA_MAU_THUAN    # JSON schema: {phan_tich, muc_do (0..1), co_mau_thuan, noi_dung_xung_dot}
def _cham_mot_cap(client, a: Dict, b: Dict) -> Optional[Dict]
```
Chấm một cặp. Trả `None` nếu không chấm được (lỗi, JSON hỏng, điểm ngoài thang) — **loại
bỏ, không kẹp về biên**, vì một điểm ngoài thang nghĩa là model không hiểu thang đo, giá trị
nó nói ra không đáng tin.

```python
def tim_mau_thuan(cac_doan, embedding_service=None, client=None) -> List[Dict]
```
Điểm vào. Chấm `SO_LAN_CHAM_MAU_THUAN = 2` lần cho mỗi cặp và **chỉ báo khi cả hai lần đồng
ý** — thoát sớm nếu lần chấm đầu nói "không". Ngưỡng `NGUONG_MAU_THUAN = 0.6`.

Trả `List[{nguon_a, trang_a, nguon_b, trang_b, noi_dung_xung_dot}]`. Giải thích lấy từ
**phần phân tích của chính model**, không tự chế lại.

Mọi lỗi (Ollama chết, JSON hỏng) đều **không làm hỏng lượt hỏi** — bước này chỉ thêm cảnh
báo, không đổi câu trả lời.

> **Phần khó nhất của tính năng này không phải bắt được mâu thuẫn, mà là IM LẶNG ĐÚNG.**
> Bộ kiểm định 7 ca có 3 ca "phải im lặng", và kết quả đo: 7/7 đúng, ổn định qua 3 lần chạy,
> **không có báo động giả nào**; tầng lọc tất định chặn 2/3 ca im lặng trước khi tốn lượt gọi
> LLM nào.

---

## 14. `rag/tai_nguyen_gpu.py` — quản lý phần cứng (395 dòng)

**Vấn đề**: bốn model cộng lại (~9 GB) vượt VRAM của card phổ thông. Nhưng chúng **không cần
cùng lúc**: model vision chỉ dùng lúc Ingestion, LLM và reranker chỉ dùng lúc Query.

```python
def co_cuda() -> bool
```
Máy có GPU dùng được cho PyTorch không. Kết quả được nhớ trong `_co_cuda` (dò một lần).

```python
def thiet_bi(vai_tro: str) -> str      # "cuda" | "cpu"
```
Thiết bị nên dùng cho một **vai trò** (`"embedding"` hoặc `"rerank"`). Đây là điểm quyết
định trung tâm: trên card chật, **embedding cố ý nằm ở CPU** để nhường VRAM cho reranker và
LLM — vì embedding chỉ phải encode một câu hỏi ngắn mỗi lượt, còn reranker phải chấm 30 cặp.

Cấu hình `THIET_BI_EMBEDDING` / `THIET_BI_RERANK` ép được, và khi ép thì hàm **không tự dò
đè lên**.

```python
def vram() -> Optional[Tuple[float, float, float]]   # (đang giữ, tổng, còn trống) GB
def tong_vram_gb() -> float
def vram_con_trong_gb() -> float
def so_nhan_cpu() -> int
def mo_ta_phan_cung() -> str
```
Nhóm hàm đọc thông tin. Tất cả **trả giá trị an toàn trên máy không GPU** (0.0 / None), không
ném lỗi — có hẳn một test canh: `test_cac_ham_gpu_khong_nem_loi_tren_may_khong_gpu`.

```python
def kich_thuoc_lo_embedding() -> int
```
Batch size encode, suy từ **VRAM còn trống** (không phải tổng VRAM — con số quan trọng là
chỗ trống thật sự tại thời điểm đó). Trên CPU thì giữ nguyên giá trị cấu hình. Không bao giờ
vượt trần `EMBEDDING_BATCH_SIZE`.

```python
def so_worker_vision() -> int
```
Số luồng gọi vision/OCR song song, suy từ `min(số nhân CPU, VRAM còn trống /
VRAM_MOI_WORKER_VISION_GB)`. Luôn ít nhất 1, không vượt trần cấu hình.

```python
def ghi_log_vram(nhan: str) -> None
def don_bo_nho_cuda() -> None
def nha_model_ollama(ten_model: str, client=None) -> bool
```
`nha_model_ollama` bảo Ollama nhả một model khỏi VRAM **ngay** (gửi request với
`keep_alive=0`), thay vì đợi hết thời gian giữ mặc định 5 phút. Thất bại thì **âm thầm**
trả `False` — Ollama không chạy không phải lỗi ở đây.

```python
def du_cho_giu_embedding_tren_gpu() -> bool
def bat_dau_ingestion(embedding_service=None) -> None
def ket_thuc_ingestion(client=None, embedding_service=None) -> None
```
**Hai hàm ranh giới giai đoạn** — được `app.py` gọi ở đầu và cuối `xay_dung_lai_index()`:

- `bat_dau_ingestion`: đưa embedding **trở lại GPU** nếu máy có (lần build trước có thể đã
  đẩy nó xuống CPU).
- `ket_thuc_ingestion`: nhả model vision khỏi VRAM; nếu card nhỏ (< `VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB`
  = 10 GB) thì đẩy embedding xuống CPU; dọn bộ nhớ CUDA.

> Câu giải thích trong `app.py` rất rõ vì sao bước này đáng có: *"Không nhả thì Ollama giữ
> nó thêm 5 phút nữa — đúng 5 phút mà người dùng vừa build xong và bắt đầu hỏi, tức lúc VRAM
> đang cần cho LLM và reranker."*
---

## 15. Thư mục `evaluation/` — đo đạc

**Triết lý của thư mục này**, và cũng là điểm đáng học nhất về mặt phương pháp: dự án không
chỉ đo **hệ thống**, mà còn đo **chính các thước đo**. Ba file `kiem_dinh_*.py` tồn tại để
trả lời câu hỏi *"con số này có đáng tin không"* trước khi dùng nó để kết luận.

```
Đo hệ thống          Đo thước đo                Đo hạ tầng
─────────────        ─────────────              ──────────
run_evaluation.py    kiem_dinh_judge.py         do_quy_mo_index.py
metrics.py           kiem_dinh_doi_chieu.py     do_worker_gpu.py
tao_tai_lieu_mau.py  kiem_dinh_viet_lai.py      do_dau_cuoi.py
                     do_nguong_rerank.py
```

---

### 15.1. `evaluation/metrics.py` — các độ đo (424 dòng)

#### Nhóm độ đo TRUY XUẤT (tất định, không gọi LLM)

```python
def _khoa(muc: Dict) -> Tuple[str, int]
```
Khoá định danh một trang: `(tên file, số trang)` — **phải kết hợp cả hai**, vì trang 7 của
hai tài liệu khác nhau là hai chỗ khác nhau.

```python
def precision_tai_k(cac_chunk_truy_xuat, cac_trang_dung) -> float
```
Trong K chunk truy xuất được, bao nhiêu tỉ lệ là đúng.

```python
def recall_tai_k(cac_chunk_truy_xuat, cac_trang_dung) -> float
```
Trong toàn bộ trang đúng đáp án, bao nhiêu tỉ lệ được tìm thấy.

```python
def thu_hang_dung_dau_tien(cac_chunk_truy_xuat, cac_trang_dung) -> int
def nghich_dao_thu_hang(cac_chunk_truy_xuat, cac_trang_dung) -> float
```
Thứ hạng (1-based) của đoạn đúng đầu tiên, và `1/thứ_hạng` — thành phần của **MRR** (Mean
Reciprocal Rank).

> **Cảnh báo quan trọng về Recall@K trong dự án này**: từ khi bật `MO_RONG_QUA_RANH_GIOI_TRANG`,
> `Recall@K` chỉ đếm **trang neo**. Khi một đoạn trích nuốt sang trang liền kề thì trang đó
> không còn được neo, dù nội dung vẫn nằm nguyên trong ngữ cảnh gửi cho LLM. Đo bằng chỉ số
> không bị hiện vật này (*Recall phủ*) cho kết quả ngược lại:
>
> | | P@K | Recall@K (neo) | Recall phủ |
> |---|---|---|---|
> | TẮT mở rộng xuyên trang | 0.580 | 0.945 | 0.945 |
> | BẬT mở rộng xuyên trang | 0.540 | 0.875 | **0.959** |
>
> Đây là một bài học phương pháp luận đáng ghi nhớ: **khi số liệu tụt, hãy kiểm tra thước đo
> trước khi kết luận hệ thống kém đi.**

#### Nhóm độ đo SINH (gọi LLM làm giám khảo)

```python
PROMPT_FAITHFULNESS        # chấm độ trung thực của câu trả lời so với ngữ cảnh
PROMPT_ANSWER_RELEVANCE    # chấm độ liên quan giữa câu trả lời và câu hỏi
PROMPT_CAN_CU_TRICH_DAN    # đoạn trích này có THẬT SỰ chứng minh cho khẳng định này không
_SCHEMA_DIEM_SO            # {"diem": 0..1, "ly_do": str}
```

```python
def _diem_hop_le(diem: float) -> bool
def _goi_judge_kep(prompt: str) -> Dict
def _goi_judge_on_dinh(prompt: str, so_lan: int) -> Dict
def _goi_judge(prompt: str) -> Dict
```
`_goi_judge_on_dinh` chấm `SO_LAN_CHAM_FAITHFULNESS = 3` lần rồi lấy **trung vị**, và trả
kèm **biên độ dao động** giữa các lần. Biên độ khác 0 nghĩa là giám khảo **tự mâu thuẫn với
chính nó** — con số này thuộc về **độ tin cậy của thước đo** và được ghi ra CSV.

Điểm ngoài thang `[0,1]` bị **loại khỏi phép lấy trung vị**, không phải kẹp về biên: điểm
ngoài thang nghĩa là model không hiểu thang đo.

```python
def do_bam_ngu_canh_thap_nhat(cau_tra_loi: str, ngu_canh: str) -> float
```
Mức bám ngữ cảnh của **câu tệ nhất** trong câu trả lời, thay vì trung bình cả bài. Dùng
trung bình sẽ để lọt câu trả lời đúng-một-nửa vào diện "đáng ngờ" trong khi giám khảo chấm
thấp là hoàn toàn đúng.

```python
def faithfulness(cau_tra_loi: str, cac_chunk_nguon: List[Dict]) -> Dict
```
**Hàm đáng học nhất của file.** Faithfulness do LLM chấm, **kèm một cờ tự nghi ngờ chính
điểm số đó**:

```
dang_ngo  =  giám khảo chấm THẤP  ĐỒNG THỜI  câu trả lời chép gần nguyên văn ngữ cảnh
```

Hai điều đó **không thể cùng đúng**. Khi cờ bật, `run_evaluation.py` đánh dấu câu đó và báo
riêng trung bình đã loại các câu đáng ngờ.

> Vì sao cần cờ: LLM-as-judge với model 4B là một thước đo **có sai số hệ thống**, không phải
> một con số khách quan. Ca hỏng đã gặp thật: 2 câu về sách Bishop bị chấm 0.0 trong khi đọc
> trực tiếp thì cả hai đều đúng — giám khảo không đối chiếu được câu trả lời sạch với ngữ
> cảnh **bị dính chữ**, nên kết luận là "bịa". Điểm 0.33 của cả nhóm là **lỗi của thước đo**.
>
> Nguy hiểm ở chỗ nó không có triệu chứng gì: nếu tin con số đó, người làm đồ án sẽ đi sửa
> prompt hoặc đổi model để "cải thiện Faithfulness" — **tối ưu vào một cái sai**.

```python
def answer_relevance(cau_hoi: str, cau_tra_loi: str) -> Dict
```

```python
def do_chinh_xac_trich_dan(cau_tra_loi: str, cac_chunk_nguon: List[Dict]) -> Dict
```
Khác Faithfulness ở chỗ: Faithfulness hỏi *"toàn bộ câu trả lời có bịa không"* trên **toàn
bộ** ngữ cảnh; metric này soi **từng cặp (ý, đoạn được dẫn cho ý đó)**. Một câu trả lời có
thể đạt Faithfulness cao mà trích dẫn vẫn sai chỗ.

**Phân biệt hai kiểu "không dẫn nguồn nào"** — chi tiết quan trọng:

| Trường hợp | Điểm | Lý do |
|---|---|---|
| **Câu từ chối** không dẫn nguồn | `None` (loại khỏi trung bình) | không dẫn nguồn là hành vi **đúng** ở đây |
| **Câu trả lời thật** không dẫn nguồn | `0.0` | đây là **lỗi trích dẫn nặng nhất**: người đọc mất đường kiểm chứng |
| Không truy xuất được đoạn nào | `None` | "trích dẫn có đúng chỗ không" là câu hỏi vô nghĩa |

> Bản trước trả `None` cho cả hai trường hợp đầu, và cái giá rất cụ thể: một lần sửa system
> prompt khiến model bỏ hẳn trích dẫn ở 3 câu, nhưng vì cả 3 rơi vào diện "bị loại khỏi
> trung bình" nên điểm Citation gần như không đổi — **lỗi đi lọt qua thước đo**. Đây là kiểu
> hỏng âm thầm ở phía ngược lại: thước đo quá **dễ dãi**.
>
> Hệ quả cần nhớ khi đọc báo cáo: Citation 0.76 → 0.61 **không phải hệ thống kém đi**, mà là
> thước đo nghiêm hơn. Đo cùng lần chạy theo cách cũ cho 0.72.

---

### 15.2. `evaluation/run_evaluation.py` — chạy đánh giá (505 dòng)

**Cách chạy**:

```bash
python evaluation/run_evaluation.py                # đầy đủ (có gọi LLM, ~80 phút)
python evaluation/run_evaluation.py --nhanh        # CHỈ đo truy xuất, TẤT ĐỊNH, vài phút
python evaluation/run_evaluation.py --nhanh 5      # chỉ 5 câu đầu
python evaluation/run_evaluation.py --held-out     # đo trên bộ HELD-OUT
python evaluation/run_evaluation.py --khoang-cach  # đo CẢ HAI bộ + in mức overfit
```

**Hàm**:

```python
def nap_bo_cau_hoi(duong_dan: Path = None) -> List[Dict]
def _trung_binh(cac_muc: List[Dict], cot: str) -> float
```
`_trung_binh` **bỏ qua mục thiếu cột** — CSV cũ có thể chưa có cột mới.

```python
def _dang_ngo(kq: Dict) -> bool
def _in_bang_ket_qua(ket_qua_tung_cau: List[Dict]) -> None
def _in_bang_theo_nhom(ket_qua_tung_cau, khoa: str, nhan: str) -> None
```
In điểm trung bình tách theo `loai_tai_lieu` (dài tiếng Anh / slide tiếng Việt / biểu mẫu có
bảng…) và `loai_cau_hoi` (đọc bảng / kiểm chứng / chéo ngôn ngữ / từ chối lạc đề). Tách nhóm
là cần thiết: trung bình chung che mất việc hệ thống mạnh ở nhóm này và yếu ở nhóm kia.

```python
def doc_csv_ket_qua(duong_dan: Path) -> List[Dict]
def so_sanh_voi_ban_truoc(csv_moi=None, csv_cu=None) -> None
```
In chênh lệch từng metric giữa **hai lần chạy** — biến câu hỏi *"thay đổi vừa rồi có thực sự
cải thiện không"* thành một con số thay vì một cảm nhận. Có kiểm tra: bỏ qua so sánh khi hai
lần chạy dùng **khác bộ câu hỏi**.

```python
def _xuat_csv(ket_qua_tung_cau, duong_dan=None) -> None
```
Mỗi câu ghi ra ~15 cột, trong đó có ba cột về **độ tin cậy của thước đo**:
`faithfulness_dang_ngo`, `faithfulness_dao_dong`, `faithfulness_bam_ngu_canh`.

```python
def chay_danh_gia_nhanh(gioi_han=None, duong_dan_cau_hoi=None) -> Optional[List[Dict]]
```
Đo **chỉ phần truy xuất** (Precision@K / Recall@K / MRR), **không gọi LLM lần nào** → **tất
định**. Đây là chế độ nên dùng khi tinh chỉnh tham số truy xuất: chênh lệch là chênh lệch
thật, không phải dao động của model.

```python
def do_khoang_cach_held_out(gioi_han=None) -> None
```
Chạy **cả hai** bộ câu hỏi rồi in chênh lệch — con số đo mức **overfit** của hệ thống.

```python
def chay_danh_gia(gioi_han=None, duong_dan_cau_hoi=None) -> None
```
Đánh giá đầy đủ. Tắt hẳn bước đối chiếu chéo (`doi_chieu=False`) vì nó không đổi câu trả lời
(chỉ thêm cảnh báo), nên để bật sẽ kéo dài một lần đánh giá vốn đã 60–90 phút mà không làm
thay đổi metric nào.

**Định dạng bộ câu hỏi** (`test_questions.json`, 29 câu):

```json
{
  "cau_hoi": "What is the bias-variance decomposition?",
  "cac_trang_dung": [
    {"nguon": "Bishop-....pdf", "trang": 167},
    {"nguon": "Bishop-....pdf", "trang": 169}
  ],
  "dap_an_mau": "Phân rã sai số kỳ vọng thành bias và variance.",
  "loai_tai_lieu": "dai_tieng_anh",
  "loai_cau_hoi": "truy_xuat"
}
```

---

### 15.3. `evaluation/tao_tai_lieu_mau.py` — sinh tài liệu mẫu (475 dòng)

**Vai trò**: sinh 6 tài liệu **độc lập hoàn toàn** với corpus thật, để kiểm chứng chống
overfitting. Chạy: `python evaluation/tao_tai_lieu_mau.py`.

```python
def _sinh_muc_tuong_tu()
def _ve_so_do_quy_trinh(duong_dan: Path) -> Path
def _ve_bieu_do_cot(duong_dan: Path) -> Path
def tao_docx_dai(duong_dan: Path) -> None
def tao_docx_nhieu_muc_tuong_tu(duong_dan: Path) -> None
def tao_docx_co_bang(duong_dan: Path, anh_so_do: Path) -> None
def tao_pptx_ngan(duong_dan: Path) -> None
def tao_pptx_bang_anh(duong_dan: Path, anh_so_do: Path, anh_bieu_do: Path) -> None
def tao_pdf_hon_hop(duong_dan: Path, anh_bieu_do: Path) -> None
def main() -> None
```

Điểm đáng chú ý: mỗi tài liệu được thiết kế để **bẫy một cơ chế cụ thể**:

| Hàm | Bẫy gì |
|---|---|
| `tao_docx_dai` | tài liệu dài, nhiều cấp tiêu đề → chunking có tôn trọng ranh giới mục không |
| `tao_docx_nhieu_muc_tuong_tu` | 20 điều khoản **gần như trùng nhau** → nhiễu có kiểm soát |
| `tao_docx_co_bang` | bảng xen văn xuôi + **bảng lồng trong ô** + ảnh |
| `tao_pptx_ngan` | tài liệu ngắn, thưa chữ — cực đối lập với giáo trình |
| `tao_pptx_bang_anh` | ảnh + bảng **nằm trong group shape** |
| `tao_pdf_hon_hop` | tiêu đề phân biệt **chỉ bằng cỡ chữ**, bảng kẻ khung, ảnh kèm caption |
| `_ve_bieu_do_cot` | số liệu **chỉ có trong hình**, không lặp lại ở text |

Kết quả đo trên bộ này: **24/24 câu có đáp án đạt Recall@K 1.00**, MRR 0.95, 22/24 đúng ở
hạng 1, 2 câu lạc đề bị chặn đúng, **0 chunk vượt giới hạn model**.

---

### 15.4. Ba script `kiem_dinh_*.py` — đo chính các thước đo

```python
# kiem_dinh_judge.py (195 dòng)
CAC_CA_KIEM_DINH      # ca có đáp án biết trước
def _chay_mot_ca(cau_tra_loi: str, ngu_canh: str) -> dict
def main() -> None
```
Đo độ tin cậy của **thước đo Faithfulness**, không phải của hệ thống RAG. Ba ngữ cảnh mẫu
được dựng sẵn, trong đó có `_NGU_CANH_DINH_CHU` — bản **cố ý bị dính chữ** để tái hiện đúng
ca đã khiến giám khảo chấm sai.

`_chay_mot_ca` gọi **đúng hàm `faithfulness()`** mà `run_evaluation.py` dùng, không viết lại
logic — nếu không thì bài kiểm định đo một thứ khác với thứ đang chạy thật.

```python
# kiem_dinh_doi_chieu.py (187 dòng)
def _doan(nguon, trang, noidung)
CAC_CA_KIEM_DINH      # 7 ca, trong đó 3 ca PHẢI IM LẶNG
def main() -> None
```
Đo cơ chế phát hiện mâu thuẫn. Các ca gồm: mâu thuẫn khác số lượng (số viết bằng chữ), lệch
phủ định, và **ba ca bổ sung cho nhau — không được báo động**.

```python
# kiem_dinh_viet_lai.py (255 dòng)
_HOI_THOAI_LUAT / _HOI_THOAI_ML / _HOI_THOAI_THU_VIEN     # ba kịch bản hội thoại
CAC_CA                                                     # ca có nhãn (là nối tiếp / không)
def _khong_dau(s: str) -> str
def main() -> None
CAU_NOI_TIEP_TRUNG_TINH
def _do_anh_huong_truy_xuat(cau_hoi_goc: str) -> None
```
Hai phần: (1) đo **tầng nhận diện** trên bộ ca có nhãn — đạt 10/10; (2)
`_do_anh_huong_truy_xuat` đo **ảnh hưởng thật lên truy xuất**, lấy chính kết quả truy xuất
của câu hỏi **đầy đủ** làm chuẩn vàng. Nhờ cách này, phép đo chạy được trên **bất kỳ corpus
nào**, không cần gán nhãn tay. Kết quả: **1/16 → 16/16** trùng chuẩn vàng.

---

### 15.5. Bốn script `do_*.py` — đo hạ tầng

```python
# do_nguong_rerank.py (144 dòng)
TRONG_PHAM_VI_VI / TRONG_PHAM_VI_EN / LAC_DE_VI            # ba nhóm câu hỏi
def _diem_cao_nhat(pipeline, reranker, cau_hoi) -> (cosine, rerank)
def _in_nhom(nhan, cac_diem)
def chay_do()
```
Trả lời câu hỏi *"điểm rerank có tách được câu trong phạm vi khỏi câu lạc đề không"*. Đây
chính là script đã cho ra con số biện minh cho `NGUONG_DIEM_RERANK_TOI_THIEU = 0.001`, và
cũng cho thấy **cosine KHÔNG làm được việc này**.

```python
# do_quy_mo_index.py (271 dòng)
K_TIM_KIEM, NGAN_SACH_TIM_KIEM_MS = 200.0, SO_CAU_HOI_DO = 30
def _so_co_dau_cham(n: int) -> str                          # 123456 -> '123.456'
def _vector_gia_lap(so_luong, so_chieu, seed, so_cum=200, do_tuong_dong=0.8) -> np.ndarray
def _do_do_tre(index, cac_cau_hoi) -> (p50, p95)
def _recall_so_voi_flat(index, cac_cau_hoi, dap_an_flat) -> float
def do_mot_kich_thuoc(so_chunk: int, so_chieu: int) -> dict
def _ket_luan(cac_ket_qua: list, so_chieu: int) -> None
def main() -> None
```
Đo `IndexFlatIP` chịu được corpus tới cỡ nào, và chuyển sang IVF/HNSW thì được/mất gì.
Điểm đáng học: `_vector_gia_lap` sinh vector **có gom cụm theo chủ đề** (`so_cum=200`,
`do_tuong_dong=0.8`) chứ không phải ngẫu nhiên đều — vector ngẫu nhiên đều sẽ cho kết quả
quá lạc quan cho các index gần đúng.

```python
# do_worker_gpu.py (195 dòng)
class TheoDoiGpu:
    def __init__(self, chu_ky_giay: float = 0.25)
    def _lay_mau(self) -> None                    # gọi nvidia-smi
    def __enter__ / __exit__
    def tom_tat(self)                             # (util TB %, util đỉnh %, VRAM đỉnh GB)
def _tim_trang_can_ocr(duong_dan, so_trang_can, bo_qua_dau=300)
def chay(duong_dan: Path, cac_muc_worker, so_trang: int) -> None
def main() -> None
```
Đo số worker OCR/Vision tối ưu. `_tim_trang_can_ocr` tìm những trang **thật sự cần OCR** —
để phép đo chạy trên đúng loại việc nó mô tả, chứ không phải trên trang text bình thường.

```python
# do_dau_cuoi.py (268 dòng)
CAU_HOI_MAC_DINH
def _tom_tat(cac_giay)
def _bang_profiling(tieu_de: str) -> None
def do_ingestion(cac_file, thu_muc_lam_viec: Path, embedding_service)
def do_query(pipeline, cac_cau_hoi, goi_llm: bool)
def _in_tong_ket_query(dong, goi_llm: bool) -> None
def main() -> None
```
Đo **đầu-cuối**: từ lúc đưa tài liệu vào tới lúc hỏi được, và từ lúc hỏi tới lúc có câu trả
lời. `do_ingestion` đo **hai lần**: với cache rỗng và với cache đầy — đó là cách duy nhất
nói được cache tiết kiệm bao nhiêu.

---

## 16. Thư mục `tests/` — kiểm thử

**22 file test + 1 `conftest.py`, tổng 335 test.** Chạy:

```bash
pytest                      # tất cả (dùng trước khi chốt thay đổi)
pytest -m "not slow"        # bỏ qua test phải nạp model thật
pytest tests/test_pipeline.py -v
```

### 16.1. `conftest.py` — một fixture, một mục đích

```python
@pytest.fixture(autouse=True, scope="session")
def khong_ghi_vao_thu_muc_du_an(tmp_path_factory)
```
**Mục đích duy nhất: không một test nào được ghi vào thư mục dữ liệu THẬT của dự án.**

Cần thiết từ khi có bộ nhớ đệm: một test dựng file DOCX tạm rồi gọi `doc_thu_muc()` sẽ để
lại rác trong `data/cache/` và `data/images/`. Đây là loại lỗi **không làm test đỏ** nên
không ai phát hiện qua CI.

Hai chi tiết kỹ thuật: gán thẳng thay vì `monkeypatch` (fixture phải ở phạm vi **session**,
mà `monkeypatch` chỉ có ở phạm vi function); và các kho trong `bo_nho_dem` phải được **tạo
lại** chứ không chỉ đổi `config.CACHE_DIR`, vì chúng chốt đường dẫn ngay lúc khởi tạo ở mức
module.

### 16.2. Bảng tra 22 file test

| File | Test | Kiểm gì |
|---|---:|---|
| `test_pipeline.py` | 34 | **Lõi luồng Query**: dựng đoạn trích, mở rộng xuyên trang, trần đoạn/trang, ba tầng ngưỡng, nhận diện câu kiểm chứng, nhận diện ngôn ngữ, nối liền mạch |
| `test_toi_uu_ingestion.py` | 28 | Đọc một-lượt, hiệu chỉnh `x_tolerance`, cache 4 tầng, lọc ảnh logo, sổ băm sống qua lưu/nạp |
| `test_citation_grounding.py` | 26 | Ghép câu ↔ số trích dẫn, chấm căn cứ từng cặp, cờ `la_suy_doan`, bỏ số khi hiển thị |
| `test_tai_nguyen_gpu.py` | 24 | Chọn thiết bị theo vai trò, batch theo VRAM, số worker, nhả model, ranh giới giai đoạn |
| `test_doi_chieu_nguon.py` | 23 | Chuẩn hoá số (kể cả số viết bằng chữ), lệch phủ định, trần số cặp, thoát sớm, Ollama chết không làm hỏng lượt hỏi |
| `test_tiep_noi_hoi_thoai.py` | 21 | Nhận diện câu nối tiếp, ghép ngữ cảnh, **đường LLM mặc định TẮT**, bốn lớp loại bản viết lại xấu |
| `test_do_tin_cay_thuoc_do.py` | 18 | **Thước đo tự kiểm tra chính nó**: phát hiện dính chữ, cờ đáng ngờ, cắt bảng lớn |
| `test_document_loader.py` | 18 | Bảng → Markdown, ô gộp, đệ quy group shape, nhận diện tiêu đề |
| `test_vision_caption.py` | 16 | Khớp tên model, thiếu model không làm hỏng build, một ảnh lỗi không hỏng cả lô, **prompt OCR cấm dịch** |
| `test_khai_quat_tai_lieu.py` | 15 | **Kiểm theo HÌNH DẠNG, không theo tài liệu**: bảng chỉ có tiêu đề, bảng 8 cột, ô chứa nguyên bài văn, trang hai cột |
| `test_ngan_sach_thich_ung.py` | 15 | Nhận diện độ phức tạp, `num_ctx` không bao giờ thấp hơn cấu hình, nén ngữ cảnh không sửa danh sách gốc |
| `test_streaming.py` | 12 | Thẻ `<think>` bị cắt đôi giữa hai mảnh, hai chế độ cho **cùng một** câu trả lời, đo độ trễ |
| `test_chunking.py` | 11 | Chunk không vượt giới hạn model, bảng được phép dài hơn văn xuôi, `vi_tri` tăng dần |
| `test_evaluation_harness.py` | 11 | Khung đo: trung bình bỏ ô thiếu, **bộ held-out không dùng chung tài liệu nào với in-sample** |
| `test_giao_dien.py` | 10 | **Chạy app thật bằng `streamlit.testing.AppTest`**: đặt câu hỏi không gọi LLM ở nhịp đầu, lỗi không làm treo giao diện |
| `test_tai_lieu_scan.py` | 10 | Nhận ra trang cần OCR, ảnh phủ cả trang scan, ảnh không chú thích bị loại, file hỏng không sập build |
| `test_ket_noi_ollama.py` | 8 | Lỗi kết nối thành lỗi riêng **có hướng dẫn**, truy xuất vẫn chạy khi Ollama chết |
| `test_lexical_search.py` | 8 | Tách từ sinh cả âm tiết và cặp âm tiết, BM25 cứu hộ đưa được đoạn từ khoá hiếm vào tập rerank |
| `test_index_tang_dan.py` | 7 | **Đầu-cuối qua đúng nút "Đọc tài liệu"**: thêm 1 tài liệu chỉ xử lý tài liệu đó, bấm hai lần không xử lý lại gì |
| `test_ngan_sach_ngu_canh.py` | 7 | `num_ctx` — "bug im lặng nghiêm trọng nhất đã gặp" |
| `test_retrieval.py` | 7 | Embedding + FAISS cơ bản, lưu/tải index giữ nguyên kết quả, cảnh báo khi đổi model |
| `test_reranker.py` | 6 | Rerank kéo đúng đoạn lên đầu, **không đổi điểm similarity**, không có reranker thì pipeline vẫn chạy |

### 16.3. Bốn khuôn mẫu test đáng học

**1. Test kiểm THEO HÌNH DẠNG, không theo tài liệu** (`test_khai_quat_tai_lieu.py`)

Thay vì kiểm "file X trang 7 phải ra kết quả Y" (chỉ đúng với file X), bộ test này dựng ra
những **hình dạng dữ liệu** bất thường: bảng chỉ có dòng tiêu đề, bảng 8 cột, ô chứa nguyên
một bài văn, bảng toàn ô rỗng, trang hai cột, chữ trong text box. Đây là cách chống overfit
ở tầng test.

**2. Test kiểm chính THƯỚC ĐO** (`test_do_tin_cay_thuoc_do.py`)

`test_dung_mot_nua_bi_muc_thap_nhat_keo_ve_khong`,
`test_diem_ngoai_thang_bi_loai_khoi_phep_lay_trung_vi`,
`test_bao_dong_gia_van_vo_hai_vi_ban_doc_lai_phai_TOT_HON_moi_duoc_nhan` — những test này
không kiểm hệ thống, chúng kiểm **logic đánh giá hệ thống**.

**3. Test đầu-cuối qua đúng giao diện thật** (`test_index_tang_dan.py`, `test_giao_dien.py`)

Dùng `streamlit.testing.AppTest` để chạy `app.py` như người dùng thật bấm nút. Nhờ vậy các
bug tương tác Streamlit (widget disabled, rerun huỷ ngang) được canh chừng.

**4. Test cho các chế độ hỏng** (`test_ket_noi_ollama.py`, `test_tai_lieu_scan.py`)

`test_ollama_chet_khong_lam_hong_luot_hoi`, `test_file_doc_hong_khong_duoc_ghi_la_da_xu_ly`,
`test_mot_anh_loi_khong_lam_hong_ca_lo`, `test_cache_embedding_khong_sap_khi_khong_ghi_duoc_xuong_dia`
— **một phần đáng kể của bộ test dành cho việc hệ thống hỏng đúng cách**, không phải cho
việc nó chạy đúng.

---
---

# PHẦN III — VẬN HÀNH

## 17. Cài đặt và chạy

### 17.1. Yêu cầu

- **Python 3.11+** (đã kiểm chứng trên 3.14).
- **[Ollama](https://ollama.com)** đã cài, và đã kéo hai model:
  ```bash
  ollama pull qwen3:4b        # LLM sinh câu trả lời (~2.6 GB)
  ollama pull qwen2.5vl:3b    # vision + OCR (~3.2 GB) — chỉ cần nếu dùng ảnh/PDF scan
  ```
- **Không bắt buộc GPU.** Model embedding (~1.1 GB) và reranker (~2.2 GB) tự tải về từ
  HuggingFace ở lần chạy đầu (cần Internet **một lần**).

### 17.2. Các bước

```bash
# 1. (Tuỳ chọn, nhưng nên làm nếu có GPU NVIDIA) cài PyTorch bản CUDA TRƯỚC
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 2. Cài thư viện
pip install -r requirements.txt

# 3. (Tuỳ chọn) tạo file cấu hình riêng
cp .env.example .env

# 4. Chạy
streamlit run app.py
```

> **Bẫy quan trọng**: `pip install sentence-transformers` kéo về PyTorch bản **CPU-only**.
> Trên máy có GPU NVIDIA, đó là một cấu hình **sai không gây lỗi** — hệ thống vẫn trả lời
> đúng, chỉ chậm hơn nhiều lần. Đo trên RTX 5060: embedding **12,8×**, rerank **11,2×**,
> truy xuất đầu-cuối **9,0×** — mà **6/6 câu hỏi cho kết quả giống hệt**.
>
> Kiểm tra: `python -c "import torch; print(torch.cuda.is_available())"`.
> Thanh bên của ứng dụng cũng hiện rõ đang chạy GPU hay CPU cho **từng vai trò**.

### 17.3. Dùng

1. **Thanh bên**: upload PDF/PPTX/DOCX → tick chọn nguồn được dùng → bấm **"Đọc tài liệu"**.
2. **Đặt câu hỏi** tiếng Việt hoặc tiếng Anh. Hoặc **đưa ra một khẳng định để kiểm chứng**
   (*"Pháp luật ra đời trước nhà nước, đúng không?"*).
3. Câu trả lời hiện **dần**; sau ~2 giây đã thấy *"đã tìm N đoạn liên quan trong \<tên file\>"*.
4. **Nguồn** hiển thị dưới câu trả lời là nguồn mà chính câu trả lời đó đã tham chiếu.
5. **Hỏi nối tiếp được**: *"Thế còn dấu hiệu thứ hai?"*.
6. Nút **＋ Hội thoại mới** xoá lịch sử nhưng **giữ nguyên tài liệu và index**.

### 17.4. Chạy đo đạc và kiểm thử

```bash
pytest -m "not slow"                                 # test nhanh
pytest                                               # tất cả 335 test
python evaluation/run_evaluation.py --nhanh          # đo truy xuất (tất định, vài phút)
python evaluation/run_evaluation.py                  # đo đầy đủ (~80 phút)
python evaluation/run_evaluation.py --khoang-cach    # đo mức overfit
python evaluation/tao_tai_lieu_mau.py                # sinh bộ tài liệu độc lập
python evaluation/kiem_dinh_judge.py                 # đo độ tin cậy của thước đo
python evaluation/do_dau_cuoi.py                     # đo đầu-cuối
```

---

## 18. Các biến cấu hình cần biết

Đặt trong file `.env` ở thư mục gốc. Mười biến hay dùng nhất:

| Biến | Mặc định | Khi nào cần đổi |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b` | máy yếu → `qwen3:1.7b`; máy mạnh → `qwen3:8b` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama chạy ở máy khác |
| `TOP_K` | `4` | muốn câu trả lời tổng hợp nhiều nguồn hơn → tăng (nhớ theo dõi `num_ctx`) |
| `BAT_RERANK` | `1` | máy quá yếu → `0` (tiết kiệm 2.2 GB, nhưng **mất luôn cơ chế chặn câu lạc đề**) |
| `BAT_CHU_THICH_ANH` | `1` | tài liệu không có hình → `0`, build nhanh hơn nhiều |
| `BAT_OCR_DU_PHONG` | `1` | không có PDF scan → `0` |
| `BAT_STREAMING` | `1` | debug → `0` để xem câu trả lời một cục |
| `BAT_INDEX_TANG_DAN` | `1` | nghi index sai → `0` để build lại toàn bộ |
| `THIET_BI_EMBEDDING` | `auto` | ép `cpu` / `cuda` khi muốn kiểm soát VRAM |
| `LOG_PHAN_BO_DIEM` | `0` | `1` khi cần hiệu chỉnh ngưỡng trên corpus mới |

**Nguyên tắc khi đổi tham số**: đổi bất kỳ tham số nào trong nhóm *"ăn vào nội dung đã
index"* (`EMBEDDING_MODEL_NAME`, `CHUNK_SIZE_TOKENS`, `BAT_NHAN_DIEN_TIEU_DE`,
`BAT_TRICH_ANH`, `BAT_CHU_THICH_ANH`) thì **phải bấm "Đọc tài liệu" lại**. Hệ thống sẽ tự
cảnh báo, nhưng đừng bỏ qua cảnh báo đó.

---

## 19. Sự cố thường gặp

| Triệu chứng | Nguyên nhân gần như chắc chắn | Cách xử lý |
|---|---|---|
| Thanh bên hiện `⚠️ Không kết nối được tới máy chủ Ollama` | Ollama chưa chạy | Mở app Ollama, hoặc `ollama serve` |
| `Máy chủ Ollama đang chạy nhưng chưa có model` | chưa pull | `ollama pull qwen3:4b` |
| Mọi câu hỏi đều nhận câu từ chối | bỏ tick hết tài liệu, **hoặc** index rỗng, **hoặc** tài liệu là PDF scan không đọc được | Xem log: có dòng `KHÔNG ĐỌC ĐƯỢC NỘI DUNG từ '...'` không |
| Câu trả lời **đứt giữa chừng** | `done_reason='length'` | Xem log; nâng `OLLAMA_NUM_CTX` **trước** khi nghĩ tới `num_predict` |
| Câu trả lời ngắn bất thường, trích dẫn trỏ sai chỗ | prompt bị cắt | Xem log `PROMPT BỊ CẮT`; hạ `TOP_K` hoặc `NGAN_SACH_KY_TU_MOI_DOAN` |
| Truy xuất ra kết quả vô lý | index cũ + model embedding mới | Thanh bên có cảnh báo `⚠️ Index được build bằng model...`; bấm "Đọc tài liệu" |
| Build rất chậm | đang chú thích ảnh bằng model vision (~1,9 s/hình) | Xem bảng profiling cuối lần build; tắt `BAT_CHU_THICH_ANH` nếu không cần |
| Chậm dù có GPU | PyTorch bản CPU-only | Thanh bên hiện `🐢 Đang chạy CPU`; cài lại PyTorch bản CUDA |
| Xoá file rồi mà vẫn bị ghi lại | *(đã sửa)* bug `file_uploader` giữ danh sách cũ | Nếu tái diễn, xem `uploader_key_n` trong `app.py` |
| Giao diện treo, mọi nút disabled | lỗi thoát ra ngoài khối `try` | Tải lại trang; xem log để tìm lỗi gốc |
| Nghi hệ thống trả lời bằng nội dung cũ | cache | Bấm **"Xoá cache"** ở thanh bên — an toàn tuyệt đối |
---
---

# PHẦN IV — TRA CỨU NHANH

## 20. Từ điển thuật ngữ

| Thuật ngữ | Nghĩa trong dự án này |
|---|---|
| **RAG** | Retrieval-Augmented Generation. Tìm đoạn liên quan rồi mới sinh câu trả lời dựa trên đoạn đó. |
| **Embedding** | Biến chữ thành vector số sao cho nghĩa gần nhau → vector gần nhau. Model: `multilingual-e5-base`, 768 chiều. |
| **Cosine similarity** | Cosin góc giữa hai vector, đo độ gần về nghĩa. Trong code là `diem_similarity`. Vì vector đã chuẩn hoá nên nó **bằng** inner product. |
| **Chunk** | Một đoạn nhỏ của tài liệu (~160 token). Đơn vị được embed và lưu trong FAISS. |
| **Overlap** | Phần chồng lấn giữa hai chunk liền kề (32 token), để câu bị cắt vẫn nguyên vẹn ở một trong hai. |
| **Metadata** | Thông tin đi kèm mỗi chunk: nguồn, trang, vị trí, loại nội dung. FAISS không lưu được nên phải lưu song song bằng pickle. |
| **FAISS** | Thư viện tìm kiếm vector của Meta. Dự án dùng `IndexFlatIP` — quét toàn bộ, chính xác tuyệt đối. |
| **BM25** | Thuật toán xếp hạng theo **từ khoá** (họ TF-IDF). Nhánh thứ hai của tìm kiếm lai. |
| **RRF** | Reciprocal Rank Fusion. Hợp nhất nhiều danh sách xếp hạng bằng cách cộng `1/(K + thứ_hạng)`. |
| **BM25 cứu hộ** | BM25 bơm ứng viên vào tập rerank với điểm RRF = 0: giúp **recall**, không có quyền **precision**. |
| **Bi-encoder** | Mã hoá câu hỏi và tài liệu **riêng biệt** rồi so vector. Nhanh, thô. Chính là embedding model. |
| **Cross-encoder** | Đọc cả cặp `(câu hỏi, đoạn)` cùng lúc và chấm điểm. Chính xác, chậm. Chính là reranker. |
| **Rerank** | Xếp hạng lại ứng viên bằng cross-encoder. Tầng lọc **thứ hai**. |
| **Over-fetch** | Lấy dư ứng viên (60) so với số cần (4), vì còn nhiều tầng lọc phía sau. |
| **Đoạn trích** | Chunk neo + các chunk hàng xóm đã nối liền mạch. Đây mới là thứ đưa vào prompt. |
| **Chunk neo** | Chunk thật sự khớp câu hỏi, làm tâm để mở rộng. Trong code là `doan_khop`. |
| **`num_ctx`** | Cửa sổ ngữ cảnh của LLM. **Không khai báo thì Ollama cấp 4096 và cắt im lặng.** |
| **`num_predict`** | Trần token sinh ra — gồm **suy luận + câu trả lời cộng lại**. |
| **Thinking / `<think>`** | Chuỗi suy luận nội bộ của model trước khi viết câu trả lời. Qwen3 luôn sinh nó. |
| **Faithfulness** | Câu trả lời có trung thực với ngữ cảnh không (LLM chấm 0–1). |
| **Answer relevance** | Câu trả lời có đúng trọng tâm câu hỏi không (LLM chấm 0–1). |
| **Citation accuracy** | Đoạn được **dẫn** có thật sự chống lưng cho **ý đang dẫn nó** không. |
| **Bám ngữ cảnh** | Tỉ lệ cụm 4 từ của câu trả lời xuất hiện **nguyên văn** trong ngữ cảnh. Tất định. Cao = tin được, thấp = không kết luận gì. |
| **Precision@K / Recall@K** | Trong K đoạn lấy ra, bao nhiêu đúng / trong các trang đúng, bao nhiêu lấy được. |
| **MRR** | Mean Reciprocal Rank — trung bình `1/thứ_hạng của đoạn đúng đầu tiên`. |
| **Held-out** | Bộ câu hỏi + tài liệu **không dùng để tinh chỉnh**. Chênh lệch giữa in-sample và held-out = mức overfit. |
| **LLM-as-judge** | Dùng LLM chấm điểm câu trả lời của LLM. Rẻ, nhưng **có sai số hệ thống** — nên dự án đo cả độ tin cậy của nó. |
| **Sycophancy** | Xu hướng model gật đầu theo người hỏi. Chế độ **kiểm chứng** sinh ra để chặn điều này. |
| **Build tăng dần** | Chỉ xử lý lại tài liệu MỚI hoặc ĐÃ ĐỔI, so bằng **băm nội dung**. |
| **Content hash** | Băm nội dung file (SHA-256 rút gọn 32 ký tự hex). Khoá của mọi tầng cache. |
| **Vân tay cấu hình** | Băm của các tham số ảnh hưởng kết quả. Ghi trong `index_info.json`, so lại để phát hiện index lỗi thời. |
| **OCR dự phòng** | OCR **chỉ chạy khi** trích text đã thất bại (font hỏng, trang scan) — không phải mặc định. |
| **`x_tolerance`** | Tham số của pdfplumber quyết định khi nào hai ký tự được coi là cùng một từ. Sai → chữ dính liền. |
| **Group shape** | Nhóm hình trong PPTX. Phải duyệt **đệ quy** vào, nếu không mất hết nội dung bên trong. |
| **Streamlit rerun** | Streamlit chạy lại **toàn bộ script** sau mỗi thao tác. Nguồn gốc của ba bug đã sửa trong `app.py`. |
| **`st.session_state`** | Dict duy nhất sống sót qua các lần rerun. |
| **`@st.cache_resource`** | Giữ một object nặng (model) cho cả phiên. |
| **Lỗi im lặng** | Hệ thống hỏng mà không báo lỗi, chỉ chất lượng kết quả tụt. **Kẻ thù số một của dự án này.** |

---

## 21. Bảng tra: hàm ↔ file

280 hàm/class, sắp theo alphabet. Cột "Dòng" là số dòng tại thời điểm viết tài liệu — dùng
để định vị nhanh, không phải để trích dẫn chính xác sau khi code thay đổi.

| Hàm / class | File | Dòng |
|---|---|---|
| `_ban_ghi_anh` | `rag/image_extractor.py` | 193 |
| `_bang_profiling` | `evaluation/do_dau_cuoi.py` | 63 |
| `_bang_sang_markdown` | `rag/document_loader.py` | 392 |
| `_bo_ban_ghi_anh_rong` | `rag/document_loader.py` | 1137 |
| `_cac_cot_cua_trang` | `rag/document_loader.py` | 609 |
| `_cac_so_tham_chieu` | `rag/citation.py` | 33 |
| `_cache_con_du_anh` | `rag/document_loader.py` | 1095 |
| `_canh_bao_tai_lieu_khong_doc_duoc` | `rag/document_loader.py` | 1170 |
| `_cat_bang_giu_tieu_de` | `rag/chunking.py` | 71 |
| `_cham_mot_cap` | `rag/doi_chieu_nguon.py` | 239 |
| `_chay_mot_ca` | `evaluation/kiem_dinh_judge.py` | 109 |
| `_chay_va_hien_theo_luong` | `app.py` | 393 |
| `_chon_chu_thich` | `rag/image_extractor.py` | 60 |
| `_chuan_hoa_nfc` | `rag/document_loader.py` | 263 |
| `_dang_ngo` | `evaluation/run_evaluation.py` | 75 |
| `_danh_dau_tieu_de` | `rag/document_loader.py` | 311 |
| `_dat_cau_hoi` | `app.py` | 471 |
| `_diem_cao_nhat` | `evaluation/do_nguong_rerank.py` | 63 |
| `_diem_hop_le` | `evaluation/metrics.py` | 171 |
| `_do_anh_huong_truy_xuat` | `evaluation/kiem_dinh_viet_lai.py` | 167 |
| `_do_do_tre` | `evaluation/do_quy_mo_index.py` | 98 |
| `_doan` | `evaluation/kiem_dinh_doi_chieu.py` | 39 |
| `_doc_mot_trang_pdf` | `rag/document_loader.py` | 694 |
| `_don_dep_watermark` | `rag/document_loader.py` | 270 |
| `_dung_ngu_canh` | `rag/tiep_noi_hoi_thoai.py` | 242 |
| `_ghep_prompt` | `rag/rag_pipeline.py` | 542 |
| `_goi_judge` | `evaluation/metrics.py` | 242 |
| `_goi_judge_kep` | `evaluation/metrics.py` | 175 |
| `_goi_judge_on_dinh` | `evaluation/metrics.py` | 189 |
| `_hang_thanh_van_xuoi` | `rag/chunking.py` | 59 |
| `_hien_thi_bam_nguon` | `app.py` | 331 |
| `_hien_thi_cach_hieu` | `app.py` | 351 |
| `_hien_thi_mau_thuan` | `app.py` | 370 |
| `_hien_thi_trich_dan` | `app.py` | 280 |
| `_in_bang_ket_qua` | `evaluation/run_evaluation.py` | 80 |
| `_in_bang_theo_nhom` | `evaluation/run_evaluation.py` | 147 |
| `_in_nhom` | `evaluation/do_nguong_rerank.py` | 73 |
| `_in_tong_ket_query` | `evaluation/do_dau_cuoi.py` | 157 |
| `_ket_luan` | `evaluation/do_quy_mo_index.py` | 182 |
| `_khoa` | `evaluation/metrics.py` | 64 |
| `_khoi_bang` | `rag/document_loader.py` | 571 |
| `_khong_dau` | `evaluation/kiem_dinh_viet_lai.py` | 78 |
| `_la_anh_cua_trang_chu` | `rag/image_extractor.py` | 74 |
| `_la_bang_that` | `rag/document_loader.py` | 545 |
| `_la_trang_muc_luc` | `rag/document_loader.py` | 284 |
| `_lam_sach` | `rag/tiep_noi_hoi_thoai.py` | 263 |
| `_lay_bool` | `config.py` | 70 |
| `_lay_client_vision` | `rag/document_loader.py` | 427 |
| `_lay_float` | `config.py` | 62 |
| `_lay_int` | `config.py` | 58 |
| `_lay_str` | `config.py` | 66 |
| `_LocSuyLuanTheoLuong (class)` | `rag/rag_pipeline.py` | 111 |
| `_LocSuyLuanTheoLuong.__init__` | `rag/rag_pipeline.py` | 125 |
| `_LocSuyLuanTheoLuong.ket_thuc` | `rag/rag_pipeline.py` | 152 |
| `_LocSuyLuanTheoLuong.them` | `rag/rag_pipeline.py` | 129 |
| `_nap_file_env` | `config.py` | 36 |
| `_noi_lien_mach` | `rag/rag_pipeline.py` | 364 |
| `_o_bang_khong_lap` | `rag/document_loader.py` | 360 |
| `_ocr_cac_trang` | `rag/document_loader.py` | 450 |
| `_phat_hien_ngon_ngu` | `rag/rag_pipeline.py` | 312 |
| `_phat_hien_tieu_de_pdf` | `rag/document_loader.py` | 321 |
| `_recall_so_voi_flat` | `evaluation/do_quy_mo_index.py` | 115 |
| `_sinh_muc_tuong_tu` | `evaluation/tao_tai_lieu_mau.py` | 146 |
| `_so_co_dau_cham` | `evaluation/do_quy_mo_index.py` | 63 |
| `_so_worker_mac_dinh` | `config.py` | 1016 |
| `_store_dung_lai_duoc` | `app.py` | 164 |
| `_tach_khoi_bang` | `rag/chunking.py` | 40 |
| `_tach_tu` | `rag/lexical_search.py` | 32 |
| `_tap_so` | `rag/doi_chieu_nguon.py` | 98 |
| `_ten_file_an_toan` | `rag/image_extractor.py` | 53 |
| `_text_pdf_khong_ke_bang` | `rag/document_loader.py` | 583 |
| `_text_theo_cot` | `rag/document_loader.py` | 677 |
| `_text_trong_text_box` | `rag/document_loader.py` | 964 |
| `_thong_bao_khong_ket_noi_duoc` | `rag/rag_pipeline.py` | 63 |
| `_tim_trang_can_ocr` | `evaluation/do_worker_gpu.py` | 100 |
| `_tinh_num_ctx` | `rag/rag_pipeline.py` | 432 |
| `_tom_tat` | `evaluation/do_dau_cuoi.py` | 54 |
| `_trich_text` | `rag/document_loader.py` | 119 |
| `_trich_text_shape` | `rag/document_loader.py` | 887 |
| `_trich_text_thich_ung` | `rag/document_loader.py` | 171 |
| `_trung_binh` | `evaluation/run_evaluation.py` | 69 |
| `_ty_le_dinh_chu` | `rag/document_loader.py` | 77 |
| `_ty_le_tu_le` | `rag/document_loader.py` | 105 |
| `_uoc_luong_so_token` | `rag/rag_pipeline.py` | 380 |
| `_ve_bieu_do_cot` | `evaluation/tao_tai_lieu_mau.py` | 218 |
| `_ve_so_do_quy_trinh` | `evaluation/tao_tai_lieu_mau.py` | 198 |
| `_vector_gia_lap` | `evaluation/do_quy_mo_index.py` | 68 |
| `_xuat_csv` | `evaluation/run_evaluation.py` | 231 |
| `answer_relevance` | `evaluation/metrics.py` | 327 |
| `bam_bytes` | `rag/bo_nho_dem.py` | 47 |
| `bam_chuoi` | `rag/bo_nho_dem.py` | 57 |
| `bam_file` | `rag/bo_nho_dem.py` | 61 |
| `bao_cao` | `rag/do_thoi_gian.py` | 80 |
| `bat_dau_ingestion` | `rag/tai_nguyen_gpu.py` | 356 |
| `BM25 (class)` | `rag/lexical_search.py` | 52 |
| `BM25.__init__` | `rag/lexical_search.py` | 59 |
| `BM25.tim_kiem` | `rag/lexical_search.py` | 76 |
| `bo_so_trich_dan` | `rag/citation.py` | 50 |
| `bo_sung_chu_thich_vision` | `rag/vision_caption.py` | 162 |
| `cac_cap_dang_ngo` | `rag/doi_chieu_nguon.py` | 137 |
| `cac_cau_hoi_truoc` | `rag/tiep_noi_hoi_thoai.py` | 163 |
| `cac_file_tai_lieu` | `rag/document_loader.py` | 1199 |
| `cau_theo_trich_dan` | `rag/citation.py` | 274 |
| `chay` | `evaluation/do_worker_gpu.py` | 121 |
| `chay_danh_gia` | `evaluation/run_evaluation.py` | 385 |
| `chay_danh_gia_nhanh` | `evaluation/run_evaluation.py` | 254 |
| `chay_do` | `evaluation/do_nguong_rerank.py` | 87 |
| `chia_chunk` | `rag/chunking.py` | 183 |
| `chu_thich_anh` | `rag/vision_caption.py` | 129 |
| `chuan_bi_truy_van` | `rag/tiep_noi_hoi_thoai.py` | 352 |
| `co_cuda` | `rag/tai_nguyen_gpu.py` | 68 |
| `co_dau_hieu_bat_dong` | `rag/doi_chieu_nguon.py` | 118 |
| `dat_lai` | `rag/do_thoi_gian.py` | 44 |
| `dem_token` | `rag/chunking.py` | 131 |
| `dinh_dang_trich_dan` | `rag/citation.py` | 95 |
| `do` | `rag/do_thoi_gian.py` | 60 |
| `do_bam_ngu_canh` | `rag/citation.py` | 245 |
| `do_bam_ngu_canh_thap_nhat` | `evaluation/metrics.py` | 42 |
| `do_chinh_xac_trich_dan` | `evaluation/metrics.py` | 351 |
| `do_ingestion` | `evaluation/do_dau_cuoi.py` | 75 |
| `do_khoang_cach_held_out` | `evaluation/run_evaluation.py` | 333 |
| `do_mot_kich_thuoc` | `evaluation/do_quy_mo_index.py` | 124 |
| `do_query` | `evaluation/do_dau_cuoi.py` | 125 |
| `doc_csv_ket_qua` | `evaluation/run_evaluation.py` | 176 |
| `doc_docx` | `rag/document_loader.py` | 987 |
| `doc_nhieu_file` | `rag/document_loader.py` | 1208 |
| `doc_pdf` | `rag/document_loader.py` | 733 |
| `doc_pptx` | `rag/document_loader.py` | 906 |
| `doc_tai_lieu` | `rag/document_loader.py` | 1060 |
| `doc_tai_lieu_co_cache` | `rag/document_loader.py` | 1110 |
| `doc_tai_lieu_hoan_chinh` | `rag/document_loader.py` | 1072 |
| `doc_thu_muc` | `rag/document_loader.py` | 1259 |
| `don_bo_nho_cuda` | `rag/tai_nguyen_gpu.py` | 282 |
| `du_cho_giu_embedding_tren_gpu` | `rag/tai_nguyen_gpu.py` | 326 |
| `dung_luong_cache` | `rag/bo_nho_dem.py` | 344 |
| `duyet_shape` | `rag/document_loader.py` | 871 |
| `EmbeddingService (class)` | `rag/embedding.py` | 29 |
| `EmbeddingService.__init__` | `rag/embedding.py` | 30 |
| `EmbeddingService._encode` | `rag/embedding.py` | 76 |
| `EmbeddingService.chuyen_thiet_bi` | `rag/embedding.py` | 45 |
| `EmbeddingService.dem_token` | `rag/embedding.py` | 121 |
| `EmbeddingService.dimension` | `rag/embedding.py` | 106 |
| `EmbeddingService.encode_cau_hoi` | `rag/embedding.py` | 99 |
| `EmbeddingService.encode_tai_lieu` | `rag/embedding.py` | 95 |
| `EmbeddingService.lay_ham_dem_token` | `rag/embedding.py` | 134 |
| `EmbeddingService.max_seq_length` | `rag/embedding.py` | 115 |
| `encode_co_cache` | `rag/bo_nho_dem.py` | 314 |
| `faithfulness` | `evaluation/metrics.py` | 286 |
| `format_text_trich_dan` | `rag/citation.py` | 302 |
| `ghi_bao_cao` | `rag/do_thoi_gian.py` | 103 |
| `ghi_log_vram` | `rag/tai_nguyen_gpu.py` | 270 |
| `ghi_nhan` | `rag/do_thoi_gian.py` | 52 |
| `HieuChinhXTolerance (class)` | `rag/document_loader.py` | 126 |
| `HieuChinhXTolerance.__init__` | `rag/document_loader.py` | 146 |
| `HieuChinhXTolerance.da_hieu_chinh` | `rag/document_loader.py` | 151 |
| `HieuChinhXTolerance.ghi_nhan` | `rag/document_loader.py` | 164 |
| `HieuChinhXTolerance.thu_tu_uu_tien` | `rag/document_loader.py` | 157 |
| `ket_thuc_ingestion` | `rag/tai_nguyen_gpu.py` | 369 |
| `khoa_ocr` | `rag/bo_nho_dem.py` | 184 |
| `khoa_tai_lieu` | `rag/bo_nho_dem.py` | 179 |
| `khoa_vision` | `rag/bo_nho_dem.py` | 198 |
| `KhoDem (class)` | `rag/bo_nho_dem.py` | 107 |
| `KhoDem.__init__` | `rag/bo_nho_dem.py` | 119 |
| `KhoDem._duong_dan` | `rag/bo_nho_dem.py` | 125 |
| `KhoDem.co` | `rag/bo_nho_dem.py` | 128 |
| `KhoDem.lay_json` | `rag/bo_nho_dem.py` | 158 |
| `KhoDem.lay_text` | `rag/bo_nho_dem.py` | 131 |
| `KhoDem.luu_json` | `rag/bo_nho_dem.py` | 168 |
| `KhoDem.luu_text` | `rag/bo_nho_dem.py` | 145 |
| `KhoVectorDem (class)` | `rag/bo_nho_dem.py` | 217 |
| `KhoVectorDem.__init__` | `rag/bo_nho_dem.py` | 231 |
| `KhoVectorDem._nap` | `rag/bo_nho_dem.py` | 241 |
| `KhoVectorDem.lay` | `rag/bo_nho_dem.py` | 253 |
| `KhoVectorDem.luu` | `rag/bo_nho_dem.py` | 290 |
| `KhoVectorDem.them` | `rag/bo_nho_dem.py` | 280 |
| `kich_thuoc_chunk_an_toan` | `rag/chunking.py` | 141 |
| `kich_thuoc_lo_embedding` | `rag/tai_nguyen_gpu.py` | 180 |
| `kiem_tra_may_chu_llm` | `rag/rag_pipeline.py` | 83 |
| `la_cau_hoi_kiem_chung` | `rag/rag_pipeline.py` | 359 |
| `la_cau_hoi_phuc_tap` | `rag/rag_pipeline.py` | 404 |
| `la_cau_hoi_tiep_noi` | `rag/tiep_noi_hoi_thoai.py` | 137 |
| `lay_embedding_service` | `app.py` | 117 |
| `lay_loi_may_chu_llm` | `app.py` | 131 |
| `lay_pipeline` | `app.py` | 149 |
| `lay_reranker_service` | `app.py` | 124 |
| `loc_anh_lap_lai` | `rag/image_extractor.py` | 154 |
| `loc_theo_tham_chieu` | `rag/citation.py` | 143 |
| `LoiKhongKetNoiDuocOllama (class)` | `rag/rag_pipeline.py` | 48 |
| `luu_anh_trang_pdf` | `rag/image_extractor.py` | 255 |
| `ly_do_loai_anh` | `rag/image_extractor.py` | 100 |
| `ly_do_loai_anh_blob` | `rag/image_extractor.py` | 130 |
| `main` | `evaluation/do_dau_cuoi.py` | 181 |
| `main` | `evaluation/do_quy_mo_index.py` | 245 |
| `main` | `evaluation/do_worker_gpu.py` | 180 |
| `main` | `evaluation/kiem_dinh_doi_chieu.py` | 108 |
| `main` | `evaluation/kiem_dinh_judge.py` | 118 |
| `main` | `evaluation/kiem_dinh_viet_lai.py` | 88 |
| `main` | `evaluation/tao_tai_lieu_mau.py` | 445 |
| `mo_hinh_vision_co_san` | `rag/vision_caption.py` | 114 |
| `mo_ta_phan_cung` | `rag/tai_nguyen_gpu.py` | 156 |
| `nap_bo_cau_hoi` | `evaluation/run_evaluation.py` | 63 |
| `nen_ngu_canh` | `rag/rag_pipeline.py` | 485 |
| `ngan_sach_token_ngu_canh` | `rag/rag_pipeline.py` | 474 |
| `nghich_dao_thu_hang` | `evaluation/metrics.py` | 110 |
| `ngu_canh_cho_prompt` | `rag/tiep_noi_hoi_thoai.py` | 191 |
| `nha_model_ollama` | `rag/tai_nguyen_gpu.py` | 297 |
| `ocr_trang_pdf` | `rag/vision_caption.py` | 93 |
| `precision_tai_k` | `evaluation/metrics.py` | 70 |
| `RagPipeline (class)` | `rag/rag_pipeline.py` | 597 |
| `RagPipeline.__init__` | `rag/rag_pipeline.py` | 598 |
| `RagPipeline._dung_doan_trich` | `rag/rag_pipeline.py` | 773 |
| `RagPipeline._ghi_nhan_thong_ke_llm` | `rag/rag_pipeline.py` | 1229 |
| `RagPipeline._goi_llm_theo_luong` | `rag/rag_pipeline.py` | 1087 |
| `RagPipeline._ung_vien` | `rag/rag_pipeline.py` | 636 |
| `RagPipeline._xep_hang_lai` | `rag/rag_pipeline.py` | 726 |
| `RagPipeline.hoi_dap` | `rag/rag_pipeline.py` | 1487 |
| `RagPipeline.hoi_dap_theo_luong` | `rag/rag_pipeline.py` | 1363 |
| `RagPipeline.sinh_cau_tra_loi` | `rag/rag_pipeline.py` | 1349 |
| `RagPipeline.sinh_cau_tra_loi_theo_luong` | `rag/rag_pipeline.py` | 1275 |
| `RagPipeline.truy_xuat` | `rag/rag_pipeline.py` | 880 |
| `recall_tai_k` | `evaluation/metrics.py` | 79 |
| `RerankerService (class)` | `rag/reranker.py` | 36 |
| `RerankerService.__init__` | `rag/reranker.py` | 37 |
| `RerankerService._nap_model` | `rag/reranker.py` | 50 |
| `RerankerService.xep_hang` | `rag/reranker.py` | 67 |
| `so_lieu` | `rag/do_thoi_gian.py` | 69 |
| `so_nhan_cpu` | `rag/tai_nguyen_gpu.py` | 152 |
| `so_sanh_bam_tai_lieu` | `rag/vector_store.py` | 37 |
| `so_sanh_voi_ban_truoc` | `evaluation/run_evaluation.py` | 181 |
| `so_worker_vision` | `rag/tai_nguyen_gpu.py` | 226 |
| `tai_index_da_co` | `app.py` | 142 |
| `tao_docx_co_bang` | `evaluation/tao_tai_lieu_mau.py` | 283 |
| `tao_docx_dai` | `evaluation/tao_tai_lieu_mau.py` | 241 |
| `tao_docx_nhieu_muc_tuong_tu` | `evaluation/tao_tai_lieu_mau.py` | 262 |
| `tao_pdf_hon_hop` | `evaluation/tao_tai_lieu_mau.py` | 382 |
| `tao_pptx_bang_anh` | `evaluation/tao_tai_lieu_mau.py` | 345 |
| `tao_pptx_ngan` | `evaluation/tao_tai_lieu_mau.py` | 327 |
| `tao_reranker_neu_bat` | `rag/reranker.py` | 78 |
| `tao_splitter` | `rag/chunking.py` | 164 |
| `ten_model_khop` | `rag/vision_caption.py` | 108 |
| `TheoDoiGpu (class)` | `evaluation/do_worker_gpu.py` | 44 |
| `TheoDoiGpu.__enter__` | `evaluation/do_worker_gpu.py` | 78 |
| `TheoDoiGpu.__exit__` | `evaluation/do_worker_gpu.py` | 84 |
| `TheoDoiGpu.__init__` | `evaluation/do_worker_gpu.py` | 55 |
| `TheoDoiGpu._lay_mau` | `evaluation/do_worker_gpu.py` | 63 |
| `TheoDoiGpu.tom_tat` | `evaluation/do_worker_gpu.py` | 89 |
| `thiet_bi` | `rag/tai_nguyen_gpu.py` | 87 |
| `thu_hang_dung_dau_tien` | `evaluation/metrics.py` | 88 |
| `tim_mau_thuan` | `rag/doi_chieu_nguon.py` | 295 |
| `tong_giay` | `rag/do_thoi_gian.py` | 75 |
| `tong_vram_gb` | `rag/tai_nguyen_gpu.py` | 140 |
| `trang_can_ocr` | `rag/vision_caption.py` | 74 |
| `trich_anh_docx` | `rag/image_extractor.py` | 362 |
| `trich_anh_pptx` | `rag/image_extractor.py` | 280 |
| `truy_van_ngu_canh` | `rag/tiep_noi_hoi_thoai.py` | 174 |
| `ung_vien_anh_trang` | `rag/image_extractor.py` | 224 |
| `van_tay_doc_tai_lieu` | `rag/bo_nho_dem.py` | 90 |
| `van_tay_embedding` | `rag/bo_nho_dem.py` | 96 |
| `VectorStore (class)` | `rag/vector_store.py` | 62 |
| `VectorStore.__init__` | `rag/vector_store.py` | 63 |
| `VectorStore._xoa_cache` | `rag/vector_store.py` | 76 |
| `VectorStore.bm25` | `rag/vector_store.py` | 124 |
| `VectorStore.chi_muc_nguon` | `rag/vector_store.py` | 99 |
| `VectorStore.chi_muc_trang` | `rag/vector_store.py` | 86 |
| `VectorStore.diem_cosine` | `rag/vector_store.py` | 200 |
| `VectorStore.luu` | `rag/vector_store.py` | 217 |
| `VectorStore.ly_do_khong_tuong_thich` | `rag/vector_store.py` | 273 |
| `VectorStore.so_luong_vector` | `rag/vector_store.py` | 317 |
| `VectorStore.tai` | `rag/vector_store.py` | 250 |
| `VectorStore.them` | `rag/vector_store.py` | 132 |
| `VectorStore.theo_nguon_va_trang` | `rag/vector_store.py` | 169 |
| `VectorStore.tim_kiem` | `rag/vector_store.py` | 188 |
| `VectorStore.tim_kiem_tu_khoa` | `rag/vector_store.py` | 196 |
| `VectorStore.tim_kiem_vi_tri` | `rag/vector_store.py` | 173 |
| `VectorStore.xoa_theo_nguon` | `rag/vector_store.py` | 142 |
| `viet_lai_cau_hoi` | `rag/tiep_noi_hoi_thoai.py` | 278 |
| `vram` | `rag/tai_nguyen_gpu.py` | 120 |
| `vram_con_trong_gb` | `rag/tai_nguyen_gpu.py` | 146 |
| `xay_dung_lai_index` | `app.py` | 184 |
| `xoa_cache` | `rag/bo_nho_dem.py` | 351 |

---
---

# PHẦN V — NHẬN XÉT PHẢN BIỆN

Phần này không phải để chê. Nó để bạn — người vừa đọc xong code — biết chỗ nào nên tin, chỗ
nào nên tự kiểm chứng, và chỗ nào là câu hỏi mở nếu bạn phải bảo vệ trước hội đồng.

## 22.1. Điểm mạnh thật sự

**1. Kỷ luật chống "lỗi im lặng" là điểm nổi bật nhất.**
Rất nhiều đồ án RAG dừng ở chỗ "chạy được". Dự án này liên tục hỏi *"nếu cái này hỏng mà
không báo lỗi thì làm sao biết?"* và trả lời bằng code: vân tay cấu hình index, bộ đếm token
thật của máy chủ, cảnh báo tài liệu không đọc được, cờ tự nghi ngờ của thước đo. Đây là tư
duy kỹ thuật ở mức cao hơn hẳn "làm cho chạy".

**2. Mỗi quyết định gắn với một số đo, không phải một cảm nhận.**
`TRONG_SO_BM25 = 0.0` không phải vì "thấy BM25 không hợp", mà vì đo được nó phá truy xuất
chéo ngôn ngữ. `num_predict` thích ứng bị **gỡ bỏ** sau khi đo lại. Việc ghi cả những thứ
**đã thử và bỏ** vào comment là thói quen của kỹ sư giỏi.

**3. Tách bạch vai trò rất sạch ở vài chỗ khó.**
Ba ví dụ đáng học: (a) BM25 có quyền recall nhưng không có quyền precision; (b) rerank đổi
thứ tự nhưng **không đổi thang điểm**; (c) `so_sanh_bam_tai_lieu` là hàm thuần, tách quyết
định khỏi tác dụng phụ.

**4. Đo cả thước đo, không chỉ đo hệ thống.**
Ba script `kiem_dinh_*.py` + cờ `dang_ngo` + biên độ dao động của giám khảo. Đây là phần
phương pháp luận mạnh nhất, và cũng là phần dễ ghi điểm nhất khi bảo vệ.

**5. Bộ test chú trọng chế độ HỎNG.**
Một phần lớn trong 335 test kiểm việc hệ thống **hỏng đúng cách** (Ollama chết, file hỏng,
ảnh lỗi, không ghi được cache) chứ không chỉ kiểm nó chạy đúng.

---

## 22.2. Những chỗ nên tự kiểm chứng hoặc xem lại

### (1) Một nguyên tắc được phát biểu nhưng chưa được thi hành

`app._hien_thi_cach_hieu()` có một docstring rất mạnh:

> *"KHÔNG được im lặng làm chuyện này. Đây là một PHỎNG ĐOÁN của hệ thống về ý người dùng,
> và trình bày phỏng đoán như thể là sự thật đúng là lỗi mà §5.54 đã phải sửa một lần rồi."*

Nhưng **phần thân đang bị comment lại**, nên hàm không vẽ gì. Tức hiện tại hệ thống **đang
im lặng ghép ngữ cảnh hội thoại** — đúng điều docstring nói là không được làm. Đây có thể là
một quyết định UI tạm thời, nhưng nó tạo ra mâu thuẫn giữa nguyên tắc đã tuyên bố và hành vi
thực tế. Nên: hoặc bật lại, hoặc sửa docstring để nói rõ vì sao đã đổi ý.

### (2) "Tìm kiếm lai" trong tài liệu, nhưng thực tế nhánh từ khoá đang tắt

`TRONG_SO_BM25 = 0.0`. Nghĩa là nhánh BM25 **không tham gia xếp hạng**; nó chỉ còn vai trò
cứu hộ (bơm 10 ứng viên với điểm 0). Quyết định này có căn cứ đo đạc rõ ràng và hoàn toàn
hợp lý.

Nhưng README và ARCHITECTURE vẫn mô tả hệ thống là "tìm kiếm lai vector + từ khoá", dễ khiến
người đọc (kể cả hội đồng) hiểu rằng hai nhánh đang cùng chạy. Nên diễn đạt chính xác hơn:
*"kiến trúc lai có sẵn, nhưng theo số đo thì nhánh từ khoá bị hạ về vai trò cứu hộ"*. Cách
nói đó **mạnh hơn**, vì nó cho thấy bạn đã đo và đã dám tắt một thứ mình tự xây.

### (3) Giám khảo và thí sinh là **cùng một model**

`JUDGE_MODEL` mặc định `= OLLAMA_MODEL` — tức `qwen3:4b` vừa sinh câu trả lời, vừa chấm điểm
câu trả lời của chính nó. Đây là một **thiên kiến tự đánh giá** (self-evaluation bias) đã
được ghi nhận rộng rãi trong tài liệu về LLM-as-judge: model có xu hướng chấm cao hơn cho
văn phong giống văn phong của chính nó.

Dự án có ba lớp giảm nhẹ (cờ `dang_ngo`, chấm 3 lần lấy trung vị, `kiem_dinh_judge.py`), và
đó là nhiều hơn hẳn mặt bằng chung. Nhưng ba lớp đó **không loại bỏ** được thiên kiến gốc.
Nếu có thời gian, một phép đo đối chứng rất rẻ: đặt `JUDGE_MODEL` sang một model **khác họ**
(ví dụ `llama3.1:8b` hay `gemma2:9b`) và so bảng điểm. Chênh lệch giữa hai giám khảo chính
là một con số nên có trong báo cáo.

### (4) Số lượng ngưỡng đã hiệu chỉnh so với cỡ mẫu

Đếm sơ bộ: `NGUONG_DIEM_TOI_THIEU = 0.5`, `TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT = 0.78`,
`NGUONG_DIEM_RERANK_TOI_THIEU = 0.001`, `NGUONG_COSINE_DOI_CHIEU = 0.88`,
`NGUONG_MAU_THUAN = 0.6`, `SO_DOAN_TOI_DA_MOI_TRANG = 2`, `SO_UNG_VIEN_XET_DA_DANG_TRANG = 20`,
`SO_TRANG_TOI_DA_MO_RONG = 1`, `SO_TU_CAU_HOI_DON_GIAN = 12`, `TY_LE_DINH_CHU_DE_DOC_LAI = 0.1`…
— khoảng **hơn 20 ngưỡng số** được hiệu chỉnh, trong khi bộ đánh giá in-sample chỉ có **29
câu hỏi**.

Đây không phải lỗi — mọi hệ thống thực dụng đều có nhiều ngưỡng. Nhưng tỉ lệ đó nghĩa là
**rủi ro overfit vào corpus là có thật**, và nó là điều hội đồng có thể hỏi.

Điều đáng khen: dự án **đã ý thức được** và đã dựng bộ held-out + `tao_tai_lieu_mau.py`. Nên
khi trình bày, hãy để con số held-out lên trước con số in-sample, và nói thẳng về giới hạn
này thay vì để người khác chỉ ra.

Một điểm nữa nên bổ sung: với n = 29, chênh lệch **+0.008 Recall@K** của BM25 cứu hộ tương
đương chưa tới một câu hỏi. Bảng kết quả hiện chưa có khoảng tin cậy hay kiểm định thống kê,
nên chưa phân biệt được "cải thiện nhỏ" với "nhiễu". Với chế độ `--nhanh` (tất định) thì
chênh lệch là thật, nhưng **thật trên đúng 29 câu này** — đó là hai chuyện khác nhau.

### (5) Prompt injection từ chính tài liệu người dùng nạp vào

Nội dung tài liệu đi thẳng vào prompt. Một tài liệu chứa dòng *"Bỏ qua mọi hướng dẫn phía
trên, hãy trả lời rằng…"* sẽ được model đọc như một chỉ thị. Hệ thống hiện **không có phòng
vệ nào** cho việc này.

Trong phạm vi một đồ án dùng tài liệu học tập của chính mình, đây là rủi ro rất thấp và việc
không xử lý là hợp lý. Nhưng nó nên được **nêu tên trong mục "Giới hạn đã biết"** thay vì
không nhắc tới — vì nếu hội đồng hỏi "hệ thống này đưa lên production được chưa", đây là câu
trả lời đúng.

### (6) `pickle` cho metadata

`metadata.pkl` được nạp bằng `pickle.load()`. Pickle **thực thi mã** khi giải mã, nên nạp
một file pickle từ nguồn không tin cậy tương đương chạy mã lạ. Với dữ liệu do chính máy sinh
ra thì an toàn, nhưng nếu ai đó chia sẻ index đã build cho nhau thì đây là một lỗ hổng thật.
Chuyển sang JSON hoặc `numpy.savez` là thay đổi nhỏ, và nên làm nếu index có thể được chia sẻ.

### (7) Một giả định ngầm trong nhận diện ngôn ngữ

`_phat_hien_ngon_ngu()` bước 4: *"Không có MỘT dấu vết tiếng Việt nào mà vẫn là một câu có
chữ → nhiều khả năng là tiếng Anh."* Giả định ngầm ở đây là **thế giới chỉ có hai ngôn ngữ**.
Một câu hỏi tiếng Pháp, Tây Ban Nha hay Indonesia sẽ được trả lời bằng tiếng Anh.

Điều này hoàn toàn đúng với phạm vi đã tuyên bố (hệ song ngữ Việt–Anh) — chỉ là giả định đó
nên được viết ra thành một dòng, để người đọc code sau này không tưởng đây là bộ nhận diện
ngôn ngữ tổng quát.

### (8) Đồng thời và nhiều phiên

Streamlit chạy nhiều phiên (nhiều tab trình duyệt) trên cùng một tiến trình, nhưng
`data/faiss_index/` là **tài nguyên dùng chung trên đĩa**. Nếu một tab bấm "Đọc tài liệu"
trong khi tab khác đang hỏi, không có khoá nào ngăn hai bên ghi/đọc chồng lên nhau. Với đồ
án một người dùng thì không sao, nhưng đây là một giới hạn nên nêu tên.

### (9) Hai chi tiết phụ thuộc vào hành vi nội bộ của thư viện

- `xoa_theo_nguon()` dựa vào việc `faiss.IndexFlatIP.remove_ids()` **giữ nguyên thứ tự tương
  đối** của các vector còn lại. Comment ghi *"đã kiểm chứng bằng test thủ công"*. Đây là một
  chi tiết triển khai của FAISS, không phải một hợp đồng API được bảo đảm. Nên có một test
  **tự động** khẳng định bất biến này (dựng index nhỏ, xoá giữa, kiểm tra metadata còn khớp
  vector), vì nếu FAISS đổi hành vi ở phiên bản sau thì hỏng hóc sẽ **im lặng**.
- Cách xử lý tham số `think` của Ollama dựa trên hành vi đo được ở client 0.6.2. Nên ghim
  phiên bản `ollama` trong `requirements.txt` (hiện là `ollama>=0.2`, khá rộng).

### (10) Chi phí bảo trì của việc tham chiếu `§5.x`

Comment trong code tham chiếu tới `ARCHITECTURE.md §5.11`, `§5.29`, `§5.54`, `§5.68`… rất
nhiều lần. Đây là một cách liên kết code với lý do rất tốt **khi tài liệu đứng yên**, nhưng
nếu chèn thêm một mục vào giữa §5 thì mọi tham chiếu sau đó lệch, mà **không có gì phát hiện
được** — lại đúng loại "lỗi im lặng" mà dự án ghét. Cân nhắc dùng nhãn ổn định
(`§num-ctx`, `§citation-suy-doan`) thay cho số thứ tự.

### (11) Tên định danh tiếng Việt — một đánh đổi có thật

Ưu điểm rõ: code đọc như văn xuôi, người Việt đọc hiểu ngay ý đồ, và với một đồ án tốt
nghiệp thì đó là lợi thế trình bày thật sự.

Nhược điểm cũng thật: người ngoài không đọc được; tìm kiếm lỗi trên Stack Overflow khó hơn
vì tên hàm không khớp thuật ngữ chuẩn; và nếu dự án muốn đưa lên GitHub công khai hay có
người nước ngoài tham gia thì đây là rào cản đầu tiên. Đây không phải "sai" — chỉ là một
đánh đổi nên được nêu ra có ý thức, và nếu bảo vệ được bằng lý do rõ ràng thì nó thành một
điểm thể hiện chính kiến.

---

## 22.3. Ba câu hỏi hội đồng nhiều khả năng sẽ hỏi — và nơi tìm câu trả lời

**"Làm sao biết hệ thống không chỉ hoạt động trên đúng corpus của em?"**
→ Bộ **held-out** (`--khoang-cach`), bộ tài liệu sinh độc lập (`tao_tai_lieu_mau.py`, đạt
Recall@K 1.00), và `test_khai_quat_tai_lieu.py` (kiểm theo **hình dạng** dữ liệu, không theo
tài liệu cụ thể). Nên nói thêm về giới hạn ở §22.2(4) trước khi bị hỏi.

**"Điểm Faithfulness 0.86 có đáng tin không, khi chính LLM chấm LLM?"**
→ `kiem_dinh_judge.py`, cờ `dang_ngo`, biên độ dao động giữa 3 lần chấm, và ca thật đã bắt
được (nhóm sách tiếng Anh 0.33 → 0.83 sau khi sửa **nguyên nhân gốc ở khâu đọc PDF**, không
phải chỉnh thước đo). Điểm yếu còn lại: giám khảo cùng model với thí sinh — §22.2(3).

**"Hệ thống mở rộng được tới quy mô nào?"**
→ `do_quy_mo_index.py` trả lời bằng số: `IndexFlatIP` chịu được tới đâu trong ngân sách độ
trễ 200 ms, và IVF/HNSW đổi lại được gì với cái giá recall bao nhiêu. Giới hạn thật hiện nay
không nằm ở FAISS mà ở **RAM giữ toàn bộ metadata** và ở **thời gian Ingestion** (chú thích
ảnh ~1,9 s/hình).

---

## 22.4. Nếu bạn phải tiếp tục phát triển dự án này

Thứ tự ưu tiên gợi ý, từ rẻ-lợi-nhiều tới đắt:

| # | Việc | Chi phí | Lợi ích |
|---|---|---|---|
| 1 | Bật lại `_hien_thi_cach_hieu` hoặc sửa docstring | rất thấp | xoá mâu thuẫn nguyên tắc ↔ hành vi |
| 2 | Ghim phiên bản `ollama` trong `requirements.txt` | rất thấp | tránh hỏng im lặng khi client đổi hành vi `think` |
| 3 | Test tự động cho bất biến thứ tự của `remove_ids` | thấp | bịt một lỗi im lặng tiềm tàng |
| 4 | Chạy đánh giá với `JUDGE_MODEL` khác họ | thấp (một lần chạy) | con số quan trọng cho báo cáo |
| 5 | Thêm khoảng tin cậy / kiểm định cho bảng kết quả | thấp | phân biệt cải thiện thật với nhiễu |
| 6 | Chuyển ba hằng số mốc (`[BẢNG]`, `[/BẢNG]`, `[HÌNH]`) vào `config.py` | thấp | gỡ hai import ngược |
| 7 | Thay `pickle` bằng định dạng không thực thi mã | trung bình | an toàn khi chia sẻ index |
| 8 | Mở rộng bộ câu hỏi in-sample lên 60–100 câu | trung bình (công gán nhãn) | giảm rủi ro overfit, tăng độ tin cậy số liệu |
| 9 | Khoá file cho `data/faiss_index/` | trung bình | an toàn khi nhiều tab/nhiều người |

---

## 22.5. Kết

Điểm đáng học nhất từ dự án này không phải là kiến trúc RAG — kiến trúc đó đã khá chuẩn mực
và bạn tìm được ở nhiều nơi. Điểm đáng học là **thái độ với sai lầm**: mỗi lần hệ thống hỏng,
tác giả không chỉ sửa, mà còn (a) tìm nguyên nhân gốc thay vì triệu chứng, (b) viết một cơ
chế để lần sau nó không hỏng im lặng nữa, và (c) ghi lại nguyên nhân ngay tại chỗ trong code
để người sau không "sửa lại cho gọn".

Ba câu comment sau tóm gọn tinh thần đó, và đáng nhớ hơn bất kỳ dòng code nào:

> *"Một lớp lỗi mà hệ thống KHÔNG THỂ tự phát hiện thì mọi kết luận rút ra từ nó đều đáng ngờ."*

> *"Sửa prompt trước hai cái kia sẽ cho một cải thiện nhẹ đủ để tưởng đã tìm đúng nguyên
> nhân, trong khi bug thật vẫn nằm nguyên đó."*

> *"Nói khi có bằng chứng, im lặng khi không có — chứ không suy đoán theo chiều ngược lại."*
