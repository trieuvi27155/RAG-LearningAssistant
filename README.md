# Hệ thống RAG hỏi đáp tài liệu học tập (tiếng Việt)

Đồ án tốt nghiệp: hệ thống hỏi đáp thông minh trên tài liệu học tập (PDF/PPTX/DOCX) bằng kỹ
thuật RAG, hỗ trợ song ngữ Việt/Anh (tự nhận diện ngôn ngữ câu hỏi để trả lời đúng ngôn ngữ
đó), chạy hoàn toàn local (không dùng API trả phí), có trích dẫn nguồn (tên file + trang/slide),
hiểu được **câu hỏi nối tiếp** trong hội thoại, **cảnh báo khi hai tài liệu nói ngược nhau**,
và có module đánh giá định lượng.

## Kiến trúc

> Chi tiết đầy đủ — input/output từng module, lý do chọn từng thư viện, toàn bộ quyết định
> thiết kế kèm số liệu đã đo: [ARCHITECTURE.md](ARCHITECTURE.md).

Hai luồng dữ liệu tách biệt, dùng chung 1 Embedding Model và 1 FAISS Index:

```
LUỒNG INGESTION (khi có tài liệu mới, bấm "Đọc tài liệu" trên UI):
Tài liệu (PDF/PPTX/DOCX)
  → Document Loader (rag/document_loader.py)   giữ metadata: tên file, trang/slide
  → Chunking (rag/chunking.py)                 Recursive Character Splitting, có overlap
  → Embedding Model (rag/embedding.py)         sentence-transformers, đa ngôn ngữ
  → FAISS Index (rag/vector_store.py)          lưu vector + metadata đi kèm

LUỒNG QUERY (mỗi lần người dùng hỏi):
Câu hỏi
  → Ngữ cảnh hội thoại (rag/tiep_noi_hoi_thoai.py)  câu NỐI TIẾP được ghép thêm các câu hỏi
                                                    trước — tất định, 0 lượt gọi LLM
  → Embedding Model (cùng model với Ingestion)
  → Tìm kiếm vector (FAISS) — có thể bật thêm nhánh BM25, hợp nhất bằng RRF CÓ TRỌNG SỐ
  → Xếp hạng lại bằng cross-encoder (rerank)
  → Dựng đoạn trích quanh chunk khớp nhất, mở rộng sang chunk liền kề cùng trang
  → Ghép Prompt (System Prompt thường hoặc prompt KIỂM CHỨNG + đoạn trích + câu hỏi)
  → LLM qua Ollama (local), TRẢ VỀ THEO LUỒNG (streaming)
  → Câu trả lời + Citation đúng nguồn đã tham chiếu (rag/citation.py)
  → Đối chiếu CHÉO các nguồn (rag/doi_chieu_nguon.py)  cảnh báo khi hai tài liệu nói ngược
                                                       nhau; chạy SAU khi đã hiện câu trả lời
```

Không có backend API riêng — `app.py` (Streamlit) gọi thẳng các hàm/class ở `rag/*.py`.

## Cấu trúc thư mục

```
rag-do-an/
├── app.py                      # Streamlit app chính
├── config.py                   # Cấu hình: model, chunk size, top_k... (đọc từ .env)
├── rag/
│   ├── document_loader.py      # Đọc PDF/PPTX/DOCX một lượt, giữ metadata
│   ├── bo_nho_dem.py           # Cache theo content hash: tài liệu, OCR, chú thích ảnh, embedding
│   ├── tai_nguyen_gpu.py       # Dò phần cứng (GPU/CPU) + quản lý VRAM theo giai đoạn
│   ├── do_thoi_gian.py         # Đo thời gian từng bước Ingestion (bảng tổng kết sau mỗi lần build)
│   ├── chunking.py             # Recursive Character Splitting
│   ├── embedding.py            # Wrapper cho sentence-transformers
│   ├── vector_store.py         # Wrapper cho FAISS (build, save, load, search)
│   ├── lexical_search.py       # BM25 - nhánh tìm kiếm theo từ khoá
│   ├── reranker.py             # Cross-encoder xếp hạng lại (tầng lọc thứ 2)
│   ├── image_extractor.py      # Trích ảnh + gắn chú thích lân cận
│   ├── vision_caption.py       # Model vision đọc nội dung trong hình + OCR dự phòng
│   ├── tiep_noi_hoi_thoai.py   # Hiểu câu hỏi nối tiếp (ghép ngữ cảnh hội thoại vào truy vấn)
│   ├── doi_chieu_nguon.py      # Phát hiện mâu thuẫn giữa các nguồn
│   ├── rag_pipeline.py         # Ghép retrieval lai + rerank + prompt + gọi LLM
│   └── citation.py             # Format câu trả lời kèm trích dẫn
├── evaluation/
│   ├── metrics.py              # Precision@K, Recall@K, MRR, Faithfulness, Relevance, Citation
│   ├── run_evaluation.py       # Chạy đánh giá, in bảng + xuất CSV
│   ├── test_questions.json     # Bộ câu hỏi test (tự chuẩn bị, xem hướng dẫn bên dưới)
│   ├── tao_tai_lieu_mau.py     # Sinh bộ tài liệu mẫu độc lập để đo (dài/ngắn/bảng/ảnh)
│   ├── kiem_dinh_judge.py      # Đo độ tin cậy của CHÍNH thước đo Faithfulness
│   ├── kiem_dinh_doi_chieu.py  # Đo độ tin cậy của cơ chế phát hiện mâu thuẫn
│   ├── kiem_dinh_viet_lai.py   # Đo nhận diện câu nối tiếp + ảnh hưởng thật lên truy xuất
│   ├── do_nguong_rerank.py     # Đo xem điểm rerank có tách được câu lạc đề không
│   ├── do_quy_mo_index.py      # Đo ngưỡng quy mô của FAISS (Flat vs IVF vs HNSW)
│   ├── do_worker_gpu.py        # Đo số worker OCR/Vision tối ưu (kèm GPU util + VRAM)
│   └── do_dau_cuoi.py          # Đo ĐẦU-CUỐI: nạp tài liệu → hỏi được, và hỏi → trả lời xong
├── data/
│   ├── raw/                    # Tài liệu gốc upload vào
│   ├── images/                 # Ảnh trích từ tài liệu (tự sinh khi build index)
│   ├── cache/                  # Bộ nhớ đệm Ingestion (xoá lúc nào cũng an toàn - xem bên dưới)
│   └── faiss_index/            # Index đã build (index.faiss + metadata.pkl + index_info.json)
├── tests/                      # pytest — chunking, retrieval, citation, streaming, OCR...
├── .streamlit/config.toml      # Bảng màu giao diện (theme chính thức của Streamlit)
├── pytest.ini                  # Cấu hình pytest + marker "slow"
├── requirements.txt
├── .env.example
├── README.md
└── ARCHITECTURE.md
```

## Cài đặt và chạy

**Yêu cầu:** Python 3.11+ (đã kiểm chứng trên 3.14) · [Ollama](https://ollama.com) đã cài và
đã `ollama pull qwen3:4b` · **không bắt buộc GPU** (embedding ~1.1GB + reranker ~2.2GB, tải 1 lần).

**Có GPU NVIDIA thì nên cài PyTorch bản CUDA** — `pip install sentence-transformers` kéo về
bản **CPU-only**, và đó là một cấu hình sai *không gây lỗi*: hệ thống vẫn trả lời đúng, chỉ
chậm hơn nhiều lần. Đo được trên RTX 5060: embedding **12,8×**, rerank **11,2×**, truy xuất
đầu-cuối **9,0×** — mà **6/6 câu hỏi cho kết quả giống hệt** (KET_QUA_DO_DAC.md §8).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Thay `cu130` cho khớp driver (`nvidia-smi` in ra CUDA Version; RTX 50xx cần ≥ cu128). Kiểm
tra: `python -c "import torch; print(torch.cuda.is_available())"`. Thanh bên của ứng dụng
cũng hiện rõ đang chạy GPU hay CPU, nên không cần nhớ kiểm tra. **Không có GPU thì bỏ qua
hoàn toàn mục này** — hệ thống tự dò và lùi về CPU, không phải chỉnh gì.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Lần đầu chạy, `sentence-transformers` tự tải model embedding và rerank từ HuggingFace về cache
local (cần Internet 1 lần; các lần sau chạy offline được, đặt `HF_HUB_OFFLINE=1` nếu muốn tắt
hẳn việc kiểm tra phiên bản).

Sao chép `.env.example` → `.env` nếu muốn đổi giá trị mặc định (không bắt buộc — thiếu `.env`
thì `config.py` tự dùng mặc định). Mọi tham số quan trọng đều nằm ở `config.py`, không hard-code
trong logic ở các file khác.

## Sử dụng

Giao diện theo lối ứng dụng chat: **thanh bên** (nguồn tài liệu) — **khung chính** (hỏi đáp, ô
nhập ghim đáy). Thu gọn thanh bên bằng nút **«** là được toàn màn hình.

1. **Thanh bên:** upload tài liệu PDF/PPTX/DOCX, tick chọn nguồn nào được dùng để trả lời, xoá
   tài liệu không cần → bấm **"Đọc tài liệu"** (chạy toàn bộ luồng Ingestion, lưu index xuống
   `data/faiss_index/`).
2. **Đặt câu hỏi** — tiếng Việt hoặc tiếng Anh, hệ thống tự trả lời đúng ngôn ngữ câu hỏi. Có
   thể **đưa ra một khẳng định để kiểm chứng** ("Pháp luật ra đời trước nhà nước, đúng không?")
   — hệ thống đối chiếu với tài liệu và kết luận ĐÚNG/SAI/KHÔNG ĐỀ CẬP kèm trích nguyên văn căn
   cứ, thay vì trả lời thuận theo giả định.
3. Câu trả lời hiện **dần theo luồng**: sau ~2 giây đã thấy "đã tìm N đoạn liên quan trong
   \<tên file\>", rồi chữ chạy tiếp. Dưới mỗi câu trả lời có dòng số liệu độ trễ (*truy xuất …
   · chữ đầu tiên … · tổng …*). Tắt bằng `BAT_STREAMING=0`.
4. **Nguồn** hiển thị dưới câu trả lời là nguồn mà chính câu trả lời đó đã tham chiếu — tên file
   + trang/slide, đủ để bạn mở tài liệu gốc tới đúng chỗ mà tự đối chiếu. Bên trong, hệ thống
   bắt LLM gắn số đoạn trích `[1]`, `[2]`… cho từng ý và đọc lại chính các số đó để biết câu
   trả lời dùng nguồn nào; các số này **không hiện ra màn hình** vì chúng là thứ tự đoạn trích
   trong prompt — một thứ tự bạn không nhìn thấy nên cũng không tra ngược được (§5.14). Giao diện cố ý KHÔNG
   in lại nguyên văn đoạn đã dùng: đoạn in kèm chỉ là bản cắt ngắn và mất định dạng gốc, dễ
   khiến người đọc dừng ở đó thay vì kiểm tra trong tài liệu thật. Câu trả lời từ chối thì không
   kèm nguồn nào.
5. **Hỏi nối tiếp được**: sau khi hỏi "Vi phạm pháp luật gồm những dấu hiệu nào?", bạn hỏi
   tiếp "Thế còn dấu hiệu thứ hai?" là hệ thống hiểu. Nó ghép các câu hỏi trước vào truy vấn
   (tất định, không tốn lượt gọi model nào) và **nói ra** việc đã làm vậy — thấy nó nối nhầm
   chỗ thì bạn hỏi lại đầy đủ hơn. Tắt bằng `BAT_TRUY_VAN_NGU_CANH=0`.
6. **Cảnh báo khi hai nguồn nói ngược nhau**: nếu câu trả lời dựa trên nhiều tài liệu mà chúng
   không thống nhất (giáo trình cũ ghi khác slide mới), hệ thống hiện một cảnh báo kèm đủ toạ
   độ hai chỗ để bạn tự mở ra đối chiếu. Nó **không tự phân xử** nguồn nào đúng — hệ thống
   không biết tài liệu nào mới hơn. Bước này chạy sau khi câu trả lời đã hiện xong nên không
   làm chậm chữ đầu tiên. Tắt bằng `BAT_DOI_CHIEU_NGUON=0`.
7. Lịch sử chat chỉ giữ trong phiên Streamlit hiện tại. Nút **＋ Hội thoại mới** xoá lịch sử
   đang hiển thị nhưng **giữ nguyên tài liệu và index**. Index đã build được lưu trên đĩa, lần
   sau mở app không cần build lại (trừ khi thêm/xoá tài liệu hoặc đổi model embedding / chunk
   size).

## Chi phí xử lý tài liệu

Nguyên tắc: **chỉ trả chi phí tính toán khi thực sự cần**. Tài liệu text tốt đọc rất nhanh;
tài liệu scan chấp nhận chậm vì OCR; tài liệu nhiều hình chỉ gọi model vision cho những hình
có giá trị; và **tài liệu đã xử lý rồi thì không xử lý lại**. Không có tuỳ chọn nào ở đây tắt
OCR, Vision hay reranker để lấy tốc độ — đó là đổi chất lượng lấy thời gian, tức giải một bài
toán khác.

**Thêm một tài liệu không phải trả giá cho cả corpus.** Mỗi tài liệu được ghi kèm băm nội dung
vào `index_info.json`; bấm "Đọc tài liệu" chỉ xử lý lại những file **mới hoặc đã đổi nội
dung**, giữ nguyên vector của phần còn lại và gỡ vector của file đã bị xoá. So bằng băm nội
dung chứ không phải thời điểm sửa file — `git checkout`, sao chép file hay đồng bộ cloud đều
đổi `mtime` mà không đổi nội dung. Tắt bằng `BAT_INDEX_TANG_DAN=0`.

**Bộ nhớ đệm ở `data/cache/`** giữ lại bốn thứ đắt nhất, tất cả đánh khoá theo băm **nội
dung**: kết quả đọc từng tài liệu, OCR từng trang, chú thích ảnh của model vision, và vector
embedding của từng chunk. Nhờ khoá theo nội dung, một hình dùng lại ở 20 slide chỉ tốn **một**
lượt gọi model vision, và đổi tên file không làm mất cache. Đổi một tuỳ chọn ăn vào kết quả
đọc (bật/tắt OCR, DPI render, ngưỡng dính chữ…) thì cache **tự trượt** — không có chuyện hệ
thống lặng lẽ trả kết quả theo cấu hình cũ.

Xoá cả thư mục `data/cache/` bất cứ lúc nào cũng **an toàn tuyệt đối**: mọi thứ trong đó đều
tính lại được, chỉ mất thời gian chứ không mất dữ liệu. Thanh bên có sẵn nút **"Xoá cache"**
kèm dung lượng đang chiếm. Tắt hẳn cache bằng `BAT_CACHE_INGESTION=0` (dùng khi cần **đo** chi
phí thật của một lần build từ đầu).

**Chỗ nào đang chậm thì có số để trả lời.** Sau mỗi lần "Đọc tài liệu", log in một bảng tổng
kết thời gian từng bước (đọc text, nhận diện tiêu đề, render OCR, gọi model vision, chunking,
embedding…) kèm số lần gọi và phần trăm. Tắt bằng `BAT_PROFILING_INGESTION=0`.

Bảng dưới là output thật khi đọc giáo trình Bishop (758 trang, cache tắt để đo chi phí gốc):

```
PROFILING INGESTION (tổng 45.0s)
BƯỚC                               SỐ LẦN   TỔNG (s)   TB (ms)      %
---------------------------------------------------------------------
pdf_doc_text_trang                    758       39.5      52.1  87.8%
pdf_trich_anh                           1        4.1    4121.5   9.2%
pdf_nhan_dien_tieu_de                 758        0.7       1.0   1.6%
```

Đọc bảng này ra được ngay một kết luận: với tài liệu **thuần chữ** thì 87,5% thời gian nằm ở
việc đọc text, chứ không phải ở trích ảnh hay nhận diện tiêu đề — tức mọi nỗ lực tối ưu hai
bước kia đều là tối ưu nhầm chỗ. Với tài liệu **scan** thì bức tranh đảo ngược hoàn toàn:
OCR chiếm gần như toàn bộ (đo được 5,3 giây **mỗi trang** trên GPU, xem KET_QUA_DO_DAC.md §8.5).

**Ảnh không mang nội dung bị loại trước khi render.** Icon, logo góc trang, dải trang trí và
hình lặp lại kiểu watermark đều bị lọc trước cả bước render — mỗi ảnh giữ lại kéo theo một
lượt render, một file trên đĩa, một lượt gọi model vision (~1,9 giây) và một vector trong
index. Đo trên corpus thật, số bản ghi ảnh giảm 401 → 261 ở một bài giảng IoT và 219 → 168 ở
một file DOCX, **trong khi số bản ghi văn bản không đổi một đơn vị nào**.

## Tận dụng phần cứng

Hệ thống **tự dò phần cứng** rồi tự chọn cách chạy — không có gì phải cấu hình bằng tay, và
không có hằng số nào hiệu chỉnh riêng cho một loại card. Có GPU dùng được thì embedding và
reranker chạy trên đó; không có thì lùi về CPU và mọi thứ vẫn đúng, chỉ chậm hơn.

| Tham số | Suy từ đâu |
|---|---|
| Thiết bị cho embedding / rerank | có CUDA hay không (ép riêng từng vai trò bằng `THIET_BI_*`) |
| Batch size encode | VRAM **còn trống**, chặn trên bởi `EMBEDDING_BATCH_SIZE` |
| Số worker OCR/Vision | min(trần cấu hình, số nhân CPU, VRAM còn trống) |

**Quản lý VRAM theo giai đoạn.** Đo thật bằng `nvidia-smi`, ba model của giai đoạn truy vấn
cộng lại **8,07 GB** (LLM 4,75 + reranker 2,20 + embedding 1,12) — **không vừa card 8 GB**.
Mà tràn VRAM thì *không báo lỗi*: quan sát được VRAM còn trống tụt xuống **288 MB**, model
reranker mất ~58 giây mới nạp xong, và lượt hỏi đầu tiên báo **50,8 giây** cho bước truy
xuất. Không một dòng lỗi nào.

Vì người dùng bấm "Đọc tài liệu" rồi mới hỏi, hai giai đoạn không bao giờ chạy cùng lúc:

```
INGESTION  →  Vision/OCR + embedding(GPU)   … kết thúc: còn 0,79 GB trống
    ↓ nhả model vision · dọn bộ đệm CUDA    … còn 5,70 GB
    ↓ card chật → embedding xuống CPU       … còn 6,79 GB
QUERY      →  reranker(GPU) + LLM(GPU)
```

Thứ bị hy sinh là embedding, vì lúc truy vấn nó chỉ mã hoá 1–3 chuỗi ngắn (GPU nhanh hơn CPU
đúng 14 ms), còn reranker chấm vài chục cặp mỗi câu (GPU nhanh hơn 11,2×). Kết quả: truy
xuất còn **0,45 giây** mỗi câu. Card lớn hơn `VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB` (mặc định
10 GB) thì giữ tất cả trên GPU, không phải đánh đổi gì.

**Đo trên máy của bạn** — con số tối ưu khác nhau ở mỗi máy, đừng chép của người khác:

```bash
python evaluation/do_worker_gpu.py
```

```bash
python evaluation/do_dau_cuoi.py
```

Script thứ nhất thử lần lượt 1/2/4 worker OCR, đo kèm GPU utilization và VRAM, rồi tự đề
xuất giá trị `SO_WORKER_VISION`. Script thứ hai đo **đầu-cuối**: từ lúc đưa tài liệu vào tới
lúc hỏi được, và từ lúc gửi câu hỏi tới lúc nhận câu trả lời hoàn chỉnh — cộng các phép đo
lẻ lại luôn ra con số nhỏ hơn thực tế, vì chúng đều đo với model đã nạp sẵn và cache đã ấm.

**Nút thắt hiện nay nằm ở đâu** (đo được, không phải phỏng đoán):

| Giai đoạn | Chi phí lớn nhất | Tỉ trọng |
|---|---|---:|
| Nạp tài liệu lần đầu (tài liệu nhiều hình) | chú thích ảnh bằng model vision | **89,8%** |
| Nạp lại (cache đầy) | — (3,48 s cho 1209 chunk, nhanh hơn 81×) | — |
| Mỗi câu hỏi | LLM sinh chữ (chủ yếu là suy luận nội bộ của qwen3) | **~98%** |

Truy xuất chỉ còn **0,45 giây** mỗi câu (trung vị), tức khoảng **1,3%** thời gian một lượt
hỏi — không còn là chỗ đáng tối ưu tiếp. Trong 0,45 giây đó: rerank 0,45 s, mã hoá câu hỏi
0,02 s, còn FAISS thì dưới 1 mili giây.

## Sự cố thường gặp

**`ConnectError: [WinError 10061] …` / `Failed to connect to Ollama`** — máy chủ Ollama chưa
chạy. Ollama là tiến trình nền riêng, KHÔNG tự khởi động cùng `streamlit run app.py`: chưa bật
thì hệ thống vẫn đọc và truy xuất tài liệu bình thường nhưng không sinh được câu trả lời nào.

1. Mở ứng dụng **Ollama** (Windows: biểu tượng ở khay hệ thống), hoặc chạy `ollama serve`.
2. Kiểm tra: `ollama list` — chưa thấy `qwen3:4b` thì `ollama pull qwen3:4b`.
3. Ollama chạy ở máy/cổng khác thì sửa `OLLAMA_HOST` trong `.env`.

Thanh bên có sẵn cảnh báo cho cả hai trường hợp và tự biến mất trong ~10 giây sau khi sửa xong.

**Câu trả lời nào cũng là "Không tìm thấy thông tin trong tài liệu"** — thường do bỏ tick hết
nguồn ở thanh bên, hoặc chưa bấm **Đọc tài liệu** sau khi thêm tài liệu mới. Cũng có thể câu
hỏi thật sự nằm ngoài phạm vi tài liệu — đó là hành vi đúng.

**Câu trả lời cụt ngủn** — cửa sổ ngữ cảnh của Ollama mặc định chỉ 4096 token bất kể model hỗ
trợ bao nhiêu, và nó chứa **prompt + thinking + câu trả lời**. Đo thật trên corpus của đồ án:
một câu hỏi thường tốn **7.001 token** tổng cộng, câu yêu cầu "liệt kê đầy đủ" tốn **10.860** —
tức gấp 2,7 lần cửa sổ mặc định. Model viết được vài dòng rồi chạm trần và dừng, **không có lỗi
nào báo ra**. Đã sửa bằng `OLLAMA_NUM_CTX=16384`; hệ thống nay ghi `prompt_eval_count` và
`done_reason` thật vào log mỗi lượt, kèm cảnh báo `PROMPT BỊ CẮT`. Chi tiết: ARCHITECTURE.md §5.60.

**Đừng hạ `OLLAMA_NUM_CTX` để lấy lại tốc độ** — hạ `TOP_K` hoặc `NGAN_SACH_KY_TU_MOI_DOAN` thay
vào đó. Hạ `num_ctx` không làm prompt ngắn đi, nó chỉ khiến prompt bị cắt trở lại.

**Trả lời chậm hẳn sau khi cập nhật** — `OLLAMA_NUM_CTX=16384` khiến KV-cache lớn hơn và bước
prefill lâu hơn (trên CPU, prefill ~5000 token có thể mất 10–20 giây). Đây là cái giá đã biết
của việc sửa bug trên. `TOP_K` **đã được hạ 6 → 4** để bù lại (xem bảng đo bên dưới); nếu vẫn
chậm thì hạ tiếp `NGAN_SACH_KY_TU_MOI_DOAN` (1600 → 1200). **Không phải hạ `num_ctx`** — làm
vậy là tái tạo lại đúng bug prompt bị cắt im lặng.

## Kiểm thử

```bash
pytest tests/ -v
```

**402 test.** Một số ít phải nạp model thật nên chạy chậm; bỏ qua chúng bằng
`pytest -m "not slow"` trong lúc đang sửa code.

`tests/conftest.py` trỏ `data/cache/` và `data/images/` sang thư mục tạm cho cả phiên test —
không test nào được ghi vào thư mục dữ liệu thật của dự án. Đây là loại lỗi **không làm test
đỏ** nên sẽ không ai phát hiện qua CI, vì vậy phải chặn từ gốc một lần cho mọi test.

## Đánh giá (Evaluation)

1. Build index từ tài liệu thật của bạn (qua UI).
2. Điền câu hỏi test vào `evaluation/test_questions.json`:
   ```json
   [
     {
       "cau_hoi": "Học máy có giám sát là gì?",
       "cac_trang_dung": [{"nguon": "ten_file.pdf", "trang": 3}],
       "dap_an_mau": "Là phương pháp huấn luyện mô hình dựa trên dữ liệu đã có nhãn."
     }
   ]
   ```
   - `cac_trang_dung`: danh sách (nguồn, trang) **đúng** chứa câu trả lời — dùng để tính
     Precision@K/Recall@K. Câu cố tình không có đáp án (test hành vi từ chối) thì để `[]`.
   - Chuẩn bị 15–30 câu để kết quả đáng tin cậy.
   - Hai trường **tuỳ chọn** giúp tách bảng kết quả theo nhóm: `"loai_tai_lieu"` (vd `"dai"`,
     `"co_bang"`, `"co_anh"`) và `"loai_cau_hoi"` (vd `"truy_xuat"`, `"kiem_chung"`,
     `"tu_choi"`). Có chúng thì `run_evaluation.py` in thêm bảng tách theo từng nhóm — đây
     chính là thứ trả lời câu "hệ thống có ổn định trên nhiều loại tài liệu khác nhau không".
   - Chưa có tài liệu để thử? `python evaluation/tao_tai_lieu_mau.py` sinh sẵn một bộ tài liệu
     đa dạng (dài/ngắn/có bảng/có ảnh/nhiều mục na ná nhau) để chạy thử đường ống.
3. Chạy:
   ```bash
   python evaluation/run_evaluation.py
   ```
   In bảng Precision@K / Recall@K / Faithfulness / Answer Relevance / Citation accuracy, đồng
   thời xuất chi tiết ra `evaluation/ket_qua_danh_gia.csv`.

   **Chạy đầy đủ rất chậm** — mỗi câu tốn 1 lượt LLM sinh câu trả lời (~40 giây trên CPU) cộng
   nhiều lượt chấm điểm (faithfulness chấm 3 lần lấy trung vị, relevance, và 1 lượt cho MỖI ý
   có trích dẫn). Bộ 29 câu mất khoảng 60–90 phút. Trong lúc đang sửa code dùng chế độ nhanh —
   chỉ đo truy xuất, không gọi LLM, xong trong vài giây:
   ```bash
   python evaluation/run_evaluation.py --nhanh
   ```
   So sánh với lần chạy trước: chép `ket_qua_danh_gia.csv` thành `ket_qua_danh_gia_truoc.csv`
   trước khi chạy lại, script sẽ tự in bảng chênh lệch từng metric.

Ngoài ra có bốn script kiểm định — mỗi cái đo độ tin cậy của MỘT cơ chế trên bộ ca đã biết
trước đáp án, và con số rút ra là thứ nên đặt cạnh tính năng trong báo cáo:

```bash
python evaluation/kiem_dinh_judge.py                 # thước đo Faithfulness (§5.43)
python evaluation/kiem_dinh_doi_chieu.py --so-lan 3  # cơ chế phát hiện mâu thuẫn (§5.59)
python evaluation/kiem_dinh_viet_lai.py --chi-tang-1 # nhận diện câu nối tiếp (§5.58)
python evaluation/kiem_dinh_viet_lai.py --truy-xuat "Một câu hỏi đầy đủ về tài liệu của bạn"
python evaluation/do_quy_mo_index.py                 # ngưỡng quy mô FAISS trên máy bạn (§5.44)
```

### Bộ HELD-OUT — đo mức overfit, không chỉ đo điểm

Mọi hằng số của hệ thống (ngưỡng cosine, ngưỡng rerank, trần đoạn mỗi trang, số ứng viên
rerank, chunk size) đều được chọn bằng cách tối ưu trên chính `test_questions.json`, tức
**tuning trên tập test**. Điểm in-sample vì thế luôn đẹp và không nói được gì về tài liệu mới —
đúng lý do hệ thống từng "tụt hạng" khi gặp corpus lạ.

`evaluation/test_questions_held_out.json` là **46 câu (44 có đáp án) trên 12 tài liệu chưa từng
dùng để chỉnh bất kỳ tham số nào**, phủ đủ các nhóm của bộ in-sample (truy xuất, chéo ngôn ngữ,
đọc bảng, kiểm chứng khẳng định, từ chối câu lạc đề).

> Với 6 tài liệu `Bai*.docx` (bài giảng Khai phá dữ liệu), DOCX gần như không có ngắt trang nên
> **Recall@K suy biến** — xem cảnh báo ở ARCHITECTURE.md §5.64. Đọc kèm Citation accuracy.

```bash
python evaluation/run_evaluation.py --nhanh --held-out  # chỉ bộ held-out
python evaluation/run_evaluation.py --khoang-cach       # cả hai bộ + chênh lệch
```

Chênh lệch Recall@K giữa hai bộ **chính là con số đo mức overfit**. Cách đọc:

| Quan sát | Kết luận |
|---|---|
| Khoảng cách thu hẹp | thay đổi giúp hệ thống tổng quát hơn |
| Cả hai cùng tăng, khoảng cách giữ nguyên | thay đổi tốt nhưng không chữa overfit |
| In-sample tăng, held-out giảm | đang tối ưu vào đúng bộ tài liệu cũ — phải bỏ |

> **Quy tắc bất di bất dịch:** không bao giờ chỉnh tham số theo kết quả của bộ held-out.
> Chỉnh một lần là nó trở thành bộ in-sample thứ hai và con số này mất sạch ý nghĩa.

**Kết quả đo** (corpus đầy đủ 26 tài liệu / 9285 chunk chứa cả hai bộ):

| Metric | in-sample (25 câu) | held-out (44 câu) | khoảng cách |
|---|---|---|---|
| Recall@K | 0.845 | **0.905** | **−0.060** |
| MRR | 0.980 | 0.843 | **+0.137** |
| Đoạn đúng ở hạng 1 | 24/25 (96%) | 33/44 (75%) | −21 điểm % |
| Faithfulness | 0.980 | 0.977 | **+0.003** |
| Citation accuracy | 0.714 | **0.784** | **−0.070** |
| P@K | 0.620 | 0.364 | +0.256 *(không so được)* |

Kết quả chia làm ba phần, và đó mới là thông tin có giá trị:

- **Tìm đúng nội dung: tổng quát tốt.** Recall@K trên tài liệu chưa từng thấy còn cao hơn.
  Chunking + embedding + truy hồi **không** overfit vào corpus cũ.
- **Xếp đúng thứ tự: có overfit thật.** MRR 0.980 → 0.843, tỷ lệ đoạn đúng ở hạng 1 rơi từ 96%
  xuống 75%. Đúng vào tầng mà mọi ngưỡng của hệ thống đang tác động (rerank + các ngưỡng lọc).
- **Chất lượng câu trả lời: không overfit** (Faithfulness chênh 0.003) — nhưng kết luận này đã
  từng bị đo SAI: bộ held-out 22 câu cho khoảng cách +0.155, nhân đôi bộ lên 44 câu thì co về
  +0.003. Cùng lúc khoảng cách MRR lại **nở ra** (+0.101 → +0.137). **Thêm mẫu rồi xem khoảng
  cách nở ra hay co lại** là phép thử rẻ nhất để phân biệt tín hiệu thật với nhiễu — chi tiết
  ở ARCHITECTURE.md §5.64.
- **P@K không so được giữa hai bộ.** Precision@K phụ thuộc số trang đúng mỗi câu: in-sample
  trung bình 2.84 trang/câu, held-out chỉ 1.20. Chênh lệch +0.256 phần lớn là hiện vật của
  cách ra đề, không phải bằng chứng overfit.

Chi tiết đầy đủ: [`KET_QUA_DO_DAC.md`](KET_QUA_DO_DAC.md) · ARCHITECTURE.md §5.64.

### ⚠ Ba lưu ý bắt buộc đọc trước khi diễn giải số liệu

**1. Phần lớn metric KHÔNG tất định.** Đã đo trực tiếp: cùng một câu hỏi, cùng index, cùng
prompt, chạy 4 lần thì có lần model gắn 6 số trích dẫn, có lần không gắn số nào; giám khảo chấm
cùng một ca 8 lần cho `[0,1,1,1,1,1,1,1]`.

| Metric | Tất định? | Diễn giải chênh lệch |
|---|---|---|
| Precision@K, Recall@K, MRR | **Có** | So sánh trực tiếp được |
| Faithfulness, Answer Relevance | Đã bớt dao động (trung vị 3 lần) | Chênh lệch nhỏ vẫn cần dè dặt |
| **Citation accuracy** | **Không** — dao động mạnh nhất | Chênh dưới ~0.1 **không nên diễn giải là gì cả** |

Muốn con số chắc chắn để chốt cho báo cáo: chạy `run_evaluation.py` **ba lần rồi lấy trung vị
từng metric** (~4,5 giờ). Đáng làm một lần trước khi nộp, không đáng làm trong lúc đang sửa code
(lúc đó dùng `--nhanh`, tất định hoàn toàn). Chi tiết: ARCHITECTURE.md §5.46.

**2. Precision@K/Recall@K SUY BIẾN với DOCX không có ngắt trang.** DOCX không có khái niệm
"trang" cố định, nên file không có ngắt trang nào được coi là MỘT trang. Hai metric này so khớp
theo (nguồn, trang), nên với file như vậy chỉ cần lấy về một chunk bất kỳ là Recall@K = 1.00 —
bất kể chunk đó có chứa câu trả lời hay không. Với loại tài liệu này chỉ Citation accuracy còn
mang thông tin (§5.39).

**3. Câu hỏi cố tình không có đáp án** (`cac_trang_dung: []`) luôn cho Precision@K/Recall@K = 0
theo định nghĩa; Faithfulness của câu từ chối luôn được chấm 1.0, còn Answer Relevance có thể
thấp vì câu trả lời không cung cấp nội dung cụ thể — đây là quy ước bình thường của metric này
(giống cách RAGAS xử lý câu trả lời "không biết"), không phải lỗi hệ thống.

**Kiểm định giám khảo, đã chạy:** `python evaluation/kiem_dinh_judge.py --so-lan 3` cho
**7/7 ca đúng khoảng, dao động giữa các lần chấm 0,00**. Nên phát biểu là *"Faithfulness
0,980, đo bằng thước đo đã kiểm định đúng 7/7 ca"* thay vì con số trơ trọi. Lưu ý 7 ca này
dứt khoát rõ ràng; các câu dao động trong lần chạy thật là câu nằm ở ranh giới — giám khảo
không hỏng, nhưng cũng không phân xử được ca mập mờ.

**Thước đo tự kiểm tra chính nó.** LLM-as-judge có sai số và sai số đó **không có triệu chứng**:
điểm 0.0 chấm cho một câu trả lời đúng trông y hệt điểm 0.0 chấm cho một câu bịa. Bốn cơ chế đã
thêm (chi tiết §5.43, §5.48): chấm 3 lần lấy trung vị · cờ `!` cạnh điểm Faithfulness đáng ngờ
(giám khảo chấm ≤ 0.5 *đồng thời* mọi câu đều có nguyên văn trong ngữ cảnh — hai điều không thể
cùng đúng) · bộ kiểm định giám khảo 7 ca · chặn điểm ngoài thang [0,1] (một lần chạy thật đã
nhận về `100.0` và `5.0`, chỉ 2 trong 29 câu đủ kéo Faithfulness từ 0.88 lên **4.43**).

## Kết quả đánh giá trên tài liệu thật

> **Số liệu mới nhất nằm ở [`KET_QUA_DO_DAC.md`](KET_QUA_DO_DAC.md)** — đo trên **26 tài liệu
> / 3.648 trang / 9.285 chunk**, kèm môi trường đo, toàn bộ tham số, khoảng cách
> in-sample vs held-out và cách tái lập từng con số. Bảng bên dưới là kết quả của corpus 13
> tài liệu ở đợt đo trước, giữ lại để đối chiếu — **hai bảng không so trực tiếp được với nhau**
> vì khác corpus, khác `TOP_K`, và Recall@K đã đổi ý nghĩa (xem §6.1 của file đó).

13 tài liệu thật (1221 trang PDF + PPTX + DOCX, **5642 chunk**), 29 câu hỏi song ngữ. Chi tiết
và cách diễn giải: ARCHITECTURE.md §5.38.

| Loại tài liệu | Số câu | Recall@K | Faithfulness | Relevance | Citation |
|---|---|---|---|---|---|
| Biểu mẫu có bảng (DOCX) | 4 | 1.00 | 0.75 | 1.00 | 0.47 |
| Slide tiếng Anh | 8 | 0.95 | 0.88 | 1.00 | 0.87 |
| Giáo trình dài tiếng Việt (237 trang) | 6 | 0.94 | 0.92 | 1.00 | 0.38 |
| Slide tiếng Việt (PPTX) | 4 | 0.87 | 1.00 | 1.00 | 0.81 |
| Sách dài tiếng Anh (758 trang) | 3 | 0.83 | **0.83** *(trước khi sửa: 0.33)* | 1.00 | 0.46 |

| Loại câu hỏi | Số câu | Recall@K | Faithfulness | Relevance | Citation |
|---|---|---|---|---|---|
| Đọc bảng | 3 | 1.00 | 0.67 | 1.00 | 0.58 |
| Kiểm chứng khẳng định sai | 3 | 1.00 | 1.00 | 1.00 | 0.63 |
| Truy xuất thường | 14 | 0.94 | 0.86 | 1.00 | 0.58 |
| Chéo ngôn ngữ | 5 | 0.83 | 1.00 | 1.00 | 0.80 |
| Từ chối câu lạc đề | 4 | — | đúng hành vi | — | — |

**Trung bình toàn bộ:** Precision@K 0.44 · Recall@K 0.80 · Faithfulness **0.86** (0.93 nếu loại
2 câu bị cờ tự nghi ngờ đánh dấu) · Answer Relevance 0.90 · Citation 0.61 · 38,7 giây/câu.
Chỉ đo riêng truy xuất (`--nhanh`, **tất định**): **Recall@K 0.93, MRR 0.98, 24/25 câu có đoạn
đúng ở hạng 1**.

### Đo lại sau đợt sửa `num_ctx` + truy xuất

Bật lần lượt từng thay đổi trên **cùng một index** (12 tài liệu, 5554 chunk, 29 câu hỏi song
ngữ, rerank BẬT, chế độ `--nhanh` nên **tất định**):

| Cấu hình | P@K | Recall@K | MRR | hạng 1 | chặn lạc đề |
|---|---|---|---|---|---|
| GỐC (trước khi sửa) | 0.500 | 0.937 | 0.980 | 24/25 | 2/4 |
| + ngưỡng tương đối | 0.500 | 0.937 | 0.980 | 24/25 | 2/4 |
| + mở rộng xuyên trang | 0.467 | 0.875 | 0.980 | 24/25 | 2/4 |
| + trần trang thích ứng | **0.567** | 0.937 | 0.980 | 24/25 | 2/4 |
| + BM25 cứu hộ | 0.507 | **0.945** | 0.980 | 24/25 | 2/4 |

- **Ngưỡng tương đối không đổi con số nào** — đúng thiết kế: 0.78 tái tạo chính điểm hiệu
  chỉnh cũ (0.70 / 0.90) nên trên corpus cũ nó *phải* không đổi. Giá trị của nó nằm ở corpus
  có phân bố cosine khác.
- **Trần trang thích ứng: +0.067 P@K**, không đụng tới MRR lẫn khả năng chặn câu lạc đề.
- **BM25 cứu hộ: +0.008 Recall@K** — nhỏ đúng như dự đoán, và quan trọng hơn là **không làm
  hỏng gì**, đúng thiết kế "xấu nhất là không cải thiện".
- **Mở rộng xuyên trang trông như làm tụt Recall@K 0.062 — nhưng chính metric mới là thứ
  sai.** `Recall@K` chỉ đếm *trang neo*; khi một đoạn trích nuốt sang trang liền kề thì trang
  đó không còn được neo, dù nội dung vẫn nằm nguyên trong ngữ cảnh gửi cho LLM. Đo bằng chỉ
  số không bị hiện vật này (*Recall phủ* — trang đúng có nằm trong các trang mà đoạn trích đi
  qua không):

  | | P@K | Recall@K (neo) | **Recall phủ** |
  |---|---|---|---|
  | TẮT mở rộng xuyên trang | 0.580 | 0.945 | 0.945 |
  | BẬT mở rộng xuyên trang | 0.540 | 0.875 | **0.959** |

  Không câu nào mất nội dung; ngữ cảnh phủ 10–16 trang thay vì đúng 6. Chi tiết và bài học
  rút ra: ARCHITECTURE.md §5.65.

> Hệ quả cần nhớ khi đọc báo cáo: **Recall@K trước và sau đợt sửa này không so trực tiếp được
> nữa**, vì mở rộng xuyên trang phá vỡ giả định "mỗi đoạn trích nằm gọn trong một trang" mà
> metric dựa vào. Con số 0.875 được giữ nguyên trong bảng thay vì sửa cho đẹp.

**Kiểm chứng chống overfitting** trên bộ tài liệu ĐỘC LẬP (`python evaluation/tao_tai_lieu_mau.py`
sinh 6 tài liệu + 26 câu hỏi riêng, không dùng để tinh chỉnh gì): toàn bộ 24 câu có đáp án đạt
**Recall@K 1.00**, MRR 0.95, 22/24 câu đúng ở hạng 1, 2 câu lạc đề bị chặn đúng, và **0 chunk
vượt giới hạn model**. Ngoài ra `tests/test_khai_quat_tai_lieu.py` kiểm theo **hình dạng** chứ
không theo tài liệu (bảng chỉ có tiêu đề, bảng 8 cột, ô chứa nguyên một bài văn…). Chi tiết §5.45.

**Hai chỗ phải đọc kỹ khi so với số liệu bản trước:**

1. **Faithfulness nhóm sách tiếng Anh 0.33 → 0.83** là kết quả của việc sửa **nguyên nhân gốc**
   ở khâu đọc PDF (§5.40), không phải chỉnh thước đo cho đẹp: trước đây ngữ cảnh trích ra bị
   dính chữ nên giám khảo không đối chiếu được và chấm 0 cho những câu trả lời hoàn toàn đúng.
2. **Citation accuracy 0.76 → 0.61 KHÔNG phải hệ thống kém đi** mà là thước đo nghiêm hơn: bản
   trước loại khỏi trung bình mọi câu không dẫn nguồn, nay chỉ câu **từ chối** mới được loại,
   còn câu trả lời thật mà quên dẫn nguồn bị tính **0 điểm** (§5.28). Đo cùng lần chạy này theo
   cách cũ cho **0.72** — nằm trong dải dao động của chính metric đó.

## Kết quả đo hai tính năng hội thoại

**Hiểu câu hỏi nối tiếp** — đo trên index thật, lấy chính kết quả truy xuất của câu hỏi ĐẦY ĐỦ
làm chuẩn vàng (nhờ vậy chạy được trên bất kỳ corpus nào, không cần gán nhãn tay). Phép đo
**tất định**, nên chênh lệch là chênh lệch thật chứ không phải dao động của model:

| Câu nối tiếp | Trùng chuẩn vàng | Điểm rerank |
|---|---|---|
| "Giải thích thêm đi" | 0/4 → **4/4** | 0.0328 → **0.8832** |
| "Cho ví dụ" | 0/4 → **4/4** | 0.2698 → **0.8324** |
| "Tell me more" | 0/4 → **4/4** | 0.2055 → **0.8377** |
| "Cái đó cụ thể là thế nào?" | 1/4 → **4/4** | 0.0219 → **0.8831** |
| **Tổng** | **1/16 → 16/16** | |

Cột đáng lo nhất là điểm rerank, không phải cột trùng khớp: 0.0219 và 0.0328 nằm sát ngưỡng
từ chối (`NGUONG_DIEM_RERANK_TOI_THIEU`), tức những câu nối tiếp hoàn toàn hợp lệ đang ở ranh
giới **mất hẳn câu trả lời** chứ không chỉ "lấy nhầm đoạn". Tầng nhận diện câu nối tiếp đạt
**10/10** trên bộ ca có nhãn. Chi tiết: ARCHITECTURE.md §5.58.

**Phát hiện mâu thuẫn giữa các nguồn** — bộ kiểm định 7 ca, chạy 3 lần:

| | Kết quả |
|---|---|
| Đúng | **7/7**, ổn định qua 3 lần chạy |
| Trong đó **im lặng đúng** | **3/3** — không có báo động giả nào |
| Tầng lọc tất định | chặn 2/3 ca im lặng trước khi tốn lượt gọi LLM nào |

Ba trên bảy ca là ca **phải im lặng**, và đó mới là phần khó: bắt mâu thuẫn hiển nhiên thì dễ,
khó là không báo động trên hai đoạn chỉ bổ sung cho nhau. Chi tiết: ARCHITECTURE.md §5.59.

## Các quyết định kỹ thuật quan trọng

Lý do đầy đủ nằm trong comment ngay tại chỗ trong code và ở ARCHITECTURE.md §5. Tóm tắt:

- **`IndexFlatIP` thay vì `IndexFlatL2`**: vector đã chuẩn hoá nên inner product == cosine
  similarity — đúng thước đo ngữ nghĩa cần dùng, thay vì khoảng cách Euclid thô.
- **Model embedding `multilingual-e5-base`**: model huấn luyện cho *retrieval* (câu hỏi ngắn ↔
  đoạn tài liệu dài), không phải cho *paraphrase* như lựa chọn ban đầu — giới hạn 512 token
  thay vì 128. Câu hỏi và tài liệu mã hoá bằng 2 hàm riêng vì họ E5 cần tiền tố
  `query: `/`passage: ` khác nhau (§5.19).
- **Chunk 160 token, overlap 32**, đo bằng **đúng tokenizer của model**: tiktoken đếm gấp ~1.9
  lần trên tiếng Việt, khiến chunk nhỏ hơn dự định rất nhiều và nội dung bị băm vụn. Chunk tự
  co lại nếu vượt giới hạn model đang dùng (§5.2, §5.3, §5.18).
- **Đoạn trích dựng quanh chunk khớp**, không gộp nguyên trang: giữ được phần bị chunking cắt
  ngang mà không kéo theo cả trang không liên quan (§5.11).
- **Rerank bằng cross-encoder** trên 30 ứng viên: +6 giây nhưng MRR 0.417 → 0.642. Điểm cosine
  được giữ nguyên, rerank chỉ đổi thứ tự chọn (§5.24).
- **BM25 mặc định TẮT** — một kết quả âm tính đo được: trên corpus song ngữ, BM25 không giúp gì
  ngay trên sở trường của nó (từ khoá hiếm đã đạt MRR 1.000 nhờ riêng dense) và gây hại nặng
  cho truy xuất chéo ngôn ngữ (0.703 → 0.536). Code giữ nguyên, bật lại bằng `TRONG_SO_BM25`
  nếu corpus của bạn khác (§5.30).
- **Từ chối câu lạc đề dựa trên điểm rerank, không phải cosine**: đã đo, không tồn tại ngưỡng
  cosine nào tách được (câu tiếng Anh đúng chủ đề cho cosine *thấp hơn* câu tiếng Việt lạc đề);
  điểm rerank thì tách được ~37.000 lần (§5.29, §5.57).
- **Phát hiện khẳng định sai**: câu dạng "… đúng không?" đi theo system prompt riêng, bắt buộc
  ra phán quyết ĐÚNG/SAI/KHÔNG ĐỀ CẬP kèm trích nguyên văn căn cứ, và kết luận phải viết SAU
  phần đối chiếu chứ không phải trước (§5.22, §5.56).
- **Trả lời theo luồng**: không làm model nhanh hơn, nhưng đổi hẳn cảm nhận — dấu hiệu đầu tiên
  sau ~2 giây thay vì spinner câm cả phút. Chỉ có MỘT đường gọi LLM: bản không streaming dùng
  cho evaluation gom lại từ chính generator mà giao diện dùng, để số đo trong báo cáo luôn nói
  về thứ người dùng thấy (§5.42).
- **Không trình bày phỏng đoán như thể là nguồn**: khi model không tự gắn số `[n]`, giao diện
  nói rõ *"câu trả lời không tự dẫn nguồn — đoạn dưới là đoạn liên quan nhất do hệ thống chọn"*.
  Đo được: 4/29 câu rơi vào diện này (§5.54).
- **Chỉ báo bám nguồn** (*"✓ 78% nội dung trùng nguyên văn với đoạn trích đã dẫn"*) — phép đo
  tất định, không gọi model. Chỉ hiện khi CAO: mức thấp không chứng minh được gì vì diễn đạt
  lại bằng lời của mình cũng cho mức thấp (§5.55).
- **Tự OCR tài liệu scan** (mặc định BẬT): PDF scan trước đây cho ra index rỗng mà **không báo
  lỗi gì** — đã gặp thật với một giáo trình 383 trang. Nay trang nào ĐO ĐƯỢC là không đọc được
  thì tự OCR; tài liệu có lớp text bình thường tốn **đúng 0 chi phí** (§5.37, §5.49).
- **Đọc PDF nhiều cột theo từng cột**: pdfplumber đọc theo dòng ngang suốt bề ngang trang nên
  trang 2 cột bị nối câu cột trái vào cột phải thành câu vô nghĩa. Rãnh giữa cột chỉ được công
  nhận khi có chữ ở **cả hai bên** (§5.51).
- **Đọc lại trang PDF bị dính chữ**: font sát nhau của sách LaTeX khiến pdfplumber nuốt mọi
  khoảng trắng. Chỉ trang nào ĐO ĐƯỢC là dính mới đọc lại, tham số **dò theo từng trang** và
  bản đọc lại chỉ được nhận khi vừa đỡ dính hơn vừa không làm vỡ từ thêm (§5.40, §5.45).
- **Bảng quá lớn cắt theo HÀNG, lặp lại dòng tiêu đề**: bản cũ đẩy bảng vượt giới hạn vào
  splitter văn xuôi nên mọi mảnh sau mất dòng tiêu đề cột — nguyên nhân của một câu trả lời SAI
  thật đã đo được (§5.41).
- **Đọc được text box trong DOCX** (sơ đồ, khung "Lưu ý", trích dẫn nổi bật) và **một file hỏng
  không làm sập cả lần build** (§5.52, §5.53).
- **Ingestion đọc MỘT LƯỢT, và không đọc lại thứ đã đọc**: bản trước duyệt mỗi PDF hai lần và
  đọc lại mỗi trang dính chữ tới 5 lần. Nay mỗi trang đi qua đúng một lượt, mức `x_tolerance`
  được hiệu chỉnh theo tài liệu, và kết quả đọc / OCR / chú thích ảnh / embedding đều được
  cache theo **băm nội dung**. Đo trên Bishop (trung vị 3 lần): 58,8s → **48,3s** với **cùng
  809 bản ghi** và chữ tách **tốt hơn** ở 540 trang; lần đọc thứ hai gần như bằng 0 (§5.66).
- **Song song hoá đúng chỗ có lợi**: OCR và chú thích ảnh chạy nhiều luồng vì chúng ngồi chờ
  Ollama trả lời — nhưng mức lợi nhỏ hơn trực giác nhiều: GPU đã bão hoà ~85% ngay từ MỘT
  worker, nên 1→2 worker lợi 1,55× còn **2→4 không lợi gì** (§8.5). Đọc PDF thì **cố ý
  không** song song hoá — nó là việc thuần CPU bị GIL chặn, và sau khi có cache thì lần build
  thứ hai gần như không còn đọc lại tài liệu nào (§5.66).
- **Dò phần cứng thay vì giả định**: embedding và reranker tự chạy trên GPU nếu máy có, tự
  lùi về CPU nếu không; batch size và số worker suy từ VRAM còn trống chứ không phải hằng số
  hợp với một loại card. Điều này quan trọng vì `torch` bản CPU-only trên máy CÓ GPU là một
  cấu hình sai **không gây lỗi** — chỉ chậm hơn 13× ở embedding và 5× ở rerank, nên phải
  được hệ thống tự nói ra. Đo được: truy xuất đầu-cuối nhanh **12×** mà **6/6 câu cho kết
  quả giống hệt** (§5.68).
- **Chia VRAM theo giai đoạn**: riêng ba model của giai đoạn truy vấn đã là 8,07 GB, không
  vừa card 7,96 GB, mà tràn VRAM thì *không báo lỗi* — chỉ chậm hơn cả chạy CPU. Hết
  ingestion là nhả model vision và đẩy embedding về CPU (đo được: 0,79 GB trống → 6,79 GB).
  Máy không GPU thì bước này tự bỏ qua (§5.68).
- **Ngữ cảnh quá lớn thì cắt ĐOẠN, không hạ `num_ctx`**: hạ `num_ctx` không làm prompt ngắn
  lại, nó chỉ chuyển quyền quyết định cắt chỗ nào sang Ollama — mà Ollama luôn cắt từ **đầu**,
  tức xoá đúng đoạn trích liên quan nhất (§5.60). Nay hệ thống tự bỏ các đoạn xếp hạng **thấp
  nhất** và ghi rõ vào log. Câu hỏi đơn giản cũng được cấp ngân sách nhỏ hơn (12 thay vì 30
  ứng viên rerank); `TOP_K` thì **cố ý giữ nguyên** vì mọi ngưỡng của hệ thống đã hiệu chỉnh
  trên nó (§5.67).
- **LLM-as-judge dùng JSON Schema** để ép Ollama trả đúng field, nhưng schema **không** ràng
  buộc được khoảng giá trị nên việc kiểm thang điểm phải nằm ở code (§5.6, §5.48).
- **Hiểu câu hỏi nối tiếp bằng cách TẤT ĐỊNH, không phải bằng LLM**: đã thử query rewriting
  bằng LLM trước và nó thất bại đo được (qwen3:4b trả về rỗng 0/7 ca vì tiêu hết ngân sách
  token cho chuỗi suy luận; đủ chỗ thì tốn thêm ~30 giây mỗi câu). Cách dùng — ghép các câu
  hỏi trước vào truy vấn rồi để RRF hợp nhất — không tốn lượt gọi model nào, tất định nên đo
  được bằng chính các metric tất định, và câu gốc vẫn là một nhánh riêng nên ghép sai chỉ làm
  nhiễu thứ hạng chứ không xoá được kết quả đúng (§5.58).
- **Đối chiếu chéo các nguồn theo hai tầng**, nghiêng hẳn về phía im lặng: cặp phải khác nguồn
  + cùng chủ đề + có dấu hiệu bất đồng (tất định) rồi mới được LLM chấm, và phải được chấm "có
  mâu thuẫn" ở MỌI lần chấm. Báo động giả làm người dùng mất niềm tin vào chính tài liệu của
  họ — tệ hơn hẳn bỏ sót, vốn chỉ đưa hệ thống về hành vi cũ (§5.59).
- **`think=False` an toàn ĐÚNG MỘT CHỖ: khi đi kèm `format=<JSON Schema>`.** §5.23 cấm dùng nó
  với đầu ra tự do (suy luận đổ thẳng vào `content`), nhưng khi có grammar JSON thì chính
  grammar chặn suy luận. Đo được: `format=schema` không truyền `think` cho `content` RỖNG sau
  5.8s; thêm `think=False` cho JSON đúng trong 1.1s (§5.59).

## Giới hạn đã biết

- **Đổi model embedding thì phải bấm "Đọc tài liệu"** — quên build lại không gây lỗi (2 model có
  thể cùng số chiều) mà chỉ khiến kết quả sai âm thầm; hệ thống tự đối chiếu và cảnh báo trên
  UI, `run_evaluation.py` thì dừng hẳn.
- Câu hỏi thường mất ~25–33 giây với `qwen3:4b` trên GPU (truy xuất chỉ ~2,6s trong số đó — xem KET_QUA_DO_DAC.md §8; bản chạy CPU-only trước đây mất ~35–70 giây), do model luôn
  sinh phần suy luận nội bộ dài trước khi trả lời — đặc tính của model, `/no_think` lẫn
  `think=False` đều không rút ngắn được (§5.23). Câu bị từ chối chỉ mất ~6 giây vì không gọi
  LLM. Tổng thời gian không rút ngắn được, nhưng thời gian **nhìn thấy màn hình đứng yên** thì
  có: streaming đưa dấu hiệu đầu tiên về ~2 giây.
- **Chú thích ảnh bằng model vision mặc định BẬT** — cần `ollama pull qwen2.5vl:3b` trước. Chi
  phí thực đo 1,9 giây/hình khi model đã nạp (291 ảnh ~9 phút). Tắt bằng `BAT_CHU_THICH_ANH=0`;
  khi đó ảnh vẫn tìm được qua chú thích lân cận.
- **PDF nặng công thức toán**: khi PDF nhúng font không kèm bảng ToUnicode, công thức bị đọc ra
  thành mã `(cid:NN)` vô nghĩa. Hệ thống lọc bỏ rác này (1761 → 1 chunk trên giáo trình Bishop)
  nhưng **không khôi phục được nội dung công thức** trừ khi bật OCR. Câu hỏi về khái niệm vẫn
  trả lời tốt; câu hỏi về công thức cụ thể thì không.
- **DOCX không có khái niệm "trang" cố định** — hệ thống tách theo dấu ngắt trang cứng nếu có,
  hoặc coi cả file là 1 "trang". Bảng trong DOCX đã đọc được, nhưng `python-docx` chỉ thấy bảng
  ở tầng ngoài cùng — bảng lồng trong ô của bảng khác vẫn không thấy.
- **Nhận diện tiêu đề trong PDF là suy đoán theo cỡ chữ** (PDF không lưu cấu trúc logic), nên có
  thể bỏ sót hoặc nhận nhầm. Tắt bằng `BAT_NHAN_DIEN_TIEU_DE=0`.
- **Trang trong trích dẫn là vị trí vật lý trong file PDF**, không phải số trang in trên tài
  liệu — file có bìa/lời nói đầu chưa đánh số thì hai con số lệch nhau một khoảng cố định.
- **Nhận diện câu nối tiếp dựa trên danh sách dấu hiệu hồi chỉ** ("thế còn", "cái đó",
  "what about"...), không phân tích cú pháp. Câu nối tiếp diễn đạt theo cách không có trong
  danh sách sẽ bị bỏ sót và quay về hành vi single-turn. Hướng ưu tiên cố ý nghiêng về độ phủ
  (thà nhận nhầm) vì nhận nhầm gần như miễn phí, còn bỏ sót thì truy xuất trượt hẳn (§5.58).
- **Ngữ cảnh hội thoại chỉ gồm CÁC CÂU HỎI trước, không gồm câu trả lời trước.** Nên câu nối
  tiếp trỏ vào một chi tiết chỉ xuất hiện trong câu trả lời (chứ không trong câu hỏi) vẫn có
  thể trượt. Đây là đánh đổi có chủ đích: đưa câu trả lời cũ vào sẽ lấn át vector truy vấn, và
  cho model trích lại lời của chính nó như thể là tài liệu thì trích dẫn mất ý nghĩa (§5.58).
- **Phát hiện mâu thuẫn chỉ soi các đoạn ĐÃ ĐƯỢC TRUY XUẤT cho câu hỏi hiện tại**, không quét
  toàn bộ corpus. Hai tài liệu mâu thuẫn nhau ở phần không liên quan tới câu đang hỏi thì
  không được phát hiện — đó là bài toán khác (kiểm định nhất quán toàn corpus), tốn kém hơn
  nhiều bậc và nằm ngoài phạm vi đồ án.
- Không có backend API riêng — đúng phạm vi đồ án đã chốt.
- Nếu đường dẫn project chứa ký tự Unicode đặc biệt, một số thao tác ghi file cấp thấp của FAISS
  có thể lỗi — giữ project trong đường dẫn thuần ASCII để an toàn.

## Hướng phát triển (kèm ngưỡng cụ thể)

**1. Đổi index FAISS khi corpus lớn hơn — ngưỡng đã đo, không ước.** `IndexFlatIP` là tìm kiếm
vét cạn nên độ trễ tăng tuyến tính theo số chunk. Chạy `python evaluation/do_quy_mo_index.py` để
đo lại trên máy của bạn; kết quả trên máy làm đồ án (768 chiều, k=60, p95 từng câu):

| Số chunk | FlatIP p95 | RAM | HNSW p95 / recall | IVFFlat p95 / recall |
|---|---|---|---|---|
| 10.000 | 1.4 ms | 29 MB | 0.8 ms / 0.87 | 0.2 ms / 0.90 |
| 50.000 | 7.2 ms | 146 MB | 1.2 ms / 0.74 | 1.1 ms / 0.96 |
| 100.000 | 12.9 ms | 293 MB | 1.5 ms / 0.61 | 1.3 ms / 0.97 |

- **Ngưỡng theo tốc độ: ~1,5 triệu chunk** (mức FlatIP chạm 200 ms). Corpus hiện tại đang ở
  **0,4%** ngưỡng này.
- **Ngưỡng thực tế là BỘ NHỚ: ~700.000 chunk (≈2 GB RAM cho index)**, tức khoảng **145.000
  trang** với mật độ 4,8 chunk/trang của corpus này.
- **Khi đổi thì chọn `IndexIVFFlat`, không phải HNSW**: ở 100.000 chunk cả hai đều nhanh hơn
  Flat ~10 lần, nhưng IVF giữ recall 0.97 còn HNSW chỉ 0.61 — và recall HNSW tụt dần khi corpus
  to lên nếu giữ nguyên `efSearch`.
- Đổi index thì **bắt buộc** chạy lại `run_evaluation.py`: đoạn bỏ sót hoàn toàn có thể là đoạn
  chứa câu trả lời.

**2. Giảm độ trễ thật (chứ không chỉ độ trễ cảm nhận).** Ba hướng theo chi phí tăng dần: dùng
model không sinh suy luận cho câu hỏi truy xuất thường (giữ qwen3 cho câu kiểm chứng), lượng tử
hoá thấp hơn (q4 → q3), hoặc chạy GPU. Cả ba đều phải đo lại chất lượng trước khi chốt.

*Phần chi phí INGESTION và phần GPU đã làm xong* (§5.66, §5.68; KET_QUA_DO_DAC.md §8).
Sau hai đợt đó, nút thắt đã chuyển hẳn: **~98% thời gian mỗi câu hỏi nay nằm ở việc LLM sinh
chữ**, mà phần lớn là chuỗi suy luận nội bộ của qwen3. Đường khả dĩ còn lại là dùng model
không sinh suy luận cho câu hỏi thường — nhưng đã kiểm chứng lại trên Ollama 0.33.2 rằng
`think=False` **không** làm được việc đó (nó chỉ đổ chuỗi lập luận thẳng vào câu trả lời,
§8), nên phải đổi hẳn model và đo lại Faithfulness trước khi chốt.

Việc còn lại ở phía truy xuất: **adaptive TOP-K**. Nó bị hoãn có chủ đích chứ không phải bỏ quên — `TOP_K=4` là giá trị
mà toàn bộ Recall@K, MRR và các ngưỡng lọc đã hiệu chỉnh trên đó, nên hạ nó cho "câu hỏi đơn
giản" mà không đo lại chính là đổi độ chính xác lấy tốc độ. Cần một phép đo Recall@K tách theo
nhóm độ phức tạp câu hỏi trước đã.

**3. Giám khảo mạnh hơn cho evaluation.** Với `qwen3:4b`, 1/8 lần chấm cho kết quả ngược hẳn
(§5.43) — trung vị 3 lần là vá chứ không phải sửa gốc. Đặt `JUDGE_MODEL` sang model lớn hơn rồi
chạy `evaluation/kiem_dinh_judge.py` để xem có đáng đổi không.

**4. Nhận diện ô KHOÁ–GIÁ TRỊ trong biểu mẫu.** §5.41 đã cứu được phần lớn vấn đề bảng lớn,
nhưng biểu mẫu hành chính vẫn là ca khó: "LĨNH VỰC" và giá trị của nó nằm chung một hàng với cả
tiêu đề đề tài dài 100 chữ, nên vector của chunk đó bị tiêu đề lấn át.

## Ngoài phạm vi (cố tình không triển khai)

Conversation memory **qua nhiều phiên** (session ID, lưu lịch sử xuống đĩa), backend API riêng
(FastAPI/Flask), so sánh lại FAISS với vector DB khác — xem chi tiết lý do trong yêu cầu gốc
của đồ án.

> **Phân biệt hai chuyện hay bị gộp làm một.** "Không lưu lịch sử qua nhiều phiên" là quyết
> định phạm vi ở trên — đóng tab là mất lịch sử, đúng như đã chốt. Còn việc **truy xuất không
> nhìn thấy lịch sử ngay trong cùng một phiên** là một khiếm khuyết thật, và nó đã được sửa
> (§5.58): câu hỏi nối tiếp nay được ghép ngữ cảnh trước khi đi vào truy xuất.

*(Reranking trước đây nằm trong mục này. Quyết định đã đảo lại có chủ đích — xem §5.24.)*
