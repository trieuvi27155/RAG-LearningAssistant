# Chẩn đoán: truy xuất kém trên tài liệu mới & câu trả lời quá ngắn

**Dự án:** `rag-do-an`
**Ngày phân tích:** 2026-09-03
**Phạm vi đã đọc:** `config.py`, `rag/chunking.py`, `rag/embedding.py`, `rag/vector_store.py`, `rag/lexical_search.py`, `rag/reranker.py`, `rag/rag_pipeline.py`, `app.py`, `requirements.txt`, `data/faiss_index/index_info.json`

---

Code của bạn viết rất tốt, phần lý giải trong comment thuộc loại hiếm thấy ở đồ án. Nhưng đúng vì thế mà lỗi ở đây không phải lỗi "code sai" — nó là **lỗi giả định**.

Có 1 bug thật và 4 vấn đề kiến trúc. Xếp theo mức độ chắc chắn.

---

## 1. Bug chắc chắn: chưa bao giờ set `num_ctx` — cửa sổ ngữ cảnh thật chỉ có 4096 token

`rag_pipeline.py:806`:

```python
options={
    "temperature": config.OLLAMA_TEMPERATURE,
    "num_predict": config.OLLAMA_NUM_PREDICT,   # 12000
},
```

Không có `num_ctx`. **Ollama mặc định cấp cho model một cửa sổ 4096 token**, bất kể model hỗ trợ bao nhiêu.

Comment ở `config.py:394-395` viết:

> *"context window của model rất lớn (262144 token) nên không có lý do phải cắt sớm hơn để tiết kiệm"*

Đây chính là giả định ngầm sai. 262144 là năng lực **kiến trúc** của qwen3:4b. 4096 là ngân sách **runtime** mà Ollama thực sự cấp phát. Hai con số khác nhau, và Ollama không báo lỗi khi vượt — nó cắt im lặng.

### Ước lượng prompt thật

(tiếng Việt ≈ 2.5 ký tự/token với tokenizer Qwen)

| Thành phần | Ký tự | ≈ Token |
|---|---|---|
| System prompt VI | 1.894 | ~760 |
| 6 đoạn × `NGAN_SACH_KY_TU_MOI_DOAN` 1600 + nhãn nguồn | ~9.960 | ~4.000 |
| Câu hỏi + hướng dẫn | ~350 | ~140 |
| **Tổng** | | **~4.900** |

Prompt **một mình đã vượt 4096**, chưa tính một token nào cho thinking và câu trả lời.

### Vì sao điều này gây ra *đúng cả hai* triệu chứng

**(a) Câu trả lời ngắn.**

qwen3:4b luôn sinh thinking dài (chính comment `config.py:632-635` của bạn đo được 5891 ký tự thinking cho một tác vụ nhỏ). Khi prompt đã ăn hết 4096, phần còn lại cho thinking + answer gần bằng 0. Model viết được vài câu rồi chạm trần và dừng.

Bằng chứng bạn đã tự ghi lại mà chẩn đoán nhầm nguyên nhân — `rag_pipeline.py:879-889`:

```python
# Model nhét TẤT CẢ vào phần suy luận và không viết câu trả lời nào (hiếm, nhưng
# đã gặp khi model bị cắt ngang vì chạm num_predict)
```

Không phải `num_predict=12000` (con số đó chưa bao giờ với tới được). Là `num_ctx=4096`.

**(b) Truy xuất "kém" trên tài liệu mới.**

Đây là phần quan trọng nhất. Khi prompt vượt `num_ctx`, Ollama giữ system message và cắt từ **đầu** phần user content. Mà `_ghep_prompt()` xếp đoạn trích **tốt nhất trước** (`enumerate(cac_chunk, start=1)`, `cac_chunk` đã sắp theo rerank).

→ **Phần bị xoá chính xác là `[1]`, `[2]` — đoạn trích liên quan nhất.**

Retrieval của bạn có thể đã tìm đúng. LLM chỉ không bao giờ được nhìn thấy nó.

### Vì sao chỉ lộ ra trên tài liệu mới

Corpus cũ nhiều slide và trang thưa chữ → mỗi đoạn trích thực tế ngắn hơn nhiều so với trần 1600. Prompt tổng có thể chỉ ~2500 token, vừa lọt 4096.

Tài liệu mới dày chữ → mỗi đoạn trích **chạm trần 1600** → prompt phình lên ~4900 → bắt đầu bị cắt.

Nghĩa là: hệ thống không "kém hơn trên tài liệu mới". Nó luôn có bug này; tài liệu cũ chỉ tình cờ nằm dưới ngưỡng gây lỗi.

### Cách tự kiểm chứng trong 5 phút

Ollama trả về `prompt_eval_count` ở chunk cuối của stream. Thêm vào cuối vòng lặp trong `_goi_llm_theo_luong`:

```python
if manh.get("done"):
    logger.warning(
        "prompt_eval_count=%s  eval_count=%s  done_reason=%s",
        manh.get("prompt_eval_count"),
        manh.get("eval_count"),
        manh.get("done_reason"),
    )
```

Hỏi một câu trên tài liệu mới:

- Nếu `prompt_eval_count` dừng quanh ~4096 trong khi prompt ghép ra dài hơn → prompt đã bị cắt, giả thuyết được xác nhận.
- `done_reason` là `"length"` thay vì `"stop"` → câu trả lời bị cắt cụt.

### Sửa

```python
# config.py
# num_ctx PHẢI khai báo tường minh: Ollama mặc định 4096 bất kể model hỗ trợ bao nhiêu.
# Ngân sách = system prompt + TOP_K đoạn trích + thinking + câu trả lời.
OLLAMA_NUM_CTX = _lay_int("OLLAMA_NUM_CTX", 16384)
```

```python
# rag/rag_pipeline.py
options={
    "temperature": config.OLLAMA_TEMPERATURE,
    "num_predict": config.OLLAMA_NUM_PREDICT,
    "num_ctx": config.OLLAMA_NUM_CTX,
},
```

**Cái giá phải trả** — nên biết trước, đừng để nó là bất ngờ: KV-cache tỉ lệ tuyến tính với `num_ctx`. 16384 với qwen3:4b tốn thêm khoảng vài trăm MB RAM và làm prefill chậm hơn. Trên CPU, prefill ~5000 token có thể mất 10-20 giây. Đo lại thời gian phản hồi sau khi sửa; nếu quá chậm, hạ `TOP_K` hoặc `NGAN_SACH_KY_TU_MOI_DOAN` chứ **đừng** hạ `num_ctx` xuống dưới độ dài prompt.

Tốt hơn nữa là **tính động và cảnh báo**: đếm token prompt bằng tokenizer, đặt `num_ctx = prompt_tokens + 4000`, và log warning nếu prompt vượt một ngưỡng an toàn. Như vậy lỗi này không bao giờ tái diễn im lặng khi đổi `TOP_K` hay đổi model.

---

## 2. Ngưỡng đang là ngưỡng tuyệt đối, hiệu chỉnh in-sample

`NGUONG_DIEM_TOI_THIEU = 0.70`, `NGUONG_DIEM_RERANK_TOI_THIEU = 0.001`, `NGUONG_COSINE_DOI_CHIEU = 0.88` — cả ba đều là hằng số tuyệt đối, đo trên đúng corpus của bạn.

Cosine của E5 **không phải thang đo tuyệt đối**. Giá trị tuyệt đối phụ thuộc domain, ngôn ngữ, độ dài chunk, phong cách văn bản. Một corpus mới (nhiều bảng, nhiều công thức, nhiều số) dịch cả phân bố xuống — và ngưỡng cố định bắt đầu cắt oan. Chunk là mảnh bảng hay chú thích ảnh vốn đã cho cosine thấp hơn văn xuôi.

Có một vấn đề kiến trúc sâu hơn ở `truy_xuat()`: **rerank quyết định thứ tự, cosine quyết định sống chết**.

```python
cac_doan = [d for d in cac_doan if d["diem_similarity"] >= config.NGUONG_DIEM_TOI_THIEU]
```

`diem_similarity` là cosine (bạn cố ý giữ nguyên, `_xep_hang_lai` docstring giải thích rõ). Nên một đoạn được cross-encoder xếp hạng 1 nhưng cosine 0.68 vẫn bị vứt. Hai thang đo khác nhau cùng ra quyết định trong một pipeline — bạn đã tránh trộn chúng vào cùng một *trường*, nhưng chưa tránh việc chúng ra quyết định *mâu thuẫn nhau*.

Và đây là nguyên nhân **thứ hai** của "câu trả lời ngắn": nếu 4/6 đoạn rớt ngưỡng, context chỉ còn 2 đoạn → không có gì để tổng hợp.

### Hướng sửa — ngưỡng tương đối

```python
# Giữ mọi đoạn không quá kém so với đoạn tốt nhất, thay vì so với hằng số tuyệt đối.
# Phân bố cosine dịch theo domain, tỷ lệ giữa các đoạn TRONG CÙNG một lượt thì không.
if cac_doan:
    diem_cao_nhat = max(d["diem_similarity"] for d in cac_doan)
    cac_doan = [d for d in cac_doan
                if d["diem_similarity"] >= diem_cao_nhat * config.TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT]
```

Với `TY_LE_... = 0.92` chẳng hạn. Sàn tuyệt đối vẫn giữ, nhưng hạ xuống rất thấp (0.5) để chỉ còn đúng vai trò chặn rác như comment của bạn nói.

### Trước khi sửa, hãy đo phân bố trước đã

Đừng đổi ngưỡng theo cảm tính — đúng nguyên tắc bạn đã áp dụng với BM25.

```python
logger.info(
    "Điểm 6 đoạn đầu: %s | rerank cao nhất: %s | còn lại sau ngưỡng: %d",
    [round(d["diem_similarity"], 3) for d in cac_doan],
    self.diem_rerank_cao_nhat,
    len([d for d in cac_doan if d["diem_similarity"] >= config.NGUONG_DIEM_TOI_THIEU]),
)
```

Chạy trên 10 câu hỏi ở tài liệu cũ và 10 câu ở tài liệu mới:

- Nếu số đoạn sống sót trên tài liệu mới thấp hơn hẳn → giả thuyết được xác nhận.
- Nếu vẫn 6/6 → vấn đề nằm hoàn toàn ở `num_ctx`, và bạn không cần đụng tới ngưỡng.

---

## 3. Mở rộng ngữ cảnh bị chặn cứng ở ranh giới trang

`_dung_doan_trich()` chỉ mở rộng trong cùng `(nguon, trang)`:

```python
vi_tri_trong_trang = self.vector_store.chi_muc_trang[(neo["nguon"], neo["trang"])]
```

Với slide, điều này đúng: mỗi slide là một đơn vị nội dung tự đóng.

Với **PDF văn bản chảy liên tục**, một định nghĩa bắt đầu cuối trang 12 và kết thúc đầu trang 13 sẽ **không bao giờ** được nối lại — chunk neo nằm ở cuối trang 12, mở rộng sang phải chạm hết mảng rồi dừng.

Lại đúng cái pattern "chỉ lộ trên tài liệu mới": corpus cũ nhiều slide, corpus mới nhiều PDF văn xuôi.

**Sửa:** thêm một chỉ số thứ tự toàn cục cho chunk (`thu_tu_tai_lieu`) lúc chunking, và cho phép `_dung_doan_trich` mở rộng qua ranh giới trang **trong cùng một nguồn**, có đánh dấu để citation vẫn ghi đúng trang của chunk neo.

---

## 4. `SO_DOAN_TOI_DA_MOI_TRANG = 2` phản tác dụng trên tài liệu ngắn

Trần này sinh ra để chống việc các chunk liền kề trong giáo trình 230 trang chiếm hết `TOP_K`. Lý do đúng. Nhưng nó là hằng số, và với tài liệu mới ngắn hơn nó gây hại:

Nếu toàn bộ câu trả lời nằm trên **một trang** (rất phổ biến: một mục định nghĩa, một bảng tiêu chí, một quy trình), bạn chỉ được lấy 2 đoạn từ trang đó. Bốn suất còn lại đi cho những trang kém liên quan.

Kết quả: context vừa **thiếu** phần đúng, vừa **loãng** vì phần sai — và câu trả lời ngắn đi vì model không có đủ nguyên liệu.

**Sửa:** cho trần thích ứng — chỉ áp dụng khi số trang ứng viên đủ đa dạng.

```python
# Trần chỉ có ý nghĩa khi CÓ nhiều trang để phân bổ. Với câu hỏi mà toàn bộ câu trả lời
# nằm gọn trong một trang, ép lấy đoạn từ trang khác chỉ làm loãng ngữ cảnh.
so_trang_ung_vien = len({(md["nguon"], md["trang"]) for md, _ in ung_vien[:20]})
tran_moi_trang = (config.SO_DOAN_TOI_DA_MOI_TRANG
                  if so_trang_ung_vien >= top_k else top_k)
```

---

## 5. BM25 tắt — nhưng cách sửa không phải là bật lại

Kết quả âm tính của bạn (`config.py:216-231`) là đo đạc tốt và tôi không nghĩ bạn nên lật ngược nó. Nhưng hãy đọc lại chính comment bạn viết:

> *"corpus khác có thể cho kết quả khác (vd corpus THUẦN một ngôn ngữ, hoặc dày mã định danh mà model embedding chưa từng thấy)"*

"Tài liệu mới mà hệ thống chưa từng nhìn thấy" là **chính xác** trường hợp đó: tên riêng, mã hiệu, thuật ngữ OOV mà `multilingual-e5-base` chưa từng gặp.

**Điểm mấu chốt:** cái hại bạn đo được **không phải do BM25 tìm sai**, mà do **RRF cho BM25 quyền xếp hạng ngang dense**. Hạng 1 của BM25 (một tài liệu tiếng Việt sai) được coi ngang hạng 1 của dense.

Nên hướng đúng không phải "bật lại với trọng số 0.2", mà là **tách vai trò**:

- BM25 chỉ **bơm ứng viên** vào tập rerank (recall), **không** đóng góp điểm RRF (precision).
- Cross-encoder — vốn đọc cả cặp và bạn đã đo là phân biệt tốt gấp 60.000 lần — quyết định thứ hạng cuối.

```python
# BM25 làm nhiệm vụ CỨU HỘ: chỉ đảm bảo đoạn chứa từ khoá hiếm có mặt trong tập
# ứng viên đưa vào rerank, KHÔNG được quyền đẩy thứ hạng. Cách này giữ lại lợi ích
# của BM25 (recall trên từ OOV) mà tránh đúng cơ chế gây hại đã đo ở §TRONG_SO_BM25
# (RRF coi hạng 1 của BM25 ngang hạng 1 của dense, phá truy xuất chéo ngôn ngữ).
```

Cụ thể: lấy top-10 của BM25, thêm vào cuối danh sách ứng viên nếu chưa có, gán điểm RRF bằng 0 (xếp cuối), rồi để `_xep_hang_lai` chấm chúng cùng 30 ứng viên đầu. Nếu chúng thực sự liên quan, cross-encoder sẽ đẩy lên. Nếu không, chúng nằm yên ở cuối.

Trường hợp xấu nhất theo thiết kế là "không cải thiện", không phải "làm hỏng" — đúng nguyên tắc §5.45(a) bạn đã dùng ở nhánh câu hỏi gốc.

Và phải đo lại bằng đúng thí nghiệm cũ trước khi giữ.

---

## 6. Prompt: quy tắc 5 và 6 xung đột nhau

```
5. Trả lời ĐẦY ĐỦ ... tổng hợp mọi thông tin liên quan
6. Trả lời ĐI THẲNG VÀO TRỌNG TÂM, không lặp lại ... không viết phần mở đầu/kết luận thừa
```

Với model 4B, hai chỉ thị ngược chiều đặt cạnh nhau thường khiến model chọn cái **dễ tuân thủ hơn** — và "ngắn gọn" luôn dễ hơn "đầy đủ".

Đây là nguyên nhân **thứ ba** (yếu nhất) của câu trả lời ngắn. Tôi để nó cuối cùng có chủ ý: **đừng sửa prompt trước khi sửa `num_ctx`**. Nếu sửa prompt trước, bạn sẽ thấy cải thiện nhẹ, tưởng đã tìm đúng nguyên nhân, và bug thật vẫn nằm nguyên đó.

---

## 7. Vấn đề nằm dưới tất cả: toàn bộ tham số đang được hiệu chỉnh in-sample

Đây mới là bài học kiến trúc thật của đồ án này, và nó xứng đáng vào báo cáo.

Bạn có `evaluation/test_questions.json` sinh từ chính corpus, và mọi hằng số — 0.70, 0.001, 0.88, 2, 1, 30, 160 — đều được chọn bằng cách tối ưu trên đúng bộ đó. Đó là **tuning trên tập test**.

Kết quả tất yếu: hệ thống rất tốt trên corpus đã dùng để chỉnh, và tụt trên corpus mới. Đúng hiện tượng bạn đang gặp.

Việc cần làm, theo thứ tự:

1. Tách một bộ **held-out**: 3-5 tài liệu chưa từng dùng để chỉnh bất kỳ tham số nào, kèm 15-20 câu hỏi có nhãn. Không bao giờ tune trên bộ này.
2. Chạy `run_evaluation.py` trên cả hai bộ. Khoảng cách Recall@K giữa in-sample và held-out **chính là con số đo mức overfit** của hệ thống.
3. Mỗi lần sửa, báo cáo cả hai con số.

Khoảng cách đó, kèm giải thích vì sao nó tồn tại, là thứ phân biệt một đồ án RAG với một tutorial "chat with PDF". Hầu như không ai đo nó.

---

## Thứ tự thực hiện

| # | Việc | Công sức | Kỳ vọng |
|---|---|---|---|
| 1 | Log `prompt_eval_count` / `done_reason` để **xác nhận** giả thuyết trước khi sửa | 5 phút | — |
| 2 | Set `num_ctx` | 10 phút | Sửa phần lớn cả hai triệu chứng |
| 3 | Log phân bố điểm, đo số đoạn sống sót sau ngưỡng | 15 phút | Xác nhận hoặc loại bỏ nguyên nhân #2 |
| 4 | Dựng bộ held-out + đo baseline | 1-2 giờ | Không có nó, mọi thay đổi sau đều là đoán |
| 5 | Ngưỡng tương đối; trần đoạn/trang thích ứng | 1 giờ | Vừa |
| 6 | Mở rộng ngữ cảnh qua ranh giới trang | 2-3 giờ | Vừa, quan trọng với PDF văn xuôi |
| 7 | BM25 recall-only | 2 giờ | Nhỏ, chỉ đáng làm sau khi có held-out để đo |

Bước 1 và 2 gần như chắc chắn là nguyên nhân chính.

Bước 3-7 là giả thuyết có căn cứ nhưng **cần đo mới biết** — và tôi cố tình không đưa con số cụ thể cho ngưỡng mới, vì chọn chúng mà không đo lại đúng là sai lầm đã tạo ra tình trạng hiện tại.

---

## Nguồn tham khảo

- [Ollama FAQ — default context window](https://docs.ollama.com/faq) (xác nhận mặc định 4096 token, cách đổi qua `num_ctx` / `OLLAMA_CONTEXT_LENGTH`)
