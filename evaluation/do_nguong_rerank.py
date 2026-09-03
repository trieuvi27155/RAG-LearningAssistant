"""Đo xem điểm reranker có tách được câu hỏi TRONG phạm vi tài liệu khỏi câu LẠC ĐỀ hay không.

VÌ SAO CẦN ĐO RIÊNG:
config.NGUONG_DIEM_TOI_THIEU (điểm cosine) đã được đo và kết luận là KHÔNG dùng để phát
hiện câu lạc đề được - vì câu hỏi đúng chủ đề hỏi bằng tiếng Anh cho điểm cosine thấp hơn
cả câu tiếng Việt lạc đề (tài liệu viết bằng tiếng Việt nên tương đồng xuyên ngôn ngữ luôn
bị thiệt). Việc từ chối vì thế đang giao hoàn toàn cho LLM qua quy tắc trong system prompt.

Cross-encoder (rerank) đọc CẢ CẶP (câu hỏi, đoạn) cùng lúc thay vì mã hoá 2 phía độc lập,
nên về lý thuyết nó đánh giá được "đoạn này có trả lời được câu hỏi này không" thay vì chỉ
"hai đoạn text này có giống nhau không". Câu hỏi: điều đó có đúng trên thực tế không, và
có đủ tách bạch để dùng làm ngưỡng chặn không?

KHÔNG ĐƯỢC GIẢ ĐỊNH LÀ CÓ. Script này tồn tại để đo, và kết quả âm tính (không tách được)
cũng là một kết quả hợp lệ - khi đó giữ nguyên cách làm hiện tại và ghi nhận vào tài liệu,
đúng như đã làm với ngưỡng cosine.

Cách chạy:  python evaluation/do_nguong_rerank.py
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.embedding import EmbeddingService
from rag.rag_pipeline import RagPipeline
from rag.reranker import RerankerService
from rag.vector_store import VectorStore

# Câu hỏi ĐÚNG chủ đề tài liệu mẫu (thư viện số / quản lý thư viện), hỏi bằng tiếng Việt.
TRONG_PHAM_VI_VI = [
    "Thư viện số đầu tiên được xây dựng vào năm nào?",
    "Quy trình mượn tài liệu gồm những bước nào?",
    "Phí phạt trả sách trễ được tính như thế nào?",
    "Thẻ thư viện có thời hạn bao lâu?",
    "Bạn đọc được mượn tối đa bao nhiêu cuốn?",
    "Kho tài liệu được sắp xếp theo nguyên tắc nào?",
]

# Cũng đúng chủ đề, nhưng hỏi bằng TIẾNG ANH - đây chính là nhóm mà ngưỡng cosine đánh
# trượt. Nếu reranker cũng đánh trượt nhóm này thì nó không dùng làm ngưỡng được.
TRONG_PHAM_VI_EN = [
    "When was the first digital library built?",
    "What are the steps to borrow a document?",
    "How is the late return fee calculated?",
    "How long is a library card valid?",
]

# Hoàn toàn ngoài phạm vi tài liệu, hỏi bằng tiếng Việt.
LAC_DE_VI = [
    "Công thức tính diện tích hình tròn là gì?",
    "Cách nấu phở bò ngon nhất?",
    "Messi ghi bao nhiêu bàn ở World Cup 2022?",
    "Giá vàng hôm nay bao nhiêu?",
    "Thuật toán sắp xếp nhanh hoạt động thế nào?",
    "Triệu chứng của bệnh cúm mùa là gì?",
]


def _diem_cao_nhat(pipeline: RagPipeline, reranker: RerankerService, cau_hoi: str):
    """Trả về (điểm cosine cao nhất, điểm rerank cao nhất) cho 1 câu hỏi."""
    cac_doan = pipeline.truy_xuat(cau_hoi)
    if not cac_doan:
        return None, None
    cosine = max(d["diem_similarity"] for d in cac_doan)
    diem_rerank = reranker.xep_hang(cau_hoi, [d["noidung"] for d in cac_doan])
    return cosine, float(max(diem_rerank))


def _in_nhom(nhan: str, cac_diem):
    hop_le = [(c, r) for c, r in cac_diem if c is not None]
    if not hop_le:
        print(f"{nhan:<28} (không có kết quả nào)")
        return
    cosine = [c for c, _ in hop_le]
    rerank = [r for _, r in hop_le]
    print(
        f"{nhan:<28} "
        f"{min(cosine):>7.3f} {statistics.mean(cosine):>7.3f} {max(cosine):>7.3f}   "
        f"{min(rerank):>9.3f} {statistics.mean(rerank):>9.3f} {max(rerank):>9.3f}"
    )


def chay_do():
    if not (config.FAISS_INDEX_FILE.exists() and config.METADATA_MAPPING_FILE.exists()):
        print("Chưa có FAISS index. Hãy build index (qua app.py) rồi chạy lại.")
        return

    print("Đang tải model embedding, reranker và FAISS index...")
    embedding_service = EmbeddingService()
    vector_store = VectorStore.tai()
    ly_do = vector_store.ly_do_khong_tuong_thich()
    if ly_do:
        print(f"DỪNG: {ly_do}\nHãy build lại index rồi chạy lại.")
        return
    reranker = RerankerService()
    pipeline = RagPipeline(embedding_service, vector_store, reranker_service=reranker)

    cac_nhom = [
        ("Đúng chủ đề (tiếng Việt)", TRONG_PHAM_VI_VI),
        ("Đúng chủ đề (tiếng Anh)", TRONG_PHAM_VI_EN),
        ("Lạc đề (tiếng Việt)", LAC_DE_VI),
    ]
    ket_qua = {}
    for nhan, cac_cau in cac_nhom:
        ket_qua[nhan] = [_diem_cao_nhat(pipeline, reranker, c) for c in cac_cau]

    print()
    print(f"{'':<28} {'--------- COSINE ---------':^23}   {'-------- RERANK ---------':^29}")
    print(f"{'Nhóm câu hỏi':<28} {'min':>7} {'tb':>7} {'max':>7}   {'min':>9} {'tb':>9} {'max':>9}")
    print("-" * 88)
    for nhan, _ in cac_nhom:
        _in_nhom(nhan, ket_qua[nhan])

    # Kết luận tự động: có tồn tại ngưỡng nào vừa nhận HẾT câu đúng chủ đề (cả 2 ngôn ngữ)
    # vừa loại HẾT câu lạc đề không? Đây đúng là điều kiện để dùng làm ngưỡng chặn.
    dung_chu_de = [
        r for nhan in ("Đúng chủ đề (tiếng Việt)", "Đúng chủ đề (tiếng Anh)")
        for _, r in ket_qua[nhan] if r is not None
    ]
    lac_de = [r for _, r in ket_qua["Lạc đề (tiếng Việt)"] if r is not None]
    print()
    if dung_chu_de and lac_de and min(dung_chu_de) > max(lac_de):
        print(
            f"=> TÁCH ĐƯỢC: câu đúng chủ đề thấp nhất ({min(dung_chu_de):.3f}) vẫn cao hơn "
            f"câu lạc đề cao nhất ({max(lac_de):.3f}).\n"
            f"   Có thể đặt NGUONG_DIEM_RERANK_TOI_THIEU trong khoảng "
            f"({max(lac_de):.3f}, {min(dung_chu_de):.3f})."
        )
    else:
        print(
            f"=> KHÔNG TÁCH ĐƯỢC: câu đúng chủ đề thấp nhất ({min(dung_chu_de):.3f}) KHÔNG "
            f"cao hơn câu lạc đề cao nhất ({max(lac_de):.3f}).\n"
            "   Mọi ngưỡng đều sẽ hoặc loại nhầm câu hợp lệ, hoặc bỏ lọt câu lạc đề.\n"
            "   Giữ nguyên cách hiện tại: giao việc từ chối cho LLM qua quy tắc system prompt."
        )


if __name__ == "__main__":
    chay_do()
