"""Test cho tầng xếp hạng lại bằng cross-encoder.

Chia 2 loại theo đúng quy ước sẵn có của bộ test:
  - Test LOGIC (RerankerGia, điểm dựng tay): kiểm việc xếp lại thứ tự có đúng không, chạy
    trong mili giây, không nạp model nào.
  - Test MODEL THẬT (1 ca duy nhất): kiểm chính bản thân model có phân biệt được liên quan
    và lạc đề không - thứ chỉ model thật mới trả lời được.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import config
from rag.rag_pipeline import RagPipeline
from rag.vector_store import VectorStore

SO_CHIEU = 4


class EmbeddingGia:
    def __init__(self, vector_cau_hoi):
        self.vector = np.array([vector_cau_hoi], dtype="float32")

    def encode_cau_hoi(self, texts):
        return self.vector


class RerankerGia:
    """Chấm điểm theo bảng tra dựng tay: đoạn nào chứa từ khoá thì điểm cao."""

    def __init__(self, tu_khoa_diem_cao: str):
        self.tu_khoa = tu_khoa_diem_cao
        self.so_lan_goi = 0

    def xep_hang(self, cau_hoi, cac_doan):
        self.so_lan_goi += 1
        return np.array([9.0 if self.tu_khoa in doan else 0.1 for doan in cac_doan])


def _chunk(nguon, trang, vi_tri, noidung):
    return {"chunk_id": f"{nguon}-{trang}-{vi_tri}", "nguon": nguon, "trang": trang,
            "vi_tri": vi_tri, "noidung": noidung}


def _vector_don_vi(chi_so):
    v = [0.0] * SO_CHIEU
    v[chi_so] = 1.0
    return v


def _tao_store(cac_chunk, cac_vector):
    store = VectorStore(dimension=SO_CHIEU)
    store.them(np.array(cac_vector, dtype="float32"), cac_chunk)
    return store


@pytest.fixture(autouse=True)
def tat_tim_kiem_tu_khoa(monkeypatch):
    """Tắt BM25 để thứ hạng đầu vào do đúng vector quyết định - test này kiểm phần RERANK,
    không kiểm phần hợp nhất RRF (đã có test riêng)."""
    monkeypatch.setattr(config, "TRONG_SO_BM25", 0.0)


def test_rerank_keo_dung_doan_len_dau():
    """Đây là lý do tồn tại của cả tầng rerank: đoạn khớp CHI TIẾT được hỏi phải thắng đoạn
    chỉ cùng chủ đề chung chung, kể cả khi vector xếp đoạn chung chung cao hơn."""
    cac_chunk = [
        _chunk("a.pdf", 1, 0, "Thư viện số là hệ thống lưu trữ tài liệu điện tử nói chung."),
        _chunk("a.pdf", 2, 0, "Phí phạt mỗi ngày với tạp chí đóng tập là 5.000 đồng."),
    ]
    # Vector cố tình xếp đoạn CHUNG CHUNG (chunk 0) lên trước đoạn đúng chi tiết (chunk 1).
    cac_vector = [_vector_don_vi(0), [0.9, 0.44, 0.0, 0.0]]
    store = _tao_store(cac_chunk, cac_vector)

    khong_rerank = RagPipeline(EmbeddingGia(_vector_don_vi(0)), store)
    assert khong_rerank.truy_xuat("phí phạt tạp chí", top_k=1)[0]["trang"] == 1

    co_rerank = RagPipeline(
        EmbeddingGia(_vector_don_vi(0)), store, reranker_service=RerankerGia("5.000 đồng")
    )
    assert co_rerank.truy_xuat("phí phạt tạp chí", top_k=1)[0]["trang"] == 2


def test_rerank_khong_doi_diem_similarity():
    """Rerank chỉ đổi THỨ TỰ, không đổi điểm công bố. Nếu điểm rerank lọt vào
    diem_similarity thì sàn NGUONG_DIEM_TOI_THIEU và điểm hiển thị trên UI sẽ nằm ở 2 thang
    đo khác nhau - kiểu lỗi rất khó lần ra vì không có gì báo lỗi."""
    cac_chunk = [_chunk("a.pdf", 1, 0, "Nội dung có từ khoá đặc biệt ở đây.")]
    store = _tao_store(cac_chunk, [_vector_don_vi(0)])

    ket_qua = RagPipeline(
        EmbeddingGia(_vector_don_vi(0)), store, reranker_service=RerankerGia("đặc biệt")
    ).truy_xuat("câu hỏi", top_k=1)

    # Cosine của 2 vector đơn vị trùng nhau = 1.0, không phải 9.0 của RerankerGia.
    assert ket_qua[0]["diem_similarity"] == pytest.approx(1.0, abs=1e-5)


def test_khong_co_reranker_thi_pipeline_van_chay():
    """reranker_service=None là đường chạy mặc định khi BAT_RERANK=0 - không được nổ."""
    cac_chunk = [_chunk("a.pdf", 1, 0, "Một đoạn nội dung bất kỳ đủ dài để giữ lại.")]
    store = _tao_store(cac_chunk, [_vector_don_vi(0)])
    ket_qua = RagPipeline(EmbeddingGia(_vector_don_vi(0)), store).truy_xuat("hỏi", top_k=1)
    assert len(ket_qua) == 1


def test_chi_rerank_toi_da_so_ung_vien_cau_hinh(monkeypatch):
    """Trần SO_UNG_VIEN_RERANK là thứ giữ chi phí ở mức chấp nhận được: mỗi ứng viên vượt
    trần đều là một lượt chạy model thật. Phần đuôi phải được GIỮ LẠI (không cắt bỏ) để các
    bước lọc sau vẫn đủ ứng viên lấp TOP_K."""
    monkeypatch.setattr(config, "SO_UNG_VIEN_RERANK", 2)
    # Tắt mở rộng xuyên trang: mỗi chunk ở đây nằm một trang riêng, để bật thì các đoạn
    # trích liền kề nhập vào nhau và số đoạn trả về giảm đi - đúng hành vi mong muốn của
    # tính năng đó, nhưng nó che mất thứ test này đang kiểm (phần đuôi ngoài trần rerank).
    monkeypatch.setattr(config, "MO_RONG_QUA_RANH_GIOI_TRANG", False)
    cac_chunk = [_chunk("a.pdf", i + 1, 0, f"Đoạn nội dung số {i} đủ dài để không bị lọc bỏ.")
                 for i in range(5)]
    store = _tao_store(cac_chunk, [_vector_don_vi(0)] * 5)

    reranker = RerankerGia("số 1")
    ket_qua = RagPipeline(
        EmbeddingGia(_vector_don_vi(0)), store, reranker_service=reranker
    ).truy_xuat("câu hỏi", top_k=5)

    assert reranker.so_lan_goi == 1
    assert len(ket_qua) == 5, "phần đuôi ngoài trần rerank vẫn phải được giữ lại"


def test_doan_rong_khong_gay_loi():
    from rag.reranker import RerankerService

    assert len(RerankerService.xep_hang(object.__new__(RerankerService), "hỏi", [])) == 0


@pytest.mark.slow
def test_model_that_phan_biet_duoc_lien_quan_va_lac_de():
    """Ca duy nhất nạp model thật - kiểm chính bản thân model, thứ mà điểm dựng tay không
    thể thay thế. Kiểm cả tiếng Việt lẫn tiếng Anh vì hệ thống yêu cầu song ngữ."""
    from rag.reranker import RerankerService

    dich_vu = RerankerService()
    diem_vi = dich_vu.xep_hang(
        "Phí phạt tạp chí đóng tập là bao nhiêu?",
        ["Tạp chí đóng tập bị phạt 5.000 đồng mỗi ngày.",
         "Công thức tính diện tích hình tròn là pi nhân bán kính bình phương."],
    )
    assert diem_vi[0] > diem_vi[1]

    diem_en = dich_vu.xep_hang(
        "How many items can a card borrow at once?",
        ["Each card may borrow up to five items at a time.",
         "The kitchen was renovated last summer with new tiles."],
    )
    assert diem_en[0] > diem_en[1]
