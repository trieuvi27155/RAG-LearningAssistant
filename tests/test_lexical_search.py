"""Test cho nhánh tìm kiếm theo từ khoá (BM25) - phần bù cho tìm kiếm vector.

Không cần model embedding nên chạy rất nhanh.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.lexical_search import BM25, _tach_tu


def test_tach_tu_sinh_ca_am_tiet_va_cap_am_tiet():
    """Tiếng Việt viết rời từng âm tiết nên phải lập chỉ mục cả cặp âm tiết liền nhau,
    nếu không "pháp"/"luật" riêng lẻ sẽ khớp tràn lan khắp tài liệu."""
    cac_tu = _tach_tu("Pháp luật Việt Nam")
    assert "pháp" in cac_tu and "luật" in cac_tu
    assert "pháp_luật" in cac_tu
    assert "luật_việt" in cac_tu


def test_tach_tu_khong_phan_biet_dang_unicode():
    """document_loader chuẩn hoá NFC; nếu BM25 tách từ trên chuỗi NFD thì cùng một từ sẽ
    bị coi là 2 từ khác nhau và không bao giờ khớp."""
    import unicodedata

    nfc = "quy phạm"
    nfd = unicodedata.normalize("NFD", nfc)
    assert _tach_tu(nfc) == _tach_tu(nfd)


def test_tim_kiem_uu_tien_doan_chua_tu_khoa_hiem():
    """Đây chính là điểm yếu của tìm kiếm thuần vector mà BM25 bù lại: từ khoá cụ thể và
    hiếm (số hiệu điều luật) phải kéo đúng đoạn chứa nó lên đầu, thay vì đoạn cùng chủ đề."""
    van_ban = [
        "Nhà nước là tổ chức quyền lực chính trị đặc biệt của xã hội.",
        "Điều 15 quy định về quyền và nghĩa vụ cơ bản của công dân.",
        "Pháp luật là hệ thống quy tắc xử sự chung do nhà nước ban hành.",
    ]
    bm25 = BM25(van_ban)
    ket_qua = bm25.tim_kiem("Điều 15 nói về cái gì?", top_n=3)

    assert ket_qua, "phải tìm được ít nhất 1 kết quả"
    assert ket_qua[0][0] == 1


def test_tim_kiem_tra_ve_rong_khi_khong_co_tu_nao_trung():
    bm25 = BM25(["Nội dung hoàn toàn khác."])
    assert bm25.tim_kiem("zzzzz qqqqq", top_n=5) == []


def test_diem_luon_duong():
    """Bản BM25 nguyên gốc có thể ra IDF ÂM với từ xuất hiện ở quá nửa số tài liệu, khiến
    một từ phổ biến lại TRỪ điểm của chính đoạn chứa nó - biến thể đang dùng phải tránh
    được điều đó."""
    van_ban = ["pháp luật là gì", "pháp luật do ai ban hành", "pháp luật và đạo đức"]
    bm25 = BM25(van_ban)
    for _, diem in bm25.tim_kiem("pháp luật", top_n=3):
        assert diem > 0


def test_corpus_rong_khong_gay_loi():
    assert BM25([]).tim_kiem("bất kỳ", top_n=5) == []


# ======================================================================
# BM25 ở vai trò CỨU HỘ (recall-only) trong pipeline
# ======================================================================
# Kết quả đo trên corpus song ngữ thật cho thấy BM25 gây HẠI khi được RRF cho quyền xếp
# hạng ngang dense (xem config.TRONG_SO_BM25). Nhưng cái hại đó đến từ QUYỀN XẾP HẠNG, không
# phải từ việc BM25 tìm sai. Vì vậy nó được giữ lại ở đúng một vai trò: bơm ứng viên vào tập
# đưa đi rerank, với điểm RRF bằng 0 nên tự nó không đẩy được gì lên - cross-encoder mới là
# nơi quyết định. Hai test dưới đây khoá chặt đúng hai nửa của hợp đồng đó.

import numpy as np
import pytest

import config
from rag.rag_pipeline import RagPipeline
from rag.vector_store import VectorStore

_SO_CHIEU = 4


class _EmbeddingGia:
    def __init__(self, vector):
        self.vector = np.array([vector], dtype="float32")

    def encode_cau_hoi(self, texts):
        return self.vector


class _RerankerGia:
    """Chấm cao cho đoạn chứa từ khoá - thay cho cross-encoder thật."""

    def __init__(self, tu_khoa):
        self.tu_khoa = tu_khoa
        self.cac_doan_da_cham = []

    def xep_hang(self, cau_hoi, cac_doan):
        self.cac_doan_da_cham = list(cac_doan)
        return np.array([9.0 if self.tu_khoa in d else 0.01 for d in cac_doan])


def _store_hai_doan():
    cac_chunk = [
        {"chunk_id": "c0", "nguon": "a.pdf", "trang": 1, "vi_tri": 0,
         "noidung": "Đoạn nói chung chung về quyền và nghĩa vụ của công dân."},
        {"chunk_id": "c1", "nguon": "b.pdf", "trang": 9, "vi_tri": 0,
         "noidung": "Điều 15 quy định cụ thể về nghĩa vụ nộp thuế của công dân."},
    ]
    store = VectorStore(dimension=_SO_CHIEU)
    # Chunk 1 (đoạn ĐÚNG) cố ý cho vector LỆCH hẳn khỏi câu hỏi: đúng tình huống mà từ khoá
    # hiếm bị embedding "hoà tan", tức chỗ dense một mình bó tay.
    store.them(
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype="float32"), cac_chunk
    )
    return store


@pytest.fixture
def chi_lay_mot_ung_vien_dense(monkeypatch):
    """Ép nhánh dense chỉ lấy về đúng 1 ứng viên, để đoạn còn lại CHỈ có thể vào tập ứng
    viên qua đường cứu hộ của BM25. Không ép thì store nhỏ nên dense lấy hết, và test không
    kiểm được gì."""
    monkeypatch.setattr(config, "HE_SO_OVER_FETCH", 1)
    monkeypatch.setattr(config, "SO_UNG_VIEN_TOI_THIEU", 1)
    monkeypatch.setattr(config, "TRONG_SO_BM25", 0.0)
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)
    monkeypatch.setattr(config, "TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT", 0.0)
    monkeypatch.setattr(config, "NGUONG_DIEM_RERANK_TOI_THIEU", 0.0)


def test_bm25_cuu_ho_dua_doan_tu_khoa_hiem_vao_tap_rerank(chi_lay_mot_ung_vien_dense, monkeypatch):
    """Nửa thứ nhất của hợp đồng: đoạn chứa từ khoá hiếm PHẢI có mặt trong tập ứng viên,
    và phải được cross-encoder chấm - nếu bị bỏ ngoài trần rerank thì cả cơ chế vô nghĩa."""
    monkeypatch.setattr(config, "SO_UNG_VIEN_BM25_CUU_HO", 10)
    reranker = _RerankerGia("Điều 15")

    ket_qua = RagPipeline(
        _EmbeddingGia([1.0, 0.0, 0.0, 0.0]), _store_hai_doan(), reranker_service=reranker
    ).truy_xuat("Điều 15 quy định gì?", top_k=1)

    assert any("Điều 15" in d for d in reranker.cac_doan_da_cham)
    assert ket_qua[0]["nguon"] == "b.pdf", "cross-encoder phải được quyền đẩy nó lên hạng 1"


def test_tat_cuu_ho_thi_doan_do_khong_vao_duoc_tap_ung_vien(chi_lay_mot_ung_vien_dense, monkeypatch):
    """Nửa thứ hai: khi tắt, hành vi phải quay về đúng như trước - đây là đường lui nếu đo
    lại trên corpus thật cho kết quả xấu."""
    monkeypatch.setattr(config, "SO_UNG_VIEN_BM25_CUU_HO", 0)
    reranker = _RerankerGia("Điều 15")

    ket_qua = RagPipeline(
        _EmbeddingGia([1.0, 0.0, 0.0, 0.0]), _store_hai_doan(), reranker_service=reranker
    ).truy_xuat("Điều 15 quy định gì?", top_k=1)

    assert not any("Điều 15" in d for d in reranker.cac_doan_da_cham)
    assert [d["nguon"] for d in ket_qua] == ["a.pdf"]
