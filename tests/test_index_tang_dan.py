"""Test ĐẦU-CUỐI cho build index TĂNG DẦN, chạy qua đúng nút "Đọc tài liệu" của giao diện."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import streamlit as st
from docx import Document
from streamlit.testing.v1 import AppTest

import config
import rag.document_loader as document_loader

DUONG_DAN_APP = str(Path(__file__).resolve().parent.parent / "app.py")


class _EmbeddingGia:
    dimension = 4
    thiet_bi = "cpu"
    max_seq_length = 512

    def encode_cau_hoi(self, texts):
        return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts), dtype="float32")

    def encode_tai_lieu(self, texts):
        return np.array(
            [[1.0, len(t) % 7, (len(t) * 3) % 5, 1.0] for t in texts], dtype="float32"
        )

    def lay_ham_dem_token(self):
        return lambda t: len(t.split())

    def chuyen_thiet_bi(self, moi):
        """Có mặt để khớp interface thật của EmbeddingService, tránh lỗi lệch interface chỉ lộ ra
        dưới dạng AttributeError giữa luồng build."""
        self.thiet_bi = moi
        return False


def _tao_docx(duong_dan: Path, noi_dung: str) -> None:
    tai_lieu = Document()
    tai_lieu.add_paragraph(noi_dung)
    tai_lieu.save(duong_dan)


# Đủ dài để vượt ngưỡng lọc chunk quá ngắn của chunking.py.
def _van_ban(nhan: str) -> str:
    return (
        f"Tai lieu {nhan}. Day la noi dung that cua tai lieu {nhan}, du dai de khong bi loc "
        f"boi nguong do dai chunk toi thieu, va khac han cac tai lieu con lai."
    )


@pytest.fixture
def moi_truong(monkeypatch):
    """Thư mục tài liệu + thư mục index riêng, model giả, không đụng tới Ollama."""
    with tempfile.TemporaryDirectory() as goc:
        goc = Path(goc)
        thu_muc_raw = goc / "raw"
        thu_muc_index = goc / "faiss_index"
        thu_muc_raw.mkdir()
        thu_muc_index.mkdir()

        monkeypatch.setattr(config, "RAW_DOCS_DIR", thu_muc_raw)
        monkeypatch.setattr(config, "FAISS_INDEX_FILE", thu_muc_index / "index.faiss")
        monkeypatch.setattr(config, "METADATA_MAPPING_FILE", thu_muc_index / "metadata.pkl")
        monkeypatch.setattr(config, "INDEX_INFO_FILE", thu_muc_index / "index_info.json")
        monkeypatch.setattr(config, "BAT_TRICH_ANH", False)
        monkeypatch.setattr(config, "BAT_CHU_THICH_ANH", False)
        monkeypatch.setattr(config, "BAT_INDEX_TANG_DAN", True)
        monkeypatch.setattr("rag.embedding.EmbeddingService", _EmbeddingGia)
        monkeypatch.setattr("rag.reranker.tao_reranker_neu_bat", lambda: None)
        monkeypatch.setattr("rag.rag_pipeline.kiem_tra_may_chu_llm", lambda: None)

        da_xu_ly: list = []
        goc_ham = document_loader.doc_tai_lieu_co_cache

        def _dem(duong_dan):
            da_xu_ly.append(duong_dan.name)
            return goc_ham(duong_dan)

        monkeypatch.setattr(document_loader, "doc_tai_lieu_co_cache", _dem)

        st.cache_data.clear()
        st.cache_resource.clear()
        yield thu_muc_raw, da_xu_ly


def _bam_nut_doc_tai_lieu(at):
    nut = next(b for b in at.sidebar.button if "Đọc tài liệu" in b.label)
    nut.click().run()
    assert not at.exception, at.exception
    return at


def _chay(moi_truong, session_state=None):
    at = AppTest.from_file(DUONG_DAN_APP, default_timeout=60)
    for khoa, gia_tri in (session_state or {}).items():
        at.session_state[khoa] = gia_tri
    at.run()
    assert not at.exception, at.exception
    return _bam_nut_doc_tai_lieu(at)


def test_them_mot_tai_lieu_chi_xu_ly_dung_tai_lieu_do(moi_truong):
    """Đây là kịch bản mà cả tính năng này tồn tại để phục vụ."""
    thu_muc_raw, da_xu_ly = moi_truong
    for nhan in ("a", "b", "c"):
        _tao_docx(thu_muc_raw / f"{nhan}.docx", _van_ban(nhan))

    at = _chay(moi_truong)
    store = at.session_state["vector_store"]
    assert sorted(da_xu_ly) == ["a.docx", "b.docx", "c.docx"]
    so_chunk_ban_dau = store.so_luong_vector
    assert so_chunk_ban_dau > 0

    da_xu_ly.clear()
    _tao_docx(thu_muc_raw / "d.docx", _van_ban("d"))
    at = _chay(moi_truong, {"vector_store": store})

    assert da_xu_ly == ["d.docx"], "ba tài liệu cũ không được đụng tới"
    store_moi = at.session_state["vector_store"]
    assert store_moi.so_luong_vector > so_chunk_ban_dau
    assert set(store_moi.bam_tai_lieu) == {"a.docx", "b.docx", "c.docx", "d.docx"}


def test_bam_nut_hai_lan_lien_tiep_khong_xu_ly_lai_gi(moi_truong):
    thu_muc_raw, da_xu_ly = moi_truong
    _tao_docx(thu_muc_raw / "a.docx", _van_ban("a"))

    at = _chay(moi_truong)
    store = at.session_state["vector_store"]
    so_chunk = store.so_luong_vector

    da_xu_ly.clear()
    at = _chay(moi_truong, {"vector_store": store})

    assert da_xu_ly == [], "không có gì đổi thì không tài liệu nào được xử lý lại"
    assert at.session_state["vector_store"].so_luong_vector == so_chunk, (
        "số chunk phải giữ nguyên - nhân đôi vector là dấu hiệu index chứa hai bản của "
        "cùng một tài liệu"
    )


def test_sua_noi_dung_thi_index_mang_ban_moi_va_khong_giu_ban_cu(moi_truong):
    """Giữ lại bản cũ là kiểu hỏng tệ nhất: trích dẫn trỏ vào nội dung không còn tồn tại."""
    thu_muc_raw, da_xu_ly = moi_truong
    _tao_docx(thu_muc_raw / "a.docx", _van_ban("a"))
    _tao_docx(thu_muc_raw / "b.docx", _van_ban("b"))

    at = _chay(moi_truong)
    store = at.session_state["vector_store"]
    so_chunk_ban_dau = store.so_luong_vector

    da_xu_ly.clear()
    _tao_docx(thu_muc_raw / "a.docx", _van_ban("a da duoc sua lai hoan toan khac truoc"))
    at = _chay(moi_truong, {"vector_store": store})

    assert da_xu_ly == ["a.docx"]
    store_moi = at.session_state["vector_store"]
    noi_dung = " ".join(m["noidung"] for m in store_moi.metadata if m["nguon"] == "a.docx")
    assert "da duoc sua lai" in noi_dung, "phải có bản mới"
    assert store_moi.so_luong_vector <= so_chunk_ban_dau + 2, (
        "bản cũ của a.docx phải bị gỡ, không được nằm lại trong index"
    )


def test_xoa_file_khoi_thu_muc_thi_vector_cua_no_bi_go(moi_truong):
    thu_muc_raw, da_xu_ly = moi_truong
    _tao_docx(thu_muc_raw / "a.docx", _van_ban("a"))
    _tao_docx(thu_muc_raw / "b.docx", _van_ban("b"))

    at = _chay(moi_truong)
    store = at.session_state["vector_store"]

    da_xu_ly.clear()
    (thu_muc_raw / "b.docx").unlink()
    at = _chay(moi_truong, {"vector_store": store})

    store_moi = at.session_state["vector_store"]
    assert da_xu_ly == [], "xoá file không làm tài liệu nào phải đọc lại"
    assert not any(m["nguon"] == "b.docx" for m in store_moi.metadata)
    assert set(store_moi.bam_tai_lieu) == {"a.docx"}


def test_file_doc_hong_khong_duoc_ghi_la_da_xu_ly(moi_truong):
    """Ghi băm cho một file đọc hỏng = nó bị bỏ qua ở MỌI lần build sau, im lặng mãi mãi."""
    thu_muc_raw, da_xu_ly = moi_truong
    _tao_docx(thu_muc_raw / "tot.docx", _van_ban("tot"))
    (thu_muc_raw / "hong.pdf").write_bytes(b"day khong phai file PDF hop le")

    at = _chay(moi_truong)
    store = at.session_state["vector_store"]
    assert set(store.bam_tai_lieu) == {"tot.docx"}, "file hỏng không được vào sổ băm"

    da_xu_ly.clear()
    at = _chay(moi_truong, {"vector_store": store})
    assert da_xu_ly == ["hong.pdf"], "file hỏng phải được thử lại ở lần build sau"


def test_tat_tang_dan_thi_build_lai_toan_bo(moi_truong, monkeypatch):
    """Đường lui về hành vi cũ, để đo đối chứng chi phí một lần build từ đầu."""
    thu_muc_raw, da_xu_ly = moi_truong
    monkeypatch.setattr(config, "BAT_INDEX_TANG_DAN", False)
    for nhan in ("a", "b"):
        _tao_docx(thu_muc_raw / f"{nhan}.docx", _van_ban(nhan))

    at = _chay(moi_truong)
    store = at.session_state["vector_store"]

    da_xu_ly.clear()
    at = _chay(moi_truong, {"vector_store": store})
    assert sorted(da_xu_ly) == ["a.docx", "b.docx"]
    assert at.session_state["vector_store"].so_luong_vector == store.so_luong_vector


def test_thanh_ben_hien_dung_luong_cache_va_xoa_duoc(moi_truong):
    """Người dùng phải có một cách dứt khoát để loại bỏ giả thuyết "hệ thống đang trả lời
    bằng nội dung cũ" - và phải nhìn thấy cache đang chiếm bao nhiêu chỗ trước khi xoá."""
    from rag import bo_nho_dem

    thu_muc_raw, _ = moi_truong
    _tao_docx(thu_muc_raw / "a.docx", _van_ban("a"))

    at = _chay(moi_truong)
    assert bo_nho_dem.dung_luong_cache() > 0, "build xong thì cache phải có nội dung"

    nhan = " ".join(c.value for c in at.sidebar.caption)
    assert "Cache đọc tài liệu" in nhan and "MB" in nhan

    nut_xoa = next(b for b in at.sidebar.button if "Xoá cache" in b.label)
    nut_xoa.click().run()
    assert not at.exception, at.exception
    assert bo_nho_dem.dung_luong_cache() == 0

