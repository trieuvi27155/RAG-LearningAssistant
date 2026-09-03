"""Test cơ bản cho retrieval (Embedding + FAISS) - chỉ 3-5 test theo đúng phạm vi đồ án.

Lưu ý: các test này tải model embedding thật (không mock) nên lần chạy đầu sẽ hơi
chậm hơn (phải tải model từ HuggingFace nếu máy chưa có sẵn trong cache)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from rag.embedding import EmbeddingService
from rag.vector_store import VectorStore


@pytest.fixture(scope="module")
def embedding_service():
    # scope="module": chỉ load model 1 lần cho toàn bộ file test này, vì load model
    # tốn vài giây - không cần load lại cho từng test riêng lẻ.
    return EmbeddingService()


def test_tim_kiem_tra_ve_ket_qua_giong_nhat_nhieu_nhat(embedding_service):
    van_ban = [
        "Python là một ngôn ngữ lập trình phổ biến, dùng nhiều trong khoa học dữ liệu.",
        "Con mèo là một loài động vật nuôi phổ biến trong các gia đình.",
        "Hà Nội là thủ đô của Việt Nam.",
    ]
    metadata = [{"nguon": "test.pdf", "trang": i + 1, "noidung": t} for i, t in enumerate(van_ban)]
    vectors = embedding_service.encode_tai_lieu(van_ban)

    store = VectorStore(dimension=embedding_service.dimension)
    store.them(vectors, metadata)

    cau_hoi = "Ngôn ngữ lập trình nào được nhắc đến trong tài liệu?"
    ket_qua = store.tim_kiem(embedding_service.encode_cau_hoi([cau_hoi]), top_k=1)

    assert ket_qua[0][0]["trang"] == 1  # chunk về Python phải là kết quả gần nhất


def test_diem_similarity_nam_trong_khoang_hop_le(embedding_service):
    van_ban = ["Một câu ví dụ đơn giản để kiểm tra độ tương đồng."]
    metadata = [{"nguon": "test.pdf", "trang": 1, "noidung": van_ban[0]}]
    vectors = embedding_service.encode_tai_lieu(van_ban)

    store = VectorStore(dimension=embedding_service.dimension)
    store.them(vectors, metadata)

    ket_qua = store.tim_kiem(embedding_service.encode_tai_lieu(van_ban), top_k=1)
    diem = ket_qua[0][1]

    assert -1.0001 <= diem <= 1.0001  # cosine similarity luôn nằm trong [-1, 1]
    assert diem > 0.99  # vector giống hệt câu đã lưu -> similarity phải gần 1


def test_cau_hoi_va_tai_lieu_duoc_ma_hoa_khac_nhau(embedding_service):
    """Model họ E5 được huấn luyện bất đối xứng (tiền tố "query: " / "passage: "). Nếu 2
    hàm này cho ra cùng một vector nghĩa là tiền tố không được áp dụng - lỗi âm thầm làm
    tụt chất lượng truy xuất mà không có dấu hiệu nào lộ ra."""
    if not config.EMBEDDING_QUERY_PREFIX and not config.EMBEDDING_PASSAGE_PREFIX:
        pytest.skip("Model đang dùng không cần tiền tố bất đối xứng")

    text = ["Nhà nước là tổ chức quyền lực chính trị đặc biệt."]
    assert not (embedding_service.encode_cau_hoi(text) == embedding_service.encode_tai_lieu(text)).all()


def test_chunk_khong_vuot_gioi_han_token_cua_model(embedding_service):
    """Ràng buộc quan trọng nhất giữa chunking và embedding: chunk dài hơn max_seq_length
    bị model cắt bỏ âm thầm."""
    from rag.chunking import chia_chunk

    cac_trang = [{"nguon": "a.pdf", "trang": 1, "noidung": "Nội dung tiếng Việt để kiểm tra. " * 200}]
    cac_chunk = chia_chunk(
        cac_trang,
        dem_token_fn=embedding_service.lay_ham_dem_token(),
        max_seq_length=embedding_service.max_seq_length,
    )

    assert cac_chunk
    for chunk in cac_chunk:
        assert embedding_service.dem_token(chunk["noidung"]) <= embedding_service.max_seq_length


def test_luu_va_tai_lai_index_giu_nguyen_ket_qua(embedding_service):
    # Cố tình KHÔNG dùng fixture tmp_path mặc định của pytest: trên máy có tên hiển thị
    # Windows (display name) chứa ký tự tiếng Việt có dấu, pytest tạo thư mục tạm dạng
    # "pytest-of-<display-name>", và FAISS (dùng fopen theo ANSI codepage ở tầng C++)
    # không ghi được vào đường dẫn đó. tempfile.gettempdir() trả về đường dẫn an toàn hơn.
    van_ban = ["Nội dung A về chủ đề X.", "Nội dung B về chủ đề Y."]
    metadata = [{"nguon": "x.pdf", "trang": i + 1, "noidung": t} for i, t in enumerate(van_ban)]
    vectors = embedding_service.encode_tai_lieu(van_ban)

    store = VectorStore(dimension=embedding_service.dimension)
    store.them(vectors, metadata)

    with tempfile.TemporaryDirectory() as thu_muc_tam:
        thu_muc_tam = Path(thu_muc_tam)
        duong_dan = dict(
            index_path=thu_muc_tam / "index.faiss",
            metadata_path=thu_muc_tam / "metadata.pkl",
            info_path=thu_muc_tam / "index_info.json",
        )
        store.luu(**duong_dan)

        store_tai_lai = VectorStore.tai(**duong_dan)
        assert store_tai_lai.so_luong_vector == store.so_luong_vector
        assert store_tai_lai.metadata == store.metadata
        # Index vừa build bằng đúng cấu hình hiện tại -> không được báo không tương thích.
        assert store_tai_lai.ly_do_khong_tuong_thich() is None


def test_canh_bao_khi_index_build_bang_model_khac(embedding_service, monkeypatch):
    """Đổi model embedding mà quên build lại index KHÔNG gây crash (2 model có thể cùng số
    chiều) - chỉ khiến kết quả truy xuất sai âm thầm. Phải phát hiện được bằng vân tay."""
    store = VectorStore(dimension=embedding_service.dimension)
    store.them(embedding_service.encode_tai_lieu(["Một đoạn."]), [{"nguon": "x.pdf", "trang": 1, "noidung": "Một đoạn."}])
    store.thong_tin = {"embedding_model": "model-cu-nao-do", "chunk_size_tokens": config.CHUNK_SIZE_TOKENS}

    ly_do = store.ly_do_khong_tuong_thich()
    assert ly_do and "model-cu-nao-do" in ly_do


def test_xoa_theo_nguon_lam_moi_chi_muc_phu(embedding_service):
    """Chỉ mục trang và BM25 đều ánh xạ theo vị trí trong metadata; xoá vector làm các vị
    trí đó dịch đi, nếu giữ lại cache cũ thì tìm kiếm sẽ trả về nội dung của chunk khác."""
    van_ban = ["Nội dung của file A.", "Nội dung của file B."]
    metadata = [
        {"nguon": "a.pdf", "trang": 1, "vi_tri": 0, "noidung": van_ban[0]},
        {"nguon": "b.pdf", "trang": 1, "vi_tri": 0, "noidung": van_ban[1]},
    ]
    store = VectorStore(dimension=embedding_service.dimension)
    store.them(embedding_service.encode_tai_lieu(van_ban), metadata)

    assert len(store.theo_nguon_va_trang("a.pdf", 1)) == 1  # ép dựng cache trước khi xoá
    assert store.bm25.so_tai_lieu == 2

    assert store.xoa_theo_nguon("a.pdf") == 1
    assert store.theo_nguon_va_trang("a.pdf", 1) == []
    assert store.theo_nguon_va_trang("b.pdf", 1)[0]["noidung"] == van_ban[1]
    assert store.bm25.so_tai_lieu == 1
