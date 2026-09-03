"""Test cho việc xử lý lỗi KHÔNG KẾT NỐI ĐƯỢC máy chủ Ollama.

Vì sao đáng có một file test riêng cho một lỗi hạ tầng: đây là hỏng hóc số một khi đem hệ
thống chạy trên một máy mới (Ollama là tiến trình nền riêng, không tự lên cùng Streamlit),
và cái bẫy nằm ở chỗ rất dễ tưởng là đã xử lý rồi:

  - Thư viện ollama CÓ dịch httpx.ConnectError sang ConnectionError kèm thông báo tử tế -
    nhưng chỉ ở đường gọi không streaming (`_request_raw`).
  - Đường streaming (đường DUY NHẤT hệ thống này dùng để sinh câu trả lời) mở kết nối bên
    trong generator, tức là ở lần lặp ĐẦU TIÊN chứ không phải lúc gọi hàm - nằm ngoài khối
    bọc lỗi đó. Kết quả: người dùng nhận nguyên traceback "ConnectError: [WinError 10061]
    ... target machine actively refused it", không một chữ nào nhắc tới Ollama.

Nên các test dưới đây đều bắt chước ĐÚNG cách hỏng đó (lỗi chỉ bung ra khi lặp), chứ không
phải cách hỏng dễ test hơn là ném lỗi ngay lúc gọi.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import numpy as np
import pytest

import config
from rag.rag_pipeline import (
    LoiKhongKetNoiDuocOllama,
    RagPipeline,
    kiem_tra_may_chu_llm,
)
from rag.vector_store import VectorStore

SO_CHIEU = 4


class EmbeddingGia:
    def __init__(self):
        v = [0.0] * SO_CHIEU
        v[0] = 1.0
        self.vector = np.array([v], dtype="float32")

    def encode_cau_hoi(self, texts):
        return self.vector


class ClientOllamaChuaChay:
    """Bắt chước ollama.Client khi máy chủ chưa chạy, ở chế độ stream=True.

    Điểm mấu chốt: `chat()` TRẢ VỀ BÌNH THƯỜNG, lỗi chỉ nổ ra khi generator được lặp - đúng
    như httpx.Client.stream() thật. Một fake ném lỗi ngay tại `chat()` sẽ vẫn xanh kể cả
    khi bug quay lại, nên nó vô dụng.
    """

    def __init__(self, loi=None):
        self.loi = loi or httpx.ConnectError("[WinError 10061] ... actively refused it")

    def chat(self, **_):
        def sinh():
            raise self.loi
            yield  # pragma: no cover - chỉ để hàm này là generator

        return sinh()


def _pipeline(client) -> RagPipeline:
    store = VectorStore(dimension=SO_CHIEU)
    store.them(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"),
        [{"chunk_id": "a-1-0", "nguon": "a.pdf", "trang": 1, "vi_tri": 0,
          "noidung": "Nội dung mẫu trong tài liệu để có gì mà truy xuất."}],
    )
    p = RagPipeline(EmbeddingGia(), store)
    p._ollama_client = client
    return p


@pytest.fixture(autouse=True)
def khong_chan_boi_nguong(monkeypatch):
    """Bỏ ngưỡng điểm để câu hỏi luôn đi tới bước gọi LLM - chỗ cần kiểm nằm ở đó."""
    monkeypatch.setattr(config, "TRONG_SO_BM25", 0.0)
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)


# ======================================================================
# Lỗi kết nối phải thành một lỗi CÓ NGHĨA, không phải traceback httpx thô
# ======================================================================

def test_loi_ket_noi_thanh_loi_rieng_co_huong_dan():
    pipeline = _pipeline(ClientOllamaChuaChay())

    with pytest.raises(LoiKhongKetNoiDuocOllama) as loi:
        pipeline.hoi_dap("Tài liệu nói về gì?")

    thong_bao = str(loi.value)
    # Ba thứ người dùng cần để tự sửa: hỏng ở đâu, host nào, chạy lệnh gì.
    assert "Ollama" in thong_bao
    assert config.OLLAMA_HOST in thong_bao
    assert f"ollama pull {config.OLLAMA_MODEL}" in thong_bao


def test_giu_nguyen_loi_goc_lam_nguyen_nhan():
    """`raise ... from loi` chứ không nuốt: log phía server vẫn phải truy được về WinError."""
    goc = httpx.ConnectError("[WinError 10061] ... actively refused it")
    pipeline = _pipeline(ClientOllamaChuaChay(loi=goc))

    with pytest.raises(LoiKhongKetNoiDuocOllama) as loi:
        pipeline.hoi_dap("Tài liệu nói về gì?")

    assert loi.value.__cause__ is goc


def test_bat_ca_ConnectionError_cua_thu_vien_ollama():
    """Bản không streaming của ollama dịch lỗi sang ConnectionError - cũng phải bắt được,
    để hai đường gọi không cho ra hai trải nghiệm khác nhau."""
    pipeline = _pipeline(ClientOllamaChuaChay(loi=ConnectionError("Failed to connect to Ollama")))

    with pytest.raises(LoiKhongKetNoiDuocOllama):
        pipeline.hoi_dap("Tài liệu nói về gì?")


def test_truy_xuat_van_chay_khi_ollama_chet():
    """Chỉ bước SINH câu trả lời phụ thuộc Ollama. Truy xuất là FAISS + model local, phải
    còn nguyên - đây là căn cứ để thông báo lỗi dám nói 'vẫn đọc được tài liệu'."""
    pipeline = _pipeline(ClientOllamaChuaChay())

    cac_doan = pipeline.truy_xuat("Tài liệu nói về gì?", top_k=1)

    assert len(cac_doan) == 1
    assert cac_doan[0]["nguon"] == "a.pdf"


# ======================================================================
# Kiểm tra sẵn sàng TRƯỚC khi người dùng hỏi (cảnh báo ở thanh bên)
# ======================================================================

class _Model:
    def __init__(self, ten):
        self.model = ten


class _DanhSach:
    def __init__(self, ten_cac_model):
        self.models = [_Model(t) for t in ten_cac_model]


def _gia_lap_client(monkeypatch, *, ten_cac_model=None, loi=None):
    class ClientGia:
        def __init__(self, host=None):
            pass

        def list(self):
            if loi:
                raise loi
            return _DanhSach(ten_cac_model or ())

    monkeypatch.setattr("rag.rag_pipeline.ollama.Client", ClientGia)


def test_kiem_tra_bao_loi_khi_chua_bat_ollama(monkeypatch):
    _gia_lap_client(monkeypatch, loi=ConnectionError("Failed to connect to Ollama"))

    thong_bao = kiem_tra_may_chu_llm()

    assert thong_bao and "Ollama" in thong_bao
    assert "ollama serve" in thong_bao


def test_kiem_tra_bao_loi_khi_thieu_model(monkeypatch):
    """Ollama chạy nhưng chưa pull model là một hỏng hóc KHÁC, cần một hướng dẫn khác -
    bảo người ta 'bật Ollama lên' trong lúc Ollama đang chạy chỉ làm họ đi sai đường."""
    _gia_lap_client(monkeypatch, ten_cac_model=("mot-model-khac:7b",))

    thong_bao = kiem_tra_may_chu_llm()

    assert thong_bao and f"ollama pull {config.OLLAMA_MODEL}" in thong_bao
    assert "ollama serve" not in thong_bao


def test_kiem_tra_im_lang_khi_moi_thu_binh_thuong(monkeypatch):
    _gia_lap_client(monkeypatch, ten_cac_model=(config.OLLAMA_MODEL,))

    assert kiem_tra_may_chu_llm() is None


def test_kiem_tra_chap_nhan_ten_model_khong_ghi_tag(monkeypatch):
    """Người dùng đặt OLLAMA_MODEL='qwen3' trong khi Ollama liệt kê 'qwen3:4b' - coi là
    khớp, thay vì bắt họ nhớ đúng tag rồi báo lỗi sai nguyên nhân."""
    monkeypatch.setattr(config, "OLLAMA_MODEL", "qwen3")
    _gia_lap_client(monkeypatch, ten_cac_model=("qwen3:4b",))

    assert kiem_tra_may_chu_llm() is None
