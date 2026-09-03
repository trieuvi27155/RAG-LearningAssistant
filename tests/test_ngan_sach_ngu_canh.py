"""Test cho NGÂN SÁCH CỬA SỔ NGỮ CẢNH (num_ctx) - bug im lặng nghiêm trọng nhất đã gặp.

Ollama cấp mặc định 4096 token bất kể model hỗ trợ bao nhiêu, và khi prompt vượt quá thì
nó KHÔNG báo lỗi mà cắt từ ĐẦU phần user content - tức xoá đúng các đoạn trích liên quan
nhất, vì _ghep_prompt() xếp đoạn tốt nhất lên trước. Triệu chứng lộ ra là "câu trả lời
ngắn" và "truy xuất kém trên tài liệu mới", cả hai đều dẫn người sửa đi sai hướng.

Vì vậy thứ cần test ở đây không phải chất lượng câu trả lời (không đo được bằng unit test)
mà là ba điều tất định:
  1. num_ctx CÓ MẶT trong options gửi lên Ollama - thiếu nó là quay lại đúng bug.
  2. Cửa sổ được nới khi prompt dài, và không bao giờ vượt trần RAM đã đặt.
  3. Bộ đếm token thật do máy chủ trả về được ghi nhận lại, để lần sau lỗi này không còn
     im lặng nữa.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import config
from rag.rag_pipeline import RagPipeline, _tinh_num_ctx, _uoc_luong_so_token
from rag.vector_store import VectorStore

SO_CHIEU = 4


class EmbeddingGia:
    def __init__(self, vector):
        self.vector = np.array([vector], dtype="float32")

    def encode_cau_hoi(self, texts):
        return self.vector


class OllamaGia:
    def __init__(self, cac_manh):
        self.cac_manh = cac_manh
        self.tham_so_da_dung = None

    def chat(self, **tham_so):
        self.tham_so_da_dung = tham_so
        return iter(self.cac_manh)


def _tao_pipeline(cac_manh, noi_dung_chunk="Nội dung tài liệu đủ dài để không bị lọc bỏ."):
    chunk = {"chunk_id": "c1", "nguon": "a.pdf", "trang": 1, "vi_tri": 0,
             "noidung": noi_dung_chunk}
    store = VectorStore(dimension=SO_CHIEU)
    store.them(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), [chunk])
    pipeline = RagPipeline(EmbeddingGia([1.0, 0.0, 0.0, 0.0]), store)
    pipeline._ollama_client = OllamaGia(cac_manh)
    return pipeline


# ======================================================================
# Tính num_ctx
# ======================================================================

def test_uoc_luong_token_luon_du_chu_khong_thieu():
    """Ước lượng chỉ dùng để CẤP PHÁT ngân sách: cấp dư thì vô hại, cấp thiếu thì tái tạo
    lại đúng bug. Nên tỷ lệ ký tự/token phải đặt thấp hơn giá trị đo được cho tiếng Việt
    (~2.5), tức ước lượng ra SỐ TOKEN LỚN HƠN thực tế."""
    assert config.SO_KY_TU_MOI_TOKEN_UOC_LUONG < 2.5
    assert _uoc_luong_so_token("x" * 1000) > 1000 / 2.5


def test_num_ctx_khong_bao_gio_duoi_muc_cau_hinh(monkeypatch):
    """Sàn = OLLAMA_NUM_CTX. Prompt ngắn KHÔNG được co cửa sổ lại: Ollama coi num_ctx là
    một phần định danh của model đang nạp, đổi giá trị giữa hai lượt hỏi khiến nó nạp lại
    model (hàng chục giây trên CPU)."""
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 16384)
    assert _tinh_num_ctx(10) == 16384
    assert _tinh_num_ctx(5000) == 16384


def test_num_ctx_duoc_noi_khi_prompt_qua_dai(monkeypatch):
    """Đây là thứ khiến bug không tái diễn im lặng khi ai đó tăng TOP_K hay
    NGAN_SACH_KY_TU_MOI_DOAN: cửa sổ tự nới theo thay vì cắt prompt."""
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 8192)
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX_TOI_DA", 65536)
    monkeypatch.setattr(config, "OLLAMA_DU_PHONG_TOKEN_SINH", 4000)

    # 8192 không đủ cho prompt 9000 token + 4000 token dự phòng -> phải nới lên bậc kế tiếp.
    assert _tinh_num_ctx(9000) == 16384
    assert _tinh_num_ctx(30000) == 65536


def test_num_ctx_bi_chan_boi_tran_ram(monkeypatch, caplog):
    """Trần là chốt chặn RAM. Vượt trần thì phải NÓI RA chỗ cần sửa, vì triệu chứng của việc
    thiếu cửa sổ trông y hệt "model tự nhiên trả lời ngắn đi"."""
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 8192)
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX_TOI_DA", 16384)

    with caplog.at_level("WARNING"):
        assert _tinh_num_ctx(100000) == 16384
    assert any("OLLAMA_NUM_CTX_TOI_DA" in r.getMessage() for r in caplog.records)


# ======================================================================
# num_ctx thật sự đi tới Ollama
# ======================================================================

def test_num_ctx_co_trong_options_gui_len_ollama(monkeypatch):
    """Test QUAN TRỌNG NHẤT của file này: thiếu đúng một khoá trong dict options là quay
    lại nguyên vẹn bug cũ, mà không có lỗi nào báo ra."""
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)
    monkeypatch.setattr(config, "TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT", 0.0)
    pipeline = _tao_pipeline([{"message": {"content": "Câu trả lời."}}])

    list(pipeline.sinh_cau_tra_loi_theo_luong("câu hỏi", pipeline.truy_xuat("câu hỏi")))

    options = pipeline._ollama_client.tham_so_da_dung["options"]
    assert "num_ctx" in options, "thiếu num_ctx -> Ollama lặng lẽ cấp 4096 và cắt prompt"
    assert options["num_ctx"] >= config.OLLAMA_NUM_CTX


def test_ghi_nhan_bo_dem_token_that_cua_may_chu(monkeypatch):
    """prompt_eval_count là con số DUY NHẤT chứng minh được prompt có bị cắt hay không.
    Không ghi lại nó thì lớp lỗi này không quan sát được, và mọi kết luận rút ra từ chất
    lượng câu trả lời đều thiếu căn cứ."""
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)
    monkeypatch.setattr(config, "TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT", 0.0)
    pipeline = _tao_pipeline([
        {"message": {"content": "Câu trả lời."}},
        {"message": {"content": ""}, "done": True, "done_reason": "stop",
         "prompt_eval_count": 1234, "eval_count": 56},
    ])

    list(pipeline.sinh_cau_tra_loi_theo_luong("câu hỏi", pipeline.truy_xuat("câu hỏi")))

    assert pipeline.thong_ke_llm["prompt_eval_count"] == 1234
    assert pipeline.thong_ke_llm["eval_count"] == 56
    assert pipeline.thong_ke_llm["done_reason"] == "stop"
    assert pipeline.thong_ke_llm["num_ctx"] >= config.OLLAMA_NUM_CTX


def test_canh_bao_khi_prompt_cham_tran_cua_so(monkeypatch, caplog):
    """Máy chủ nạp đúng bằng (hoặc hơn) num_ctx nghĩa là prompt ĐÃ bị cắt. Đây là lúc duy
    nhất hệ thống nhìn thấy được sự việc - im lặng ở đây là để lỗi trôi vào báo cáo."""
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)
    monkeypatch.setattr(config, "TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT", 0.0)
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 4096)
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX_TOI_DA", 4096)
    pipeline = _tao_pipeline([
        {"message": {"content": "Trả lời cụt"}},
        {"message": {"content": ""}, "done": True, "done_reason": "length",
         "prompt_eval_count": 4096, "eval_count": 0},
    ])

    with caplog.at_level("WARNING"):
        list(pipeline.sinh_cau_tra_loi_theo_luong("câu hỏi", pipeline.truy_xuat("câu hỏi")))

    thong_bao = " ".join(r.getMessage() for r in caplog.records)
    assert "PROMPT BỊ CẮT" in thong_bao
    assert "done_reason='length'" in thong_bao
