"""Test cho GIAO DIỆN (app.py), chạy bằng AppTest của chính Streamlit.

Vì sao cần: app.py là "composition root" - nơi duy nhất mọi module rag/* được ghép lại, và
là nơi giữ toàn bộ trạng thái phiên. Nhưng nó cũng là phần DUY NHẤT của hệ thống trước nay
không có test nào: mọi lỗi ở đây chỉ lộ ra khi có người mở trình duyệt lên bấm thử.

Bố cục vừa được viết lại (§5.47) nên rủi ro cao nhất nằm đúng ở những chỗ khó thấy bằng mắt:
  - luồng hỏi-đáp 2 nhịp (đặt câu hỏi -> rerun -> mới gọi LLM). Đây là cơ chế chống việc một
    cú bấm bất kỳ huỷ ngang lần chạy đang gọi Ollama (bug đã gặp thực tế). Nhìn màn hình
    không thể biết nó còn đúng hay không.
  - trích dẫn phải nằm TRONG từng tin nhắn, không dùng biến dùng chung (bug lệch pha đã gặp).
  - hai lối đặt câu hỏi (ô nhập và nút gợi ý) phải đi đúng một đường mã.

AppTest chạy thẳng script Streamlit trong tiến trình, không cần trình duyệt - nên test này
nhanh, tất định, và chạy được trong CI. LLM và model embedding đều được thay bằng đồ giả:
thứ cần kiểm ở đây là LUỒNG GIAO DIỆN, không phải chất lượng câu trả lời.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
from rag.rag_pipeline import LoiKhongKetNoiDuocOllama

DUONG_DAN_APP = str(Path(__file__).resolve().parent.parent / "app.py")

CAU_TRA_LOI_GIA = "Nhà nước có tính giai cấp và quyền lực công cộng đặc biệt [1]."


class _EmbeddingGia:
    dimension = 4
    thiet_bi = "cpu"
    max_seq_length = 512

    def encode_cau_hoi(self, texts):
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")

    def encode_tai_lieu(self, texts):
        return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts), dtype="float32")

    def lay_ham_dem_token(self):
        return lambda t: len(t.split())

    def chuyen_thiet_bi(self, moi):
        """Có mặt để khớp interface thật của EmbeddingService (rag/tai_nguyen_gpu.py gọi tới
        ở ranh giới giai đoạn). Test double thiếu method này thì lỗi lệch interface sẽ hiện
        ra dưới dạng AttributeError giữa luồng build, chứ không phải một test đỏ rõ ràng."""
        self.thiet_bi = moi
        return False


class _PipelineGia:
    """Thay RagPipeline: trả về đúng hình dạng dữ liệu thật, không gọi Ollama."""

    def __init__(self):
        self.cac_cau_hoi_da_nhan = []

    def _chunk_nguon(self):
        return [{
            "nguon": "phapluat.pdf", "trang": 12,
            "noidung": "Nhà nước có tính giai cấp và quyền lực công cộng đặc biệt.",
            "doan_khop": "Nhà nước có tính giai cấp và quyền lực công cộng đặc biệt.",
            "diem_similarity": 0.9, "loai_noi_dung": "van_ban",
        }]

    def _ket_qua(self, cau_hoi):
        self.cac_cau_hoi_da_nhan.append(cau_hoi)
        return {
            "cau_tra_loi": CAU_TRA_LOI_GIA,
            "cac_chunk_nguon": self._chunk_nguon(),
            "la_kiem_chung": False,
            "truy_van": {"cau_hoi_goc": cau_hoi, "cau_hoi_truy_xuat": cau_hoi,
                         "da_viet_lai": False, "la_tiep_noi": False},
            "mau_thuan": [],
            "do_tre": {"truy_xuat": 0.5, "hien_dau_tien": 0.6, "chu_dau_tien": 1.2, "tong": 2.0},
        }

    def hoi_dap(self, cau_hoi, top_k=None, nguon_cho_phep=None, lich_su=None, doi_chieu=None):
        self.lich_su_da_nhan = lich_su
        return self._ket_qua(cau_hoi)

    def hoi_dap_theo_luong(self, cau_hoi, top_k=None, nguon_cho_phep=None,
                           lich_su=None, doi_chieu=None):
        self.lich_su_da_nhan = lich_su
        ket_qua = self._ket_qua(cau_hoi)
        yield {"loai": "truy_xuat_xong", "cac_chunk": ket_qua["cac_chunk_nguon"], "giay": 0.5}
        yield {"loai": "suy_luan", "them": "đang nghĩ"}
        yield {"loai": "cau_tra_loi", "them": CAU_TRA_LOI_GIA}
        yield {"loai": "xong", "ket_qua": ket_qua}


class _PipelineHong:
    """Pipeline ném lỗi ngay khi bắt đầu sinh câu trả lời - đúng như lúc Ollama chưa chạy."""

    def __init__(self, loi=None):
        self.loi = loi or LoiKhongKetNoiDuocOllama("Không kết nối được tới máy chủ Ollama ở ...")

    def hoi_dap(self, cau_hoi, top_k=None, nguon_cho_phep=None, lich_su=None, doi_chieu=None):
        raise self.loi

    def hoi_dap_theo_luong(self, cau_hoi, top_k=None, nguon_cho_phep=None,
                           lich_su=None, doi_chieu=None):
        raise self.loi
        yield  # pragma: no cover - chỉ để hàm này là generator


class _StoreGia:
    so_luong_vector = 42

    def ly_do_khong_tuong_thich(self):
        return None


@pytest.fixture
def app(monkeypatch, tmp_path):
    """AppTest đã thay sẵn model/pipeline bằng đồ giả và trỏ thư mục tài liệu sang tmp."""
    thu_muc = tmp_path / "raw"
    thu_muc.mkdir()
    (thu_muc / "phapluat.pdf").write_bytes(b"%PDF-1.4 noi dung gia")
    monkeypatch.setattr(config, "RAW_DOCS_DIR", thu_muc)

    # Thanh bên có kiểm tra máy chủ Ollama trước khi người dùng kịp hỏi. Test giao diện
    # không được phụ thuộc vào việc máy chạy test có bật Ollama hay không, nên cắt hẳn lời
    # gọi ra ngoài (kiểm tra đó có test riêng ở test_ket_noi_ollama.py).
    monkeypatch.setattr("rag.rag_pipeline.kiem_tra_may_chu_llm", lambda: None)
    st.cache_data.clear()

    at = AppTest.from_file(DUONG_DAN_APP, default_timeout=30)
    pipeline_gia = _PipelineGia()
    at.session_state["vector_store"] = _StoreGia()
    at.session_state["pipeline"] = pipeline_gia
    at.session_state["pipeline_cho_store"] = at.session_state["vector_store"]
    at.pipeline_gia = pipeline_gia
    return at


# ======================================================================
# Bố cục: thanh bên có đủ phần quản lý nguồn, khung chính có ô nhập
# ======================================================================

def test_thanh_ben_co_du_phan_quan_ly_nguon(app):
    at = app.run()

    assert not at.exception, at.exception
    nhan_nut = [b.label for b in at.sidebar.button]
    assert any("Hội thoại mới" in n for n in nhan_nut)
    assert any("Đọc tài liệu" in n for n in nhan_nut)
    assert at.sidebar.checkbox, "phải có checkbox chọn nguồn cho từng tài liệu"
    assert at.chat_input, "khung chính phải có ô nhập câu hỏi"


def test_man_hinh_trong_co_goi_y_bam_duoc(app):
    at = app.run()
    assert not at.exception, at.exception
    # 3 nút gợi ý nằm ở KHUNG CHÍNH (không phải thanh bên).
    nut_chinh = [b for b in at.main.button]
    assert len(nut_chinh) >= 3, f"cần ít nhất 3 gợi ý, đang có {len(nut_chinh)}"


# ======================================================================
# Luồng hỏi-đáp 2 nhịp
# ======================================================================

def test_dat_cau_hoi_khong_goi_llm_ngay_o_nhip_dau(app):
    """Nhịp 1 chỉ được ghi câu hỏi vào state rồi rerun. Nếu gọi LLM ngay tại đây thì một cú
    bấm bất kỳ trong lúc chờ sẽ huỷ ngang lần chạy và mất câu trả lời đang sinh dở."""
    at = app.run()
    at.chat_input[0].set_value("Nhà nước có đặc điểm gì?").run()

    assert at.session_state["dang_xu_ly"] is False, "sau nhịp 2 thì cờ phải được hạ xuống"
    assert app.pipeline_gia.cac_cau_hoi_da_nhan == ["Nhà nước có đặc điểm gì?"], (
        "LLM phải được gọi đúng MỘT lần, ở nhịp 2"
    )


def test_cau_tra_loi_va_trich_dan_duoc_luu_trong_chinh_tin_nhan(app):
    """Trích dẫn phải nằm TRONG tin nhắn, không dùng một biến 'trích dẫn hiện tại' dùng
    chung - đó là nguyên nhân bug lệch pha đã gặp (hỏi câu mới, trích dẫn còn của câu cũ)."""
    at = app.run()
    at.chat_input[0].set_value("Nhà nước có đặc điểm gì?").run()

    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == CAU_TRA_LOI_GIA
    assert messages[1]["trich_dan"], "câu trả lời có dẫn [1] thì phải kèm trích dẫn"
    assert messages[1]["trich_dan"][0]["nguon"] == "phapluat.pdf"
    assert messages[1]["do_tre"]["tong"] == 2.0


def test_nut_goi_y_va_o_nhap_di_cung_mot_duong(app):
    """Hai lối vào phải cho ra cùng một trạng thái - nếu một lối lách được cơ chế 2 nhịp thì
    bug huỷ-ngang quay lại chỉ ở đúng lối đó, rất khó phát hiện."""
    at = app.run()
    nhan_goi_y = at.main.button[0].label
    at.main.button[0].click().run()

    assert app.pipeline_gia.cac_cau_hoi_da_nhan == [nhan_goi_y]
    assert at.session_state["messages"][0]["content"] == nhan_goi_y
    assert at.session_state["messages"][1]["role"] == "assistant"


def test_hoi_thoai_moi_xoa_lich_su_nhung_giu_index(app):
    at = app.run()
    at.chat_input[0].set_value("Câu hỏi bất kỳ").run()
    assert at.session_state["messages"]

    nut = next(b for b in at.sidebar.button if "Hội thoại mới" in b.label)
    nut.click().run()

    assert at.session_state["messages"] == []
    assert at.session_state["vector_store"] is not None, "index tuyệt đối không được mất"


# ======================================================================
# Lỗi ở bước gọi LLM không được khoá cứng cả phiên làm việc
# ======================================================================

def test_loi_khi_sinh_cau_tra_loi_khong_lam_treo_giao_dien(app):
    """Đây là hậu quả THẬT của việc app.py không bắt lỗi: dang_xu_ly kẹt ở True, nên ô nhập
    và mọi nút vẫn disabled - người dùng không làm được gì nữa kể cả sau khi đã bật Ollama
    lên, cho tới lúc tự tải lại trang. Một lỗi hạ tầng tạm thời hoá ra làm hỏng cả phiên."""
    app.session_state["pipeline"] = _PipelineHong()
    at = app.run()
    at.chat_input[0].set_value("Nhà nước có đặc điểm gì?").run()

    assert not at.exception, at.exception
    assert at.session_state["dang_xu_ly"] is False, "cờ bận phải được hạ, nếu không app kẹt"
    assert at.session_state["cau_hoi_dang_xu_ly"] is None
    assert at.chat_input[0].disabled is False, "phải hỏi lại được ngay sau khi sửa xong"


def test_loi_ket_noi_hien_thanh_tin_nhan_kem_huong_dan(app):
    """Thông báo phải NẰM LẠI trong lịch sử chứ không chỉ st.error(): app rerun ngay sau đó,
    st.error vẽ xong là mất, người dùng chưa kịp đọc đã không còn gì trên màn hình."""
    app.session_state["pipeline"] = _PipelineHong()
    at = app.run()
    at.chat_input[0].set_value("Nhà nước có đặc điểm gì?").run()

    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert "Ollama" in messages[1]["content"]
    assert messages[1]["trich_dan"] == [], "câu trả lời hỏng thì tuyệt đối không kèm nguồn"


def test_loi_la_khong_bi_nuot_im_lang(app):
    """Bắt Exception rộng nhưng phải nói ra được là lỗi gì - nếu không, mọi hỏng hóc khác
    đều hiện ra y hệt nhau và không còn cách nào lần ra nguyên nhân."""
    app.session_state["pipeline"] = _PipelineHong(loi=ValueError("chunk_id trùng"))
    at = app.run()
    at.chat_input[0].set_value("Nhà nước có đặc điểm gì?").run()

    assert not at.exception, at.exception
    noi_dung = at.session_state["messages"][1]["content"]
    assert "ValueError" in noi_dung and "chunk_id trùng" in noi_dung


def test_so_trich_dan_khong_hien_len_man_hinh(app):
    """Số [n] là thứ tự đoạn trích TRONG PROMPT - một thứ tự người đọc không nhìn thấy nên
    cũng không tra ngược được. Hiện ra chỉ thêm nhiễu.

    Nhưng chúng PHẢI còn nguyên trong dữ liệu: đó là căn cứ để loc_theo_tham_chieu() biết câu
    trả lời dùng nguồn nào, và để metrics.do_chinh_xac_trich_dan() chấm Citation accuracy.
    Test khoá lại cả hai vế cùng lúc - gỡ số khỏi dữ liệu thay vì khỏi hiển thị sẽ phá đúng
    thứ mà các con số này sinh ra để phục vụ, mà không có lỗi nào báo ra."""
    at = app.run()
    at.chat_input[0].set_value("Nhà nước có đặc điểm gì?").run()

    # Vế 1: không còn số nào trên màn hình.
    van_ban_hien = " ".join(m.value for m in at.markdown)
    assert "[1]" not in van_ban_hien
    assert "quyền lực công cộng đặc biệt." in van_ban_hien, "nội dung phải còn nguyên"

    # Vế 2: dữ liệu gốc vẫn giữ số, và trích dẫn vẫn chọn được nguồn nhờ chính số đó.
    tin_nhan = at.session_state["messages"][1]
    assert "[1]" in tin_nhan["content"]
    assert tin_nhan["trich_dan"][0]["nguon"] == "phapluat.pdf"
