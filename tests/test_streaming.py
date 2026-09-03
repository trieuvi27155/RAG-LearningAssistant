"""Test cho chế độ trả lời theo LUỒNG (streaming).

Phần đáng test ở đây không phải "Ollama có trả về từng mảnh không" (đó là việc của thư
viện), mà là hai thứ dễ hỏng của riêng hệ thống này:

1. BÓC PHẦN SUY LUẬN khi nó tới theo từng mảnh. Bản không streaming dùng một regex chạy
   trên chuỗi đã hoàn chỉnh; streaming không có chuỗi hoàn chỉnh nào - thẻ <think> có thể
   bị cắt đôi giữa hai mảnh. Nếu máy trạng thái sai, người dùng nhìn thấy đúng phần suy
   luận nội bộ mà cả hệ thống đang cố giấu.

2. HAI CHẾ ĐỘ KHÔNG ĐƯỢC TRÔI RA KHỎI NHAU. sinh_cau_tra_loi() (dùng cho evaluation) và
   sinh_cau_tra_loi_theo_luong() (dùng cho giao diện) phải cho ra đúng cùng một câu trả
   lời, nếu không thì con số đo được trong báo cáo không nói gì về thứ người dùng thấy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import config
from rag.rag_pipeline import RagPipeline, _LocSuyLuanTheoLuong
from rag.vector_store import VectorStore

SO_CHIEU = 4


class EmbeddingGia:
    def __init__(self, vector_cau_hoi):
        self.vector = np.array([vector_cau_hoi], dtype="float32")

    def encode_cau_hoi(self, texts):
        return self.vector


class OllamaGia:
    """Thay cho ollama.Client: phát lại một kịch bản mảnh đã định sẵn.

    `cac_manh` là list dict giống hệt hình dạng Ollama trả về ở chế độ stream=True.
    """

    def __init__(self, cac_manh):
        self.cac_manh = cac_manh
        self.tham_so_da_dung = None

    def chat(self, **tham_so):
        self.tham_so_da_dung = tham_so
        return iter(self.cac_manh)


def _manh(noi_dung="", suy_luan=""):
    tin_nhan = {"content": noi_dung}
    if suy_luan:
        tin_nhan["thinking"] = suy_luan
    return {"message": tin_nhan}


def _tao_pipeline(cac_manh):
    chunk = {
        "chunk_id": "c1", "nguon": "a.pdf", "trang": 1, "vi_tri": 0,
        "noidung": "Nhà nước có tính giai cấp và quyền lực công cộng đặc biệt.",
    }
    store = VectorStore(dimension=SO_CHIEU)
    store.them(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), [chunk])
    pipeline = RagPipeline(EmbeddingGia([1.0, 0.0, 0.0, 0.0]), store)
    pipeline._ollama_client = OllamaGia(cac_manh)
    return pipeline


@pytest.fixture(autouse=True)
def cau_hinh_on_dinh(monkeypatch):
    # Ngưỡng mặc định tính trên embedding thật; ở đây vector dựng tay nên hạ về 0 để đoạn
    # duy nhất trong store luôn qua được, và tắt BM25 cho thứ hạng do vector quyết định.
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)
    monkeypatch.setattr(config, "TRONG_SO_BM25", 0.0)


# ======================================================================
# Bóc <think> theo từng mảnh
# ======================================================================

def test_the_think_bi_cat_doi_giua_hai_manh_van_bi_boc_sach():
    """Đây là ca hỏng thật của streaming: regex chạy trên chuỗi hoàn chỉnh không cứu được."""
    loc = _LocSuyLuanTheoLuong()
    ra_tra_loi, ra_suy_luan = [], []
    for manh in ["Chào ", "<thi", "nk>đang nghĩ", " tiếp</th", "ink>Đáp án là A."]:
        tl, sl = loc.them(manh)
        ra_tra_loi.append(tl)
        ra_suy_luan.append(sl)
    tl, sl = loc.ket_thuc()
    ra_tra_loi.append(tl)
    ra_suy_luan.append(sl)

    assert "".join(ra_tra_loi) == "Chào Đáp án là A."
    assert "".join(ra_suy_luan) == "đang nghĩ tiếp"


def test_khong_co_the_think_thi_giu_nguyen_toan_bo_noi_dung():
    loc = _LocSuyLuanTheoLuong()
    tl, sl = loc.them("Câu trả lời bình thường, không có thẻ nào.")
    con_lai, _ = loc.ket_thuc()
    assert tl + con_lai == "Câu trả lời bình thường, không có thẻ nào."
    assert sl == ""


def test_dau_the_mo_chua_dong_thi_khong_ro_ri_ra_cau_tra_loi():
    """Luồng đứt giữa chừng khi đang trong <think> - phần dở dang phải về nhánh suy luận,
    tuyệt đối không được rơi vào câu trả lời hiển thị cho người dùng."""
    loc = _LocSuyLuanTheoLuong()
    tl, sl = loc.them("<think>mới nghĩ được nửa chừng")
    tl_cuoi, sl_cuoi = loc.ket_thuc()
    assert tl == "" and tl_cuoi == ""
    assert sl + sl_cuoi == "mới nghĩ được nửa chừng"


def test_suy_luan_o_truong_rieng_cua_ollama_khong_lot_vao_cau_tra_loi():
    pipeline = _tao_pipeline([
        _manh(suy_luan="Okay, let me figure out..."),
        _manh(noi_dung="Nhà nước có tính giai cấp [1]."),
    ])
    ket_qua = pipeline.hoi_dap("Nhà nước có đặc điểm gì?")
    assert ket_qua["cau_tra_loi"] == "Nhà nước có tính giai cấp [1]."


# ======================================================================
# Trình tự sự kiện và tính nhất quán giữa hai chế độ
# ======================================================================

def test_trinh_tu_su_kien_dung_thu_tu_va_ket_thuc_bang_xong():
    pipeline = _tao_pipeline([
        _manh(suy_luan="đang nghĩ"),
        _manh(noi_dung="Đáp "),
        _manh(noi_dung="án [1]."),
    ])
    cac_su_kien = list(pipeline.hoi_dap_theo_luong("Nhà nước có đặc điểm gì?"))

    assert cac_su_kien[0]["loai"] == "truy_xuat_xong"
    assert cac_su_kien[0]["cac_chunk"], "phải báo được đã tìm thấy đoạn nào trước khi gọi LLM"
    assert [sk["loai"] for sk in cac_su_kien[1:-1]] == ["suy_luan", "cau_tra_loi", "cau_tra_loi"]
    assert cac_su_kien[-1]["loai"] == "xong"
    assert cac_su_kien[-1]["ket_qua"]["cau_tra_loi"] == "Đáp án [1]."


def test_hai_che_do_cho_ra_cung_mot_cau_tra_loi():
    """Nếu hai đường này lệch nhau thì con số trong báo cáo không nói gì về thứ người dùng
    nhìn thấy - lỗi âm thầm và rất khó phát hiện, nên chốt lại bằng test."""
    cac_manh = [_manh(noi_dung="Nhà "), _manh(noi_dung="nước [1]."), _manh(suy_luan="x")]
    theo_luong = _tao_pipeline(cac_manh).hoi_dap("Nhà nước có đặc điểm gì?")["cau_tra_loi"]

    pipeline = _tao_pipeline(cac_manh)
    mot_cuc = pipeline.sinh_cau_tra_loi(
        "Nhà nước có đặc điểm gì?", pipeline.truy_xuat("Nhà nước có đặc điểm gì?")
    )
    assert theo_luong == mot_cuc == "Nhà nước [1]."


def test_do_tre_duoc_do_va_tra_kem_ket_qua():
    """Số liệu độ trễ phải đi kèm kết quả, vì đây là thứ giao diện hiển thị và báo cáo
    trích dẫn - tính lại ở tầng ngoài thì mỗi chỗ ra một con số khác nhau."""
    pipeline = _tao_pipeline([_manh(noi_dung="Đáp án [1].")])
    do_tre = pipeline.hoi_dap("Nhà nước có đặc điểm gì?")["do_tre"]

    assert do_tre["truy_xuat"] >= 0
    assert do_tre["chu_dau_tien"] is not None
    assert do_tre["tong"] >= do_tre["chu_dau_tien"] >= do_tre["truy_xuat"]


def test_cau_bi_tu_choi_khong_goi_llm_va_van_co_do_tre():
    """Câu lạc đề bị chặn ở tầng truy xuất - phải trả lời ngay, không mở luồng tới Ollama."""
    pipeline = _tao_pipeline([_manh(noi_dung="KHÔNG ĐƯỢC GỌI")])
    pipeline.vector_store.metadata.clear()
    pipeline.vector_store.index.reset()

    ket_qua = pipeline.hoi_dap("Câu hỏi bất kỳ")
    assert ket_qua["cau_tra_loi"] == config.CAU_TU_CHOI["vi"]
    assert pipeline._ollama_client.tham_so_da_dung is None
    assert ket_qua["do_tre"]["tong"] >= 0


def test_model_chi_sinh_suy_luan_thi_khong_tra_ve_bong_bong_rong():
    """Bong bóng chat trống trơn trông y hệt như hệ thống bị lỗi - thà đưa ra phần suy luận
    dở dang còn hơn không có gì (đúng hành vi dự phòng của bản không streaming trước đây)."""
    pipeline = _tao_pipeline([_manh(suy_luan="Okay, tôi đang nghĩ mãi mà chưa xong thì hết hạn mức")])
    ket_qua = pipeline.hoi_dap("Nhà nước có đặc điểm gì?")
    assert "đang nghĩ mãi" in ket_qua["cau_tra_loi"]


def test_ket_qua_kem_chi_so_bam_nguon():
    """Chỉ số bám nguồn được tính ngay trong pipeline (tất định, mili giây) để giao diện nói
    được với người đọc mức độ bám nguồn của câu trả lời họ đang xem."""
    pipeline = _tao_pipeline([
        _manh(noi_dung="Nhà nước có tính giai cấp và quyền lực công cộng đặc biệt [1].")
    ])
    ket_qua = pipeline.hoi_dap("Nhà nước có đặc điểm gì?")
    assert ket_qua["bam_nguon"] > 0.8, "câu trả lời chép nguyên văn ngữ cảnh phải bám cao"


def test_bam_nguon_thap_khi_cau_tra_loi_khong_lien_quan_ngu_canh():
    pipeline = _tao_pipeline([_manh(noi_dung="Thủ đô nước Pháp là Paris có nhiều bảo tàng.")])
    assert pipeline.hoi_dap("Nhà nước có đặc điểm gì?")["bam_nguon"] < 0.2


def test_cau_tu_choi_co_bam_nguon_bang_khong():
    """Không truy xuất được đoạn nào thì không có ngữ cảnh để mà bám - phải là 0, không phải
    lỗi chia cho 0."""
    pipeline = _tao_pipeline([_manh(noi_dung="KHÔNG ĐƯỢC GỌI")])
    pipeline.vector_store.metadata.clear()
    pipeline.vector_store.index.reset()
    assert pipeline.hoi_dap("Câu hỏi bất kỳ")["bam_nguon"] == 0.0
