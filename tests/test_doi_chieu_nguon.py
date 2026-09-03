"""Test cho việc phát hiện MÂU THUẪN giữa các nguồn.

Trọng tâm là ba tính chất mà thiết kế dựa vào, chứ không phải chất lượng phán đoán của model
(thứ đó đo bằng evaluation/kiem_dinh_doi_chieu.py trên bộ ca đã biết trước đáp án):

1. CHI PHÍ CÓ TRẦN. Tầng lọc tất định phải cắt được số cặp; nếu nó để lọt, mỗi câu trả lời
   sẽ gánh thêm hàng chục lượt gọi LLM. Test đếm thẳng số lần model bị gọi.
2. NGHIÊNG VỀ PHÍA IM LẶNG. Mọi đường không chắc chắn - model bất đồng giữa các lần chấm,
   model hỏng, điểm ngoài thang - đều phải cho ra danh sách RỖNG, không phải một cảnh báo.
3. KHÔNG BAO GIỜ NÉM LỖI. Đây là một lớp thông tin thêm, chạy sau khi câu trả lời đã sinh
   xong; nó hỏng thì người dùng vẫn phải nhận được câu trả lời.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pytest

import config
from rag.doi_chieu_nguon import (
    _tap_so,
    cac_cap_dang_ngo,
    co_dau_hieu_bat_dong,
    tim_mau_thuan,
)


class ClientGia:
    """Thay ollama.Client: trả lần lượt các phán quyết định sẵn, và ĐẾM số lần bị gọi."""

    def __init__(self, cac_phan_hoi=None, loi=None):
        self.cac_phan_hoi = list(cac_phan_hoi or [])
        self.loi = loi
        self.so_lan_goi = 0

    def chat(self, **kwargs):
        self.so_lan_goi += 1
        if self.loi:
            raise self.loi
        phan_hoi = self.cac_phan_hoi.pop(0) if self.cac_phan_hoi else self.cac_phan_hoi
        return {"message": {"content": json.dumps(phan_hoi)}}


# Thứ tự key ở đây theo đúng _SCHEMA_MAU_THUAN: phân tích TRƯỚC, phán quyết SAU. Không phải
# chi tiết trang trí - đó là cách sửa cho đúng lỗi §5.56 (model bị ép chốt phán quyết trước
# khi viết được chữ lập luận nào), và fixture phải phản ánh đúng hình dạng thật.
def _co(muc_do=0.9, mo_ta="Một bên nói năm, bên kia nói bốn"):
    return {"phan_tich": mo_ta, "muc_do": muc_do, "co_mau_thuan": True}


def _khong():
    return {"phan_tich": "Hai đoạn bổ sung cho nhau.", "muc_do": 0.0, "co_mau_thuan": False}


def _doan(nguon, trang, noidung):
    return {"nguon": nguon, "trang": trang, "noidung": noidung, "doan_khop": noidung,
            "diem_similarity": 0.9}


# Hai đoạn nói ngược nhau về CÙNG một chuyện - ca chính mà tính năng này sinh ra để bắt.
DOAN_NAM = _doan("giaotrinh.pdf", 12, "Nhà nước có năm đặc điểm cơ bản.")
DOAN_BON = _doan("slide.pptx", 3, "Nhà nước có bốn đặc điểm cơ bản.")

# Ma trận vector cho 2 đoạn giống chủ đề (cosine ~0.99).
VECTOR_GIONG = np.array([[1.0, 0.0], [0.99, 0.141]], dtype="float32")


@pytest.fixture(autouse=True)
def cau_hinh_on_dinh(monkeypatch):
    """Khoá cấu hình về giá trị mặc định, để test không phụ thuộc .env của máy đang chạy."""
    monkeypatch.setattr(config, "BAT_DOI_CHIEU_NGUON", True)
    monkeypatch.setattr(config, "SO_CAP_DOI_CHIEU_TOI_DA", 3)
    monkeypatch.setattr(config, "SO_LAN_CHAM_MAU_THUAN", 2)
    monkeypatch.setattr(config, "NGUONG_MAU_THUAN", 0.6)
    monkeypatch.setattr(config, "NGUONG_COSINE_DOI_CHIEU", 0.88)


# ======================================================================
# Tầng 1: lọc tất định
# ======================================================================

def test_chuan_hoa_so_de_khong_bao_dong_gia():
    """Cùng một con số hay được viết khác nhau giữa hai tài liệu. Không chuẩn hoá thì mọi
    cặp đều "khác số" và tầng lọc mất sạch tác dụng - nó sẽ đẩy hết cặp lên tầng LLM."""
    assert _tap_so("phí là 1.000 đồng") == _tap_so("phí là 1000 đồng")
    assert _tap_so("tỉ lệ 2,5%") == _tap_so("tỉ lệ 2.5%")


def test_khac_so_la_dau_hieu_bat_dong():
    assert co_dau_hieu_bat_dong("có năm đặc điểm, tổng 5 mục", "có bốn đặc điểm, tổng 4 mục")


def test_lech_phu_dinh_la_dau_hieu_bat_dong():
    assert co_dau_hieu_bat_dong(
        "Doanh nghiệp tư nhân có tư cách pháp nhân.",
        "Doanh nghiệp tư nhân không có tư cách pháp nhân.",
    )


def test_hai_doan_noi_cung_mot_y_khong_co_dau_hieu():
    assert not co_dau_hieu_bat_dong(
        "Nhà nước có quyền lực công cộng đặc biệt.",
        "Quyền lực công cộng đặc biệt là một đặc điểm của nhà nước.",
    )


def test_cap_cung_mot_nguon_bi_loai():
    """Hai đoạn trong cùng một file "mâu thuẫn" thì hầu hết là do đọc hỏng (bảng bị cắt, cột
    bị nối sai) - đó là bài toán của khâu đọc tài liệu, không phải của module này."""
    cung_nguon = [
        _doan("a.pdf", 1, "Nhà nước có năm đặc điểm."),
        _doan("a.pdf", 9, "Nhà nước có bốn đặc điểm."),
    ]
    assert cac_cap_dang_ngo(cung_nguon, VECTOR_GIONG) == []


def test_cap_khac_chu_de_bi_loai_theo_cosine():
    """Hai đoạn phải nói về CÙNG một chuyện thì mới mâu thuẫn được."""
    vector_khac_han = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    assert cac_cap_dang_ngo([DOAN_NAM, DOAN_BON], vector_khac_han) == []


def test_cap_dung_chu_de_va_khac_so_thi_qua_duoc_tang_1():
    assert cac_cap_dang_ngo([DOAN_NAM, DOAN_BON], VECTOR_GIONG) == [(0, 1)]


def test_tran_so_cap_chan_bung_no_chi_phi(monkeypatch):
    """Không có trần, 6 đoạn từ 6 nguồn khác nhau sinh ra 15 cặp = 15 lượt gọi LLM cộng vào
    sau MỖI câu trả lời."""
    monkeypatch.setattr(config, "SO_CAP_DOI_CHIEU_TOI_DA", 2)
    cac_doan = [_doan(f"f{i}.pdf", 1, f"Có {i} đặc điểm cơ bản.") for i in range(6)]
    vector = np.ones((6, 2), dtype="float32") / np.sqrt(2)
    assert len(cac_cap_dang_ngo(cac_doan, vector)) == 2


# ======================================================================
# Tầng 2 + hợp nhất: nghiêng về phía im lặng
# ======================================================================

def test_phat_hien_duoc_mau_thuan_that():
    client = ClientGia([_co(0.9), _co(0.85)])
    ket_qua = tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=None, client=client)
    assert len(ket_qua) == 1
    assert {ket_qua[0]["nguon_a"], ket_qua[0]["nguon_b"]} == {"giaotrinh.pdf", "slide.pptx"}
    # Lấy mức độ THẤP NHẤT giữa các lần chấm, không lấy trung bình: khi các lần không thống
    # nhất thì phải nghiêng về phía dè dặt hơn.
    assert ket_qua[0]["muc_do"] == 0.85


def test_cac_lan_cham_bat_dong_thi_khong_bao_gi():
    """§5.43 đã đo: model 4B lật phán quyết 1/8 lần dù temperature=0. Báo động giả ở đây làm
    người dùng mất niềm tin vào chính tài liệu của họ - tệ hơn hẳn việc bỏ sót."""
    client = ClientGia([_co(0.9), _khong()])
    assert tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=None, client=client) == []


def test_thoat_som_khi_lan_cham_dau_tien_noi_khong():
    """Phần lớn cặp bị loại ngay ở lần chấm đầu, nên chi phí thực tế gần với MỘT lần chấm
    chứ không phải SO_LAN_CHAM_MAU_THUAN lần."""
    client = ClientGia([_khong(), _co(0.9)])
    assert tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=None, client=client) == []
    assert client.so_lan_goi == 1


def test_muc_do_duoi_nguong_bi_bo_qua():
    client = ClientGia([_co(0.5), _co(0.5)])
    assert tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=None, client=client) == []


def test_diem_ngoai_thang_bi_loai_chu_khong_bi_kep():
    """Đúng lỗi §5.48: giám khảo đổi sang thang phần trăm. Kẹp 100 -> 1.0 là đoán ý model
    rồi ghi kết quả đoán ra màn hình cho người dùng đọc."""
    client = ClientGia([_co(100.0), _co(0.9)])
    assert tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=None, client=client) == []


def test_chi_mot_nguon_thi_khong_ton_luot_goi_nao():
    """Không có gì để đối chiếu chéo - phải thoát TRƯỚC khi encode, vì encode cả tập đoạn là
    phần đắt nhất của tầng lọc."""
    mot_nguon = [_doan("a.pdf", 1, "Có năm đặc điểm."), _doan("a.pdf", 2, "Có bốn đặc điểm.")]
    client = ClientGia([_co()])
    assert tim_mau_thuan(mot_nguon, embedding_service=None, client=client) == []
    assert client.so_lan_goi == 0


def test_khong_co_dau_hieu_bat_dong_thi_khong_goi_llm():
    """Đây là chỗ tiết kiệm chính: đại đa số lượt hỏi không có cặp nào qua nổi tầng 1."""
    hoa_thuan = [
        _doan("a.pdf", 1, "Nhà nước có quyền lực công cộng đặc biệt."),
        _doan("b.pdf", 2, "Quyền lực công cộng đặc biệt là đặc điểm của nhà nước."),
    ]
    client = ClientGia([_co()])
    assert tim_mau_thuan(hoa_thuan, embedding_service=None, client=client) == []
    assert client.so_lan_goi == 0


def test_tat_cau_hinh_thi_khong_chay_gi(monkeypatch):
    monkeypatch.setattr(config, "BAT_DOI_CHIEU_NGUON", False)
    client = ClientGia([_co()])
    assert tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=None, client=client) == []
    assert client.so_lan_goi == 0


# ======================================================================
# Không bao giờ ném lỗi ra ngoài
# ======================================================================

def test_ollama_chet_khong_lam_hong_luot_hoi():
    client = ClientGia(loi=ConnectionError("chưa bật Ollama"))
    assert tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=None, client=client) == []


def test_model_tra_json_hong_khong_lam_hong_luot_hoi():
    class ClientRac:
        so_lan_goi = 0

        def chat(self, **kwargs):
            return {"message": {"content": "đây không phải JSON"}}

    assert tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=None, client=ClientRac()) == []


def test_embedding_hong_thi_bo_dieu_kien_cung_chu_de_chu_khong_dung_han():
    """Suy giảm êm: mất phép lọc theo chủ đề thì chỉ tốn thêm vài lượt chấm, còn dừng hẳn
    thì mất luôn tính năng."""
    class EmbeddingHong:
        def encode_tai_lieu(self, texts):
            raise RuntimeError("model chưa nạp được")

    client = ClientGia([_co(0.9), _co(0.9)])
    ket_qua = tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=EmbeddingHong(), client=client)
    assert len(ket_qua) == 1


def test_so_viet_bang_chu_duoc_quy_ve_cung_dang_voi_chu_so():
    """Nếu không quy về cùng dạng thì "5 đặc điểm" và "năm đặc điểm" - hai cách viết CÙNG
    một con số - lại bị coi là mâu thuẫn."""
    assert not co_dau_hieu_bat_dong("Có năm đặc điểm cơ bản.", "Có 5 đặc điểm cơ bản.")


def test_so_viet_bang_chu_khac_nhau_van_bat_duoc():
    """Ca mẫu của cả đồ án (§5.22) không chứa lấy một chữ số nào."""
    assert co_dau_hieu_bat_dong("Nhà nước có năm đặc điểm.", "Nhà nước có bốn đặc điểm.")


def test_mot_khong_duoc_tinh_la_so():
    """"một" đóng vai mạo từ trong vô số câu tiếng Việt. Tính nó là số thì gần như cặp nào
    cũng "khác số", và tầng lọc mất tác dụng cắt chi phí."""
    assert not co_dau_hieu_bat_dong(
        "Đây là một cách tiếp cận phổ biến trong ngành.",
        "Cách tiếp cận này phổ biến trong ngành hiện nay.",
    )


def test_giai_thich_lay_tu_phan_tich_cua_model():
    """Chuỗi hiện cho người dùng phải là LẬP LUẬN của model, không phải một nhãn chung chung.

    Khoá lại luôn tên field: schema đặt `phan_tich` TRƯỚC `co_mau_thuan` để model viết lập
    luận rồi mới chốt phán quyết (§5.56 / §5.59). Đổi tên field mà quên chỗ đọc thì cảnh báo
    hiện ra vẫn đúng/sai bình thường, chỉ có phần giải thích lặng lẽ rỗng."""
    client = ClientGia([_co(0.9, "Đoạn A nói năm, đoạn B nói bốn."),
                        _co(0.9, "Đoạn A nói năm, đoạn B nói bốn.")])
    ket_qua = tim_mau_thuan([DOAN_NAM, DOAN_BON], embedding_service=None, client=client)
    assert ket_qua[0]["noi_dung_xung_dot"] == "Đoạn A nói năm, đoạn B nói bốn."
