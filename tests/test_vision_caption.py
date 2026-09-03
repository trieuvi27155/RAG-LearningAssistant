"""Test cho chú thích ảnh bằng model vision.

Phần LOGIC (giảm cấp khi thiếu model, nuốt lỗi, nối mô tả vào chú thích sẵn có) được test
bằng client giả - không cần Ollama chạy, không cần model, chạy trong mili giây. Phần gọi
model THẬT tách riêng và tự bỏ qua nếu model chưa được pull, vì nó kiểm tra hành vi của
model chứ không phải của code ta viết.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from rag.vision_caption import (
    bo_sung_chu_thich_vision,
    chu_thich_anh,
    mo_hinh_vision_co_san,
    ocr_trang_pdf,
    ten_model_khop,
    trang_can_ocr,
)


class ModelGia:
    def __init__(self, ten):
        self.model = ten


class PhanHoiListGia:
    def __init__(self, ten_cac_model):
        self.models = [ModelGia(t) for t in ten_cac_model]


class ClientGia:
    """Thay cho ollama.Client - không gọi mạng."""

    def __init__(self, ten_cac_model=("qwen2.5vl:3b",), noi_dung="Sơ đồ có 3 ô: A, B, C.", loi=None):
        self._ten_cac_model = ten_cac_model
        self._noi_dung = noi_dung
        self._loi = loi
        self.so_lan_goi = 0

    def list(self):
        if self._loi == "list":
            raise ConnectionError("Ollama chưa chạy")
        return PhanHoiListGia(self._ten_cac_model)

    def chat(self, **kwargs):
        self.so_lan_goi += 1
        if self._loi == "chat":
            raise RuntimeError("model lỗi")
        return {"message": {"content": self._noi_dung}}


@pytest.fixture(autouse=True)
def bat_chu_thich(monkeypatch):
    monkeypatch.setattr(config, "BAT_CHU_THICH_ANH", True)
    monkeypatch.setattr(config, "VISION_MODEL_NAME", "qwen2.5vl:3b")


def _ban_ghi_anh(chu_thich="[HÌNH] Hình 3: Sơ đồ bộ máy nhà nước"):
    return {
        "nguon": "a.pdf",
        "trang": 5,
        "noidung": chu_thich,
        "loai_noi_dung": "anh",
        "duong_dan_anh": "data/images/a_5_1.png",
    }


# ======================================================================
# Nhận diện model có sẵn
# ======================================================================

def test_khop_ten_model_khong_can_dung_tag():
    """Ollama trả "qwen2.5vl:3b" nhưng người dùng hay cấu hình "qwen2.5vl" - bắt họ nhớ
    đúng tag chỉ tạo ra một cách hỏng vô nghĩa."""
    assert ten_model_khop("qwen2.5vl:3b", "qwen2.5vl")
    assert ten_model_khop("qwen2.5vl:3b", "qwen2.5vl:3b")
    assert not ten_model_khop("qwen3:4b", "qwen2.5vl")


def test_bao_thieu_model_khi_chua_pull():
    assert not mo_hinh_vision_co_san(ClientGia(ten_cac_model=("qwen3:4b",)))
    assert mo_hinh_vision_co_san(ClientGia(ten_cac_model=("qwen2.5vl:3b",)))


def test_ollama_khong_chay_thi_coi_nhu_khong_co_model():
    """Không được để lỗi kết nối bung ra giữa lần build index."""
    assert not mo_hinh_vision_co_san(ClientGia(loi="list"))


# ======================================================================
# Giảm cấp thay vì hỏng
# ======================================================================

def test_thieu_model_thi_bo_qua_khong_lam_hong_build(caplog):
    """Bật nhầm tuỳ chọn không được phép làm sập cả lần build index - bản ghi ảnh phải
    giữ nguyên chú thích lân cận để vẫn tìm được."""
    cac_anh = [_ban_ghi_anh()]
    goc = cac_anh[0]["noidung"]

    so_luong = bo_sung_chu_thich_vision(cac_anh, client=ClientGia(ten_cac_model=("qwen3:4b",)))

    assert so_luong == 0
    assert cac_anh[0]["noidung"] == goc
    assert "chưa được pull" in caplog.text


def test_mot_anh_loi_khong_lam_hong_ca_lo():
    cac_anh = [_ban_ghi_anh(), _ban_ghi_anh()]
    assert bo_sung_chu_thich_vision(cac_anh, client=ClientGia(loi="chat")) == 0
    assert all(a["noidung"].startswith("[HÌNH]") for a in cac_anh)


def test_tat_tuy_chon_thi_khong_goi_model():
    """Tài liệu thuần văn bản không được trả bất kỳ chi phí nào cho tính năng này."""
    client = ClientGia()
    config.BAT_CHU_THICH_ANH = False
    try:
        assert bo_sung_chu_thich_vision([_ban_ghi_anh()], client=client) == 0
    finally:
        config.BAT_CHU_THICH_ANH = True
    assert client.so_lan_goi == 0


# ======================================================================
# Nối mô tả vào chú thích sẵn có
# ======================================================================

def test_mo_ta_duoc_noi_them_chu_khong_thay_the_chu_thich():
    """Hai nguồn bổ khuyết nhau: chú thích cho biết tài liệu GỌI hình đó là gì (từ khoá
    người dùng sẽ hỏi), model vision cho biết BÊN TRONG hình có gì."""
    cac_anh = [_ban_ghi_anh()]
    bo_sung_chu_thich_vision(cac_anh, client=ClientGia(noi_dung="Sơ đồ có 3 ô: A, B, C."))

    noi_dung = cac_anh[0]["noidung"]
    assert "Hình 3: Sơ đồ bộ máy nhà nước" in noi_dung  # chú thích gốc còn nguyên
    assert "Sơ đồ có 3 ô: A, B, C." in noi_dung          # mô tả mới được thêm
    assert cac_anh[0]["co_chu_thich_vision"] is True


def test_mo_ta_rong_thi_khong_danh_dau_da_chu_thich():
    cac_anh = [_ban_ghi_anh()]
    assert bo_sung_chu_thich_vision(cac_anh, client=ClientGia(noi_dung="   ")) == 0
    assert "co_chu_thich_vision" not in cac_anh[0]


def test_danh_sach_rong_khong_goi_model():
    client = ClientGia()
    assert bo_sung_chu_thich_vision([], client=client) == 0
    assert client.so_lan_goi == 0


# ======================================================================
# Model THẬT (tự bỏ qua nếu chưa pull)
# ======================================================================

@pytest.mark.slow
def test_model_vision_that_doc_duoc_chu_trong_anh(tmp_path):
    """Kiểm tra model thật đọc được chữ trong hình - đây là điều kiện tiên quyết để tính
    năng có ích, và là thứ chỉ model mới trả lời được (không mock thay được)."""
    ollama = pytest.importorskip("ollama")
    from PIL import Image, ImageDraw

    client = ollama.Client(host=config.OLLAMA_HOST)
    if not mo_hinh_vision_co_san(client):
        pytest.skip(f"Chưa pull model {config.VISION_MODEL_NAME}")

    duong_dan = tmp_path / "so_do.png"
    anh = Image.new("RGB", (400, 160), "white")
    ve = ImageDraw.Draw(anh)
    ve.rectangle([20, 40, 180, 120], outline="black", width=3)
    ve.text((60, 75), "QUOC HOI", fill="black")
    ve.rectangle([220, 40, 380, 120], outline="black", width=3)
    ve.text((250, 75), "CHINH PHU", fill="black")
    anh.save(duong_dan)

    mo_ta = chu_thich_anh(client, str(duong_dan)).upper()
    assert "QUOC HOI" in mo_ta or "CHINH PHU" in mo_ta, f"model không đọc được chữ: {mo_ta}"


# ======================================================================
# OCR dự phòng cho trang PDF đọc hỏng
# ======================================================================

def test_nhan_dien_trang_font_hong():
    """PDF nhúng font không kèm bảng ánh xạ ToUnicode (hay gặp ở font toán học) trả về mã
    (cid:NN) thay vì chữ. Đo trên giáo trình Bishop: 355/758 trang (47%) bị vậy."""
    text_hong = "142 3.LINEARMODELS (cid:22) (cid:23) (cid:2)N 0= tn (cid:10) (cid:11) w ML"
    assert trang_can_ocr(text_hong, so_anh_trong_trang=0)


def test_trang_binh_thuong_khong_can_ocr():
    """Trang đọc tốt thì KHÔNG được OCR - mỗi trang OCR tốn ~14 giây, chạy thừa trên tài
    liệu lành lặn là lãng phí lớn (sách 700 trang = hàng giờ vô ích)."""
    text_tot = " ".join(["Đây là một câu văn bình thường đọc được hoàn chỉnh."] * 20)
    assert not trang_can_ocr(text_tot, so_anh_trong_trang=0)
    assert not trang_can_ocr(text_tot, so_anh_trong_trang=3)


def test_vai_ma_cid_le_te_khong_kich_hoat_ocr():
    """Trang dài đọc tốt mà lẫn vài ký tự đặc biệt không nên bị coi là hỏng."""
    text = " ".join(["Nội dung bình thường của trang tài liệu này."] * 40) + " (cid:3)"
    assert not trang_can_ocr(text, so_anh_trong_trang=0)


def test_nhan_dien_trang_scan():
    """Trang gần như không có chữ nhưng có ảnh -> nhiều khả năng là trang scan."""
    assert trang_can_ocr("Hình 3.1", so_anh_trong_trang=1)
    # Không có ảnh thì là trang trống thật, OCR cũng không cứu được gì.
    assert not trang_can_ocr("Hình 3.1", so_anh_trong_trang=0)


def test_ocr_that_bai_tra_ve_rong_khong_lam_sap_build():
    assert ocr_trang_pdf(ClientGia(loi="chat"), "khong-ton-tai.png") == ""


def test_prompt_ocr_cam_dich():
    """Bản đầu dùng prompt tiếng Việt, model liền DỊCH cả trang sách tiếng Anh sang tiếng
    Việt và dịch sai bét ("LINEAR MODELS FOR REGRESSION" -> "LINHỆ MÔIẾN TRÊN REGRESSION"),
    tức phá hỏng nội dung thay vì cứu nó."""
    from rag.vision_caption import PROMPT_OCR_TRANG

    assert "do not translate" in PROMPT_OCR_TRANG.lower()
    assert "original language" in PROMPT_OCR_TRANG.lower()
