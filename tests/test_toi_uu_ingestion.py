"""Test cho đợt tối ưu luồng Ingestion: đọc một-lượt, cache theo content hash, lọc ảnh.

Bối cảnh: luồng build cũ quét cùng một tài liệu nhiều lần. Mỗi trang PDF có thể bị đọc tới
5 lần (dò x_tolerance), mỗi bảng bị trích 2 lần, và toàn bộ PDF được duyệt thêm một lượt nữa
chỉ để trích ảnh. Trên giáo trình Bishop 758 trang, đó là gần một phút cho MỘT tài liệu, lặp
lại đầy đủ mỗi lần bấm "Đọc tài liệu" - kể cả khi người dùng chỉ vừa thêm một file khác.

Các test dưới đây khoá lại từng cơ chế đã sửa, và quan trọng hơn: khoá lại RANH GIỚI của
chúng - chỗ mà tối ưu KHÔNG được phép đổi kết quả. Một tối ưu ingestion làm nội dung index
xấu đi là một tối ưu đã thất bại, dù nó nhanh tới đâu.

Không gọi model thật ở đâu cả: OCR và vision đều thay bằng hàm giả, vì thứ cần kiểm là luồng
quyết định (đọc lại mấy lần, cache trúng hay trượt, ảnh nào bị loại), không phải chất lượng
model.
"""

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import config
from rag import bo_nho_dem
from rag.bo_nho_dem import KhoDem, KhoVectorDem, bam_bytes, encode_co_cache
from rag.document_loader import (
    HieuChinhXTolerance,
    _trich_text_thich_ung,
    doc_tai_lieu_co_cache,
)
from rag.image_extractor import (
    KICH_THUOC_ANH_TOI_THIEU,
    MOC_ANH,
    loc_anh_lap_lai,
    ly_do_loai_anh,
)
from rag.vision_caption import bo_sung_chu_thich_vision


@pytest.fixture
def cache_tam(tmp_path, monkeypatch):
    """Trỏ toàn bộ cache sang thư mục tạm - test không được đụng vào data/cache thật."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(bo_nho_dem, "kho_tai_lieu", KhoDem("tai_lieu", ".json"))
    monkeypatch.setattr(bo_nho_dem, "kho_ocr", KhoDem("ocr", ".txt"))
    monkeypatch.setattr(bo_nho_dem, "kho_vision", KhoDem("vision", ".txt"))
    monkeypatch.setattr(config, "BAT_CACHE_INGESTION", True)
    return tmp_path


class TrangGia:
    """Trang PDF giả: trả text khác nhau tuỳ x_tolerance, và ĐẾM số lần bị đọc.

    Bộ đếm mới là thứ đang được kiểm: số lần extract_text() bị gọi chính là chi phí đọc lại
    mà đợt tối ưu này nhắm vào.
    """

    def __init__(self, theo_tolerance: dict, mac_dinh: str):
        self.theo_tolerance = theo_tolerance
        self.mac_dinh = mac_dinh
        self.so_lan_doc = 0

    def extract_text(self, x_tolerance=None, **_):
        self.so_lan_doc += 1
        return self.theo_tolerance.get(x_tolerance, self.mac_dinh)


# Một câu dài viết liền, đủ để _ty_le_dinh_chu() vượt ngưỡng đọc lại (cụm >= 25 ký tự).
_DINH = " ".join(["thequickbrownfoxjumpsoverthelazydogagainandagain"] * 6)
# Cùng nội dung nhưng đã tách từ - độ dính về 0.
_SACH = " ".join(["the quick brown fox jumps over the lazy dog again and again"] * 6)
# Bản "đỡ dính": một nửa số cụm đã tách, nửa còn lại vẫn liền -> dính khoảng 50%, tức đã
# GIẢM so với bản gốc nhưng CHƯA sạch.
_DO_DINH = " ".join(
    ["the quick brown fox jumps over the lazy dog again and again",
     "thequickbrownfoxjumpsoverthelazydogagainandagain"] * 3
)


# ======================================================================
# 1. Dò x_tolerance: dừng sớm nhưng KHÔNG được dừng khi chữ còn dính
# ======================================================================

def test_dung_som_chi_khi_text_da_that_su_sach():
    """Hồi quy đã gặp thật trên PaperQA.pdf.

    Bản đầu của phép dừng sớm dùng chung ngưỡng với TY_LE_DINH_CHU_DE_DOC_LAI (0.10), nên nó
    chấp nhận ngay mức x_tolerance đầu tiên đưa độ dính xuống dưới 10% - bỏ qua mức tốt hơn
    nằm ngay sau đó. Kết quả là những dòng như "RAGmodelsretrievetextfromacorpus" đi thẳng
    vào index. Test này khoá lại: chỉ được dừng khi đã SẠCH.
    """
    trang = TrangGia({2.0: _DO_DINH, 1.5: _SACH}, _DINH)
    ket_qua = _trich_text_thich_ung(trang, "thu.pdf", 1)
    assert ket_qua == _SACH, "phải đọc tiếp tới mức làm sạch được, không dừng ở mức 'đỡ dính'"


def test_dung_ngay_khi_muc_dau_tien_da_sach():
    """Đã sạch thì thử tiếp chỉ tốn thêm lượt đọc trang mà không cứu thêm chữ nào."""
    trang = TrangGia({2.0: _SACH}, _DINH)
    assert _trich_text_thich_ung(trang, "thu.pdf", 1) == _SACH
    # 1 lượt đọc gốc + 1 lượt với x_tolerance=2.0, rồi dừng. Bản cũ đọc 1 + 4 = 5 lượt.
    assert trang.so_lan_doc == 2


def test_khong_muc_nao_dat_thi_giu_nguyen_ban_goc():
    """Trang thật sự viết liền: đừng đánh đổi bản gốc lấy một bản cũng hỏng."""
    trang = TrangGia({}, _DINH)
    assert _trich_text_thich_ung(trang, "thu.pdf", 1) == _DINH


# ======================================================================
# 2. Hiệu chỉnh x_tolerance theo TÀI LIỆU
# ======================================================================

def test_hieu_chinh_thu_muc_da_dung_duoc_truoc_tien():
    hieu_chinh = HieuChinhXTolerance()
    assert hieu_chinh.thu_tu_uu_tien() == list(config.CAC_X_TOLERANCE_THU)

    x_cuoi = config.CAC_X_TOLERANCE_THU[-1]
    hieu_chinh.ghi_nhan(x_cuoi)
    thu_tu = hieu_chinh.thu_tu_uu_tien()
    assert thu_tu[0] == x_cuoi, "mức đã dùng được phải được thử TRƯỚC"
    assert sorted(thu_tu) == sorted(config.CAC_X_TOLERANCE_THU), "không được đánh rơi mức nào"


def test_hieu_chinh_chi_duoc_coi_la_on_dinh_sau_du_so_trang():
    hieu_chinh = HieuChinhXTolerance()
    for _ in range(config.SO_TRANG_HIEU_CHINH_X_TOLERANCE - 1):
        hieu_chinh.ghi_nhan(1.0)
    assert not hieu_chinh.da_hieu_chinh
    hieu_chinh.ghi_nhan(1.0)
    assert hieu_chinh.da_hieu_chinh

    # Một trang cần mức KHÁC -> bộ đếm phải reset, không được coi là đã ổn định nữa. Đây là
    # chốt cho tài liệu trộn nhiều font (phụ lục scan, chương chèn từ nguồn khác).
    hieu_chinh.ghi_nhan(0.7)
    assert not hieu_chinh.da_hieu_chinh


def test_hieu_chinh_giam_han_so_lan_doc_lai_tu_trang_thu_hai():
    """Đây là khoản tiết kiệm chính trên tài liệu dính chữ toàn tập như Bishop."""
    hieu_chinh = HieuChinhXTolerance()
    x_tot = config.CAC_X_TOLERANCE_THU[-1]

    trang_dau = TrangGia({x_tot: _SACH}, _DINH)
    _trich_text_thich_ung(trang_dau, "sach.pdf", 1, hieu_chinh)
    so_lan_trang_dau = trang_dau.so_lan_doc

    trang_sau = TrangGia({x_tot: _SACH}, _DINH)
    _trich_text_thich_ung(trang_sau, "sach.pdf", 2, hieu_chinh)

    assert trang_sau.so_lan_doc == 2, "trang sau chỉ cần đọc gốc + đúng mức đã hiệu chỉnh"
    assert trang_sau.so_lan_doc < so_lan_trang_dau


def test_hieu_chinh_van_do_lai_khi_muc_da_nho_khong_dung_duoc():
    """Giá trị đã nhớ chỉ được thử TRƯỚC, không được tin tưởng vô điều kiện."""
    hieu_chinh = HieuChinhXTolerance()
    hieu_chinh.ghi_nhan(config.CAC_X_TOLERANCE_THU[0])
    # Mức đã nhớ (phần tử đầu) không cứu được trang này; mức cuối mới cứu được.
    trang = TrangGia({config.CAC_X_TOLERANCE_THU[-1]: _SACH}, _DINH)
    assert _trich_text_thich_ung(trang, "sach.pdf", 9, hieu_chinh) == _SACH


# ======================================================================
# 3. Cache tài liệu theo content hash
# ======================================================================

def _tai_lieu_gia(monkeypatch, ban_ghi, bo_dem):
    """Thay hàm đọc thật bằng hàm giả có đếm số lần được gọi."""
    def gia(duong_dan):
        bo_dem.append(duong_dan.name)
        return [dict(m) for m in ban_ghi]

    monkeypatch.setattr("rag.document_loader.doc_tai_lieu_hoan_chinh", gia)


def test_cache_tai_lieu_khong_doc_lai_khi_noi_dung_khong_doi(cache_tam, monkeypatch):
    f = cache_tam / "bai.pdf"
    f.write_bytes(b"%PDF-1.4 noi dung goc")
    bo_dem = []
    _tai_lieu_gia(monkeypatch, [{"nguon": "bai.pdf", "trang": 1, "noidung": "xin chao"}], bo_dem)

    lan_1 = doc_tai_lieu_co_cache(f)
    lan_2 = doc_tai_lieu_co_cache(f)

    assert lan_1 == lan_2, "cache phải trả về đúng nội dung đã đọc, không được biến dạng"
    assert len(bo_dem) == 1, "lần thứ hai phải lấy từ cache, không đọc lại tài liệu"


def test_cache_truot_khi_noi_dung_file_thay_doi(cache_tam, monkeypatch):
    """Khoá cache là NỘI DUNG - sửa file thì bắt buộc phải đọc lại."""
    f = cache_tam / "bai.pdf"
    f.write_bytes(b"%PDF-1.4 ban dau")
    bo_dem = []
    _tai_lieu_gia(monkeypatch, [{"nguon": "bai.pdf", "trang": 1, "noidung": "x"}], bo_dem)

    doc_tai_lieu_co_cache(f)
    f.write_bytes(b"%PDF-1.4 da sua noi dung")
    doc_tai_lieu_co_cache(f)

    assert len(bo_dem) == 2


def test_cache_truot_khi_doi_cau_hinh_doc_tai_lieu(cache_tam, monkeypatch):
    """Đổi tuỳ chọn ăn vào kết quả đọc mà vẫn trả cache cũ = sai âm thầm."""
    f = cache_tam / "bai.pdf"
    f.write_bytes(b"%PDF-1.4 noi dung")
    bo_dem = []
    _tai_lieu_gia(monkeypatch, [{"nguon": "bai.pdf", "trang": 1, "noidung": "x"}], bo_dem)

    doc_tai_lieu_co_cache(f)
    monkeypatch.setattr(config, "BAT_OCR_DU_PHONG", not config.BAT_OCR_DU_PHONG)
    doc_tai_lieu_co_cache(f)

    assert len(bo_dem) == 2


def test_cache_truot_khi_file_anh_da_bi_xoa(cache_tam, monkeypatch):
    """Nội dung nằm trong cache nhưng ảnh nằm ở data/images - hai chỗ có thể lệch nhau.

    Trả về bản ghi trỏ vào ảnh không còn tồn tại sẽ hỏng đúng ở chỗ người dùng nhìn thấy:
    trích dẫn có hình nhưng hình không mở được.
    """
    f = cache_tam / "bai.pdf"
    f.write_bytes(b"%PDF-1.4 noi dung")
    anh = cache_tam / "hinh1.png"
    anh.write_bytes(b"PNG-gia")
    bo_dem = []
    _tai_lieu_gia(
        monkeypatch,
        [{
            "nguon": "bai.pdf", "trang": 1, "noidung": f"{MOC_ANH} so do",
            "loai_noi_dung": "anh", "duong_dan_anh": str(anh),
        }],
        bo_dem,
    )

    doc_tai_lieu_co_cache(f)
    anh.unlink()
    doc_tai_lieu_co_cache(f)

    assert len(bo_dem) == 2


def test_tat_cache_thi_luon_doc_lai(cache_tam, monkeypatch):
    """Đường lui để đo chi phí THẬT của một lần build từ đầu."""
    monkeypatch.setattr(config, "BAT_CACHE_INGESTION", False)
    f = cache_tam / "bai.pdf"
    f.write_bytes(b"%PDF-1.4 noi dung")
    bo_dem = []
    _tai_lieu_gia(monkeypatch, [{"nguon": "bai.pdf", "trang": 1, "noidung": "x"}], bo_dem)

    doc_tai_lieu_co_cache(f)
    doc_tai_lieu_co_cache(f)
    assert len(bo_dem) == 2


# ======================================================================
# 4. Lọc ảnh trước khi render / gọi model vision
# ======================================================================

_DIEN_TICH_TRANG = 595.0 * 841.0        # A4
_DIEN_TICH_TRANG_LON = 1920.0 * 1080.0  # trang khổ lớn / poster


@pytest.mark.parametrize(
    "mo_ta, rong, cao, bi_loai",
    [
        ("icon nhỏ", 40.0, 40.0, True),
        ("đường kẻ ngang dưới tiêu đề slide", 900.0, 6.0, True),
        ("thanh màu dọc bên lề", 8.0, 700.0, True),
        ("biểu đồ thật giữa trang", 400.0, 300.0, False),
        ("sơ đồ rộng ngang trang", 500.0, 180.0, False),
        ("hình vuông vừa phải", 130.0, 130.0, False),
    ],
)
def test_loc_anh_theo_hinh_dang(mo_ta, rong, cao, bi_loai):
    assert bool(ly_do_loai_anh(rong, cao, _DIEN_TICH_TRANG)) is bi_loai, mo_ta


def test_tran_dien_tich_chi_can_thiep_o_trang_kho_lon():
    """Chốt diện tích là lưới an toàn cho TRANG LỚN, không phải chốt chính trên A4.

    Trên khổ A4, KICH_THUOC_ANH_TOI_THIEU (120 điểm) đã tương đương ~2,9% diện tích trang -
    tức mọi ảnh lọt qua chốt kích thước đều tự khắc vượt ngưỡng diện tích. Chốt diện tích chỉ
    thật sự cắn ở trang khổ lớn, nơi 120 điểm chỉ còn là một chấm nhỏ. Ghi lại quan hệ này
    thành test để không ai chỉnh một trong hai con số mà tưởng cái kia vẫn còn tác dụng.
    """
    assert ly_do_loai_anh(140.0, 140.0, _DIEN_TICH_TRANG) is None
    assert ly_do_loai_anh(140.0, 140.0, _DIEN_TICH_TRANG_LON) is not None


def test_khong_ap_tran_dien_tich_khi_khong_biet_kich_thuoc_trang():
    """PPTX/DOCX không có khái niệm diện tích trang - chỉ áp được hai chốt hình dạng."""
    assert ly_do_loai_anh(130.0, 130.0) is None
    assert ly_do_loai_anh(float(KICH_THUOC_ANH_TOI_THIEU - 1), 300.0) == "quá nhỏ"


def _ban_ghi_anh_gia(tmp_path, ten, noi_dung_anh):
    duong_dan = tmp_path / ten
    duong_dan.write_bytes(noi_dung_anh)
    return {
        "nguon": "bai.pptx", "trang": 1, "noidung": f"{MOC_ANH}",
        "loai_noi_dung": "anh", "duong_dan_anh": str(duong_dan),
        "bam_anh": bam_bytes(noi_dung_anh),
    }


def test_loai_hinh_lap_lai_kieu_logo(tmp_path):
    logo = b"LOGO-TRUONG"
    cac_ban_ghi = [
        _ban_ghi_anh_gia(tmp_path, f"logo{i}.png", logo)
        for i in range(config.SO_LAN_LAP_COI_LA_LOGO)
    ]
    cac_ban_ghi.append(_ban_ghi_anh_gia(tmp_path, "so_do.png", b"SO-DO-THAT"))

    con_lai = loc_anh_lap_lai(cac_ban_ghi, "bai.pptx")
    assert len(con_lai) == 1
    assert con_lai[0]["duong_dan_anh"].endswith("so_do.png")


def test_giu_hinh_that_duoc_nhac_lai_vai_trang(tmp_path):
    """Hình tổng quan mở đầu mỗi chương lặp 2-3 lần là chuyện bình thường - đừng giết nó."""
    hinh = b"HINH-TONG-QUAN"
    cac_ban_ghi = [
        _ban_ghi_anh_gia(tmp_path, f"h{i}.png", hinh)
        for i in range(config.SO_LAN_LAP_COI_LA_LOGO - 1)
    ]
    assert loc_anh_lap_lai(cac_ban_ghi, "bai.pdf") == cac_ban_ghi


# ======================================================================
# 5. Chú thích ảnh: gộp ảnh trùng + cache + không gọi lại model
# ======================================================================

class ClientVisionGia:
    def __init__(self, noi_dung="Sơ đồ gồm 3 ô A, B, C."):
        self.noi_dung = noi_dung
        self.so_lan_goi = 0

    def list(self):
        class M:
            model = config.VISION_MODEL_NAME

        class KQ:
            models = [M()]

        return KQ()

    def chat(self, **_):
        self.so_lan_goi += 1
        return {"message": {"content": self.noi_dung}}


def test_anh_trung_noi_dung_chi_ton_mot_luot_goi_model(cache_tam, tmp_path, monkeypatch):
    """Một hình dùng lại ở 20 slide phải chỉ tốn đúng 1 lượt gọi, không phải 20."""
    monkeypatch.setattr(config, "BAT_CHU_THICH_ANH", True)
    monkeypatch.setattr(config, "SO_WORKER_VISION", 1)
    cac_anh = [_ban_ghi_anh_gia(tmp_path, f"a{i}.png", b"CUNG-MOT-HINH") for i in range(20)]

    client = ClientVisionGia()
    so_thanh_cong = bo_sung_chu_thich_vision(cac_anh, client=client)

    assert client.so_lan_goi == 1
    assert so_thanh_cong == 20, "mọi bản ghi trùng nội dung đều phải nhận được mô tả"
    assert all("Sơ đồ gồm 3 ô" in a["noidung"] for a in cac_anh)


def test_lan_build_sau_lay_chu_thich_tu_cache(cache_tam, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BAT_CHU_THICH_ANH", True)
    monkeypatch.setattr(config, "SO_WORKER_VISION", 1)

    client = ClientVisionGia()
    bo_sung_chu_thich_vision([_ban_ghi_anh_gia(tmp_path, "a.png", b"HINH")], client=client)
    assert client.so_lan_goi == 1

    lan_sau = [_ban_ghi_anh_gia(tmp_path, "a.png", b"HINH")]
    bo_sung_chu_thich_vision(lan_sau, client=client)
    assert client.so_lan_goi == 1, "ảnh đã chú thích rồi thì không được gọi model lần nữa"
    assert lan_sau[0]["co_chu_thich_vision"] is True


def test_chu_thich_song_song_van_gan_dung_mo_ta_cho_dung_anh(cache_tam, tmp_path, monkeypatch):
    """Chạy nhiều luồng không được làm lẫn mô tả của ảnh này sang ảnh khác."""
    monkeypatch.setattr(config, "BAT_CHU_THICH_ANH", True)
    monkeypatch.setattr(config, "SO_WORKER_VISION", 4)
    cac_anh = [_ban_ghi_anh_gia(tmp_path, f"a{i}.png", f"HINH-{i}".encode()) for i in range(8)]

    class ClientTheoAnh(ClientVisionGia):
        def chat(self, **tham_so):
            self.so_lan_goi += 1
            duong_dan = tham_so["messages"][0]["images"][0]
            return {"message": {"content": f"mo ta cua {Path(duong_dan).stem}"}}

    bo_sung_chu_thich_vision(cac_anh, client=ClientTheoAnh())
    for i, anh in enumerate(cac_anh):
        assert f"mo ta cua a{i}" in anh["noidung"]


# ======================================================================
# 6. Cache embedding
# ======================================================================

class EmbeddingGia:
    dimension = 4

    def __init__(self):
        self.da_encode = []

    def encode_tai_lieu(self, cac_text):
        self.da_encode.extend(cac_text)
        return np.array([[float(len(t)), 0.0, 0.0, 1.0] for t in cac_text], dtype="float32")


def test_cache_embedding_chi_encode_chunk_moi(cache_tam):
    dich_vu = EmbeddingGia()
    kho = KhoVectorDem()

    v1 = encode_co_cache(dich_vu, ["alpha", "beta"], kho)
    assert len(dich_vu.da_encode) == 2

    kho_moi = KhoVectorDem()  # nạp lại từ đĩa, như một lần build sau
    v2 = encode_co_cache(dich_vu, ["alpha", "beta", "gamma"], kho_moi)

    assert dich_vu.da_encode[2:] == ["gamma"], "chỉ chunk mới được encode lại"
    assert np.allclose(v1, v2[:2]), "vector lấy từ cache phải trùng khít vector đã encode"


def test_cache_embedding_giu_dung_thu_tu_dau_vao(cache_tam):
    """Trả sai thứ tự = gán nhầm vector cho chunk, sai âm thầm và không thể lần ra."""
    dich_vu = EmbeddingGia()
    kho = KhoVectorDem()
    encode_co_cache(dich_vu, ["mot", "haiii"], kho)

    ket_qua = encode_co_cache(dich_vu, ["baaaaa", "mot", "haiii"], kho)
    assert [v[0] for v in ket_qua] == [6.0, 3.0, 5.0]


# ======================================================================
# 7. Sổ băm tài liệu trong index - nền tảng của build tăng dần
# ======================================================================

@contextmanager
def _thu_muc_faiss_ghi_duoc():
    """Thư mục tạm mà FAISS ghi được.

    Cố tình KHÔNG dùng fixture tmp_path của pytest: trên máy có display name Windows chứa
    ký tự tiếng Việt có dấu, pytest tạo thư mục "pytest-of-<display-name>", và FAISS (dùng
    fopen theo ANSI codepage ở tầng C++) không ghi được vào đường dẫn đó. Cùng lý do đã ghi
    ở tests/test_retrieval.py.
    """
    with tempfile.TemporaryDirectory() as thu_muc:
        yield Path(thu_muc)


def _duong_dan_index(thu_muc: Path) -> dict:
    return dict(
        index_path=thu_muc / "index.faiss",
        metadata_path=thu_muc / "metadata.pkl",
        info_path=thu_muc / "index_info.json",
    )


def _store_hai_tai_lieu():
    from rag.vector_store import VectorStore

    store = VectorStore(dimension=4)
    store.them(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype="float32"),
        [
            {"nguon": "a.pdf", "trang": 1, "vi_tri": 0, "noidung": "noi dung a1"},
            {"nguon": "a.pdf", "trang": 2, "vi_tri": 0, "noidung": "noi dung a2"},
            {"nguon": "b.pdf", "trang": 1, "vi_tri": 0, "noidung": "noi dung b1"},
        ],
    )
    store.bam_tai_lieu = {"a.pdf": "bam-a", "b.pdf": "bam-b"}
    return store


def test_so_bam_tai_lieu_song_qua_luu_va_nap():
    """Không lưu được sổ băm thì mở lại app là mất sạch khả năng build tăng dần."""
    from rag.vector_store import VectorStore

    store = _store_hai_tai_lieu()
    with _thu_muc_faiss_ghi_duoc() as thu_muc:
        duong_dan = _duong_dan_index(thu_muc)
        store.luu(**duong_dan)
        nap_lai = VectorStore.tai(**duong_dan)
    assert nap_lai.bam_tai_lieu == {"a.pdf": "bam-a", "b.pdf": "bam-b"}


def test_xoa_tai_lieu_thi_xoa_luon_khoi_so_bam():
    """Còn tên trong sổ băm mà không còn vector nào = lần build sau bỏ qua nó, index thiếu
    một tài liệu mà không có dấu hiệu gì."""
    store = _store_hai_tai_lieu()
    assert store.xoa_theo_nguon("a.pdf") == 2
    assert "a.pdf" not in store.bam_tai_lieu
    assert store.bam_tai_lieu == {"b.pdf": "bam-b"}


def test_index_build_bang_ban_cu_khong_co_so_bam_thi_lui_ve_doc_lai_tat_ca():
    """Giảm cấp chứ không hỏng: thiếu sổ băm nghĩa là mọi tài liệu bị coi là chưa xử lý."""
    import json

    from rag.vector_store import VectorStore

    store = _store_hai_tai_lieu()
    with _thu_muc_faiss_ghi_duoc() as thu_muc:
        duong_dan = _duong_dan_index(thu_muc)
        store.luu(**duong_dan)

        thong_tin = json.loads(duong_dan["info_path"].read_text(encoding="utf-8"))
        del thong_tin["bam_tai_lieu"]
        duong_dan["info_path"].write_text(
            json.dumps(thong_tin, ensure_ascii=False), encoding="utf-8"
        )
        nap_lai = VectorStore.tai(**duong_dan)

    assert nap_lai.bam_tai_lieu == {}
    assert nap_lai.so_luong_vector == 3, "dữ liệu cũ vẫn còn nguyên, chỉ mất thông tin phụ"


# ======================================================================
# 8. Quyết định của build tăng dần: file nào đọc lại, file nào gỡ ra
# ======================================================================

@pytest.mark.parametrize(
    "mo_ta, trong_index, tren_dia, can_doc, can_xoa, giu_nguyen",
    [
        ("index rỗng: mọi tài liệu đều phải đọc",
         {}, {"a.pdf": "1", "b.pdf": "2"}, ["a.pdf", "b.pdf"], [], []),
        ("không có gì đổi: không đọc lại file nào",
         {"a.pdf": "1"}, {"a.pdf": "1"}, [], [], ["a.pdf"]),
        ("thêm 1 file vào 2 file cũ: chỉ đọc file mới",
         {"a.pdf": "1", "b.pdf": "2"}, {"a.pdf": "1", "b.pdf": "2", "c.pdf": "3"},
         ["c.pdf"], [], ["a.pdf", "b.pdf"]),
        ("sửa nội dung 1 file: chỉ đọc lại file đó",
         {"a.pdf": "1", "b.pdf": "2"}, {"a.pdf": "1", "b.pdf": "MOI"},
         ["b.pdf"], [], ["a.pdf"]),
        ("xoá file khỏi thư mục: phải gỡ vector của nó",
         {"a.pdf": "1", "b.pdf": "2"}, {"a.pdf": "1"}, [], ["b.pdf"], ["a.pdf"]),
        ("đổi TÊN file (nội dung giữ nguyên): gỡ tên cũ, đọc tên mới",
         {"cu.pdf": "1"}, {"moi.pdf": "1"}, ["moi.pdf"], ["cu.pdf"], []),
    ],
)
def test_quyet_dinh_build_tang_dan(mo_ta, trong_index, tren_dia, can_doc, can_xoa, giu_nguyen):
    from rag.vector_store import so_sanh_bam_tai_lieu

    assert so_sanh_bam_tai_lieu(trong_index, tren_dia) == (can_doc, can_xoa, giu_nguyen), mo_ta


def test_giu_thu_tu_tren_dia():
    """Thứ tự xử lý phải khớp thứ tự người dùng nhìn thấy trên giao diện."""
    from rag.vector_store import so_sanh_bam_tai_lieu

    tren_dia = {f"{i:02d}.pdf": str(i) for i in range(10)}
    can_doc, _, _ = so_sanh_bam_tai_lieu({}, tren_dia)
    assert can_doc == list(tren_dia)


def test_cache_embedding_khong_sap_khi_khong_ghi_duoc_xuong_dia(cache_tam, monkeypatch):
    """Sự cố ghi cache chỉ được phép làm hệ thống CHẬM lại, không được làm sập lần build."""
    dich_vu = EmbeddingGia()
    kho = KhoVectorDem()
    monkeypatch.setattr(
        kho, "luu", lambda: None  # giả lập ghi đĩa thất bại: sổ khoá đã cập nhật, file thì chưa
    )

    v1 = encode_co_cache(dich_vu, ["alpha", "beta"], kho)
    v2 = encode_co_cache(dich_vu, ["alpha", "gamma"], kho)

    assert np.allclose(v1[0], v2[0]), "vector chưa kịp ghi đĩa vẫn phải dùng lại được trong phiên"
    assert dich_vu.da_encode == ["alpha", "beta", "gamma"], "chỉ 'gamma' là chunk thật sự mới"

