"""Test cho khả năng đọc TÀI LIỆU SCAN — thứ đã khiến hệ thống trả lời sai hoàn toàn."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from rag.document_loader import _bo_ban_ghi_anh_rong
from rag.image_extractor import MOC_ANH, _la_anh_cua_trang_chu
from rag.vision_caption import trang_can_ocr


@pytest.mark.parametrize(
    "mo_ta, text, so_anh, ket_qua_mong_doi",
    [
        ("trang scan: không chữ, có ảnh", "", 1, True),
        ("trang scan: vài chữ lạc, có ảnh", "12", 1, True),
        ("trang chữ bình thường", " ".join(["từ"] * 200), 0, False),
        ("trang chữ có kèm hình minh hoạ", " ".join(["từ"] * 200), 3, False),
        ("trang trống hoàn toàn, KHÔNG ảnh", "", 0, False),
        ("trang hỏng font: toàn mã (cid:)", "(cid:12) " * 40, 0, True),
    ],
)
def test_nhan_ra_trang_can_ocr(mo_ta, text, so_anh, ket_qua_mong_doi):
    """Trang trống mà KHÔNG có ảnh thì OCR cũng vô ích - không được phí một lượt gọi model."""
    assert trang_can_ocr(text, so_anh) is ket_qua_mong_doi, mo_ta


def test_anh_phu_phan_lon_trang_scan_bi_coi_la_anh_cua_trang_chu():
    assert _la_anh_cua_trang_chu(495.0, 821.0, 595.0 * 841.0)


def test_hinh_minh_hoa_nho_tren_trang_scan_van_duoc_giu():
    """Một trang scan vẫn có thể có thêm hình nhỏ nhúng riêng - hình đó phải giữ lại."""
    assert not _la_anh_cua_trang_chu(200.0, 150.0, 595.0 * 841.0)


def test_ty_le_dien_tich_bang_khong_khong_gay_chia_cho_khong():
    assert not _la_anh_cua_trang_chu(100.0, 100.0, 0.0)


def test_anh_khong_co_chu_thich_bi_loai_khoi_index():
    """Chunk có nội dung đúng bằng "[HÌNH]" không mang thông tin nào để tra cứu, nhưng vẫn
    chiếm một vector và vẫn lọt được vào top-K. Đúng 379 chunk như thế đã tạo nên sự cố."""
    cac_trang = [
        {"nguon": "a.pdf", "trang": 1, "noidung": MOC_ANH, "loai_noi_dung": "anh"},
        {"nguon": "a.pdf", "trang": 2, "noidung": f"{MOC_ANH}  ", "loai_noi_dung": "anh"},
        {"nguon": "a.pdf", "trang": 3, "noidung": f"{MOC_ANH} Hình 2: sơ đồ quy trình",
         "loai_noi_dung": "anh"},
        {"nguon": "a.pdf", "trang": 4, "noidung": "Đoạn văn xuôi bình thường."},
    ]
    con_lai = _bo_ban_ghi_anh_rong(cac_trang)

    assert len(con_lai) == 2
    assert [m["trang"] for m in con_lai] == [3, 4]


def test_khong_dung_nham_van_ban_ngan_thanh_anh_rong():
    """Bộ lọc chỉ được đụng tới bản ghi ẢNH - một đoạn văn ngắn vẫn phải giữ nguyên."""
    cac_trang = [{"nguon": "a.pdf", "trang": 1, "noidung": "Ngắn."}]
    assert _bo_ban_ghi_anh_rong(cac_trang) == cac_trang


def test_canh_bao_khi_tai_lieu_khong_doc_duoc_chu_nao(tmp_path, caplog):
    """Build "thành công" một tài liệu không đọc được là kiểu hỏng im lặng - phải nói ra."""
    from rag.document_loader import _canh_bao_tai_lieu_khong_doc_duoc

    (tmp_path / "sach_scan.pdf").write_bytes(b"%PDF-1.4")
    cac_trang = [
        {"nguon": "sach_scan.pdf", "trang": 1, "noidung": f"{MOC_ANH} x", "loai_noi_dung": "anh"},
    ]
    with caplog.at_level("ERROR"):
        _canh_bao_tai_lieu_khong_doc_duoc(cac_trang, [tmp_path / "sach_scan.pdf"])

    assert any("KHÔNG ĐỌC ĐƯỢC NỘI DUNG" in r.message for r in caplog.records)
    assert any("sach_scan.pdf" in str(r.args) for r in caplog.records)


def test_khong_canh_bao_khi_tai_lieu_doc_duoc_binh_thuong(tmp_path, caplog):
    from rag.document_loader import _canh_bao_tai_lieu_khong_doc_duoc

    (tmp_path / "binh_thuong.pdf").write_bytes(b"%PDF-1.4")
    cac_trang = [
        {"nguon": "binh_thuong.pdf", "trang": 1, "noidung": "x" * 500},
    ]
    with caplog.at_level("ERROR"):
        _canh_bao_tai_lieu_khong_doc_duoc(cac_trang, [tmp_path / "binh_thuong.pdf"])

    assert not [r for r in caplog.records if "KHÔNG ĐỌC ĐƯỢC" in r.message]


def test_ocr_bat_mac_dinh():
    """Mặc định TẮT từng khiến một cuốn sách 383 trang thành index rỗng mà không báo lỗi.
    OCR chỉ chạy cho trang ĐO ĐƯỢC là không đọc được, nên tài liệu bình thường tốn 0 chi phí."""
    assert config.BAT_OCR_DU_PHONG is True


def test_file_hong_khong_lam_sap_ca_lan_build(tmp_path, caplog, monkeypatch):
    """File hỏng (đặt mật khẩu, tải về dở dang, sai định dạng) không được làm sập cả lần build và
    làm mất công build của mọi tài liệu còn lại."""
    import config
    from docx import Document

    from rag.document_loader import doc_thu_muc

    monkeypatch.setattr(config, "BAT_TRICH_ANH", False)
    monkeypatch.setattr(config, "BAT_CHU_THICH_ANH", False)

    d = Document()
    d.add_paragraph("Noi dung that cua tai lieu doc duoc binh thuong, du dai de khong bi loc.")
    d.save(tmp_path / "tot.docx")
    (tmp_path / "hong.pdf").write_bytes(b"day khong phai la mot file PDF hop le")

    with caplog.at_level("ERROR"):
        cac_trang = doc_thu_muc(tmp_path)

    assert any(m["nguon"] == "tot.docx" for m in cac_trang), (
        "tài liệu đọc được vẫn phải vào index dù có file khác bị hỏng"
    )
    assert any("KHÔNG ĐỌC ĐƯỢC FILE" in r.message for r in caplog.records), (
        "phải báo lỗi rõ ràng, tuyệt đối không nuốt lỗi"
    )
