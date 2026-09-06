"""Trích xuất hình ảnh từ tài liệu và gắn chúng với văn bản xung quanh."""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docx.opc.constants import RELATIONSHIP_TYPE
from pptx.enum.shapes import MSO_SHAPE_TYPE

import config
from rag.bo_nho_dem import bam_bytes, bam_file

logger = logging.getLogger(__name__)

MOC_ANH = "[HÌNH]"

KICH_THUOC_ANH_TOI_THIEU = 120

DO_DAI_CHU_THICH_LAN_CAN = 400

_MAU_DONG_CHU_THICH = re.compile(
    r"^\s*(hình|hinh|figure|fig|biểu đồ|bieu do|sơ đồ|so do|bảng|table)\s*\d+\s*[:.\-]",
    re.IGNORECASE,
)


def _ten_file_an_toan(nguon: str, trang: int, thu_tu: int, duoi: str) -> str:
    """Tên file ảnh suy ra từ (nguồn, trang, thứ tự) nên ổn định giữa các lần build - build
    lại index không tạo ra một đống file rác trùng nội dung khác tên."""
    goc = re.sub(r"[^\w\-.]", "_", Path(nguon).stem)[:60]
    return f"{goc}__t{trang}_{thu_tu}{duoi}"


def _chon_chu_thich(cac_dong: List[str]) -> str:
    """Chọn văn bản mô tả hình từ các dòng lân cận."""
    cac_dong = [d.strip() for d in cac_dong if d and d.strip()]
    dong_chu_thich = [d for d in cac_dong if _MAU_DONG_CHU_THICH.match(d)]
    if dong_chu_thich:
        return " ".join(dong_chu_thich)[:DO_DAI_CHU_THICH_LAN_CAN]
    return " ".join(cac_dong)[:DO_DAI_CHU_THICH_LAN_CAN]


def _la_anh_cua_trang_chu(rong: float, cao: float, dien_tich_trang: float) -> bool:
    """Ảnh này có phải là ảnh chụp của một TRANG CHỮ (PDF scan) không?"""
    if dien_tich_trang <= 0:
        return False
    return (rong * cao) / dien_tich_trang >= config.TY_LE_DIEN_TICH_ANH_TOAN_TRANG


def ly_do_loai_anh(rong: float, cao: float, dien_tich_trang: float = 0.0) -> Optional[str]:
    """Ảnh này có đáng đưa vào index không? Trả về LÝ DO loại, hoặc None nếu giữ."""
    if rong < KICH_THUOC_ANH_TOI_THIEU or cao < KICH_THUOC_ANH_TOI_THIEU:
        return "quá nhỏ"
    canh_dai, canh_ngan = max(rong, cao), max(min(rong, cao), 1e-9)
    if config.TY_LE_CANH_ANH_TRANG_TRI > 0 and canh_dai / canh_ngan > config.TY_LE_CANH_ANH_TRANG_TRI:
        return "dải trang trí (tỉ lệ cạnh quá dẹt)"
    if dien_tich_trang > 0 and config.TY_LE_DIEN_TICH_ANH_TOI_THIEU > 0:
        if (rong * cao) / dien_tich_trang < config.TY_LE_DIEN_TICH_ANH_TOI_THIEU:
            return "chiếm quá ít diện tích trang (icon/logo)"
    return None


def ly_do_loai_anh_blob(du_lieu: bytes) -> Optional[str]:
    """Như ly_do_loai_anh() nhưng cho ảnh PPTX/DOCX - đọc kích thước thẳng từ bytes."""
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(du_lieu)) as anh:
            rong, cao = anh.size
    except Exception:  # noqa: BLE001
        return None
    return ly_do_loai_anh(float(rong), float(cao))


def loc_anh_lap_lai(cac_ban_ghi: List[Dict], nguon: str) -> List[Dict]:
    """Loại những ảnh có NỘI DUNG GIỐNG HỆT lặp lại nhiều lần trong cùng một tài liệu."""
    if not cac_ban_ghi or config.SO_LAN_LAP_COI_LA_LOGO <= 0:
        return cac_ban_ghi

    for ban_ghi in cac_ban_ghi:
        if not ban_ghi.get("bam_anh"):
            try:
                ban_ghi["bam_anh"] = bam_file(Path(ban_ghi["duong_dan_anh"]))
            except OSError:
                ban_ghi["bam_anh"] = ""

    so_lan = Counter(b["bam_anh"] for b in cac_ban_ghi if b["bam_anh"])
    bam_logo = {b for b, n in so_lan.items() if n >= config.SO_LAN_LAP_COI_LA_LOGO}
    if not bam_logo:
        return cac_ban_ghi

    giu = [b for b in cac_ban_ghi if b["bam_anh"] not in bam_logo]
    logger.info(
        "'%s': bỏ %d bản ghi ảnh thuộc %d hình lặp lại từ %d lần trở lên (logo/watermark/"
        "khung mẫu slide) - chúng chỉ tạo ra các chunk giống hệt nhau trong index.",
        nguon, len(cac_ban_ghi) - len(giu), len(bam_logo), config.SO_LAN_LAP_COI_LA_LOGO,
    )
    return giu


def _ban_ghi_anh(
    nguon: str, trang: int, duong_dan_anh: Path, chu_thich: str, bam_anh: str = ""
) -> Dict:
    """Một ảnh trở thành một "trang" riêng trong luồng dữ liệu."""
    return {
        "nguon": nguon,
        "trang": trang,
        "noidung": f"{MOC_ANH} {chu_thich}".strip(),
        "loai_noi_dung": "anh",
        "duong_dan_anh": str(duong_dan_anh),
        "bam_anh": bam_anh,
    }


def ung_vien_anh_trang(trang) -> List[Tuple[Tuple[float, float, float, float], bool]]:
    """Liệt kê ảnh ĐÁNG GIỮ trên một trang PDF - CHƯA render gì cả."""
    dien_tich_trang = float(trang.width or 0) * float(trang.height or 0)
    ket_qua = []
    for anh in trang.images:
        rong = float(anh.get("width") or 0)
        cao = float(anh.get("height") or 0)
        ly_do = ly_do_loai_anh(rong, cao, dien_tich_trang)
        if ly_do:
            continue
        bbox = (
            max(anh["x0"], trang.bbox[0]), max(anh["top"], trang.bbox[1]),
            min(anh["x1"], trang.bbox[2]), min(anh["bottom"], trang.bbox[3]),
        )
        ket_qua.append((bbox, _la_anh_cua_trang_chu(rong, cao, dien_tich_trang)))
    return ket_qua


def luu_anh_trang_pdf(
    nguon: str, trang, so_trang: int, cac_bbox: List[Tuple[float, float, float, float]],
    cac_dong_text: List[str],
) -> List[Dict]:
    """Render + lưu ra file những ảnh đã được ung_vien_anh_trang() chọn."""
    ket_qua = []
    chu_thich = _chon_chu_thich(cac_dong_text)
    for thu_tu, bbox in enumerate(cac_bbox, start=1):
        try:
            pil = trang.crop(bbox).to_image(resolution=110).original
        except Exception as loi:  # noqa: BLE001
            logger.warning("Bỏ qua 1 ảnh ở trang %d của '%s': %s",
                           so_trang, nguon, type(loi).__name__)
            continue
        dich = config.IMAGES_DIR / _ten_file_an_toan(nguon, so_trang, thu_tu, ".png")
        pil.save(dich)
        ket_qua.append(_ban_ghi_anh(nguon, so_trang, dich, chu_thich))
    return ket_qua


def trich_anh_pptx(duong_dan: Path, trinh_chieu) -> List[Dict]:
    """Trích ảnh từng slide, kể cả ảnh nằm trong group shape."""
    from rag.document_loader import duyet_shape

    ket_qua = []
    for so_slide, slide in enumerate(trinh_chieu.slides, start=1):
        cac_shape = list(duyet_shape(slide.shapes))
        cac_dong = [
            s.text_frame.text for s in cac_shape
            if s.has_text_frame and s.text_frame.text.strip()
        ]
        thu_tu = 0
        da_lay = set()
        for shape in cac_shape:
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            try:
                anh = shape.image
            except (ValueError, KeyError, AttributeError):
                continue
            if ly_do_loai_anh_blob(anh.blob):
                da_lay.add(getattr(anh, "sha1", None) or "")
                continue
            thu_tu += 1
            dich = config.IMAGES_DIR / _ten_file_an_toan(
                duong_dan.name, so_slide, thu_tu, f".{anh.ext}"
            )
            dich.write_bytes(anh.blob)
            da_lay.add(getattr(anh, "sha1", None) or dich.name)
            ket_qua.append(
                _ban_ghi_anh(
                    duong_dan.name, so_slide, dich, _chon_chu_thich(cac_dong),
                    bam_bytes(anh.blob),
                )
            )

        for quan_he in slide.part.rels.values():
            if quan_he.reltype != RELATIONSHIP_TYPE.IMAGE or quan_he.is_external:
                continue
            try:
                phan_anh = quan_he.target_part
                khoa = getattr(phan_anh, "sha1", None) or str(phan_anh.partname)
                if khoa in da_lay:
                    continue
                du_lieu = phan_anh.blob
                duoi = Path(str(phan_anh.partname)).suffix or ".png"
            except Exception as loi:  # noqa: BLE001
                logger.warning(
                    "Slide %d của '%s': bỏ qua 1 ảnh (%s).",
                    so_slide, duong_dan.name, type(loi).__name__,
                )
                continue
            da_lay.add(khoa)
            if ly_do_loai_anh_blob(du_lieu):
                continue
            thu_tu += 1
            dich = config.IMAGES_DIR / _ten_file_an_toan(
                duong_dan.name, so_slide, thu_tu, duoi
            )
            dich.write_bytes(du_lieu)
            ket_qua.append(
                _ban_ghi_anh(
                    duong_dan.name, so_slide, dich, _chon_chu_thich(cac_dong),
                    bam_bytes(du_lieu),
                )
            )
    return ket_qua


def trich_anh_docx(duong_dan: Path, document) -> List[Dict]:
    """Trích ảnh DOCX qua quan hệ (rels) của phần thân tài liệu."""
    ket_qua = []
    cac_dong = [p.text for p in document.paragraphs if p.text.strip()]
    thu_tu = 0
    for quan_he in document.part.rels.values():
        if quan_he.reltype != RELATIONSHIP_TYPE.IMAGE:
            continue
        try:
            du_lieu = quan_he.target_part.blob
            duoi = f".{quan_he.target_part.image.ext}"
        except Exception as loi:  # noqa: BLE001
            logger.warning("Bỏ qua 1 ảnh trong '%s': %s", duong_dan.name, type(loi).__name__)
            continue
        if ly_do_loai_anh_blob(du_lieu):
            continue
        thu_tu += 1
        dich = config.IMAGES_DIR / _ten_file_an_toan(duong_dan.name, 1, thu_tu, duoi)
        dich.write_bytes(du_lieu)
        ket_qua.append(
            _ban_ghi_anh(
                duong_dan.name, 1, dich, _chon_chu_thich(cac_dong), bam_bytes(du_lieu)
            )
        )
    return ket_qua
