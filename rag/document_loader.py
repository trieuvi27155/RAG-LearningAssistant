"""Đọc tài liệu PDF/PPTX/DOCX, giữ metadata (tên file, số trang/slide) ngay từ bước đọc."""

import logging
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pdfplumber
from docx import Document
from docx.table import Table as BangDocx
from docx.text.paragraph import Paragraph as DoanVanDocx
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

import config
from rag import bo_nho_dem, do_thoi_gian, tai_nguyen_gpu
from rag.image_extractor import (
    MOC_ANH,
    loc_anh_lap_lai,
    luu_anh_trang_pdf,
    trich_anh_docx,
    trich_anh_pptx,
    ung_vien_anh_trang,
)
from rag.vision_caption import (
    bo_sung_chu_thich_vision,
    mo_hinh_vision_co_san,
    ocr_trang_pdf,
    trang_can_ocr,
)

logger = logging.getLogger(__name__)

CAC_DUOI_HO_TRO = (".pdf", ".pptx", ".docx")

_client_vision = None
_da_canh_bao_vision = False

MOC_BANG_MO = "[BẢNG]"
MOC_BANG_DONG = "[/BẢNG]"

_MAU_WATERMARK_STUDOCU = re.compile(r"lOMoARcPSD\|\d+|Downloaded by .+?\([^)]*@[^)]*\)")

_MAU_CID_PDF = re.compile(r"\(cid:\d+\)")

_MAU_CUM_CHU = re.compile(r"[A-Za-zÀ-ỹ]+")


def _ty_le_dinh_chu(text: str) -> float:
    """Tỉ lệ ký tự chữ nằm trong những cụm dài bất thường (không có khoảng trắng ngăn từ)."""
    do_dai = [len(t) for t in _MAU_CUM_CHU.findall(text)]
    tong = sum(do_dai)
    if tong < config.SO_KY_TU_TOI_THIEU_DE_DO:
        return 0.0
    return sum(d for d in do_dai if d >= config.DO_DAI_CUM_DINH_CHU) / tong


def _ty_le_tu_le(text: str) -> float:
    """Tỉ lệ "từ" chỉ có ĐÚNG MỘT chữ cái - dấu hiệu ngược lại của dính chữ."""
    cac_tu = _MAU_CUM_CHU.findall(text)
    if not cac_tu:
        return 0.0
    return sum(1 for t in cac_tu if len(t) == 1) / len(cac_tu)


def _trich_text(doi_tuong, x_tolerance=None) -> str:
    """extract_text() có/không tham số x_tolerance - gom vào 1 chỗ để 2 nhánh dùng chung."""
    if x_tolerance is None:
        return doi_tuong.extract_text() or ""
    return doi_tuong.extract_text(x_tolerance=x_tolerance) or ""


class HieuChinhXTolerance:
    """Nhớ x_tolerance đã dò được cho MỘT tài liệu, để các trang sau không phải dò lại."""

    def __init__(self):
        self.x_da_chon: Optional[float] = None
        self.so_lan_dong_y = 0

    @property
    def da_hieu_chinh(self) -> bool:
        return (
            self.x_da_chon is not None
            and self.so_lan_dong_y >= config.SO_TRANG_HIEU_CHINH_X_TOLERANCE
        )

    def thu_tu_uu_tien(self) -> List[float]:
        """Thứ tự thử các mức x_tolerance cho trang tiếp theo."""
        if self.x_da_chon is None:
            return list(config.CAC_X_TOLERANCE_THU)
        con_lai = [x for x in config.CAC_X_TOLERANCE_THU if x != self.x_da_chon]
        return [self.x_da_chon] + con_lai

    def ghi_nhan(self, x_tolerance: float) -> None:
        if x_tolerance == self.x_da_chon:
            self.so_lan_dong_y += 1
        else:
            self.x_da_chon, self.so_lan_dong_y = x_tolerance, 1


def _trich_text_thich_ung(
    doi_tuong, ten_file: str = "", so_trang=None, hieu_chinh: Optional[HieuChinhXTolerance] = None
) -> str:
    """Đọc text một trang PDF, TỰ DÒ tham số đọc lại khi phát hiện chữ bị dính liền."""
    text = _trich_text(doi_tuong)
    if not config.BAT_DOC_LAI_TRANG_DINH_CHU:
        return text
    ty_le_goc = _ty_le_dinh_chu(text)
    if ty_le_goc < config.TY_LE_DINH_CHU_DE_DOC_LAI:
        return text

    tran_tu_le = _ty_le_tu_le(text) + config.MUC_TANG_TU_LE_CHAP_NHAN
    thu_tu = hieu_chinh.thu_tu_uu_tien() if hieu_chinh else list(config.CAC_X_TOLERANCE_THU)
    tot_nhat, ty_le_tot_nhat, x_tot_nhat = None, ty_le_goc, None
    for x_tolerance in thu_tu:
        thu = _trich_text(doi_tuong, x_tolerance=x_tolerance)
        if not thu:
            continue
        ty_le_thu = _ty_le_dinh_chu(thu)
        if ty_le_thu >= ty_le_tot_nhat or _ty_le_tu_le(thu) > tran_tu_le:
            continue
        tot_nhat, ty_le_tot_nhat, x_tot_nhat = thu, ty_le_thu, x_tolerance
        if ty_le_thu <= config.TY_LE_DINH_CHU_DAT_YEU_CAU:
            break

    if tot_nhat is None:
        return text
    if hieu_chinh is not None:
        hieu_chinh.ghi_nhan(x_tot_nhat)
    ghi_log = logger.debug if (hieu_chinh and hieu_chinh.da_hieu_chinh) else logger.info
    ghi_log(
        "Trang %s của '%s' bị dính chữ (%.0f%%) - đọc lại với x_tolerance=%.1f, còn %.0f%%.",
        so_trang if so_trang is not None else "?", ten_file or "?",
        ty_le_goc * 100, x_tot_nhat, ty_le_tot_nhat * 100,
    )
    return tot_nhat
_MAU_KHOANG_TRANG_THUA = re.compile(r"[ \t]{2,}")

_CAC_TIEU_DE_MUC_LUC = ("mục lục", "table of contents", "contents")

_MAU_DONG_KET_THUC_BANG_SO = re.compile(r"\d{1,4}\s*$")
_NGUONG_TY_LE_MUC_LUC = 0.5
_SO_DONG_TOI_THIEU_DE_XET = 6


def _chuan_hoa_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _don_dep_watermark(text: str) -> str:
    """Dọn nhiễu lặp lại trong text vừa đọc: watermark và ký tự PDF không giải mã được."""
    text = _MAU_WATERMARK_STUDOCU.sub("", text)
    text = _MAU_CID_PDF.sub("", text)
    return _MAU_KHOANG_TRANG_THUA.sub(" ", text)


def _la_trang_muc_luc(text: str) -> bool:
    """Đoán 1 trang có phải Mục lục (hoặc là 1 trang TIẾP THEO của mục lục nhiều trang) hay
    không. 2 cách nhận diện, chỉ cần khớp 1:"""
    dau_trang = text.strip().lower()[:40]
    if any(dau_trang.startswith(tieu_de) for tieu_de in _CAC_TIEU_DE_MUC_LUC):
        return True

    cac_dong = [dong for dong in text.split("\n") if dong.strip()]
    if len(cac_dong) < _SO_DONG_TOI_THIEU_DE_XET:
        return False
    so_dong_ket_thuc_bang_so = sum(1 for dong in cac_dong if _MAU_DONG_KET_THUC_BANG_SO.search(dong))
    return (so_dong_ket_thuc_bang_so / len(cac_dong)) >= _NGUONG_TY_LE_MUC_LUC


def _danh_dau_tieu_de(text: str, cap: int = 2) -> str:
    """Bọc 1 dòng thành dấu tiêu đề kiểu Markdown để splitter ưu tiên cắt tại đây."""
    return f"\n{'#' * cap} {text.strip()}\n"


def _phat_hien_tieu_de_pdf(trang) -> set:
    """Đoán những dòng nào trong trang PDF là tiêu đề, dựa vào cỡ chữ và độ dài."""
    cac_ky_tu = trang.chars
    if not cac_ky_tu:
        return set()

    theo_dong: Dict[int, List] = {}
    for ky_tu in cac_ky_tu:
        theo_dong.setdefault(round(ky_tu["top"]), []).append(ky_tu)

    dem_co_chu = Counter(round(k["size"], 1) for k in cac_ky_tu)
    co_chu_ap_dao = dem_co_chu.most_common(1)[0][0]
    if co_chu_ap_dao <= 0:
        return set()

    tieu_de = set()
    for cac_ky_tu_dong in theo_dong.values():
        noi_dung = "".join(k["text"] for k in cac_ky_tu_dong).strip()
        if not noi_dung or len(noi_dung) > config.DO_DAI_TOI_DA_TIEU_DE:
            continue
        co_chu_dong = sum(k["size"] for k in cac_ky_tu_dong) / len(cac_ky_tu_dong)
        if co_chu_dong >= co_chu_ap_dao * config.TY_LE_KICH_THUOC_CHU_TIEU_DE:
            tieu_de.add(noi_dung)
    return tieu_de


def _o_bang_khong_lap(hang) -> List[str]:
    """Lấy text các ô của 1 hàng, KHỬ phần nhân bản do ô gộp (merged cell)."""
    ket_qua, da_thay = [], set()
    for o in hang.cells:
        nen = getattr(o, "_tc", None)
        if nen is None:
            nen = getattr(o, "_element", None)
        khoa = id(nen) if nen is not None else None
        if khoa is not None and khoa in da_thay:
            continue
        if khoa is not None:
            da_thay.add(khoa)
        ket_qua.append(o.text)
    return ket_qua


def _bang_sang_markdown(bang: List[List]) -> str:
    """Đổi bảng (list hàng × ô) thành bảng Markdown."""
    cac_hang = [
        ["" if o is None else " ".join(str(o).split()) for o in hang]
        for hang in bang
        if hang and any(o is not None and str(o).strip() for o in hang)
    ]
    if not cac_hang:
        return ""
    so_cot = max(len(h) for h in cac_hang)
    cac_hang = [h + [""] * (so_cot - len(h)) for h in cac_hang]

    cot_co_chu = [i for i in range(so_cot) if any(h[i].strip() for h in cac_hang)]
    if cot_co_chu and len(cot_co_chu) < so_cot:
        cac_hang = [[h[i] for i in cot_co_chu] for h in cac_hang]
        so_cot = len(cot_co_chu)

    dong = ["| " + " | ".join(cac_hang[0]) + " |",
            "| " + " | ".join(["---"] * so_cot) + " |"]
    dong += ["| " + " | ".join(h) + " |" for h in cac_hang[1:]]
    return "\n".join(dong)


def _lay_client_vision():
    """Tạo Ollama client dùng chung cho OCR, chỉ một lần cho cả lần build."""
    global _client_vision, _da_canh_bao_vision
    if _client_vision is None and not _da_canh_bao_vision:
        import ollama

        client = ollama.Client(host=config.OLLAMA_HOST)
        if mo_hinh_vision_co_san(client):
            _client_vision = client
        else:
            _da_canh_bao_vision = True
            logger.warning(
                "BAT_OCR_DU_PHONG đang bật nhưng model vision '%s' chưa được pull - bỏ qua "
                "OCR. Chạy: ollama pull %s",
                config.VISION_MODEL_NAME, config.VISION_MODEL_NAME,
            )
    return _client_vision


def _ocr_cac_trang(
    pdf, cac_so_trang: List[int], ten_file: str, bam_tai_lieu: str
) -> Dict[int, str]:
    """OCR một LOẠT trang PDF cùng lúc, trả về {số trang: text đọc được}."""
    if not cac_so_trang:
        return {}

    ket_qua: Dict[int, str] = {}
    can_goi: List[tuple] = []
    dung_cache = config.BAT_CACHE_INGESTION and bool(bam_tai_lieu)

    client = _lay_client_vision()
    for so_trang in cac_so_trang:
        khoa = bo_nho_dem.khoa_ocr(bam_tai_lieu, so_trang) if dung_cache else None
        if khoa:
            da_co = bo_nho_dem.kho_ocr.lay_text(khoa)
            if da_co is not None:
                ket_qua[so_trang] = da_co
                continue
        if client is None:
            continue
        duong_dan_tam = config.IMAGES_DIR / f"_ocr_tam_{so_trang}.png"
        try:
            with do_thoi_gian.do("ocr_render_trang"):
                trang = pdf.pages[so_trang - 1]
                trang.to_image(resolution=config.DPI_RENDER_TRANG_OCR).original.save(duong_dan_tam)
        except Exception as loi:  # noqa: BLE001
            logger.warning("Không render được trang %d của '%s': %s", so_trang, ten_file, loi)
            duong_dan_tam.unlink(missing_ok=True)
            continue
        can_goi.append((so_trang, duong_dan_tam, khoa))

    if not can_goi:
        if ket_qua:
            logger.info("OCR: cả %d trang đều lấy lại được từ cache.", len(ket_qua))
        return ket_qua

    so_worker = max(1, min(tai_nguyen_gpu.so_worker_vision(), len(can_goi)))
    logger.info(
        "OCR '%s': %d trang cần đọc lại bằng model vision (%d trang lấy từ cache), %d luồng.",
        ten_file, len(can_goi), len(ket_qua), so_worker,
    )

    def _chay(viec):
        so_trang, duong_dan_tam, khoa = viec
        try:
            text = ocr_trang_pdf(client, str(duong_dan_tam))
        except Exception as loi:  # noqa: BLE001
            logger.warning("Không OCR được trang %d của '%s': %s", so_trang, ten_file, loi)
            text = ""
        finally:
            duong_dan_tam.unlink(missing_ok=True)
        if text and khoa:
            bo_nho_dem.kho_ocr.luu_text(khoa, text)
        return so_trang, text

    with do_thoi_gian.do("ocr_goi_model"):
        if so_worker == 1:
            cac_ket_qua = [_chay(v) for v in can_goi]
        else:
            with ThreadPoolExecutor(max_workers=so_worker) as bo_chay:
                cac_ket_qua = list(bo_chay.map(_chay, can_goi))

    so_doc_duoc = 0
    for so_trang, text in cac_ket_qua:
        if text:
            ket_qua[so_trang] = text
            so_doc_duoc += 1
    logger.info(
        "Đã OCR lại %d/%d trang của '%s'.", so_doc_duoc, len(can_goi), ten_file,
    )
    return ket_qua


def _la_bang_that(bang: List[List]) -> bool:
    """Lọc "bảng" do thuật toán dò nhầm ra khỏi bảng thật."""
    cac_hang = [h for h in bang if h and any(o is not None and str(o).strip() for o in h)]
    if len(cac_hang) < 2:
        return False
    so_cot_co_chu = max(
        sum(1 for o in h if o is not None and str(o).strip()) for h in cac_hang
    )
    return so_cot_co_chu >= 2


def _khoi_bang(bang: List[List]) -> str:
    """Bọc 1 bảng trong cặp mốc để chunking nhận ra và giữ nguyên khối."""
    if not _la_bang_that(bang):
        return ""
    markdown = _bang_sang_markdown(bang)
    return f"\n\n{MOC_BANG_MO}\n{markdown}\n{MOC_BANG_DONG}\n" if markdown else ""


def _text_pdf_khong_ke_bang(
    trang, cac_bang, ten_file: str = "", hieu_chinh: Optional[HieuChinhXTolerance] = None
) -> str:
    """Lấy văn xuôi của trang, ĐÃ LOẠI vùng chiếm bởi bảng."""
    so_trang = getattr(trang, "page_number", "?")
    try:
        vung = trang
        for bang in cac_bang:
            vung = vung.outside_bbox(bang.bbox)
        return _trich_text_thich_ung(vung, ten_file, so_trang, hieu_chinh)
    except (ValueError, TypeError) as loi:
        logger.warning(
            "Không loại được vùng bảng ở trang %s (%s) - đọc cả trang, bảng có thể bị lặp.",
            so_trang, type(loi).__name__,
        )
        return _trich_text_thich_ung(trang, ten_file, so_trang, hieu_chinh)


def _cac_cot_cua_trang(trang) -> List[tuple]:
    """Trả về danh sách khoảng (x_trái, x_phải) của từng cột; rỗng nếu trang MỘT cột."""
    try:
        cac_tu = trang.extract_words()
    except Exception:  # noqa: BLE001
        return []
    rong = float(trang.width or 0)
    if len(cac_tu) < config.SO_TU_TOI_THIEU_DE_DO_COT or rong <= 0:
        return []

    so_o = config.SO_O_DO_COT
    co_chu = [False] * so_o
    for t in cac_tu:
        dau = max(0, min(so_o - 1, int(t["x0"] / rong * so_o)))
        cuoi = max(0, min(so_o - 1, int(t["x1"] / rong * so_o)))
        for i in range(dau, cuoi + 1):
            co_chu[i] = True

    o_co_chu = [i for i, x in enumerate(co_chu) if x]
    if not o_co_chu or o_co_chu[-1] - o_co_chu[0] < 12:
        return []

    cac_ranh, dau = [], None
    for i in range(o_co_chu[0] + 1, o_co_chu[-1]):
        if not co_chu[i]:
            dau = i if dau is None else dau
        else:
            if dau is not None and i - dau >= config.SO_O_RANH_TOI_THIEU:
                cac_ranh.append((dau + i) / 2 * rong / so_o)
            dau = None

    moc = []
    for giua in cac_ranh:
        ben_trai = sum(1 for t in cac_tu if t["x1"] <= giua)
        ben_phai = sum(1 for t in cac_tu if t["x0"] >= giua)
        if min(ben_trai, ben_phai) >= config.TY_LE_TU_MOI_COT * len(cac_tu):
            moc.append(giua)
    if not moc:
        return []

    ranh_gioi = [0.0] + moc + [rong]
    return [(ranh_gioi[i], ranh_gioi[i + 1]) for i in range(len(ranh_gioi) - 1)]


def _text_theo_cot(
    trang, cac_cot: List[tuple], ten_file: str, so_trang: int,
    hieu_chinh: Optional[HieuChinhXTolerance] = None,
) -> str:
    """Đọc từng cột riêng rồi nối lại theo thứ tự trái -> phải (thứ tự đọc của người)."""
    cac_phan = []
    for x0, x1 in cac_cot:
        try:
            vung = trang.crop((x0, trang.bbox[1], x1, trang.bbox[3]))
        except (ValueError, TypeError):
            continue
        phan = _trich_text_thich_ung(vung, ten_file, so_trang, hieu_chinh).strip()
        if phan:
            cac_phan.append(phan)
    return "\n\n".join(cac_phan)


def _doc_mot_trang_pdf(trang, ten_file: str, so_trang: int, hieu_chinh) -> str:
    """Toàn bộ việc đọc TEXT của một trang PDF, gói lại thành đúng MỘT lượt."""
    cac_bang = []
    for bang in trang.find_tables():
        cac_hang = bang.extract()
        if _la_bang_that(cac_hang):
            cac_bang.append((bang, cac_hang))

    if cac_bang:
        noidung = _text_pdf_khong_ke_bang(
            trang, [b for b, _ in cac_bang], ten_file, hieu_chinh
        )
        return noidung + "".join(_khoi_bang(cac_hang) for _, cac_hang in cac_bang)

    cac_cot = _cac_cot_cua_trang(trang) if config.BAT_DOC_THEO_COT else []
    if cac_cot:
        logger.info(
            "Trang %d của '%s' có bố cục %d cột - đọc từng cột riêng.",
            so_trang, ten_file, len(cac_cot),
        )
        return _text_theo_cot(trang, cac_cot, ten_file, so_trang, hieu_chinh)
    return _trich_text_thich_ung(trang, ten_file, so_trang, hieu_chinh)


def doc_pdf(duong_dan: Path) -> List[Dict]:
    """Đọc từng trang PDF bằng pdfplumber, trả về list dict {nguon, trang, noidung}."""
    ten_file = duong_dan.name
    hieu_chinh = HieuChinhXTolerance()
    bam_tai_lieu = (
        bo_nho_dem.bam_file(duong_dan) if config.BAT_CACHE_INGESTION else ""
    )
    cac_trang_tho: List[Dict] = []
    ung_vien_anh: Dict[int, list] = {}

    with pdfplumber.open(duong_dan) as pdf:
        for so_trang, trang in enumerate(pdf.pages, start=1):
            with do_thoi_gian.do("pdf_doc_text_trang"):
                noidung = _doc_mot_trang_pdf(trang, ten_file, so_trang, hieu_chinh)

            can_ocr = config.BAT_OCR_DU_PHONG and trang_can_ocr(noidung, len(trang.images))

            if config.BAT_NHAN_DIEN_TIEU_DE:
                with do_thoi_gian.do("pdf_nhan_dien_tieu_de"):
                    cac_tieu_de = _phat_hien_tieu_de_pdf(trang)
                for dong_tieu_de in cac_tieu_de:
                    noidung = noidung.replace(dong_tieu_de, _danh_dau_tieu_de(dong_tieu_de), 1)

            if config.BAT_TRICH_ANH:
                cac_ung_vien = ung_vien_anh_trang(trang)
                if cac_ung_vien:
                    ung_vien_anh[so_trang] = cac_ung_vien

            cac_trang_tho.append(
                {"so_trang": so_trang, "noidung": noidung, "can_ocr": can_ocr}
            )
            trang.flush_cache()

        ket_qua_ocr = _ocr_cac_trang(
            pdf, [t["so_trang"] for t in cac_trang_tho if t["can_ocr"]], ten_file, bam_tai_lieu
        )

        ket_qua: List[Dict] = []
        cac_trang_ocr_ra_chu = set()
        noi_dung_theo_trang: Dict[int, str] = {}
        for tho in cac_trang_tho:
            so_trang = tho["so_trang"]
            noidung = tho["noidung"]
            noidung_ocr = ket_qua_ocr.get(so_trang, "")
            if noidung_ocr:
                noidung = noidung_ocr
                if len(noidung_ocr.split()) >= config.SO_TU_TOI_THIEU_TRANG_CO_CHU:
                    cac_trang_ocr_ra_chu.add(so_trang)

            noidung = _don_dep_watermark(noidung)
            noidung = _chuan_hoa_nfc(noidung).strip()
            noi_dung_theo_trang[so_trang] = noidung
            if not noidung:
                logger.warning(
                    "Trang %d của '%s' không có text (có thể là ảnh scan) - bỏ qua.",
                    so_trang, ten_file,
                )
                continue
            if _la_trang_muc_luc(noidung):
                logger.warning(
                    "Trang %d của '%s' có vẻ là trang Mục lục - bỏ qua (không phải nội "
                    "dung thật, dễ gây nhiễu retrieval).",
                    so_trang, ten_file,
                )
                continue
            ket_qua.append({"nguon": ten_file, "trang": so_trang, "noidung": noidung})

        if config.BAT_TRICH_ANH and ung_vien_anh:
            with do_thoi_gian.do("pdf_trich_anh"):
                cac_ban_ghi_anh, so_anh_toan_trang = [], 0
                for so_trang, cac_ung_vien in ung_vien_anh.items():
                    la_trang_scan_chu = so_trang in cac_trang_ocr_ra_chu
                    cac_bbox = []
                    for bbox, phu_ca_trang in cac_ung_vien:
                        if la_trang_scan_chu and phu_ca_trang:
                            so_anh_toan_trang += 1
                            continue
                        cac_bbox.append(bbox)
                    if not cac_bbox:
                        continue
                    cac_ban_ghi_anh.extend(
                        luu_anh_trang_pdf(
                            ten_file, pdf.pages[so_trang - 1], so_trang, cac_bbox,
                            noi_dung_theo_trang.get(so_trang, "").split("\n"),
                        )
                    )
                    pdf.pages[so_trang - 1].flush_cache()
                if so_anh_toan_trang:
                    logger.info(
                        "Bỏ qua %d ảnh chụp cả trang ở '%s' (PDF scan - nội dung của chúng "
                        "là chữ, đã được OCR đọc ra).", so_anh_toan_trang, ten_file,
                    )
                ket_qua.extend(loc_anh_lap_lai(cac_ban_ghi_anh, ten_file))
    return ket_qua


def duyet_shape(cac_shape) -> Iterator:
    """Duyệt shape của slide, ĐỆ QUY vào các nhóm (group shape)."""
    for shape in cac_shape:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from duyet_shape(shape.shapes)
        else:
            yield shape


def _trich_text_shape(shape) -> List[str]:
    """Lấy text từ 1 shape trong slide: text_frame và table (nếu có)."""
    cac_doan = []
    if shape.has_text_frame:
        for doan_van in shape.text_frame.paragraphs:
            text = "".join(run.text for run in doan_van.runs)
            if text.strip():
                cac_doan.append(text)
    if shape.has_table:
        cac_doan.append(
            _khoi_bang([_o_bang_khong_lap(hang) for hang in shape.table.rows])
        )
    return cac_doan


def doc_pptx(duong_dan: Path) -> List[Dict]:
    """Đọc từng slide PPTX bằng python-pptx, trả về list dict {nguon, trang, noidung}."""
    ket_qua = []
    trinh_chieu = Presentation(duong_dan)
    for so_slide, slide in enumerate(trinh_chieu.slides, start=1):
        shape_tieu_de = slide.shapes.title if slide.shapes.title is not None else None
        cac_doan = []
        id_tieu_de = None
        if config.BAT_NHAN_DIEN_TIEU_DE and shape_tieu_de is not None:
            van_ban_tieu_de = shape_tieu_de.text_frame.text.strip()
            if van_ban_tieu_de:
                cac_doan.append(_danh_dau_tieu_de(van_ban_tieu_de))
                id_tieu_de = shape_tieu_de.shape_id
        for shape in duyet_shape(slide.shapes):
            if id_tieu_de is not None and shape.shape_id == id_tieu_de:
                continue
            cac_doan.extend(_trich_text_shape(shape))
        noidung = _don_dep_watermark("\n".join(cac_doan))
        noidung = _chuan_hoa_nfc(noidung).strip()
        if not noidung:
            logger.warning(
                "Slide %d của '%s' không có text (có thể chỉ chứa ảnh) - bỏ qua.",
                so_slide,
                duong_dan.name,
            )
            continue
        if _la_trang_muc_luc(noidung):
            logger.warning(
                "Slide %d của '%s' có vẻ là slide Mục lục - bỏ qua (không phải nội dung "
                "thật, dễ gây nhiễu retrieval).",
                so_slide,
                duong_dan.name,
            )
            continue
        ket_qua.append({"nguon": duong_dan.name, "trang": so_slide, "noidung": noidung})
    if config.BAT_TRICH_ANH:
        with do_thoi_gian.do("pptx_trich_anh"):
            ket_qua.extend(
                loc_anh_lap_lai(trich_anh_pptx(duong_dan, trinh_chieu), duong_dan.name)
            )
    return ket_qua


_NS_WORD = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def _text_trong_text_box(doan_van) -> str:
    """Lấy chữ nằm trong TEXT BOX của một đoạn văn Word (python-docx không thấy được)."""
    try:
        cac_o = doan_van._element.findall(".//w:txbxContent//w:t", _NS_WORD)
    except Exception:  # noqa: BLE001
        return ""
    cac_chu = [o.text for o in cac_o if o.text and o.text.strip()]
    return (" ".join(cac_chu) + "\n") if cac_chu else ""


def doc_docx(duong_dan: Path) -> List[Dict]:
    """Đọc file DOCX bằng python-docx, tách thành các "trang" theo dấu ngắt trang cứng
    (Insert > Page Break trong Word)."""
    document = Document(duong_dan)
    cac_trang_tho: List[str] = [""]
    for phan in document.iter_inner_content():
        if isinstance(phan, BangDocx):
            cac_trang_tho[-1] += _khoi_bang([_o_bang_khong_lap(hang) for hang in phan.rows])
            continue
        if not isinstance(phan, DoanVanDocx):
            continue
        if phan.text.strip():
            ten_style = (phan.style.name or "") if phan.style is not None else ""
            if config.BAT_NHAN_DIEN_TIEU_DE and ten_style.startswith("Heading"):
                cap = int(ten_style.split()[-1]) if ten_style.split()[-1].isdigit() else 1
                cac_trang_tho[-1] += _danh_dau_tieu_de(phan.text, cap=min(cap, 2))
            else:
                cac_trang_tho[-1] += phan.text + "\n"
        cac_trang_tho[-1] += _text_trong_text_box(phan)
        if any(run._element.xpath(".//w:br[@w:type='page']") for run in phan.runs):
            cac_trang_tho.append("")

    ket_qua = []
    for so_trang, noidung_tho in enumerate(cac_trang_tho, start=1):
        noidung = _don_dep_watermark(noidung_tho)
        noidung = _chuan_hoa_nfc(noidung).strip()
        if not noidung:
            logger.warning(
                "Trang %d của '%s' không có text - bỏ qua.", so_trang, duong_dan.name
            )
            continue
        if _la_trang_muc_luc(noidung):
            logger.warning(
                "Trang %d của '%s' có vẻ là trang Mục lục - bỏ qua (không phải nội dung "
                "thật, dễ gây nhiễu retrieval).",
                so_trang,
                duong_dan.name,
            )
            continue
        ket_qua.append({"nguon": duong_dan.name, "trang": so_trang, "noidung": noidung})
    if config.BAT_TRICH_ANH:
        with do_thoi_gian.do("docx_trich_anh"):
            ket_qua.extend(
                loc_anh_lap_lai(trich_anh_docx(duong_dan, document), duong_dan.name)
            )
    return ket_qua


def doc_tai_lieu(duong_dan: Path) -> List[Dict]:
    """Đọc 1 file, tự chọn hàm đọc phù hợp theo phần đuôi file."""
    duoi = duong_dan.suffix.lower()
    if duoi == ".pdf":
        return doc_pdf(duong_dan)
    if duoi == ".pptx":
        return doc_pptx(duong_dan)
    if duoi == ".docx":
        return doc_docx(duong_dan)
    raise ValueError(f"Định dạng không hỗ trợ: '{duoi}' (chỉ hỗ trợ {CAC_DUOI_HO_TRO})")


def doc_tai_lieu_hoan_chinh(duong_dan: Path) -> List[Dict]:
    """Đọc 1 file và làm TRỌN VẸN mọi bước sinh nội dung: đọc + chú thích ảnh + bỏ ảnh rỗng."""
    ket_qua = doc_tai_lieu(duong_dan)
    if config.BAT_TRICH_ANH and config.BAT_CHU_THICH_ANH:
        cac_anh = [m for m in ket_qua if m.get("loai_noi_dung") == "anh"]
        if cac_anh:
            logger.info("'%s': %d ảnh cần chú thích.", duong_dan.name, len(cac_anh))
            bo_sung_chu_thich_vision(cac_anh)
    return _bo_ban_ghi_anh_rong(ket_qua)


def _cache_con_du_anh(cac_trang: List[Dict]) -> bool:
    """Mọi file ảnh mà bản ghi trong cache trỏ tới còn tồn tại không?"""
    return all(
        Path(m["duong_dan_anh"]).exists()
        for m in cac_trang
        if m.get("loai_noi_dung") == "anh" and m.get("duong_dan_anh")
    )


def doc_tai_lieu_co_cache(duong_dan: Path) -> List[Dict]:
    """doc_tai_lieu_hoan_chinh() nhưng lấy lại kết quả cũ khi nội dung file không đổi."""
    if not config.BAT_CACHE_INGESTION:
        return doc_tai_lieu_hoan_chinh(duong_dan)

    khoa = bo_nho_dem.khoa_tai_lieu(duong_dan)
    da_co = bo_nho_dem.kho_tai_lieu.lay_json(khoa)
    if da_co is not None and _cache_con_du_anh(da_co):
        logger.info(
            "'%s': lấy lại %d bản ghi từ cache (nội dung file không đổi).",
            duong_dan.name, len(da_co),
        )
        do_thoi_gian.ghi_nhan("tai_lieu_trung_cache", 0.0)
        return da_co

    with do_thoi_gian.do("tai_lieu_doc_moi"):
        ket_qua = doc_tai_lieu_hoan_chinh(duong_dan)
    bo_nho_dem.kho_tai_lieu.luu_json(khoa, ket_qua)
    return ket_qua


def _bo_ban_ghi_anh_rong(cac_trang: List[Dict]) -> List[Dict]:
    """Bỏ những bản ghi ảnh rốt cuộc KHÔNG có chú thích nào."""
    giu, bo = [], 0
    for m in cac_trang:
        if m.get("loai_noi_dung") == "anh" and m["noidung"].replace(MOC_ANH, "").strip() == "":
            bo += 1
            continue
        giu.append(m)
    if bo:
        logger.warning(
            "Bỏ %d ảnh không có chú thích nào (không có chữ lân cận, và model vision không "
            "chạy hoặc đang tắt) - chúng chỉ chứa mốc '%s' nên vô dụng cho tra cứu.",
            bo, MOC_ANH,
        )
    return giu


def _canh_bao_tai_lieu_khong_doc_duoc(cac_trang: List[Dict], cac_duong_dan: List[Path]) -> None:
    """Cảnh báo to khi một tài liệu gần như không đọc được chữ nào."""
    theo_nguon: Dict[str, int] = {}
    for m in cac_trang:
        if m.get("loai_noi_dung") != "anh":
            theo_nguon[m["nguon"]] = theo_nguon.get(m["nguon"], 0) + len(m["noidung"])

    for duong_dan in cac_duong_dan:
        so_ky_tu = theo_nguon.get(duong_dan.name, 0)
        if so_ky_tu >= config.SO_KY_TU_TOI_THIEU_MOT_TAI_LIEU:
            continue
        logger.error(
            "KHÔNG ĐỌC ĐƯỢC NỘI DUNG từ '%s' (chỉ %d ký tự văn bản). Gần như chắc chắn đây "
            "là PDF SCAN (ảnh chụp trang, không có lớp text). Hệ thống sẽ không trả lời "
            "đúng được về tài liệu này. Cách xử lý: bật OCR bằng BAT_OCR_DU_PHONG=1 và "
            "'ollama pull %s', rồi bấm Đọc tài liệu.",
            duong_dan.name, so_ky_tu, config.VISION_MODEL_NAME,
        )


def cac_file_tai_lieu(thu_muc: Path) -> List[Path]:
    """Danh sách file tài liệu hỗ trợ được trong thư mục, thứ tự ổn định."""
    return [d for d in sorted(thu_muc.glob("*")) if d.suffix.lower() in CAC_DUOI_HO_TRO]


def doc_nhieu_file(cac_duong_dan: List[Path]) -> List[Dict]:
    """Đọc một DANH SÁCH file cụ thể (có cache), gom lỗi lại thay vì để một file hỏng
    làm sập cả lần build."""
    ket_qua = []
    cac_file_loi = []
    for thu_tu, duong_dan in enumerate(cac_duong_dan, start=1):
        logger.info("[%d/%d] Đang xử lý '%s'...", thu_tu, len(cac_duong_dan), duong_dan.name)
        try:
            ket_qua.extend(doc_tai_lieu_co_cache(duong_dan))
        except Exception as loi:  # noqa: BLE001
            cac_file_loi.append((duong_dan.name, f"{type(loi).__name__}: {loi}"))
            logger.error(
                "KHÔNG ĐỌC ĐƯỢC FILE '%s' (%s: %s) - bỏ qua file này và tiếp tục với các "
                "tài liệu còn lại. File có thể đặt mật khẩu, tải về dở dang, hoặc sai định "
                "dạng so với phần đuôi.",
                duong_dan.name, type(loi).__name__, loi,
            )

    if cac_file_loi:
        logger.error(
            "TỔNG KẾT: %d/%d tài liệu KHÔNG vào được index: %s",
            len(cac_file_loi),
            len(cac_file_loi) + len({m["nguon"] for m in ket_qua}),
            ", ".join(ten for ten, _ in cac_file_loi),
        )

    _canh_bao_tai_lieu_khong_doc_duoc(ket_qua, cac_duong_dan)
    return ket_qua


def doc_thu_muc(thu_muc: Path) -> List[Dict]:
    """Đọc toàn bộ file PDF/PPTX/DOCX trong 1 thư mục, dùng cho luồng Ingestion."""
    return doc_nhieu_file(cac_file_tai_lieu(thu_muc))
