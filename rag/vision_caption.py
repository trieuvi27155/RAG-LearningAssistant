"""Mô tả nội dung hình ảnh bằng model vision chạy local qua Ollama."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import ollama

import config
from rag import bo_nho_dem, do_thoi_gian, tai_nguyen_gpu

logger = logging.getLogger(__name__)

PROMPT_CHU_THICH_VI = """Mô tả nội dung hình này bằng tiếng Việt, phục vụ mục đích tra cứu.

YÊU CẦU:
1. Ghi lại TOÀN BỘ chữ xuất hiện trong hình (nhãn, tiêu đề, số liệu, chú giải) - giữ nguyên văn.
2. Nếu là sơ đồ/lưu đồ: nêu rõ các thành phần và quan hệ giữa chúng (cái nào thuộc cái nào, mũi tên đi từ đâu tới đâu).
3. Nếu là biểu đồ: nêu loại biểu đồ, tên các trục, và các giá trị/xu hướng chính.
4. Nếu là bảng chụp thành ảnh: đọc lại nội dung theo từng hàng.
5. Chỉ mô tả những gì THẬT SỰ nhìn thấy. Không suy đoán, không thêm kiến thức bên ngoài.
6. Viết liền mạch, không mở đầu kiểu "Hình này cho thấy...", đi thẳng vào nội dung."""


PROMPT_OCR_TRANG = """Transcribe ALL text in this document page image, verbatim and in order.

CRITICAL RULES:
1. Keep the ORIGINAL LANGUAGE of the page. Do NOT translate anything. If the page is in
   English, output English. If Vietnamese, output Vietnamese.
2. Include mathematical formulas, written with ordinary symbols.
3. Include figure labels and captions.
4. Do NOT summarise, do NOT comment, do NOT add anything. Only transcribe."""

_MAU_CID = re.compile(r"\(cid:\d+\)")


def trang_can_ocr(text_da_doc: str, so_anh_trong_trang: int) -> bool:
    """Trang này có cần đọc lại bằng OCR không?"""
    so_tu = len(_MAU_CID.sub("", text_da_doc).split())
    so_cid = len(_MAU_CID.findall(text_da_doc))
    if so_cid >= config.SO_CID_TOI_THIEU_DE_OCR and so_cid > so_tu * config.TY_LE_CID_DE_OCR:
        return True
    return so_tu < config.SO_TU_TOI_THIEU_TRANG_CO_CHU and so_anh_trong_trang > 0


def ocr_trang_pdf(client: ollama.Client, duong_dan_anh: str, ten_model: Optional[str] = None) -> str:
    """Đọc lại một trang PDF đã render thành ảnh, bằng model vision. Rỗng nếu thất bại."""
    ten_model = ten_model or config.VISION_MODEL_NAME
    try:
        phan_hoi = client.chat(
            model=ten_model,
            messages=[{"role": "user", "content": PROMPT_OCR_TRANG, "images": [duong_dan_anh]}],
            options={"temperature": 0, "num_predict": config.OCR_NUM_PREDICT},
        )
    except Exception as loi:
        logger.warning("OCR thất bại cho '%s': %s", duong_dan_anh, loi)
        return ""
    return (phan_hoi["message"]["content"] or "").strip()


def ten_model_khop(ten_co_san: str, ten_can: str) -> bool:
    """So tên model có/không có tag. Ollama trả về "qwen2.5vl:3b" nhưng người dùng có thể
    cấu hình "qwen2.5vl" - coi là khớp để không bắt họ nhớ đúng tag."""
    return ten_co_san == ten_can or ten_co_san.split(":")[0] == ten_can.split(":")[0]


def mo_hinh_vision_co_san(client: ollama.Client, ten_model: Optional[str] = None) -> bool:
    """Model vision đã được pull về máy chưa."""
    ten_model = ten_model or config.VISION_MODEL_NAME
    try:
        cac_model = client.list().models
    except Exception as loi:
        logger.warning("Không kiểm tra được danh sách model của Ollama (%s).", loi)
        return False
    return any(ten_model_khop(m.model or "", ten_model) for m in cac_model)


def chu_thich_anh(
    client: ollama.Client, duong_dan_anh: str, ten_model: Optional[str] = None
) -> str:
    """Gọi model vision mô tả 1 hình, trả về chuỗi mô tả (rỗng nếu thất bại)."""
    ten_model = ten_model or config.VISION_MODEL_NAME
    try:
        phan_hoi = client.chat(
            model=ten_model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_CHU_THICH_VI,
                    "images": [duong_dan_anh],
                }
            ],
            options={
                "temperature": 0,
                "num_predict": config.VISION_NUM_PREDICT,
            },
        )
    except Exception as loi:
        logger.warning("Không chú thích được ảnh '%s': %s", duong_dan_anh, loi)
        return ""
    return (phan_hoi["message"]["content"] or "").strip()


def bo_sung_chu_thich_vision(cac_ban_ghi_anh: list, client: Optional[ollama.Client] = None) -> int:
    """Thêm mô tả của model vision vào các bản ghi ảnh (sửa tại chỗ)."""
    if not cac_ban_ghi_anh or not config.BAT_CHU_THICH_ANH:
        return 0

    client = client or ollama.Client(host=config.OLLAMA_HOST)
    if not mo_hinh_vision_co_san(client):
        logger.warning(
            "BAT_CHU_THICH_ANH đang bật nhưng model vision '%s' chưa được pull về máy - "
            "bỏ qua bước chú thích ảnh. Chạy: ollama pull %s",
            config.VISION_MODEL_NAME,
            config.VISION_MODEL_NAME,
        )
        return 0

    theo_noi_dung: Dict[str, list] = {}
    for i, ban_ghi in enumerate(cac_ban_ghi_anh):
        duong_dan = ban_ghi.get("duong_dan_anh")
        if not duong_dan:
            continue
        khoa = ban_ghi.get("bam_anh") or bo_nho_dem.bam_chuoi(f"{i}|{duong_dan}")
        theo_noi_dung.setdefault(khoa, []).append(ban_ghi)

    if not theo_noi_dung:
        return 0

    mo_ta_theo_khoa: Dict[str, str] = {}
    can_goi: list = []
    for khoa, cac_ban_ghi in theo_noi_dung.items():
        duong_dan = Path(cac_ban_ghi[0]["duong_dan_anh"])
        khoa_cache = bo_nho_dem.khoa_vision(duong_dan) if config.BAT_CACHE_INGESTION else None
        da_co = bo_nho_dem.kho_vision.lay_text(khoa_cache) if khoa_cache else None
        if da_co is not None:
            mo_ta_theo_khoa[khoa] = da_co
        else:
            can_goi.append((khoa, str(duong_dan), khoa_cache))

    so_worker = max(1, min(tai_nguyen_gpu.so_worker_vision(), len(can_goi)))
    logger.info(
        "Chú thích ảnh: %d bản ghi -> %d ảnh khác nhau; %d lấy từ cache, %d cần gọi model "
        "(%d luồng).",
        len(cac_ban_ghi_anh), len(theo_noi_dung), len(mo_ta_theo_khoa), len(can_goi), so_worker,
    )

    if can_goi:
        with do_thoi_gian.do("vision_chu_thich_anh"):
            def _chay(viec):
                khoa, duong_dan, khoa_cache = viec
                mo_ta = chu_thich_anh(client, duong_dan)
                if mo_ta and khoa_cache:
                    bo_nho_dem.kho_vision.luu_text(khoa_cache, mo_ta)
                return khoa, mo_ta

            if so_worker == 1:
                for viec in can_goi:
                    khoa, mo_ta = _chay(viec)
                    mo_ta_theo_khoa[khoa] = mo_ta
            else:
                with ThreadPoolExecutor(max_workers=so_worker) as bo_chay:
                    for khoa, mo_ta in bo_chay.map(_chay, can_goi):
                        mo_ta_theo_khoa[khoa] = mo_ta

    so_thanh_cong = 0
    for khoa, cac_ban_ghi in theo_noi_dung.items():
        mo_ta = mo_ta_theo_khoa.get(khoa)
        if not mo_ta:
            continue
        for ban_ghi in cac_ban_ghi:
            ban_ghi["noidung"] = f"{ban_ghi['noidung']}\n{mo_ta}".strip()
            ban_ghi["co_chu_thich_vision"] = True
            so_thanh_cong += 1
    logger.info("Đã chú thích %d/%d ảnh bằng model vision.", so_thanh_cong, len(cac_ban_ghi_anh))
    return so_thanh_cong
