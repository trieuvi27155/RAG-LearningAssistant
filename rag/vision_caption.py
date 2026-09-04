"""Mô tả nội dung hình ảnh bằng model vision chạy local qua Ollama.

VÌ SAO CẦN, khi đã có chú thích lân cận (rag/image_extractor.py):
Chú thích chỉ cho biết hình đó TÊN là gì ("Hình 3: Sơ đồ bộ máy nhà nước"), không cho biết
BÊN TRONG hình có gì. Câu hỏi "cơ quan nào trực thuộc Chính phủ?" mà đáp án chỉ nằm trong
các ô của sơ đồ thì chú thích không giúp được - không có từ nào khớp. Model vision đọc hình
rồi viết ra thành lời, biến nội dung trực quan thành text tìm kiếm được như mọi đoạn khác.

CÁI GIÁ, VÀ BỐN CÁCH ĐÃ LÀM ĐỂ KHÔNG PHẢI TRẢ NÓ HAI LẦN (config.BAT_CHU_THICH_ANH):
Mỗi hình tốn một lượt gọi model riêng - tài liệu 200 hình là 200 lượt, đo thực tế ~1,9 giây
mỗi hình (3-6 giây trên máy chỉ có CPU), tức có thể thêm cả chục phút vào một lần Build
Index. Đó là lý do bốn chốt sau đều nằm TRƯỚC lượt gọi model, không phải sau:

  1. LỌC ẢNH (rag/image_extractor.py): icon, logo góc trang, dải trang trí và hình lặp lại
     kiểu watermark bị loại trước khi tới đây - chúng không mang nội dung tra cứu được.
  2. GỘP ẢNH TRÙNG NỘI DUNG: một hình dùng lại ở 20 slide chỉ tốn đúng 1 lượt gọi.
  3. CACHE THEO BĂM NỘI DUNG ẢNH (rag/bo_nho_dem.py): ảnh đã chú thích ở lần build trước thì
     không gọi lại, kể cả khi tài liệu đổi tên hay được đọc lại vì lý do khác.
  4. GỌI SONG SONG (config.SO_WORKER_VISION): phần còn lại chạy nhiều luồng, vì toàn bộ thời
     gian chờ ở đây là chờ mạng chứ không phải chờ CPU của chính mình.

Nhờ bốn chốt đó mà tuỳ chọn này mặc định BẬT: chi phí thật chỉ còn rơi vào những hình MỚI và
THẬT SỰ có nội dung, thay vì rơi vào mọi hình ở mọi lần build.

GIẢM CẤP THAY VÌ HỎNG: nếu model chưa được pull, hệ thống ghi cảnh báo và bỏ qua bước chú
thích - phần còn lại của luồng Ingestion vẫn chạy bình thường. Người dùng bật nhầm một
tuỳ chọn không được phép làm hỏng cả lần build index.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import ollama

import config
from rag import bo_nho_dem, do_thoi_gian

logger = logging.getLogger(__name__)

# Prompt cố ý yêu cầu LIỆT KÊ chữ và quan hệ trong hình, không phải "tả cho đẹp": mục đích
# duy nhất của đoạn mô tả này là để TÌM KIẾM khớp được, nên nhãn trong sơ đồ, số trên biểu
# đồ, tên các ô và mũi tên nối chúng mới là thứ có giá trị - văn phong không quan trọng.
PROMPT_CHU_THICH_VI = """Mô tả nội dung hình này bằng tiếng Việt, phục vụ mục đích tra cứu.

YÊU CẦU:
1. Ghi lại TOÀN BỘ chữ xuất hiện trong hình (nhãn, tiêu đề, số liệu, chú giải) - giữ nguyên văn.
2. Nếu là sơ đồ/lưu đồ: nêu rõ các thành phần và quan hệ giữa chúng (cái nào thuộc cái nào, mũi tên đi từ đâu tới đâu).
3. Nếu là biểu đồ: nêu loại biểu đồ, tên các trục, và các giá trị/xu hướng chính.
4. Nếu là bảng chụp thành ảnh: đọc lại nội dung theo từng hàng.
5. Chỉ mô tả những gì THẬT SỰ nhìn thấy. Không suy đoán, không thêm kiến thức bên ngoài.
6. Viết liền mạch, không mở đầu kiểu "Hình này cho thấy...", đi thẳng vào nội dung."""


# Prompt cho OCR DỰ PHÒNG - khác hẳn prompt chú thích ảnh ở trên, và khác ở 2 điểm sống còn:
#   - Viết bằng TIẾNG ANH và cấm dịch. Bản đầu viết prompt tiếng Việt, model liền dịch cả
#     trang sách tiếng Anh sang tiếng Việt, dịch sai bét ("LINEAR MODELS FOR REGRESSION" ->
#     "LINHỆ MÔIẾN TRÊN REGRESSION") - tức phá hỏng nội dung thay vì cứu nó.
#   - Yêu cầu CHÉP LẠI nguyên văn, không mô tả. Chú thích ảnh cần mô tả; OCR cần bản sao.
PROMPT_OCR_TRANG = """Transcribe ALL text in this document page image, verbatim and in order.

CRITICAL RULES:
1. Keep the ORIGINAL LANGUAGE of the page. Do NOT translate anything. If the page is in
   English, output English. If Vietnamese, output Vietnamese.
2. Include mathematical formulas, written with ordinary symbols.
3. Include figure labels and captions.
4. Do NOT summarise, do NOT comment, do NOT add anything. Only transcribe."""

_MAU_CID = re.compile(r"\(cid:\d+\)")


def trang_can_ocr(text_da_doc: str, so_anh_trong_trang: int) -> bool:
    """Trang này có cần đọc lại bằng OCR không?

    Hai dấu hiệu, chỉ cần khớp một:
    1. Nhiều mã `(cid:NN)` so với số từ đọc được -> PDF nhúng font không kèm bảng ánh xạ
       ToUnicode, thường gặp ở font toán học. Chữ vẫn hiển thị bình thường khi mở file,
       nhưng trích xuất ra thì thành mã vô nghĩa.
    2. Gần như không có chữ nhưng trang lại có ảnh -> nhiều khả năng là trang scan.

    Cả hai đều là trường hợp mà bản thân việc đọc file đã thất bại, nên đọc lại bằng mắt
    (model vision) là cách duy nhất còn lại.
    """
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
    """Model vision đã được pull về máy chưa.

    Kiểm tra TRƯỚC khi build index thay vì để lỗi bung ra giữa chừng: một lần build có thể
    chạy vài phút, hỏng ở hình thứ 150 nghĩa là mất trắng toàn bộ công sức trước đó.
    """
    ten_model = ten_model or config.VISION_MODEL_NAME
    try:
        cac_model = client.list().models
    except Exception as loi:  # máy chủ Ollama chưa chạy, hoặc phiên bản client khác
        logger.warning("Không kiểm tra được danh sách model của Ollama (%s).", loi)
        return False
    return any(ten_model_khop(m.model or "", ten_model) for m in cac_model)


def chu_thich_anh(
    client: ollama.Client, duong_dan_anh: str, ten_model: Optional[str] = None
) -> str:
    """Gọi model vision mô tả 1 hình, trả về chuỗi mô tả (rỗng nếu thất bại).

    Truyền ĐƯỜNG DẪN file cho Ollama thay vì đọc sẵn ra bytes: client tự đọc và mã hoá
    base64, đỡ giữ cả ảnh trong RAM khi xử lý hàng trăm hình liên tiếp.

    Nuốt mọi lỗi và trả chuỗi rỗng (có ghi log): một hình hỏng/định dạng lạ không đáng làm
    sập cả lần build index - phần chú thích lân cận của hình đó vẫn còn dùng được.
    """
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
                "temperature": 0,  # mô tả phải ổn định, không cần sáng tạo
                "num_predict": config.VISION_NUM_PREDICT,
            },
        )
    except Exception as loi:
        logger.warning("Không chú thích được ảnh '%s': %s", duong_dan_anh, loi)
        return ""
    return (phan_hoi["message"]["content"] or "").strip()


def bo_sung_chu_thich_vision(cac_ban_ghi_anh: list, client: Optional[ollama.Client] = None) -> int:
    """Thêm mô tả của model vision vào các bản ghi ảnh (sửa tại chỗ).

    Mô tả được NỐI THÊM vào sau chú thích lân cận chứ không thay thế nó: hai nguồn thông tin
    bổ khuyết cho nhau - chú thích cho biết hình này được gọi tên là gì trong tài liệu (từ
    khoá người đọc sẽ dùng khi hỏi), còn model vision cho biết bên trong hình có gì.

    Trả về số ảnh đã chú thích thành công.

    BA TẦNG TIẾT KIỆM, xếp theo thứ tự chi phí giảm dần - và cả ba đều KHÔNG đổi kết quả,
    chỉ đổi số lượt gọi model:

      1. GỘP ẢNH TRÙNG NỘI DUNG. Các bản ghi được gom theo băm nội dung ảnh; mỗi nội dung
         chỉ gọi model MỘT lần rồi phát cùng một mô tả cho mọi bản ghi trùng. Hình dùng lại
         giữa các slide, ảnh chèn hai lần trong cùng một chương - rất phổ biến trong bài
         giảng - từ N lượt gọi còn đúng 1.
      2. CACHE TRÊN ĐĨA. Ảnh đã từng được chú thích ở lần build trước thì lấy lại kết quả cũ,
         kể cả khi tài liệu đã đổi tên hay được đọc lại vì một trang khác thay đổi.
      3. GỌI SONG SONG. Phần còn lại chạy trên config.SO_WORKER_VISION luồng. Đây là bước
         đắt nhất của cả luồng Ingestion (~1,9 giây mỗi ảnh theo benchmark của project) và
         cũng là bước NHÀN RỖI nhất về phía Python - toàn bộ thời gian đó là ngồi chờ Ollama
         trả lời qua HTTP.

    Thứ tự này quan trọng: gộp trước, tra cache sau, rồi mới gọi model. Làm ngược lại (gọi
    song song trước) chỉ khiến hệ thống chú thích cùng một cái logo trên 8 luồng cùng lúc.
    """
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

    # Gom theo NỘI DUNG ảnh. Khoá rỗng (không băm được) thì mỗi bản ghi tự thành một nhóm,
    # tức lùi về đúng hành vi tuần tự cũ cho riêng ảnh đó thay vì bị bỏ qua.
    theo_noi_dung: Dict[str, list] = {}
    for i, ban_ghi in enumerate(cac_ban_ghi_anh):
        duong_dan = ban_ghi.get("duong_dan_anh")
        if not duong_dan:
            continue
        khoa = ban_ghi.get("bam_anh") or bo_nho_dem.bam_chuoi(f"{i}|{duong_dan}")
        theo_noi_dung.setdefault(khoa, []).append(ban_ghi)

    if not theo_noi_dung:
        return 0

    # Tra cache trước khi gọi model. Khoá cache khác khoá gộp ở trên (nó còn gộp thêm tên
    # model và VISION_NUM_PREDICT) nên đổi model vision sẽ làm trượt cache đúng như phải thế.
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

    so_worker = max(1, min(config.SO_WORKER_VISION, len(can_goi)))
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
