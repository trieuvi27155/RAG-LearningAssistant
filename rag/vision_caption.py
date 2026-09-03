"""Mô tả nội dung hình ảnh bằng model vision chạy local qua Ollama.

VÌ SAO CẦN, khi đã có chú thích lân cận (rag/image_extractor.py):
Chú thích chỉ cho biết hình đó TÊN là gì ("Hình 3: Sơ đồ bộ máy nhà nước"), không cho biết
BÊN TRONG hình có gì. Câu hỏi "cơ quan nào trực thuộc Chính phủ?" mà đáp án chỉ nằm trong
các ô của sơ đồ thì chú thích không giúp được - không có từ nào khớp. Model vision đọc hình
rồi viết ra thành lời, biến nội dung trực quan thành text tìm kiếm được như mọi đoạn khác.

VÌ SAO MẶC ĐỊNH TẮT (config.BAT_CHU_THICH_ANH):
Mỗi hình tốn một lượt gọi model riêng - tài liệu 200 hình là 200 lượt, đo thực tế ~3-6 giây
mỗi hình trên CPU, tức có thể thêm cả chục phút vào một lần Build Index. Tài liệu thuần văn
bản không được lợi gì từ chi phí đó. Nên đây là thứ người dùng BẬT khi biết tài liệu của
mình nhiều sơ đồ/biểu đồ, không phải thứ bắt mọi người trả giá mặc định.

GIẢM CẤP THAY VÌ HỎNG: nếu model chưa được pull, hệ thống ghi cảnh báo và bỏ qua bước chú
thích - phần còn lại của luồng Ingestion vẫn chạy bình thường. Người dùng bật nhầm một
tuỳ chọn không được phép làm hỏng cả lần build index.
"""

import logging
import re
from typing import Optional

import ollama

import config

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

    so_thanh_cong = 0
    for i, ban_ghi in enumerate(cac_ban_ghi_anh, start=1):
        duong_dan = ban_ghi.get("duong_dan_anh")
        if not duong_dan:
            continue
        logger.info("Đang chú thích ảnh %d/%d...", i, len(cac_ban_ghi_anh))
        mo_ta = chu_thich_anh(client, duong_dan)
        if mo_ta:
            ban_ghi["noidung"] = f"{ban_ghi['noidung']}\n{mo_ta}".strip()
            ban_ghi["co_chu_thich_vision"] = True
            so_thanh_cong += 1
    logger.info("Đã chú thích %d/%d ảnh bằng model vision.", so_thanh_cong, len(cac_ban_ghi_anh))
    return so_thanh_cong
