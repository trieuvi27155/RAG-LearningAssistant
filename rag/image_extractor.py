"""Trích xuất hình ảnh từ tài liệu và gắn chúng với văn bản xung quanh.

Bài toán: nội dung nằm TRONG hình (sơ đồ, biểu đồ, ảnh chụp bảng) hoàn toàn vô hình với
hệ thống chỉ đọc lớp text. Người dùng hỏi "sơ đồ quy trình gồm những bước nào" thì dù tài
liệu có sơ đồ đó, hệ thống vẫn không tìm ra.

Có 2 mức xử lý, tách rời có chủ đích:

  Mức 1 (module này, luôn bật): trích ảnh ra file + lấy VĂN BẢN LÂN CẬN làm chú thích.
      Không cần model nào, gần như miễn phí. Đủ để tìm được hình qua caption kiểu
      "Hình 3: Bốn bước xử lý mượn tài liệu" - vốn là cách tài liệu thật hay đánh số hình.

  Mức 2 (rag/vision_caption.py, tuỳ chọn): cho model vision mô tả nội dung bên trong hình.
      Bắt được cả thứ không có trong caption (số liệu trên biểu đồ, nhãn trong sơ đồ),
      nhưng mỗi ảnh tốn một lượt gọi model nên chỉ bật khi thật sự cần.

Tách 2 mức để tài liệu thuần văn bản không phải trả bất kỳ chi phí nào, còn tài liệu nhiều
hình vẫn dùng được đầy đủ - đúng yêu cầu "hoạt động ổn định trên nhiều loại tài liệu".
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from docx.opc.constants import RELATIONSHIP_TYPE
from pptx.enum.shapes import MSO_SHAPE_TYPE

import config

logger = logging.getLogger(__name__)

MOC_ANH = "[HÌNH]"

# Ảnh quá nhỏ gần như luôn là logo, đường kẻ trang trí, bullet icon - không mang nội dung
# nhưng vẫn chiếm 1 vector và 1 lượt gọi model vision nếu bật. Lọc theo kích thước là cách
# rẻ và tổng quát hơn nhiều so với dò tên file hay vị trí.
KICH_THUOC_ANH_TOI_THIEU = 120  # điểm ảnh, áp cho cả chiều rộng lẫn chiều cao

# Số ký tự văn bản lân cận lấy làm chú thích cho ảnh.
DO_DAI_CHU_THICH_LAN_CAN = 400

# Dòng bắt đầu bằng "Hình 3:", "Figure 2.", "Biểu đồ 1 -"... gần như chắc chắn là chú thích
# của hình ngay cạnh. Ưu tiên những dòng này hơn văn bản lân cận chung chung.
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
    """Chọn văn bản mô tả hình từ các dòng lân cận.

    Ưu tiên dòng có dạng chú thích chuẩn ("Hình 3: ..."); nếu không có thì gộp văn bản lân
    cận làm ngữ cảnh thay thế. Lý do ưu tiên: dòng chú thích mô tả ĐÚNG hình đó, còn văn bản
    lân cận chỉ là "quanh đây nói về gì" - dùng được nhưng kém chính xác hơn.
    """
    cac_dong = [d.strip() for d in cac_dong if d and d.strip()]
    dong_chu_thich = [d for d in cac_dong if _MAU_DONG_CHU_THICH.match(d)]
    if dong_chu_thich:
        return " ".join(dong_chu_thich)[:DO_DAI_CHU_THICH_LAN_CAN]
    return " ".join(cac_dong)[:DO_DAI_CHU_THICH_LAN_CAN]


def _la_anh_cua_trang_chu(rong: float, cao: float, dien_tich_trang: float) -> bool:
    """Ảnh này có phải là ảnh chụp của một TRANG CHỮ (PDF scan) không?

    Chỉ được gọi cho những trang mà OCR ĐÃ ĐỌC RA CẢ MỘT TRANG CHỮ (xem doc_pdf). Nói cách
    khác, câu hỏi "trang này là chữ hay là hình" KHÔNG được trả lời bằng cách đoán qua kích
    thước ảnh, mà bằng kết quả đo: OCR lấy ra được bao nhiêu chữ từ chính trang đó.

    Vì sao phải làm vậy - hai lần thử trước đều sai theo hai kiểu khác nhau:
      - Chỉ so DIỆN TÍCH với ngưỡng 0.85: trượt ngay, vì ảnh scan của một giáo trình thật
        chỉ phủ 81.2% trang (sách có lề ~50pt mỗi bên).
      - Hạ ngưỡng xuống 0.6 kèm điều kiện "trang không có chữ": bắt được sách scan, nhưng
        nuốt luôn 5 ảnh nền của slide tiêu đề và bìa sách trong corpus cũ - những trang có
        rất ít chữ nhưng ảnh của chúng KHÔNG phải là ảnh chụp chữ.

    Điều phân biệt hai trường hợp đó không phải kích thước, mà là NỘI DUNG của ảnh: ảnh chụp
    trang sách chứa cả nghìn ký tự chữ, còn ảnh nền của slide tiêu đề thì không. Chỉ OCR mới
    biết được điều đó, nên quyết định phải đi sau OCR.

    Phần diện tích còn lại chỉ là chốt phụ: trên một trang scan vẫn có thể có thêm một hình
    nhỏ được nhúng riêng, và hình nhỏ đó thì vẫn nên giữ.
    """
    if dien_tich_trang <= 0:
        return False
    return (rong * cao) / dien_tich_trang >= config.TY_LE_DIEN_TICH_ANH_TOAN_TRANG


def _ban_ghi_anh(nguon: str, trang: int, duong_dan_anh: Path, chu_thich: str) -> Dict:
    """Một ảnh trở thành một "trang" riêng trong luồng dữ liệu.

    Dùng lại đúng schema {nguon, trang, noidung} của document_loader thay vì tạo luồng
    riêng: nhờ vậy ảnh đi qua chunking/embedding/vector_store/citation y hệt văn bản, không
    module nào phải biết đến khái niệm "ảnh". Chỉ 2 trường phụ được thêm để UI hiển thị
    được ảnh và để chunking biết không cắt nhỏ bản ghi này.

    Bản ghi ở đây vẫn được tạo kể cả khi chưa có chú thích nào: chú thích bằng model vision
    được thêm SAU, ở doc_thu_muc(), nên bỏ ảnh ngay tại đây sẽ giết luôn những ảnh sắp được
    model đọc nội dung. Việc loại bỏ ảnh rốt cuộc vẫn rỗng nằm ở
    document_loader._bo_ban_ghi_anh_rong(), chạy sau bước chú thích.
    """
    return {
        "nguon": nguon,
        "trang": trang,
        "noidung": f"{MOC_ANH} {chu_thich}".strip(),
        "loai_noi_dung": "anh",
        "duong_dan_anh": str(duong_dan_anh),
    }


def trich_anh_pdf(
    duong_dan: Path, pdf, cac_trang_ocr_ra_chu: Optional[Set[int]] = None
) -> List[Dict]:
    """Trích ảnh từng trang PDF.

    pdfplumber không giải mã được stream ảnh gốc, nên phải RENDER LẠI vùng chứa ảnh thành
    pixel (crop -> to_image). Đánh đổi: không giữ được file gốc nguyên vẹn, nhưng đủ tốt để
    model vision đọc và để hiển thị cho người dùng đối chiếu - đúng 2 mục đích đang cần.
    """
    ket_qua = []
    so_anh_toan_trang = 0
    for so_trang, trang in enumerate(pdf.pages, start=1):
        cac_dong = (trang.extract_text() or "").split("\n")
        dien_tich_trang = float(trang.width or 0) * float(trang.height or 0)
        la_trang_scan_chu = so_trang in (cac_trang_ocr_ra_chu or set())
        thu_tu = 0
        for anh in trang.images:
            rong = float(anh.get("width") or 0)
            cao = float(anh.get("height") or 0)
            if rong < KICH_THUOC_ANH_TOI_THIEU or cao < KICH_THUOC_ANH_TOI_THIEU:
                continue
            if la_trang_scan_chu and _la_anh_cua_trang_chu(rong, cao, dien_tich_trang):
                # Đây là ảnh chụp CẢ TRANG (PDF scan), không phải hình minh hoạ trong trang.
                # Nội dung thật của nó là CHỮ, và chữ đó đã được OCR lấy ra ở doc_pdf rồi -
                # trích thêm một "hình" nữa chỉ tạo một chunk rác cho mỗi trang, tốn thêm một
                # lượt model vision cho mỗi trang, và khiến trích dẫn trỏ vào "hình" thay vì
                # vào đoạn chữ mà câu trả lời thật sự dùng.
                so_anh_toan_trang += 1
                continue
            try:
                # Giới hạn bbox trong khung trang: ảnh tràn mép trang sẽ khiến crop() ném lỗi.
                bbox = (
                    max(anh["x0"], trang.bbox[0]), max(anh["top"], trang.bbox[1]),
                    min(anh["x1"], trang.bbox[2]), min(anh["bottom"], trang.bbox[3]),
                )
                pil = trang.crop(bbox).to_image(resolution=110).original
            except Exception as loi:  # noqa: BLE001 - pdfplumber ném nhiều loại lỗi khác nhau
                logger.warning("Bỏ qua 1 ảnh ở trang %d của '%s': %s",
                               so_trang, duong_dan.name, type(loi).__name__)
                continue
            thu_tu += 1
            dich = config.IMAGES_DIR / _ten_file_an_toan(duong_dan.name, so_trang, thu_tu, ".png")
            pil.save(dich)
            ket_qua.append(
                _ban_ghi_anh(duong_dan.name, so_trang, dich, _chon_chu_thich(cac_dong))
            )
    if so_anh_toan_trang:
        logger.info(
            "Bỏ qua %d ảnh chụp cả trang ở '%s' (PDF scan - nội dung của chúng là chữ, đã "
            "được OCR đọc ra).", so_anh_toan_trang, duong_dan.name,
        )
    return ket_qua


def trich_anh_pptx(duong_dan: Path, trinh_chieu) -> List[Dict]:
    """Trích ảnh từng slide, kể cả ảnh nằm trong group shape."""
    from rag.document_loader import duyet_shape  # nhập tại chỗ để tránh phụ thuộc vòng

    ket_qua = []
    for so_slide, slide in enumerate(trinh_chieu.slides, start=1):
        cac_shape = list(duyet_shape(slide.shapes))
        cac_dong = [
            s.text_frame.text for s in cac_shape
            if s.has_text_frame and s.text_frame.text.strip()
        ]
        thu_tu = 0
        da_lay = set()  # partname của ảnh đã lấy, để bước dự phòng không lấy trùng
        for shape in cac_shape:
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            # shape.image ném lỗi trong 2 trường hợp thực tế đã gặp: ảnh LIÊN KẾT ngoài
            # (Insert > Link to File, không có byte nào trong file), và ảnh SVG do
            # PowerPoint đời mới chèn (blip trỏ tới ảnh qua phần mở rộng mà python-pptx
            # không lần ra được). Không bắt lỗi ở đây thì cả lần build index hỏng.
            try:
                anh = shape.image
            except (ValueError, KeyError, AttributeError):
                continue  # để bước quét rels bên dưới lo
            thu_tu += 1
            dich = config.IMAGES_DIR / _ten_file_an_toan(
                duong_dan.name, so_slide, thu_tu, f".{anh.ext}"
            )
            dich.write_bytes(anh.blob)
            da_lay.add(getattr(anh, "sha1", None) or dich.name)
            ket_qua.append(
                _ban_ghi_anh(duong_dan.name, so_slide, dich, _chon_chu_thich(cac_dong))
            )

        # DỰ PHÒNG: quét quan hệ (rels) của slide để bắt những ảnh mà API shape không với
        # tới được. Đo trên một bài giảng thật: 11 shape ảnh đều ném lỗi qua API shape,
        # nhưng gói .pptx chứa 8 ảnh nhúng thật (9 MB, gồm cả SVG) - nếu chỉ dựa vào API
        # shape thì toàn bộ số ảnh đó biến mất khỏi index mà không có dấu hiệu gì.
        # Cùng kỹ thuật đã dùng cho DOCX, nhưng ở đây quét theo TỪNG SLIDE nên vẫn giữ
        # được thông tin ảnh thuộc slide nào.
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
            except Exception as loi:  # noqa: BLE001 - gói hỏng thì bỏ qua đúng ảnh đó
                logger.warning(
                    "Slide %d của '%s': bỏ qua 1 ảnh (%s).",
                    so_slide, duong_dan.name, type(loi).__name__,
                )
                continue
            thu_tu += 1
            da_lay.add(khoa)
            dich = config.IMAGES_DIR / _ten_file_an_toan(
                duong_dan.name, so_slide, thu_tu, duoi
            )
            dich.write_bytes(du_lieu)
            ket_qua.append(
                _ban_ghi_anh(duong_dan.name, so_slide, dich, _chon_chu_thich(cac_dong))
            )
    return ket_qua


def trich_anh_docx(duong_dan: Path, document) -> List[Dict]:
    """Trích ảnh DOCX qua quan hệ (rels) của phần thân tài liệu.

    Không dùng `document.inline_shapes` vì bộ đó CHỈ chứa ảnh dạng inline - ảnh neo/nổi
    (floating, kiểu bọc chữ quanh ảnh) không nằm trong đó và sẽ bị bỏ sót. Duyệt rels bắt
    được cả hai loại.

    Đánh đổi: rels không cho biết ảnh nằm ở trang nào, nên mọi ảnh DOCX được gán trang 1.
    Chấp nhận được vì bản thân "trang" trong DOCX cũng chỉ là ước lượng theo ngắt trang
    cứng (xem doc_docx), và chú thích lân cận mới là thứ giúp tìm ra hình.
    """
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
        thu_tu += 1
        dich = config.IMAGES_DIR / _ten_file_an_toan(duong_dan.name, 1, thu_tu, duoi)
        dich.write_bytes(du_lieu)
        ket_qua.append(_ban_ghi_anh(duong_dan.name, 1, dich, _chon_chu_thich(cac_dong)))
    return ket_qua
