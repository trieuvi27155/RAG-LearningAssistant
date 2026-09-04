"""Bộ nhớ đệm theo CONTENT HASH cho luồng Ingestion: tài liệu, OCR, chú thích ảnh, embedding.

VẤN ĐỀ ĐANG GIẢI:
Luồng build cũ đọc TOÀN BỘ thư mục, chunk toàn bộ, encode toàn bộ rồi tạo index mới - mỗi
lần bấm "Đọc tài liệu". Thêm 1 tài liệu thứ 11 vào 10 tài liệu đã xử lý nghĩa là trả lại
toàn bộ chi phí của 10 tài liệu kia: đọc lại PDF, OCR lại những trang hỏng, gọi lại model
vision cho từng hình, encode lại từng chunk. Với corpus thật (giáo trình Bishop + slide CV +
bài giảng IoT) đó là hàng chục phút cho một việc lẽ ra chỉ tốn vài chục giây.

NGUYÊN TẮC: khoá cache là NỘI DUNG, không phải tên file hay thời gian sửa file.
  - Tên file đổi mà nội dung không đổi -> vẫn trúng cache (đúng: nội dung index không đổi).
  - Nội dung đổi mà tên file giữ nguyên -> trượt cache (đúng: phải đọc lại).
  - Copy cùng một hình vào 20 slide -> 20 bản ghi nhưng chỉ MỘT lượt gọi model vision.
  - mtime không đáng tin: git checkout, copy file, đồng bộ cloud đều đổi mtime mà không đổi
    nội dung; ngược lại một số công cụ ghi đè file mà giữ nguyên mtime.

VÂN TAY CẤU HÌNH: khoá cache tài liệu còn gộp thêm "vân tay" của các tuỳ chọn ăn vào KẾT
QUẢ ĐỌC (bật/tắt OCR, ngưỡng dính chữ, DPI render...). Đổi một tuỳ chọn như vậy phải làm
trượt cache, nếu không hệ thống sẽ lặng lẽ trả về kết quả đọc theo cấu hình CŨ - đúng loại
lỗi không có triệu chứng mà cả project này đang cố gắng tránh (xem config.INDEX_INFO_FILE).

DỌN DẸP: cache chỉ là bản sao của thứ tính lại được. Xoá cả thư mục data/cache bất cứ lúc
nào cũng an toàn - lần build sau chỉ chậm lại đúng bằng chi phí gốc, không mất dữ liệu gì.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# Kích thước khối đọc file khi băm: file PDF có thể vài trăm MB (giáo trình quét ảnh), đọc
# nguyên vào RAM chỉ để băm là không cần thiết.
_KICH_THUOC_KHOI = 1 << 20  # 1 MB


# ============================================================
# BĂM
# ============================================================
def bam_bytes(du_lieu: bytes) -> str:
    """Băm một khối bytes -> chuỗi hex 32 ký tự (nửa đầu SHA-256).

    Cắt còn 32 ký tự vì khoá này chỉ dùng làm TÊN FILE cache trên máy cá nhân: 128 bit đã
    dư sức để không đụng độ trong một corpus vài nghìn tài liệu, mà tên file ngắn thì đường
    dẫn không chạm giới hạn 260 ký tự của Windows.
    """
    return hashlib.sha256(du_lieu).hexdigest()[:32]


def bam_chuoi(text: str) -> str:
    return bam_bytes(text.encode("utf-8"))


def bam_file(duong_dan: Path) -> str:
    """Băm nội dung một file, đọc theo khối để không nạp cả file vào RAM."""
    bam = hashlib.sha256()
    with open(duong_dan, "rb") as f:
        for khoi in iter(lambda: f.read(_KICH_THUOC_KHOI), b""):
            bam.update(khoi)
    return bam.hexdigest()[:32]


# Các tuỳ chọn ĂN VÀO KẾT QUẢ ĐỌC TÀI LIỆU. Đổi bất kỳ giá trị nào trong đây thì nội dung
# trang đọc ra sẽ khác, nên cache tài liệu phải trượt. Cố ý liệt kê TÊN BIẾN thay vì quét
# toàn bộ config: đa số tham số (TOP_K, ngưỡng rerank, num_ctx...) chỉ ảnh hưởng lúc TRUY
# VẤN, gộp chúng vào đây sẽ khiến mỗi lần chỉnh một ngưỡng truy xuất là mất trắng cache đọc
# tài liệu - tức cache gần như không bao giờ trúng.
_THAM_SO_ANH_HUONG_DOC_TAI_LIEU = (
    "BAT_NHAN_DIEN_TIEU_DE", "TY_LE_KICH_THUOC_CHU_TIEU_DE", "DO_DAI_TOI_DA_TIEU_DE",
    "BAT_TRICH_ANH", "BAT_CHU_THICH_ANH", "VISION_MODEL_NAME", "VISION_NUM_PREDICT",
    "BAT_OCR_DU_PHONG", "SO_CID_TOI_THIEU_DE_OCR", "TY_LE_CID_DE_OCR",
    "SO_TU_TOI_THIEU_TRANG_CO_CHU", "OCR_NUM_PREDICT", "DPI_RENDER_TRANG_OCR",
    "BAT_DOC_LAI_TRANG_DINH_CHU", "CAC_X_TOLERANCE_THU", "MUC_TANG_TU_LE_CHAP_NHAN",
    "TY_LE_DINH_CHU_DE_DOC_LAI", "TY_LE_DINH_CHU_DAT_YEU_CAU", "DO_DAI_CUM_DINH_CHU",
    "SO_KY_TU_TOI_THIEU_DE_DO",
    "BAT_DOC_THEO_COT", "SO_O_DO_COT", "SO_O_RANH_TOI_THIEU", "TY_LE_TU_MOI_COT",
    "SO_TU_TOI_THIEU_DE_DO_COT", "TY_LE_DIEN_TICH_ANH_TOAN_TRANG",
    "TY_LE_DIEN_TICH_ANH_TOI_THIEU", "TY_LE_CANH_ANH_TRANG_TRI",
    "SO_LAN_LAP_COI_LA_LOGO", "SO_TRANG_HIEU_CHINH_X_TOLERANCE",
)


def van_tay_doc_tai_lieu() -> str:
    """Vân tay của cấu hình ĐỌC TÀI LIỆU - thành phần thứ hai của khoá cache tài liệu."""
    cac_gia_tri = {ten: repr(getattr(config, ten, None)) for ten in _THAM_SO_ANH_HUONG_DOC_TAI_LIEU}
    return bam_chuoi(json.dumps(cac_gia_tri, sort_keys=True))


def van_tay_embedding() -> str:
    """Vân tay của model embedding - vector sinh bởi model khác thì không dùng lẫn được."""
    return bam_chuoi(
        f"{config.EMBEDDING_MODEL_NAME}|{config.EMBEDDING_QUERY_PREFIX}"
        f"|{config.EMBEDDING_PASSAGE_PREFIX}"
    )


# ============================================================
# KHO TEXT / JSON TRÊN ĐĨA
# ============================================================
class KhoDem:
    """Kho khoá-giá trị đơn giản trên đĩa, mỗi giá trị là một file trong data/cache/<ten>/.

    Chia thư mục con theo 2 ký tự đầu của khoá: một corpus vài nghìn ảnh sẽ tạo vài nghìn
    file, và thư mục phẳng cỡ đó khiến chính File Explorer lẫn các thao tác glob chậm hẳn đi
    trên Windows. Hai ký tự đầu cho tối đa 256 thư mục con, đủ để mỗi thư mục còn vài chục file.

    MỌI lỗi đọc/ghi đều được nuốt kèm log: cache hỏng (đĩa đầy, file ghi dở vì tắt máy giữa
    chừng, quyền ghi bị chặn) chỉ được phép làm hệ thống CHẬM lại đúng bằng lúc chưa có
    cache, tuyệt đối không được làm hỏng một lần build.
    """

    def __init__(self, ten: str, duoi: str = ".txt"):
        self.thu_muc = config.CACHE_DIR / ten
        self.duoi = duoi
        self.so_trung = 0
        self.so_truot = 0

    def _duong_dan(self, khoa: str) -> Path:
        return self.thu_muc / khoa[:2] / f"{khoa}{self.duoi}"

    def co(self, khoa: str) -> bool:
        return self._duong_dan(khoa).exists()

    def lay_text(self, khoa: str) -> Optional[str]:
        duong_dan = self._duong_dan(khoa)
        if not duong_dan.exists():
            self.so_truot += 1
            return None
        try:
            noi_dung = duong_dan.read_text(encoding="utf-8")
        except OSError as loi:
            logger.warning("Không đọc được cache '%s': %s", duong_dan.name, loi)
            self.so_truot += 1
            return None
        self.so_trung += 1
        return noi_dung

    def luu_text(self, khoa: str, noi_dung: str) -> None:
        duong_dan = self._duong_dan(khoa)
        try:
            duong_dan.parent.mkdir(parents=True, exist_ok=True)
            # Ghi ra file tạm rồi đổi tên: tắt máy giữa lúc ghi sẽ để lại file tạm dở dang
            # thay vì một mục cache "có thật nhưng thiếu nội dung" - thứ mà lần build sau sẽ
            # tin tưởng dùng luôn.
            tam = duong_dan.with_suffix(duong_dan.suffix + ".tam")
            tam.write_text(noi_dung, encoding="utf-8")
            tam.replace(duong_dan)
        except OSError as loi:
            logger.warning("Không ghi được cache '%s': %s", duong_dan.name, loi)

    def lay_json(self, khoa: str):
        noi_dung = self.lay_text(khoa)
        if noi_dung is None:
            return None
        try:
            return json.loads(noi_dung)
        except json.JSONDecodeError:
            logger.warning("Cache '%s' hỏng định dạng JSON - bỏ qua, đọc lại từ đầu.", khoa)
            return None

    def luu_json(self, khoa: str, du_lieu) -> None:
        self.luu_text(khoa, json.dumps(du_lieu, ensure_ascii=False))


# Các kho dùng chung cho cả tiến trình. Tạo sẵn ở mức module (không phải mỗi lần gọi) để
# bộ đếm trúng/trượt cộng dồn được cho cả một lần build.
kho_tai_lieu = KhoDem("tai_lieu", ".json")
kho_ocr = KhoDem("ocr", ".txt")
kho_vision = KhoDem("vision", ".txt")


def khoa_tai_lieu(duong_dan: Path) -> str:
    """Khoá cache của một tài liệu = băm(nội dung file) + vân tay cấu hình đọc."""
    return bam_chuoi(f"{bam_file(duong_dan)}|{van_tay_doc_tai_lieu()}")


def khoa_ocr(bam_tai_lieu_: str, so_trang: int) -> str:
    """Khoá cache OCR một trang.

    Băm theo (nội dung tài liệu, số trang, DPI render) thay vì theo ảnh đã render: nhờ vậy
    khi trúng cache thì KHÔNG PHẢI RENDER ảnh nữa. Render một trang ở 150 DPI mất khoảng
    0,2-0,4 giây - với một cuốn sách scan 400 trang thì riêng phần render đã là vài phút,
    tức nếu khoá theo ảnh thì cache OCR chỉ tiết kiệm được một nửa chi phí.
    """
    return bam_chuoi(
        f"{bam_tai_lieu_}|{so_trang}|{config.DPI_RENDER_TRANG_OCR}|{config.VISION_MODEL_NAME}"
        f"|{config.OCR_NUM_PREDICT}"
    )


def khoa_vision(duong_dan_anh: Path) -> Optional[str]:
    """Khoá cache chú thích ảnh = băm NỘI DUNG ẢNH (+ model + độ dài mô tả).

    Băm theo nội dung chứ không theo đường dẫn là điều làm nên phần lớn giá trị ở đây: logo
    trường lặp lại trên 60 slide, hình minh hoạ dùng chung giữa hai bài giảng, hay chính một
    tài liệu được đọc lại sau khi đổi tên - tất cả đều quy về MỘT lượt gọi model vision.
    """
    try:
        return bam_chuoi(
            f"{bam_file(duong_dan_anh)}|{config.VISION_MODEL_NAME}|{config.VISION_NUM_PREDICT}"
        )
    except OSError as loi:
        logger.warning("Không băm được ảnh '%s': %s", duong_dan_anh, loi)
        return None


# ============================================================
# KHO VECTOR (EMBEDDING)
# ============================================================
class KhoVectorDem:
    """Cache embedding theo băm nội dung chunk, lưu trong MỘT file .npz duy nhất.

    Vì sao một file thay vì mỗi vector một file như các kho ở trên: corpus thật có ~6.000
    chunk, tức 6.000 file 3KB. Đọc từng file một lúc khởi động mất lâu hơn hẳn so với đọc
    một mảng liền 18MB, và trên Windows thì chênh lệch còn lớn hơn nữa. Vector lại là dữ
    liệu số kích thước đều nhau nên rất hợp với một mảng 2 chiều.

    Vì sao vẫn đáng cache dù encode "chỉ" tốn 316 giây cho cả corpus (xem
    config.EMBEDDING_MODEL_NAME): 316 giây đó lặp lại mỗi lần bấm "Đọc tài liệu", kể cả khi
    người dùng chỉ vừa thêm một file 3 trang. Phần lớn chunk giữa hai lần build là y hệt nhau
    - kể cả với tài liệu ĐÃ SỬA, vì sửa vài đoạn không làm đổi những trang còn lại.
    """

    def __init__(self):
        self.duong_dan = config.CACHE_DIR / "embedding" / f"{van_tay_embedding()}.npz"
        self._theo_khoa: Dict[str, int] = {}
        self._vector: Optional[np.ndarray] = None
        self._khoa_moi: List[str] = []
        self._vector_moi: List[np.ndarray] = []
        self.so_trung = 0
        self.so_truot = 0
        self._nap()

    def _nap(self) -> None:
        if not self.duong_dan.exists():
            return
        try:
            with np.load(self.duong_dan, allow_pickle=False) as du_lieu:
                khoa = du_lieu["khoa"]
                self._vector = du_lieu["vector"]
            self._theo_khoa = {str(k): i for i, k in enumerate(khoa)}
        except Exception as loi:  # noqa: BLE001 - file cache hỏng không được làm sập build
            logger.warning("Cache embedding hỏng (%s) - bỏ qua, encode lại từ đầu.", loi)
            self._theo_khoa, self._vector = {}, None

    def lay(self, text: str) -> Optional[np.ndarray]:
        """Vector của chunk này, hoặc None nếu chưa có.

        Phải xét CẢ phần chưa kịp ghi xuống đĩa (`_vector_moi`), không chỉ phần đã nạp từ
        file. `them()` ghi khoá vào sổ ngay lập tức để không encode trùng trong cùng một lần
        build, nên có một khoảng thời gian mà khoá đã có mặt trong sổ nhưng vector của nó
        chưa nằm trong `_vector` - và nếu lần `luu()` trước đó thất bại (đĩa đầy, quyền ghi
        bị chặn) thì khoảng đó kéo dài tới hết lần build. Đọc thẳng `_vector[vi_tri]` lúc ấy
        là IndexError, tức một sự cố ghi cache làm sập cả lần build: đúng điều mà lớp này cam
        kết không bao giờ để xảy ra.
        """
        khoa = bam_chuoi(text)
        vi_tri = self._theo_khoa.get(khoa)
        if vi_tri is None:
            self.so_truot += 1
            return None
        so_da_luu = 0 if self._vector is None else len(self._vector)
        if vi_tri < so_da_luu:
            self.so_trung += 1
            return self._vector[vi_tri]
        thu_tu_moi = vi_tri - so_da_luu
        if thu_tu_moi < len(self._vector_moi):
            self.so_trung += 1
            return self._vector_moi[thu_tu_moi]
        self.so_truot += 1
        return None

    def them(self, text: str, vector: np.ndarray) -> None:
        khoa = bam_chuoi(text)
        if khoa in self._theo_khoa:
            return
        # Đánh dấu vị trí ngay để không encode trùng trong CÙNG một lần build (hai chunk
        # trùng nội dung ở hai tài liệu khác nhau là chuyện thường gặp với slide dùng lại).
        self._theo_khoa[khoa] = len(self._theo_khoa)
        self._khoa_moi.append(khoa)
        self._vector_moi.append(np.asarray(vector, dtype="float32"))

    def luu(self) -> None:
        """Gộp phần mới vào file cache. Không có gì mới thì không đụng vào đĩa."""
        if not self._khoa_moi:
            return
        moi = np.vstack(self._vector_moi)
        vector = moi if self._vector is None else np.vstack([self._vector, moi])
        khoa = np.array(
            [k for k, _ in sorted(self._theo_khoa.items(), key=lambda kv: kv[1])], dtype=object
        )
        try:
            self.duong_dan.parent.mkdir(parents=True, exist_ok=True)
            # Tên file tạm PHẢI kết thúc bằng ".npz": np.savez tự nối thêm đuôi đó khi thiếu,
            # nên một tên như "....npz.tam" sẽ được ghi thành "....npz.tam.npz" và lệnh đổi
            # tên ngay dưới sẽ không tìm thấy file - cache im lặng không bao giờ được ghi.
            tam = self.duong_dan.with_name(self.duong_dan.name + ".tam.npz")
            np.savez(tam, khoa=khoa.astype("U32"), vector=vector)
            tam.replace(self.duong_dan)
        except Exception as loi:  # noqa: BLE001
            logger.warning("Không ghi được cache embedding: %s", loi)
            return
        self._vector = vector
        self._khoa_moi, self._vector_moi = [], []


def encode_co_cache(embedding_service, cac_text: List[str], kho: Optional[KhoVectorDem] = None):
    """encode_tai_lieu() nhưng chỉ encode những chunk CHƯA có trong cache.

    Trả về mảng vector đúng thứ tự `cac_text`. Giữ nguyên chữ ký "vào list text, ra mảng
    vector" của EmbeddingService để chỗ gọi không phải biết cache tồn tại.
    """
    if not cac_text:
        return np.zeros((0, embedding_service.dimension), dtype="float32")
    if not config.BAT_CACHE_INGESTION:
        return embedding_service.encode_tai_lieu(cac_text)

    kho = kho if kho is not None else KhoVectorDem()
    ket_qua: List[Optional[np.ndarray]] = [kho.lay(t) for t in cac_text]
    can_encode = [i for i, v in enumerate(ket_qua) if v is None]
    if can_encode:
        vector_moi = embedding_service.encode_tai_lieu([cac_text[i] for i in can_encode])
        for i, vector in zip(can_encode, vector_moi):
            ket_qua[i] = vector
            kho.them(cac_text[i], vector)
        kho.luu()
    logger.info(
        "Embedding: %d/%d chunk lấy từ cache, %d chunk phải encode lại.",
        len(cac_text) - len(can_encode), len(cac_text), len(can_encode),
    )
    return np.vstack(ket_qua).astype("float32")


# ============================================================
# DỌN DẸP
# ============================================================
def dung_luong_cache() -> int:
    """Tổng số byte cache đang chiếm - để giao diện nói được con số thật khi mời xoá."""
    if not config.CACHE_DIR.exists():
        return 0
    return sum(f.stat().st_size for f in config.CACHE_DIR.rglob("*") if f.is_file())


def xoa_cache() -> None:
    """Xoá toàn bộ cache. An toàn tuyệt đối - mọi thứ trong đó đều tính lại được."""
    if config.CACHE_DIR.exists():
        shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Đã xoá toàn bộ cache ingestion tại %s.", config.CACHE_DIR)
