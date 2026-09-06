"""Bộ nhớ đệm theo CONTENT HASH cho luồng Ingestion: tài liệu, OCR, chú thích ảnh, embedding."""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

_KICH_THUOC_KHOI = 1 << 20


def bam_bytes(du_lieu: bytes) -> str:
    """Băm một khối bytes -> chuỗi hex 32 ký tự (nửa đầu SHA-256)."""
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


class KhoDem:
    """Kho khoá-giá trị đơn giản trên đĩa, mỗi giá trị là một file trong data/cache/<ten>/."""

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


kho_tai_lieu = KhoDem("tai_lieu", ".json")
kho_ocr = KhoDem("ocr", ".txt")
kho_vision = KhoDem("vision", ".txt")


def khoa_tai_lieu(duong_dan: Path) -> str:
    """Khoá cache của một tài liệu = băm(nội dung file) + vân tay cấu hình đọc."""
    return bam_chuoi(f"{bam_file(duong_dan)}|{van_tay_doc_tai_lieu()}")


def khoa_ocr(bam_tai_lieu_: str, so_trang: int) -> str:
    """Khoá cache OCR một trang."""
    return bam_chuoi(
        f"{bam_tai_lieu_}|{so_trang}|{config.DPI_RENDER_TRANG_OCR}|{config.VISION_MODEL_NAME}"
        f"|{config.OCR_NUM_PREDICT}"
    )


def khoa_vision(duong_dan_anh: Path) -> Optional[str]:
    """Khoá cache chú thích ảnh = băm NỘI DUNG ẢNH (+ model + độ dài mô tả)."""
    try:
        return bam_chuoi(
            f"{bam_file(duong_dan_anh)}|{config.VISION_MODEL_NAME}|{config.VISION_NUM_PREDICT}"
        )
    except OSError as loi:
        logger.warning("Không băm được ảnh '%s': %s", duong_dan_anh, loi)
        return None


class KhoVectorDem:
    """Cache embedding theo băm nội dung chunk, lưu trong MỘT file .npz duy nhất."""

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
        except Exception as loi:  # noqa: BLE001
            logger.warning("Cache embedding hỏng (%s) - bỏ qua, encode lại từ đầu.", loi)
            self._theo_khoa, self._vector = {}, None

    def lay(self, text: str) -> Optional[np.ndarray]:
        """Vector của chunk này, hoặc None nếu chưa có."""
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
            tam = self.duong_dan.with_name(self.duong_dan.name + ".tam.npz")
            np.savez(tam, khoa=khoa.astype("U32"), vector=vector)
            tam.replace(self.duong_dan)
        except Exception as loi:  # noqa: BLE001
            logger.warning("Không ghi được cache embedding: %s", loi)
            return
        self._vector = vector
        self._khoa_moi, self._vector_moi = [], []


def encode_co_cache(embedding_service, cac_text: List[str], kho: Optional[KhoVectorDem] = None):
    """encode_tai_lieu() nhưng chỉ encode những chunk CHƯA có trong cache."""
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
