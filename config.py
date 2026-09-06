"""Cấu hình tập trung cho toàn bộ hệ thống RAG."""

import logging
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _ten_logger in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub", "faiss.loader"):
    logging.getLogger(_ten_logger).setLevel(logging.WARNING)


def _nap_file_env(duong_dan_env: Path) -> None:
    """Đọc file .env (nếu có) và nạp các dòng KEY=VALUE vào os.environ."""
    if not duong_dan_env.exists():
        return
    for dong in duong_dan_env.read_text(encoding="utf-8").splitlines():
        dong = dong.strip()
        if not dong or dong.startswith("#") or "=" not in dong:
            continue
        key, _, value = dong.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_nap_file_env(BASE_DIR / ".env")


def _lay_int(ten_bien: str, mac_dinh: int) -> int:
    return int(os.environ.get(ten_bien, mac_dinh))


def _lay_float(ten_bien: str, mac_dinh: float) -> float:
    return float(os.environ.get(ten_bien, mac_dinh))


def _lay_str(ten_bien: str, mac_dinh: str) -> str:
    return os.environ.get(ten_bien, mac_dinh)


def _lay_bool(ten_bien: str, mac_dinh: bool) -> bool:
    gia_tri = os.environ.get(ten_bien)
    if gia_tri is None:
        return mac_dinh
    return gia_tri.strip().lower() in ("1", "true", "yes", "on")


DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
METADATA_MAPPING_FILE = FAISS_INDEX_DIR / "metadata.pkl"
INDEX_INFO_FILE = FAISS_INDEX_DIR / "index_info.json"
IMAGES_DIR = DATA_DIR / "images"
CACHE_DIR = DATA_DIR / "cache"

for _thu_muc in (RAW_DOCS_DIR, FAISS_INDEX_DIR, IMAGES_DIR, CACHE_DIR):
    _thu_muc.mkdir(parents=True, exist_ok=True)


EMBEDDING_MODEL_NAME = _lay_str("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")

_LA_HO_E5 = "e5" in EMBEDDING_MODEL_NAME.lower()
EMBEDDING_QUERY_PREFIX = _lay_str("EMBEDDING_QUERY_PREFIX", "query: " if _LA_HO_E5 else "")
EMBEDDING_PASSAGE_PREFIX = _lay_str("EMBEDDING_PASSAGE_PREFIX", "passage: " if _LA_HO_E5 else "")

EMBEDDING_BATCH_SIZE = _lay_int("EMBEDDING_BATCH_SIZE", 64)


CHUNK_SIZE_TOKENS = _lay_int("CHUNK_SIZE_TOKENS", 160)
CHUNK_OVERLAP_TOKENS = _lay_int("CHUNK_OVERLAP_TOKENS", 32)
BIEN_AN_TOAN_TOKEN = _lay_int("BIEN_AN_TOAN_TOKEN", 16)

CHUNK_SEPARATORS = ["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""]

BAT_NHAN_DIEN_TIEU_DE = _lay_bool("BAT_NHAN_DIEN_TIEU_DE", True)
TY_LE_KICH_THUOC_CHU_TIEU_DE = _lay_float("TY_LE_KICH_THUOC_CHU_TIEU_DE", 1.15)
DO_DAI_TOI_DA_TIEU_DE = _lay_int("DO_DAI_TOI_DA_TIEU_DE", 90)

TIKTOKEN_ENCODING = _lay_str("TIKTOKEN_ENCODING", "cl100k_base")


TOP_K = _lay_int("TOP_K", 4)

HE_SO_OVER_FETCH = _lay_int("HE_SO_OVER_FETCH", 10)
SO_UNG_VIEN_TOI_THIEU = _lay_int("SO_UNG_VIEN_TOI_THIEU", 60)

TRONG_SO_BM25 = _lay_float("TRONG_SO_BM25", 0.0)

SO_UNG_VIEN_BM25_CUU_HO = _lay_int("SO_UNG_VIEN_BM25_CUU_HO", 10)
RRF_K = _lay_int("RRF_K", 60)

NGAN_SACH_KY_TU_MOI_DOAN = _lay_int("NGAN_SACH_KY_TU_MOI_DOAN", 1600)

MO_RONG_QUA_RANH_GIOI_TRANG = _lay_bool("MO_RONG_QUA_RANH_GIOI_TRANG", True)

SO_TRANG_TOI_DA_MO_RONG = _lay_int("SO_TRANG_TOI_DA_MO_RONG", 1)

SO_DOAN_TOI_DA_MOI_TRANG = _lay_int("SO_DOAN_TOI_DA_MOI_TRANG", 2)

SO_UNG_VIEN_XET_DA_DANG_TRANG = _lay_int("SO_UNG_VIEN_XET_DA_DANG_TRANG", 20)

SO_DOAN_ANH_TOI_DA = _lay_int("SO_DOAN_ANH_TOI_DA", 1)

BAT_TRICH_ANH = _lay_bool("BAT_TRICH_ANH", True)

BAT_CHU_THICH_ANH = _lay_bool("BAT_CHU_THICH_ANH", True)
VISION_MODEL_NAME = _lay_str("VISION_MODEL_NAME", "qwen2.5vl:3b")
VISION_NUM_PREDICT = _lay_int("VISION_NUM_PREDICT", 400)


BAT_RERANK = _lay_bool("BAT_RERANK", True)
RERANKER_MODEL_NAME = _lay_str("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
SO_UNG_VIEN_RERANK = _lay_int("SO_UNG_VIEN_RERANK", 30)


NGUONG_DIEM_TOI_THIEU = _lay_float("NGUONG_DIEM_TOI_THIEU", 0.50 if _LA_HO_E5 else 0.15)

TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT = _lay_float("TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT", 0.78)

LOG_PHAN_BO_DIEM = _lay_bool("LOG_PHAN_BO_DIEM", False)

NGUONG_DIEM_RERANK_TOI_THIEU = _lay_float("NGUONG_DIEM_RERANK_TOI_THIEU", 0.001)


OLLAMA_MODEL = _lay_str("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_HOST = _lay_str("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TEMPERATURE = _lay_float("OLLAMA_TEMPERATURE", 0.1)
OLLAMA_NUM_PREDICT = _lay_int("OLLAMA_NUM_PREDICT", 12000)

OLLAMA_NUM_CTX = _lay_int("OLLAMA_NUM_CTX", 16384)

OLLAMA_NUM_CTX_TOI_DA = _lay_int("OLLAMA_NUM_CTX_TOI_DA", 32768)

OLLAMA_DU_PHONG_TOKEN_SINH = _lay_int("OLLAMA_DU_PHONG_TOKEN_SINH", 4000)

SO_KY_TU_MOI_TOKEN_UOC_LUONG = _lay_float("SO_KY_TU_MOI_TOKEN_UOC_LUONG", 2.2)

BAT_THINKING_KHI_KIEM_CHUNG = _lay_bool("BAT_THINKING_KHI_KIEM_CHUNG", True)

SO_TRICH_DAN_HIEN_THI = _lay_int("SO_TRICH_DAN_HIEN_THI", 3)

CAU_TU_CHOI = {
    "vi": "Không tìm thấy thông tin trong tài liệu.",
    "en": "The documents do not contain this information.",
}


EVAL_DIR = BASE_DIR / "evaluation"
TEST_QUESTIONS_FILE = EVAL_DIR / "test_questions.json"

TEST_QUESTIONS_HELD_OUT_FILE = EVAL_DIR / "test_questions_held_out.json"
JUDGE_MODEL = _lay_str("JUDGE_MODEL", OLLAMA_MODEL)

BAT_OCR_DU_PHONG = _lay_bool("BAT_OCR_DU_PHONG", True)
SO_CID_TOI_THIEU_DE_OCR = _lay_int("SO_CID_TOI_THIEU_DE_OCR", 5)
TY_LE_CID_DE_OCR = _lay_float("TY_LE_CID_DE_OCR", 0.02)
SO_TU_TOI_THIEU_TRANG_CO_CHU = _lay_int("SO_TU_TOI_THIEU_TRANG_CO_CHU", 15)
OCR_NUM_PREDICT = _lay_int("OCR_NUM_PREDICT", 1200)
DPI_RENDER_TRANG_OCR = _lay_int("DPI_RENDER_TRANG_OCR", 150)
SO_KY_TU_TOI_THIEU_MOT_TAI_LIEU = _lay_int("SO_KY_TU_TOI_THIEU_MOT_TAI_LIEU", 200)
TY_LE_DIEN_TICH_ANH_TOAN_TRANG = _lay_float("TY_LE_DIEN_TICH_ANH_TOAN_TRANG", 0.60)

BAT_DOC_LAI_TRANG_DINH_CHU = _lay_bool("BAT_DOC_LAI_TRANG_DINH_CHU", True)
CAC_X_TOLERANCE_THU = [
    float(x) for x in _lay_str("CAC_X_TOLERANCE_THU", "2.0,1.5,1.0,0.7").split(",") if x.strip()
]
MUC_TANG_TU_LE_CHAP_NHAN = _lay_float("MUC_TANG_TU_LE_CHAP_NHAN", 0.03)
TY_LE_DINH_CHU_DE_DOC_LAI = _lay_float("TY_LE_DINH_CHU_DE_DOC_LAI", 0.10)
DO_DAI_CUM_DINH_CHU = _lay_int("DO_DAI_CUM_DINH_CHU", 25)
SO_KY_TU_TOI_THIEU_DE_DO = _lay_int("SO_KY_TU_TOI_THIEU_DE_DO", 200)

BAT_STREAMING = _lay_bool("BAT_STREAMING", True)
GIAN_CACH_VE_LAI_GIAY = _lay_float("GIAN_CACH_VE_LAI_GIAY", 0.12)
SO_KY_TU_SUY_LUAN_HIEN = _lay_int("SO_KY_TU_SUY_LUAN_HIEN", 500)

NGUONG_DIEM_JUDGE_THAP = _lay_float("NGUONG_DIEM_JUDGE_THAP", 0.5)
NGUONG_BAM_NGU_CANH_DE_NGHI_NGO = _lay_float("NGUONG_BAM_NGU_CANH_DE_NGHI_NGO", 0.30)
SO_LAN_CHAM_FAITHFULNESS = _lay_int("SO_LAN_CHAM_FAITHFULNESS", 3)

BAT_DOC_THEO_COT = _lay_bool("BAT_DOC_THEO_COT", True)
SO_O_DO_COT = _lay_int("SO_O_DO_COT", 60)
SO_O_RANH_TOI_THIEU = _lay_int("SO_O_RANH_TOI_THIEU", 3)
TY_LE_TU_MOI_COT = _lay_float("TY_LE_TU_MOI_COT", 0.25)
SO_TU_TOI_THIEU_DE_DO_COT = _lay_int("SO_TU_TOI_THIEU_DE_DO_COT", 60)

NGUONG_BAM_NGUON_HIEN_THI = _lay_float("NGUONG_BAM_NGUON_HIEN_THI", 0.30)


BAT_TRUY_VAN_NGU_CANH = _lay_bool("BAT_TRUY_VAN_NGU_CANH", True)

BAT_VIET_LAI_CAU_HOI = _lay_bool("BAT_VIET_LAI_CAU_HOI", False)

NUM_PREDICT_VIET_LAI = _lay_int("NUM_PREDICT_VIET_LAI", 200)

SO_LUOT_NGU_CANH = _lay_int("SO_LUOT_NGU_CANH", 3)

DO_DAI_TRA_LOI_TRONG_NGU_CANH = _lay_int("DO_DAI_TRA_LOI_TRONG_NGU_CANH", 300)

SO_TU_TOI_DA_CAU_VIET_LAI = _lay_int("SO_TU_TOI_DA_CAU_VIET_LAI", 60)

TRONG_SO_TRUY_VAN_GOC = _lay_float("TRONG_SO_TRUY_VAN_GOC", 1.0)


BAT_DOI_CHIEU_NGUON = _lay_bool("BAT_DOI_CHIEU_NGUON", True)

NGUONG_COSINE_DOI_CHIEU = _lay_float("NGUONG_COSINE_DOI_CHIEU", 0.88)

SO_CAP_DOI_CHIEU_TOI_DA = _lay_int("SO_CAP_DOI_CHIEU_TOI_DA", 3)

SO_LAN_CHAM_MAU_THUAN = _lay_int("SO_LAN_CHAM_MAU_THUAN", 2)

NGUONG_MAU_THUAN = _lay_float("NGUONG_MAU_THUAN", 0.6)


BAT_CACHE_INGESTION = _lay_bool("BAT_CACHE_INGESTION", True)

BAT_INDEX_TANG_DAN = _lay_bool("BAT_INDEX_TANG_DAN", True)

BAT_PROFILING_INGESTION = _lay_bool("BAT_PROFILING_INGESTION", True)


def _so_worker_mac_dinh(toi_da: int) -> int:
    """Số worker mặc định suy từ chính máy đang chạy, chặn trên bằng `toi_da`."""
    return max(1, min(toi_da, (os.cpu_count() or 2) // 2))


SO_WORKER_VISION = _lay_int("SO_WORKER_VISION", _so_worker_mac_dinh(2))

SO_WORKER_DOC = _lay_int("SO_WORKER_DOC", 1)


TY_LE_DINH_CHU_DAT_YEU_CAU = _lay_float("TY_LE_DINH_CHU_DAT_YEU_CAU", 0.02)


SO_TRANG_HIEU_CHINH_X_TOLERANCE = _lay_int("SO_TRANG_HIEU_CHINH_X_TOLERANCE", 3)


TY_LE_DIEN_TICH_ANH_TOI_THIEU = _lay_float("TY_LE_DIEN_TICH_ANH_TOI_THIEU", 0.015)

TY_LE_CANH_ANH_TRANG_TRI = _lay_float("TY_LE_CANH_ANH_TRANG_TRI", 12.0)

SO_LAN_LAP_COI_LA_LOGO = _lay_int("SO_LAN_LAP_COI_LA_LOGO", 4)


BAT_NGAN_SACH_THICH_UNG = _lay_bool("BAT_NGAN_SACH_THICH_UNG", True)

SO_TU_CAU_HOI_DON_GIAN = _lay_int("SO_TU_CAU_HOI_DON_GIAN", 12)

SO_UNG_VIEN_RERANK_DON_GIAN = _lay_int("SO_UNG_VIEN_RERANK_DON_GIAN", 12)

BAT_NEN_NGU_CANH = _lay_bool("BAT_NEN_NGU_CANH", True)


THIET_BI_EMBEDDING = _lay_str("THIET_BI_EMBEDDING", "auto")
THIET_BI_RERANK = _lay_str("THIET_BI_RERANK", "auto")

BAT_QUAN_LY_VRAM = _lay_bool("BAT_QUAN_LY_VRAM", True)

NHA_MODEL_SAU_INGESTION = _lay_bool("NHA_MODEL_SAU_INGESTION", True)

VRAM_DU_CHO_LO_LON_GB = _lay_float("VRAM_DU_CHO_LO_LON_GB", 3.0)
VRAM_DU_CHO_LO_VUA_GB = _lay_float("VRAM_DU_CHO_LO_VUA_GB", 2.0)

VRAM_MOI_WORKER_VISION_GB = _lay_float("VRAM_MOI_WORKER_VISION_GB", 1.5)

VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB = _lay_float("VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB", 10.0)
