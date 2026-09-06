"""Wrapper cho FAISS - build, lưu, load, tìm kiếm (vector + từ khoá)."""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

import config
from rag.lexical_search import BM25

logger = logging.getLogger(__name__)


def so_sanh_bam_tai_lieu(
    bam_trong_index: Dict[str, str], bam_tren_dia: Dict[str, str]
) -> Tuple[List[str], List[str], List[str]]:
    """So sổ băm của index với thực tế thư mục -> (cần đọc lại, cần xoá, giữ nguyên)."""
    can_doc, giu_nguyen = [], []
    for ten, bam in bam_tren_dia.items():
        (giu_nguyen if bam_trong_index.get(ten) == bam else can_doc).append(ten)
    can_xoa = [ten for ten in bam_trong_index if ten not in bam_tren_dia]
    return can_doc, can_xoa, giu_nguyen


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []
        self.thong_tin: Dict = {}
        self.bam_tai_lieu: Dict[str, str] = {}
        self._xoa_cache()

    def _xoa_cache(self) -> None:
        """Xoá cache chỉ mục trang và BM25 - gọi sau MỌI thay đổi dữ liệu vì vị trí trong
        self.metadata bị dịch đi."""
        self._chi_muc_trang: Optional[Dict[Tuple[str, int], List[int]]] = None
        self._chi_muc_nguon: Optional[Dict[str, List[int]]] = None
        self._bm25: Optional[BM25] = None

    @property
    def chi_muc_trang(self) -> Dict[Tuple[str, int], List[int]]:
        """(nguon, trang) -> danh sách vị trí các chunk của trang đó, đã sắp theo thứ tự
        xuất hiện trong trang gốc ("vi_tri")."""
        if self._chi_muc_trang is None:
            chi_muc: Dict[Tuple[str, int], List[int]] = {}
            for i, m in enumerate(self.metadata):
                chi_muc.setdefault((m["nguon"], m["trang"]), []).append(i)
            for danh_sach in chi_muc.values():
                danh_sach.sort(key=lambda i: self.metadata[i].get("vi_tri", 0))
            self._chi_muc_trang = chi_muc
        return self._chi_muc_trang

    @property
    def chi_muc_nguon(self) -> Dict[str, List[int]]:
        """nguon -> vị trí MỌI chunk của tài liệu đó, đã sắp theo thứ tự đọc trong tài liệu."""
        if self._chi_muc_nguon is None:
            chi_muc: Dict[str, List[int]] = {}
            for i, m in enumerate(self.metadata):
                chi_muc.setdefault(m["nguon"], []).append(i)
            for danh_sach in chi_muc.values():
                danh_sach.sort(key=lambda i: (self.metadata[i]["trang"],
                                              self.metadata[i].get("vi_tri", 0)))
            self._chi_muc_nguon = chi_muc
        return self._chi_muc_nguon

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([m["noidung"] for m in self.metadata])
        return self._bm25

    def them(self, vectors: np.ndarray, metadata_list: List[Dict]) -> None:
        """Thêm 1 batch vector + metadata tương ứng vào index."""
        if len(vectors) != len(metadata_list):
            raise ValueError(
                f"Số vector ({len(vectors)}) phải khớp số metadata ({len(metadata_list)})"
            )
        self.index.add(vectors)
        self.metadata.extend(metadata_list)
        self._xoa_cache()

    def xoa_theo_nguon(self, ten_file: str) -> int:
        """Xóa toàn bộ vector + metadata thuộc về 1 file khỏi index ngay lập tức.
        Trả về số vector đã xóa (0 nếu file không có trong index)."""
        self.bam_tai_lieu.pop(ten_file, None)
        vi_tri_xoa = [i for i, m in enumerate(self.metadata) if m["nguon"] == ten_file]
        if not vi_tri_xoa:
            return 0
        self.index.remove_ids(np.array(vi_tri_xoa, dtype="int64"))
        for i in sorted(vi_tri_xoa, reverse=True):
            del self.metadata[i]
        self._xoa_cache()
        return len(vi_tri_xoa)

    def theo_nguon_va_trang(self, nguon: str, trang: int) -> List[Dict]:
        """Toàn bộ chunk của đúng 1 (nguon, trang), đã sắp theo thứ tự trong trang gốc."""
        return [self.metadata[i] for i in self.chi_muc_trang.get((nguon, trang), [])]

    def tim_kiem_vi_tri(self, vector_cau_hoi: np.ndarray, top_k: int = None) -> List[Tuple[int, float]]:
        """Như tim_kiem() nhưng trả về VỊ TRÍ trong metadata thay vì bản thân metadata."""
        top_k = top_k or config.TOP_K
        top_k = min(top_k, self.index.ntotal)
        if top_k <= 0:
            return []
        diem_so, vi_tri = self.index.search(vector_cau_hoi, top_k)
        return [(int(i), float(d)) for i, d in zip(vi_tri[0], diem_so[0]) if i != -1]

    def tim_kiem(self, vector_cau_hoi: np.ndarray, top_k: int = None) -> List[Tuple[Dict, float]]:
        """Tìm top_k chunk có cosine similarity cao nhất với vector_cau_hoi."""
        return [(self.metadata[i], diem) for i, diem in self.tim_kiem_vi_tri(vector_cau_hoi, top_k)]

    def tim_kiem_tu_khoa(self, cau_hoi: str, top_n: int) -> List[Tuple[int, float]]:
        """Nhánh tìm kiếm theo từ khoá (BM25) - trả về [(vị trí, điểm)] giảm dần."""
        return self.bm25.tim_kiem(cau_hoi, top_n)

    def diem_cosine(self, vi_tri: List[int], vector_cau_hoi: np.ndarray) -> Dict[int, float]:
        """Tính cosine similarity giữa câu hỏi và các chunk ở những vị trí cho trước."""
        if not vi_tri:
            return {}
        vector_goc = np.vstack([self.index.reconstruct(int(i)) for i in vi_tri])
        return {i: float(d) for i, d in zip(vi_tri, vector_goc @ vector_cau_hoi[0])}

    def luu(self, index_path: Path = None, metadata_path: Path = None, info_path: Path = None) -> None:
        index_path = index_path or config.FAISS_INDEX_FILE
        metadata_path = metadata_path or config.METADATA_MAPPING_FILE
        info_path = info_path or config.INDEX_INFO_FILE
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        self.thong_tin = {
            "bam_tai_lieu": dict(getattr(self, "bam_tai_lieu", {}) or {}),
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            "nhan_dien_tieu_de": config.BAT_NHAN_DIEN_TIEU_DE,
            "trich_anh": config.BAT_TRICH_ANH,
            "chu_thich_anh_vision": config.BAT_CHU_THICH_ANH,
            "so_chunk": len(self.metadata),
            "thoi_diem_build": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        Path(info_path).write_text(
            json.dumps(self.thong_tin, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def tai(cls, index_path: Path = None, metadata_path: Path = None, info_path: Path = None) -> "VectorStore":
        index_path = index_path or config.FAISS_INDEX_FILE
        metadata_path = metadata_path or config.METADATA_MAPPING_FILE
        info_path = info_path or config.INDEX_INFO_FILE
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            obj.metadata = pickle.load(f)
        obj.thong_tin = {}
        obj.bam_tai_lieu = {}
        if Path(info_path).exists():
            try:
                obj.thong_tin = json.loads(Path(info_path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Không đọc được %s - bỏ qua kiểm tra tương thích index.", info_path)
        obj.bam_tai_lieu = dict(obj.thong_tin.get("bam_tai_lieu") or {})
        obj._xoa_cache()
        return obj

    def ly_do_khong_tuong_thich(self) -> Optional[str]:
        """Trả về mô tả lý do index trên đĩa không còn khớp cấu hình hiện tại (hoặc None)."""
        if not self.metadata:
            return None
        if not self.thong_tin:
            return (
                "Index này được build bằng phiên bản cũ của hệ thống (chưa ghi lại thông tin "
                "cấu hình) nên không kiểm tra được có khớp model embedding hiện tại hay không."
            )
        model_cu = self.thong_tin.get("embedding_model")
        if model_cu != config.EMBEDDING_MODEL_NAME:
            return (
                f"Index được build bằng model embedding '{model_cu}', nhưng hệ thống đang "
                f"dùng '{config.EMBEDDING_MODEL_NAME}'. Vector câu hỏi và vector tài liệu "
                "sẽ không cùng một không gian ngữ nghĩa nên kết quả truy xuất sẽ sai."
            )
        if self.thong_tin.get("chunk_size_tokens") != config.CHUNK_SIZE_TOKENS:
            return (
                f"Index được build với chunk size {self.thong_tin.get('chunk_size_tokens')} "
                f"token, khác cấu hình hiện tại ({config.CHUNK_SIZE_TOKENS})."
            )
        for khoa, gia_tri_hien_tai, mo_ta in (
            ("nhan_dien_tieu_de", config.BAT_NHAN_DIEN_TIEU_DE, "nhận diện tiêu đề"),
            ("trich_anh", config.BAT_TRICH_ANH, "trích xuất hình ảnh"),
            ("chu_thich_anh_vision", config.BAT_CHU_THICH_ANH, "chú thích ảnh bằng model vision"),
        ):
            if khoa in self.thong_tin and self.thong_tin[khoa] != gia_tri_hien_tai:
                return (
                    f"Index được build khi tuỳ chọn '{mo_ta}' đang "
                    f"{'BẬT' if self.thong_tin[khoa] else 'TẮT'}, nhưng hiện tại đang "
                    f"{'BẬT' if gia_tri_hien_tai else 'TẮT'} - nội dung đã index không khớp."
                )
        return None

    @property
    def so_luong_vector(self) -> int:
        return self.index.ntotal
