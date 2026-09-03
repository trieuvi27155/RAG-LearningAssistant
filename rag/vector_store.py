"""Wrapper cho FAISS - build, lưu, load, tìm kiếm (vector + từ khoá).

Dùng IndexFlatIP (inner product) thay vì IndexFlatL2 (khoảng cách Euclid) vì vector
embedding đã được chuẩn hóa (normalize) ở embedding.py - với vector đã chuẩn hóa,
inner product == cosine similarity, đúng thước đo "độ liên quan ngữ nghĩa" mà đồ án cần,
thay vì khoảng cách hình học thô giữa 2 điểm.

FAISS tự nó chỉ lưu vector (số), không lưu được metadata dạng text (tên file, số trang,
nội dung gốc...). Vì vậy cần thêm 1 list `metadata` song song: vị trí i trong metadata
tương ứng với vector thứ i đã add vào index, lưu riêng bằng pickle.

Ngoài chỉ mục vector, class này còn giữ 3 cấu trúc phụ được dựng LƯỜI (chỉ tính khi cần
lần đầu, sau đó dùng lại) và tự huỷ mỗi khi dữ liệu đổi:
  - chỉ mục (nguồn, trang) -> vị trí các chunk: để mở rộng ngữ cảnh sang chunk liền kề mà
    không phải quét tuyến tính toàn bộ metadata cho từng trang, từng câu hỏi.
  - chỉ mục nguồn -> vị trí các chunk theo thứ tự đọc: cùng mục đích nhưng XUYÊN TRANG, cho
    tài liệu văn bản chảy liên tục (xem chi_muc_nguon).
  - chỉ mục BM25: nhánh tìm kiếm theo từ khoá (xem rag/lexical_search.py).
"""

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


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []
        self.thong_tin: Dict = {}
        self._xoa_cache()

    # ------------------------------------------------------------------
    # Cấu trúc phụ dựng lười
    # ------------------------------------------------------------------
    def _xoa_cache(self) -> None:
        """Gọi sau MỌI thay đổi dữ liệu. Chỉ mục trang và BM25 đều ánh xạ theo vị trí trong
        self.metadata, mà thêm/xoá vector làm các vị trí đó dịch đi - giữ lại cache cũ sẽ
        khiến tìm kiếm trả về nội dung của chunk khác (sai âm thầm, không báo lỗi).
        """
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
        """nguon -> vị trí MỌI chunk của tài liệu đó, đã sắp theo thứ tự đọc trong tài liệu.

        Bổ sung cho chi_muc_trang, phục vụ việc mở rộng ngữ cảnh QUA ranh giới trang: với
        PDF văn bản chảy liên tục, một định nghĩa bắt đầu cuối trang 12 và kết thúc đầu
        trang 13 chỉ nối lại được khi có một thứ tự đọc xuyên trang (xem
        config.MO_RONG_QUA_RANH_GIOI_TRANG).

        Thứ tự đọc suy ra từ (trang, vi_tri) - HAI trường đã có sẵn trong mọi metadata - chứ
        KHÔNG thêm một trường "thứ tự toàn cục" mới. Chủ ý: thêm trường mới thì mọi index đã
        build đều thiếu nó và phải build lại, mà build lại tốn nguyên một lượt chú thích ảnh
        bằng model vision (~9 phút cho 291 ảnh). Thông tin cần thiết vốn đã nằm trong dữ
        liệu, không có lý do bắt người dùng trả giá đó để lấy lại đúng thứ mình đang có.
        """
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

    # ------------------------------------------------------------------
    # Ghi dữ liệu
    # ------------------------------------------------------------------
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
        """Xóa toàn bộ vector + metadata thuộc về 1 file khỏi index NGAY LẬP TỨC, không
        cần build lại từ đầu như §5.10 (quyết định đó chỉ áp dụng khi THÊM tài liệu mới -
        xóa 1 file không có lý do gì phải tính toán lại toàn bộ corpus còn lại).

        IndexFlatIP lưu vector trong 1 mảng liền; remove_ids nén mảng lại (dồn các vector
        còn lại lên, GIỮ NGUYÊN thứ tự tương đối - đã kiểm chứng bằng test thủ công) - nên
        metadata cũng phải xóa đúng các vị trí tương ứng theo thứ tự GIẢM DẦN (xoá từ cuối
        lên) để không bị lệch chỉ số khi xoá nhiều phần tử liên tiếp bằng del.

        Trả về số vector đã xóa (0 nếu file không có trong index).
        """
        vi_tri_xoa = [i for i, m in enumerate(self.metadata) if m["nguon"] == ten_file]
        if not vi_tri_xoa:
            return 0
        self.index.remove_ids(np.array(vi_tri_xoa, dtype="int64"))
        for i in sorted(vi_tri_xoa, reverse=True):
            del self.metadata[i]
        self._xoa_cache()
        return len(vi_tri_xoa)

    # ------------------------------------------------------------------
    # Truy vấn
    # ------------------------------------------------------------------
    def theo_nguon_va_trang(self, nguon: str, trang: int) -> List[Dict]:
        """Toàn bộ chunk của đúng 1 (nguon, trang), đã sắp theo thứ tự trong trang gốc."""
        return [self.metadata[i] for i in self.chi_muc_trang.get((nguon, trang), [])]

    def tim_kiem_vi_tri(self, vector_cau_hoi: np.ndarray, top_k: int = None) -> List[Tuple[int, float]]:
        """Như tim_kiem() nhưng trả về VỊ TRÍ trong metadata thay vì bản thân metadata.

        rag_pipeline cần vị trí (không phải nội dung) để hợp nhất kết quả với nhánh BM25 và
        để tra ra chunk liền kề - dùng dict metadata làm khoá thì không được (dict không
        hash được, và 2 chunk trùng nội dung sẽ lẫn vào nhau).
        """
        top_k = top_k or config.TOP_K
        top_k = min(top_k, self.index.ntotal)  # tránh lỗi khi index có ít hơn top_k vector
        if top_k <= 0:
            return []
        diem_so, vi_tri = self.index.search(vector_cau_hoi, top_k)
        # FAISS trả -1 ở ô chỉ số khi không tìm đủ top_k kết quả -> bỏ qua các ô đó.
        return [(int(i), float(d)) for i, d in zip(vi_tri[0], diem_so[0]) if i != -1]

    def tim_kiem(self, vector_cau_hoi: np.ndarray, top_k: int = None) -> List[Tuple[Dict, float]]:
        """Tìm top_k chunk có cosine similarity cao nhất với vector_cau_hoi.

        Trả về list (metadata, diem_similarity), đã sắp xếp giảm dần theo độ liên quan
        (FAISS tự trả về theo thứ tự này).
        """
        return [(self.metadata[i], diem) for i, diem in self.tim_kiem_vi_tri(vector_cau_hoi, top_k)]

    def tim_kiem_tu_khoa(self, cau_hoi: str, top_n: int) -> List[Tuple[int, float]]:
        """Nhánh tìm kiếm theo từ khoá (BM25) - trả về [(vị trí, điểm)] giảm dần."""
        return self.bm25.tim_kiem(cau_hoi, top_n)

    def diem_cosine(self, vi_tri: List[int], vector_cau_hoi: np.ndarray) -> Dict[int, float]:
        """Tính cosine similarity giữa câu hỏi và các chunk ở những vị trí cho trước.

        Cần cho các chunk chỉ do BM25 tìm ra (không nằm trong top của FAISS nên chưa có
        điểm cosine): mọi ngưỡng lọc và điểm hiển thị trên UI đều quy về cùng 1 thang đo
        cosine, nếu để lẫn điểm BM25 (thang đo hoàn toàn khác, không chặn trên) vào thì
        ngưỡng lọc sẽ vô nghĩa. IndexFlat lưu nguyên vector nên đọc lại được chính xác
        bằng reconstruct(), không phải tính xấp xỉ.
        """
        if not vi_tri:
            return {}
        vector_goc = np.vstack([self.index.reconstruct(int(i)) for i in vi_tri])
        return {i: float(d) for i, d in zip(vi_tri, vector_goc @ vector_cau_hoi[0])}

    # ------------------------------------------------------------------
    # Lưu / nạp
    # ------------------------------------------------------------------
    def luu(self, index_path: Path = None, metadata_path: Path = None, info_path: Path = None) -> None:
        index_path = index_path or config.FAISS_INDEX_FILE
        metadata_path = metadata_path or config.METADATA_MAPPING_FILE
        info_path = info_path or config.INDEX_INFO_FILE
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        # "Vân tay" cấu hình lúc build - xem giải thích ở config.INDEX_INFO_FILE.
        self.thong_tin = {
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            # Các tuỳ chọn ĂN VÀO NỘI DUNG đã index (đổi chúng thì chunk khác đi, phải build
            # lại). Ghi kèm để ly_do_khong_tuong_thich() phát hiện được - cùng cơ chế đã dùng
            # cho model embedding, không phát minh cách mới.
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
        # Dùng __new__ thay vì __init__ để tránh tạo 1 IndexFlatIP rỗng rồi bỏ đi ngay -
        # index thật sẽ được đọc trực tiếp từ file bằng faiss.read_index.
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            obj.metadata = pickle.load(f)
        obj.thong_tin = {}
        if Path(info_path).exists():
            try:
                obj.thong_tin = json.loads(Path(info_path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Không đọc được %s - bỏ qua kiểm tra tương thích index.", info_path)
        obj._xoa_cache()
        return obj

    def ly_do_khong_tuong_thich(self) -> Optional[str]:
        """Trả về mô tả lý do index trên đĩa không còn khớp cấu hình hiện tại (hoặc None).

        Đây là loại lỗi nguy hiểm vì KHÔNG gây crash: đổi EMBEDDING_MODEL_NAME sang model
        khác cùng số chiều rồi quên build lại index thì FAISS vẫn chạy bình thường, chỉ có
        điều vector câu hỏi và vector tài liệu nằm ở 2 không gian ngữ nghĩa khác nhau nên
        kết quả trả về gần như ngẫu nhiên. Không có cách nào phát hiện qua kết quả (nó vẫn
        "trông giống" kết quả thật), nên phải đối chiếu bằng vân tay đã ghi lúc build.
        """
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
        # Chỉ so những tuỳ chọn ĐÃ được ghi lại: index build bằng bản cũ hơn không có các
        # khoá này, và việc thiếu khoá không có nghĩa là cấu hình khác nhau.
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
