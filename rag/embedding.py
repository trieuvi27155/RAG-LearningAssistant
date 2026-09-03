"""Wrapper cho sentence-transformers - model embedding đa ngôn ngữ, chạy local.

Đóng gói thành 1 class EmbeddingService để sau này muốn đổi sang model khác
(ví dụ bge-m3) chỉ cần đổi EMBEDDING_MODEL_NAME trong config.py, không phải sửa
code ở rag_pipeline.py hay vector_store.py - chúng chỉ gọi qua interface
.encode_tai_lieu()/.encode_cau_hoi()/.dimension.

Điểm quan trọng: câu hỏi và đoạn tài liệu được mã hoá bằng 2 hàm KHÁC NHAU
(encode_cau_hoi/encode_tai_lieu) chứ không phải 1 hàm encode() dùng chung. Các model
retrieval hiện đại (họ E5, BGE, GTE...) được huấn luyện bất đối xứng: cùng một đoạn text
phải gắn tiền tố khác nhau tuỳ vai trò "câu hỏi" hay "đoạn cần tìm". Tách 2 hàm khiến
việc gắn đúng tiền tố là chuyện của module này, chỗ gọi không thể quên - nếu để 1 hàm
encode() dùng chung, sai sót kiểu "quên tiền tố ở luồng query" sẽ không gây lỗi mà chỉ âm
thầm làm tụt chất lượng truy xuất.
"""

import logging
from typing import Callable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self._model = SentenceTransformer(self.model_name)

    # ---------- Mã hoá ----------

    def _encode(self, texts: List[str], tien_to: str) -> np.ndarray:
        """normalize_embeddings=True: chuẩn hóa vector về độ dài 1, để khi FAISS tính
        inner product (IndexFlatIP) thì kết quả chính là cosine similarity.
        float32: FAISS yêu cầu kiểu dữ liệu này cho index.
        """
        if tien_to:
            texts = [tien_to + t for t in texts]
        embeddings = self._model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def encode_tai_lieu(self, texts: List[str]) -> np.ndarray:
        """Mã hoá các ĐOẠN TÀI LIỆU (dùng ở luồng Ingestion)."""
        return self._encode(texts, config.EMBEDDING_PASSAGE_PREFIX)

    def encode_cau_hoi(self, texts: List[str]) -> np.ndarray:
        """Mã hoá CÂU HỎI (dùng ở luồng Query)."""
        return self._encode(texts, config.EMBEDDING_QUERY_PREFIX)

    # ---------- Thông tin về model (chunking cần để chia chunk cho vừa) ----------

    @property
    def dimension(self) -> int:
        # sentence-transformers 6.0 đổi tên get_sentence_embedding_dimension ->
        # get_embedding_dimension (tên cũ vẫn chạy nhưng cảnh báo FutureWarning). Dùng tên
        # mới nếu có, lùi về tên cũ cho các bản cũ hơn - requirements.txt cho phép từ 2.7.
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        """Số token tối đa model xử lý được cho 1 đoạn text. Nội dung vượt quá bị CẮT BỎ
        âm thầm khi encode (không có lỗi báo ra) nên chunking bắt buộc phải biết con số này.
        """
        return int(getattr(self._model, "max_seq_length", 512) or 512)

    def dem_token(self, text: str) -> int:
        """Đếm token bằng ĐÚNG tokenizer của model đang dùng.

        Thay cho bộ đếm xấp xỉ tiktoken ở bản trước: đã đo trên tài liệu tiếng Việt thật,
        tiktoken đếm ra số token gấp ~1.9 lần tokenizer thật của model, khiến chunk bị chia
        nhỏ hơn nhiều so với dự định (chỉ dùng ~31% giới hạn cho phép) và nội dung bị băm
        vụn - một trong những nguyên nhân gốc khiến truy xuất trên tài liệu dài kém chính xác.

        add_special_tokens=False: chỉ đếm token của chính nội dung, phần token đặc biệt
        ([CLS]/[SEP]) đã được trừ hao riêng qua config.BIEN_AN_TOAN_TOKEN.
        """
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def lay_ham_dem_token(self) -> Optional[Callable[[str], int]]:
        """Trả về hàm đếm token của model để truyền sang chunking, hoặc None nếu model
        không có tokenizer truy cập được (khi đó chunking tự lùi về dùng tiktoken).
        """
        try:
            self.dem_token("kiểm tra")
        except Exception:  # pragma: no cover - chỉ xảy ra với backend model bất thường
            logger.warning(
                "Không lấy được tokenizer của '%s' - chunking sẽ dùng bộ đếm xấp xỉ tiktoken.",
                self.model_name,
            )
            return None
        return self.dem_token
