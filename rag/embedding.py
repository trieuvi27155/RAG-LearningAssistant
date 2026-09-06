"""Wrapper cho sentence-transformers - model embedding đa ngôn ngữ, chạy local."""

import logging
from typing import Callable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from rag import tai_nguyen_gpu

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = None, thiet_bi: str = None):
        """thiet_bi: "cuda" / "cpu"; bỏ trống thì TỰ DÒ theo phần cứng của máy đang chạy."""
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.thiet_bi = thiet_bi or tai_nguyen_gpu.thiet_bi("embedding")
        self._model = SentenceTransformer(self.model_name, device=self.thiet_bi)
        logger.info("Model embedding '%s' chạy trên %s.", self.model_name, self.thiet_bi)

    def chuyen_thiet_bi(self, thiet_bi_moi: str) -> bool:
        """Chuyển model sang thiết bị khác. Trả True nếu thật sự có chuyển."""
        if thiet_bi_moi == self.thiet_bi:
            return False
        try:
            self._model = self._model.to(thiet_bi_moi)
        except Exception as loi:  # noqa: BLE001
            logger.warning(
                "Không chuyển được model embedding sang %s (%s) - giữ nguyên %s.",
                thiet_bi_moi, type(loi).__name__, self.thiet_bi,
            )
            return False
        logger.info("Model embedding chuyển %s -> %s.", self.thiet_bi, thiet_bi_moi)
        self.thiet_bi = thiet_bi_moi
        return True

    def _encode(self, texts: List[str], tien_to: str) -> np.ndarray:
        """normalize_embeddings=True để FAISS (IndexFlatIP) tính ra đúng cosine similarity;
        float32 là kiểu dữ liệu FAISS yêu cầu cho index."""
        if tien_to:
            texts = [tien_to + t for t in texts]
        embeddings = self._model.encode(
            texts,
            batch_size=tai_nguyen_gpu.kich_thuoc_lo_embedding(),
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

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        """Số token tối đa model xử lý được cho 1 đoạn text. Nội dung vượt quá bị CẮT BỎ
        âm thầm khi encode (không có lỗi báo ra) nên chunking bắt buộc phải biết con số này."""
        return int(getattr(self._model, "max_seq_length", 512) or 512)

    def dem_token(self, text: str) -> int:
        """Đếm token bằng ĐÚNG tokenizer của model đang dùng."""
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def lay_ham_dem_token(self) -> Optional[Callable[[str], int]]:
        """Trả về hàm đếm token của model để truyền sang chunking, hoặc None nếu model
        không có tokenizer truy cập được (khi đó chunking tự lùi về dùng tiktoken)."""
        try:
            self.dem_token("kiểm tra")
        except Exception:  # pragma: no cover
            logger.warning(
                "Không lấy được tokenizer của '%s' - chunking sẽ dùng bộ đếm xấp xỉ tiktoken.",
                self.model_name,
            )
            return None
        return self.dem_token
