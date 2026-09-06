"""Xếp hạng lại (rerank) bằng cross-encoder - tầng lọc THỨ HAI sau tìm kiếm lai."""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

import config
from rag import tai_nguyen_gpu

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_name: str = None, thiet_bi: str = None):
        """thiet_bi: "cuda" / "cpu"; bỏ trống thì TỰ DÒ theo phần cứng của máy đang chạy."""
        self.model_name = model_name or config.RERANKER_MODEL_NAME
        self.thiet_bi = thiet_bi or tai_nguyen_gpu.thiet_bi("rerank")
        self._model = self._nap_model()
        logger.info("Model rerank '%s' chạy trên %s.", self.model_name, self.thiet_bi)

    def _nap_model(self) -> CrossEncoder:
        """Nạp model, tự xử lý trường hợp model yêu cầu trust_remote_code."""
        try:
            return CrossEncoder(self.model_name, device=self.thiet_bi)
        except (ValueError, OSError, ImportError) as loi:
            logger.info(
                "Nạp '%s' thất bại (%s) - thử lại với trust_remote_code=True.",
                self.model_name, type(loi).__name__,
            )
            return CrossEncoder(self.model_name, trust_remote_code=True, device=self.thiet_bi)

    def xep_hang(self, cau_hoi: str, cac_doan: List[str]) -> np.ndarray:
        """Chấm điểm liên quan cho từng đoạn so với câu hỏi."""
        if not cac_doan:
            return np.array([])
        return self._model.predict([(cau_hoi, doan) for doan in cac_doan])


def tao_reranker_neu_bat() -> Optional[RerankerService]:
    """Tạo RerankerService khi cấu hình bật, ngược lại trả None."""
    if not config.BAT_RERANK:
        return None
    return RerankerService()
