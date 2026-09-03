"""Xếp hạng lại (rerank) bằng cross-encoder - tầng lọc THỨ HAI sau tìm kiếm lai.

Vì sao cần thêm một tầng nữa khi đã có dense + BM25 + RRF:

Embedding (bi-encoder, rag/embedding.py) mã hoá câu hỏi và đoạn văn ĐỘC LẬP thành 2 vector
rồi mới so sánh. Cách đó bắt buộc phải như vậy để tìm kiếm nhanh - vector tài liệu tính sẵn
một lần lúc build index, lúc hỏi chỉ việc so sánh. Nhưng cái giá là model không bao giờ được
nhìn câu hỏi và đoạn văn CÙNG LÚC, nên nó không đánh giá được những quan hệ chỉ lộ ra khi đặt
cạnh nhau: đoạn này có trả lời đúng CÂU HỎI NÀY không, hay chỉ tình cờ cùng chủ đề.

Cross-encoder đọc cả cặp (câu hỏi, đoạn) trong một lượt nên đánh giá đúng hơn hẳn. Đổi lại,
nó KHÔNG thể tính trước: mỗi cặp phải chạy qua model một lần, mỗi lần hỏi lại phải chạy lại.
Chạy nó trên toàn corpus là bất khả thi. Vì vậy kiến trúc chuẩn là 2 tầng: tầng 1 (dense +
BM25) quét nhanh toàn bộ để lấy ra vài chục ứng viên, tầng 2 (cross-encoder) đọc kỹ đúng vài
chục ứng viên đó rồi xếp lại thứ tự.

Đo thực tế trên model đang dùng (BAAI/bge-reranker-v2-m3), cặp liên quan vs cặp lạc đề:
    "Nhà nước là gì?" + đoạn về nhà nước        -> 0.998
    "Nhà nước là gì?" + đoạn về diện tích hình tròn -> 0.000016
Khoảng cách ~60.000 lần, trong khi cosine của cùng 2 cặp đó chỉ chênh nhau khoảng 0.1 điểm.
Độ phân biệt đó là lý do tầng này đáng giá thêm vài giây mỗi câu hỏi.
"""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.RERANKER_MODEL_NAME
        self._model = self._nap_model()

    def _nap_model(self) -> CrossEncoder:
        """Nạp model, tự xử lý trường hợp model yêu cầu trust_remote_code.

        Một số model trên HuggingFace kèm code kiến trúc riêng và chỉ nạp được khi bật cờ
        này. Không thể biết trước model người dùng cấu hình có cần hay không, mà bật sẵn cho
        mọi model thì tức là chạy code tuỳ ý tải từ mạng ngay cả khi không cần. Nên: thử
        KHÔNG bật trước (an toàn hơn), chỉ bật khi model thật sự đòi, và log rõ khi phải bật.
        """
        try:
            return CrossEncoder(self.model_name)
        except (ValueError, OSError, ImportError) as loi:
            logger.info(
                "Nạp '%s' thất bại (%s) - thử lại với trust_remote_code=True.",
                self.model_name, type(loi).__name__,
            )
            return CrossEncoder(self.model_name, trust_remote_code=True)

    def xep_hang(self, cau_hoi: str, cac_doan: List[str]) -> np.ndarray:
        """Chấm điểm liên quan cho từng đoạn so với câu hỏi.

        Trả về mảng điểm SONG SONG với cac_doan (chưa sắp xếp) - việc sắp xếp để chỗ gọi tự
        làm, vì nó còn cần giữ liên kết giữa điểm và vị trí chunk trong index.
        """
        if not cac_doan:
            return np.array([])
        return self._model.predict([(cau_hoi, doan) for doan in cac_doan])


def tao_reranker_neu_bat() -> Optional[RerankerService]:
    """Tạo RerankerService khi cấu hình bật, ngược lại trả None.

    Gom điều kiện vào đây để cả app.py lẫn run_evaluation.py dùng chung một cách bật/tắt,
    và để khi tắt thì model KHÔNG bị nạp (tiết kiệm ~2GB RAM và vài giây khởi động, đúng
    mục đích của việc tắt).
    """
    if not config.BAT_RERANK:
        return None
    return RerankerService()
