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
from rag import tai_nguyen_gpu

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = None, thiet_bi: str = None):
        """thiet_bi: "cuda" / "cpu"; bỏ trống thì TỰ DÒ theo phần cứng của máy đang chạy.

        Vì sao phải khai báo tường minh thay vì để sentence-transformers tự chọn: thư viện
        tự chọn ĐÚNG, nhưng nó chọn dựa trên bản PyTorch đang cài. Một máy có GPU mà lỡ cài
        bản `torch` CPU-only sẽ chạy toàn bộ trên CPU, không lỗi, không cảnh báo, chỉ chậm
        hơn khoảng 13 lần (đo được: 512 chunk mất 19,90 s trên CPU so với 1,55 s trên GPU).
        Đi qua tai_nguyen_gpu.thiet_bi() khiến lựa chọn đó được ghi ra log và chỉnh được
        bằng cấu hình, thay vì là một thứ vô hình.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.thiet_bi = thiet_bi or tai_nguyen_gpu.thiet_bi("embedding")
        self._model = SentenceTransformer(self.model_name, device=self.thiet_bi)
        logger.info("Model embedding '%s' chạy trên %s.", self.model_name, self.thiet_bi)

    def chuyen_thiet_bi(self, thiet_bi_moi: str) -> bool:
        """Chuyển model sang thiết bị khác. Trả True nếu thật sự có chuyển.

        Dùng ở ranh giới giai đoạn (rag/tai_nguyen_gpu.py): lúc build index, embedding chạy
        theo lô hàng nghìn chunk nên GPU thắng đậm (12,8×); lúc truy vấn thì mỗi lần chỉ mã
        hoá 1-3 chuỗi ngắn, và ở quy mô đó GPU chỉ nhanh hơn CPU **14 mili giây** (6,1 ms so
        với 20,3 ms) - một con số không ai cảm nhận được bên cạnh câu trả lời hàng chục giây.

        Đổi lại, rời khỏi GPU trả về **1,12 GB VRAM** cho reranker và LLM. Trên card 8 GB thì
        đó là khác biệt giữa "vừa đủ chỗ" và "ba model tranh nhau tới mức phải nạp/nhả liên
        tục" - đo được: khi cả ba cùng nằm trên GPU, VRAM còn trống tụt xuống 288 MB.

        Nuốt lỗi và trả False nếu chuyển không được: giữ nguyên thiết bị cũ luôn là một trạng
        thái hợp lệ, không đáng làm hỏng lần build hay lượt hỏi đang chạy.
        """
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

    # ---------- Mã hoá ----------

    def _encode(self, texts: List[str], tien_to: str) -> np.ndarray:
        """normalize_embeddings=True: chuẩn hóa vector về độ dài 1, để khi FAISS tính
        inner product (IndexFlatIP) thì kết quả chính là cosine similarity.
        float32: FAISS yêu cầu kiểu dữ liệu này cho index.
        """
        if tien_to:
            texts = [tien_to + t for t in texts]
        # Batch size hỏi lại ở MỖI lần encode chứ không chốt lúc khởi tạo: VRAM còn trống
        # thay đổi theo giai đoạn (Ollama nạp/nhả model vision và LLM ngay bên cạnh), nên
        # con số đúng lúc nạp model có thể đã sai vào lúc thật sự encode.
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
