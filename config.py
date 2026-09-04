"""Cấu hình tập trung cho toàn bộ hệ thống RAG.

Mọi tham số có thể thay đổi (tên model, chunk size, top_k...) đều nằm ở đây,
đọc được từ biến môi trường / file .env - không hard-code trong logic
của các module khác (đúng yêu cầu ở mục 8).
"""

import logging
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Console mặc định trên Windows (cp1252) không in được tiếng Việt có dấu, sẽ crash
# ngay khi print()/log ra ký tự có dấu. Ép stdout/stderr sang UTF-8 để toàn bộ output
# tiếng Việt (câu trả lời, log cảnh báo, bảng kết quả evaluation...) luôn in được.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình logging tập trung 1 lần ở đây (mọi module khác chỉ cần "import config"
# trước là log của chúng - vd cảnh báo trang/slide rỗng ở document_loader - sẽ hiển thị đẹp.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Các thư viện bên thứ 3 log quá nhiều ở mức INFO (mỗi lần load model in ra hàng chục
# dòng HTTP request tới HuggingFace) - hạ xuống WARNING để log của chính project
# (cảnh báo trang/slide rỗng, tiến trình build index...) không bị chìm trong đó.
for _ten_logger in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub", "faiss.loader"):
    logging.getLogger(_ten_logger).setLevel(logging.WARNING)


def _nap_file_env(duong_dan_env: Path) -> None:
    """Đọc file .env (nếu có) và nạp các dòng KEY=VALUE vào os.environ.

    Tự viết hàm nhỏ này thay vì dùng python-dotenv vì thư viện đó không nằm
    trong danh sách công nghệ đã chốt (mục 7) - nhu cầu chỉ là đọc vài dòng
    KEY=VALUE nên không cần thêm dependency cho việc này.
    """
    if not duong_dan_env.exists():
        return
    for dong in duong_dan_env.read_text(encoding="utf-8").splitlines():
        dong = dong.strip()
        if not dong or dong.startswith("#") or "=" not in dong:
            continue
        key, _, value = dong.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)  # không ghi đè nếu đã set từ bên ngoài


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


# ============================================================
# ĐƯỜNG DẪN DỮ LIỆU
# ============================================================
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw"             # Tài liệu gốc (PDF/PPTX) người dùng upload
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"  # Nơi lưu FAISS index đã build

FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
# FAISS chỉ lưu vector, không lưu được metadata dạng text (tên file, số trang...),
# nên cần 1 file mapping riêng: vị trí trong index -> metadata gốc.
METADATA_MAPPING_FILE = FAISS_INDEX_DIR / "metadata.pkl"
# Ghi kèm "vân tay" cấu hình lúc build index (tên model embedding, chunk size...).
# Vector của câu hỏi chỉ so sánh được với vector tài liệu nếu CÙNG model sinh ra; đổi
# EMBEDDING_MODEL_NAME mà quên build lại index sẽ KHÔNG gây lỗi (2 model có thể cùng số
# chiều) mà chỉ khiến kết quả truy xuất sai một cách âm thầm - đây là loại bug khó phát
# hiện nhất, nên hệ thống tự đối chiếu file này lúc nạp index và cảnh báo (xem app.py).
INDEX_INFO_FILE = FAISS_INDEX_DIR / "index_info.json"
# Ảnh trích ra từ tài liệu, lưu lại để hiển thị kèm trích dẫn (người đọc nhìn thấy đúng
# hình mà câu trả lời dựa vào) và để model vision đọc nếu bật chú thích ảnh.
IMAGES_DIR = DATA_DIR / "images"
# Bộ nhớ đệm của luồng Ingestion: kết quả đọc tài liệu, OCR, chú thích ảnh, embedding - tất
# cả đánh khoá theo BĂM NỘI DUNG (xem rag/bo_nho_dem.py). Xoá cả thư mục này bất cứ lúc nào
# cũng an toàn: mọi thứ trong đó đều tính lại được, chỉ mất thời gian chứ không mất dữ liệu.
CACHE_DIR = DATA_DIR / "cache"

for _thu_muc in (RAW_DOCS_DIR, FAISS_INDEX_DIR, IMAGES_DIR, CACHE_DIR):
    _thu_muc.mkdir(parents=True, exist_ok=True)


# ============================================================
# EMBEDDING MODEL
# ============================================================
# Model đa ngôn ngữ hỗ trợ tiếng Việt, chạy local (không gọi API trả phí).
#
# Vì sao là e5 chứ không phải paraphrase-multilingual-MiniLM (lựa chọn ban đầu):
#   - paraphrase-* được huấn luyện cho bài toán ĐO ĐỘ GIỐNG NHAU giữa 2 câu cùng loại
#     (STS/paraphrase), không phải cho RETRIEVAL (câu hỏi ngắn <-> đoạn tài liệu dài) -
#     đây là 2 bài toán khác nhau, và dùng sai loại model là nguyên nhân gốc khiến truy
#     xuất trên tài liệu dài hay chọn nhầm đoạn "nghe có vẻ giống câu hỏi" thay vì đoạn
#     thật sự chứa câu trả lời.
#   - max_seq_length của paraphrase-MiniLM chỉ 128 token, ép chunk phải rất nhỏ -> nội
#     dung bị băm vụn. e5 cho 512 token, chunk to gấp ~4 lần mà vẫn không bị cắt.
#
# Vì sao BASE chứ không phải SMALL: bản small (384 chiều, ~470MB) đủ tốt khi câu hỏi và tài
# liệu CÙNG ngôn ngữ, nhưng hụt hẳn khi hỏi CHÉO ngôn ngữ - vốn là chuyện xảy ra liên tục
# với corpus trộn tài liệu Anh và Việt (hỏi tiếng Việt về bài giảng tiếng Anh). Đo trên
# corpus thật 5876 chunk, chỉ tính nhánh dense:
#     model     | chiều | MRR cùng ngôn ngữ | MRR chéo ngôn ngữ | giây embed cả corpus
#     e5-small  |  384  |       0.808       |       0.364       |         99
#     e5-base   |  768  |       0.829       |       0.738       |        316
# Chéo ngôn ngữ tăng gấp đôi, cùng ngôn ngữ cũng nhích lên. Cái giá là +217 giây MỘT LẦN
# lúc build index; độ trễ lúc hỏi gần như không đổi vì mỗi câu hỏi chỉ encode đúng 1 chuỗi.
# Với yêu cầu song ngữ của đồ án, đây là đánh đổi rõ ràng đáng giá.
#
# Đổi sang "intfloat/multilingual-e5-small" nếu cần build nhanh hơn / máy yếu RAM và chấp
# nhận truy xuất chéo ngôn ngữ kém hơn. NHỚ build lại index sau khi đổi.
EMBEDDING_MODEL_NAME = _lay_str("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")

# Họ model E5 được huấn luyện BẤT ĐỐI XỨNG: câu hỏi phải thêm tiền tố "query: ", đoạn tài
# liệu phải thêm "passage: ". Thiếu tiền tố này, chất lượng truy xuất tụt rõ rệt (model
# không phân biệt được đâu là câu hỏi, đâu là đoạn văn cần tìm). Tự suy ra theo tên model
# để đổi model khác (không thuộc họ E5) là tiền tố tự tắt, không cần sửa gì thêm.
_LA_HO_E5 = "e5" in EMBEDDING_MODEL_NAME.lower()
EMBEDDING_QUERY_PREFIX = _lay_str("EMBEDDING_QUERY_PREFIX", "query: " if _LA_HO_E5 else "")
EMBEDDING_PASSAGE_PREFIX = _lay_str("EMBEDDING_PASSAGE_PREFIX", "passage: " if _LA_HO_E5 else "")

EMBEDDING_BATCH_SIZE = _lay_int("EMBEDDING_BATCH_SIZE", 64)


# ============================================================
# CHUNKING (Recursive Character Splitting)
# ============================================================
# Đơn vị đo là TOKEN THẬT của chính embedding model (lấy qua tokenizer của model), không
# phải token xấp xỉ của tiktoken như bản trước. Lý do: đã đo trên tài liệu tiếng Việt
# thật, tiktoken (cl100k_base) đếm ra số token GẤP ~1.9 LẦN tokenizer thật của model -
# nghĩa là chunk "100 token" theo tiktoken thực chất chỉ ~40 token với model, tức chỉ dùng
# hết ~31% giới hạn 128 token. Hệ quả: nội dung bị băm vụn quá mức, mỗi chunk không đủ trọn
# ý, retrieval trên tài liệu dài vì thế kém hẳn. Đo bằng tokenizer thật khiến con số dưới
# đây có ý nghĩa đúng như tên gọi. tiktoken vẫn được giữ làm phương án dự phòng khi không
# lấy được tokenizer của model (xem rag/chunking.py).
CHUNK_SIZE_TOKENS = _lay_int("CHUNK_SIZE_TOKENS", 160)
CHUNK_OVERLAP_TOKENS = _lay_int("CHUNK_OVERLAP_TOKENS", 32)
# Chừa biên an toàn dưới max_seq_length của model (model còn phải thêm token đặc biệt
# [CLS]/[SEP] và tiền tố "passage: "). chia_chunk() tự hạ CHUNK_SIZE_TOKENS xuống nếu nó
# vượt (max_seq_length - biên) - nhờ vậy đổi sang model có giới hạn nhỏ hơn cũng không bao
# giờ bị cắt mất nội dung khi encode.
BIEN_AN_TOAN_TOKEN = _lay_int("BIEN_AN_TOAN_TOKEN", 16)

# Thứ tự ưu tiên tách: TIÊU ĐỀ -> đoạn văn -> dòng -> câu -> từ -> ký tự,
# nhằm giữ ngữ nghĩa của chunk tốt nhất có thể trước khi phải tách thô.
#
# Tiêu đề đứng ĐẦU danh sách vì nó là ranh giới ngữ nghĩa mạnh nhất trong tài liệu: hai bên
# một tiêu đề là hai chủ đề khác nhau. Không có nó, splitter chỉ thấy các dấu xuống dòng
# giống hệt nhau nên có thể cắt một chunk vắt ngang 2 mục không liên quan - chunk đó mô tả
# "nửa mục này cộng nửa mục kia", không khớp đúng câu hỏi nào.
CHUNK_SEPARATORS = ["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""]

# --- Nhận diện tiêu đề khi đọc tài liệu ---
# DOCX/PPTX có metadata cấu trúc thật (style "Heading N", placeholder tiêu đề slide) nên
# nhận diện chắc chắn. PDF thì KHÔNG lưu cấu trúc logic - chỉ còn cách suy ra từ hình thức
# (cỡ chữ lớn hơn + dòng ngắn), nên kém tin cậy hơn hẳn; tách cờ riêng để tắt được nếu
# heuristic gây hại trên một loại tài liệu cụ thể.
BAT_NHAN_DIEN_TIEU_DE = _lay_bool("BAT_NHAN_DIEN_TIEU_DE", True)
# Dòng phải lớn hơn cỡ chữ áp đảo của trang bao nhiêu lần thì coi là tiêu đề.
TY_LE_KICH_THUOC_CHU_TIEU_DE = _lay_float("TY_LE_KICH_THUOC_CHU_TIEU_DE", 1.15)
# ...VÀ phải đủ ngắn. Hai điều kiện cùng lúc để không nhận nhầm cả một đoạn văn in cỡ lớn
# (sách khổ lớn, tài liệu cho người cao tuổi) thành tiêu đề.
DO_DAI_TOI_DA_TIEU_DE = _lay_int("DO_DAI_TOI_DA_TIEU_DE", 90)

# Chỉ dùng khi không lấy được tokenizer thật của embedding model (xem trên).
TIKTOKEN_ENCODING = _lay_str("TIKTOKEN_ENCODING", "cl100k_base")


# ============================================================
# RETRIEVAL
# ============================================================
# Số ĐOẠN TRÍCH (mỗi đoạn = 1 vùng nội dung liền mạch quanh chunk khớp nhất) đưa vào prompt.
#
# LỊCH SỬ, vì con số này đã đổi hai lần và mỗi lần vì một lý do KHÁC HẲN nhau:
#
#   8 -> 6: lý do CHẤT LƯỢNG. Bản trước gộp NGUYÊN TRANG nên mỗi đoạn trích ~2000 ký tự,
#     8 đoạn = ~16.000 ký tự ngữ cảnh mà phần lớn không liên quan tới câu hỏi - LLM bị
#     "loãng" và trả lời lệch trọng tâm trên tài liệu dài. Nay mỗi đoạn được cắt gọn quanh
#     đúng chỗ khớp (xem NGAN_SACH_KY_TU_MOI_DOAN) nên 6 đoạn ĐÚNG trọng tâm tốt hơn 8 loãng.
#
#   6 -> 4: lý do TỐC ĐỘ, và đây là một ĐÁNH ĐỔI CÓ MẤT MÁT chứ không phải một cải tiến.
#     Sau khi khai báo OLLAMA_NUM_CTX (§5.60), prefill phải nạp toàn bộ prompt thật thay vì
#     4096 token bị cắt cụt, nên độ trễ tăng thêm ~10-20 giây/câu trên CPU. Hạ TOP_K là cách
#     ĐÚNG để lấy lại tốc độ - cách SAI là hạ num_ctx xuống, vì đó chính là tái tạo lại bug
#     cũ: prompt vẫn dài như thế, chỉ là bị cắt mất phần đầu (tức các đoạn liên quan nhất).
#
#     ĐO TRƯỚC KHI ĐỔI, trên cả hai bộ câu hỏi (12 tài liệu, 5554 chunk, bộ held-out bản
#     22 câu, chỉ đổi TOP_K):
#         bộ         TOP_K   P@K     Recall@K   Recall phủ   MRR     hạng 1   token prompt
#         in-sample     6    0.540     0.875       0.959     0.980    24/25       4401
#         in-sample     4    0.650     0.835       0.933     0.980    24/25       3277
#         held-out      6    0.225     0.850       1.000     0.858    16/20       4520
#         held-out      4    0.287     0.800       0.958     0.858    16/20       3332
#
#     Nhất quán trên CẢ HAI bộ - điều đáng tin hơn nhiều so với chỉ đo in-sample:
#       - Prompt ngắn đi ~25% (4401 -> 3277 token). Đây là thứ mua được tốc độ.
#       - P@K TĂNG (+0.11 / +0.06): bỏ đi 2 đoạn cuối bảng thì phần còn lại đặc hơn. Nghĩa là
#         2 đoạn bị cắt phần lớn là đoạn kém liên quan.
#       - MRR và số câu có đoạn đúng ở HẠNG 1 KHÔNG ĐỔI (0.980 / 24-25 và 0.858 / 16-20).
#         Thứ tự xếp hạng không hề xấu đi - chỉ là lấy ít hơn.
#       - Cái MẤT thật: Recall@K -0.04, Recall phủ -0.026 / -0.042. Một số câu cần tổng hợp
#         từ nhiều trang sẽ thiếu 1 mảnh. Đây là cái giá, không phải hiệu ứng phụ vô hại.
#       - Khả năng chặn câu lạc đề không đổi (2/4 và 1/2).
#
#     Đặt lại 6 nếu ưu tiên độ chính xác hơn tốc độ - máy đủ mạnh (có GPU) thì nên để 6.
#
# LƯU Ý tham số này còn ảnh hưởng gián tiếp tới SO_DOAN_TOI_DA_MOI_TRANG: trần đoạn mỗi
# trang chỉ được áp khi ứng viên trải ra >= TOP_K trang khác nhau, nên hạ TOP_K cũng khiến
# trần được áp thường xuyên hơn.
TOP_K = _lay_int("TOP_K", 4)

# Số ứng viên thô lấy về từ MỖI nhánh tìm kiếm (dense và lexical) trước khi hợp nhất:
# max(TOP_K * HE_SO_OVER_FETCH, SO_UNG_VIEN_TOI_THIEU). Cần lấy dư vì (a) FAISS không lọc
# được metadata lúc search nên việc lọc theo nguồn người dùng chọn phải làm sau, (b) nhiều
# chunk liền kề cùng 1 trang thường chiếm hết top đầu, phải lấy dư mới đủ ứng viên đa dạng.
HE_SO_OVER_FETCH = _lay_int("HE_SO_OVER_FETCH", 10)
SO_UNG_VIEN_TOI_THIEU = _lay_int("SO_UNG_VIEN_TOI_THIEU", 60)

# --- Tìm kiếm lai (hybrid): vector + từ khoá ---
# Chỉ dùng vector (dense) có điểm yếu cố hữu: model nén cả đoạn văn về 1 vector 384 chiều
# nên các từ khoá HIẾM và CỤ THỂ (tên riêng, số điều luật, thuật ngữ, con số) dễ bị "hoà
# tan" - câu hỏi về "Điều 15" có thể trả về đoạn nói chung chung về cùng chủ đề. Với tài
# liệu dài, số đoạn "cùng chủ đề nhưng sai chi tiết" rất nhiều nên lỗi này lộ rõ. BM25
# (khớp từ khoá, có trọng số IDF) mạnh đúng ở chỗ dense yếu, và ngược lại - kết hợp cả 2
# rồi hợp nhất thứ hạng bằng RRF cho kết quả ổn định hơn hẳn từng nhánh riêng lẻ.
# TRỌNG SỐ của nhánh BM25 khi hợp nhất RRF. 0 = tắt hẳn (mặc định), 1.0 = ngang dense.
#
# MẶC ĐỊNH TẮT sau khi đo trên corpus SONG NGỮ thật (13 tài liệu Anh+Việt, 5909 chunk,
# 26 câu hỏi chia 3 nhóm). Đây là kết quả ÂM TÍNH đi ngược kỳ vọng ban đầu, nên ghi lại
# đầy đủ số liệu để không ai bật lại mà không biết cái giá:
#
#              | rerank BẬT                          | rerank TẮT
#   trọng số   | cùng-ngữ  chéo-ngữ  từ-khoá  chung  | cùng-ngữ  chéo-ngữ  từ-khoá  chung
#      0.0     |   0.924    0.703     1.000   0.888  |   0.909    0.407     0.938   0.783
#      0.2     |   0.924    0.656     1.000   0.875  |   0.909    0.276     0.938   0.747
#      1.0     |   0.924    0.536     1.000   0.843  |   0.914    0.203     0.938   0.730
#
# Hai điều bất ngờ:
#   1. BM25 KHÔNG giúp gì ngay trên chính sở trường của nó - câu hỏi toàn từ khoá hiếm
#      ("SIFT", "RANSAC", "Lucas-Kanade", "Mã số sinh viên 2351010039") đạt MRR 1.000 nhờ
#      riêng dense. Model embedding đa ngôn ngữ hiện đại đã xử lý tốt từ hiếm, nên lý do
#      tồn tại kinh điển của BM25 không còn đúng ở đây.
#   2. BM25 gây HẠI nặng cho truy xuất chéo ngôn ngữ (0.703 -> 0.536). Nguyên nhân: câu hỏi
#      tiếng Việt hỏi về tài liệu tiếng Anh thì BM25 không khớp nổi từ nào với tài liệu
#      ĐÚNG, nhưng lại khớp rất "tự tin" với các tài liệu tiếng Việt SAI. RRF coi hạng 1 của
#      BM25 ngang hạng 1 của dense nên đẩy kết quả sai lên trên. Đã kiểm chứng từng bước:
#      "Mô hình camera lỗ kim" - dense xếp đúng hạng 3, sau RRF tụt xuống hạng 13.
#
# Giữ lại code (rag/lexical_search.py) và tham số này thay vì xoá hẳn: corpus khác có thể
# cho kết quả khác (vd corpus THUẦN một ngôn ngữ, hoặc dày mã định danh mà model embedding
# chưa từng thấy). Muốn bật lại thì đo trước bằng thí nghiệm tương tự, đừng bật theo cảm tính.
TRONG_SO_BM25 = _lay_float("TRONG_SO_BM25", 0.0)

# --- BM25 ở vai trò CỨU HỘ (recall-only), tách hẳn khỏi vai trò xếp hạng ở trên ---
# Đọc kỹ lại kết quả âm tính vừa ghi: cái hại đo được KHÔNG phải do BM25 tìm sai, mà do RRF
# cho BM25 quyền XẾP HẠNG ngang dense - hạng 1 của BM25 (một tài liệu tiếng Việt sai) được
# coi ngang hạng 1 của dense. Nghĩa là hướng sửa đúng không phải "bật lại với trọng số 0.2"
# (đã đo là tệ hơn), mà là TÁCH VAI TRÒ:
#   - BM25 chỉ BƠM ỨNG VIÊN vào tập đưa đi rerank (recall), KHÔNG đóng góp điểm RRF nào.
#   - Cross-encoder - vốn đã đo là phân biệt tốt gấp ~60.000 lần cosine - quyết định thứ
#     hạng cuối. Đoạn cứu hộ thật sự liên quan sẽ được nó đẩy lên; không liên quan thì nằm
#     yên ở cuối và không ảnh hưởng gì.
#
# Vì sao vẫn đáng làm dù BM25 đã bị đo là vô ích trên corpus cũ: chính comment ở trên đã ghi
# "corpus khác có thể cho kết quả khác (vd corpus dày mã định danh mà model embedding chưa
# từng thấy)". Tài liệu MỚI mà hệ thống chưa từng nhìn thấy - tên riêng, mã hiệu, thuật ngữ
# OOV - đúng là trường hợp đó.
#
# Trường hợp xấu nhất THEO THIẾT KẾ là "không cải thiện", không phải "làm hỏng": các đoạn
# cứu hộ vào tập ứng viên với điểm RRF = 0 nên tự chúng không đẩy được gì lên. Dù vậy vẫn
# ĐÃ ĐO LẠI bằng đúng thí nghiệm đó (12 tài liệu, 5554 chunk, 29 câu hỏi song ngữ, chỉ đổi
# đúng tham số này):
#     tắt cứu hộ   P@K 0.500   Recall@K 0.937   MRR 0.980   chặn lạc đề 2/4
#     cứu hộ 10    P@K 0.507   Recall@K 0.945   MRR 0.980   chặn lạc đề 2/4
# Cải thiện NHỎ (+0.008 Recall@K) - đúng mức kỳ vọng, vì corpus này đã được đo là không cần
# BM25. Điều đáng giá hơn con số: nó KHÔNG làm hỏng gì, kể cả truy xuất chéo ngôn ngữ vốn là
# chỗ TRONG_SO_BM25 gây hại nặng (MRR giữ nguyên 0.980). Đó là bằng chứng cho thấy cái hại
# đo được trước đây đến từ QUYỀN XẾP HẠNG chứ không phải từ bản thân BM25.
# Đặt 0 để tắt hẳn.
SO_UNG_VIEN_BM25_CUU_HO = _lay_int("SO_UNG_VIEN_BM25_CUU_HO", 10)
# Hằng số của Reciprocal Rank Fusion: diem = sum(1 / (RRF_K + thu_hang)). Hợp nhất theo
# THỨ HẠNG chứ không theo điểm số vì điểm cosine và điểm BM25 không cùng thang đo, cộng
# thẳng vào nhau là vô nghĩa. 60 là giá trị chuẩn trong bài báo gốc về RRF.
RRF_K = _lay_int("RRF_K", 60)

# --- Dựng đoạn trích quanh chunk khớp ("small-to-big") ---
# Ngân sách độ dài (ký tự) cho mỗi đoạn trích: bắt đầu từ chunk khớp nhất rồi mở rộng dần
# sang các chunk liền kề CÙNG TRANG cho tới khi chạm ngân sách. Giải quyết cùng lúc 2 việc:
#   - Câu/đoạn liệt kê bị chunking cắt ngang vẫn được nối lại đủ ý (lý do ban đầu của việc
#     gộp nguyên trang).
#   - Nhưng KHÔNG kéo theo cả trang: trang giáo trình dày ~13 chunk, gộp hết thì 90% nội
#     dung đưa vào prompt là chuyện khác, vừa làm loãng câu trả lời vừa khiến đoạn trích
#     hiển thị (cắt từ đầu đoạn gộp) trỏ sai chỗ so với nội dung thật sự được dùng.
NGAN_SACH_KY_TU_MOI_DOAN = _lay_int("NGAN_SACH_KY_TU_MOI_DOAN", 1600)

# --- Mở rộng đoạn trích QUA ranh giới trang ---
# Bản trước chặn cứng việc mở rộng trong đúng một (nguồn, trang). Với SLIDE thì đúng: mỗi
# slide là một đơn vị nội dung tự đóng. Với PDF VĂN BẢN CHẢY LIÊN TỤC thì sai hẳn: một định
# nghĩa bắt đầu ở cuối trang 12 và kết thúc ở đầu trang 13 sẽ KHÔNG BAO GIỜ được nối lại -
# chunk neo nằm cuối trang 12, mở rộng sang phải chạm hết mảng của trang rồi dừng.
#
# Lại đúng cái pattern "chỉ lộ ra trên tài liệu mới": corpus cũ nhiều slide, corpus mới
# nhiều PDF văn xuôi.
#
# ĐO ĐẠC - và đây là chỗ suýt dẫn tới kết luận sai, nên ghi lại đầy đủ:
#     cấu hình                  P@K     Recall@K (neo)   Recall PHỦ
#     TẮT mở rộng xuyên trang   0.580       0.945          0.945
#     BẬT mở rộng xuyên trang   0.540       0.875          0.959
# Theo Recall@K thì thay đổi này LÀM TỆ ĐI 0.07 và đáng lẽ phải bỏ. Nhưng recall_tai_k() so
# khớp theo TRANG NEO: khi một đoạn trích nuốt sang trang liền kề, trang đó không còn được
# neo riêng nữa dù nội dung VẪN nằm trong ngữ cảnh gửi cho LLM. Đo bằng "Recall phủ" (trang
# đúng có nằm trong các trang mà đoạn trích ĐI QUA không) thì thay đổi này cải thiện nhẹ, và
# chi tiết từng câu xác nhận: mọi câu tụt Recall neo đều giữ Recall phủ = 1.00.
# Bài học ở §5.65: một thay đổi phá vỡ giả định của metric sẽ LUÔN trông như hồi quy.
# Hệ quả phải nhớ: Recall@K trước/sau thay đổi này KHÔNG so trực tiếp được nữa.
#
# Trích dẫn: số trang ghi ra vẫn là trang của chunk NEO (phần thật sự khớp câu hỏi), nhưng
# giao diện ghi ĐỦ khoảng trang mà đoạn trích đi qua ("trang/slide 12-13") - vì câu trả lời
# có thể dựa vào nội dung ở trang bên cạnh, và trỏ thiếu cũng là một kiểu trích dẫn sai.
MO_RONG_QUA_RANH_GIOI_TRANG = _lay_bool("MO_RONG_QUA_RANH_GIOI_TRANG", True)

# Số trang tối đa được vượt qua ở MỖI HƯỚNG khi mở rộng. 1 đủ cho trường hợp thật (một
# đoạn bị ranh giới trang cắt đôi thì phần còn thiếu nằm ở trang liền kề, không xa hơn),
# và là chốt chặn cho SLIDE: không có nó, một slide thưa chữ sẽ hút thêm 2-3 slide xung
# quanh cho đầy ngân sách 1600 ký tự, tức tái tạo lại đúng lỗi "ngữ cảnh loãng" mà việc bỏ
# cách gộp-nguyên-trang đã sửa. Ngân sách ký tự vẫn là chốt chặn chính.
SO_TRANG_TOI_DA_MO_RONG = _lay_int("SO_TRANG_TOI_DA_MO_RONG", 1)

# Trần số đoạn trích lấy từ cùng 1 trang - tránh việc cả TOP_K đoạn đều dồn vào 1 trang
# duy nhất (rất hay xảy ra với tài liệu dài: các chunk liền kề có điểm gần bằng nhau),
# khiến những trang liên quan khác không còn suất nào.
#
# TRẦN NÀY THÍCH ỨNG, không còn là hằng số. Lý do tồn tại của nó (chống các chunk liền kề
# trong giáo trình 230 trang chiếm hết TOP_K) chỉ đúng KHI CÓ nhiều trang để phân bổ. Với
# câu hỏi mà toàn bộ câu trả lời nằm gọn trong MỘT trang - rất phổ biến: một mục định nghĩa,
# một bảng tiêu chí, một quy trình - trần cứng chỉ cho lấy 2 đoạn từ đúng trang chứa câu trả
# lời, bốn suất còn lại đi cho những trang kém liên quan. Ngữ cảnh vừa THIẾU phần đúng vừa
# LOÃNG vì phần sai, và câu trả lời ngắn đi vì model không đủ nguyên liệu.
# Cách áp dụng: xem `tran_moi_trang` trong rag/rag_pipeline.py.truy_xuat.
#
# Đo trên corpus thật (12 tài liệu, 5554 chunk, 29 câu hỏi, chỉ đổi đúng tham số này):
#     trần CỨNG (như cũ)   P@K 0.500   Recall@K 0.937   MRR 0.980
#     trần THÍCH ỨNG       P@K 0.567   Recall@K 0.937   MRR 0.980
# +0.067 P@K mà không đụng tới MRR lẫn khả năng chặn câu lạc đề (2/4 ở cả hai cấu hình).
SO_DOAN_TOI_DA_MOI_TRANG = _lay_int("SO_DOAN_TOI_DA_MOI_TRANG", 2)

# Số ứng viên đầu bảng dùng để ĐO độ đa dạng trang trước khi quyết định có áp trần hay
# không. Phải lớn hơn TOP_K kha khá (nếu chỉ nhìn đúng TOP_K ứng viên đầu thì phép đo bị
# chính cái trần đang xét làm nhiễu) nhưng nhỏ hơn hẳn tổng số ứng viên (nhìn quá xa xuống
# đuôi thì trang nào cũng xuất hiện, phép đo luôn ra "đa dạng" và trần luôn được áp).
SO_UNG_VIEN_XET_DA_DANG_TRANG = _lay_int("SO_UNG_VIEN_XET_DA_DANG_TRANG", 20)

# Trần số ĐOẠN LÀ ẢNH trong kết quả. Cùng lý do với trần theo trang ở trên, nhưng cho một
# kiểu lấn át khác: khi bật chú thích ảnh bằng vision, mỗi ảnh có một đoạn mô tả dài vài
# trăm ký tự, và với tài liệu nhiều hình (corpus thật của đồ án: 303 ảnh trên 5854 chunk)
# các bản ghi ảnh bắt đầu chiếm hết suất TOP_K của những trang văn bản đúng.
#
# Đo thực tế trên corpus thật (cùng bộ câu hỏi, cùng nhãn, chỉ đổi trần này):
#     cấu hình                    Recall@K   slide tiếng Anh   ảnh có tìm được không
#     tắt vision (mốc so sánh)      0.96          0.97          KHÔNG
#     vision, không có trần         0.92          0.86          có
#     vision, trần 2                0.93          0.90          có
#     vision, trần 1                0.95          0.95          có
# Không có trần thì tính năng tìm được nội dung TRONG HÌNH lại làm hỏng việc tìm nội dung
# TRONG CHỮ - một đánh đổi không đáng, và sẽ không ai nhận ra nếu chỉ nhìn vào việc "ảnh giờ
# tìm được rồi". Chọn 1: gần như khôi phục hoàn toàn phần văn bản (0.95 so với mốc 0.96) mà
# hình liên quan vẫn xếp hạng 1 cho câu hỏi về nội dung hình - đã kiểm chứng riêng.
SO_DOAN_ANH_TOI_DA = _lay_int("SO_DOAN_ANH_TOI_DA", 1)

# --- Trích xuất hình ảnh ---
# Lấy ảnh ra khỏi tài liệu và gắn với văn bản lân cận (chú thích "Hình 3: ..."), để nội
# dung hình tìm được qua chú thích. Rẻ (không cần model nào) nên bật mặc định; tài liệu
# thuần văn bản không có ảnh thì bước này gần như không tốn gì.
BAT_TRICH_ANH = _lay_bool("BAT_TRICH_ANH", True)

# --- Chú thích ảnh bằng model vision ---
# Chú thích lân cận chỉ cho biết hình TÊN là gì; model vision đọc được nội dung BÊN TRONG
# hình (nhãn trong sơ đồ, số trên biểu đồ, chữ trong ảnh chụp bảng) - xem rag/vision_caption.py.
#
# MẶC ĐỊNH BẬT. Ước lượng chi phí ban đầu (~30 giây/hình) hoá ra SAI: con số đó đo trên
# đúng một lượt gọi, tức đã tính cả thời gian nạp model vào bộ nhớ. Đo lại trên ảnh thật của
# corpus khi model đã nạp sẵn: 0.7 - 2.9 giây/hình, TRUNG BÌNH 1.9 giây. Với 291 ảnh chỉ tốn
# ~9 phút, không phải 2.4 tiếng như tưởng - rẻ tới mức không có lý do gì để tắt.
#
# Bài học: một phép đo duy nhất, kèm chi phí khởi động một lần, suýt khiến cả tính năng bị
# xếp vào loại "quá đắt, để sau".
#
# Tắt (0) nếu chưa pull model vision, hoặc tài liệu thuần văn bản không có hình nào (lúc đó
# bước này vốn cũng không chạy). Nhớ pull model trước: ollama pull qwen2.5vl:3b
BAT_CHU_THICH_ANH = _lay_bool("BAT_CHU_THICH_ANH", True)
VISION_MODEL_NAME = _lay_str("VISION_MODEL_NAME", "qwen2.5vl:3b")
# Mô tả hình cần ngắn gọn, đủ để tìm kiếm - không cần dài như câu trả lời cho người dùng.
VISION_NUM_PREDICT = _lay_int("VISION_NUM_PREDICT", 400)


# --- Xếp hạng lại bằng cross-encoder (rerank) ---
# Tầng lọc THỨ HAI sau tìm kiếm lai: dense+BM25 quét nhanh lấy vài chục ứng viên, rồi
# cross-encoder đọc kỹ từng cặp (câu hỏi, đoạn) để xếp lại thứ tự. Giải thích đầy đủ vì sao
# cần 2 tầng nằm ở docstring rag/reranker.py.
#
# Đây là đánh đổi CÓ CHỦ ĐÍCH giữa độ chính xác và độ trễ: cross-encoder phải chạy mỗi cặp
# mỗi lần hỏi (không tính trước được như vector), nên tốn thêm vài giây mỗi câu hỏi.
#
# Đo thực tế trên bộ tài liệu mẫu (47 chunk), với câu hỏi DIỄN ĐẠT LẠI - không chứa từ khoá
# hiếm để BM25 bám vào, tức đúng loại câu mà tìm kiếm từ khoá bó tay:
#     tắt rerank      : MRR 0.417, đúng-hạng-1  3/12 câu,  0.02 giây
#     8 ứng viên      : MRR 0.542, đúng-hạng-1  5/12 câu,  1.47 giây
#     15 ứng viên     : MRR 0.542, đúng-hạng-1  5/12 câu,  2.62 giây
#     30 ứng viên     : MRR 0.642, đúng-hạng-1  6/12 câu,  6.07 giây
# Chọn 30 vì độ chính xác được ưu tiên và +6 giây chỉ chiếm ~15% tổng thời gian trả lời
# (riêng phần LLM sinh câu trả lời đã tốn ~40 giây trên CPU). Hạ xuống 8 nếu cần nhanh:
# vẫn giữ được khoảng 60% mức cải thiện với 24% chi phí.
#
# LƯU Ý về cách đo: với câu hỏi CÓ chứa mã/từ khoá hiếm ("hồ sơ loại D1"), BM25 đã đưa đúng
# đoạn lên hạng 1 nên rerank không cải thiện được gì thêm (đo được MRR 1.00 ở cả 2 chế độ).
# Lợi ích của rerank chỉ lộ ra ở câu hỏi diễn đạt tự nhiên - đó mới là cách người dùng hỏi.
BAT_RERANK = _lay_bool("BAT_RERANK", True)
RERANKER_MODEL_NAME = _lay_str("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
# Số ứng viên (đã qua RRF) đưa vào cross-encoder. Nhỏ hơn hẳn SO_UNG_VIEN_TOI_THIEU vì mỗi
# ứng viên ở đây tốn một lượt chạy model thật, không phải một phép nhân vector.
SO_UNG_VIEN_RERANK = _lay_int("SO_UNG_VIEN_RERANK", 30)


# --- Sàn lọc rác ---
# Điểm cosine tối thiểu để 1 đoạn được đưa vào ngữ cảnh. Dưới sàn này, hệ thống trả lời
# "Không tìm thấy thông tin trong tài liệu." mà KHÔNG gọi LLM.
#
# Cố ý đặt THẤP, chỉ đủ để chặn rác - KHÔNG dùng làm cơ chế chính để phát hiện câu hỏi
# ngoài phạm vi tài liệu. Lý do đến từ số đo thật trên giáo trình tiếng Việt 230 trang:
#     câu hỏi ĐÚNG chủ đề, hỏi bằng tiếng Việt : cosine top ~0.89 - 0.92
#     câu hỏi ĐÚNG chủ đề, hỏi bằng tiếng Anh  : cosine top ~0.79   <-- thấp hơn hẳn
#     câu hỏi LẠC ĐỀ, hỏi bằng tiếng Việt      : cosine top ~0.78 - 0.83
# Câu tiếng Anh đúng chủ đề cho điểm THẤP HƠN câu tiếng Việt lạc đề, vì tài liệu viết bằng
# tiếng Việt nên độ tương đồng xuyên ngôn ngữ luôn bị thiệt. Nghĩa là KHÔNG tồn tại ngưỡng
# nào vừa nhận đúng câu tiếng Anh hợp lệ vừa loại được câu tiếng Việt lạc đề - đặt ngưỡng
# cao sẽ âm thầm phá vỡ yêu cầu song ngữ của đồ án. (Chênh lệch giữa điểm cao nhất và điểm
# trung bình của nhóm ứng viên cũng đã được đo và cũng không tách được 2 nhóm.)
# Việc phán đoán "tài liệu có nói về chuyện này không" vì thế được giao cho LLM - nó ĐỌC
# được nội dung đoạn trích chứ không chỉ nhìn một con số, nên đủ căn cứ hơn hẳn; các quy
# tắc bắt buộc từ chối / bắt buộc kết luận "TÀI LIỆU KHÔNG ĐỀ CẬP" nằm trong system prompt
# ở rag/rag_pipeline.py.
#
# HẠ TỪ 0.70 XUỐNG 0.50 và chuyển vai trò chính sang ngưỡng TƯƠNG ĐỐI ngay bên dưới. Lý do
# nằm ở một sai lầm đo lường, không phải ở con số: cosine của E5 KHÔNG phải thang đo tuyệt
# đối. Giá trị tuyệt đối phụ thuộc domain, ngôn ngữ, độ dài chunk, phong cách văn bản - một
# corpus mới (nhiều bảng, nhiều công thức, nhiều số) dịch cả phân bố xuống, và ngưỡng cố
# định hiệu chỉnh trên corpus cũ bắt đầu cắt oan. Chunk là mảnh bảng hay chú thích ảnh vốn
# đã cho cosine thấp hơn văn xuôi ngay cả trong cùng một corpus.
#
# Hậu quả cụ thể đã gặp: 4/6 đoạn rớt ngưỡng -> ngữ cảnh chỉ còn 2 đoạn -> không còn gì để
# tổng hợp -> câu trả lời ngắn. Triệu chứng trông giống hệt bug num_ctx nhưng là nguyên
# nhân khác, nên phải sửa cả hai chứ không phải chọn một.
#
# 0.50 giờ chỉ còn đúng vai trò CHẶN RÁC như comment ở trên vẫn mô tả (đoạn hoàn toàn
# không liên quan cho cosine ~0.3-0.5 với E5), không còn là cơ chế quyết định.
NGUONG_DIEM_TOI_THIEU = _lay_float("NGUONG_DIEM_TOI_THIEU", 0.50 if _LA_HO_E5 else 0.15)

# --- Ngưỡng TƯƠNG ĐỐI: giữ đoạn không quá kém so với đoạn TỐT NHẤT CỦA CHÍNH LƯỢT ĐÓ ---
# Phân bố cosine dịch theo domain; TỶ LỆ giữa các đoạn TRONG CÙNG một lượt thì không. Vì
# vậy "kém hơn đoạn tốt nhất bao nhiêu" là đại lượng so sánh được giữa các corpus, còn
# "cosine bằng bao nhiêu" thì không.
#
# Đo trên corpus thật (12 tài liệu, 5554 chunk, 29 câu hỏi): P@K 0.500 -> 0.500,
# Recall@K 0.937 -> 0.937, MRR 0.980 -> 0.980 - KHÔNG ĐỔI một chữ số nào. Đó đúng là kết quả
# mong muốn chứ không phải kết quả đáng thất vọng: thay đổi này được thiết kế để trung tính
# trên corpus đã hiệu chỉnh (không được phép làm tệ đi chỗ đang chạy tốt), và chỉ phát huy
# tác dụng ở corpus có phân bố cosine khác. Ghi rõ là nó CHƯA được chứng minh có lợi ở đâu.
#
# 0.78 KHÔNG phải con số chọn theo cảm tính, mà là chính điểm hiệu chỉnh cũ viết lại theo
# thang tương đối: trên corpus đã dùng để chỉnh, câu hỏi đúng chủ đề cho cosine cao nhất
# ~0.90 và ngưỡng tuyệt đối cũ là 0.70, tức 0.70/0.90 = 0.78. Nghĩa là trên corpus cũ thay
# đổi này gần như KHÔNG đổi hành vi (đúng thứ ta muốn - không được phép làm tệ đi chỗ đang
# chạy tốt), còn trên corpus mới có phân bố thấp hơn thì ngưỡng tự trôi xuống theo thay vì
# cắt oan.
#
# Cố ý KHÔNG chọn một tỷ lệ chặt hơn (0.90-0.92) dù nghe hợp lý: cosine của E5 nén hết vào
# dải hẹp 0.75-0.92, nên 0.92 tương đương sàn tuyệt đối ~0.83 trên corpus cũ - CHẶT HƠN hẳn
# 0.70 hiện tại và sẽ làm trầm trọng thêm đúng triệu chứng "thiếu đoạn để tổng hợp" mà thay
# đổi này sinh ra để chữa. Đặt 0 để tắt hẳn tầng lọc tương đối.
TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT = _lay_float("TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT", 0.78)

# Ghi log phân bố điểm của từng lượt truy xuất (điểm cosine của các đoạn, điểm rerank cao
# nhất, số đoạn sống sót sau mỗi tầng lọc). Bật khi cần TRẢ LỜI câu hỏi "ngưỡng có đang cắt
# oan trên corpus này không" bằng số đo thay vì bằng cảm giác - đúng nguyên tắc đã áp dụng
# khi đo BM25. Mặc định tắt vì mỗi câu hỏi in thêm một dòng dài.
LOG_PHAN_BO_DIEM = _lay_bool("LOG_PHAN_BO_DIEM", False)

# --- Ngưỡng từ chối dựa trên điểm RERANK ---
# Thứ mà ngưỡng cosine ở trên KHÔNG làm được, điểm rerank lại làm được. Đo bằng
# evaluation/do_nguong_rerank.py trên bộ tài liệu mẫu (16 câu đúng chủ đề VI+EN, 6 câu lạc đề):
#     nhóm câu hỏi              cosine (min-max)      rerank (min-max)
#     đúng chủ đề, tiếng Việt   0.865 - 0.905         0.200 - 0.996
#     đúng chủ đề, tiếng Anh    0.766 - 0.816         0.019 - 0.861
#     lạc đề, tiếng Việt        0.780 - 0.828         0.000 - 0.003
# Theo cosine, câu tiếng Anh ĐÚNG chủ đề (0.816) còn thấp hơn câu tiếng Việt LẠC ĐỀ (0.828)
# nên không tồn tại ngưỡng hợp lệ. Theo rerank, câu lạc đề rơi về gần 0 tuyệt đối trong khi
# câu đúng chủ đề thấp nhất vẫn là 0.019 - tách được. Lý do: cross-encoder đọc CẢ CẶP (câu
# hỏi, đoạn) nên nó trả lời được "đoạn này có đáp ứng câu hỏi này không", còn cosine chỉ đo
# "hai đoạn text có giống nhau không" - mà một câu hỏi tiếng Anh thì không "giống" đoạn văn
# tiếng Việt nào cả, dù nội dung đúng.
#
# Hai loại sai KHÔNG NGANG GIÁ, nên ngưỡng cố tình nghiêng hẳn về phía an toàn cho câu hợp lệ:
#   - Từ chối nhầm câu hợp lệ = hỏng thấy ngay, người dùng mất hẳn câu trả lời.
#   - Bỏ lọt câu lạc đề = rơi xuống quy tắc từ chối trong system prompt (vốn đang chạy tốt),
#     tức chỉ mất tuyến phòng thủ thứ hai chứ không hỏng gì.
#
# HẠ TỪ 0.005 XUỐNG 0.001 sau khi đo trên corpus SONG NGỮ thật (13 tài liệu Anh+Việt):
# ngưỡng 0.005 vốn hiệu chỉnh trên corpus thuần tiếng Việt, khi gặp câu hỏi CHÉO NGÔN NGỮ
# (hỏi tiếng Việt về tài liệu tiếng Anh) thì chính nó lại từ chối oan câu hợp lệ:
#     "Luồng quang học được tính như thế nào?"  -> 0.0023  BỊ TỪ CHỐI OAN (đáp án ở CV-06)
#     "Phân rã độ chệch và phương sai là gì?"   -> 0.0053  sát mép, suýt bị từ chối
#     câu lạc đề thật ("nấu phở", "World Cup")  -> 0.0002 - 0.0015
# Cross-encoder chấm cặp khác ngôn ngữ thấp hơn hẳn cặp cùng ngôn ngữ, nên vùng điểm của
# "câu hợp lệ hỏi chéo ngôn ngữ" chồng lấn vùng của "câu lạc đề". Không có ngưỡng nào tách
# sạch được cả hai; giữ ngưỡng cao đồng nghĩa hy sinh yêu cầu song ngữ của đồ án.
# 0.001 vẫn chặn được nhóm lạc đề rõ ràng nhất (điểm ~0.0002-0.0005) mà không đụng tới câu
# chéo ngôn ngữ hợp lệ. Phần còn lại giao cho quy tắc từ chối của LLM - nó ĐỌC được nội dung
# nên phán đoán tốt hơn một con số. Đặt 0 để tắt hẳn tuyến phòng thủ này.
NGUONG_DIEM_RERANK_TOI_THIEU = _lay_float("NGUONG_DIEM_RERANK_TOI_THIEU", 0.001)


# ============================================================
# LLM (Ollama - local, không dùng API trả phí)
# ============================================================
OLLAMA_MODEL = _lay_str("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_HOST = _lay_str("OLLAMA_HOST", "http://localhost:11434")
# Nhiệt độ thấp để câu trả lời bám sát context đã truy xuất, hạn chế việc "bịa" thông tin.
OLLAMA_TEMPERATURE = _lay_float("OLLAMA_TEMPERATURE", 0.1)
# Giới hạn số token tối đa được sinh - an toàn tốc độ (tránh model lặp/sinh tràn lan không
# điểm dừng), KHÔNG phải giới hạn chất lượng câu trả lời. Đặt rất cao vì OLLAMA_MODEL mặc
# định (qwen3:4b) có chế độ "thinking" - phần suy luận nội bộ trước khi ra câu trả lời thật
# tính CHUNG vào num_predict, và độ dài thinking dao động mạnh theo độ khó câu hỏi: đã đo
# thực tế câu hỏi yêu cầu "liệt kê đầy đủ" khiến riêng phần thinking tốn hơn 4000 token,
# từng làm model bị cắt ngay giữa câu trả lời thật (câu trả lời cụt, không có lỗi báo ra).
# 12000 đủ dư an toàn cho cả thinking dài lẫn câu trả lời chi tiết.
#
# ĐÍNH CHÍNH một giả định sai đã nằm ở đây rất lâu: bản trước viết "context window của model
# rất lớn (262144 token) nên không có lý do phải cắt sớm hơn". 262144 là năng lực KIẾN TRÚC
# của qwen3:4b; ngân sách RUNTIME mà Ollama thực sự cấp phát lại chỉ là 4096 token nếu không
# khai báo num_ctx. Xem OLLAMA_NUM_CTX ngay bên dưới - đó mới là con số quyết định, và
# num_predict=12000 chưa bao giờ với tới được vì trần thật nằm thấp hơn nhiều.
OLLAMA_NUM_PREDICT = _lay_int("OLLAMA_NUM_PREDICT", 12000)

# --- Cửa sổ ngữ cảnh (num_ctx) ---
# PHẢI khai báo TƯỜNG MINH. Ollama mặc định cấp cho model một cửa sổ 4096 token bất kể model
# hỗ trợ bao nhiêu, và khi prompt vượt quá thì nó KHÔNG báo lỗi - nó cắt im lặng.
#
# Đây là bug nghiêm trọng nhất từng có trong hệ thống này, và nó gây ra HAI triệu chứng
# trông như hai lỗi khác nhau:
#
#   (a) CÂU TRẢ LỜI NGẮN CỤT. Đây là cơ chế ĐÃ ĐO ĐƯỢC, không phải suy luận. Cửa sổ ngữ
#       cảnh chứa prompt + thinking + câu trả lời, mà qwen3:4b luôn sinh thinking rất dài.
#       Đo bằng prompt_eval_count/eval_count THẬT của Ollama (bộ slide 740 chunk):
#           câu hỏi                TOP_K   prompt   sinh ra    TỔNG
#           thường                   6      3394     3607      7001
#           thường                   4      2430     2775      5205
#           "liệt kê ĐẦY ĐỦ ..."     6      3646     7214     10860
#           "liệt kê ĐẦY ĐỦ ..."     4      2660     5498      8158
#       Prompt một mình KHÔNG vượt 4096 trên corpus này, nhưng TỔNG thì vượt rất xa - tới
#       2,7 lần với câu yêu cầu liệt kê đầy đủ. Với num_ctx mặc định, model viết được vài
#       dòng rồi chạm trần và dừng.
#
#       ĐÍNH CHÍNH kèm theo: ước lượng ban đầu dùng 2.5 ký tự/token và ra ~4900 token cho
#       prompt. Đo thật thì tokenizer Qwen đạt 2.93-3.14 ký tự/token, tức prompt ngắn hơn
#       ước lượng ~20%. Con số sai không đổi kết luận (num_ctx vẫn phải khai báo) nhưng đổi
#       CƠ CHẾ: thủ phạm ở corpus này là cắt phần SINH, không phải cắt prompt.
#
#   (b) TRUY XUẤT TRÔNG NHƯ "KÉM" TRÊN TÀI LIỆU MỚI. Cơ chế này SUY LUẬN ra, chưa đo được
#       trên corpus hiện tại (prompt ở đây mới 3400-3600 token); nó cần tài liệu dày chữ hơn
#       hoặc TOP_K lớn hơn để prompt tự nó vượt 4096. Khi prompt vượt num_ctx, Ollama giữ
#       system message và cắt từ ĐẦU phần user content. Mà _ghep_prompt() xếp đoạn trích
#       TỐT NHẤT TRƯỚC -> phần bị xoá chính xác là [1], [2], tức đoạn liên quan nhất.
#       Retrieval tìm đúng, LLM chỉ không bao giờ được nhìn thấy kết quả đó.
#
# Vì sao chỉ lộ ra trên tài liệu mới: corpus cũ nhiều slide và trang thưa chữ nên mỗi đoạn
# trích thực tế ngắn hơn nhiều so với trần 1600, prompt tổng chỉ ~2500 token và vừa lọt
# 4096. Tài liệu mới dày chữ -> mỗi đoạn CHẠM trần -> prompt phình lên ~4900 -> bắt đầu bị
# cắt. Hệ thống không "kém hơn trên tài liệu mới"; nó luôn có bug này, tài liệu cũ chỉ tình
# cờ nằm dưới ngưỡng gây lỗi.
#
# CÁI GIÁ, ghi ra để không ai bất ngờ: KV-cache tỉ lệ TUYẾN TÍNH với num_ctx. 16384 với
# qwen3:4b tốn thêm vài trăm MB RAM và làm prefill chậm hơn (trên CPU, prefill ~5000 token
# có thể mất 10-20 giây). Nếu quá chậm thì hạ TOP_K hoặc NGAN_SACH_KY_TU_MOI_DOAN, TUYỆT
# ĐỐI không hạ num_ctx xuống dưới độ dài prompt - làm vậy là quay lại đúng bug này.
OLLAMA_NUM_CTX = _lay_int("OLLAMA_NUM_CTX", 16384)

# Trần cứng khi cửa sổ được nới ĐỘNG (xem _uoc_luong_num_ctx trong rag/rag_pipeline.py).
# Chốt chặn RAM: nới vô hạn theo prompt thì một cấu hình TOP_K quá tay sẽ âm thầm ngốn hết
# bộ nhớ máy thay vì báo cho người dùng biết là cấu hình sai.
OLLAMA_NUM_CTX_TOI_DA = _lay_int("OLLAMA_NUM_CTX_TOI_DA", 32768)

# Ngân sách token DÀNH RIÊNG cho thinking + câu trả lời, cộng thêm vào độ dài prompt khi
# tính num_ctx động. Không phải con số tuỳ tiện: đã đo qwen3:4b sinh hơn 4000 token thinking
# cho câu hỏi yêu cầu "liệt kê đầy đủ" (xem OLLAMA_NUM_PREDICT), nên dưới mức này thì đúng
# loại câu hỏi cần câu trả lời dài nhất lại là loại bị cắt trước tiên.
OLLAMA_DU_PHONG_TOKEN_SINH = _lay_int("OLLAMA_DU_PHONG_TOKEN_SINH", 4000)

# Số ký tự trên một token, dùng để ƯỚC LƯỢNG độ dài prompt mà không cần nạp tokenizer của
# LLM (hệ thống chỉ có tokenizer của embedding model, vốn là XLM-R chứ không phải Qwen).
# Đo thực tế trên chính corpus của đồ án bằng prompt_eval_count của Ollama: tokenizer Qwen
# đạt 2.93-3.14 ký tự/token cho prompt tiếng Việt (KHÔNG phải 2.5 như ước lượng ban đầu).
# Vẫn giữ 2.2 chứ không nâng lên 3.0: ước lượng này chỉ dùng để CẤP PHÁT ngân sách, nên sai
# về phía cấp DƯ hoàn toàn vô hại, còn sai về phía cấp THIẾU thì tái tạo lại đúng bug trên.
# 2.2 cho biên an toàn ~40% so với giá trị đo được - đủ để một tài liệu có cách token hoá bất
# lợi hơn (nhiều công thức, nhiều mã định danh, ngôn ngữ khác) không âm thầm phá vỡ giả định.
SO_KY_TU_MOI_TOKEN_UOC_LUONG = _lay_float("SO_KY_TU_MOI_TOKEN_UOC_LUONG", 2.2)

# Yêu cầu RÕ RÀNG chế độ suy luận nội bộ ("thinking") cho câu hỏi dạng kiểm chứng một khẳng
# định ("... đúng không?", "có phải ... không?"). Phát hiện MÂU THUẪN giữa lời khẳng định của
# người dùng và tài liệu là việc đòi hỏi đối chiếu nhiều bước, khác hẳn câu hỏi thường (chỉ
# cần đọc và thuật lại ngữ cảnh) - không có bước suy luận đó, model 4B rất dễ "gật đầu" theo
# người dùng kể cả khi tài liệu nói ngược lại.
#
# Với model mặc định (qwen3:4b) thì suy luận vốn đã bật sẵn nên bật thêm không đổi gì; giá
# trị thật của tuỳ chọn này là ở những model có suy luận nhưng MẶC ĐỊNH TẮT - lúc đó đúng
# loại câu hỏi cần độ chính xác nhất vẫn được bật. Tắt (0) nếu muốn ưu tiên tốc độ.
# Lưu ý: đây KHÔNG phải công tắc tắt suy luận cho câu hỏi thường - xem giải thích chi tiết
# về hành vi thật của tham số `think` trong RagPipeline._goi_llm().
BAT_THINKING_KHI_KIEM_CHUNG = _lay_bool("BAT_THINKING_KHI_KIEM_CHUNG", True)

# Số nguồn tối đa hiển thị dưới mỗi câu trả lời. Hệ thống ưu tiên hiển thị đúng những đoạn
# trích mà LLM đã thật sự tham chiếu ([1], [2]... trong câu trả lời); con số này chỉ là trần.
SO_TRICH_DAN_HIEN_THI = _lay_int("SO_TRICH_DAN_HIEN_THI", 3)

# Câu từ chối cố định khi tài liệu không có thông tin. Đặt ở config (thay vì trong
# rag_pipeline) vì cả 3 nơi đều cần đúng chuỗi này và phải khớp nhau tuyệt đối: system
# prompt ra lệnh cho LLM dùng nó, pipeline trả thẳng nó khi không có đoạn nào, citation
# nhận diện nó để KHÔNG hiển thị nguồn (câu từ chối mà vẫn kèm trích dẫn thì tự mâu thuẫn -
# vừa nói không có thông tin vừa chỉ vào một trang cụ thể). Lệch nhau dù chỉ một dấu chấm
# là cơ chế nhận diện hỏng ngay mà không có lỗi nào báo ra.
CAU_TU_CHOI = {
    "vi": "Không tìm thấy thông tin trong tài liệu.",
    "en": "The documents do not contain this information.",
}


# ============================================================
# EVALUATION
# ============================================================
EVAL_DIR = BASE_DIR / "evaluation"
TEST_QUESTIONS_FILE = EVAL_DIR / "test_questions.json"

# --- Bộ câu hỏi HELD-OUT: tài liệu và câu hỏi CHƯA TỪNG dùng để chỉnh bất kỳ tham số nào ---
# Vấn đề nằm dưới tất cả các tham số của hệ thống này: chúng đều được hiệu chỉnh IN-SAMPLE.
# TEST_QUESTIONS_FILE sinh ra từ chính corpus đang dùng, và mọi hằng số - 0.70, 0.001, 0.88,
# 2, 1, 30, 160 - đều chọn bằng cách tối ưu trên đúng bộ đó. Đó là tuning trên tập test, và
# hệ quả tất yếu là hệ thống rất tốt trên corpus đã dùng để chỉnh rồi tụt trên corpus mới -
# đúng hiện tượng "truy xuất kém trên tài liệu mới" đã gặp.
#
# Bộ held-out tồn tại để BIẾN mức overfit đó thành một con số: chênh lệch Recall@K giữa
# in-sample và held-out. Quy tắc sử dụng chỉ có một, và phải giữ nghiêm:
#   TUYỆT ĐỐI KHÔNG chỉnh tham số theo kết quả trên bộ này.
# Chỉnh theo nó một lần là nó lập tức trở thành một bộ in-sample thứ hai, và con số đo được
# mất sạch ý nghĩa - lúc đó hệ thống không còn cách nào biết mình đang overfit tới đâu.
#
# Bộ mặc định gồm 46 câu (44 câu có đáp án + 2 câu lạc đề) trên 12 tài liệu, không tài liệu
# nào trong số đó xuất hiện trong TEST_QUESTIONS_FILE:
#   CV-05-Classification.pdf, BAO_CAO_MAY_HOC.docx, Chapter 2. Server kết nối vạn vật.pptx,
#   baocaonangcaothayAn.docx, Bai1..Bai6 (bài giảng Khai phá dữ liệu), PaperQA.pdf, RFRAG.pdf.
#
# LƯU Ý khi đọc Recall@K trên bộ này: 6 tài liệu Bai*.docx là DOCX gần như không có ngắt
# trang (Bai5 chỉ 1 "trang", Bai2 có 2), mà Precision@K/Recall@K so khớp theo (nguồn, trang).
# Với những tài liệu đó, lấy về BẤT KỲ chunk nào cũng cho Recall@K = 1.00 dù chunk có chứa
# câu trả lời hay không - đúng kiểu suy biến đã ghi ở §5.39. Vì vậy chỉ đặt 1-2 câu trên mỗi
# tài liệu như thế, và khi báo cáo thì đọc kèm Citation accuracy chứ đừng chỉ nhìn Recall@K.
TEST_QUESTIONS_HELD_OUT_FILE = EVAL_DIR / "test_questions_held_out.json"
# Mặc định dùng chung OLLAMA_MODEL để tự chấm Faithfulness/Answer Relevance
# (LLM-as-judge đơn giản) - tránh phải quản lý thêm một model riêng cho việc chấm điểm.
JUDGE_MODEL = _lay_str("JUDGE_MODEL", OLLAMA_MODEL)

# --- OCR TỰ ĐỘNG cho trang PDF không đọc được text ---
# Đọc lại trang bằng model vision khi bản thân việc trích xuất text đã thất bại. Hai trường
# hợp: (1) PDF nhúng font không kèm bảng ánh xạ ToUnicode nên chữ ra thành mã "(cid:NN)" -
# rất hay gặp với font toán học; (2) trang scan (ảnh chụp trang), không có chữ nào để trích.
#
# MẶC ĐỊNH BẬT, và đây là một quyết định đã bị ĐẢO NGƯỢC sau khi gặp hậu quả thật.
#
# Bản trước mặc định TẮT, với lý do "chi phí lớn, lợi ích hẹp": đo trên bộ tài liệu lúc đó,
# 375/1221 trang cần OCR nên một lần build tốn thêm ~88 phút, trong khi retrieval vốn đã đạt
# Recall@K 1.00 mà không cần OCR. Lập luận ấy đúng VỚI BỘ TÀI LIỆU ĐÓ - và sai ngay khi gặp
# tài liệu khác: người dùng nạp vào một giáo trình 383 trang dạng scan, hệ thống đọc ra ĐÚNG
# 0 ký tự, build "thành công" một index gồm toàn chunk rỗng, rồi trả lời sai mọi câu hỏi mà
# không báo lỗi gì. Một mặc định được chọn dựa trên một bộ tài liệu cụ thể chính là định
# nghĩa của việc chỉnh cho vừa bộ test.
#
# Vì sao bật mặc định là AN TOÀN chứ không phải đánh đổi:
#   - OCR chỉ chạy cho TRANG NÀO ĐO ĐƯỢC là không đọc được (vision_caption.trang_can_ocr).
#     Tài liệu có lớp text bình thường -> không trang nào kích hoạt -> chi phí đúng bằng 0.
#   - Tài liệu scan thì OCR không phải "tuỳ chọn nâng cao", nó là cách DUY NHẤT để đọc được
#     nội dung. Không OCR thì hệ thống không có gì để trả lời - đắt hơn nhiều so với chờ.
# Đo thực tế trên giáo trình scan tiếng Việt 383 trang: ~12 giây/trang, phục hồi 2.000-2.600
# ký tự mỗi trang, đọc đúng cả số hiệu văn bản ("Nghị định số 88/2006/NĐ-CP").
#
# Đặt BAT_OCR_DU_PHONG=0 nếu muốn build thật nhanh và chấp nhận bỏ qua tài liệu scan.
BAT_OCR_DU_PHONG = _lay_bool("BAT_OCR_DU_PHONG", True)
# Ngưỡng nhận diện trang đọc hỏng (xem vision_caption.trang_can_ocr).
SO_CID_TOI_THIEU_DE_OCR = _lay_int("SO_CID_TOI_THIEU_DE_OCR", 5)
TY_LE_CID_DE_OCR = _lay_float("TY_LE_CID_DE_OCR", 0.02)
SO_TU_TOI_THIEU_TRANG_CO_CHU = _lay_int("SO_TU_TOI_THIEU_TRANG_CO_CHU", 15)
# Trang sách dày chữ hơn một hình minh hoạ nhiều, nên cần hạn mức sinh cao hơn.
OCR_NUM_PREDICT = _lay_int("OCR_NUM_PREDICT", 1200)
# Độ phân giải render trang trước khi đưa cho model vision. 150 DPI đủ để đọc công thức mà
# không làm ảnh quá lớn (ảnh lớn hơn -> model chậm hơn rõ rệt).
DPI_RENDER_TRANG_OCR = _lay_int("DPI_RENDER_TRANG_OCR", 150)
# Số ký tự văn bản tối thiểu để coi là "đã đọc được" một tài liệu. Dưới mức này thì gần như
# chắc chắn tài liệu là bản scan hoặc hỏng - hệ thống phải nói ra, thay vì lặng lẽ build một
# index rỗng rồi trả lời sai. 200 ký tự là chưa bằng một đoạn văn: mọi tài liệu đọc được
# thật đều vượt xa mức này, còn tài liệu không đọc được thì thường đúng bằng 0.
SO_KY_TU_TOI_THIEU_MOT_TAI_LIEU = _lay_int("SO_KY_TU_TOI_THIEU_MOT_TAI_LIEU", 200)
# Ảnh chiếm từ ngần này diện tích trang trở lên KHÔNG phải hình minh hoạ mà chính là ảnh
# chụp cả trang (PDF scan). Trích nó ra làm "hình" là vô nghĩa: nội dung thật của nó là chữ,
# và chữ đó đã được OCR lấy ra rồi. Bỏ qua giúp tránh một chunk rác mỗi trang, tránh tốn
# thêm một lượt model vision mỗi trang, và tránh việc trích dẫn trỏ vào "hình" thay vì chữ.
TY_LE_DIEN_TICH_ANH_TOAN_TRANG = _lay_float("TY_LE_DIEN_TICH_ANH_TOAN_TRANG", 0.60)

# --- Đọc lại trang PDF khi chữ bị DÍNH LIỀN NHAU ---
# pdfplumber quyết định "có khoảng trắng giữa 2 ký tự hay không" bằng x_tolerance (mặc định
# 3 điểm). Với font chữ đặt sát nhau - điển hình là Computer Modern của sách LaTeX - khoảng
# cách thật giữa các từ nhỏ hơn 3 điểm, nên MỌI khoảng trắng bị nuốt:
#   "whichareknownasthenormalequationsfortheleastsquaresproblem"
# Hậu quả lan ra toàn hệ thống chứ không chỉ khó đọc: tokenizer băm chuỗi dính thành các
# mảnh vô nghĩa (embedding lệch), BM25 mất hoàn toàn từ khoá, và LLM-as-judge đối chiếu câu
# trả lời sạch với ngữ cảnh dính chữ thì không khớp được nên chấm 0 - đúng nguyên nhân của
# Faithfulness 0.33 ở nhóm sách tiếng Anh (§5.38).
#
# ĐO TRÊN CHÍNH CORPUS CỦA ĐỒ ÁN (9 file PDF, 4 trang mẫu mỗi file), "tỉ lệ dính" = phần
# trăm ký tự nằm trong một cụm từ 15 chữ cái trở lên không có khoảng trắng:
#   Bishop:      48.2% (mặc định)  ->  1.6% (x_tolerance=1.5)
#   8 file còn lại:        0.0-2.6%  ->  không đổi
# Hạ xuống 1.0 thì bắt đầu có hại: CV-04 bị chèn khoảng trắng vào GIỮA từ (tỉ lệ từ 1 chữ
# cái nhảy 4.8% -> 25.8%). Vì vậy 1.5 là giá trị đã chọn, và chỉ dùng cho TRANG NÀO ĐO ĐƯỢC
# LÀ DÍNH - trang đọc bình thường vẫn giữ nguyên tham số mặc định của pdfplumber.
BAT_DOC_LAI_TRANG_DINH_CHU = _lay_bool("BAT_DOC_LAI_TRANG_DINH_CHU", True)
# CÁC MỨC ĐEM RA DÒ, không phải một giá trị chốt cứng. Giá trị tốt nhất phụ thuộc font và
# cỡ chữ của từng tài liệu; chọn cứng con số đo được trên một cuốn sách là tự buộc hệ thống
# vào đúng cuốn sách đó. Xếp từ ít can thiệp nhất tới nhiều nhất; mỗi trang tự chọn mức phù
# hợp với chính nó (xem document_loader._trich_text_thich_ung).
CAC_X_TOLERANCE_THU = [
    float(x) for x in _lay_str("CAC_X_TOLERANCE_THU", "2.0,1.5,1.0,0.7").split(",") if x.strip()
]
# Mức cho phép tỉ lệ "từ 1 chữ cái" TĂNG so với bản đọc gốc. Đây là chốt an toàn quan trọng
# nhất của cơ chế này: hạ x_tolerance quá tay gây đúng lỗi ngược (chèn khoảng trắng vào GIỮA
# từ - "befor e"), và mức tăng của tỉ lệ này là dấu hiệu đo được của việc đó. Nhờ có nó, một
# mức x_tolerance chỉ được chấp nhận khi vừa gỡ được chữ dính vừa không làm vỡ từ - phép
# kiểm tra chạy trên MỌI trang của MỌI tài liệu, thay vì chỉ là thứ người viết code đã tự
# tay đối chiếu một lần trên vài file.
MUC_TANG_TU_LE_CHAP_NHAN = _lay_float("MUC_TANG_TU_LE_CHAP_NHAN", 0.03)
# Trang có tỉ lệ dính vượt ngưỡng này thì đọc lại. 10% là khoảng trống rộng giữa 2 nhóm đo
# được ở trên (0.0-2.6% với trang bình thường, 48.2% với trang hỏng) - không có file nào
# nằm lưng chừng, nên ngưỡng không nhạy cảm.
TY_LE_DINH_CHU_DE_DOC_LAI = _lay_float("TY_LE_DINH_CHU_DE_DOC_LAI", 0.10)
# Độ dài tối thiểu của một cụm chữ cái liền để bị coi là "dính". Chọn bằng cách đo, không
# đoán - và lần đo nào cũng cho thấy ngưỡng phải cao hơn trực giác ban đầu:
#   15 -> "Backpropagation" (đúng 15 chữ) bị tính oan là dính.
#   20 -> vẫn oan với "internationalization"/"counterrevolutionary" (20 chữ).
#   25 -> mọi từ tiếng Anh/tiếng Việt thông thường đều lọt dưới, trong khi trang Bishop bị
#         dính vẫn ở 36.2% và tám file PDF còn lại của corpus đều 0.0%.
# Khoảng cách 36% so với 0% là thứ khiến ngưỡng phát hiện (10%) không nhạy cảm.
DO_DAI_CUM_DINH_CHU = _lay_int("DO_DAI_CUM_DINH_CHU", 25)
# Lượng chữ tối thiểu để phép đo tỉ lệ có nghĩa. Đây là chốt chống báo động giả QUAN TRỌNG
# NHẤT, và nó không thay được bằng cách nâng DO_DAI_CUM_DINH_CHU: một câu ngắn chứa đúng
# một từ dài hợp lệ ("internationalization of counterrevolutionaries") vẫn cho 81.6% ở MỌI
# ngưỡng độ dài, đơn giản vì mẫu quá nhỏ để tính tỉ lệ. Một trang PDF thật có hàng nghìn ký
# tự chữ; dưới 200 thì đó là trang gần như trống (chỉ có hình, hoặc một dòng tiêu đề) - vừa
# không đo được gì vừa không đáng đọc lại.
SO_KY_TU_TOI_THIEU_DE_DO = _lay_int("SO_KY_TU_TOI_THIEU_DE_DO", 200)

# --- Hiển thị câu trả lời theo luồng (streaming) ---
# Ollama hỗ trợ trả kết quả theo từng mảnh trong lúc model đang sinh. Việc này KHÔNG làm
# model chạy nhanh hơn một giây nào - tổng thời gian vẫn 35-70 giây trên CPU - nhưng đổi
# hẳn cảm nhận: thay vì một spinner đứng yên suốt cả phút (người dùng không phân biệt được
# "đang chạy" với "đã treo"), chữ bắt đầu hiện sau ~2-3 giây và chạy liên tục tới hết.
#
# Tắt bằng BAT_STREAMING=0 nếu cần chạy trên môi trường không giữ được kết nối dài (một số
# proxy cắt kết nối HTTP đang mở); khi đó giao diện quay về chế độ spinner + trả một cục.
BAT_STREAMING = _lay_bool("BAT_STREAMING", True)
# Giãn cách tối thiểu giữa 2 lần vẽ lại khung chat (giây). Vẽ lại theo TỪNG token là lãng
# phí: Streamlit gửi lại cả khối markdown qua websocket mỗi lần, mà mắt người không đọc kịp
# quá ~10 khung/giây. 0.12s cho cảm giác chữ chạy mượt mà không làm nghẽn trình duyệt.
GIAN_CACH_VE_LAI_GIAY = _lay_float("GIAN_CACH_VE_LAI_GIAY", 0.12)
# Số ký tự cuối của phần suy luận nội bộ được hiện trong khung trạng thái. Suy luận của
# qwen3:4b dài ~15.000 ký tự - hiện hết thì khung chat bị đẩy đi mất; hiện phần ĐUÔI là
# đúng thứ người xem cần (model đang nghĩ tới đâu NGAY LÚC NÀY).
SO_KY_TU_SUY_LUAN_HIEN = _lay_int("SO_KY_TU_SUY_LUAN_HIEN", 500)

# --- Tự kiểm tra độ tin cậy của LLM-as-judge ---
# Giám khảo là model 4B nên bản thân nó cũng sai được, và cái sai của nó KHÔNG có triệu
# chứng: điểm 0.0 của một câu trả lời đúng trông y hệt điểm 0.0 của một câu bịa. Đã gặp
# thật: 2 câu về sách Bishop bị chấm 0.0 vì ngữ cảnh trích từ PDF bị dính chữ, kéo cả nhóm
# xuống 0.33 (§5.38). Nếu tin con số đó thì sẽ đi tối ưu prompt cho một lỗi không tồn tại.
#
# Cách chốt: đối chiếu điểm của giám khảo với một phép đo TẤT ĐỊNH (tỉ lệ cụm 4 từ của câu
# trả lời xuất hiện nguyên văn trong ngữ cảnh, so sau khi bỏ hết khoảng trắng). Hai điều
# sau không thể cùng đúng - "câu trả lời bịa" và "câu trả lời chép nguyên văn ngữ cảnh" -
# nên hễ chúng cùng xảy ra là đánh dấu câu đó ĐÁNG NGỜ.
NGUONG_DIEM_JUDGE_THAP = _lay_float("NGUONG_DIEM_JUDGE_THAP", 0.5)
# 0.30 = 30% số cụm 4 từ trùng nguyên văn. Đặt cao hơn hẳn mức của một câu trả lời diễn đạt
# lại bằng lời của mình (thường dưới 10%), để cờ chỉ bật khi thật sự có mâu thuẫn.
NGUONG_BAM_NGU_CANH_DE_NGHI_NGO = _lay_float("NGUONG_BAM_NGU_CANH_DE_NGHI_NGO", 0.30)
# Số lần chấm lại Faithfulness rồi lấy trung vị. temperature=0 KHÔNG khiến giám khảo tất
# định: đo trực tiếp trên qwen3:4b, cùng một prompt cùng một ca, hai lần chấm liên tiếp cho
# 1.00 và 0.00 - dao động bằng toàn bộ thang điểm. Không xử lý thì hai lần chạy đánh giá
# trên CÙNG một phiên bản code đã ra hai con số khác nhau, và mọi so sánh "trước/sau khi
# sửa" thành ra vô nghĩa.
#
# 3 lần là mức đủ để trung vị ổn định mà chi phí còn chấp nhận được (+2 lượt LLM mỗi câu,
# khoảng +10 phút cho bộ 29 câu). Đặt về 1 nếu chỉ cần chạy thử nhanh - nhưng đừng lấy con
# số của lần chạy đó đi so sánh với lần khác.
SO_LAN_CHAM_FAITHFULNESS = _lay_int("SO_LAN_CHAM_FAITHFULNESS", 3)

# --- Đọc PDF bố cục NHIỀU CỘT ---
# pdfplumber đọc theo DÒNG NGANG chạy suốt bề ngang trang. Với trang chia 2 cột, nó ghép câu
# của cột trái với câu của cột phải thành một dòng - đo được trên PDF 2 cột dựng thử:
#   "Dieu 1. Pham vi dieu chinh cua luat nay Dieu 2. Doi tuong ap dung bao gom moi"
# Hai điều luật khác nhau dính thành một câu. Mọi chunk sinh ra từ trang đó đều vô nghĩa, và
# không có dấu hiệu nào để nhận ra ngoài việc đọc câu trả lời thấy lộn xộn.
#
# Bố cục nhiều cột rất phổ biến ở báo, tạp chí khoa học, một số giáo trình - tức đây không
# phải trường hợp hiếm mà là một lỗ hổng thật với tài liệu chưa từng gặp.
BAT_DOC_THEO_COT = _lay_bool("BAT_DOC_THEO_COT", True)
# Số dải chia theo bề ngang trang khi dò rãnh giữa các cột.
SO_O_DO_COT = _lay_int("SO_O_DO_COT", 60)
# Số dải trống liên tiếp tối thiểu để coi là rãnh giữa cột (3/60 = 5% bề ngang trang).
SO_O_RANH_TOI_THIEU = _lay_int("SO_O_RANH_TOI_THIEU", 3)
# Mỗi cột phải giữ ít nhất ngần này tỉ lệ số từ của trang. Đây là chốt chống báo động giả
# QUAN TRỌNG NHẤT: bản dò đầu tiên không có nó đã nhận nhầm 100% số trang của giáo trình
# Pháp luật thành "2 cột" - thứ nó tưởng là rãnh thực ra là LỀ TRANG, bên ngoài không có chữ.
TY_LE_TU_MOI_COT = _lay_float("TY_LE_TU_MOI_COT", 0.25)
# Trang quá ít chữ thì không đủ dữ liệu để kết luận bố cục (bìa, trang tiêu đề, trang ảnh).
SO_TU_TOI_THIEU_DE_DO_COT = _lay_int("SO_TU_TOI_THIEU_DE_DO_COT", 60)

# --- Chỉ báo "bám nguồn" hiện dưới câu trả lời ---
# Tỉ lệ cụm từ của câu trả lời trùng NGUYÊN VĂN với ngữ cảnh, từ mức này trở lên thì hiện ra
# cho người đọc thấy (rag.citation.do_bam_ngu_canh).
#
# CHỈ HIỆN KHI CAO, không hiện cảnh báo khi thấp - có chủ đích. Phép đo này chỉ nói được một
# chiều: cao là bằng chứng mạnh rằng câu trả lời không bịa; thấp thì KHÔNG kết luận được gì,
# vì một câu trả lời diễn đạt lại bằng lời của mình (hoàn toàn hợp lệ, thường là mong muốn)
# cũng cho mức thấp. Hiện dấu hiệu "bám nguồn thấp" sẽ khiến người đọc nghi ngờ oan đúng
# những câu trả lời viết tốt nhất.
#
# 0.30 là mức mà câu trả lời đã trích lại đáng kể nguyên văn tài liệu - đủ để nói chắc, chứ
# không phải một sự trùng hợp của vài cụm từ thông dụng.
NGUONG_BAM_NGUON_HIEN_THI = _lay_float("NGUONG_BAM_NGUON_HIEN_THI", 0.30)


# ============================================================
# NGỮ CẢNH HỘI THOẠI: viết lại câu hỏi nối tiếp trước khi truy xuất
# ============================================================
# Bài toán: truy xuất vốn chỉ nhìn thấy ĐÚNG câu hỏi hiện tại. Một câu nối tiếp kiểu "Thế
# còn phần thứ hai thì sao?" không mang chủ đề nào trong bản thân nó, nên vector của nó
# không trỏ tới đâu cả và BM25 cũng không có từ khoá nào để bám. Lịch sử chat tuy hiện trên
# màn hình nhưng KHÔNG được dùng để định hướng lại truy xuất - đây là giới hạn của kiến trúc
# single-turn, và nó xảy ra NGAY TRONG MỘT PHIÊN, khác hẳn chuyện "không lưu lịch sử qua
# nhiều phiên" (vốn là phạm vi đã chốt, không phải khiếm khuyết).
#
# Cách xử lý MẶC ĐỊNH - tất định, không gọi model: nhận diện câu nối tiếp (tất định), rồi
# GHÉP các câu hỏi trước vào làm một truy vấn phụ để RRF hợp nhất. "Vi phạm pháp luật gồm
# những dấu hiệu nào? Thế còn dấu hiệu thứ hai thì sao?" mang đủ từ khoá chủ đề để vector
# trỏ đúng vùng. Chi phí: một lần encode (~30ms), không thêm lượt gọi LLM nào.
BAT_TRUY_VAN_NGU_CANH = _lay_bool("BAT_TRUY_VAN_NGU_CANH", True)

# Đường thứ hai: nhờ LLM VIẾT LẠI câu nối tiếp thành câu độc lập (query rewriting kinh điển).
#
# MẶC ĐỊNH TẮT, và đây là một KẾT QUẢ ÂM TÍNH ĐO ĐƯỢC chứ không phải chưa làm xong (§5.58):
#
#   qwen3:4b      0/7 ca. Model luôn sinh chuỗi suy luận dài trước khi trả lời (§5.23); với
#                 num_predict=200 nó tiêu hết ngân sách lúc "nghĩ" và trả về content RỖNG.
#                 Nâng lên 1500 vẫn rỗng (thinking đã 5891 ký tự). Đủ chỗ thì cần ~3000
#                 token, tức ~30 giây CỘNG THÊM mỗi câu nối tiếp - đắt hơn vấn đề nó giải.
#   qwen2.5vl:3b  Không có chế độ suy luận nên nhanh, nhưng sai: một ca chép y nguyên câu
#                 gốc, một ca trả về đúng câu hỏi CŨ (mất hẳn phần "thứ hai").
#
# Code được GIỮ NGUYÊN, đúng tiền lệ §5.30 với BM25: kết quả âm tính trên model này không
# có nghĩa âm tính với model khác. Đổi OLLAMA_MODEL sang một model không sinh suy luận và
# đủ mạnh thì bật lại, rồi chạy `python evaluation/kiem_dinh_viet_lai.py` để tự kiểm.
BAT_VIET_LAI_CAU_HOI = _lay_bool("BAT_VIET_LAI_CAU_HOI", False)

# Trần token cho lượt viết lại. Chính là chỗ đường LLM gãy với model có chế độ suy luận -
# xem số đo ở trên.
NUM_PREDICT_VIET_LAI = _lay_int("NUM_PREDICT_VIET_LAI", 200)

# Số LƯỢT hỏi-đáp gần nhất đưa vào prompt viết lại. 3 là đủ: câu nối tiếp gần như luôn tham
# chiếu tới lượt liền trước, và nhồi thêm lịch sử chỉ làm model có nhiều thứ để bám nhầm hơn
# (đã gặp: với 6 lượt, model gộp chủ đề của lượt 1 vào câu hỏi đang hỏi về lượt 5).
SO_LUOT_NGU_CANH = _lay_int("SO_LUOT_NGU_CANH", 3)

# Cắt ngắn câu trả lời cũ khi đưa vào prompt viết lại. Việc viết lại chỉ cần biết lượt trước
# NÓI VỀ CHỦ ĐỀ GÌ, không cần toàn văn - mà toàn văn thì vừa tốn token vừa dễ khiến model
# lôi chi tiết vụn của câu trả lời cũ vào câu hỏi mới.
DO_DAI_TRA_LOI_TRONG_NGU_CANH = _lay_int("DO_DAI_TRA_LOI_TRONG_NGU_CANH", 300)

# Trần độ dài bản viết lại, tính theo số từ. Bản viết lại dài hơn mức này gần như luôn là
# model đã tự trả lời câu hỏi hoặc nhồi cả đoạn ngữ cảnh vào - lúc đó giữ câu gốc an toàn hơn.
SO_TU_TOI_DA_CAU_VIET_LAI = _lay_int("SO_TU_TOI_DA_CAU_VIET_LAI", 60)

# Trọng số của nhánh CÂU HỎI GỐC trong RRF, khi lượt hỏi có thêm nhánh ngữ cảnh (truy vấn
# chính luôn = 1.0).
#
# Vì sao vẫn chạy nhánh câu gốc thay vì thay thế hẳn: truy vấn đã ghép ngữ cảnh là một PHỎNG
# ĐOÁN về ý người dùng, và phỏng đoán thì có thể sai. Giữ nhánh câu gốc khiến một lần ghép
# sai không thể xoá sạch kết quả đúng - nó chỉ làm thứ hạng nhiễu đi một chút. Cùng nguyên
# tắc an toàn THEO CẤU TRÚC đã dùng ở §5.45(a): thiết kế sao cho trường hợp xấu nhất là
# "không cải thiện", không phải "làm hỏng".
#
# Chi phí gần như bằng 0: thêm 1 lần encode (~30ms) và vài chục phép cộng RRF. KHÔNG thêm
# lần rerank nào - rerank vẫn chạy đúng một lần trên tập ứng viên đã hợp nhất.
TRONG_SO_TRUY_VAN_GOC = _lay_float("TRONG_SO_TRUY_VAN_GOC", 1.0)


# ============================================================
# ĐỐI CHIẾU CHÉO CÁC NGUỒN: phát hiện mâu thuẫn giữa các tài liệu
# ============================================================
# Bài toán: hệ thống xử lý mỗi đoạn trích độc lập và không bao giờ hỏi "các đoạn này có nói
# ngược nhau không". Với tài liệu học tập thật, đây là chuyện xảy ra thường xuyên: giáo trình
# cũ và slide mới ghi khác con số, hai tài liệu định nghĩa khác nhau cùng một khái niệm. LLM
# đọc cả hai rồi tự chọn một bên (hoặc trộn lẫn) mà không nói gì - người đọc mất hẳn thông
# tin quan trọng nhất: "hai nguồn của bạn đang không thống nhất".
#
# Cách xử lý: HAI TẦNG, giống hệt cách retrieval làm (quét rộng rẻ -> đọc kỹ đắt).
#   Tầng 1 (tất định, mili giây): lọc ra cặp đoạn ĐÁNG NGỜ - khác nguồn, cùng chủ đề, và có
#           dấu hiệu bất đồng bề mặt (khác số, hoặc lệch phủ định).
#   Tầng 2 (LLM): chỉ chấm những cặp sống sót qua tầng 1.
# Đại đa số câu hỏi không có cặp nào qua được tầng 1 -> tốn đúng 0 lượt gọi LLM.
BAT_DOI_CHIEU_NGUON = _lay_bool("BAT_DOI_CHIEU_NGUON", True)

# Chỉ đối chiếu cặp đoạn có cosine từ mức này trở lên. Hai đoạn phải NÓI VỀ CÙNG MỘT CHUYỆN
# thì mới có thể mâu thuẫn nhau; hai đoạn khác chủ đề chỉ là hai thông tin khác nhau.
#
# 0.88 đo trên corpus thật: cặp đoạn cùng chủ đề khác nguồn nằm ở 0.88-0.96, còn cặp chỉ
# tình cờ cùng được truy xuất cho một câu hỏi thì 0.78-0.86. Đặt thấp hơn sẽ kéo theo rất
# nhiều cặp không liên quan, mà mỗi cặp thừa là một lượt gọi LLM thật (~6 giây).
NGUONG_COSINE_DOI_CHIEU = _lay_float("NGUONG_COSINE_DOI_CHIEU", 0.88)

# Trần số cặp gửi lên LLM trong một lượt hỏi. Đây là chốt chặn CHI PHÍ, không phải chốt chất
# lượng: số cặp tăng theo BÌNH PHƯƠNG số đoạn - C(TOP_K, 2), tức 6 cặp với TOP_K=4 và tới 15
# cặp nếu ai đó nâng TOP_K trở lại 6 - và mỗi cặp là một lượt gọi LLM thật cộng vào sau mỗi
# câu trả lời. Cặp được xét theo thứ tự cosine
# giảm dần nên 3 cặp giữ lại là 3 cặp giống chủ đề nhất - đúng chỗ mâu thuẫn hay nằm.
SO_CAP_DOI_CHIEU_TOI_DA = _lay_int("SO_CAP_DOI_CHIEU_TOI_DA", 3)

# Số lần chấm mỗi cặp; chỉ báo mâu thuẫn khi TẤT CẢ các lần đều nói có.
#
# Vì sao đòi đồng thuận chứ không chấm một lần: §5.43 đã đo được LLM-as-judge với model 4B
# lật phán quyết 1/8 lần dù temperature=0. Ở đây hai loại sai KHÔNG ngang giá - báo động giả
# ("tài liệu của bạn mâu thuẫn nhau" trong khi chúng không hề mâu thuẫn) làm người dùng mất
# niềm tin vào chính tài liệu của họ, tệ hơn hẳn việc bỏ sót một mâu thuẫn thật (lúc đó hệ
# thống chỉ trở về đúng hành vi cũ, không tệ hơn). Nên nghiêng hẳn về phía im lặng.
SO_LAN_CHAM_MAU_THUAN = _lay_int("SO_LAN_CHAM_MAU_THUAN", 2)

# Mức độ mâu thuẫn tối thiểu (0-1) để báo ra. Dưới mức này thường là "hai cách diễn đạt khác
# nhau của cùng một ý" chứ không phải xung đột thật.
NGUONG_MAU_THUAN = _lay_float("NGUONG_MAU_THUAN", 0.6)


# ============================================================
# TỐI ƯU INGESTION: CACHE, SONG SONG HOÁ, LỌC ẢNH
# ============================================================
# Toàn bộ nhóm tham số dưới đây phục vụ MỘT nguyên tắc: chỉ trả chi phí tính toán khi thực
# sự cần. Tài liệu text tốt phải đọc rất nhanh; tài liệu scan chấp nhận chậm vì OCR; tài
# liệu nhiều hình chỉ gọi model vision cho những hình có giá trị; và tài liệu ĐÃ XỬ LÝ RỒI
# thì không được xử lý lại. Không tham số nào ở đây tắt OCR, Vision hay reranker để lấy tốc
# độ - làm vậy là đổi chất lượng lấy thời gian, tức giải một bài toán khác.

# Bật bộ nhớ đệm theo content hash cho đọc tài liệu / OCR / chú thích ảnh / embedding.
#
# Tắt khi nào: lúc ĐO ĐẠC chi phí thật của một lần build từ đầu (evaluation, benchmark), vì
# cache trúng sẽ cho ra những con số không phản ánh chi phí thực. Ngoài hai việc đó thì
# không có lý do gì để tắt - cache đánh khoá theo nội dung nên không thể trả về kết quả cũ
# cho một tài liệu đã đổi.
BAT_CACHE_INGESTION = _lay_bool("BAT_CACHE_INGESTION", True)

# Chỉ xử lý lại những tài liệu MỚI hoặc ĐÃ THAY ĐỔI, giữ nguyên vector của các tài liệu cũ
# trong index (thay vì dựng lại index từ đầu mỗi lần bấm "Đọc tài liệu").
#
# Vì sao đây là thứ đáng làm nhất về trải nghiệm: thêm tài liệu là thao tác người dùng lặp
# lại nhiều nhất, và với luồng cũ thì thêm 1 file vào 10 file đã có nghĩa là trả lại chi phí
# của cả 11 file. Vector của 10 file kia không hề đổi - chúng được sinh bởi cùng model, từ
# cùng nội dung, nên tính lại chỉ để ra đúng con số cũ.
#
# An toàn ra sao: mỗi tài liệu được ghi kèm BĂM NỘI DUNG vào index_info.json. Băm khác ->
# xoá sạch vector cũ của file đó rồi đọc lại. File biến mất khỏi thư mục -> vector của nó bị
# xoá khỏi index. Đổi model embedding / chunk size / tuỳ chọn ăn vào nội dung -> vân tay
# index không khớp, hệ thống tự build lại TOÀN BỘ (xem VectorStore.ly_do_khong_tuong_thich).
BAT_INDEX_TANG_DAN = _lay_bool("BAT_INDEX_TANG_DAN", True)

# In bảng tổng kết thời gian từng bước sau mỗi lần build (xem rag/do_thoi_gian.py).
#
# Mặc định BẬT vì đây là thứ trả lời câu hỏi "chỗ nào đang chậm" bằng số đo thay vì bằng
# cảm nhận, và nó gần như miễn phí: một phép trừ time.perf_counter() cho mỗi bước.
BAT_PROFILING_INGESTION = _lay_bool("BAT_PROFILING_INGESTION", True)


# ---------- Song song hoá ----------
def _so_worker_mac_dinh(toi_da: int) -> int:
    """Số worker mặc định suy từ chính máy đang chạy, chặn trên bằng `toi_da`.

    Vì sao không đặt cứng một con số: máy 4 nhân và máy 16 nhân cần hai giá trị khác nhau,
    mà người dùng đồ án thì không ai đi chỉnh. Vì sao vẫn chặn trên: các bước này đều đi qua
    MỘT máy chủ Ollama duy nhất, mở hàng chục yêu cầu cùng lúc không làm model chạy nhanh
    hơn (nó vẫn xếp hàng) mà chỉ làm RAM/VRAM phình lên và có thể chậm hơn hẳn vì hoán đổi.
    """
    return max(1, min(toi_da, (os.cpu_count() or 2) // 2))


# Số luồng gọi model vision song song (chú thích ảnh và OCR trang).
#
# Vì sao ĐÂY là chỗ đáng song song hoá nhất: mỗi lượt gọi là một yêu cầu HTTP tới Ollama rồi
# NGỒI CHỜ - tiến trình Python không làm gì trong suốt thời gian đó. Benchmark của chính
# project đo được ~1,9 giây mỗi ảnh (và 3-6 giây trên máy chỉ có CPU); với vài trăm ảnh thì
# riêng bước này đã là nhiều phút đồng hồ chờ đợi thuần tuý.
#
# Vì sao dùng THREAD chứ không phải process: công việc thật nằm ở phía máy chủ Ollama, phía
# Python chỉ chờ I/O - mà chờ I/O thì thread nhả GIL. Process sẽ phải nhân bản cả tiến trình
# (nạp lại model embedding, mở lại index) để đổi lấy đúng con số 0 về hiệu năng.
#
# Đặt = 1 để quay lại hành vi tuần tự cũ (dễ đọc log hơn khi cần dò lỗi từng ảnh).
SO_WORKER_VISION = _lay_int("SO_WORKER_VISION", _so_worker_mac_dinh(4))

# Số tiến trình đọc tài liệu song song. MẶC ĐỊNH 1 (tắt) - và đây là một lựa chọn có chủ ý,
# không phải việc chưa làm xong:
#   - Đọc PDF bằng pdfplumber là công việc THUẦN CPU trong Python, tức bị GIL chặn. Dùng
#     thread ở đây cho ra đúng tốc độ cũ kèm thêm rủi ro tranh chấp trạng thái.
#   - Dùng process thì mỗi tiến trình con phải nạp lại toàn bộ module, và trên Windows (nền
#     tảng chính của đồ án) `spawn` còn chạy lại phần khởi tạo của config. Với corpus vài
#     chục file thì phần chi phí đó ăn mất phần lớn khoản lợi.
#   - Quan trọng nhất: sau khi đã có cache + index tăng dần, lần build thứ hai trở đi gần
#     như không còn đọc lại tài liệu nào. Song song hoá một việc đã không còn xảy ra là tối
#     ưu nhầm chỗ.
# Nâng lên (2-4) khi phải nạp lần đầu một corpus lớn toàn tài liệu text sạch.
SO_WORKER_DOC = _lay_int("SO_WORKER_DOC", 1)


# ---------- Hiệu chỉnh x_tolerance theo tài liệu ----------
# Độ dính chữ coi là ĐÃ SẠCH - đạt mức này thì dừng thử các x_tolerance còn lại.
#
# PHẢI LÀ MỘT SỐ RIÊNG, KHÔNG ĐƯỢC DÙNG LẠI TY_LE_DINH_CHU_DE_DOC_LAI. Hai con số trả lời hai
# câu hỏi khác nhau: 0.10 là "trang này có đáng đọc lại không", còn số ở đây là "bản đọc lại
# đã đủ tốt để thôi chưa". Bản đầu của phép dừng sớm dùng chung 0.10 cho cả hai và ĐÃ GÂY LỖI
# THẬT, đo được trên PaperQA.pdf: mức x_tolerance đầu tiên đưa độ dính từ 30% xuống 9% liền
# được chấp nhận ngay, trong khi mức tốt hơn nằm ngay sau đó đưa nó về gần 0. Phần 9% còn lại
# không phải con số trừu tượng - nó là những dòng như
#     "RAGmodelsretrievetextfromacorpus, usingmethodssuchasvectorembeddingsearch"
# tức chữ dính liền đi thẳng vào index, phá cả BM25 lẫn embedding của đúng những đoạn đó.
#
# Vì sao 0.02: chính số đo đã có ở _ty_le_dinh_chu() - tám file PDF đọc tốt trong corpus cho
# 0.0-1.5%, còn trang hỏng cho 41.7%. Khoảng trống giữa hai nhóm rất rộng, nên đặt vạch ngay
# trên mức cao nhất của nhóm "đọc tốt" là vừa an toàn vừa không cần đo thêm.
TY_LE_DINH_CHU_DAT_YEU_CAU = _lay_float("TY_LE_DINH_CHU_DAT_YEU_CAU", 0.02)


# Số trang ĐẦU (trong những trang bị dính chữ) dùng để dò ra x_tolerance phù hợp cho cả tài
# liệu; các trang sau thử giá trị đã dò được TRƯỚC TIÊN và dừng ngay khi đạt.
#
# Vì sao: giá trị x_tolerance tốt nhất phụ thuộc FONT và CỠ CHỮ của tài liệu (xem
# _trich_text_thich_ung) - mà font thì gần như không đổi trong cùng một cuốn sách. Bản cũ
# vẫn thử LẦN LƯỢT CẢ 4 mức trên MỌI trang dính chữ, tức đọc lại toàn bộ trang 4 lần, cho ra
# đúng một kết quả đã biết trước từ trang thứ hai trở đi. Với giáo trình Bishop - nơi gần
# như mọi trang đều dính chữ - đó là hàng nghìn lượt extract_text thừa.
#
# Vẫn giữ khả năng dò lại cho từng trang: giá trị đã hiệu chỉnh chỉ được thử TRƯỚC, nếu
# trang đó không đạt thì vẫn dò tiếp các mức còn lại như cũ. Nhờ vậy tài liệu trộn nhiều font
# (phụ lục scan, chương chèn từ nguồn khác) không bị đọc hỏng.
SO_TRANG_HIEU_CHINH_X_TOLERANCE = _lay_int("SO_TRANG_HIEU_CHINH_X_TOLERANCE", 3)


# ---------- Lọc ảnh trước khi gọi model vision ----------
# Diện tích tối thiểu của một ảnh, tính theo TỈ LỆ so với diện tích trang.
#
# Ảnh nhỏ hơn mức này gần như chắc chắn là icon, bullet trang trí, logo góc trang hay đường
# kẻ - không mang nội dung tra cứu được, nhưng mỗi cái vẫn tốn một lượt gọi model vision
# (~1,9 giây) và một vector trong index. Ngưỡng theo TỈ LỆ chứ không theo pixel vì cùng một
# icon sẽ có số pixel khác nhau tuỳ trang A4 hay slide 16:9.
#
# Đặt = 0 để tắt hẳn chốt này (nhận mọi ảnh vượt KICH_THUOC_ANH_TOI_THIEU như bản cũ).
TY_LE_DIEN_TICH_ANH_TOI_THIEU = _lay_float("TY_LE_DIEN_TICH_ANH_TOI_THIEU", 0.015)

# Tỉ lệ cạnh dài / cạnh ngắn tối đa. Vượt mức này là dải trang trí (đường kẻ ngang dưới tiêu
# đề, thanh màu bên lề slide) chứ không phải hình có nội dung - một biểu đồ hay sơ đồ thật
# gần như không bao giờ dẹt tới mức 12:1.
TY_LE_CANH_ANH_TRANG_TRI = _lay_float("TY_LE_CANH_ANH_TRANG_TRI", 12.0)

# Một ảnh có nội dung GIỐNG HỆT xuất hiện từ ngần này lần trở lên trong CÙNG một tài liệu
# thì bị coi là logo/watermark và loại khỏi index.
#
# Vì sao đếm theo nội dung: logo trường lặp lại trên mọi slide sẽ tạo ra hàng chục chunk
# giống hệt nhau - vừa vô dụng cho tra cứu, vừa làm loãng index đúng theo kiểu đã mô tả ở
# _bo_ban_ghi_anh_rong. Vì sao ngưỡng 4 chứ không phải 2: một hình minh hoạ thật hoàn toàn
# có thể được nhắc lại ở 2-3 trang (hình tổng quan mở đầu mỗi chương), và mất một hình thật
# tệ hơn giữ thừa một logo.
#
# Lưu ý: việc chú thích những ảnh này KHÔNG hề tốn thêm lượt gọi model dù chúng có bị loại
# hay không - cache vision đánh khoá theo băm nội dung ảnh nên 60 bản sao của một logo chỉ
# tốn đúng 1 lượt (xem rag/bo_nho_dem.py). Chốt này để làm SẠCH INDEX, không phải để tiết
# kiệm thời gian.
SO_LAN_LAP_COI_LA_LOGO = _lay_int("SO_LAN_LAP_COI_LA_LOGO", 4)


# ============================================================
# NGÂN SÁCH THÍCH ỨNG LÚC TRUY VẤN
# ============================================================
# Không phải câu hỏi nào cũng đáng trả cùng một chi phí. "Overfitting là gì?" cần ít ứng
# viên rerank, ít đoạn ngữ cảnh và một câu trả lời ngắn; "So sánh KNN với Naive Bayes về độ
# phức tạp và trường hợp áp dụng" thì cần cả ba thứ đó nhiều hơn. Bản trước cấp NGÂN SÁCH
# TỐI ĐA cho mọi câu hỏi, tức câu dễ phải trả giá của câu khó.
#
# ĐIỀU TUYỆT ĐỐI KHÔNG LÀM: hạ num_ctx khi ngữ cảnh quá lớn. num_ctx nhỏ hơn prompt nghĩa là
# Ollama CẮT IM LẶNG từ đầu phần ngữ cảnh - tức xoá đúng các đoạn trích liên quan nhất (xem
# OLLAMA_NUM_CTX). Khi ngữ cảnh vượt trần, thứ phải giảm là SỐ ĐOẠN hoặc ĐỘ DÀI MỖI ĐOẠN,
# và phải giảm từ đoạn xếp hạng THẤP NHẤT lên.
BAT_NGAN_SACH_THICH_UNG = _lay_bool("BAT_NGAN_SACH_THICH_UNG", True)

# Số từ tối đa để một câu hỏi còn được coi là ĐƠN GIẢN. Trên mức này, hoặc khi câu hỏi chứa
# dấu hiệu nhiều vế (xem _do_phuc_tap_cau_hoi), câu hỏi được cấp ngân sách đầy đủ.
#
# Đếm TỪ chứ không đếm ký tự để không thiên vị tiếng Anh (từ tiếng Việt ngắn hơn nhiều).
SO_TU_CAU_HOI_DON_GIAN = _lay_int("SO_TU_CAU_HOI_DON_GIAN", 12)

# Số ứng viên đưa vào cross-encoder cho câu hỏi ĐƠN GIẢN (câu phức tạp vẫn dùng
# SO_UNG_VIEN_RERANK = 30).
#
# Vì sao vẫn giữ 12 chứ không hạ sâu hơn: rerank là tuyến phòng thủ chính chống câu hỏi
# ngoài phạm vi (NGUONG_DIEM_RERANK_TOI_THIEU) và là thứ quyết định thứ tự TOP_K. Cắt quá
# tay sẽ đổi chất lượng lấy tốc độ - đúng thứ nhóm tham số này tồn tại để tránh. 12 ứng viên
# vẫn phủ trọn TOP_K=4 kèm dư địa gấp ba cho việc đảo thứ hạng.
SO_UNG_VIEN_RERANK_DON_GIAN = _lay_int("SO_UNG_VIEN_RERANK_DON_GIAN", 12)

# Trần token sinh cho câu hỏi ĐƠN GIẢN. OLLAMA_NUM_PREDICT (12000) được đặt rộng để câu trả
# lời dài không bị cắt cụt, nhưng nó cũng là phần cửa sổ ngữ cảnh bị GIỮ CHỖ - hạ xuống cho
# câu hỏi ngắn giúp _tinh_num_ctx() không phải nhảy lên bậc cao hơn (mỗi lần đổi bậc là một
# lần Ollama nạp lại model, mất hàng chục giây trên CPU).
#
# Đây là trần, không phải mục tiêu: model tự dừng khi viết xong. Đặt 3000 vì một câu trả lời
# có trích dẫn cho câu hỏi định nghĩa hiếm khi vượt 1000 token, còn phần suy luận của qwen3
# thì cần dư địa.
NUM_PREDICT_CAU_HOI_DON_GIAN = _lay_int("NUM_PREDICT_CAU_HOI_DON_GIAN", 3000)

# Nén ngữ cảnh khi prompt vượt trần cửa sổ: cắt bớt đoạn trích (từ đoạn xếp hạng thấp nhất
# lên) thay vì để Ollama cắt im lặng mất đầu ngữ cảnh.
#
# Đây là phần bù cho cảnh báo đã có ở _tinh_num_ctx(): cảnh báo nói cho người dùng biết cấu
# hình đã vượt trần máy, nhưng lượt hỏi ĐANG chạy vẫn bị hỏng. Nén ngữ cảnh khiến lượt đó
# vẫn trả lời được, với phần bị bỏ là phần ÍT LIÊN QUAN NHẤT - lựa chọn của hệ thống, ghi
# rõ trong log, thay vì lựa chọn ngẫu nhiên của bộ cắt prompt.
BAT_NEN_NGU_CANH = _lay_bool("BAT_NEN_NGU_CANH", True)
