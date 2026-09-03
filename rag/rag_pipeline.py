"""Ghép luồng Query: retrieval (lai vector + từ khoá) + ghép prompt + gọi LLM qua Ollama.

Dùng chung 1 EmbeddingService và 1 VectorStore với luồng Ingestion (được truyền vào
từ bên ngoài - app.py hoặc evaluation - chứ RagPipeline không tự tạo, để tránh load
lại model / index nhiều lần không cần thiết).

Hai vấn đề mà module này giải quyết (đều là lỗi đã gặp thực tế, không phải phòng xa):

1. TÀI LIỆU DÀI -> câu trả lời và trích dẫn lệch. Nguyên nhân: bản trước, khi một trang
   được xác định là liên quan thì GỘP NGUYÊN TRANG đưa vào prompt. Với giáo trình ~230
   trang, mỗi trang trung bình 13 chunk (~2000 ký tự), TOP_K=8 trang nghĩa là ~16.000 ký
   tự ngữ cảnh mà đại đa số không dính dáng gì tới câu hỏi -> LLM bị loãng, trả lời lệch
   trọng tâm; còn đoạn trích hiển thị (cắt 400 ký tự ĐẦU đoạn gộp) gần như không bao giờ
   rơi trúng chỗ thật sự khớp câu hỏi. Nay: mỗi đoạn trích được dựng QUANH ĐÚNG chunk khớp
   nhất và mở rộng dần trong ngân sách ký tự (xem _dung_doan_trich).

2. KHẲNG ĐỊNH SAI -> hệ thống vẫn gật đầu. Nguyên nhân: prompt chỉ yêu cầu "trả lời dựa
   trên ngữ cảnh", không hề yêu cầu ĐỐI CHIẾU giả định của người hỏi với tài liệu; model
   nhỏ lại có xu hướng chiều theo người dùng (sycophancy). Nay: câu hỏi dạng kiểm chứng
   được nhận diện và đi theo một system prompt riêng bắt buộc ra phán quyết ĐÚNG/SAI/KHÔNG
   ĐỀ CẬP kèm trích nguyên văn căn cứ, đồng thời bật lại chế độ suy luận của model.
"""

import itertools
import logging
import re
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

import httpx
import ollama
from langdetect import DetectorFactory, LangDetectException, detect_langs

import config
from rag.citation import do_bam_ngu_canh
from rag.doi_chieu_nguon import tim_mau_thuan
from rag.embedding import EmbeddingService
from rag.reranker import RerankerService
from rag.tiep_noi_hoi_thoai import chuan_bi_truy_van
from rag.vector_store import VectorStore
from rag.vision_caption import ten_model_khop

logger = logging.getLogger(__name__)


class LoiKhongKetNoiDuocOllama(RuntimeError):
    """Không mở được kết nối tới máy chủ Ollama (chưa chạy, hoặc OLLAMA_HOST trỏ sai chỗ).

    Cần một lớp lỗi RIÊNG vì đây là lỗi MÔI TRƯỜNG, không phải lỗi dữ liệu: người dùng sửa
    được bằng đúng một câu lệnh, nhưng chỉ khi được nói cho biết phải chạy lệnh gì.

    Thư viện ollama đã bọc lỗi này thành ConnectionError có thông báo tử tế - nhưng CHỈ ở
    đường gọi thường (`_request_raw`). Ở đường streaming - đường DUY NHẤT hệ thống này
    dùng để sinh câu trả lời - kết nối chỉ thật sự mở khi generator được lặp lần đầu, nằm
    ngoài khối bọc lỗi đó, nên httpx.ConnectError bay thẳng ra ngoài. Người dùng nhận
    nguyên một traceback "ConnectError: [WinError 10061] ... target machine actively
    refused it" - không một chữ nào nhắc tới Ollama, cũng không gợi ý phải làm gì.
    """


def _thong_bao_khong_ket_noi_duoc() -> str:
    """Soạn thông báo lỗi kèm ĐÚNG các lệnh cần chạy, đọc host/model từ cấu hình đang dùng.

    Dựng lúc gặp lỗi chứ không phải hằng số dựng lúc import, để nếu người dùng đổi
    OLLAMA_HOST/OLLAMA_MODEL trong .env thì thông báo vẫn nói đúng giá trị thật.
    """
    return (
        f"Không kết nối được tới máy chủ Ollama ở {config.OLLAMA_HOST}. "
        "Ollama là tiến trình chạy model ngôn ngữ ngay trên máy bạn — chưa bật nó thì "
        "hệ thống truy xuất được tài liệu nhưng không sinh được câu trả lời.\n\n"
        "Cách xử lý:\n\n"
        "1. Mở ứng dụng **Ollama** (biểu tượng ở khay hệ thống), hoặc chạy `ollama serve` "
        "trong một cửa sổ terminal và để nguyên cửa sổ đó.\n"
        f"2. Kiểm tra model đã tải về: `ollama list` — nếu chưa thấy `{config.OLLAMA_MODEL}` "
        f"thì chạy `ollama pull {config.OLLAMA_MODEL}`.\n"
        "3. Nếu Ollama chạy ở máy khác hoặc cổng khác, sửa `OLLAMA_HOST` trong `.env`.\n\n"
        "Xong bước trên thì hỏi lại — không cần khởi động lại ứng dụng."
    )


def kiem_tra_may_chu_llm() -> Optional[str]:
    """Máy chủ Ollama đã sẵn sàng trả lời chưa? Trả None nếu ổn, chuỗi mô tả lỗi nếu không.

    Kiểm tra TRƯỚC khi người dùng gõ câu hỏi, cùng lý do với mo_hinh_vision_co_san: hai
    hỏng hóc ở đây (chưa bật Ollama / chưa pull model) đều không dính gì tới câu hỏi và đều
    sửa được trong một phút - nhưng nếu để tới lúc gọi LLM mới lộ ra thì người dùng đã chờ
    xong cả vòng truy xuất rồi mới nhận lỗi, và dễ tưởng là hệ thống hỏng.

    Dùng đường gọi KHÔNG streaming (`list`) nên nhanh và không đánh thức model.
    """
    try:
        cac_model = ollama.Client(host=config.OLLAMA_HOST).list().models
    except Exception as loi:  # chưa bật Ollama, sai host, hoặc client khác phiên bản
        logger.warning("Không hỏi được danh sách model của Ollama (%s).", loi)
        return _thong_bao_khong_ket_noi_duoc()
    if not any(ten_model_khop(m.model or "", config.OLLAMA_MODEL) for m in cac_model):
        return (
            f"Máy chủ Ollama đang chạy nhưng chưa có model `{config.OLLAMA_MODEL}`.\n\n"
            f"Chạy `ollama pull {config.OLLAMA_MODEL}` (tải một lần, vài GB), hoặc đổi "
            "`OLLAMA_MODEL` trong `.env` sang một model đã có trong `ollama list`."
        )
    return None


_THE_MO_THINK = "<think>"
_THE_DONG_THINK = "</think>"


class _LocSuyLuanTheoLuong:
    """Bóc phần <think>...</think> ra khỏi luồng content ĐANG CHẢY, từng mảnh một.

    Ollama tách phần suy luận sang trường riêng (`message.thinking`), nhưng đã quan sát
    thực tế một số lượt vẫn để thẻ <think> lọt vào content - bản không streaming xử lý việc
    này bằng một regex chạy trên chuỗi ĐÃ HOÀN CHỈNH. Streaming không có chuỗi hoàn chỉnh
    để mà chạy regex: mảnh đang tới có thể cắt ngang giữa thẻ ("<thi" | "nk>"), và nếu cứ
    thế đẩy ra màn hình thì người dùng nhìn thấy đúng phần suy luận thô mà cả hệ thống đang
    cố giấu đi.

    Vì vậy phải là một máy trạng thái: giữ lại phần đuôi có thể là NỬA CÁI THẺ (tối đa
    len("</think>") - 1 ký tự) cho tới khi mảnh sau tới đủ để kết luận.
    """

    def __init__(self) -> None:
        self._dem = ""
        self._dang_trong_think = False

    def them(self, manh: str):
        """Nhận 1 mảnh content thô -> trả (phần_câu_trả_lời, phần_suy_luận) đã tách."""
        self._dem += manh
        ra_tra_loi, ra_suy_luan = [], []
        while True:
            the, ra = (
                (_THE_DONG_THINK, ra_suy_luan)
                if self._dang_trong_think
                else (_THE_MO_THINK, ra_tra_loi)
            )
            vi_tri = self._dem.find(the)
            if vi_tri == -1:
                # Chưa thấy thẻ: đẩy ra tất cả TRỪ phần đuôi có thể là nửa cái thẻ.
                giu_lai = len(the) - 1
                if len(self._dem) > giu_lai:
                    ra.append(self._dem[: len(self._dem) - giu_lai])
                    self._dem = self._dem[len(self._dem) - giu_lai :]
                break
            ra.append(self._dem[:vi_tri])
            self._dem = self._dem[vi_tri + len(the) :]
            self._dang_trong_think = not self._dang_trong_think
        return "".join(ra_tra_loi), "".join(ra_suy_luan)

    def ket_thuc(self):
        """Xả nốt phần đuôi còn giữ lại khi luồng đã hết."""
        con_lai, self._dem = self._dem, ""
        if self._dang_trong_think:
            return "", con_lai
        return con_lai, ""

# langdetect lấy mẫu ngẫu nhiên nên MẶC ĐỊNH KHÔNG TẤT ĐỊNH: chạy cùng một câu 8 lần cho 8
# kết quả khác nhau. Đo thực tế với "What does criminal law regulate?":
#     ['ca:0.57','en:0.29','ro:0.14'] / ['ca:0.71','en:0.29'] / ['ca:1.00'] / ['en:0.71','ca:0.29'] ...
# Với hệ song ngữ, điều này nghĩa là CÙNG một câu hỏi lúc được trả lời tiếng Anh lúc tiếng
# Việt - người dùng không thể tin được hệ thống. Cố định seed để kết quả lặp lại được.
DetectorFactory.seed = 0

# Dấu phụ riêng của tiếng Việt (ă â đ ê ô ơ ư + các tổ hợp thanh điệu). Không ngôn ngữ nào
# khác dùng đủ bộ này, nên thấy một ký tự bất kỳ trong đây là chắc chắn tiếng Việt.
_MAU_DAU_TIENG_VIET = re.compile(
    r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)

# Từ tiếng Việt KHÔNG DẤU dùng làm cứu cánh cuối khi langdetect bó tay. Cố ý chỉ chọn những
# từ KHÔNG phải từ tiếng Anh - "la", "the", "hay", "ta" đều là từ tiếng Anh hoặc quá mơ hồ
# nên bị loại, dù chúng rất phổ biến trong tiếng Việt. Thà bỏ sót còn hơn nhận nhầm câu
# tiếng Anh thành tiếng Việt.
_TU_TIENG_VIET_KHONG_DAU = frozenset({
    "gi", "khong", "nao", "duoc", "cua", "nhung", "cac", "mot", "nguoi", "hoac",
    "phai", "viec", "theo", "nhu", "voi", "trong", "cho", "tai", "boi", "vi",
    "dieu", "luat", "phap", "hinh", "su", "nha", "nuoc", "chinh", "tri",
})

# ============================================================
# SYSTEM PROMPT
# ============================================================
# System prompt bắt buộc: chỉ trả lời dựa trên context, không suy đoán, phải nói rõ
# khi tài liệu không có thông tin. Đây là ràng buộc quan trọng nhất của đồ án nên được
# viết thành quy tắc rõ ràng, đánh số, thay vì 1 câu mô tả chung chung dễ bị model bỏ qua.
# Có 2 bản VI/EN (thay vì 1 bản + câu lệnh "trả lời theo ngôn ngữ câu hỏi") để bản thân
# hướng dẫn cũng đúng ngôn ngữ mong muốn - LLM bám theo system prompt tốt hơn khi toàn bộ
# prompt nhất quán 1 ngôn ngữ, thay vì hướng dẫn tiếng Việt nhưng yêu cầu trả lời tiếng Anh.
#
# QUY TẮC 5 VÀ 6 TỪNG XUNG ĐỘT NHAU: bản trước đặt cạnh nhau "trả lời ĐẦY ĐỦ, tổng hợp mọi
# thông tin liên quan" và "trả lời ĐI THẲNG VÀO TRỌNG TÂM" mà không nói cái nào thắng khi
# hai điều đó ngược chiều. Với model 4B, đặt hai chỉ thị ngược chiều cạnh nhau thường khiến
# model chọn cái DỄ TUÂN THỦ HƠN - mà "ngắn gọn" luôn dễ hơn "đầy đủ", vì bỏ bớt thì không
# có gì để làm sai. Nay quy tắc 5 nói rõ nó có ưu tiên cao hơn, và quy tắc 6 định nghĩa lại
# "ngắn gọn" theo SỐ CHỮ THỪA chứ không theo LƯỢNG THÔNG TIN.
#
# Lưu ý về thứ tự sửa lỗi: đây là nguyên nhân YẾU NHẤT trong ba nguyên nhân của "câu trả lời
# ngắn" (hai cái kia: num_ctx bị bỏ trống, và ngưỡng cosine tuyệt đối cắt mất đoạn trích).
# Sửa prompt trước hai cái kia sẽ cho một cải thiện nhẹ đủ để tưởng đã tìm đúng nguyên nhân,
# trong khi bug thật vẫn nằm nguyên đó.
HE_THONG_PROMPT_VI = """Bạn là trợ lý học tập, chỉ được trả lời dựa trên các đoạn trích ("ngữ cảnh") được cung cấp dưới đây.

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin có trong ngữ cảnh, KHÔNG được dùng kiến thức bên ngoài, KHÔNG được suy đoán hay bịa thêm.
2. Nếu ngữ cảnh không chứa đủ thông tin để trả lời, PHẢI trả lời đúng câu: "Không tìm thấy thông tin trong tài liệu." - không cố trả lời một phần bằng suy đoán.
3. Mỗi ý trong câu trả lời PHẢI kèm số đoạn trích làm căn cứ, viết ĐÚNG dạng số trong ngoặc vuông: "... theo [2]". KHÔNG viết "theo trang 109" hay "theo Slide 110" - phải dùng số hiệu đoạn trích [1], [2], [3]... Ý nào không chỉ ra được đoạn trích chứa nó thì không được viết ra.
   Quy tắc này áp dụng cho TỪNG DÒNG khi bạn trả lời bằng danh sách - mỗi gạch đầu dòng hoặc mỗi mục đánh số là một ý riêng và phải có số đoạn trích của riêng nó. Ví dụ đúng:
   1. Nhà nước thiết lập quyền lực công cộng đặc biệt [2].
   2. Nhà nước có chủ quyền quốc gia [2].
   Đặt số ở cuối câu mở đầu rồi bỏ trống các mục còn lại là SAI.
4. NẾU CÂU HỎI CHỨA MỘT KHẲNG ĐỊNH HOẶC GIẢ ĐỊNH MÂU THUẪN VỚI NGỮ CẢNH: phải nói thẳng chỗ sai và nêu nội dung đúng theo tài liệu, kèm căn cứ. TUYỆT ĐỐI không trả lời thuận theo giả định sai chỉ vì người dùng đã nêu nó ra - người hỏi có thể nhớ nhầm, và việc của bạn là bám tài liệu chứ không phải làm hài lòng người hỏi.
5. Trả lời ĐẦY ĐỦ, bám sát đúng nội dung ngữ cảnh: tổng hợp MỌI thông tin liên quan, không chỉ chọn 1 đoạn trích rồi bỏ qua các đoạn còn lại. Nếu ngữ cảnh có nhiều ý, trình bày rõ TỪNG ý (dùng gạch đầu dòng nếu phù hợp). Đây là yêu cầu quan trọng hơn: khi phải chọn giữa ĐẦY ĐỦ và NGẮN GỌN, luôn chọn đầy đủ.
6. "Ngắn gọn" ở đây nghĩa là KHÔNG CÓ CHỮ THỪA, KHÔNG phải ít thông tin hơn. Cụ thể: không lặp lại cùng 1 ý, không trích lại nguyên văn cùng một đoạn nhiều lần, không chia nhỏ thành quá nhiều tiêu đề/mục con nếu nội dung không thực sự cần, không viết phần mở đầu/kết luận thừa. Không được BỎ BỚT thông tin có trong ngữ cảnh để câu trả lời ngắn lại, cũng không được THÊM thông tin ngoài ngữ cảnh để câu trả lời dài ra.
7. Trả lời bằng tiếng Việt, văn phong rõ ràng, mạch lạc."""

HE_THONG_PROMPT_EN = """You are a study assistant. You may only answer based on the excerpts ("context") provided below.

MANDATORY RULES:
1. Only use information present in the context. Do NOT use outside knowledge, do NOT guess or make anything up.
2. If the context does not contain enough information to answer, you MUST reply with exactly: "The documents do not contain this information." - do not attempt a partial guess.
3. Every point in your answer MUST cite the excerpt it comes from, written exactly as a bracketed number: "... according to [2]". Do NOT write "according to page 109" or "according to Slide 110" - use the excerpt numbers [1], [2], [3]... If you cannot point to an excerpt for a claim, do not write that claim.
   This applies to EVERY LINE when you answer with a list - each bullet or numbered item is its own point and needs its own excerpt number. Correct example:
   1. The state establishes a special public power [2].
   2. The state has national sovereignty [2].
   Putting the number only in the opening sentence and leaving the items bare is WRONG.
4. IF THE QUESTION CONTAINS A STATEMENT OR ASSUMPTION THAT CONTRADICTS THE CONTEXT: say plainly what is wrong and state what the documents actually say, with the citation. NEVER go along with a false assumption just because the user stated it - the user may be misremembering, and your job is to follow the documents, not to please the user.
5. Answer FULLY, staying close to the context: synthesize EVERY piece of relevant information, do not just pick one excerpt and ignore the rest. If the context covers multiple points, lay EACH one out clearly (use bullet points where appropriate). This is the higher priority: when completeness and brevity pull in opposite directions, choose completeness.
6. "Brief" here means NO WASTED WORDS, not less information. Specifically: do not repeat the same point, do not re-quote the same passage multiple times, do not split the answer into more headings/sections than the content actually needs, do not add filler intros/conclusions. Never DROP information that is in the context to make the answer shorter, and never ADD information outside the context to make it longer.
7. Answer in English, with clear, well-structured writing."""

# Prompt riêng cho câu hỏi dạng KIỂM CHỨNG một khẳng định. Khác biệt cốt lõi so với prompt
# thường: bắt buộc ra phán quyết rời rạc (ĐÚNG/SAI/KHÔNG ĐỀ CẬP) và bắt buộc trích NGUYÊN
# VĂN câu làm căn cứ. Hai ràng buộc này chặn đúng cơ chế gây lỗi: model không còn được phép
# viết một đoạn văn chung chung "nghe như đang đồng ý", mà buộc phải chỉ ra một câu cụ thể
# trong tài liệu - nếu không có câu nào chứng minh được thì tự nó lộ ra là khẳng định sai
# hoặc tài liệu không đề cập.
HE_THONG_PROMPT_KIEM_CHUNG_VI = """Bạn là trợ lý học tập kiêm người KIỂM CHỨNG thông tin. Người dùng đang đưa ra một khẳng định (hoặc một giả định ẩn trong câu hỏi) và cần biết nó có khớp với tài liệu hay không.

QUY TẮC BẮT BUỘC:
1. Chỉ dùng thông tin có trong ngữ cảnh dưới đây. KHÔNG dùng kiến thức bên ngoài, KHÔNG suy đoán.
2. TUYỆT ĐỐI KHÔNG đồng ý chỉ vì người dùng nói như vậy. Người dùng hoàn toàn có thể khẳng định sai. Nhiệm vụ của bạn là ĐỐI CHIẾU, không phải làm hài lòng người hỏi.
3. Tách khẳng định thành từng chi tiết nhỏ và đối chiếu TỪNG chi tiết với ngữ cảnh. Chỉ cần MỘT chi tiết mâu thuẫn thì kết luận là SAI, kể cả khi các phần còn lại đều đúng.
4. Đặc biệt soi kỹ những chi tiết dễ bị nhớ ngược - đây là chỗ khẳng định sai hay ẩn nấp:
   - thứ tự / trước - sau / cái nào có trước
   - quan hệ nhân - quả (cái gì sinh ra cái gì)
   - có / không, bắt buộc / không bắt buộc
   - số lượng, con số, năm, số hiệu điều khoản
   - tên gọi, ai làm gì, thuộc về ai
5. Nếu ngữ cảnh KHÔNG hề nhắc tới nội dung được khẳng định, PHẢI kết luận "TÀI LIỆU KHÔNG ĐỀ CẬP" - không được đoán là đúng, cũng không được đoán là sai.
6. Trả lời theo ĐÚNG bố cục sau, KHÔNG đảo thứ tự và không thêm phần nào khác:

Căn cứ: trích NGUYÊN VĂN câu hoặc đoạn trong ngữ cảnh nói về nội dung được khẳng định, kèm số đoạn trích. Ví dụ: theo [2]: "..."

Đối chiếu: so từng chi tiết của khẳng định với đúng câu vừa trích. Nói rõ chi tiết nào khớp, chi tiết nào không khớp, và tài liệu thật sự nói gì.

KẾT LUẬN: <chọn đúng một trong ba: ĐÚNG | SAI | TÀI LIỆU KHÔNG ĐỀ CẬP>

BẮT BUỘC viết KẾT LUẬN SAU CÙNG, sau khi đã trích căn cứ và đối chiếu xong. Không được nêu kết luận ở đầu câu trả lời. Lý do: kết luận phải là thứ RÚT RA TỪ phần đối chiếu ngay phía trên nó - nếu bạn chốt trước rồi mới đối chiếu, bạn sẽ bảo vệ kết luận đã lỡ nói thay vì đọc lại tài liệu.
Trước khi viết KẾT LUẬN, hãy đọc lại chính phần Đối chiếu bạn vừa viết: nếu ở đó bạn đã chỉ ra tài liệu nói NGƯỢC với khẳng định của người hỏi, thì KẾT LUẬN phải là SAI.

7. Viết thẳng nội dung ba phần trên. KHÔNG in lại tên quy tắc, KHÔNG viết những câu kiểu "Bố cục trả lời theo quy định", "Phân tích ngữ cảnh", "Trả lời cuối cùng" - chỉ ba nhãn Căn cứ / Đối chiếu / KẾT LUẬN và nội dung của chúng.
8. Trả lời bằng tiếng Việt."""

HE_THONG_PROMPT_KIEM_CHUNG_EN = """You are a study assistant acting as a FACT-CHECKER. The user is making a claim (or a question with a hidden assumption) and needs to know whether it matches the documents.

MANDATORY RULES:
1. Use only the information in the context below. Do NOT use outside knowledge, do NOT guess.
2. NEVER agree just because the user said so. The user may well be wrong. Your job is to CHECK the claim, not to please the user.
3. Break the claim into individual details and check EACH one against the context. If even ONE detail contradicts the context, the verdict is FALSE, even when the rest is correct.
4. Look especially hard at details that are easy to remember backwards - this is where false claims hide:
   - ordering / which came first
   - cause and effect (what gives rise to what)
   - yes/no, mandatory/optional
   - quantities, numbers, years, article numbers
   - names, who does what, what belongs to whom
5. If the context does not mention the claim at all, you MUST conclude "NOT COVERED BY THE DOCUMENTS" - do not guess true, do not guess false.
6. Answer in EXACTLY this structure, do NOT reorder it and add nothing else:

Evidence: quote VERBATIM the sentence or passage from the context that speaks to the claim, with its excerpt number. For example: according to [2]: "..."

Comparison: check each detail of the claim against the sentence you just quoted. State which details match, which do not, and what the documents actually say.

VERDICT: <exactly one of: TRUE | FALSE | NOT COVERED BY THE DOCUMENTS>

You MUST write the VERDICT LAST, after quoting the evidence and doing the comparison. Never state the verdict at the start of your answer. The reason: the verdict has to be something DERIVED FROM the comparison right above it - if you commit to it first and compare afterwards, you will defend the verdict you already announced instead of re-reading the documents.
Before writing the VERDICT, re-read the Comparison you just wrote: if it says the documents state the OPPOSITE of the user's claim, the VERDICT must be FALSE.

7. Write the three parts directly. Do NOT echo the rule names, do NOT write headings like "Required structure", "Context analysis" or "Final answer" - only the three labels Evidence / Comparison / VERDICT and their content.
8. Answer in English."""

# Dấu hiệu câu hỏi mang tính KIỂM CHỨNG một khẳng định (thay vì hỏi thông tin thuần tuý).
# Cố ý ưu tiên ĐỘ CHÍNH XÁC hơn độ phủ: một câu hỏi thường bị nhận nhầm thành kiểm chứng sẽ
# nhận về câu trả lời có bố cục phán quyết khá cứng nhắc, nên thà bỏ sót còn hơn nhận nhầm.
# Phần bỏ sót vẫn được quy tắc số 4 của prompt thường (chống a dua) đỡ lại, nên không có
# trường hợp nào rơi hoàn toàn ra ngoài lưới.
_CAC_MAU_KIEM_CHUNG = [
    r"có phải", r"phải không", r"phải ko", r"đúng không", r"đúng ko", r"đúng chứ",
    r"có đúng", r"đúng hay sai", r"sai không", r"sai ko", r"chính xác không",
    r"thật không", r"hay không", r"khẳng định", r"nhận định", r"phát biểu sau",
    r"theo tôi", r"tôi nghĩ", r"tôi cho rằng", r"tôi tưởng", r"tôi được biết",
    r"tôi nhớ", r"mình nghĩ", r"mình tưởng", r"nghe nói",
    r"is it true", r"is that true", r"is it correct", r"is this correct",
    r"am i right", r"is n't it", r"isn't it", r"true or false",
    r"i think", r"i believe", r"i heard", r"right\?", r"correct\?",
]
_MAU_KIEM_CHUNG = re.compile("|".join(_CAC_MAU_KIEM_CHUNG), re.IGNORECASE)


def _phat_hien_ngon_ngu(cau_hoi: str) -> str:
    """Phát hiện câu hỏi là tiếng Anh hay tiếng Việt.

    Hệ thống chỉ hỗ trợ 2 ngôn ngữ, nên việc cần làm là CHỌN GIỮA HAI, không phải nhận diện
    ngôn ngữ trong số hàng trăm thứ tiếng. Bản trước viết `"en" if detect(...) == "en" else
    "vi"` - tức coi mọi thứ không phải tiếng Anh là tiếng Việt. Nghe thì hợp lý nhưng sai
    thực tế: langdetect chấm "What does criminal law regulate?" là tiếng Catalan (0.71) và
    tiếng Anh chỉ 0.29, nên câu tiếng Anh rõ ràng đó bị trả lời bằng tiếng Việt. Câu hỏi
    ngắn rất hay bị đoán nhầm sang các thứ tiếng Latin họ gần (ca, tl, it...).

    Ba bước, dừng ở bước nào có bằng chứng chắc chắn nhất:
    """
    # 1. Dấu tiếng Việt là bằng chứng không thể nhầm - không ngôn ngữ nào khác dùng bộ dấu
    #    này, nên thấy là chốt luôn, khỏi cần đoán.
    if _MAU_DAU_TIENG_VIET.search(cau_hoi):
        return "vi"

    # 2. So TRỰC TIẾP xác suất của đúng 2 ngôn ngữ quan tâm, bỏ qua thứ hạng chung. Nhờ vậy
    #    "en 0.29 vs ca 0.71" vẫn ra tiếng Anh - vì tiếng Catalan không nằm trong lựa chọn.
    try:
        xac_suat = {kq.lang: kq.prob for kq in detect_langs(cau_hoi)}
    except LangDetectException:
        xac_suat = {}
    if "en" in xac_suat or "vi" in xac_suat:
        return "en" if xac_suat.get("en", 0.0) > xac_suat.get("vi", 0.0) else "vi"

    # 3. Không có cả hai (câu quá ngắn, hoặc toàn thuật ngữ): tìm dấu vết tiếng Việt KHÔNG
    #    DẤU. Người Việt hay gõ không dấu ("SIFT la gi", "luat hinh su dieu chinh gi") -
    #    langdetect chấm những câu này thành tl/it, và nếu không bắt được thì hệ thống trả
    #    lời tiếng Anh cho người hỏi tiếng Việt.
    cac_tu = set(re.findall(r"[a-z]+", cau_hoi.lower()))
    if cac_tu & _TU_TIENG_VIET_KHONG_DAU:
        return "vi"

    # 4. Không có MỘT dấu vết tiếng Việt nào (không dấu phụ, không từ chức năng không dấu)
    #    mà vẫn là một câu có chữ -> nhiều khả năng là tiếng Anh. Tiếng Việt luôn để lại một
    #    trong hai dấu vết đó; văn bản không có gì cả thì khó mà là tiếng Việt.
    #    Bước này cần vì langdetect đôi khi trả về danh sách KHÔNG chứa cả en lẫn vi (vd
    #    ['ca:1.00'] cho một câu tiếng Anh rõ ràng), lúc đó bước 2 không quyết được.
    if len(cac_tu) >= 2:
        return "en"

    # 5. Hết cách đoán (chuỗi rỗng, toàn số, toàn dấu câu) -> tiếng Việt, vì đây là ngôn ngữ
    #    chính của hệ thống (xem README.md).
    return "vi"


def la_cau_hoi_kiem_chung(cau_hoi: str) -> bool:
    """Câu hỏi có đang đưa ra một khẳng định cần kiểm chứng hay không."""
    return bool(_MAU_KIEM_CHUNG.search(cau_hoi))


def _noi_lien_mach(truoc: str, sau: str, toi_da: int = 300) -> str:
    """Nối 2 chunk liền kề, bỏ phần bị lặp do overlap khi chia chunk.

    Chunking cố ý cho 2 chunk liên tiếp chồng lấn nhau (CHUNK_OVERLAP_TOKENS) để câu bị cắt
    đôi vẫn xuất hiện trọn ở một trong hai chunk. Nhưng khi nối lại để hiển thị/đưa vào
    prompt, phần chồng lấn đó thành ra lặp nguyên một đoạn - vừa khó đọc trong trích dẫn,
    vừa tốn ngân sách ngữ cảnh cho nội dung trùng. Tìm đoạn cuối của `truoc` trùng với đoạn
    đầu của `sau` rồi bỏ đi đúng một bản.
    """
    gioi_han = min(len(truoc), len(sau), toi_da)
    for do_dai in range(gioi_han, 20, -1):
        if truoc.endswith(sau[:do_dai]):
            return truoc + sau[do_dai:]
    return truoc + " " + sau


def _uoc_luong_so_token(*cac_phan: str) -> int:
    """Ước lượng số token của prompt mà KHÔNG cần tokenizer của LLM.

    Hệ thống chỉ có tokenizer của embedding model (họ XLM-R), không phải của Qwen, và nạp
    thêm một tokenizer nữa chỉ để đếm là cái giá không đáng: con số này dùng để CẤP PHÁT
    ngân sách cửa sổ ngữ cảnh, nên sai về phía cấp DƯ hoàn toàn vô hại, còn sai về phía cấp
    THIẾU thì tái tạo lại đúng bug num_ctx (xem config.OLLAMA_NUM_CTX). Vì vậy dùng tỷ lệ
    ký tự/token cố ý đặt THẤP hơn giá trị đo được cho tiếng Việt để luôn ước lượng dư.
    """
    return int(sum(len(p) for p in cac_phan) / config.SO_KY_TU_MOI_TOKEN_UOC_LUONG) + 1


def _tinh_num_ctx(so_token_prompt: int) -> int:
    """Cửa sổ ngữ cảnh cần cấp cho một prompt dài `so_token_prompt` token.

    Vì sao TÍNH ĐỘNG chứ không đặt một hằng số rồi thôi: hằng số đủ dùng hôm nay sẽ âm thầm
    không đủ vào ngày ai đó tăng TOP_K, tăng NGAN_SACH_KY_TU_MOI_DOAN, hay đổi sang model
    có system prompt dài hơn - và triệu chứng của việc thiếu (câu trả lời cụt, đoạn trích
    liên quan nhất bị cắt mất) KHÔNG hề giống một lỗi cấu hình, nên sẽ bị chẩn đoán nhầm
    đúng như đã xảy ra một lần rồi. Tính động + cảnh báo khiến lỗi này không tái diễn im lặng.

    Nhưng KHÔNG cấp đúng-vừa-đủ theo từng câu hỏi: Ollama coi num_ctx là một phần định danh
    của phiên bản model đang nạp, nên đổi giá trị này giữa hai lượt hỏi sẽ khiến nó NẠP LẠI
    model (mất hàng chục giây trên CPU). Vì vậy giá trị được làm tròn lên theo thang gấp đôi
    và bắt đầu từ OLLAMA_NUM_CTX: gần như mọi câu hỏi rơi vào cùng một bậc, không có lần nạp
    lại nào, mà cấu hình quá tay vẫn được nới thay vì bị cắt.
    """
    can = so_token_prompt + config.OLLAMA_DU_PHONG_TOKEN_SINH
    tran = max(config.OLLAMA_NUM_CTX_TOI_DA, config.OLLAMA_NUM_CTX)
    num_ctx = config.OLLAMA_NUM_CTX
    while num_ctx < can and num_ctx < tran:
        num_ctx *= 2
    num_ctx = min(num_ctx, tran)

    if can > num_ctx:
        # Đã chạm trần RAM. Prompt vẫn có thể lọt (phần bị ép là ngân sách sinh), nhưng đây
        # là dấu hiệu cấu hình đã vượt quá thứ máy này gánh nổi - nói thẳng ra chỗ cần sửa
        # thay vì để người dùng gặp lại triệu chứng "câu trả lời tự nhiên ngắn đi".
        logger.warning(
            "Prompt ~%d token + %d token dự phòng sinh = %d, vượt trần OLLAMA_NUM_CTX_TOI_DA=%d. "
            "Hạ TOP_K hoặc NGAN_SACH_KY_TU_MOI_DOAN (ĐỪNG hạ num_ctx), hoặc nâng trần nếu máy đủ RAM.",
            so_token_prompt, config.OLLAMA_DU_PHONG_TOKEN_SINH, can, tran,
        )
    return num_ctx


def _ghep_prompt(
    cau_hoi: str,
    cac_chunk: List[Dict],
    ngon_ngu: str,
    la_kiem_chung: bool,
    ngu_canh_hoi_thoai: str = "",
) -> str:
    """Ghép Top-K đoạn trích vào prompt, đánh số từng đoạn kèm nguồn để LLM trích dẫn đúng
    theo số thứ tự [1], [2]... khớp với thứ tự hiển thị ở citation.py.

    ngon_ngu ("vi"/"en") quyết định nhãn/tiêu đề trong prompt (không phải nội dung chunk,
    vốn giữ nguyên ngôn ngữ gốc của tài liệu) - để toàn bộ prompt nhất quán 1 ngôn ngữ.
    """
    if ngon_ngu == "en":
        nhan_nguon, nhan_trang = "Source", "page/slide"
        tieu_de_ngu_canh, tieu_de_cau_hoi = "CONTEXT", "QUESTION"
        huong_dan = (
            "Check the claim above against the context and answer in the required structure:"
            if la_kiem_chung
            else "Answer based on the context above:"
        )
    else:
        nhan_nguon, nhan_trang = "Nguồn", "trang/slide"
        tieu_de_ngu_canh, tieu_de_cau_hoi = "NGỮ CẢNH", "CÂU HỎI"
        huong_dan = (
            "Đối chiếu khẳng định trên với ngữ cảnh và trả lời theo đúng bố cục đã quy định:"
            if la_kiem_chung
            else "Trả lời dựa trên ngữ cảnh trên:"
        )

    cac_doan = [
        f"[{i}] ({nhan_nguon}: {chunk['nguon']}, {nhan_trang} {chunk['trang']})\n{chunk['noidung']}"
        for i, chunk in enumerate(cac_chunk, start=1)
    ]
    ngu_canh = "\n\n".join(cac_doan)

    # Khối ngữ cảnh hội thoại (chỉ có với câu hỏi nối tiếp) đặt SAU đoạn trích và TRƯỚC câu
    # hỏi, dưới nhãn riêng của nó. Ba chi tiết đều có lý do:
    #   - Nhãn riêng, kèm câu "KHÔNG PHẢI nguồn thông tin, tuyệt đối không trích dẫn": ngữ
    #     cảnh này chứa câu hỏi của chính người dùng, không phải tài liệu. Cho model coi nó
    #     ngang hàng với đoạn trích là mở đúng cánh cửa mà cả hệ thống tồn tại để đóng.
    #   - Chỉ chứa CÁC CÂU HỎI trước, không chứa câu trả lời trước: câu trả lời cũ là lời
    #     của model, và để model trích lại lời của chính nó như thể là tài liệu thì trích dẫn
    #     mất sạch ý nghĩa.
    #   - Đặt sát câu hỏi để model đọc nó đúng lúc cần giải nghĩa "cái thứ hai" là gì.
    khoi_ngu_canh = f"\n{ngu_canh_hoi_thoai}\n" if ngu_canh_hoi_thoai else ""

    return f"""{tieu_de_ngu_canh}:
{ngu_canh}
{khoi_ngu_canh}
{tieu_de_cau_hoi}: {cau_hoi}

{huong_dan}"""


class RagPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        reranker_service: Optional[RerankerService] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        # None = không rerank. Truyền từ ngoài vào (không tự tạo) theo đúng lý do ở §5.8:
        # model reranker nặng ~2GB, phải nạp đúng 1 lần cho cả phiên chứ không phải mỗi
        # lần khởi tạo pipeline.
        self.reranker_service = reranker_service
        # Điểm rerank cao nhất của LƯỢT TRUY XUẤT GẦN NHẤT, dùng cho ngưỡng từ chối
        # (NGUONG_DIEM_RERANK_TOI_THIEU). None khi chưa truy xuất lần nào hoặc khi tắt
        # rerank - lúc đó ngưỡng này tự bỏ qua, hệ thống lùi về đúng hành vi cũ.
        self.diem_rerank_cao_nhat = None
        # Kết quả của bước chuẩn bị truy vấn ở lượt GẦN NHẤT (rag/tiep_noi_hoi_thoai.py):
        # {"cau_hoi_goc", "cau_hoi_truy_xuat", "da_viet_lai", "la_tiep_noi"}. Tầng trên đọc
        # nó để ghép prompt bằng đúng câu đã truy xuất, và để nói cho người dùng biết hệ
        # thống đã hiểu câu hỏi nối tiếp của họ thành câu gì.
        self.truy_van_da_dung = None
        # Bộ đếm token THẬT do Ollama trả về ở lượt gọi gần nhất: {"prompt_eval_count",
        # "eval_count", "done_reason", "num_ctx", "uoc_luong_token_prompt"}. Xem
        # _ghi_nhan_thong_ke_llm - đây là thứ duy nhất phát hiện được prompt bị cắt.
        self.thong_ke_llm = None
        self._ollama_client = ollama.Client(host=config.OLLAMA_HOST)
        # Đặt False khi máy chủ Ollama báo model không hỗ trợ chế độ suy luận, để những lần
        # gọi sau không phải thử-rồi-hỏng thêm lần nào nữa (xem _goi_llm).
        self._ho_tro_thinking = True

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------
    def _ung_vien(
        self, cac_truy_van: List[tuple], so_ung_vien: int, nguon_cho_phep: Optional[Set[str]]
    ) -> Tuple[List[tuple], Set[int]]:
        """Lấy ứng viên từ mọi nhánh tìm kiếm rồi hợp nhất bằng RRF.

        cac_truy_van: [(cau_hoi, vector, trong_so)]. Phần tử ĐẦU TIÊN là truy vấn CHÍNH -
        điểm cosine trả ra được đo theo nó, vì mọi ngưỡng trong hệ thống đều hiệu chỉnh trên
        một thang cosine duy nhất và trộn cosine của hai truy vấn khác nhau vào cùng một
        trường là kiểu lỗi rất khó lần ra (cùng lý do đã giữ nguyên cosine khi rerank, §5.24).

        Trả về ([(vi_tri, diem_cosine)] sắp xếp theo thứ hạng hợp nhất, tốt nhất trước;
        tập vị trí do BM25 CỨU HỘ bơm vào). Tập thứ hai cần cho _xep_hang_lai - xem
        config.SO_UNG_VIEN_BM25_CUU_HO.

        Reciprocal Rank Fusion cộng nghịch đảo THỨ HẠNG chứ không cộng điểm số, vì cosine
        (chặn trong [-1, 1]) và BM25 (không chặn trên, phụ thuộc độ hiếm từ khoá) là 2 thang
        đo khác hẳn nhau - cộng thẳng thì nhánh nào có thang lớn hơn sẽ nuốt trọn nhánh kia.
        Chuẩn hoá điểm về cùng thang cũng là một hướng, nhưng phải chọn cách chuẩn hoá và nó
        rất nhạy với ngoại lệ; RRF không cần tham số nào ngoài RRF_K nên ổn định hơn nhiều.

        NHIỀU TRUY VẤN đi qua đúng cơ chế đó, không cần cơ chế mới: câu hỏi gốc và câu hỏi đã
        viết lại theo ngữ cảnh hội thoại (rag/tiep_noi_hoi_thoai.py) chỉ là thêm hai danh
        sách xếp hạng nữa để RRF hợp nhất - vốn đúng việc RRF sinh ra để làm. Nhờ vậy một bản
        viết lại sai KHÔNG xoá được kết quả đúng của câu gốc, nó chỉ làm thứ hạng nhiễu đi.
        """
        # FAISS/BM25 đều không lọc được theo metadata trong lúc tìm, nên việc lọc theo nguồn
        # người dùng tick chọn ở UI phải làm tại đây (lý do phải over-fetch từ đầu).
        def duoc_phep(vi_tri: int) -> bool:
            return (
                nguon_cho_phep is None
                or self.vector_store.metadata[vi_tri]["nguon"] in nguon_cho_phep
            )

        # RRF CÓ TRỌNG SỐ. Bản đầu cộng 2 nhánh ngang nhau, và chính điều đó phá truy xuất
        # chéo ngôn ngữ: câu hỏi tiếng Việt hỏi về tài liệu tiếng Anh thì BM25 không khớp nổi
        # từ nào với tài liệu ĐÚNG, nhưng khớp rất "tự tin" với tài liệu tiếng Việt SAI - mà
        # RRF lại coi hạng 1 của BM25 ngang hạng 1 của dense. Trọng số cho phép nói rõ mức
        # tin cậy của từng nhánh; mặc định BM25 = 0 (xem số đo ở config.TRONG_SO_BM25).
        diem_rrf: Dict[int, float] = defaultdict(float)
        dense_chinh: List[tuple] = []
        for thu_tu_truy_van, (cau_hoi, vector, trong_so_truy_van) in enumerate(cac_truy_van):
            ket_qua_dense = self.vector_store.tim_kiem_vi_tri(vector, top_k=so_ung_vien)
            if thu_tu_truy_van == 0:
                dense_chinh = ket_qua_dense
            ket_qua_lexical = (
                self.vector_store.tim_kiem_tu_khoa(cau_hoi, so_ung_vien)
                if config.TRONG_SO_BM25 > 0
                else []
            )
            for danh_sach, trong_so_nhanh in (
                (ket_qua_dense, 1.0),
                (ket_qua_lexical, config.TRONG_SO_BM25),
            ):
                thu_hang = 0
                for vi_tri, _ in danh_sach:
                    if not duoc_phep(vi_tri):
                        continue
                    thu_hang += 1
                    diem_rrf[vi_tri] += (
                        trong_so_truy_van * trong_so_nhanh / (config.RRF_K + thu_hang)
                    )
        # BM25 CỨU HỘ: bơm thêm ứng viên vào tập đưa đi rerank mà KHÔNG cho một điểm RRF
        # nào (điểm 0 -> luôn xếp cuối). Đây là chỗ tách bạch hai vai trò của BM25: giúp
        # RECALL (đoạn chứa từ khoá hiếm / mã định danh / tên riêng OOV chắc chắn có mặt
        # trong tập ứng viên) mà không có quyền PRECISION (không đẩy được thứ hạng của
        # chính nó lên). Việc xếp hạng để cross-encoder quyết - nếu đoạn thật sự liên quan
        # nó sẽ được đẩy lên, nếu không thì nằm yên ở cuối. Xem config.SO_UNG_VIEN_BM25_CUU_HO
        # để biết vì sao cách này KHÔNG lặp lại cái hại đã đo được ở TRONG_SO_BM25.
        vi_tri_cuu_ho: Set[int] = set()
        if config.SO_UNG_VIEN_BM25_CUU_HO > 0:
            for vi_tri, _ in self.vector_store.tim_kiem_tu_khoa(
                cac_truy_van[0][0], config.SO_UNG_VIEN_BM25_CUU_HO
            ):
                if duoc_phep(vi_tri) and vi_tri not in diem_rrf:
                    diem_rrf[vi_tri] = 0.0
                    vi_tri_cuu_ho.add(vi_tri)

        if not diem_rrf:
            return [], set()

        # Mọi ngưỡng lọc và điểm hiển thị đều quy về cosine cho cùng một thang đo; chunk chỉ
        # do BM25 hoặc do truy vấn phụ tìm ra thì chưa có điểm cosine nên phải tính bù ở đây.
        vector_chinh = cac_truy_van[0][1]
        diem_cosine = {vi_tri: diem for vi_tri, diem in dense_chinh if vi_tri in diem_rrf}
        con_thieu = [vi_tri for vi_tri in diem_rrf if vi_tri not in diem_cosine]
        diem_cosine.update(self.vector_store.diem_cosine(con_thieu, vector_chinh))

        thu_tu = sorted(diem_rrf, key=lambda i: (-diem_rrf[i], -diem_cosine[i]))
        return [(vi_tri, diem_cosine[vi_tri]) for vi_tri in thu_tu], vi_tri_cuu_ho

    def _xep_hang_lai(
        self, cau_hoi: str, ung_vien: List[tuple], vi_tri_cuu_ho: Optional[Set[int]] = None
    ) -> List[tuple]:
        """Xếp lại thứ tự ứng viên bằng cross-encoder (xem rag/reranker.py).

        Chỉ rerank SO_UNG_VIEN_RERANK ứng viên đầu, phần đuôi giữ nguyên thứ tự RRF. Lý do
        giữ đuôi thay vì cắt bỏ: các bước sau còn lọc tiếp (trần đoạn mỗi trang, sàn điểm),
        nên nếu cắt cụt ở đây thì có trường hợp không còn đủ ứng viên để lấp TOP_K.

        vi_tri_cuu_ho: các ứng viên do BM25 bơm vào với điểm RRF = 0 (config.SO_UNG_VIEN_BM25_CUU_HO).
        Chúng LUÔN được chấm, kể cả khi rơi ngoài SO_UNG_VIEN_RERANK ứng viên đầu - nếu
        không thì cả cơ chế cứu hộ vô nghĩa: điểm RRF 0 đẩy chúng xuống cuối, mà xuống cuối
        thì không bao giờ được cross-encoder nhìn tới, tức chúng chỉ tồn tại cho có.

        Chấm điểm trên NỘI DUNG CHUNK GỐC, chưa mở rộng ngữ cảnh: vừa rẻ hơn (đoạn ngắn hơn
        nhiều), vừa đúng hơn về mặt đo lường - ta đang hỏi "chunk này có khớp câu hỏi không",
        chứ không phải "cả vùng quanh nó có khớp không". Mở rộng ngữ cảnh là việc làm SAU khi
        đã chọn xong, để LLM đọc đủ ý (§5.11).

        QUAN TRỌNG: điểm cosine đi kèm mỗi ứng viên được GIỮ NGUYÊN, không thay bằng điểm
        rerank. Rerank chỉ đổi THỨ TỰ CHỌN. Nhờ vậy diem_similarity hiển thị trên UI, sàn lọc
        NGUONG_DIEM_TOI_THIEU và mọi chỗ đang đọc điểm đó vẫn giữ đúng một thang đo duy nhất
        (cosine) - trộn 2 thang đo vào cùng một trường là kiểu lỗi rất khó lần ra về sau.
        """
        if not self.reranker_service or len(ung_vien) < 2:
            return ung_vien

        vi_tri_cuu_ho = vi_tri_cuu_ho or set()
        so_dau = min(len(ung_vien), config.SO_UNG_VIEN_RERANK)
        chi_so_cham = list(range(so_dau)) + [
            i for i in range(so_dau, len(ung_vien)) if ung_vien[i][0] in vi_tri_cuu_ho
        ]
        cac_doan = [self.vector_store.metadata[ung_vien[i][0]]["noidung"] for i in chi_so_cham]
        diem_rerank = self.reranker_service.xep_hang(cau_hoi, cac_doan)
        self.diem_rerank_cao_nhat = float(max(diem_rerank)) if len(diem_rerank) else None

        diem_theo_chi_so = dict(zip(chi_so_cham, diem_rerank))
        thu_tu_moi = sorted(chi_so_cham, key=lambda i: -diem_theo_chi_so[i])
        duoi = [i for i in range(len(ung_vien)) if i not in diem_theo_chi_so]
        return [ung_vien[i] for i in thu_tu_moi] + [ung_vien[i] for i in duoi]

    def _dung_doan_trich(self, vi_tri_neo: int) -> Dict:
        """Dựng 1 đoạn trích liền mạch quanh chunk khớp nhất ("neo").

        Bắt đầu từ chính chunk neo rồi mở rộng luân phiên sang chunk liền sau / liền trước
        TRONG CÙNG TRANG, cho tới khi chạm ngân sách ký tự. Ưu tiên mở rộng về phía SAU
        trước vì kiểu mất mát hay gặp nhất là một câu hoặc một đoạn liệt kê bị ranh giới
        chunk cắt ngang, phần còn thiếu nằm ở chunk kế tiếp.

        Đây là điểm thay thế cho cách "gộp nguyên trang" của bản trước: vẫn nối lại được
        phần bị cắt (mục đích ban đầu), nhưng không kéo theo toàn bộ phần còn lại của trang
        vốn chẳng liên quan gì tới câu hỏi. Với tài liệu ngắn (slide, trang thưa chữ) thì
        cả trang thường vẫn lọt trong ngân sách, nên hành vi y hệt bản cũ - nghĩa là cách
        làm này đúng cho cả tài liệu dài lẫn ngắn, không phải đánh đổi bên này lấy bên kia.

        MỞ RỘNG QUA RANH GIỚI TRANG (config.MO_RONG_QUA_RANH_GIOI_TRANG): bản trước chặn
        cứng trong đúng một (nguồn, trang). Với slide thì đúng - mỗi slide là một đơn vị nội
        dung tự đóng. Với PDF văn bản chảy liên tục thì SAI: một định nghĩa bắt đầu cuối
        trang 12 và kết thúc đầu trang 13 không bao giờ được nối lại, vì chunk neo nằm cuối
        trang 12 và việc mở rộng chạm hết mảng của trang rồi dừng. Nay phạm vi mở rộng là
        toàn bộ tài liệu theo thứ tự đọc, có hai chốt chặn: ngân sách ký tự (như cũ) và số
        trang tối đa được vượt qua mỗi hướng (config.SO_TRANG_TOI_DA_MO_RONG) - chốt thứ hai
        giữ cho slide thưa chữ không hút thêm 2-3 slide xung quanh cho đầy ngân sách.

        Số trang của trích dẫn vẫn là trang của chunk NEO. Các trang bị đi qua được trả kèm
        ở "cac_trang" để tầng trên biết đoạn này đọc xuyên mấy trang, nhưng chúng KHÔNG được
        dùng làm nguồn trích dẫn: phần thật sự khớp câu hỏi là chunk neo, phần mở rộng chỉ
        là ngữ cảnh đọc kèm.
        """
        neo = self.vector_store.metadata[vi_tri_neo]
        if config.MO_RONG_QUA_RANH_GIOI_TRANG:
            pham_vi = self.vector_store.chi_muc_nguon[neo["nguon"]]
        else:
            pham_vi = self.vector_store.chi_muc_trang[(neo["nguon"], neo["trang"])]
        chi_so = pham_vi.index(vi_tri_neo)

        def qua_xa_trang(chi_so_chunk: int) -> bool:
            """Chunk này đã cách chunk neo quá nhiều trang chưa."""
            trang_do = self.vector_store.metadata[pham_vi[chi_so_chunk]]["trang"]
            try:
                return abs(int(trang_do) - int(neo["trang"])) > config.SO_TRANG_TOI_DA_MO_RONG
            except (TypeError, ValueError):
                # Loader nào đó không đánh số trang bằng số -> không so được khoảng cách,
                # lùi về hành vi an toàn là không cho vượt sang trang khác.
                return trang_do != neo["trang"]

        da_chon = [chi_so]
        con_lai = config.NGAN_SACH_KY_TU_MOI_DOAN - len(neo["noidung"])
        trai, phai = chi_so - 1, chi_so + 1
        uu_tien_phai = True
        while con_lai > 0 and (trai >= 0 or phai < len(pham_vi)):
            if uu_tien_phai and phai < len(pham_vi):
                ke_tiep = phai
                phai += 1
            elif trai >= 0:
                ke_tiep = trai
                trai -= 1
            elif phai < len(pham_vi):
                ke_tiep = phai
                phai += 1
            else:
                break
            uu_tien_phai = not uu_tien_phai

            noi_dung_them = self.vector_store.metadata[pham_vi[ke_tiep]]["noidung"]
            # Chunk không vừa ngân sách, hoặc đã ra ngoài phạm vi trang cho phép -> ĐÓNG HẲN
            # hướng đó lại rồi thử hướng còn lại, thay vì bỏ qua nó để lấy chunk xa hơn. Bỏ
            # qua sẽ tạo ra lỗ hổng giữa đoạn trích: nội dung nhảy cóc mà không có dấu hiệu
            # gì, người đọc tưởng 2 phần đứng liền nhau trong tài liệu. Đoạn trích buộc phải
            # liền mạch. (Với ranh giới trang, đóng hướng còn là điều DUY NHẤT đúng: các
            # chunk xa hơn ở hướng đó chỉ có thể cách xa hơn nữa.)
            if len(noi_dung_them) > con_lai or qua_xa_trang(ke_tiep):
                if ke_tiep >= chi_so:
                    phai = len(pham_vi)
                else:
                    trai = -1
                continue
            da_chon.append(ke_tiep)
            con_lai -= len(noi_dung_them)

        da_chon.sort()
        noi_dung = ""
        for chi_so_chunk in da_chon:
            phan = self.vector_store.metadata[pham_vi[chi_so_chunk]]["noidung"]
            noi_dung = _noi_lien_mach(noi_dung, phan) if noi_dung else phan

        return {
            "nguon": neo["nguon"],
            "trang": neo["trang"],
            # Mọi trang mà đoạn trích này đi qua (thường chỉ có 1). Dùng để hiển thị và để
            # đo được tần suất mở rộng xuyên trang thật sự xảy ra - không phải nguồn trích dẫn.
            "cac_trang": sorted(
                {self.vector_store.metadata[pham_vi[i]]["trang"] for i in da_chon},
                key=str,
            ),
            "noidung": noi_dung,
            # Chuyển tiếp loại nội dung + đường dẫn ảnh ra ngoài để UI hiển thị đúng dạng
            # (bảng render thành bảng, ảnh hiện ra ảnh). Lấy từ chunk NEO vì đó là phần
            # thật sự khớp câu hỏi - các chunk mở rộng xung quanh chỉ là ngữ cảnh thêm.
            "loai_noi_dung": neo.get("loai_noi_dung", "van_ban"),
            "duong_dan_anh": neo.get("duong_dan_anh", ""),
            # Chunk khớp nhất, tách riêng khỏi cả đoạn đã mở rộng: đây mới là phần thật sự
            # khiến đoạn này được chọn, nên citation.py dùng nó làm đoạn trích hiển thị thay
            # vì cắt bừa mấy trăm ký tự đầu (nguyên nhân khiến trích dẫn trước đây trỏ sai chỗ).
            "doan_khop": neo["noidung"],
            "cac_vi_tri": {pham_vi[i] for i in da_chon},
        }

    def truy_xuat(
        self,
        cau_hoi: str,
        top_k: int = None,
        nguon_cho_phep: Optional[Set[str]] = None,
        lich_su: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Tìm các đoạn trích liên quan nhất tới câu hỏi.

        Trả về list dict {"nguon", "trang", "noidung", "doan_khop", "diem_similarity"},
        sắp xếp theo độ liên quan (tốt nhất trước).

        nguon_cho_phep: tập tên file được phép dùng (None = dùng tất cả).
        lich_su: lịch sử hội thoại [{"role", "content"}] để hiểu câu hỏi NỐI TIẾP. None hoặc
        rỗng thì hành vi y hệt bản chưa có tính năng này - đây cũng là lý do evaluation và
        test không phải sửa gì: chúng hỏi từng câu độc lập nên không truyền lịch sử.

        Câu hỏi đã dùng để truy xuất (có thể là bản viết lại) nằm ở
        `self.truy_van_da_dung` sau khi hàm này chạy xong, để tầng trên còn (a) ghép prompt
        bằng đúng câu đó và (b) nói cho người dùng biết hệ thống đã hiểu câu hỏi ra sao.
        """
        top_k = top_k or config.TOP_K
        # Xoá điểm rerank của lượt TRƯỚC ngay từ đầu: nếu lượt này không chạy rerank (quá ít
        # ứng viên, hoặc tắt rerank) mà vẫn còn giá trị cũ, ngưỡng từ chối sẽ phán xét câu
        # hỏi hiện tại bằng điểm của một câu hỏi khác - sai âm thầm, rất khó lần ra.
        self.diem_rerank_cao_nhat = None

        # Đưa ngữ cảnh hội thoại vào truy vấn khi đây là câu hỏi NỐI TIẾP. Tầng nhận diện
        # tất định chạy trước, và đường mặc định (ghép câu hỏi trước) cũng tất định - nên
        # bước này không thêm lượt gọi LLM nào (§5.58).
        self.truy_van_da_dung = chuan_bi_truy_van(cau_hoi, lich_su, client=self._ollama_client)
        cau_hoi_chinh = self.truy_van_da_dung["cau_hoi_chinh"]

        if self.vector_store.so_luong_vector == 0:
            return []

        so_ung_vien = min(
            max(top_k * config.HE_SO_OVER_FETCH, config.SO_UNG_VIEN_TOI_THIEU),
            self.vector_store.so_luong_vector,
        )
        # Truy vấn CHÍNH là bản đã mang ngữ cảnh (nếu có): nó là câu đủ nghĩa, nên cosine của
        # nó mới là con số so được với các ngưỡng vốn hiệu chỉnh trên câu hỏi độc lập.
        cac_truy_van = [(
            cau_hoi_chinh,
            self.embedding_service.encode_cau_hoi([cau_hoi_chinh]),
            1.0,
        )]
        # Câu GỐC được giữ làm một nhánh riêng để một lần ghép ngữ cảnh sai không xoá được
        # kết quả đúng - xem config.TRONG_SO_TRUY_VAN_GOC.
        for van_ban, trong_so in (
            [(cau_hoi, config.TRONG_SO_TRUY_VAN_GOC)] if cau_hoi_chinh != cau_hoi else []
        ) + [(v, config.TRONG_SO_TRUY_VAN_GOC)
             for v in self.truy_van_da_dung["cac_truy_van_phu"]]:
            cac_truy_van.append(
                (van_ban, self.embedding_service.encode_cau_hoi([van_ban]), trong_so)
            )

        ung_vien, vi_tri_cuu_ho = self._ung_vien(cac_truy_van, so_ung_vien, nguon_cho_phep)
        # Rerank theo truy vấn CHÍNH: cross-encoder chấm cặp (câu hỏi, đoạn) nên nó cần một
        # câu hỏi đầy đủ nghĩa. Đưa "Thế còn cái thứ hai?" vào đây thì mọi đoạn đều bị chấm
        # gần 0 - và vì điểm rerank còn là cơ chế TỪ CHỐI (§5.29), câu nối tiếp hợp lệ sẽ bị
        # từ chối oan. Đây là lý do quan trọng nhất khiến bước này không thể chỉ là "thêm một
        # nhánh truy xuất cho vui".
        ung_vien = self._xep_hang_lai(cau_hoi_chinh, ung_vien, vi_tri_cuu_ho)

        # TRẦN SỐ ĐOẠN MỖI TRANG - THÍCH ỨNG. Trần chỉ có ý nghĩa khi CÓ nhiều trang để phân
        # bổ. Với câu hỏi mà toàn bộ câu trả lời nằm gọn trong một trang (một mục định nghĩa,
        # một bảng tiêu chí, một quy trình - rất phổ biến), ép lấy đoạn từ trang khác vừa bỏ
        # sót phần đúng vừa làm loãng ngữ cảnh bằng phần sai. Đo độ đa dạng trang trên đầu
        # bảng ứng viên rồi mới quyết định; đủ đa dạng thì áp trần như cũ, không thì bỏ trần.
        # SO_UNG_VIEN_XET_DA_DANG_TRANG = 0 -> tắt hẳn phần thích ứng, luôn áp trần cứng
        # (đường lui để đo đối chứng với hành vi cũ).
        so_trang_ung_vien = len({
            (
                self.vector_store.metadata[vi_tri]["nguon"],
                self.vector_store.metadata[vi_tri]["trang"],
            )
            for vi_tri, _ in ung_vien[: config.SO_UNG_VIEN_XET_DA_DANG_TRANG]
        }) if config.SO_UNG_VIEN_XET_DA_DANG_TRANG > 0 else top_k
        tran_moi_trang = (
            config.SO_DOAN_TOI_DA_MOI_TRANG if so_trang_ung_vien >= top_k else top_k
        )

        cac_doan: List[Dict] = []
        da_dung: set = set()  # vị trí chunk đã nằm trong một đoạn nào đó
        so_doan_theo_trang: Dict[tuple, int] = defaultdict(int)
        so_doan_anh = 0
        for vi_tri, diem in ung_vien:
            if len(cac_doan) >= top_k:
                break
            # Chunk này đã nằm trong một đoạn dựng trước đó (thường là chunk liền kề của
            # chính đoạn đó) -> không tạo đoạn mới trùng nội dung, chỉ nâng điểm của đoạn cũ.
            if vi_tri in da_dung:
                for doan in cac_doan:
                    if vi_tri in doan["cac_vi_tri"]:
                        doan["diem_similarity"] = max(doan["diem_similarity"], diem)
                        break
                continue

            khoa_trang = (
                self.vector_store.metadata[vi_tri]["nguon"],
                self.vector_store.metadata[vi_tri]["trang"],
            )
            # Trần số đoạn mỗi trang: với tài liệu dài, các chunk liền kề cùng 1 trang có
            # điểm gần bằng nhau nên rất dễ chiếm sạch TOP_K suất, đẩy hết những trang liên
            # quan khác ra ngoài. Đây chính là kiểu lỗi "câu trả lời chỉ bám 1 chỗ trong tài
            # liệu dài, bỏ sót phần còn lại" mà người dùng gặp phải.
            if so_doan_theo_trang[khoa_trang] >= tran_moi_trang:
                continue

            # Trần riêng cho đoạn LÀ ẢNH: mô tả ảnh do model vision sinh ra khá dài, nên với
            # tài liệu nhiều hình chúng dễ chiếm hết suất của các trang văn bản đúng (đo
            # thực tế: Recall@K tụt 0.96 -> 0.92 khi chưa có trần này). Xem config.SO_DOAN_ANH_TOI_DA.
            la_anh = self.vector_store.metadata[vi_tri].get("loai_noi_dung") == "anh"
            if la_anh and so_doan_anh >= config.SO_DOAN_ANH_TOI_DA:
                continue

            doan = self._dung_doan_trich(vi_tri)
            doan["diem_similarity"] = diem
            cac_doan.append(doan)
            da_dung |= doan["cac_vi_tri"]
            so_doan_theo_trang[khoa_trang] += 1
            so_doan_anh += la_anh

        # Sàn lọc rác: rỗng -> sinh_cau_tra_loi() từ chối luôn, không gọi LLM.
        # Cố ý đặt THẤP và KHÔNG dùng làm cơ chế chính để phát hiện câu hỏi ngoài phạm vi -
        # đã đo và xác nhận không tồn tại ngưỡng nào làm được việc đó (câu tiếng Anh đúng
        # chủ đề cho điểm thấp hơn câu tiếng Việt lạc đề, xem config.NGUONG_DIEM_TOI_THIEU).
        # Việc phán đoán "tài liệu có nói về chuyện này không" thuộc về LLM, vì nó ĐỌC được
        # nội dung đoạn trích chứ không chỉ nhìn một con số.
        so_doan_truoc_loc = len(cac_doan)
        cac_doan = [d for d in cac_doan if d["diem_similarity"] >= config.NGUONG_DIEM_TOI_THIEU]

        # TẦNG LỌC TƯƠNG ĐỐI. Cosine của E5 không phải thang đo tuyệt đối: giá trị của nó
        # trôi theo domain, ngôn ngữ, độ dài chunk, phong cách văn bản - nên một hằng số
        # hiệu chỉnh trên corpus này sẽ cắt oan trên corpus khác. Tỷ lệ giữa các đoạn TRONG
        # CÙNG MỘT LƯỢT thì không trôi, vì cả lượt dùng chung một câu hỏi và một model.
        # Sàn tuyệt đối ở trên vẫn giữ nhưng chỉ còn vai trò chặn rác (xem
        # config.NGUONG_DIEM_TOI_THIEU và TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT).
        if cac_doan and config.TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT > 0:
            diem_cao_nhat = max(d["diem_similarity"] for d in cac_doan)
            san_tuong_doi = diem_cao_nhat * config.TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT
            cac_doan = [d for d in cac_doan if d["diem_similarity"] >= san_tuong_doi]

        if config.LOG_PHAN_BO_DIEM:
            # Số liệu để TRẢ LỜI bằng đo đạc câu hỏi "ngưỡng có đang cắt oan trên corpus này
            # không", thay vì đổi ngưỡng theo cảm tính - đúng cách đã dùng khi đo BM25.
            # Chạy trên ~10 câu ở corpus cũ và ~10 câu ở corpus mới rồi so số đoạn sống sót.
            logger.info(
                "PHAN_BO_DIEM | cosine=%s | rerank_cao_nhat=%s | trang_ung_vien=%d "
                "tran_moi_trang=%d | song_sot=%d/%d | hoi: %.60s",
                [round(d["diem_similarity"], 3) for d in cac_doan],
                None if self.diem_rerank_cao_nhat is None else round(self.diem_rerank_cao_nhat, 5),
                so_trang_ung_vien,
                tran_moi_trang,
                len(cac_doan),
                so_doan_truoc_loc,
                cau_hoi,
            )

        if not cac_doan:
            logger.info(
                "Không đoạn nào đạt ngưỡng %.2f cho câu hỏi: %.80s",
                config.NGUONG_DIEM_TOI_THIEU,
                cau_hoi,
            )
            return []

        # Tuyến phòng thủ THỨ HAI, dựa trên điểm rerank thay vì cosine. Đây mới là thứ bắt
        # được câu hỏi ngoài phạm vi tài liệu: đo thực tế cho thấy câu lạc đề rơi về gần 0
        # tuyệt đối (0.000-0.003) trong khi câu đúng chủ đề - kể cả hỏi bằng tiếng Anh trên
        # tài liệu tiếng Việt - vẫn từ 0.019 trở lên. Cosine KHÔNG làm được việc này (xem
        # config.NGUONG_DIEM_RERANK_TOI_THIEU để biết số đo đầy đủ và lý do chọn ngưỡng thấp).
        if self.diem_rerank_cao_nhat is not None and (
            self.diem_rerank_cao_nhat < config.NGUONG_DIEM_RERANK_TOI_THIEU
        ):
            logger.info(
                "Điểm rerank cao nhất %.4f dưới ngưỡng %.4f - coi như ngoài phạm vi tài "
                "liệu, câu hỏi: %.80s",
                self.diem_rerank_cao_nhat,
                config.NGUONG_DIEM_RERANK_TOI_THIEU,
                cau_hoi,
            )
            return []

        for doan in cac_doan:
            doan.pop("cac_vi_tri", None)  # chi tiết nội bộ, không cần lộ ra ngoài pipeline
        return cac_doan

    # ------------------------------------------------------------------
    # SINH CÂU TRẢ LỜI
    # ------------------------------------------------------------------
    def _goi_llm_theo_luong(
        self, he_thong_prompt: str, prompt_nguoi_dung: str, bat_thinking: bool
    ) -> Iterator[Dict]:
        """Gọi Ollama ở chế độ STREAMING, sinh ra từng mảnh {"loai", "them"} khi model viết.

        Vì sao streaming là bắt buộc chứ không phải trang trí: trên CPU, qwen3:4b mất 35-70
        giây cho một câu hỏi vì luôn sinh một đoạn suy luận dài trước khi trả lời (đã đo,
        không tắt được bằng tham số - xem ghi chú về `think` bên dưới). Ở chế độ chờ-rồi-trả
        -một-cục, toàn bộ khoảng thời gian đó người dùng chỉ thấy một cái spinner đứng yên,
        không có cách nào phân biệt "đang chạy" với "đã treo". Streaming không làm model
        nhanh hơn một giây nào, nhưng đưa thời điểm nhìn thấy chữ đầu tiên từ ~40 giây xuống
        còn ~2-3 giây, và cho người dùng thấy hệ thống đang lập luận trên đúng tài liệu nào.

        Chỉ có MỘT đường gọi LLM trong hệ thống (đường này); bản không streaming
        (_goi_llm) chỉ là vòng lặp gom hết các mảnh lại. Cố ý làm vậy để hai chế độ không
        bao giờ trôi ra khỏi nhau về hành vi (lọc <think>, xử lý lỗi, tham số sinh).

        Về tham số `think` - đã đo trực tiếp trên qwen3:4b + ollama client 0.6.2, kết quả
        trái với trực giác nên ghi lại đây để không ai "sửa lại cho gọn":
          - think=True      -> máy chủ tách suy luận sang message["thinking"], content sạch.
          - KHÔNG truyền gì -> y hệt think=True (mặc định của model biết suy luận), content sạch.
          - think=False     -> KHÔNG tắt suy luận, mà tắt việc TÁCH nó ra: toàn bộ chuỗi suy
                               luận (thường bằng tiếng Anh, kiểu "Okay, let me figure out...")
                               đổ thẳng vào content và hiện nguyên si cho người dùng.
        Vì vậy khi không cần suy luận thì BỎ HẲN tham số, tuyệt đối không truyền False.

        Mẹo chèn hậu tố "/no_think" của bản trước cũng đã bỏ: đo thực tế cho thấy nó không
        hề tắt suy luận (model vẫn sinh ~15.000 ký tự suy luận) và không nhanh hơn đáng kể
        (43.7s so với 47.0s - trong khoảng nhiễu), trong khi lại là quy ước riêng của Qwen3,
        đổi sang model khác thì nó thành một chuỗi rác nằm ngay cuối câu hỏi.
        """
        so_token_prompt = _uoc_luong_so_token(he_thong_prompt, prompt_nguoi_dung)
        num_ctx = _tinh_num_ctx(so_token_prompt)
        tham_so = dict(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": he_thong_prompt},
                {"role": "user", "content": prompt_nguoi_dung},
            ],
            options={
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_predict": config.OLLAMA_NUM_PREDICT,
                # BẮT BUỘC khai báo. Không truyền num_ctx thì Ollama cấp 4096 token bất kể
                # model hỗ trợ bao nhiêu, và khi prompt vượt quá thì nó cắt IM LẶNG từ đầu
                # phần user content - tức xoá đúng đoạn trích [1], [2] liên quan nhất, vì
                # _ghep_prompt() xếp đoạn tốt nhất lên trước. Giải thích đầy đủ (kèm ước
                # lượng cho thấy prompt mặc định ~4900 token, một mình đã vượt 4096) nằm ở
                # config.OLLAMA_NUM_CTX.
                "num_ctx": num_ctx,
            },
            stream=True,
        )
        if bat_thinking and self._ho_tro_thinking:
            tham_so["think"] = True

        def _mo_luong(ts):
            """Mở luồng và LẤY LUÔN mảnh đầu tiên.

            Bắt buộc phải lấy mảnh đầu ngay tại đây: với stream=True, client trả về một
            generator nên lỗi phía máy chủ (vd model không hỗ trợ `think`) chỉ nổ ra ở lần
            lặp ĐẦU TIÊN chứ không phải lúc gọi. Không chạm vào generator thì khối try/except
            bên dưới sẽ không bao giờ bắt được lỗi đó.

            Cũng vì lý do đó mà lỗi KHÔNG KẾT NỐI ĐƯỢC phải bắt ngay ở đây: thư viện ollama
            chỉ dịch httpx.ConnectError sang ConnectionError có thông báo tử tế ở đường gọi
            KHÔNG streaming; đường streaming để nó lọt nguyên si ra ngoài (xem
            LoiKhongKetNoiDuocOllama).
            """
            try:
                it = iter(self._ollama_client.chat(**ts))
                return it, next(it)
            except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError) as loi:
                raise LoiKhongKetNoiDuocOllama(_thong_bao_khong_ket_noi_duoc()) from loi

        try:
            luong, manh_dau = _mo_luong(tham_so)
        except ollama.ResponseError:
            if "think" not in tham_so:
                raise
            # Model không có chế độ suy luận (llama3, mistral...) -> máy chủ trả lỗi. Ghi nhớ
            # để các lượt sau khỏi thử lại, rồi gọi lại ngay để người dùng không thấy lỗi.
            logger.info("Model '%s' không hỗ trợ tham số think - bỏ qua.", config.OLLAMA_MODEL)
            self._ho_tro_thinking = False
            tham_so.pop("think")
            luong, manh_dau = _mo_luong(tham_so)
        except StopIteration:
            return

        # Phòng vệ: phần "thinking" của model (nếu có) đã được Ollama tách sang trường riêng
        # `message.thinking`, không nằm trong content - nhưng đã quan sát thực tế một số
        # trường hợp thẻ <think>...</think> vẫn lọt vào content. Lọc theo luồng để người dùng
        # không bao giờ thấy phần suy luận nội bộ thô lẫn trong câu trả lời đang hiện dần.
        loc = _LocSuyLuanTheoLuong()
        cac_manh_suy_luan: List[str] = []
        da_co_cau_tra_loi = False
        for manh in itertools.chain([manh_dau], luong):
            if manh.get("done"):
                self._ghi_nhan_thong_ke_llm(manh, so_token_prompt, num_ctx)
            tin_nhan = manh.get("message") or {}
            suy_luan_tho = tin_nhan.get("thinking") or ""
            if suy_luan_tho:
                cac_manh_suy_luan.append(suy_luan_tho)
                yield {"loai": "suy_luan", "them": suy_luan_tho}
            noi_dung_tho = tin_nhan.get("content") or ""
            if noi_dung_tho:
                phan_tra_loi, phan_suy_luan = loc.them(noi_dung_tho)
                if phan_suy_luan:
                    cac_manh_suy_luan.append(phan_suy_luan)
                    yield {"loai": "suy_luan", "them": phan_suy_luan}
                if phan_tra_loi:
                    da_co_cau_tra_loi = da_co_cau_tra_loi or bool(phan_tra_loi.strip())
                    yield {"loai": "cau_tra_loi", "them": phan_tra_loi}

        phan_tra_loi, phan_suy_luan = loc.ket_thuc()
        if phan_suy_luan:
            cac_manh_suy_luan.append(phan_suy_luan)
            yield {"loai": "suy_luan", "them": phan_suy_luan}
        if phan_tra_loi:
            da_co_cau_tra_loi = da_co_cau_tra_loi or bool(phan_tra_loi.strip())
            yield {"loai": "cau_tra_loi", "them": phan_tra_loi}

        if not da_co_cau_tra_loi and cac_manh_suy_luan:
            # Model nhét TẤT CẢ vào phần suy luận và không viết câu trả lời nào. Thà đưa ra
            # phần suy luận - người dùng còn đọc được điều gì đó và tự thấy nó dở dang - hơn
            # là một bong bóng chat trống trơn trông y hệt như hệ thống bị lỗi. Đây cũng đúng
            # hành vi dự phòng của bản không streaming trước đây.
            #
            # Nguyên nhân đã từng bị chẩn đoán NHẦM là chạm OLLAMA_NUM_PREDICT (12000). Thủ
            # phạm thật là num_ctx: khi cửa sổ mặc định 4096 bị prompt ~4900 token ăn hết,
            # phần còn lại cho thinking + câu trả lời gần bằng 0, model viết được mấy dòng
            # suy luận rồi chạm trần. num_predict=12000 chưa bao giờ với tới được.
            logger.warning(
                "Model không sinh câu trả lời nào ngoài phần suy luận - trả về phần suy luận "
                "để không hiện bong bóng rỗng. Thống kê lượt gọi: %s (xem num_ctx và "
                "done_reason ở đây trước khi nghi OLLAMA_NUM_PREDICT=%d).",
                self.thong_ke_llm,
                config.OLLAMA_NUM_PREDICT,
            )
            yield {"loai": "cau_tra_loi", "them": "".join(cac_manh_suy_luan).strip()}

    def _ghi_nhan_thong_ke_llm(self, manh, so_token_prompt: int, num_ctx: int) -> None:
        """Đọc bộ đếm token thật của Ollama ở mảnh cuối luồng và cảnh báo nếu bị cắt.

        Đây là tuyến CHỨNG MINH cho bug num_ctx: prompt_eval_count là số token máy chủ THẬT
        SỰ đã nạp, đối chiếu được với ước lượng của ta và với num_ctx đã cấp. Trước khi có
        nó, việc prompt bị cắt không để lại một dấu vết nào - không lỗi, không cảnh báo, chỉ
        có câu trả lời tự nhiên ngắn đi và trích dẫn trỏ vào những đoạn kém liên quan. Một
        lớp lỗi mà hệ thống KHÔNG THỂ tự phát hiện thì mọi kết luận rút ra từ nó đều đáng ngờ.

        done_reason:
          "stop"   - model tự viết xong.
          "length" - chạm trần sinh (num_predict, hoặc phần còn lại của num_ctx) -> câu trả
                     lời bị cắt cụt giữa chừng.
        """
        so_token_that = manh.get("prompt_eval_count")
        ly_do_dung = manh.get("done_reason")
        self.thong_ke_llm = {
            "prompt_eval_count": so_token_that,
            "eval_count": manh.get("eval_count"),
            "done_reason": ly_do_dung,
            "num_ctx": num_ctx,
            "uoc_luong_token_prompt": so_token_prompt,
        }
        logger.info(
            "LLM: prompt %s token (ước lượng %d) / num_ctx %d, sinh %s token, dừng vì '%s'.",
            so_token_that, so_token_prompt, num_ctx, manh.get("eval_count"), ly_do_dung,
        )
        if so_token_that and so_token_that >= num_ctx:
            logger.warning(
                "PROMPT BỊ CẮT: máy chủ nạp %d token trong khi cửa sổ chỉ có %d - phần bị "
                "xoá là ĐẦU phần ngữ cảnh, tức đúng các đoạn trích liên quan nhất. Nâng "
                "OLLAMA_NUM_CTX hoặc hạ TOP_K / NGAN_SACH_KY_TU_MOI_DOAN.",
                so_token_that, num_ctx,
            )
        if ly_do_dung == "length":
            logger.warning(
                "Câu trả lời bị cắt cụt (done_reason='length'): đã sinh %s token với "
                "num_predict=%d, num_ctx=%d. Nếu tái diễn, nâng OLLAMA_NUM_CTX trước - phần "
                "còn lại của cửa sổ sau prompt mới là trần thật của câu trả lời.",
                manh.get("eval_count"), config.OLLAMA_NUM_PREDICT, num_ctx,
            )

    def sinh_cau_tra_loi_theo_luong(
        self, cau_hoi: str, cac_chunk: List[Dict], ngu_canh_hoi_thoai: str = ""
    ) -> Iterator[Dict]:
        """Sinh câu trả lời theo luồng: yield {"loai": "suy_luan"|"cau_tra_loi", "them": str}.

        Ngôn ngữ trả lời (VI/EN) tự nhận diện từ chính câu hỏi (yêu cầu đồ án: hỏi tiếng
        Anh -> trả lời tiếng Anh, hỏi tiếng Việt -> trả lời tiếng Việt).

        ngu_canh_hoi_thoai: các câu hỏi trước trong phiên, CHỈ để model giải nghĩa những từ
        trỏ ra ngoài ("dấu hiệu thứ hai" là thứ hai của cái gì). Mặc định rỗng, tức hành vi
        y hệt bản single-turn - đây cũng là lý do evaluation và test không phải sửa gì.
        """
        ngon_ngu = _phat_hien_ngon_ngu(cau_hoi)

        if not cac_chunk:
            # Không có đoạn nào đủ liên quan (index rỗng, hoặc mọi đoạn đều dưới ngưỡng) ->
            # không gọi LLM, trả lời ngay theo đúng ràng buộc "không bịa thông tin".
            yield {"loai": "cau_tra_loi", "them": config.CAU_TU_CHOI[ngon_ngu]}
            return

        la_kiem_chung = la_cau_hoi_kiem_chung(cau_hoi)
        if la_kiem_chung:
            he_thong_prompt = (
                HE_THONG_PROMPT_KIEM_CHUNG_EN if ngon_ngu == "en" else HE_THONG_PROMPT_KIEM_CHUNG_VI
            )
        else:
            he_thong_prompt = HE_THONG_PROMPT_EN if ngon_ngu == "en" else HE_THONG_PROMPT_VI

        prompt_nguoi_dung = _ghep_prompt(
            cau_hoi, cac_chunk, ngon_ngu, la_kiem_chung, ngu_canh_hoi_thoai
        )
        yield from self._goi_llm_theo_luong(
            he_thong_prompt,
            prompt_nguoi_dung,
            bat_thinking=la_kiem_chung and config.BAT_THINKING_KHI_KIEM_CHUNG,
        )

    def sinh_cau_tra_loi(self, cau_hoi: str, cac_chunk: List[Dict]) -> str:
        """Bản gom-hết-rồi-trả-một-lần của sinh_cau_tra_loi_theo_luong().

        Dùng cho evaluation và test - những chỗ không có ai ngồi nhìn màn hình nên không cần
        hiện dần. Cố ý gọi lại đúng generator ở trên thay vì gọi Ollama lần nữa, để chế độ
        đo đạc chạy đúng một đường mã với chế độ người dùng thật.
        """
        cac_manh = [
            sk["them"]
            for sk in self.sinh_cau_tra_loi_theo_luong(cau_hoi, cac_chunk)
            if sk["loai"] == "cau_tra_loi"
        ]
        return "".join(cac_manh).strip()

    def hoi_dap_theo_luong(
        self,
        cau_hoi: str,
        top_k: int = None,
        nguon_cho_phep: Optional[Set[str]] = None,
        lich_su: Optional[List[Dict]] = None,
        doi_chieu: Optional[bool] = None,
    ) -> Iterator[Dict]:
        """Chạy trọn luồng Query và tường thuật lại từng chặng cho tầng giao diện.

        Các loại sự kiện yield ra:
          {"loai": "truy_xuat_xong", "cac_chunk": [...], "giay": float, "truy_van": {...}}
          {"loai": "suy_luan",   "them": str}   - model đang lập luận, chưa phải câu trả lời
          {"loai": "cau_tra_loi","them": str}   - từng mảnh câu trả lời thật
          {"loai": "dang_doi_chieu"}            - bắt đầu đối chiếu chéo các nguồn
          {"loai": "xong",       "ket_qua": {...}}

        Tách "truy_xuat_xong" thành một sự kiện riêng là có chủ đích: nó tới sau ~2 giây,
        tức trước khi LLM kịp viết chữ nào, nên giao diện có thể hiện ngay "đã tìm được N
        đoạn trong tài liệu X" - người dùng biết hệ thống đang làm gì và trên tài liệu nào,
        thay vì nhìn một cái spinner câm suốt cả phút.

        lich_su: lịch sử hội thoại, để hiểu câu hỏi nối tiếp (xem truy_xuat).
        doi_chieu: bật/tắt bước đối chiếu chéo các nguồn cho riêng lượt này; None = theo
        config.BAT_DOI_CHIEU_NGUON. `run_evaluation.py` tắt hẳn nó vì bước này không đổi câu
        trả lời (chỉ thêm cảnh báo), nên để bật sẽ kéo dài một lần đánh giá vốn đã 60-90 phút
        mà không làm thay đổi bất kỳ metric nào đang đo.
        """
        moc_bat_dau = time.perf_counter()
        cac_chunk = self.truy_xuat(
            cau_hoi, top_k=top_k, nguon_cho_phep=nguon_cho_phep, lich_su=lich_su
        )
        giay_truy_xuat = time.perf_counter() - moc_bat_dau
        truy_van = self.truy_van_da_dung or {
            "cau_hoi_goc": cau_hoi, "cau_hoi_chinh": cau_hoi,
            "ngu_canh_llm": "", "la_tiep_noi": False, "da_viet_lai": False,
        }
        yield {
            "loai": "truy_xuat_xong",
            "cac_chunk": cac_chunk,
            "giay": giay_truy_xuat,
            "truy_van": truy_van,
        }

        # LLM nhận ĐÚNG câu hỏi người dùng đã gõ, kèm một khối ngữ cảnh hội thoại riêng khi
        # đây là câu nối tiếp. Không đưa bản đã ghép/viết lại vào đây: bản đó phục vụ truy
        # xuất, còn câu trả lời phải trả lời đúng thứ người dùng hỏi và đọc lên phải tự nhiên.
        #
        # Vì sao vẫn cần khối ngữ cảnh: LLM KHÔNG nhìn thấy lịch sử chat - messages chỉ có
        # [system, user]. Truy xuất đúng đoạn rồi mà model vẫn không biết "dấu hiệu thứ hai"
        # là thứ hai của cái gì thì câu trả lời vẫn hỏng.
        cau_hoi_cho_llm = cau_hoi
        ngu_canh_hoi_thoai = truy_van.get("ngu_canh_llm") or ""

        cac_manh: List[str] = []
        giay_hien_dau_tien = None
        giay_chu_dau_tien = None
        for su_kien in self.sinh_cau_tra_loi_theo_luong(
            cau_hoi_cho_llm, cac_chunk, ngu_canh_hoi_thoai
        ):
            if giay_hien_dau_tien is None:
                giay_hien_dau_tien = time.perf_counter() - moc_bat_dau
            if su_kien["loai"] == "cau_tra_loi":
                if giay_chu_dau_tien is None:
                    giay_chu_dau_tien = time.perf_counter() - moc_bat_dau
                cac_manh.append(su_kien["them"])
            yield su_kien

        cau_tra_loi = "".join(cac_manh).strip()

        # ĐỐI CHIẾU CHÉO CÁC NGUỒN - chạy SAU khi câu trả lời đã hiện xong, có chủ đích:
        # nó không đổi câu trả lời, chỉ thêm một lớp cảnh báo, nên không có lý do gì bắt
        # người dùng chờ nó trước khi được đọc chữ đầu tiên. Tầng lọc tất định bên trong
        # khiến đại đa số lượt hỏi không tốn lượt LLM nào (xem rag/doi_chieu_nguon.py).
        if doi_chieu is None:
            doi_chieu = config.BAT_DOI_CHIEU_NGUON
        cac_mau_thuan: List[Dict] = []
        if doi_chieu and len(cac_chunk) >= 2 and cau_tra_loi != config.CAU_TU_CHOI.get(
            _phat_hien_ngon_ngu(cau_hoi_cho_llm)
        ):
            yield {"loai": "dang_doi_chieu"}
            cac_mau_thuan = tim_mau_thuan(
                cac_chunk,
                embedding_service=self.embedding_service,
                client=self._ollama_client,
            )

        yield {
            "loai": "xong",
            "ket_qua": {
                "cau_tra_loi": cau_tra_loi,
                "cac_chunk_nguon": cac_chunk,
                "la_kiem_chung": la_cau_hoi_kiem_chung(cau_hoi_cho_llm),
                # Bản viết lại của câu hỏi nối tiếp (nếu có). Giao diện PHẢI hiện nó ra chứ
                # không được im lặng dùng: đây là một phỏng đoán của hệ thống về ý người
                # dùng, và trình bày phỏng đoán như thể là sự thật đúng là lỗi mà §5.54 đã
                # phải sửa một lần rồi. Người dùng thấy được thì họ tự sửa câu hỏi khi hệ
                # thống hiểu sai.
                "truy_van": truy_van,
                # Các cặp nguồn nói ngược nhau. Danh sách rỗng là kết quả bình thường.
                "mau_thuan": cac_mau_thuan,
                # Tỉ lệ câu trả lời trùng NGUYÊN VĂN với ngữ cảnh đã truy xuất. Tính ngay ở
                # đây (tất định, mili giây, không gọi model) để giao diện nói được với người
                # đọc mức độ bám nguồn của câu trả lời họ đang xem - thay vì bắt họ tự mở
                # từng đoạn trích ra đối chiếu.
                #
                # ĐỌC CHO ĐÚNG: cao = bằng chứng mạnh rằng KHÔNG bịa; thấp thì KHÔNG kết
                # luận được gì, vì diễn đạt lại bằng lời của mình cũng cho điểm thấp. Vì thế
                # con số này TUYỆT ĐỐI không được dùng để tự động từ chối một câu trả lời -
                # làm vậy sẽ giết đúng những câu trả lời viết tốt nhất.
                "bam_nguon": do_bam_ngu_canh(
                    cau_tra_loi, "\n\n".join(c["noidung"] for c in cac_chunk)
                ) if cac_chunk else 0.0,
                # Số liệu độ trễ đi kèm luôn kết quả: đây là thứ cần đo được để nói về UX
                # bằng con số ("chữ đầu tiên sau 2,8 giây") thay vì bằng cảm nhận.
                "do_tre": {
                    "truy_xuat": giay_truy_xuat,
                    "hien_dau_tien": giay_hien_dau_tien,
                    "chu_dau_tien": giay_chu_dau_tien,
                    "tong": time.perf_counter() - moc_bat_dau,
                },
            },
        }

    def hoi_dap(
        self,
        cau_hoi: str,
        top_k: int = None,
        nguon_cho_phep: Optional[Set[str]] = None,
        lich_su: Optional[List[Dict]] = None,
        doi_chieu: Optional[bool] = None,
    ) -> Dict:
        """Chạy trọn luồng Query: truy xuất -> sinh câu trả lời.

        Trả về dict {"cau_tra_loi", "cac_chunk_nguon", "la_kiem_chung", "truy_van",
        "mau_thuan", "bam_nguon", "do_tre"} để app.py hiển thị mọi thứ từ cùng 1 lần gọi.
        """
        for su_kien in self.hoi_dap_theo_luong(
            cau_hoi,
            top_k=top_k,
            nguon_cho_phep=nguon_cho_phep,
            lich_su=lich_su,
            doi_chieu=doi_chieu,
        ):
            if su_kien["loai"] == "xong":
                return su_kien["ket_qua"]
        raise RuntimeError("hoi_dap_theo_luong kết thúc mà không phát sự kiện 'xong'")
