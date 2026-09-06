"""Ghép luồng Query: retrieval (lai vector + từ khoá) + ghép prompt + gọi LLM qua Ollama."""

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
from rag import do_thoi_gian
from rag.citation import do_bam_ngu_canh
from rag.doi_chieu_nguon import tim_mau_thuan
from rag.embedding import EmbeddingService
from rag.reranker import RerankerService
from rag.tiep_noi_hoi_thoai import chuan_bi_truy_van
from rag.vector_store import VectorStore
from rag.vision_caption import ten_model_khop

logger = logging.getLogger(__name__)


class LoiKhongKetNoiDuocOllama(RuntimeError):
    """Không mở được kết nối tới máy chủ Ollama (chưa chạy, hoặc OLLAMA_HOST trỏ sai chỗ)."""


def _thong_bao_khong_ket_noi_duoc() -> str:
    """Soạn thông báo lỗi kèm ĐÚNG các lệnh cần chạy, đọc host/model từ cấu hình đang dùng."""
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
    """Máy chủ Ollama đã sẵn sàng trả lời chưa? Trả None nếu ổn, chuỗi mô tả lỗi nếu không."""
    try:
        cac_model = ollama.Client(host=config.OLLAMA_HOST).list().models
    except Exception as loi:
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
    """Bóc phần <think>...</think> ra khỏi luồng content ĐANG CHẢY, từng mảnh một."""

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

DetectorFactory.seed = 0

_MAU_DAU_TIENG_VIET = re.compile(
    r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)

_TU_TIENG_VIET_KHONG_DAU = frozenset({
    "gi", "khong", "nao", "duoc", "cua", "nhung", "cac", "mot", "nguoi", "hoac",
    "phai", "viec", "theo", "nhu", "voi", "trong", "cho", "tai", "boi", "vi",
    "dieu", "luat", "phap", "hinh", "su", "nha", "nuoc", "chinh", "tri",
})

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
    """Phát hiện câu hỏi là tiếng Anh hay tiếng Việt."""
    if _MAU_DAU_TIENG_VIET.search(cau_hoi):
        return "vi"

    try:
        xac_suat = {kq.lang: kq.prob for kq in detect_langs(cau_hoi)}
    except LangDetectException:
        xac_suat = {}
    if "en" in xac_suat or "vi" in xac_suat:
        return "en" if xac_suat.get("en", 0.0) > xac_suat.get("vi", 0.0) else "vi"

    cac_tu = set(re.findall(r"[a-z]+", cau_hoi.lower()))
    if cac_tu & _TU_TIENG_VIET_KHONG_DAU:
        return "vi"

    if len(cac_tu) >= 2:
        return "en"

    return "vi"


def la_cau_hoi_kiem_chung(cau_hoi: str) -> bool:
    """Câu hỏi có đang đưa ra một khẳng định cần kiểm chứng hay không."""
    return bool(_MAU_KIEM_CHUNG.search(cau_hoi))


def _noi_lien_mach(truoc: str, sau: str, toi_da: int = 300) -> str:
    """Nối 2 chunk liền kề, bỏ phần bị lặp do overlap khi chia chunk."""
    gioi_han = min(len(truoc), len(sau), toi_da)
    for do_dai in range(gioi_han, 20, -1):
        if truoc.endswith(sau[:do_dai]):
            return truoc + sau[do_dai:]
    return truoc + " " + sau


def _uoc_luong_so_token(*cac_phan: str) -> int:
    """Ước lượng số token của prompt mà KHÔNG cần tokenizer của LLM."""
    return int(sum(len(p) for p in cac_phan) / config.SO_KY_TU_MOI_TOKEN_UOC_LUONG) + 1


_MAU_CAU_HOI_NHIEU_VE = re.compile(
    r"\bso\s*sánh\b|\bphân\s*biệt\b|\bkhác\s*(nhau|biệt)\b|\bliệt\s*kê\b|\bcác\s+bước\b"
    r"|\bưu\s*(và\s*)?nhược\b|\bvì\s*sao\b|\btại\s*sao\b|\bmối\s*(liên\s*hệ|quan\s*hệ)\b"
    r"|\bcompare\b|\bdifference\b|\bversus\b|\bvs\.?\b|\blist\b|\bsteps\b|\bpros\s+and\s+cons\b"
    r"|\bwhy\b|\brelationship\b",
    re.IGNORECASE,
)


def la_cau_hoi_phuc_tap(cau_hoi: str) -> bool:
    """Câu hỏi này có xứng đáng được cấp NGÂN SÁCH ĐẦY ĐỦ không?"""
    if not config.BAT_NGAN_SACH_THICH_UNG:
        return True
    if len(cau_hoi.split()) > config.SO_TU_CAU_HOI_DON_GIAN:
        return True
    if _MAU_CAU_HOI_NHIEU_VE.search(cau_hoi):
        return True
    return la_cau_hoi_kiem_chung(cau_hoi)


def _tinh_num_ctx(so_token_prompt: int, num_predict: Optional[int] = None) -> int:
    """Cửa sổ ngữ cảnh cần cấp cho một prompt dài `so_token_prompt` token."""
    du_phong = config.OLLAMA_DU_PHONG_TOKEN_SINH
    if num_predict is not None:
        du_phong = min(du_phong, max(num_predict, 1))
    can = so_token_prompt + du_phong
    tran = max(config.OLLAMA_NUM_CTX_TOI_DA, config.OLLAMA_NUM_CTX)
    num_ctx = config.OLLAMA_NUM_CTX
    while num_ctx < can and num_ctx < tran:
        num_ctx *= 2
    num_ctx = min(num_ctx, tran)

    if can > num_ctx:
        logger.warning(
            "Prompt ~%d token + %d token dự phòng sinh = %d, vượt trần OLLAMA_NUM_CTX_TOI_DA=%d. "
            "Hạ TOP_K hoặc NGAN_SACH_KY_TU_MOI_DOAN (ĐỪNG hạ num_ctx), hoặc nâng trần nếu máy đủ RAM.",
            so_token_prompt, du_phong, can, tran,
        )
    return num_ctx


def ngan_sach_token_ngu_canh(num_predict: int, so_token_co_dinh: int) -> int:
    """Số token còn lại dành cho các ĐOẠN TRÍCH, sau khi đã trừ mọi phần cố định của prompt."""
    tran = max(config.OLLAMA_NUM_CTX_TOI_DA, config.OLLAMA_NUM_CTX)
    du_phong = min(config.OLLAMA_DU_PHONG_TOKEN_SINH, max(num_predict, 1))
    return tran - du_phong - so_token_co_dinh


def nen_ngu_canh(cac_chunk: List[Dict], ngan_sach_token: int) -> List[Dict]:
    """Ép các đoạn trích vào `ngan_sach_token`, bỏ từ đoạn XẾP HẠNG THẤP NHẤT lên."""
    if not cac_chunk or not config.BAT_NEN_NGU_CANH or ngan_sach_token <= 0:
        return cac_chunk

    _CHI_PHI_NHAN = 30
    so_token = [_uoc_luong_so_token(c["noidung"]) + _CHI_PHI_NHAN for c in cac_chunk]
    if sum(so_token) <= ngan_sach_token:
        return cac_chunk

    giu, da_dung = [], 0
    for chunk, token in zip(cac_chunk, so_token):
        if da_dung + token <= ngan_sach_token:
            giu.append(chunk)
            da_dung += token
            continue
        con_lai = ngan_sach_token - da_dung - _CHI_PHI_NHAN
        if not giu and con_lai > 0:
            so_ky_tu = int(con_lai * config.SO_KY_TU_MOI_TOKEN_UOC_LUONG)
            ban_cat = dict(chunk)
            ban_cat["noidung"] = chunk["noidung"][:so_ky_tu].rstrip() + "\n[...đoạn bị cắt bớt...]"
            giu.append(ban_cat)
        break

    logger.warning(
        "NÉN NGỮ CẢNH: %d đoạn (~%d token) vượt ngân sách %d token của cửa sổ ngữ cảnh - "
        "chỉ gửi %d đoạn xếp hạng cao nhất. Đây là lựa chọn của hệ thống (bỏ phần ÍT liên "
        "quan nhất) thay vì để Ollama cắt im lặng mất đầu ngữ cảnh. Nâng OLLAMA_NUM_CTX_TOI_DA "
        "nếu máy đủ RAM, hoặc hạ NGAN_SACH_KY_TU_MOI_DOAN / TOP_K.",
        len(cac_chunk), sum(so_token), ngan_sach_token, len(giu),
    )
    return giu


def _ghep_prompt(
    cau_hoi: str,
    cac_chunk: List[Dict],
    ngon_ngu: str,
    la_kiem_chung: bool,
    ngu_canh_hoi_thoai: str = "",
) -> str:
    """Ghép Top-K đoạn trích vào prompt, đánh số từng đoạn kèm nguồn để LLM trích dẫn đúng
    theo số thứ tự [1], [2]... khớp với thứ tự hiển thị ở citation.py."""
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
        self.reranker_service = reranker_service
        self.diem_rerank_cao_nhat = None
        self.truy_van_da_dung = None
        self.thong_ke_llm = None
        self.la_cau_hoi_phuc_tap = True
        self._ollama_client = ollama.Client(host=config.OLLAMA_HOST)
        self._ho_tro_thinking = True

    def _ung_vien(
        self, cac_truy_van: List[tuple], so_ung_vien: int, nguon_cho_phep: Optional[Set[str]]
    ) -> Tuple[List[tuple], Set[int]]:
        """Lấy ứng viên từ mọi nhánh tìm kiếm rồi hợp nhất bằng RRF."""
        # FAISS/BM25 không lọc được theo metadata lúc tìm, nên lọc theo nguồn người dùng
        # tick chọn ở UI phải làm tại đây.
        def duoc_phep(vi_tri: int) -> bool:
            return (
                nguon_cho_phep is None
                or self.vector_store.metadata[vi_tri]["nguon"] in nguon_cho_phep
            )

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

        vector_chinh = cac_truy_van[0][1]
        diem_cosine = {vi_tri: diem for vi_tri, diem in dense_chinh if vi_tri in diem_rrf}
        con_thieu = [vi_tri for vi_tri in diem_rrf if vi_tri not in diem_cosine]
        diem_cosine.update(self.vector_store.diem_cosine(con_thieu, vector_chinh))

        thu_tu = sorted(diem_rrf, key=lambda i: (-diem_rrf[i], -diem_cosine[i]))
        return [(vi_tri, diem_cosine[vi_tri]) for vi_tri in thu_tu], vi_tri_cuu_ho

    def _xep_hang_lai(
        self, cau_hoi: str, ung_vien: List[tuple], vi_tri_cuu_ho: Optional[Set[int]] = None,
        so_ung_vien_rerank: Optional[int] = None,
    ) -> List[tuple]:
        """Xếp lại thứ tự ứng viên bằng cross-encoder (xem rag/reranker.py)."""
        if not self.reranker_service or len(ung_vien) < 2:
            return ung_vien

        vi_tri_cuu_ho = vi_tri_cuu_ho or set()
        so_dau = min(len(ung_vien), so_ung_vien_rerank or config.SO_UNG_VIEN_RERANK)
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
        """Dựng 1 đoạn trích liền mạch quanh chunk khớp nhất ("neo")."""
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
            "cac_trang": sorted(
                {self.vector_store.metadata[pham_vi[i]]["trang"] for i in da_chon},
                key=str,
            ),
            "noidung": noi_dung,
            "loai_noi_dung": neo.get("loai_noi_dung", "van_ban"),
            "duong_dan_anh": neo.get("duong_dan_anh", ""),
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
        """Tìm các đoạn trích liên quan nhất tới câu hỏi."""
        top_k = top_k or config.TOP_K
        self.diem_rerank_cao_nhat = None

        self.truy_van_da_dung = chuan_bi_truy_van(cau_hoi, lich_su, client=self._ollama_client)
        cau_hoi_chinh = self.truy_van_da_dung["cau_hoi_chinh"]

        if self.vector_store.so_luong_vector == 0:
            return []

        so_ung_vien = min(
            max(top_k * config.HE_SO_OVER_FETCH, config.SO_UNG_VIEN_TOI_THIEU),
            self.vector_store.so_luong_vector,
        )
        with do_thoi_gian.do("query_ma_hoa_cau_hoi"):
            cac_truy_van = [(
                cau_hoi_chinh,
                self.embedding_service.encode_cau_hoi([cau_hoi_chinh]),
                1.0,
            )]
        for van_ban, trong_so in (
            [(cau_hoi, config.TRONG_SO_TRUY_VAN_GOC)] if cau_hoi_chinh != cau_hoi else []
        ) + [(v, config.TRONG_SO_TRUY_VAN_GOC)
             for v in self.truy_van_da_dung["cac_truy_van_phu"]]:
            cac_truy_van.append(
                (van_ban, self.embedding_service.encode_cau_hoi([van_ban]), trong_so)
            )

        with do_thoi_gian.do("query_ung_vien_dense_bm25"):
            ung_vien, vi_tri_cuu_ho = self._ung_vien(cac_truy_van, so_ung_vien, nguon_cho_phep)
        self.la_cau_hoi_phuc_tap = la_cau_hoi_phuc_tap(cau_hoi_chinh)
        with do_thoi_gian.do("query_rerank"):
            ung_vien = self._xep_hang_lai(
                cau_hoi_chinh, ung_vien, vi_tri_cuu_ho,
                so_ung_vien_rerank=(
                    config.SO_UNG_VIEN_RERANK if self.la_cau_hoi_phuc_tap
                    else config.SO_UNG_VIEN_RERANK_DON_GIAN
                ),
            )

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
        da_dung: set = set()
        so_doan_theo_trang: Dict[tuple, int] = defaultdict(int)
        so_doan_anh = 0
        for vi_tri, diem in ung_vien:
            if len(cac_doan) >= top_k:
                break
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
            if so_doan_theo_trang[khoa_trang] >= tran_moi_trang:
                continue

            la_anh = self.vector_store.metadata[vi_tri].get("loai_noi_dung") == "anh"
            if la_anh and so_doan_anh >= config.SO_DOAN_ANH_TOI_DA:
                continue

            doan = self._dung_doan_trich(vi_tri)
            doan["diem_similarity"] = diem
            cac_doan.append(doan)
            da_dung |= doan["cac_vi_tri"]
            so_doan_theo_trang[khoa_trang] += 1
            so_doan_anh += la_anh

        so_doan_truoc_loc = len(cac_doan)
        cac_doan = [d for d in cac_doan if d["diem_similarity"] >= config.NGUONG_DIEM_TOI_THIEU]

        if cac_doan and config.TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT > 0:
            diem_cao_nhat = max(d["diem_similarity"] for d in cac_doan)
            san_tuong_doi = diem_cao_nhat * config.TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT
            cac_doan = [d for d in cac_doan if d["diem_similarity"] >= san_tuong_doi]

        if config.LOG_PHAN_BO_DIEM:
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
            doan.pop("cac_vi_tri", None)
        return cac_doan

    def _goi_llm_theo_luong(
        self, he_thong_prompt: str, prompt_nguoi_dung: str, bat_thinking: bool,
        num_predict: Optional[int] = None,
    ) -> Iterator[Dict]:
        """Gọi Ollama ở chế độ STREAMING, sinh ra từng mảnh {"loai", "them"} khi model viết."""
        num_predict = num_predict or config.OLLAMA_NUM_PREDICT
        so_token_prompt = _uoc_luong_so_token(he_thong_prompt, prompt_nguoi_dung)
        num_ctx = _tinh_num_ctx(so_token_prompt, num_predict)
        tham_so = dict(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": he_thong_prompt},
                {"role": "user", "content": prompt_nguoi_dung},
            ],
            options={
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
            stream=True,
        )
        if bat_thinking and self._ho_tro_thinking:
            tham_so["think"] = True

        def _mo_luong(ts):
            """Mở luồng và LẤY LUÔN mảnh đầu tiên."""
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
            logger.info("Model '%s' không hỗ trợ tham số think - bỏ qua.", config.OLLAMA_MODEL)
            self._ho_tro_thinking = False
            tham_so.pop("think")
            luong, manh_dau = _mo_luong(tham_so)
        except StopIteration:
            return

        loc = _LocSuyLuanTheoLuong()
        cac_manh_suy_luan: List[str] = []
        da_co_cau_tra_loi = False
        for manh in itertools.chain([manh_dau], luong):
            if manh.get("done"):
                self._ghi_nhan_thong_ke_llm(manh, so_token_prompt, num_ctx, num_predict)
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
            logger.warning(
                "Model không sinh câu trả lời nào ngoài phần suy luận - trả về phần suy luận "
                "để không hiện bong bóng rỗng. Thống kê lượt gọi: %s (xem num_ctx và "
                "done_reason ở đây trước khi nghi num_predict=%d).",
                self.thong_ke_llm,
                num_predict,
            )
            yield {"loai": "cau_tra_loi", "them": "".join(cac_manh_suy_luan).strip()}

    def _ghi_nhan_thong_ke_llm(
        self, manh, so_token_prompt: int, num_ctx: int, num_predict: Optional[int] = None
    ) -> None:
        """Đọc bộ đếm token thật của Ollama ở mảnh cuối luồng và cảnh báo nếu bị cắt."""
        so_token_that = manh.get("prompt_eval_count")
        ly_do_dung = manh.get("done_reason")
        num_predict = num_predict or config.OLLAMA_NUM_PREDICT
        self.thong_ke_llm = {
            "prompt_eval_count": so_token_that,
            "eval_count": manh.get("eval_count"),
            "done_reason": ly_do_dung,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
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
                manh.get("eval_count"), num_predict, num_ctx,
            )

    def sinh_cau_tra_loi_theo_luong(
        self, cau_hoi: str, cac_chunk: List[Dict], ngu_canh_hoi_thoai: str = ""
    ) -> Iterator[Dict]:
        """Sinh câu trả lời theo luồng: yield {"loai": "suy_luan"|"cau_tra_loi", "them": str}."""
        ngon_ngu = _phat_hien_ngon_ngu(cau_hoi)

        if not cac_chunk:
            yield {"loai": "cau_tra_loi", "them": config.CAU_TU_CHOI[ngon_ngu]}
            return

        la_kiem_chung = la_cau_hoi_kiem_chung(cau_hoi)
        if la_kiem_chung:
            he_thong_prompt = (
                HE_THONG_PROMPT_KIEM_CHUNG_EN if ngon_ngu == "en" else HE_THONG_PROMPT_KIEM_CHUNG_VI
            )
        else:
            he_thong_prompt = HE_THONG_PROMPT_EN if ngon_ngu == "en" else HE_THONG_PROMPT_VI

        num_predict = config.OLLAMA_NUM_PREDICT

        cac_chunk_gui = nen_ngu_canh(
            cac_chunk,
            ngan_sach_token_ngu_canh(
                num_predict,
                _uoc_luong_so_token(he_thong_prompt, cau_hoi, ngu_canh_hoi_thoai),
            ),
        )
        prompt_nguoi_dung = _ghep_prompt(
            cau_hoi, cac_chunk_gui, ngon_ngu, la_kiem_chung, ngu_canh_hoi_thoai
        )
        yield from self._goi_llm_theo_luong(
            he_thong_prompt,
            prompt_nguoi_dung,
            bat_thinking=la_kiem_chung and config.BAT_THINKING_KHI_KIEM_CHUNG,
            num_predict=num_predict,
        )

    def sinh_cau_tra_loi(self, cau_hoi: str, cac_chunk: List[Dict]) -> str:
        """Bản gom-hết-rồi-trả-một-lần của sinh_cau_tra_loi_theo_luong()."""
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
        """Chạy trọn luồng Query và tường thuật lại từng chặng cho tầng giao diện."""
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
                "truy_van": truy_van,
                "mau_thuan": cac_mau_thuan,
                "bam_nguon": do_bam_ngu_canh(
                    cau_tra_loi, "\n\n".join(c["noidung"] for c in cac_chunk)
                ) if cac_chunk else 0.0,
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
        """Chạy trọn luồng Query: truy xuất -> sinh câu trả lời."""
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
