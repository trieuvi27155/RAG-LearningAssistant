"""Đưa NGỮ CẢNH HỘI THOẠI vào truy xuất, để hiểu được câu hỏi nối tiếp.

Vấn đề mà module này giải quyết
--------------------------------
`RagPipeline.truy_xuat()` chỉ encode đúng chuỗi câu hỏi hiện tại. Với một câu nối tiếp:

    Người dùng: "Vi phạm pháp luật gồm những dấu hiệu nào?"
    Hệ thống:   "... bốn dấu hiệu: hành vi trái pháp luật, có lỗi, ..."
    Người dùng: "Thế còn dấu hiệu thứ hai thì sao?"        <-- câu này

câu hỏi cuối KHÔNG chứa chủ đề nào cả. Vector của nó không trỏ tới vùng nào trong không gian
ngữ nghĩa của tài liệu, BM25 không có từ khoá nào để bám, và cross-encoder chấm mọi đoạn về
gần 0 - mà điểm rerank còn là cơ chế từ chối (§5.29), nên câu hợp lệ này còn có nguy cơ bị
từ chối oan. Lịch sử chat có hiện trên màn hình, nhưng nó nằm ở tầng giao diện và chưa bao
giờ đi vào truy xuất.

Đây là giới hạn của kiến trúc single-turn và nó xảy ra NGAY TRONG MỘT PHIÊN - khác hẳn việc
"không lưu lịch sử qua nhiều phiên" (§5.9), vốn là một quyết định phạm vi chứ không phải
khiếm khuyết. Hai chuyện hay bị gộp làm một khi đọc README.

Hai cách giải, và VÌ SAO CHỌN CÁCH TẤT ĐỊNH
-------------------------------------------
Cách kinh điển là *query rewriting*: hỏi LLM viết lại câu nối tiếp thành câu độc lập. Đã
cài đặt, đã đo, và **kết quả âm tính** (xem §5.58 - số đo đầy đủ ở đó):

    qwen3:4b     0/7 ca. Model luôn sinh một chuỗi suy luận dài trước khi trả lời (§5.23);
                 với num_predict=200 nó tiêu hết ngân sách trong lúc "nghĩ" và trả về
                 content RỖNG. Nâng lên 1500 vẫn rỗng (thinking đã 5891 ký tự). Muốn đủ
                 chỗ thì phải ~3000 token, tức ~30 giây CỘNG THÊM cho mỗi câu nối tiếp -
                 đắt hơn hẳn vấn đề nó giải quyết.
    qwen2.5vl:3b Không có chế độ suy luận nên nhanh, nhưng sai: một ca chép lại y nguyên
                 câu gốc, một ca trả về đúng câu hỏi CŨ (mất hẳn phần "thứ hai").

Cách thứ hai - *contextualization* - làm đúng việc cần làm mà không cần model nào: **ghép
câu hỏi trước vào câu hỏi hiện tại** thành một truy vấn phụ, rồi để RRF hợp nhất hai nhánh.
"Vi phạm pháp luật gồm những dấu hiệu nào? Thế còn dấu hiệu thứ hai thì sao?" mang đủ từ
khoá chủ đề để vector trỏ đúng vùng.

Ba lý do khiến đây là lựa chọn ĐÚNG chứ không phải lựa chọn tạm:

  1. TẤT ĐỊNH. §5.46 đã đo và kết luận: chỉ Precision@K/Recall@K/MRR là tất định, và đó là
     những con số duy nhất so sánh được giữa hai lần chạy. Một bước tất định thì đo được
     bằng chính chúng; một bước gọi LLM thì thêm một nguồn dao động nữa vào đúng chỗ đứng
     chắn trước toàn bộ truy xuất.
  2. GẦN NHƯ MIỄN PHÍ. Một lần encode (~30ms). Không thêm lượt gọi LLM nào, nên không đụng
     tới độ trễ mà §5.42 đã tốn công kéo xuống.
  3. AN TOÀN THEO CẤU TRÚC. Câu hỏi gốc vẫn là một nhánh riêng trong RRF, nên trường hợp
     xấu nhất của việc ghép sai chỉ là "thứ hạng nhiễu đi một chút", không phải "mất kết
     quả đúng" - cùng nguyên tắc đã dùng ở §5.45(a).

Đường viết lại bằng LLM được GIỮ LẠI trong code nhưng mặc định TẮT
(config.BAT_VIET_LAI_CAU_HOI), đúng theo tiền lệ §5.30 với BM25: một kết quả âm tính đo
trên corpus/model này không có nghĩa nó âm tính với model khác. Đặt `OLLAMA_MODEL` sang một
model không sinh suy luận và đủ mạnh thì bật lại rồi chạy
`python evaluation/kiem_dinh_viet_lai.py` để tự kiểm.

Hướng ưu tiên của tầng nhận diện ĐẢO NGƯỢC so với `la_cau_hoi_kiem_chung()` (§5.22)
-----------------------------------------------------------------------------------
Ở đó, nhận nhầm khiến câu hỏi thường nhận về bố cục phán quyết cứng nhắc, nên ưu tiên độ
chính xác (thà bỏ sót). Ở đây ngược lại:

  - BỎ SÓT một câu nối tiếp -> truy xuất trượt hoàn toàn. Hỏng thấy ngay.
  - NHẬN NHẦM một câu vốn đã độc lập -> truy vấn phụ chỉ là câu hỏi trước ghép thêm vào,
    và nhánh câu gốc vẫn nguyên trong RRF. Giá phải trả gần bằng 0.

Nên tầng nhận diện nghiêng về phía độ phủ. Ghi lại rõ vì hai hàm nhận diện nằm cạnh nhau
trong cùng một hệ thống mà lại chọn ngược hướng nhau - dễ bị "sửa lại cho nhất quán".
"""

import logging
import re
from typing import Dict, List, Optional

import httpx
import ollama

import config

logger = logging.getLogger(__name__)


# ============================================================
# TẦNG 1: nhận diện câu hỏi nối tiếp (tất định, không gọi model)
# ============================================================

# Dấu hiệu HỒI CHỈ: những từ chỉ trỏ ra ngoài bản thân câu hỏi. Câu chứa chúng chỉ hiểu được
# khi biết lượt trước nói gì.
#
# Danh sách cố ý KHÔNG chứa các từ vừa là hồi chỉ vừa là từ thường gặp trong câu hỏi độc lập.
# Ví dụ đã loại: "này" (có trong "tài liệu này nói gì" - một câu hỏi tự đứng được), "the"
# (mạo từ tiếng Anh, có ở gần như mọi câu). Giữ chúng lại sẽ khiến gần như câu nào cũng bị
# coi là nối tiếp, tức truy vấn phụ được dựng cho mọi lượt hỏi - vô hại nhưng vô nghĩa.
_DAU_HIEU_HOI_CHI = (
    # Tiếng Việt
    "thế còn", "vậy còn", "còn cái", "còn phần", "còn loại", "còn trường hợp",
    "thế thì", "vậy thì", "thế nào nữa", "cái đó", "điều đó", "chuyện đó", "việc đó",
    "cái này", "cái kia", "cái thứ", "phần thứ", "loại thứ", "ý thứ", "mục thứ",
    "bước thứ", "dấu hiệu thứ", "yếu tố thứ", "đặc điểm thứ", "cái còn lại",
    "phần còn lại", "những cái", "vừa nói", "vừa nêu", "vừa rồi", "ở trên",
    "bên trên", "nói thêm", "giải thích thêm", "chi tiết hơn", "rõ hơn",
    "ví dụ đi", "cho ví dụ", "tại sao vậy", "sao lại thế", "còn lại thì",
    "bước đầu tiên", "bước cuối", "cuối cùng thì",
    # Tiếng Anh
    "what about", "how about", "and the second", "and the third", "the second one",
    "the third one", "the other one", "the rest", "the latter", "the former",
    "that one", "those ones", "tell me more", "explain more", "more detail",
    "elaborate", "why is that", "give an example", "what else", "any others",
)

# Câu MỞ ĐẦU bằng liên từ nối: bản thân việc mở đầu bằng "còn", "thế", "vậy", "and", "but"...
# đã là dấu hiệu câu này treo vào một câu trước đó.
_MO_DAU_NOI_TIEP = re.compile(
    r"^\s*(còn|thế|vậy|nhưng|hoặc|hay là|với lại|ngoài ra|thêm nữa|and|but|or|so|then|also)\b",
    re.IGNORECASE,
)

# MỘT TÍN HIỆU ĐÃ BỊ BỎ, ghi lại để không ai thêm lại nó.
#
# Bản đầu có thêm một luật dự phòng: "câu ngắn (<= 8 từ), không có danh từ riêng, không có
# số thì coi là nối tiếp" - ý tưởng là câu ngắn thiếu chủ đề thì chắc đang trỏ ra ngoài. Bộ
# ca có nhãn ở tests/test_tiep_noi_hoi_thoai.py bác bỏ ngay:
#
#   "Nhà nước có những đặc điểm gì?"           6 từ  -> bị nhận nhầm
#   "Trình bày khái niệm quy phạm pháp luật."  6 từ  -> bị nhận nhầm
#   "How is the late return fee calculated?"   7 từ  -> bị nhận nhầm
#
# Tiếng Việt viết rời từng âm tiết nên một câu hỏi HOÀN CHỈNH thường xuyên chỉ có 6-8 "từ";
# độ dài đơn giản không mang thông tin về việc câu có tự đứng được hay không. Và đo lại thì
# luật đó KHÔNG bắt thêm được ca nối tiếp nào mà danh sách dấu hiệu trên chưa bắt - nó chỉ
# đóng góp báo động giả.
#
# Đây đúng là bài học §5.50: ngưỡng đặt trên một đại lượng GẦN ĐÚNG (độ dài câu) thay vì trên
# thứ thật sự cần biết (câu có từ trỏ ra ngoài không). Lời giải là đổi tín hiệu, không phải
# chỉnh ngưỡng - nên luật độ dài bị bỏ hẳn, chỉ giữ dấu hiệu trực tiếp.


def la_cau_hoi_tiep_noi(cau_hoi: str, lich_su: Optional[List[Dict]] = None) -> bool:
    """Câu hỏi này có cần ngữ cảnh của lượt trước mới hiểu được không?

    lich_su: danh sách {"role": "user"|"assistant", "content": str} theo thứ tự thời gian.
    Không có lịch sử thì không có gì để nối tiếp - trả False ngay, không xét gì thêm.
    """
    if not lich_su or not cau_hoi:
        return False
    # Phải có ít nhất một lượt hỏi TRƯỚC ĐÓ. Câu đầu tiên của phiên không thể là câu nối tiếp
    # dù nó trông giống đến đâu ("Thế còn cái kia?" hỏi ngay khi vừa mở app là câu vô nghĩa,
    # và ghép ngữ cảnh vào cũng không cứu được gì).
    if not any(m.get("role") == "user" for m in lich_su):
        return False

    thuong = cau_hoi.lower()
    if any(dau_hieu in thuong for dau_hieu in _DAU_HIEU_HOI_CHI):
        return True
    if _MO_DAU_NOI_TIEP.match(cau_hoi):
        return True
    return False


# ============================================================
# TẦNG 2A: ghép ngữ cảnh (TẤT ĐỊNH - đường mặc định)
# ============================================================

def cac_cau_hoi_truoc(lich_su: List[Dict], so_luot: Optional[int] = None) -> List[str]:
    """Lấy các câu hỏi NGƯỜI DÙNG đã đặt gần đây nhất, mới nhất đứng trước."""
    so_luot = config.SO_LUOT_NGU_CANH if so_luot is None else so_luot
    cac_cau = [
        (m.get("content") or "").strip()
        for m in lich_su
        if m.get("role") == "user" and (m.get("content") or "").strip()
    ]
    return cac_cau[::-1][:so_luot]


def truy_van_ngu_canh(cau_hoi: str, lich_su: List[Dict]) -> str:
    """Ghép câu hỏi trước vào câu hỏi hiện tại thành MỘT truy vấn mang đủ chủ đề.

    Chỉ lấy các câu HỎI, không lấy câu trả lời. Câu trả lời của lượt trước dài hơn câu hỏi
    hàng chục lần, nên ghép nó vào sẽ khiến vector truy vấn bị chính nội dung câu trả lời cũ
    lấn át - và câu trả lời cũ thì đã nằm sẵn trong tài liệu, tức nhánh này sẽ chỉ kéo về
    đúng những đoạn vừa dùng ở lượt trước, không tìm được phần MỚI mà người dùng đang hỏi.

    Câu hỏi HIỆN TẠI đặt SAU cùng, có chủ đích: nó là thứ cần được nhấn, còn phần ghép thêm
    chỉ đóng vai trò nêu chủ đề.
    """
    cac_truoc = cac_cau_hoi_truoc(lich_su)
    if not cac_truoc:
        return cau_hoi
    return " ".join(cac_truoc[::-1] + [cau_hoi])


def ngu_canh_cho_prompt(lich_su: List[Dict], ngon_ngu: str = "vi") -> str:
    """Dựng khối ngữ cảnh hội thoại đưa vào prompt của LLM sinh câu trả lời.

    Cần thiết vì LLM KHÔNG nhìn thấy lịch sử chat - messages chỉ có [system, user]. Truy xuất
    đúng đoạn rồi mà model vẫn không biết "dấu hiệu thứ hai" là thứ hai của cái gì thì câu
    trả lời vẫn hỏng.

    Đưa vào dưới một nhãn RIÊNG và nói rõ nó KHÔNG phải nguồn thông tin: ngữ cảnh hội thoại
    chứa chính câu trả lời trước của model, mà câu đó không phải tài liệu. Cho model coi nó
    ngang hàng với đoạn trích là mở đúng cánh cửa mà cả hệ thống này tồn tại để đóng - model
    trích dẫn lại lời của chính nó như thể đó là tài liệu.
    """
    cac_truoc = cac_cau_hoi_truoc(lich_su)
    if not cac_truoc:
        return ""
    danh_sach = "\n".join(f"- {c}" for c in cac_truoc[::-1])
    if ngon_ngu == "en":
        return (
            "EARLIER QUESTIONS IN THIS CONVERSATION (use ONLY to resolve what the current "
            "question refers to — this is NOT a source of facts, never cite it):\n"
            f"{danh_sach}"
        )
    return (
        "CÁC CÂU HỎI TRƯỚC TRONG HỘI THOẠI (chỉ dùng để hiểu câu hỏi hiện tại đang nhắc tới "
        "cái gì — ĐÂY KHÔNG PHẢI nguồn thông tin, tuyệt đối không trích dẫn):\n"
        f"{danh_sach}"
    )


# ============================================================
# TẦNG 2B: viết lại bằng LLM (MẶC ĐỊNH TẮT - xem §5.58)
# ============================================================

PROMPT_VIET_LAI = """Nhiệm vụ: viết lại CÂU HỎI MỚI thành một câu hỏi ĐỘC LẬP, tự hiểu được mà không cần đọc lịch sử hội thoại.

QUY TẮC BẮT BUỘC:
1. Thay mọi từ trỏ ra ngoài ("cái đó", "phần thứ hai", "nó", "the second one"...) bằng đúng tên/nội dung mà nó ám chỉ, lấy từ lịch sử bên dưới.
2. GIỮ NGUYÊN ngôn ngữ của CÂU HỎI MỚI. Câu hỏi tiếng Việt phải viết lại bằng tiếng Việt, câu hỏi tiếng Anh phải viết lại bằng tiếng Anh.
3. KHÔNG trả lời câu hỏi. KHÔNG thêm thông tin không có trong lịch sử. Chỉ viết lại.
4. Nếu CÂU HỎI MỚI vốn đã độc lập rồi thì chép lại y nguyên, không sửa gì.
5. Chỉ in ra đúng MỘT câu hỏi, không giải thích, không thêm dấu ngoặc kép.

--- LỊCH SỬ HỘI THOẠI ---
{lich_su}

--- CÂU HỎI MỚI ---
{cau_hoi}

--- CÂU HỎI ĐỘC LẬP (chỉ in đúng một dòng) ---"""


def _dung_ngu_canh(lich_su: List[Dict]) -> str:
    """Dựng phần lịch sử đưa vào prompt viết lại: vài lượt gần nhất, câu trả lời đã cắt ngắn."""
    gan_nhat = lich_su[-(config.SO_LUOT_NGU_CANH * 2):]
    cac_dong = []
    for tin_nhan in gan_nhat:
        noi_dung = (tin_nhan.get("content") or "").strip()
        if not noi_dung:
            continue
        if tin_nhan.get("role") == "user":
            cac_dong.append(f"Người dùng hỏi: {noi_dung}")
        else:
            # Cắt ngắn câu trả lời: việc viết lại chỉ cần biết lượt trước NÓI VỀ CHỦ ĐỀ GÌ.
            # Đưa toàn văn vào vừa tốn token vừa khiến model lôi chi tiết vụn của câu trả lời
            # cũ vào câu hỏi mới.
            cat = noi_dung[: config.DO_DAI_TRA_LOI_TRONG_NGU_CANH]
            if len(noi_dung) > len(cat):
                cat += "..."
            cac_dong.append(f"Hệ thống trả lời: {cat}")
    return "\n".join(cac_dong)


def _lam_sach(ket_qua_tho: str) -> str:
    """Bóc những thứ model hay thêm vào dù prompt đã cấm: nhãn, ngoặc kép, gạch đầu dòng."""
    dong = ""
    for d in (ket_qua_tho or "").splitlines():
        d = d.strip()
        # Bỏ dòng rỗng và dòng chỉ toàn dấu gạch (model hay chép lại dấu phân cách của prompt).
        if d and not set(d) <= set("-=_ "):
            dong = d
            break
    dong = re.sub(r"^(câu hỏi độc lập|câu hỏi|standalone question|question)\s*[:.\-]\s*", "",
                  dong, flags=re.IGNORECASE)
    dong = dong.strip().strip('"').strip("'").lstrip("-*•").strip()
    return dong


def viet_lai_cau_hoi(
    cau_hoi: str, lich_su: List[Dict], client: Optional[ollama.Client] = None
) -> str:
    """Trả về câu hỏi đã viết lại, hoặc CHÍNH câu hỏi gốc nếu không viết lại được.

    Không bao giờ ném lỗi ra ngoài: một tính năng phụ trợ không được phép làm hỏng lượt hỏi.
    Mọi đường thất bại đều lùi về câu gốc, tức về đúng hành vi khi tắt tính năng này.
    """
    ngu_canh = _dung_ngu_canh(lich_su)
    if not ngu_canh:
        return cau_hoi

    client = client or ollama.Client(host=config.OLLAMA_HOST)
    try:
        phan_hoi = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": PROMPT_VIET_LAI.format(lich_su=ngu_canh, cau_hoi=cau_hoi),
            }],
            options={
                # temperature=0: đây là việc BIẾN ĐỔI văn bản theo quy tắc, không phải việc
                # sáng tác - không có lý do gì để hai lần chạy cho hai kết quả khác nhau.
                "temperature": 0,
                # Trần này là chốt chặn chi phí, và cũng chính là chỗ đường viết lại gãy với
                # model có chế độ suy luận: qwen3:4b tiêu hết ngân sách trong lúc "nghĩ" rồi
                # trả về content rỗng (§5.58). Nâng trần lên cho đủ chỗ thì mỗi câu nối tiếp
                # tốn thêm ~30 giây - đắt hơn hẳn vấn đề nó giải quyết.
                "num_predict": config.NUM_PREDICT_VIET_LAI,
            },
            # KHÔNG truyền think - xem §5.23: think=False không tắt suy luận mà chỉ tắt việc
            # TÁCH nó ra, khiến toàn bộ chuỗi suy luận đổ thẳng vào content.
        )
        viet_lai = _lam_sach(phan_hoi["message"]["content"])
    except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError) as loi:
        # Ollama chưa chạy. Người dùng sẽ gặp thông báo tử tế ở bước sinh câu trả lời (nơi có
        # hướng dẫn xử lý đầy đủ); ở đây chỉ cần lùi về câu gốc, không nhân đôi thông báo lỗi.
        logger.info("Không viết lại được câu hỏi (chưa kết nối được Ollama): %s", loi)
        return cau_hoi
    except Exception:
        logger.warning("Lỗi khi viết lại câu hỏi nối tiếp - giữ câu gốc.", exc_info=True)
        return cau_hoi

    if not viet_lai:
        return cau_hoi

    # Ba chốt chặn, đều nhằm một việc: KHÔNG BAO GIỜ để bản viết lại tệ hơn câu gốc.
    #
    # (a) Quá dài -> model đã tự trả lời hoặc nhồi cả đoạn ngữ cảnh vào câu hỏi.
    if len(viet_lai.split()) > config.SO_TU_TOI_DA_CAU_VIET_LAI:
        logger.info("Bản viết lại quá dài (%d từ) - giữ câu gốc.", len(viet_lai.split()))
        return cau_hoi
    # (b) Ngắn hơn hẳn câu gốc -> viết lại phải LÀM RÕ thêm, không được làm mất thông tin.
    #     Ca đã gặp: model rút gọn "Thế còn dấu hiệu thứ hai?" thành "Dấu hiệu?".
    if len(viet_lai.split()) < len(cau_hoi.split()):
        logger.info("Bản viết lại ngắn hơn câu gốc - giữ câu gốc.")
        return cau_hoi
    # (c) Model trả lời thay vì viết lại. Dấu hiệu rẻ và chắc: câu hỏi luôn còn dấu chấm hỏi
    #     hoặc một từ để hỏi. Không có gì trong hai thứ đó thì đây không còn là câu hỏi nữa.
    if "?" not in viet_lai and not re.search(
        r"\b(gì|nào|sao|đâu|bao nhiêu|thế nào|ra sao|what|which|how|why|when|where|who)\b",
        viet_lai, re.IGNORECASE,
    ):
        logger.info("Bản viết lại không còn là câu hỏi - giữ câu gốc.")
        return cau_hoi

    logger.info("Viết lại câu hỏi nối tiếp: %r -> %r", cau_hoi, viet_lai)
    return viet_lai


# ============================================================
# ĐIỂM VÀO
# ============================================================

def chuan_bi_truy_van(
    cau_hoi: str, lich_su: Optional[List[Dict]] = None, client: Optional[ollama.Client] = None
) -> Dict:
    """Quyết định lượt hỏi này truy xuất bằng (những) truy vấn nào.

    Trả về:
      cau_hoi_goc      - đúng chuỗi người dùng đã gõ; luôn là một nhánh riêng trong RRF.
      cau_hoi_chinh    - truy vấn CHÍNH: dùng để rerank và để đo cosine. Là bản đã ghép ngữ
                         cảnh (hoặc đã viết lại) khi đây là câu nối tiếp, vì cross-encoder
                         cần một câu hỏi đủ nghĩa - đưa "Thế còn cái thứ hai?" vào đó thì
                         mọi đoạn đều bị chấm gần 0 và câu hợp lệ bị TỪ CHỐI OAN (§5.29).
      cac_truy_van_phu - các nhánh truy vấn thêm, ngoài cau_hoi_goc và cau_hoi_chinh.
      ngu_canh_llm     - khối ngữ cảnh hội thoại để ghép vào prompt sinh câu trả lời.
      la_tiep_noi      - tầng nhận diện đã kết luận đây là câu nối tiếp.
      da_viet_lai      - đường LLM (mặc định tắt) đã đổi được câu hỏi.

    Câu hỏi tự đứng được - đại đa số - đi qua đây gần như không tốn gì: một phép so chuỗi.
    """
    ket_qua = {
        "cau_hoi_goc": cau_hoi,
        "cau_hoi_chinh": cau_hoi,
        "cac_truy_van_phu": [],
        "ngu_canh_llm": "",
        "la_tiep_noi": False,
        "da_viet_lai": False,
    }
    if not lich_su or not la_cau_hoi_tiep_noi(cau_hoi, lich_su):
        return ket_qua

    ket_qua["la_tiep_noi"] = True

    # Đường TẤT ĐỊNH, mặc định: ghép câu hỏi trước vào làm truy vấn chính.
    if config.BAT_TRUY_VAN_NGU_CANH:
        ghep = truy_van_ngu_canh(cau_hoi, lich_su)
        if ghep != cau_hoi:
            ket_qua["cau_hoi_chinh"] = ghep
            ket_qua["ngu_canh_llm"] = ngu_canh_cho_prompt(lich_su)

    # Đường LLM, mặc định TẮT (§5.58). Khi bật, bản viết lại THAY cho bản ghép làm truy vấn
    # chính - nó là câu hỏi thật sự độc lập, còn bản ghép chỉ là hai câu đặt cạnh nhau.
    if config.BAT_VIET_LAI_CAU_HOI:
        viet_lai = viet_lai_cau_hoi(cau_hoi, lich_su, client=client)
        # So sánh sau khi chuẩn hoá khoảng trắng: model hay trả về đúng câu cũ nhưng khác
        # cách đặt dấu cách. Báo "đã viết lại" cho một thay đổi vô hình chỉ làm nhiễu giao diện.
        if " ".join(viet_lai.split()).lower() != " ".join(cau_hoi.split()).lower():
            if ket_qua["cau_hoi_chinh"] != cau_hoi:
                ket_qua["cac_truy_van_phu"].append(ket_qua["cau_hoi_chinh"])
            ket_qua["cau_hoi_chinh"] = viet_lai
            ket_qua["da_viet_lai"] = True

    return ket_qua
