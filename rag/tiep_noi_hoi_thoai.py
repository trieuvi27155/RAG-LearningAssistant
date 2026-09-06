"""Đưa NGỮ CẢNH HỘI THOẠI vào truy xuất, để hiểu được câu hỏi nối tiếp."""

import logging
import re
from typing import Dict, List, Optional

import httpx
import ollama

import config

logger = logging.getLogger(__name__)


_DAU_HIEU_HOI_CHI = (
    "thế còn", "vậy còn", "còn cái", "còn phần", "còn loại", "còn trường hợp",
    "thế thì", "vậy thì", "thế nào nữa", "cái đó", "điều đó", "chuyện đó", "việc đó",
    "cái này", "cái kia", "cái thứ", "phần thứ", "loại thứ", "ý thứ", "mục thứ",
    "bước thứ", "dấu hiệu thứ", "yếu tố thứ", "đặc điểm thứ", "cái còn lại",
    "phần còn lại", "những cái", "vừa nói", "vừa nêu", "vừa rồi", "ở trên",
    "bên trên", "nói thêm", "giải thích thêm", "chi tiết hơn", "rõ hơn",
    "ví dụ đi", "cho ví dụ", "tại sao vậy", "sao lại thế", "còn lại thì",
    "bước đầu tiên", "bước cuối", "cuối cùng thì",
    "what about", "how about", "and the second", "and the third", "the second one",
    "the third one", "the other one", "the rest", "the latter", "the former",
    "that one", "those ones", "tell me more", "explain more", "more detail",
    "elaborate", "why is that", "give an example", "what else", "any others",
)

_MO_DAU_NOI_TIEP = re.compile(
    r"^\s*(còn|thế|vậy|nhưng|hoặc|hay là|với lại|ngoài ra|thêm nữa|and|but|or|so|then|also)\b",
    re.IGNORECASE,
)


def la_cau_hoi_tiep_noi(cau_hoi: str, lich_su: Optional[List[Dict]] = None) -> bool:
    """Câu hỏi này có cần ngữ cảnh của lượt trước mới hiểu được không?"""
    if not lich_su or not cau_hoi:
        return False
    if not any(m.get("role") == "user" for m in lich_su):
        return False

    thuong = cau_hoi.lower()
    if any(dau_hieu in thuong for dau_hieu in _DAU_HIEU_HOI_CHI):
        return True
    if _MO_DAU_NOI_TIEP.match(cau_hoi):
        return True
    return False


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
    """Ghép câu hỏi trước vào câu hỏi hiện tại thành MỘT truy vấn mang đủ chủ đề."""
    cac_truoc = cac_cau_hoi_truoc(lich_su)
    if not cac_truoc:
        return cau_hoi
    return " ".join(cac_truoc[::-1] + [cau_hoi])


def ngu_canh_cho_prompt(lich_su: List[Dict], ngon_ngu: str = "vi") -> str:
    """Dựng khối ngữ cảnh hội thoại đưa vào prompt của LLM sinh câu trả lời."""
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
    """Trả về câu hỏi đã viết lại, hoặc CHÍNH câu hỏi gốc nếu không viết lại được."""
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
                "temperature": 0,
                "num_predict": config.NUM_PREDICT_VIET_LAI,
            },
        )
        viet_lai = _lam_sach(phan_hoi["message"]["content"])
    except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError) as loi:
        logger.info("Không viết lại được câu hỏi (chưa kết nối được Ollama): %s", loi)
        return cau_hoi
    except Exception:
        logger.warning("Lỗi khi viết lại câu hỏi nối tiếp - giữ câu gốc.", exc_info=True)
        return cau_hoi

    if not viet_lai:
        return cau_hoi

    if len(viet_lai.split()) > config.SO_TU_TOI_DA_CAU_VIET_LAI:
        logger.info("Bản viết lại quá dài (%d từ) - giữ câu gốc.", len(viet_lai.split()))
        return cau_hoi
    if len(viet_lai.split()) < len(cau_hoi.split()):
        logger.info("Bản viết lại ngắn hơn câu gốc - giữ câu gốc.")
        return cau_hoi
    if "?" not in viet_lai and not re.search(
        r"\b(gì|nào|sao|đâu|bao nhiêu|thế nào|ra sao|what|which|how|why|when|where|who)\b",
        viet_lai, re.IGNORECASE,
    ):
        logger.info("Bản viết lại không còn là câu hỏi - giữ câu gốc.")
        return cau_hoi

    logger.info("Viết lại câu hỏi nối tiếp: %r -> %r", cau_hoi, viet_lai)
    return viet_lai


def chuan_bi_truy_van(
    cau_hoi: str, lich_su: Optional[List[Dict]] = None, client: Optional[ollama.Client] = None
) -> Dict:
    """Quyết định lượt hỏi này truy xuất bằng (những) truy vấn nào."""
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

    if config.BAT_TRUY_VAN_NGU_CANH:
        ghep = truy_van_ngu_canh(cau_hoi, lich_su)
        if ghep != cau_hoi:
            ket_qua["cau_hoi_chinh"] = ghep
            ket_qua["ngu_canh_llm"] = ngu_canh_cho_prompt(lich_su)

    if config.BAT_VIET_LAI_CAU_HOI:
        viet_lai = viet_lai_cau_hoi(cau_hoi, lich_su, client=client)
        if " ".join(viet_lai.split()).lower() != " ".join(cau_hoi.split()).lower():
            if ket_qua["cau_hoi_chinh"] != cau_hoi:
                ket_qua["cac_truy_van_phu"].append(ket_qua["cau_hoi_chinh"])
            ket_qua["cau_hoi_chinh"] = viet_lai
            ket_qua["da_viet_lai"] = True

    return ket_qua
