"""Phát hiện MÂU THUẪN giữa các đoạn trích được truy xuất từ những tài liệu khác nhau."""

import json
import logging
import re
from itertools import combinations
from typing import Dict, List, Optional

import httpx
import numpy as np
import ollama

import config

logger = logging.getLogger(__name__)


_SO = re.compile(r"\b\d[\d.,]*\b")

_SO_BANG_CHU = {
    "hai": "2", "ba": "3", "bốn": "4", "năm": "5", "sáu": "6",
    "bảy": "7", "tám": "8", "chín": "9", "mười": "10",
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_TU = re.compile(r"[^\W\d_]+", re.UNICODE)

_PHU_DINH = re.compile(
    r"\b(không|chẳng|chưa|đừng|khỏi phải|không phải|không được|không có"
    r"|not|no|never|cannot|can't|don't|doesn't|isn't|aren't|without)\b",
    re.IGNORECASE,
)


def _tap_so(van_ban: str) -> set:
    """Tập các số xuất hiện trong đoạn, đã chuẩn hoá về dạng so sánh được."""
    ket_qua = set()
    for tho in _SO.findall(van_ban):
        gon = tho.replace(".", "").replace(",", "").lstrip("0")
        if gon:
            ket_qua.add(gon)
    for tu in _TU.findall(van_ban.lower()):
        if tu in _SO_BANG_CHU:
            ket_qua.add(_SO_BANG_CHU[tu])
    return ket_qua


def co_dau_hieu_bat_dong(doan_a: str, doan_b: str) -> bool:
    """Hai đoạn có dấu hiệu BỀ MẶT của việc nói khác nhau không?"""
    so_a, so_b = _tap_so(doan_a), _tap_so(doan_b)
    if so_a and so_b and so_a != so_b:
        return True
    if bool(_PHU_DINH.search(doan_a)) != bool(_PHU_DINH.search(doan_b)):
        return True
    return False


def cac_cap_dang_ngo(cac_doan: List[Dict], vector_doan: Optional[np.ndarray]) -> List[tuple]:
    """Chọn ra các cặp (i, j) đáng đem đi chấm, sắp theo cosine giảm dần."""
    ung_vien = []
    for i, j in combinations(range(len(cac_doan)), 2):
        a, b = cac_doan[i], cac_doan[j]
        if a.get("nguon") == b.get("nguon"):
            continue

        cosine = 1.0
        if vector_doan is not None:
            cosine = float(np.dot(vector_doan[i], vector_doan[j]))
            if cosine < config.NGUONG_COSINE_DOI_CHIEU:
                continue

        if not co_dau_hieu_bat_dong(a.get("noidung", ""), b.get("noidung", "")):
            continue
        ung_vien.append((cosine, i, j))

    ung_vien.sort(key=lambda x: -x[0])
    return [(i, j) for _, i, j in ung_vien[: config.SO_CAP_DOI_CHIEU_TOI_DA]]


PROMPT_DOI_CHIEU = """Bạn đang kiểm tra xem HAI đoạn trích từ HAI tài liệu khác nhau có MÂU THUẪN với nhau hay không.

MÂU THUẪN nghĩa là: hai đoạn cùng nói về MỘT chuyện nhưng đưa ra thông tin KHÔNG THỂ CÙNG ĐÚNG (khác số liệu, khác định nghĩa, một bên khẳng định một bên phủ định).

KHÔNG PHẢI mâu thuẫn:
- Hai đoạn nói về hai chuyện khác nhau.
- Một đoạn chi tiết hơn đoạn kia, nhưng không nói ngược.
- Cùng một ý diễn đạt bằng từ ngữ khác nhau.
- Hai đoạn bổ sung cho nhau (đoạn này nêu điều kiện A, đoạn kia nêu điều kiện B).

--- ĐOẠN A (nguồn: {nguon_a}, trang {trang_a}) ---
{noi_dung_a}

--- ĐOẠN B (nguồn: {nguon_b}, trang {trang_b}) ---
{noi_dung_b}

Trả lời bằng JSON, điền các trường THEO ĐÚNG THỨ TỰ SAU:

1. "phan_tich": trước hết trả lời LẦN LƯỢT hai câu hỏi.
   (a) Hai đoạn có đang nói về CÙNG MỘT đại lượng / thuộc tính / khái niệm cụ thể không?
       Hãy gọi tên đúng đại lượng đó ra. Nếu đoạn A nói về thứ X còn đoạn B nói về thứ Y
       KHÁC X (ví dụ: một bên nói về điểm trung bình, bên kia nói về điểm rèn luyện; một
       bên nêu điều kiện thứ nhất, bên kia nêu điều kiện thứ hai) thì dừng lại ở đây:
       KHÔNG mâu thuẫn.
   (b) Chỉ khi (a) là CÙNG một đại lượng: hai giá trị được nêu có thể CÙNG ĐÚNG không?
       Phép thử: có thể viết cả hai câu vào cùng một tài liệu mà không sai chỗ nào không?
       Nếu được thì KHÔNG mâu thuẫn.
2. "muc_do": 0 đến 1. Nếu (a) kết luận hai đoạn nói về hai đại lượng khác nhau thì muc_do
   PHẢI bằng 0. Không chắc thì chấm THẤP.
3. "co_mau_thuan": phải là kết luận RÚT RA TỪ phần phân tích bạn vừa viết, không được mâu
   thuẫn với nó."""

_SCHEMA_MAU_THUAN = {
    "type": "object",
    "properties": {
        "phan_tich": {"type": "string"},
        "muc_do": {"type": "number", "minimum": 0, "maximum": 1},
        "co_mau_thuan": {"type": "boolean"},
    },
    "required": ["phan_tich", "muc_do", "co_mau_thuan"],
}


def _cham_mot_cap(client: ollama.Client, a: Dict, b: Dict) -> Optional[Dict]:
    """Chấm một cặp. Trả None nếu không chấm được (lỗi, JSON hỏng, điểm ngoài thang)."""
    prompt = PROMPT_DOI_CHIEU.format(
        nguon_a=a.get("nguon", "?"), trang_a=a.get("trang", "?"),
        noi_dung_a=a.get("noidung", "")[:1500],
        nguon_b=b.get("nguon", "?"), trang_b=b.get("trang", "?"),
        noi_dung_b=b.get("noidung", "")[:1500],
    )
    tham_so = dict(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 400},
        format=_SCHEMA_MAU_THUAN,
        think=False,
    )
    try:
        try:
            phan_hoi = client.chat(**tham_so)
        except ollama.ResponseError:
            tham_so.pop("think")
            phan_hoi = client.chat(**tham_so)
        ket_qua = json.loads(phan_hoi["message"]["content"])
        ket_qua["muc_do"] = float(ket_qua["muc_do"])
    except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
        logger.info("Không đối chiếu được nguồn (chưa kết nối được Ollama).")
        return None
    except Exception:
        logger.warning("Không chấm được một cặp đối chiếu - bỏ qua cặp này.", exc_info=True)
        return None

    if not 0.0 <= ket_qua["muc_do"] <= 1.0:
        logger.warning("Mức độ mâu thuẫn %s ngoài thang [0,1] - loại mẫu.", ket_qua["muc_do"])
        return None
    return ket_qua


def tim_mau_thuan(
    cac_doan: List[Dict],
    embedding_service=None,
    client: Optional[ollama.Client] = None,
) -> List[Dict]:
    """Tìm các cặp đoạn trích MÂU THUẪN nhau trong tập đã truy xuất."""
    if not config.BAT_DOI_CHIEU_NGUON or len(cac_doan) < 2:
        return []

    if len({d.get("nguon") for d in cac_doan}) < 2:
        return []

    vector_doan = None
    if embedding_service is not None:
        try:
            van_ban = [d.get("doan_khop") or d.get("noidung", "") for d in cac_doan]
            vector_doan = embedding_service.encode_tai_lieu(van_ban)
        except Exception:
            logger.warning("Không encode được đoạn để đối chiếu - bỏ điều kiện cùng chủ đề.",
                           exc_info=True)

    cac_cap = cac_cap_dang_ngo(cac_doan, vector_doan)
    if not cac_cap:
        return []

    logger.info("Đối chiếu chéo %d cặp đoạn trích đáng ngờ.", len(cac_cap))
    client = client or ollama.Client(host=config.OLLAMA_HOST)
    ket_qua = []
    for i, j in cac_cap:
        a, b = cac_doan[i], cac_doan[j]

        cac_lan = []
        for _ in range(max(1, config.SO_LAN_CHAM_MAU_THUAN)):
            lan = _cham_mot_cap(client, a, b)
            if lan is None or not lan.get("co_mau_thuan"):
                cac_lan = []
                break
            cac_lan.append(lan)
        if not cac_lan:
            continue

        muc_do = min(l["muc_do"] for l in cac_lan)
        if muc_do < config.NGUONG_MAU_THUAN:
            continue

        ket_qua.append({
            "nguon_a": a.get("nguon"), "trang_a": a.get("trang"),
            "nguon_b": b.get("nguon"), "trang_b": b.get("trang"),
            "muc_do": muc_do,
            "noi_dung_xung_dot": cac_lan[0].get("phan_tich", "").strip(),
        })

    ket_qua.sort(key=lambda m: -m["muc_do"])
    if ket_qua:
        logger.info("Phát hiện %d mâu thuẫn giữa các nguồn.", len(ket_qua))
    return ket_qua
