"""Các độ đo dùng để đánh giá hệ thống RAG trên bộ câu hỏi test."""

import json
import logging
import re
from typing import Dict, List, Tuple

import ollama

import config
from rag.citation import cau_theo_trich_dan, dinh_dang_trich_dan, do_bam_ngu_canh

logger = logging.getLogger(__name__)

_MAU_TU = re.compile(r"[0-9A-Za-zÀ-ỹ]+")

_MAU_TACH_CAU_DON = re.compile(r"(?<=[.!?])\s+|\n+")
SO_TU_TOI_THIEU_MOT_CAU = 8


def do_bam_ngu_canh_thap_nhat(cau_tra_loi: str, ngu_canh: str) -> float:
    """Mức bám ngữ cảnh của CÂU TỆ NHẤT trong câu trả lời (thay vì trung bình cả bài)."""
    cac_muc = [
        do_bam_ngu_canh(cau, ngu_canh)
        for cau in _MAU_TACH_CAU_DON.split(cau_tra_loi or "")
        if len(_MAU_TU.findall(cau)) >= SO_TU_TOI_THIEU_MOT_CAU
    ]
    return min(cac_muc) if cac_muc else 0.0


def _khoa(muc: Dict) -> Tuple[str, int]:
    """Khóa định danh 1 trang/slide: (tên file, số trang) - phải kết hợp cả 2 vì
    corpus có thể có nhiều file, chỉ số trang thôi không đủ để định danh duy nhất."""
    return (muc["nguon"], muc["trang"])


def precision_tai_k(cac_chunk_truy_xuat: List[Dict], cac_trang_dung: List[Dict]) -> float:
    """Trong K chunk truy xuất được, bao nhiêu tỉ lệ là đúng (khớp với đáp án mẫu)."""
    if not cac_chunk_truy_xuat:
        return 0.0
    tap_dung = {_khoa(m) for m in cac_trang_dung}
    so_dung = sum(1 for c in cac_chunk_truy_xuat if _khoa(c) in tap_dung)
    return so_dung / len(cac_chunk_truy_xuat)


def recall_tai_k(cac_chunk_truy_xuat: List[Dict], cac_trang_dung: List[Dict]) -> float:
    """Trong toàn bộ trang đúng đáp án, bao nhiêu tỉ lệ được tìm thấy trong K chunk truy xuất."""
    tap_dung = {_khoa(m) for m in cac_trang_dung}
    if not tap_dung:
        return 0.0
    tap_truy_xuat = {_khoa(c) for c in cac_chunk_truy_xuat}
    return len(tap_dung & tap_truy_xuat) / len(tap_dung)


def thu_hang_dung_dau_tien(cac_chunk_truy_xuat: List[Dict], cac_trang_dung: List[Dict]) -> int:
    """Thứ hạng (1-based) của đoạn ĐÚNG đầu tiên trong danh sách truy xuất; 0 nếu không có."""
    tap_dung = {_khoa(m) for m in cac_trang_dung}
    if not tap_dung:
        return 0
    for thu_hang, chunk in enumerate(cac_chunk_truy_xuat, start=1):
        if _khoa(chunk) in tap_dung:
            return thu_hang
    return 0


def nghich_dao_thu_hang(cac_chunk_truy_xuat: List[Dict], cac_trang_dung: List[Dict]) -> float:
    """1/thứ_hạng của đoạn đúng đầu tiên (0 nếu không tìm thấy) - thành phần của MRR."""
    thu_hang = thu_hang_dung_dau_tien(cac_chunk_truy_xuat, cac_trang_dung)
    return 1.0 / thu_hang if thu_hang else 0.0


PROMPT_FAITHFULNESS = """Bạn là giám khảo đánh giá độ TRUNG THỰC (faithfulness) của một câu trả lời so với ngữ cảnh cho trước.

NGỮ CẢNH:
{ngu_canh}

CÂU TRẢ LỜI CẦN ĐÁNH GIÁ:
{cau_tra_loi}

Hãy đánh giá xem MỌI thông tin trong câu trả lời có được hỗ trợ trực tiếp bởi ngữ cảnh trên hay không (không quan tâm câu trả lời có đầy đủ hay không, chỉ quan tâm có bịa thông tin ngoài ngữ cảnh hay không).
Chấm điểm "diem" từ 0 đến 1 (số thực):
- 1.0 = mọi thông tin đều có căn cứ trong ngữ cảnh.
- 0.0 = câu trả lời bịa hoàn toàn, không dựa vào ngữ cảnh.
- Trường hợp đặc biệt: nếu câu trả lời là lời từ chối kiểu "Không tìm thấy thông tin trong tài liệu." (không đưa ra bất kỳ thông tin cụ thể nào), LUÔN chấm 1.0, vì không có thông tin nào bị bịa ra cả.

Chỉ trả lời đúng định dạng JSON, không thêm chữ nào khác:
{{"diem": <số từ 0 đến 1>, "ly_do": "<giải thích ngắn gọn bằng tiếng Việt>"}}"""

PROMPT_ANSWER_RELEVANCE = """Bạn là giám khảo đánh giá độ LIÊN QUAN (answer relevance) giữa câu trả lời và câu hỏi.

CÂU HỎI: {cau_hoi}

CÂU TRẢ LỜI CẦN ĐÁNH GIÁ:
{cau_tra_loi}

Hãy đánh giá xem câu trả lời có đúng trọng tâm câu hỏi hay không (không quan tâm đúng/sai kiến thức, chỉ quan tâm có lạc đề hay không).
Chấm điểm "diem" từ 0 đến 1 (số thực):
- 1.0 = trả lời đúng trọng tâm, đầy đủ ý câu hỏi hỏi.
- 0.0 = hoàn toàn lạc đề.

Chỉ trả lời đúng định dạng JSON, không thêm chữ nào khác:
{{"diem": <số từ 0 đến 1>, "ly_do": "<giải thích ngắn gọn bằng tiếng Việt>"}}"""


_SCHEMA_DIEM_SO = {
    "type": "object",
    "properties": {
        "diem": {"type": "number", "minimum": 0, "maximum": 1},
        "ly_do": {"type": "string"},
    },
    "required": ["diem", "ly_do"],
}


def _diem_hop_le(diem: float) -> bool:
    return 0.0 <= diem <= 1.0


def _goi_judge_kep(prompt: str) -> Dict:
    """Gọi giám khảo MỘT lần, kẹp điểm về [0,1] nếu nó trả ngoài thang."""
    ket_qua = _goi_judge(prompt)
    if not ket_qua.get("hop_le", True):
        ket_qua["diem"] = min(max(ket_qua["diem"], 0.0), 1.0)
    return ket_qua


def _goi_judge_on_dinh(prompt: str, so_lan: int) -> Dict:
    """Chấm `so_lan` lần rồi lấy TRUNG VỊ, kèm biên độ dao động giữa các lần."""
    cac_lan = [_goi_judge(prompt) for _ in range(max(so_lan, 1))]

    hop_le = [lan for lan in cac_lan if lan.get("hop_le", True)]
    if not hop_le:
        logger.error(
            "CẢ %d lần chấm đều cho điểm ngoài khoảng [0,1] - kẹp về [0,1] để chạy tiếp, "
            "nhưng con số này KHÔNG đáng tin. Cân nhắc đổi JUDGE_MODEL.", len(cac_lan),
        )
        for lan in cac_lan:
            lan["diem"] = min(max(lan["diem"], 0.0), 1.0)
        hop_le = cac_lan

    cac_diem = sorted(lan["diem"] for lan in hop_le)
    trung_vi = cac_diem[len(cac_diem) // 2]
    ket_qua = next(lan for lan in hop_le if lan["diem"] == trung_vi)
    ket_qua["diem"] = trung_vi
    ket_qua["so_lan_bi_loai"] = len(cac_lan) - len(hop_le)
    ket_qua["dao_dong_judge"] = cac_diem[-1] - cac_diem[0]
    if ket_qua["dao_dong_judge"] > 0:
        logger.info(
            "Giám khảo chấm không ổn định giữa %d lần: %s -> lấy trung vị %.2f",
            len(cac_diem), cac_diem, trung_vi,
        )
    return ket_qua


def _goi_judge(prompt: str) -> Dict:
    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.chat(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        format=_SCHEMA_DIEM_SO,
    )
    noi_dung = response["message"]["content"]
    try:
        ket_qua = json.loads(noi_dung)
        ket_qua["diem"] = float(ket_qua["diem"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("Không parse được JSON từ judge model: %s", noi_dung[:200])
        return {
            "diem": 0.0,
            "ly_do": f"Không parse được JSON từ judge: {noi_dung[:200]}",
            "hop_le": False,
        }

    if not _diem_hop_le(ket_qua["diem"]):
        logger.warning(
            "Giám khảo trả điểm %s ngoài khoảng [0,1] (thang điểm bị hiểu sai) - loại mẫu "
            "này. Lý do giám khảo đưa ra: %.120s",
            ket_qua["diem"], ket_qua.get("ly_do", ""),
        )
        ket_qua["hop_le"] = False
        return ket_qua

    ket_qua["hop_le"] = True
    return ket_qua


def faithfulness(cau_tra_loi: str, cac_chunk_nguon: List[Dict]) -> Dict:
    """Faithfulness do LLM chấm, KÈM một cờ tự nghi ngờ chính điểm số đó."""
    ngu_canh = "\n\n".join(c["noidung"] for c in cac_chunk_nguon) or "(không có ngữ cảnh)"
    prompt = PROMPT_FAITHFULNESS.format(ngu_canh=ngu_canh, cau_tra_loi=cau_tra_loi)
    ket_qua = _goi_judge_on_dinh(prompt, config.SO_LAN_CHAM_FAITHFULNESS)

    ket_qua["bam_ngu_canh"] = do_bam_ngu_canh(cau_tra_loi, ngu_canh)
    ket_qua["bam_ngu_canh_thap_nhat"] = do_bam_ngu_canh_thap_nhat(cau_tra_loi, ngu_canh)
    ket_qua["dang_ngo"] = (
        ket_qua["diem"] <= config.NGUONG_DIEM_JUDGE_THAP
        and ket_qua["bam_ngu_canh_thap_nhat"] >= config.NGUONG_BAM_NGU_CANH_DE_NGHI_NGO
    )
    if ket_qua["dang_ngo"]:
        logger.warning(
            "Faithfulness %.2f nhưng MỌI câu của câu trả lời đều có nguyên văn trong ngữ "
            "cảnh (câu bám ít nhất: %.0f%%) - nhiều khả năng giám khảo chấm sai, không phải "
            "hệ thống bịa. Lý do giám khảo đưa ra: %.150s",
            ket_qua["diem"], ket_qua["bam_ngu_canh_thap_nhat"] * 100, ket_qua.get("ly_do", ""),
        )
    return ket_qua


def answer_relevance(cau_hoi: str, cau_tra_loi: str) -> Dict:
    prompt = PROMPT_ANSWER_RELEVANCE.format(cau_hoi=cau_hoi, cau_tra_loi=cau_tra_loi)
    return _goi_judge_kep(prompt)


PROMPT_CAN_CU_TRICH_DAN = """Bạn là giám khảo kiểm tra xem một đoạn trích có THẬT SỰ chứng minh cho một câu khẳng định hay không.

ĐOẠN TRÍCH ĐƯỢC DẪN LÀM CĂN CỨ:
{doan_trich}

CÂU KHẲNG ĐỊNH ĐANG DẪN ĐOẠN TRÍCH TRÊN:
{cau_khang_dinh}

Hãy đánh giá xem nội dung của câu khẳng định có được đoạn trích trên hỗ trợ trực tiếp hay không.
Chấm điểm "diem" từ 0 đến 1 (số thực):
- 1.0 = đoạn trích nêu rõ ràng thông tin trong câu khẳng định.
- 0.5 = đoạn trích có liên quan tới chủ đề nhưng KHÔNG nêu đúng thông tin được khẳng định.
- 0.0 = đoạn trích không liên quan, hoặc nói điều ngược lại.
Chỉ xét đúng đoạn trích này, KHÔNG dùng kiến thức bên ngoài. Câu khẳng định đúng về mặt kiến thức nhưng không có trong đoạn trích vẫn phải chấm thấp - vì mục đích ở đây là kiểm tra trích dẫn có đúng chỗ không, không phải kiểm tra kiến thức.

Chỉ trả lời đúng định dạng JSON, không thêm chữ nào khác:
{{"diem": <số từ 0 đến 1>, "ly_do": "<giải thích ngắn gọn bằng tiếng Việt>"}}"""


def do_chinh_xac_trich_dan(cau_tra_loi: str, cac_chunk_nguon: List[Dict]) -> Dict:
    """Đo xem những đoạn trích được câu trả lời DẪN có thật sự chống lưng cho ý đang dẫn chúng."""
    trich_dan = dinh_dang_trich_dan(cac_chunk_nguon)
    if not trich_dan:
        return {"diem": None, "so_cap_da_kiem": 0, "chi_tiet": []}

    theo_so = cau_theo_trich_dan(cau_tra_loi)
    if not theo_so:
        la_tu_choi = any(
            (cau_tra_loi or "").strip().startswith(tu_choi)
            for tu_choi in config.CAU_TU_CHOI.values()
        )
        if la_tu_choi:
            return {"diem": None, "so_cap_da_kiem": 0, "chi_tiet": []}
        logger.warning(
            "Câu trả lời KHÔNG dẫn nguồn nào dù không phải câu từ chối - tính 0 điểm trích "
            "dẫn: %.120s", cau_tra_loi,
        )
        return {
            "diem": 0.0,
            "so_cap_da_kiem": 0,
            "chi_tiet": [{"ly_do": "Câu trả lời thật nhưng không gắn số đoạn trích nào"}],
        }

    cac_diem, chi_tiet = [], []
    for so, cac_cau in sorted(theo_so.items()):
        if not 1 <= so <= len(trich_dan):
            cac_diem.append(0.0)
            chi_tiet.append({"so": so, "diem": 0.0, "ly_do": "Dẫn số đoạn trích không tồn tại"})
            continue
        doan = trich_dan[so - 1]["doan_trich"]
        for cau in cac_cau:
            ket_qua = _goi_judge_kep(
                PROMPT_CAN_CU_TRICH_DAN.format(doan_trich=doan, cau_khang_dinh=cau)
            )
            cac_diem.append(ket_qua["diem"])
            chi_tiet.append({"so": so, "cau": cau, "diem": ket_qua["diem"], "ly_do": ket_qua["ly_do"]})

    return {
        "diem": sum(cac_diem) / len(cac_diem) if cac_diem else None,
        "so_cap_da_kiem": len(cac_diem),
        "chi_tiet": chi_tiet,
    }
