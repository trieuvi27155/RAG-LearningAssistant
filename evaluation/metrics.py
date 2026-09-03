"""Các độ đo dùng để đánh giá hệ thống RAG trên bộ câu hỏi test.

- Precision@K, Recall@K: so khớp (nguồn, trang) của chunk truy xuất được với đáp án
  đúng do người dùng tự chuẩn bị trong test_questions.json - đo chất lượng Retrieval.
- Faithfulness, Answer Relevance: dùng chính LLM cục bộ (JUDGE_MODEL) để tự chấm điểm
  (LLM-as-judge đơn giản) - đo chất lượng câu trả lời cuối cùng.

LLM-as-judge ở đây là một thước đo CÓ SAI SỐ, và sai số của nó không có triệu chứng: điểm
0.0 chấm cho một câu trả lời đúng trông y hệt điểm 0.0 chấm cho một câu bịa. Đã xảy ra thật
(§5.38). Vì vậy Faithfulness được bọc thêm hai lớp tự kiểm tra, xem §5.43:
  - chấm nhiều lần lấy TRUNG VỊ, vì temperature=0 không khiến giám khảo tất định (đo được:
    8 lần chấm cùng một ca cho [0, 1, 1, 1, 1, 1, 1, 1]);
  - đối chiếu với một phép đo TẤT ĐỊNH (do_bam_ngu_canh) để tự bật cờ khi hai bên mâu thuẫn.
"""

import json
import logging
import re
from typing import Dict, List, Tuple

import ollama

import config
from rag.citation import cau_theo_trich_dan, dinh_dang_trich_dan, do_bam_ngu_canh

logger = logging.getLogger(__name__)

_MAU_TU = re.compile(r"[0-9A-Za-zÀ-ỹ]+")

# do_bam_ngu_canh() sống ở rag/citation.py, KHÔNG có bản sao ở đây. Phép đo này được dùng ở
# cả hai nơi - tầng chạy thật (hiện chỉ báo "bám nguồn" dưới câu trả lời) và tầng đánh giá
# (bắt lỗi chấm sai của LLM-as-judge) - nên nếu để hai bản, chúng sẽ trôi khỏi nhau và con
# số trong báo cáo sẽ nói về một thứ khác với con số người dùng nhìn thấy. Cùng một lý do đã
# khiến sinh_cau_tra_loi() phải gọi lại đúng generator của chế độ streaming (§5.42).

_MAU_TACH_CAU_DON = re.compile(r"(?<=[.!?])\s+|\n+")
# Câu ngắn hơn ngần này từ thường là câu nối ("Cụ thể như sau:", "Ngoài ra:") - không mang
# nội dung để mà đối chiếu, tính vào sẽ kéo mức thấp nhất xuống 0 một cách vô nghĩa.
SO_TU_TOI_THIEU_MOT_CAU = 8


def do_bam_ngu_canh_thap_nhat(cau_tra_loi: str, ngu_canh: str) -> float:
    """Mức bám ngữ cảnh của CÂU TỆ NHẤT trong câu trả lời (thay vì trung bình cả bài).

    Vì sao phải là mức thấp nhất chứ không phải trung bình: cờ nghi ngờ dùng phép đo này để
    bác lại kết luận "câu trả lời bịa" của giám khảo, mà một câu trả lời ĐÚNG MỘT NỬA -
    nửa đầu chép nguyên văn, nửa sau bịa - vẫn cho trung bình khá cao (đo được 35% trên ca
    kiểm định), đủ để bật cờ oan trong khi giám khảo hoàn toàn đúng.

    Lấy mức thấp nhất thì mâu thuẫn mới thật sự sắc: cờ chỉ bật khi MỌI câu đều có nguyên
    văn trong ngữ cảnh, tức không còn chỗ nào cho thông tin bịa nằm. Đo trên bộ kiểm định
    (evaluation/kiem_dinh_judge.py): ca hỏng thật (ngữ cảnh dính chữ) giữ nguyên 0.53, còn
    ca đúng-một-nửa tụt từ 0.53 xuống 0.00 - hai nhóm tách hẳn nhau, không cần một ngưỡng
    tinh chỉnh mong manh.
    """
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
    """Thứ hạng (1-based) của đoạn ĐÚNG đầu tiên trong danh sách truy xuất; 0 nếu không có.

    Vì sao cần metric này bên cạnh Precision@K/Recall@K:
    Cả 2 metric kia đều là phép so TẬP HỢP - chúng không quan tâm đoạn đúng nằm ở vị trí
    thứ nhất hay thứ sáu. Nhưng THỨ TỰ mới là thứ quyết định chất lượng thực tế: LLM đọc
    đoạn [1] kỹ hơn đoạn [6], và người dùng nhìn trích dẫn đầu tiên. Quan trọng hơn, thứ tự
    chính là thứ DUY NHẤT mà rerank thay đổi - đo bằng Precision@K thì rerank luôn hiện ra
    là "không cải thiện gì", dù thực tế nó đã kéo đoạn đúng từ hạng 6 lên hạng 1.

    Điều này lộ rõ khi corpus nhỏ: với 22 chunk và TOP_K=6, mỗi lần truy xuất lấy 27% toàn
    bộ corpus nên Precision@K gần như cố định ở 1/6 bất kể xếp hạng tốt hay tệ.
    """
    tap_dung = {_khoa(m) for m in cac_trang_dung}
    if not tap_dung:
        return 0
    for thu_hang, chunk in enumerate(cac_chunk_truy_xuat, start=1):
        if _khoa(chunk) in tap_dung:
            return thu_hang
    return 0


def nghich_dao_thu_hang(cac_chunk_truy_xuat: List[Dict], cac_trang_dung: List[Dict]) -> float:
    """1/thứ_hạng của đoạn đúng đầu tiên (0 nếu không tìm thấy) - thành phần của MRR.

    Dùng nghịch đảo thay vì thứ hạng thô để trung bình cộng có ý nghĩa: chênh lệch giữa
    hạng 1 và hạng 2 (1.0 -> 0.5) đáng kể hơn nhiều so với giữa hạng 9 và hạng 10
    (0.111 -> 0.100), đúng với cảm nhận thực tế về chất lượng xếp hạng.
    """
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


# Dùng JSON Schema thật (thay vì chỉ format="json") để ép Ollama trả đúng field
# "diem"/"ly_do" - format="json" chỉ đảm bảo cú pháp JSON hợp lệ, không đảm bảo đúng
# tên field. Từng gặp trường hợp model tự trả {"answer": "0.5"} khiến parse thất bại
# và điểm luôn bị tính thành 0.0 dù câu trả lời thực tế đúng.
_SCHEMA_DIEM_SO = {
    "type": "object",
    "properties": {
        # minimum/maximum ghi vào schema cho ĐÚNG ý định, nhưng KHÔNG được tin: Ollama dịch
        # JSON Schema sang grammar để ép sinh, mà grammar không biểu diễn được ràng buộc
        # khoảng giá trị của số. Việc chặn thật nằm ở _diem_hop_le() bên dưới.
        "diem": {"type": "number", "minimum": 0, "maximum": 1},
        "ly_do": {"type": "string"},
    },
    "required": ["diem", "ly_do"],
}


def _diem_hop_le(diem: float) -> bool:
    return 0.0 <= diem <= 1.0


def _goi_judge_kep(prompt: str) -> Dict:
    """Gọi giám khảo MỘT lần, kẹp điểm về [0,1] nếu nó trả ngoài thang.

    Dùng cho những phép đo chỉ chấm một lần mỗi lượt (answer_relevance, và từng cặp của
    citation accuracy) - ở đó không có cơ chế bỏ phiếu để loại mẫu hỏng như
    _goi_judge_on_dinh(), nên lựa chọn duy nhất còn lại là kẹp. Cảnh báo đã được ghi ở
    _goi_judge; kẹp ở đây chỉ để một điểm 100.0 không âm thầm phá vỡ trung bình của cả bảng.
    """
    ket_qua = _goi_judge(prompt)
    if not ket_qua.get("hop_le", True):
        ket_qua["diem"] = min(max(ket_qua["diem"], 0.0), 1.0)
    return ket_qua


def _goi_judge_on_dinh(prompt: str, so_lan: int) -> Dict:
    """Chấm `so_lan` lần rồi lấy TRUNG VỊ, kèm biên độ dao động giữa các lần.

    Vì sao cần: `temperature=0` KHÔNG làm giám khảo tất định. Đo trực tiếp trên qwen3:4b
    (evaluation/kiem_dinh_judge.py, cùng một prompt, cùng một ca): hai lần chấm liên tiếp
    cho 1.00 và 0.00 - dao động bằng đúng toàn bộ thang điểm. Nguyên nhân là model vẫn sinh
    một chuỗi suy luận dài trước khi kết luận, và chỉ cần một mắt xích trong chuỗi đó rẽ
    khác là kết luận lật ngược.

    Hệ quả nếu bỏ qua: hai lần chạy run_evaluation.py trên CÙNG một phiên bản code cho ra
    hai con số Faithfulness khác nhau, nên mọi so sánh "trước/sau khi sửa" đều vô nghĩa -
    không phân biệt được cải tiến thật với nhiễu của thước đo.

    Lấy TRUNG VỊ chứ không phải trung bình: điểm của giám khảo hay dồn về hai cực 0 và 1,
    trung bình của {0, 1, 1} ra 0.67 - một con số không ứng với phán quyết nào cả; trung vị
    ra 1.0, đúng là phán quyết mà đa số lần chấm đưa ra.

    Số lần chấm là ĐỐI XỨNG (không phải chỉ hỏi lại khi điểm thấp): hỏi lại một chiều sẽ
    đẩy điểm lên cao một cách có hệ thống, tức đổi một loại sai số lấy một loại sai số khác
    khó thấy hơn.
    """
    cac_lan = [_goi_judge(prompt) for _ in range(max(so_lan, 1))]

    # Loại hẳn những lần chấm cho điểm ngoài thang (xem _goi_judge). Chấm nhiều lần rồi lấy
    # trung vị vốn đã là một cơ chế bỏ phiếu, nên loại mẫu hỏng ở đây gần như luôn còn đủ
    # mẫu tốt để quyết. Chỉ khi HỎNG HẾT mới phải lùi về kẹp giá trị - và lúc đó ghi log to,
    # vì đó là dấu hiệu model đang hiểu sai thang điểm một cách hệ thống chứ không phải lỡ tay.
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
    # Giữ lại lý do của ĐÚNG lần chấm có điểm bằng trung vị, để đọc lại CSV còn khớp được
    # điểm với lời giải thích, thay vì một lý do lấy từ lần chấm khác.
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
        # temperature=0 để việc chấm điểm ổn định, lặp lại được giữa các lần chạy.
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

    # ĐIỂM NGOÀI KHOẢNG [0,1] LÀ MỘT LỖI THẬT ĐÃ GẶP, không phải phòng xa. Prompt ghi rõ
    # "chấm từ 0 đến 1", JSON Schema cũng khai báo minimum/maximum - vậy mà một lần chạy
    # đánh giá thật vẫn nhận về 100.0 và 5.0 (model đổi sang thang phần trăm và thang 1-5).
    # Ollama dịch schema sang grammar để ép sinh, mà grammar không biểu diễn được ràng buộc
    # khoảng giá trị của số, nên schema không chặn được.
    #
    # Hậu quả nếu không chặn thì rất nặng và rất kín: chỉ 2 trong 29 câu đủ kéo Faithfulness
    # trung bình từ 0.88 lên 4.43 - một con số vô nghĩa, nhưng nằm gọn trong bảng kết quả
    # trông vẫn bình thường nếu đọc lướt. Đánh dấu hop_le=False để _goi_judge_on_dinh() loại
    # hẳn mẫu hỏng ra khỏi phép lấy trung vị, thay vì kẹp về [0,1] rồi coi như không có gì.
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
    """Faithfulness do LLM chấm, KÈM một cờ tự nghi ngờ chính điểm số đó.

    Vì sao cần cờ: LLM-as-judge với model 4B là một thước đo có sai số hệ thống, không phải
    một con số khách quan. Ca hỏng đã gặp thật (§5.38): 2 câu về sách Bishop bị chấm 0.0
    trong khi đọc trực tiếp thì cả hai đều đúng và có căn cứ - giám khảo không đối chiếu
    được câu trả lời sạch với ngữ cảnh bị dính chữ, nên kết luận là "bịa". Điểm 0.33 của cả
    nhóm là lỗi của THƯỚC ĐO.

    Nguy hiểm ở chỗ nó không có triệu chứng gì: nếu tin con số đó, người làm đồ án sẽ đi
    sửa prompt hoặc đổi model để "cải thiện Faithfulness" - tối ưu vào một cái sai. Nên
    thước đo phải tự khai báo lúc nào nó có thể đang sai:

        dang_ngo = giám khảo chấm THẤP  ĐỒNG THỜI  câu trả lời chép gần nguyên văn ngữ cảnh

    Hai điều đó không thể cùng đúng. Khi cờ bật, run_evaluation.py đánh dấu câu đó và báo
    riêng trung bình đã loại các câu đáng ngờ, thay vì lặng lẽ trộn số sai vào bảng kết quả.
    """
    ngu_canh = "\n\n".join(c["noidung"] for c in cac_chunk_nguon) or "(không có ngữ cảnh)"
    prompt = PROMPT_FAITHFULNESS.format(ngu_canh=ngu_canh, cau_tra_loi=cau_tra_loi)
    ket_qua = _goi_judge_on_dinh(prompt, config.SO_LAN_CHAM_FAITHFULNESS)

    ket_qua["bam_ngu_canh"] = do_bam_ngu_canh(cau_tra_loi, ngu_canh)
    # Cờ bật theo mức của CÂU TỆ NHẤT, không phải trung bình cả bài - xem
    # do_bam_ngu_canh_thap_nhat(): trung bình để lọt câu trả lời đúng-một-nửa vào diện
    # "đáng ngờ" trong khi giám khảo chấm thấp là hoàn toàn đúng.
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
    """Đo xem những đoạn trích được câu trả lời DẪN có thật sự chống lưng cho ý đang dẫn chúng.

    Khác với Faithfulness (hỏi "toàn bộ câu trả lời có bịa không" trên TOÀN BỘ ngữ cảnh),
    metric này soi từng cặp (ý, đoạn được dẫn cho ý đó). Một câu trả lời có thể đạt
    Faithfulness cao mà trích dẫn vẫn sai chỗ: mọi ý đều có căn cứ đâu đó trong ngữ cảnh,
    nhưng số [n] gắn kèm lại trỏ sang đoạn khác - người đọc bấm vào nguồn sẽ không thấy điều
    họ vừa đọc. Đó đúng là lỗi "citation nhầm chunk" cần đo riêng mới thấy.

    Chạy ở TẦNG ĐÁNH GIÁ, không phải lúc chạy thật: mỗi cặp tốn 1 lượt gọi LLM, không đáng
    để bắt người dùng chờ thêm ở mỗi câu hỏi.

    Trả về {"diem", "so_cap_da_kiem", "chi_tiet"}.

    PHÂN BIỆT HAI KIỂU "không dẫn nguồn nào", vì gộp chúng lại đã che mất một lỗi thật:
      - CÂU TỪ CHỐI ("Không tìm thấy thông tin trong tài liệu") -> diem=None, bị loại khỏi
        trung bình. Đúng: không dẫn nguồn là hành vi ĐÚNG ở đây, vừa nói không có thông tin
        vừa chỉ vào một trang cụ thể mới là tự mâu thuẫn.
      - CÂU TRẢ LỜI THẬT mà không dẫn số nào -> diem=0.0. Đây là LỖI TRÍCH DẪN nặng nhất:
        người đọc không có cách nào kiểm chứng điều vừa đọc, tức mất đúng thứ mà cả hệ thống
        sinh ra để bảo đảm.

    Bản trước trả None cho cả hai, và cái giá phải trả là cụ thể: một lần sửa system prompt
    khiến model bỏ hẳn trích dẫn ở 3 câu, nhưng vì cả 3 rơi vào diện "bị loại khỏi trung
    bình" nên điểm Citation trung bình gần như không đổi - lỗi đi lọt qua thước đo. Đúng cái
    kiểu hỏng âm thầm mà §5.43 nói tới, lần này ở phía ngược lại: thước đo quá dễ dãi.
    """
    trich_dan = dinh_dang_trich_dan(cac_chunk_nguon)
    if not trich_dan:
        # Không truy xuất được đoạn nào (index rỗng, hoặc mọi đoạn đều dưới ngưỡng). Không có
        # nguồn nào tồn tại thì "trích dẫn có đúng chỗ không" là câu hỏi vô nghĩa - khác hẳn
        # trường hợp CÓ nguồn mà model không chịu dẫn ở dưới.
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
            # LLM dẫn một số không tồn tại (vd [9] khi chỉ có 6 đoạn) - đây là trích dẫn bịa,
            # tính 0 điểm chứ không bỏ qua: bịa số nguồn là lỗi nặng nhất của citation.
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
