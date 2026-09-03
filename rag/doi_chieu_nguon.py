"""Phát hiện MÂU THUẪN giữa các đoạn trích được truy xuất từ những tài liệu khác nhau.

Vấn đề mà module này giải quyết
--------------------------------
Toàn bộ phần còn lại của hệ thống xử lý mỗi đoạn trích ĐỘC LẬP: xếp hạng độc lập, đưa vào
prompt cạnh nhau, rồi để LLM tự viết một câu trả lời gộp. Không có bước nào hỏi "các đoạn
này có nói ngược nhau không".

Với tài liệu học tập thật thì đó là chuyện xảy ra thường xuyên: giáo trình in năm cũ và slide
cập nhật ghi khác con số; hai môn định nghĩa khác nhau cùng một khái niệm; biểu mẫu cũ và
quy định mới quy định khác nhau về cùng một thủ tục. Khi đó LLM đọc cả hai rồi lặng lẽ chọn
một bên, hoặc tệ hơn là trộn lẫn - và người đọc mất đúng thông tin quan trọng nhất:

    "Hai nguồn của bạn đang không thống nhất về chuyện này."

Đó là thông tin mà một hệ thống có trích dẫn PHẢI nói ra. Trích dẫn tồn tại để người đọc tự
kiểm chứng được; giấu đi việc hai căn cứ đá nhau thì trích dẫn mất phần lớn giá trị.

Thiết kế: hai tầng, đúng khuôn của retrieval
--------------------------------------------
Chấm mọi cặp bằng LLM là O(n²): C(TOP_K, 2) cặp - 6 cặp với TOP_K=4, 15 cặp nếu TOP_K trở
lại 6 - tức chừng ấy lượt gọi model cộng vào sau MỖI
câu trả lời - không thể chấp nhận trên CPU. Nên dùng lại đúng cấu trúc mà retrieval đã dùng
(quét rộng bằng thứ rẻ, đọc kỹ bằng thứ đắt trên phần còn lại):

  Tầng 1 (`cac_cap_dang_ngo`) - tất định, mili giây, không gọi model:
      khác nguồn  +  cùng chủ đề (cosine)  +  có dấu hiệu bất đồng bề mặt (số / phủ định)
  Tầng 2 (`_cham_mot_cap`) - LLM đọc kỹ, chỉ chạy trên vài cặp sống sót.

Đại đa số câu hỏi không có cặp nào qua nổi tầng 1, tức tốn đúng 0 lượt gọi LLM.

Vì sao nghiêng hẳn về phía IM LẶNG
----------------------------------
§5.43 đã đo được LLM-as-judge với model 4B lật phán quyết 1/8 lần dù temperature=0. Ở đây
hai loại sai không ngang giá:

  - BÁO ĐỘNG GIẢ ("tài liệu của bạn mâu thuẫn nhau" trong khi chúng không hề) làm người dùng
    mất niềm tin vào chính tài liệu của họ, và họ không có cách nào rẻ để kiểm tra lại.
  - BỎ SÓT một mâu thuẫn thật chỉ đưa hệ thống về đúng hành vi cũ - không tệ hơn trước.

Nên: cặp phải qua được cả ba điều kiện tất định của tầng 1, rồi phải được chấm "có mâu thuẫn"
ở TẤT CẢ các lần chấm (config.SO_LAN_CHAM_MAU_THUAN), rồi mức độ còn phải vượt ngưỡng. Ba
lớp này đều nghiêng về một phía: thà không nói gì.
"""

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


# ============================================================
# TẦNG 1: lọc cặp đáng ngờ (tất định, không gọi model)
# ============================================================

# Số có nghĩa: số nguyên hoặc thập phân, kèm cả dạng có dấu phân cách nghìn. Cố ý BỎ QUA số
# đứng một mình quá ngắn ngữ cảnh (số thứ tự đầu dòng) bằng cách đòi ranh giới từ hai bên.
_SO = re.compile(r"\b\d[\d.,]*\b")

# SỐ VIẾT BẰNG CHỮ. Bỏ qua nhóm này thì bộ lọc gần như vô dụng trên tài liệu tiếng Việt:
# chính ca mẫu của cả đồ án - "Nhà nước có NĂM đặc điểm" đối lại "có BỐN đặc điểm" (§5.22) -
# không có lấy một chữ số nào. Test bắt được đúng chỗ này.
#
# Cố ý LOẠI "một" và "tư": "một" đóng vai mạo từ trong vô số câu tiếng Việt ("một cách",
# "một số"), còn "tư" vừa là số bốn vừa là "tư nhân"/"tư cách". Đưa chúng vào sẽ khiến gần
# như cặp nào cũng "khác số".
#
# Vẫn còn nhập nhằng đã biết: "năm" cũng là đơn vị thời gian, "ba" cũng là danh từ. Chấp nhận
# được vì đây là BỘ LỌC THÔ nghiêng về độ phủ - cặp lọt qua vẫn phải vượt tiếp điều kiện
# cosine, rồi phải được LLM xác nhận ở tầng 2. Giá của một nhận nhầm là một lượt gọi model,
# giá của một bỏ sót là mất hẳn tính năng trên đúng loại tài liệu nó sinh ra để phục vụ.
_SO_BANG_CHU = {
    "hai": "2", "ba": "3", "bốn": "4", "năm": "5", "sáu": "6",
    "bảy": "7", "tám": "8", "chín": "9", "mười": "10",
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_TU = re.compile(r"[^\W\d_]+", re.UNICODE)

# Từ phủ định. Chỉ những từ đảo hẳn nghĩa mệnh đề - không lấy "chưa chắc", "hiếm khi"... vì
# chúng là mức độ chứ không phải đảo cực, và trộn hai thứ đó vào nhau sinh báo động giả.
_PHU_DINH = re.compile(
    r"\b(không|chẳng|chưa|đừng|khỏi phải|không phải|không được|không có"
    r"|not|no|never|cannot|can't|don't|doesn't|isn't|aren't|without)\b",
    re.IGNORECASE,
)


def _tap_so(van_ban: str) -> set:
    """Tập các số xuất hiện trong đoạn, đã chuẩn hoá về dạng so sánh được.

    Chuẩn hoá vì cùng một con số hay được viết khác nhau giữa hai tài liệu ("1.000" và
    "1000", "2,5" và "2.5"). Không chuẩn hoá thì mọi cặp đều "khác số" và tầng 1 mất tác dụng
    lọc - nó sẽ đẩy hết mọi cặp lên tầng LLM, đúng thứ thiết kế này muốn tránh.
    """
    ket_qua = set()
    for tho in _SO.findall(van_ban):
        gon = tho.replace(".", "").replace(",", "").lstrip("0")
        if gon:
            ket_qua.add(gon)
    # Số viết bằng chữ được quy về CÙNG một dạng với chữ số, nên "5 đặc điểm" và "năm đặc
    # điểm" không bị coi là hai con số khác nhau.
    for tu in _TU.findall(van_ban.lower()):
        if tu in _SO_BANG_CHU:
            ket_qua.add(_SO_BANG_CHU[tu])
    return ket_qua


def co_dau_hieu_bat_dong(doan_a: str, doan_b: str) -> bool:
    """Hai đoạn có dấu hiệu BỀ MẶT của việc nói khác nhau không?

    Đây là một bộ lọc thô có chủ đích - nó chỉ cần đủ tốt để cắt số cặp phải gửi lên LLM,
    không cần đúng. Sai theo hướng nào cũng an toàn: bỏ sót thì mất một lần kiểm (hệ thống về
    hành vi cũ), nhận nhầm thì tốn thêm một lượt LLM rồi bị chính LLM bác bỏ.
    """
    so_a, so_b = _tap_so(doan_a), _tap_so(doan_b)
    # Hai bên cùng nêu số, mà tập số lại lệch nhau -> ứng viên mạnh nhất. Đây là dạng mâu
    # thuẫn rõ ràng và hay gặp nhất giữa hai tài liệu ("năm đặc điểm" / "bốn đặc điểm").
    if so_a and so_b and so_a != so_b:
        return True
    # Lệch cực: một bên phủ định, bên kia không. Yếu hơn tín hiệu số (rất nhiều câu chứa
    # "không" mà chẳng mâu thuẫn với ai), nên nó chỉ đưa cặp lên tầng 2 chứ không kết luận gì.
    if bool(_PHU_DINH.search(doan_a)) != bool(_PHU_DINH.search(doan_b)):
        return True
    return False


def cac_cap_dang_ngo(cac_doan: List[Dict], vector_doan: Optional[np.ndarray]) -> List[tuple]:
    """Chọn ra các cặp (i, j) đáng đem đi chấm, sắp theo cosine giảm dần.

    vector_doan: ma trận (n, dim) đã CHUẨN HOÁ, song song với cac_doan. None = bỏ điều kiện
    cùng chủ đề (dùng trong test, và khi không có embedding service).

    Ba điều kiện, đều rẻ:
      1. KHÁC NGUỒN. Hai đoạn trong cùng một file mà "mâu thuẫn" thì hầu hết là do đọc hỏng
         (bảng bị cắt, cột bị nối sai) chứ không phải thông tin xung đột - và đó là bài toán
         của khâu đọc tài liệu, không phải của module này. Cross-document mới là ca có giá
         trị ứng dụng: nó nói được điều mà người đọc không tự thấy khi mở từng file riêng.
      2. CÙNG CHỦ ĐỀ. Hai đoạn phải nói về cùng một chuyện thì mới mâu thuẫn được; hai đoạn
         khác chủ đề chỉ là hai thông tin khác nhau.
      3. CÓ DẤU HIỆU BẤT ĐỒNG bề mặt (xem co_dau_hieu_bat_dong).
    """
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

    # Sắp theo cosine giảm dần rồi cắt: nếu buộc phải bỏ bớt cặp vì trần chi phí thì bỏ cặp
    # ít giống chủ đề nhất, tức cặp ít khả năng là mâu thuẫn thật nhất.
    ung_vien.sort(key=lambda x: -x[0])
    return [(i, j) for _, i, j in ung_vien[: config.SO_CAP_DOI_CHIEU_TOI_DA]]


# ============================================================
# TẦNG 2: chấm bằng LLM (structured output)
# ============================================================

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

# Cùng lý do với _SCHEMA_DIEM_SO ở evaluation/metrics.py (§5.6): format="json" chỉ ép cú pháp
# JSON hợp lệ chứ không ép ĐÚNG TÊN FIELD, và model từng tự đổi tên field khiến việc parse
# thất bại rồi điểm mặc định về 0 một cách âm thầm.
#
# Lưu ý theo đúng §5.48: minimum/maximum ở đây chỉ là TÀI LIỆU, không phải hàng rào - Ollama
# dịch schema sang grammar mà grammar không biểu diễn được ràng buộc khoảng giá trị. Việc
# kiểm thang điểm nằm ở code bên dưới.
#
# THỨ TỰ FIELD Ở ĐÂY LÀ MỘT QUYẾT ĐỊNH, KHÔNG PHẢI NGẪU NHIÊN. Bản đầu đặt "co_mau_thuan"
# lên trước, và nó tái tạo lại đúng lỗi mà §5.56 đã phải sửa cho prompt kiểm chứng: JSON
# được sinh tuần tự theo grammar, nên model phải CHỐT PHÁN QUYẾT trước khi viết được một
# chữ lập luận nào, rồi không quay lại sửa được nữa.
#
# Hậu quả đo được (bộ kiểm định, ca "hai điều kiện học bổng bổ sung cho nhau"): model chấm
# co_mau_thuan=true, muc_do=1.0, rồi tự viết trong phần giải thích rằng *"hai đoạn không
# cùng nói về một chuyện"* - tức chính nó bác bỏ phán quyết của chính nó, theo đúng định
# nghĩa mà prompt đã nêu. Đảo thứ tự thành phân tích -> mức độ -> phán quyết thì lập luận
# được viết trước và phán quyết rút ra từ nó.
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
    # `think=False` KÈM `format=<schema>` - và thứ tự lập luận ở đây quan trọng, vì §5.23 đã
    # kết luận "tuyệt đối không truyền think=False".
    #
    # Kết luận đó đúng với đầu ra TỰ DO: lúc ấy think=False không tắt suy luận mà chỉ tắt việc
    # TÁCH nó ra, nên toàn bộ chuỗi "Okay, let me figure out..." đổ thẳng vào content. Nhưng
    # khi có `format`, Ollama ép sinh theo một grammar JSON - model KHÔNG THỂ sinh văn xuôi
    # tự do nữa, vì văn xuôi không phải JSON hợp lệ. Chính grammar trở thành thứ chặn suy luận.
    #
    # Đo trực tiếp trên cùng một cặp đoạn (§5.59):
    #     format=schema, không truyền think  -> thinking 1795 ký tự, content RỖNG, 5.8s
    #     format=schema + think=False        -> thinking 0,        JSON đúng,     1.1s
    #
    # Không có nó thì tính năng này KHÔNG CHẠY: bản đầu đo được 0/4 ca mâu thuẫn thật, vì
    # num_predict bị chuỗi suy luận ăn hết và content về rỗng ở mọi ca.
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
            # Model không có chế độ suy luận (llama3, mistral...) thì máy chủ từ chối tham số
            # `think`. Bỏ nó ra rồi gọi lại - những model đó vốn không sinh suy luận nên
            # không cần chặn gì.
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
        # Đúng lỗi đã gặp ở §5.48: giám khảo đổi sang thang phần trăm hoặc thang 1-5. Ở đây
        # loại hẳn mẫu thay vì kẹp giá trị - kẹp là đoán ý model rồi ghi kết quả đoán ra màn
        # hình cho người dùng đọc.
        logger.warning("Mức độ mâu thuẫn %s ngoài thang [0,1] - loại mẫu.", ket_qua["muc_do"])
        return None
    return ket_qua


def tim_mau_thuan(
    cac_doan: List[Dict],
    embedding_service=None,
    client: Optional[ollama.Client] = None,
) -> List[Dict]:
    """Tìm các cặp đoạn trích MÂU THUẪN nhau trong tập đã truy xuất.

    Trả về list {"nguon_a","trang_a","nguon_b","trang_b","muc_do","noi_dung_xung_dot"},
    sắp theo mức độ giảm dần. Danh sách RỖNG là kết quả bình thường và hay gặp nhất.

    Không bao giờ ném lỗi ra ngoài: đây là một lớp thông tin THÊM, không được phép làm hỏng
    câu trả lời vốn đã sinh xong.
    """
    if not config.BAT_DOI_CHIEU_NGUON or len(cac_doan) < 2:
        return []

    # Chỉ có một nguồn duy nhất thì không có gì để đối chiếu chéo - thoát trước khi encode,
    # vì encode cả tập đoạn là phần đắt nhất của tầng 1.
    if len({d.get("nguon") for d in cac_doan}) < 2:
        return []

    vector_doan = None
    if embedding_service is not None:
        try:
            # Encode CHUNK KHỚP chứ không phải vùng ngữ cảnh đã mở rộng: vùng mở rộng kéo
            # theo nội dung lân cận không liên quan, làm loãng phép so "hai đoạn có cùng chủ
            # đề không" - cùng lý do đã dùng khi rerank chấm trên chunk gốc (§5.24).
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

        # ĐỒNG THUẬN: chấm nhiều lần, chỉ giữ khi MỌI lần đều nói có mâu thuẫn. Thoát sớm
        # ngay khi có một lần nói không - phần lớn cặp bị loại ở lần chấm đầu, nên trên thực
        # tế chi phí gần với một lần chấm chứ không phải SO_LAN_CHAM_MAU_THUAN lần.
        cac_lan = []
        for _ in range(max(1, config.SO_LAN_CHAM_MAU_THUAN)):
            lan = _cham_mot_cap(client, a, b)
            if lan is None or not lan.get("co_mau_thuan"):
                cac_lan = []
                break
            cac_lan.append(lan)
        if not cac_lan:
            continue

        # Lấy mức độ THẤP NHẤT trong các lần chấm, không lấy trung bình: khi các lần chấm
        # không thống nhất về mức độ thì phải nghiêng về phía dè dặt hơn, đúng hướng ưu tiên
        # im lặng đã nêu ở đầu file.
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
