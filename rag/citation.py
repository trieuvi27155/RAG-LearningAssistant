"""Định dạng danh sách đoạn trích nguồn thành trích dẫn dễ hiển thị (tên file + trang/slide
+ đoạn trích), lấy trực tiếp từ metadata đã gắn từ Document Loader - không suy luận lại
thông tin nguồn, tránh sai lệch giữa câu trả lời và trích dẫn thực tế đã dùng.
"""

import re
from typing import Dict, List, Optional

import config
from rag.document_loader import MOC_BANG_DONG, MOC_BANG_MO
from rag.image_extractor import MOC_ANH

# Độ dài tối đa của đoạn trích hiển thị. Đoạn trích nay là chunk KHỚP NHẤT với câu hỏi
# (field "doan_khop"), không phải cả vùng ngữ cảnh đã mở rộng, nên độ dài của nó vốn đã
# vừa phải - con số này chỉ còn là chặn an toàn cho trường hợp chunk dài bất thường.
DO_DAI_TRICH_DAN = 600

# Số đoạn trích LLM gắn vào câu trả lời. Bắt cả dạng gộp nhiều số trong MỘT cặp ngoặc -
# "[3,4,5]", "[3, 4]", "[3;4]" - chứ không chỉ "[3][4][5]".
#
# Dạng gộp là thứ model thật sự viết ra, dù system prompt chỉ nêu ví dụ dạng "[2]". Bắt được
# nó trên một lần chạy thật: câu trả lời dẫn "[6]" và "[3,4,5]", nhưng regex cũ (\[(\d+)\])
# chỉ khớp "[6]" — ba nguồn còn lại BIẾN MẤT khỏi danh sách nguồn mà không có lỗi nào báo ra.
#
# Hậu quả lan xa hơn phần hiển thị: cau_theo_trich_dan() dùng chung mẫu này để chấm Citation
# accuracy, nên mọi ý dẫn nguồn theo dạng gộp đều bị bỏ khỏi phép chấm — tức thước đo âm thầm
# bỏ sót đúng những câu trả lời dẫn nhiều nguồn cho một ý. Đây là kiểu hỏng im lặng mà cả đồ
# án tìm cách loại bỏ, và nó chỉ lộ ra khi số hiệu được hiển thị cho người đọc thấy.
_MAU_THAM_CHIEU_KHOI = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
_MAU_SO_TRONG_KHOI = re.compile(r"\d+")


def _cac_so_tham_chieu(van_ban: str) -> List[int]:
    """Mọi số đoạn trích được dẫn trong đoạn văn bản, kể cả dạng gộp "[3,4,5]"."""
    cac_so = []
    for khoi in _MAU_THAM_CHIEU_KHOI.findall(van_ban or ""):
        cac_so.extend(int(s) for s in _MAU_SO_TRONG_KHOI.findall(khoi))
    return cac_so


# Bỏ số đoạn trích khỏi văn bản HIỂN THỊ. Kèm cả khoảng trắng đứng trước để "xử lý [6]." không
# thành "xử lý ." sau khi bỏ.
_MAU_BO_SO_TRICH_DAN = re.compile(r"[ \t]*\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
# Đuôi ngoặc còn dở khi đang stream ("... xử lý [3,"): giữ lại trên màn hình thì người dùng
# thấy một mảnh ngoặc nhấp nháy rồi biến mất. Cùng lý do với việc _LocSuyLuanTheoLuong phải
# giữ lại nửa cái thẻ <think> thay vì đẩy thẳng ra màn hình (§5.42).
_MAU_NGOAC_DANG_DO = re.compile(r"[ \t]*\[[\d,;\s]*$")


def bo_so_trich_dan(van_ban: str) -> str:
    """Bỏ các số [n] khỏi văn bản để HIỂN THỊ, giữ nguyên dữ liệu gốc.

    Chỉ dùng ở tầng giao diện. Bản gốc (còn số) vẫn là thứ được lưu trong lịch sử chat và
    được đưa vào loc_theo_tham_chieu() cùng metrics.do_chinh_xac_trich_dan() - bỏ số khỏi dữ
    liệu sẽ phá đúng cơ chế quyết định nguồn nào được hiển thị và phá luôn phép chấm Citation
    accuracy, tức làm hỏng thứ mà các con số này sinh ra để phục vụ.

    Người đọc không mất gì khi không thấy số: số hiệu là thứ tự đoạn trích TRONG PROMPT, một
    thứ tự họ không nhìn thấy nên cũng không tra ngược được.
    """
    s = _MAU_BO_SO_TRICH_DAN.sub("", van_ban or "")
    s = _MAU_NGOAC_DANG_DO.sub("", s)
    # Bỏ số xong hay để lại khoảng trắng lửng trước dấu câu ("xử lý ." / "gồm :").
    s = re.sub(r"[ \t]+([.,;:!?)])", r"\1", s)
    return re.sub(r"[ \t]{2,}", " ", s)
# Dự phòng khi LLM trích theo SỐ TRANG thay vì số đoạn trích ("theo Slide 109", "at page 12")
# dù system prompt đã cấm - model nhỏ vẫn thỉnh thoảng làm vậy vì tiêu đề mỗi đoạn trích
# trong prompt có ghi kèm số trang. Bắt lại được thì trích dẫn vẫn đúng thay vì phải lùi về
# đoán "đoạn điểm cao nhất".
_MAU_SO_TRANG = re.compile(r"(?:trang|slide|page)\s*/?\s*(?:slide\s*)?(\d+)", re.IGNORECASE)
# Tách câu để đối chiếu từng ý với căn cứ của nó. Ngoài dấu câu thường, tách cả theo xuống
# dòng và gạch đầu dòng: câu trả lời hay ở dạng danh sách, mỗi gạch đầu dòng là một ý riêng
# cần căn cứ riêng dù không kết thúc bằng dấu chấm.
_MAU_TACH_CAU = re.compile(r"(?<=[.!?:])\s+|\n+\s*[-*•]?\s*")

# Câu chỉ NÓI VỀ một đoạn trích để loại nó ra ("các phần [3][4] không liên quan tới đề
# tài"), chứ không hề dựa vào nó để khẳng định điều gì.
#
# Vì sao phải tách riêng: cau_theo_trich_dan() phục vụ phép đo "đoạn [n] có chống lưng cho
# ý đang dẫn nó không". Với một câu kiểu trên thì câu hỏi đó vô nghĩa - người viết đang
# khẳng định đoạn [3] KHÔNG chứa gì cả, nên giám khảo tất nhiên chấm 0, và điểm Citation
# accuracy bị kéo xuống bởi đúng cái hành vi thận trọng mà ta muốn khuyến khích. Đo trên bộ
# tài liệu thật: câu "Đề tài thuộc lĩnh vực nào?" có 6 cặp được kiểm thì 4 cặp là loại này,
# kéo điểm từ 1.00 xuống 0.33 (xem ARCHITECTURE.md §5.39).
_MAU_CAU_LOAI_TRU_NGUON = re.compile(
    r"không\s+(?:liên\s+quan|đề\s+cập|nhắc\s+(?:tới|đến)|nói\s+(?:tới|đến|về)"
    r"|chứa|có\s+thông\s+tin|cung\s+cấp\s+thông\s+tin)"
    r"|(?:is|are)\s+(?:not\s+relevant|unrelated)"
    r"|do(?:es)?\s+not\s+(?:mention|contain|discuss|cover|relate)"
    r"|no\s+information\s+(?:about|on)",
    re.IGNORECASE,
)


def dinh_dang_trich_dan(cac_chunk: List[Dict]) -> List[Dict]:
    """Chuẩn hoá danh sách đoạn trích nguồn thành list trích dẫn để hiển thị:
    {"nguon", "trang", "doan_trich", "diem_similarity"}.

    Ánh xạ 1-1 và GIỮ NGUYÊN THỨ TỰ của cac_chunk: phần tử thứ i ở đây tương ứng đúng với
    đoạn trích được đánh số [i+1] trong prompt gửi cho LLM. Vì vậy hàm này KHÔNG được phép
    loại trùng hay sắp xếp lại - làm vậy sẽ khiến số [2] trong câu trả lời trỏ sang một
    nguồn khác với đoạn [2] mà LLM thật sự đã đọc. Việc gộp các đoạn cùng trang khi hiển
    thị được làm ở loc_theo_tham_chieu(), sau khi đã dùng xong các con số.

    "doan_trich" lấy từ "doan_khop" - tức chunk thật sự khớp câu hỏi - chứ không cắt từ đầu
    vùng ngữ cảnh đã mở rộng. Bản trước cắt 400 ký tự đầu của cả trang đã gộp, nên với tài
    liệu dài (trang ~2000 ký tự) đoạn hiển thị gần như luôn là phần đầu trang, chẳng liên
    quan gì tới câu hỏi - đúng triệu chứng "trích dẫn không chính xác" đã gặp.
    """
    trich_dan = []
    for chunk in cac_chunk:
        doan_trich = chunk.get("doan_khop") or chunk["noidung"]
        # Mốc [BẢNG]/[HÌNH] có ích trong prompt (báo cho LLM biết đây là bảng/hình) nhưng
        # là rác khi hiển thị cho người đọc - bỏ đi ở đúng tầng hiển thị, không đụng dữ liệu.
        for moc in (MOC_BANG_MO, MOC_BANG_DONG, MOC_ANH):
            doan_trich = doan_trich.replace(moc, "")
        doan_trich = doan_trich.strip()
        if len(doan_trich) > DO_DAI_TRICH_DAN:
            doan_trich = doan_trich[:DO_DAI_TRICH_DAN].rstrip() + "..."
        trich_dan.append(
            {
                "nguon": chunk["nguon"],
                "trang": chunk["trang"],
                # Mọi trang mà đoạn trích ĐI QUA (thường chỉ có 1). Đoạn trích được phép mở
                # rộng sang trang liền kề để nối lại một định nghĩa bị ranh giới trang cắt
                # đôi (config.MO_RONG_QUA_RANH_GIOI_TRANG) - khi đó câu trả lời có thể dùng
                # nội dung nằm ở trang bên cạnh, mà trích dẫn lại chỉ ghi trang neo. Nói ra
                # đủ khoảng trang là điều kiện để người đọc mở đúng chỗ mà đối chiếu; giấu
                # đi thì trích dẫn "đúng" nhưng trỏ thiếu, đúng loại sai lệch mà §5.54 đã
                # phải sửa một lần.
                "cac_trang": chunk.get("cac_trang") or [chunk["trang"]],
                "doan_trich": doan_trich,
                "diem_similarity": chunk.get("diem_similarity"),
                # Để UI biết hiển thị đoạn trích dưới dạng nào: bảng Markdown render thành
                # bảng, ảnh hiện ra ảnh thật thay vì chỉ ghi chú thích bằng chữ.
                "loai_noi_dung": chunk.get("loai_noi_dung", "van_ban"),
                "duong_dan_anh": chunk.get("duong_dan_anh", ""),
            }
        )
    return trich_dan


def loc_theo_tham_chieu(
    cac_chunk: List[Dict], cau_tra_loi: str, so_toi_da: Optional[int] = None
) -> List[Dict]:
    """Chỉ giữ những nguồn mà câu trả lời THẬT SỰ tham chiếu tới ([1], [2]... trong nội dung).

    System prompt bắt buộc LLM gắn số đoạn trích cho từng ý, nên những con số này là bằng
    chứng trực tiếp về việc nó đã dùng đoạn nào - chính xác hơn nhiều so với cách cũ (luôn
    hiển thị đoạn có điểm similarity cao nhất). Với tài liệu dài, đoạn điểm cao nhất thường
    KHÔNG phải đoạn LLM dùng để trả lời: nhiều trang cùng chủ đề có điểm sát nhau, đoạn
    thắng về điểm số có thể chỉ nhắc tới chủ đề, còn nội dung trả lời thật sự nằm ở đoạn
    khác - đó là lý do trích dẫn hiển thị hay "lệch" so với câu trả lời.

    Ba lớp, lớp sau chỉ dùng khi lớp trước không cho kết quả nào:
      1. Số đoạn trích [n] - dạng chuẩn mà system prompt bắt buộc.
      2. Số trang được nhắc trong câu trả lời ("theo Slide 109") - model nhỏ đôi khi vẫn
         trích kiểu này dù đã bị cấm, bắt lại được thì trích dẫn vẫn đúng.
      3. Nguồn liên quan nhất - vẫn tốt hơn là không hiển thị gì.

    Riêng câu TỪ CHỐI thì không hiển thị nguồn nào: vừa nói "không tìm thấy thông tin trong
    tài liệu" vừa chỉ vào một trang cụ thể là tự mâu thuẫn, và đó chính là kiểu trích dẫn
    gây hiểu lầm nhất (người đọc tưởng trang đó có liên quan).

    Các đoạn cùng (nguồn, trang) được gộp lại khi hiển thị: người đọc chỉ cần biết trang
    nào, không cần thấy 2 dòng trỏ cùng một trang.
    """
    if not cac_chunk:
        return []
    cau_tra_loi = (cau_tra_loi or "").strip()
    if any(cau_tra_loi.startswith(tu_choi) for tu_choi in config.CAU_TU_CHOI.values()):
        return []
    so_toi_da = so_toi_da or config.SO_TRICH_DAN_HIEN_THI

    trich_dan = dinh_dang_trich_dan(cac_chunk)
    cac_so_tham_chieu = set(_cac_so_tham_chieu(cau_tra_loi))

    # SỐ HIỆU ĐI THEO ĐOẠN TRÍCH tới tận tầng hiển thị. Không có nó, người đọc thấy "[4]"
    # trong câu trả lời nhưng danh sách nguồn bên dưới chỉ ghi tên file + trang, không số
    # nào - tức con số trong câu trả lời KHÔNG TRA NGƯỢC ĐƯỢC. Với một hệ thống mà giá trị
    # cốt lõi là kiểm chứng được, đó là mất đúng nửa sau của chuỗi kiểm chứng: người đọc biết
    # câu trả lời có dẫn nguồn, nhưng không biết dẫn nguồn NÀO.
    #
    # Số hiệu là thứ tự đoạn trích TRONG PROMPT (1-based), tức đúng con số LLM đã gắn.
    duoc_dung = [
        {**t, "so_hieu": i}
        for i, t in enumerate(trich_dan, start=1)
        if i in cac_so_tham_chieu
    ]
    la_suy_doan = False
    if not duoc_dung:
        cac_trang_nhac_toi = {int(s) for s in _MAU_SO_TRANG.findall(cau_tra_loi)}
        duoc_dung = [
            {**t, "so_hieu": i}
            for i, t in enumerate(trich_dan, start=1)
            if t["trang"] in cac_trang_nhac_toi
        ]
    if not duoc_dung:
        # Model không dẫn số nào cả. Bản trước lặng lẽ lấy đoạn điểm cao nhất và HIỂN THỊ NÓ
        # NHƯ MỘT NGUỒN - tức trình bày một PHỎNG ĐOÁN của hệ thống như thể là căn cứ mà câu
        # trả lời đã dùng. Đó đúng là kiểu trích dẫn gây hiểu lầm nhất: người đọc tin rằng
        # câu trả lời dựa trên trang đó, trong khi thật ra không ai biết nó dựa trên gì.
        #
        # Đây không phải trường hợp hiếm: đo trên bộ 29 câu, 4 câu trả lời thật không gắn số
        # nào; đo lặp lại cùng một câu 4 lần thì tỉ lệ tuân thủ chỉ 50% (§5.46).
        #
        # Nay vẫn hiển thị đoạn liên quan nhất - không hiển thị gì thì người đọc mất luôn
        # đường đối chiếu - nhưng ĐÁNH DẤU RÕ là phỏng đoán, để tầng giao diện nói thật với
        # người đọc thay vì để họ tự tin nhầm.
        duoc_dung = [{**t, "so_hieu": i} for i, t in enumerate(trich_dan[:1], start=1)]
        la_suy_doan = True

    # Gộp theo (nguồn, trang) nhưng GIỮ LẠI MỌI SỐ HIỆU trỏ về đó: hai đoạn khác nhau của
    # cùng một trang là hai số khác nhau trong câu trả lời, và người đọc phải tra được cả hai.
    ket_qua, theo_khoa = [], {}
    for t in duoc_dung:
        khoa = (t["nguon"], t["trang"])
        if khoa in theo_khoa:
            theo_khoa[khoa]["cac_so"].append(t["so_hieu"])
            # Hai đoạn cùng trang neo có thể phủ khác nhau sang trang liền kề - gộp lại để
            # dòng nguồn hiển thị đủ khoảng trang thật sự đã đọc.
            theo_khoa[khoa]["cac_trang"] = sorted(
                set(theo_khoa[khoa]["cac_trang"]) | set(t["cac_trang"]), key=str
            )
            continue
        if len(ket_qua) >= so_toi_da:
            continue
        # la_suy_doan: câu trả lời KHÔNG dẫn số nào cả, nên đoạn hiển thị là phỏng đoán của
        # hệ thống. Gắn số vào đó sẽ hàm ý model đã dẫn nó - đúng kiểu trình bày phỏng đoán
        # như thể là sự thật mà §5.54 đã phải sửa một lần rồi. Để rỗng.
        muc = {**t, "la_suy_doan": la_suy_doan,
               "cac_so": [] if la_suy_doan else [t["so_hieu"]]}
        theo_khoa[khoa] = muc
        ket_qua.append(muc)

    for muc in ket_qua:
        muc["cac_so"] = sorted(set(muc["cac_so"]))

    return ket_qua


_MAU_TU_BAM = re.compile(r"[0-9A-Za-zÀ-ỹ]+")


def do_bam_ngu_canh(cau_tra_loi: str, ngu_canh: str, so_tu_moi_cum: int = 4) -> float:
    """Tỉ lệ cụm 4 từ liên tiếp của câu trả lời xuất hiện NGUYÊN VĂN trong ngữ cảnh.

    Phép đo TẤT ĐỊNH, không gọi model nào, chạy trong mili giây - nên dùng được ngay ở luồng
    trả lời thật chứ không chỉ ở tầng đánh giá.

    CÁCH ĐỌC CON SỐ NÀY CHO ĐÚNG (quan trọng, dễ dùng sai):
      - Điểm CAO là bằng chứng mạnh rằng câu trả lời KHÔNG bịa: nó đang chép gần nguyên văn
        từ ngữ cảnh.
      - Điểm THẤP thì KHÔNG chứng minh được gì cả. Một câu trả lời diễn đạt lại bằng lời của
        mình - điều hoàn toàn hợp lệ và thường là mong muốn - cũng cho điểm thấp.
    Vì vậy tuyệt đối không được dùng nó để TỪ CHỐI một câu trả lời: làm thế sẽ giết đúng
    những câu trả lời tốt nhất. Nó chỉ dùng để XÁC NHẬN, và để cảnh báo khi kết hợp với một
    tín hiệu khác (xem metrics.faithfulness).

    So sánh sau khi bỏ hết khoảng trắng ở cả hai bên, nên ngữ cảnh trích từ PDF bị dính chữ
    (§5.40) vẫn khớp được với câu trả lời viết đúng chuẩn.
    """
    cac_tu = _MAU_TU_BAM.findall(cau_tra_loi.lower())
    if len(cac_tu) < so_tu_moi_cum:
        return 0.0
    ngu_canh_rut_gon = "".join(_MAU_TU_BAM.findall(ngu_canh.lower()))
    cac_cum = [
        "".join(cac_tu[i : i + so_tu_moi_cum])
        for i in range(len(cac_tu) - so_tu_moi_cum + 1)
    ]
    return sum(1 for cum in cac_cum if cum in ngu_canh_rut_gon) / len(cac_cum)


def cau_theo_trich_dan(cau_tra_loi: str) -> Dict[int, List[str]]:
    """Ghép mỗi số trích dẫn [n] với những CÂU trong câu trả lời đã dẫn nó.

    Dùng để kiểm chứng: biết "LLM dẫn đoạn [2]" là chưa đủ, phải biết nó dẫn đoạn [2] để
    chống lưng cho Ý NÀO - có vậy mới đối chiếu được ý đó với nội dung đoạn [2] xem có thật
    sự khớp không (§5.14 mới chỉ giải quyết được vế "dẫn đoạn nào").

    Tách câu theo dấu chấm/xuống dòng và theo cả gạch đầu dòng, vì câu trả lời hay ở dạng
    danh sách - mỗi gạch đầu dòng là một ý riêng cần căn cứ riêng, dù không kết thúc bằng
    dấu chấm.

    Một câu dẫn nhiều số thì được tính cho từng số; một số được dẫn ở nhiều câu thì gom hết
    các câu đó lại.

    BỎ QUA những câu chỉ nói về đoạn trích để LOẠI nó ra ("các phần [3][4] không liên quan")
    - xem _MAU_CAU_LOAI_TRU_NGUON. Đó không phải một khẳng định đang dựa vào đoạn [3], nên
    đem đi hỏi "đoạn [3] có chứng minh điều này không" là đặt sai câu hỏi.
    """
    ket_qua: Dict[int, List[str]] = {}
    for cau in _MAU_TACH_CAU.split(cau_tra_loi or ""):
        cau = cau.strip()
        if not cau or _MAU_CAU_LOAI_TRU_NGUON.search(cau):
            continue
        for so in set(_cac_so_tham_chieu(cau)):
            ket_qua.setdefault(so, []).append(cau)
    return ket_qua


def format_text_trich_dan(cac_chunk: List[Dict]) -> str:
    """Trả về chuỗi text hiển thị nhanh trích dẫn (dùng cho script/terminal, không cần UI)."""
    return "\n".join(
        f"[{i}] {t['nguon']} - trang/slide {t['trang']}: \"{t['doan_trich']}\""
        for i, t in enumerate(dinh_dang_trich_dan(cac_chunk), start=1)
    )
