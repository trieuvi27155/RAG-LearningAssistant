"""Test cho việc kiểm chứng citation: câu nào dẫn đoạn nào, và đoạn đó có chống lưng không.

Phần tách câu-theo-trích-dẫn là logic thuần tuý nên test trực tiếp. Phần chấm điểm phải gọi
LLM nên thay judge bằng hàm giả - thứ cần kiểm ở đây là cách ghép cặp và cách tính trung
bình, không phải chất lượng chấm điểm của model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from evaluation import metrics
from rag.citation import cau_theo_trich_dan, loc_theo_tham_chieu


def _chunk(trang: int, doan_khop: str):
    return {"nguon": "a.pdf", "trang": trang, "noidung": doan_khop,
            "doan_khop": doan_khop, "diem_similarity": 0.9}


# ======================================================================
# Ghép câu với số trích dẫn nó dẫn
# ======================================================================

def test_ghep_dung_cau_voi_so_trich_dan():
    """Biết "LLM dẫn đoạn [2]" là chưa đủ - phải biết nó dẫn [2] cho Ý NÀO thì mới đối
    chiếu được ý đó với nội dung đoạn [2]."""
    ket_qua = cau_theo_trich_dan("Nhà nước có 5 đặc điểm [1]. Pháp luật do nhà nước ban hành [2].")
    assert set(ket_qua) == {1, 2}
    assert "5 đặc điểm" in ket_qua[1][0]
    assert "ban hành" in ket_qua[2][0]


def test_mot_cau_dan_nhieu_so_thi_tinh_cho_tung_so():
    ket_qua = cau_theo_trich_dan("Điều này được nêu ở cả hai chỗ [1][3].")
    assert set(ket_qua) == {1, 3}


def test_mot_so_duoc_dan_o_nhieu_cau_thi_gom_lai():
    ket_qua = cau_theo_trich_dan("Ý đầu tiên [1]. Ý thứ hai cũng vậy [1].")
    assert len(ket_qua[1]) == 2


def test_tach_duoc_ca_danh_sach_gach_dau_dong():
    """Câu trả lời hay ở dạng liệt kê - mỗi gạch đầu dòng là một ý riêng cần căn cứ riêng,
    dù không kết thúc bằng dấu chấm."""
    ket_qua = cau_theo_trich_dan("Các đặc điểm:\n- Có chủ quyền [1]\n- Ban hành pháp luật [2]")
    assert set(ket_qua) == {1, 2}
    assert "chủ quyền" in ket_qua[1][0]
    assert "chủ quyền" not in ket_qua[2][0], "hai gạch đầu dòng không được dính vào nhau"


def test_cau_tra_loi_khong_dan_so_nao():
    assert cau_theo_trich_dan("Không tìm thấy thông tin trong tài liệu.") == {}
    assert cau_theo_trich_dan("") == {}


# ======================================================================
# Chấm điểm căn cứ
# ======================================================================

@pytest.fixture
def judge_gia(monkeypatch):
    """Thay judge thật bằng hàm trả điểm theo từ khoá - kiểm cách ghép cặp và tính trung
    bình, không kiểm chất lượng model."""
    cac_lan_goi = []

    def gia(prompt: str):
        cac_lan_goi.append(prompt)
        return {"diem": 1.0 if "KHỚP" in prompt else 0.0, "ly_do": "giả"}

    monkeypatch.setattr(metrics, "_goi_judge", gia)
    return cac_lan_goi


def test_cham_diem_tung_cap_y_va_doan_duoc_dan(judge_gia):
    cac_chunk = [_chunk(1, "Nội dung KHỚP với ý"), _chunk(2, "Nội dung khác hẳn")]
    ket_qua = metrics.do_chinh_xac_trich_dan("Ý thứ nhất [1]. Ý thứ hai [2].", cac_chunk)

    assert ket_qua["so_cap_da_kiem"] == 2
    assert ket_qua["diem"] == 0.5  # 1 cặp đúng, 1 cặp sai
    assert len(judge_gia) == 2


def test_dan_so_khong_ton_tai_bi_tinh_khong_diem(judge_gia):
    """Dẫn [9] khi chỉ có 2 đoạn là trích dẫn BỊA - lỗi nặng nhất của citation, phải bị
    tính điểm 0 chứ không được bỏ qua."""
    ket_qua = metrics.do_chinh_xac_trich_dan("Theo tài liệu [9].", [_chunk(1, "KHỚP"), _chunk(2, "x")])

    assert ket_qua["diem"] == 0.0
    assert ket_qua["so_cap_da_kiem"] == 1
    assert "không tồn tại" in ket_qua["chi_tiet"][0]["ly_do"]
    assert not judge_gia, "không cần gọi judge cho số không tồn tại"


def test_khong_dan_so_nao_tra_ve_none_khong_phai_khong_diem(judge_gia):
    """"Không có gì để kiểm" khác hẳn "kiểm rồi và sai" - trả 0.0 sẽ kéo trung bình xuống
    oan cho những câu từ chối hợp lệ."""
    ket_qua = metrics.do_chinh_xac_trich_dan("Không tìm thấy thông tin trong tài liệu.", [_chunk(1, "x")])

    assert ket_qua["diem"] is None
    assert ket_qua["so_cap_da_kiem"] == 0
    assert not judge_gia


def test_khong_co_chunk_nao_thi_khong_goi_judge(judge_gia):
    assert metrics.do_chinh_xac_trich_dan("Có dẫn [1].", [])["diem"] is None
    assert not judge_gia


# ======================================================================
# Phân biệt TRÍCH DẪN THẬT với PHỎNG ĐOÁN của hệ thống
# ======================================================================

def _chunk_mau(n=3):
    return [
        {"nguon": f"tai_lieu_{i}.pdf", "trang": i, "noidung": f"Noi dung doan {i} du dai.",
         "doan_khop": f"Noi dung doan {i} du dai.", "diem_similarity": 0.9 - i * 0.1}
        for i in range(1, n + 1)
    ]


def test_trich_dan_that_khong_bi_danh_dau_la_suy_doan():
    from rag.citation import loc_theo_tham_chieu

    kq = loc_theo_tham_chieu(_chunk_mau(), "Nội dung này lấy từ [2].")
    assert kq and kq[0]["nguon"] == "tai_lieu_2.pdf"
    assert kq[0]["la_suy_doan"] is False


def test_model_khong_dan_nguon_thi_phai_danh_dau_la_suy_doan():
    """Bản trước lặng lẽ hiện đoạn điểm cao nhất dưới nhãn "Nguồn" - tức trình bày PHỎNG ĐOÁN
    của hệ thống như thể là căn cứ mà câu trả lời đã dùng. Người đọc yên tâm nhầm, và đó đúng
    là kiểu trích dẫn gây hiểu lầm nhất với một hệ thống mà giá trị cốt lõi là kiểm chứng
    được. Đo trên bộ 29 câu: 4 câu trả lời thật không gắn số nào - không phải hiếm.
    """
    from rag.citation import loc_theo_tham_chieu

    kq = loc_theo_tham_chieu(_chunk_mau(), "Cong nghe thong tin.")
    assert kq, "vẫn phải hiện đoạn liên quan nhất - không hiện gì thì mất đường đối chiếu"
    assert kq[0]["la_suy_doan"] is True


def test_cau_tu_choi_khong_kem_nguon_nao():
    """Vừa nói "không tìm thấy thông tin" vừa chỉ vào một trang cụ thể là tự mâu thuẫn."""
    import config
    from rag.citation import loc_theo_tham_chieu

    assert loc_theo_tham_chieu(_chunk_mau(), config.CAU_TU_CHOI["vi"]) == []


# ======================================================================
# Đo bám ngữ cảnh ở TẦNG CHẠY THẬT (không chỉ tầng đánh giá)
# ======================================================================

def test_do_bam_ngu_canh_dung_chung_mot_ban_giua_runtime_va_danh_gia():
    """Hai bản sao sẽ trôi khỏi nhau, và khi đó con số trong báo cáo nói về một thứ khác với
    con số người dùng nhìn thấy - cùng lý do khiến sinh_cau_tra_loi() phải gọi lại đúng
    generator của chế độ streaming."""
    from evaluation.metrics import do_bam_ngu_canh as ban_danh_gia
    from rag.citation import do_bam_ngu_canh as ban_runtime

    assert ban_runtime is ban_danh_gia


def test_bam_nguon_cao_khi_chep_nguyen_van_thap_khi_bia():
    from rag.citation import do_bam_ngu_canh

    ngu_canh = "Nha nuoc co ba dac diem co ban la tinh giai cap va quyen luc cong cong."
    assert do_bam_ngu_canh("Nha nuoc co ba dac diem co ban la tinh giai cap", ngu_canh) > 0.8
    assert do_bam_ngu_canh("Thu do nuoc Phap la Paris co nhieu bao tang", ngu_canh) == 0.0


# ======================================================================
# Số hiệu đoạn trích phải TRA NGƯỢC ĐƯỢC ở tầng hiển thị
# ======================================================================
# Người đọc thấy "[4]" trong câu trả lời thì phải tìm được nguồn [4] trong danh sách bên dưới.
# Không có điều đó, câu trả lời "có dẫn nguồn" nhưng người đọc vẫn không biết dẫn nguồn NÀO -
# tức mất đúng nửa sau của chuỗi kiểm chứng mà cả hệ thống này tồn tại để bảo đảm.

def _bo_chunk():
    return [
        {"nguon": "a.pdf", "trang": 1, "noidung": "doan mot", "doan_khop": "doan mot",
         "diem_similarity": 0.9},
        {"nguon": "a.pdf", "trang": 2, "noidung": "doan hai", "doan_khop": "doan hai",
         "diem_similarity": 0.8},
        {"nguon": "a.pdf", "trang": 1, "noidung": "doan ba", "doan_khop": "doan ba",
         "diem_similarity": 0.7},
        {"nguon": "b.pdf", "trang": 7, "noidung": "doan bon", "doan_khop": "doan bon",
         "diem_similarity": 0.6},
    ]


def test_moi_nguon_mang_dung_so_hieu_da_duoc_dan():
    kq = loc_theo_tham_chieu(_bo_chunk(), "Ý một [1]. Ý hai [2]. Ý ba [4].")
    assert [t["cac_so"] for t in kq] == [[1], [2], [4]]
    # Số KHÔNG hiển thị cho người đọc (xem bo_so_trich_dan), nhưng vẫn phải đi theo dữ liệu:
    # đó là căn cứ để biết câu trả lời dùng nguồn nào.


def test_hai_doan_cung_mot_trang_giu_ca_hai_so():
    """Gộp theo (nguồn, trang) là đúng cho người đọc, nhưng gộp xong mà mất số thì "[3]"
    trong câu trả lời thành con số cụt."""
    kq = loc_theo_tham_chieu(_bo_chunk(), "Ý một [1]. Ý ba [3].")
    assert len(kq) == 1
    assert kq[0]["cac_so"] == [1, 3]


def test_cau_khong_dan_nguon_thi_KHONG_gan_so():
    """Đoạn hiển thị lúc này là PHỎNG ĐOÁN của hệ thống, không phải căn cứ model đã dẫn.
    Gắn "[1]" vào sẽ hàm ý model có dẫn nó - đúng kiểu trình bày phỏng đoán như thể là sự
    thật mà §5.54 đã phải sửa một lần rồi."""
    kq = loc_theo_tham_chieu(_bo_chunk(), "Câu trả lời không có số nào.")
    assert kq[0]["la_suy_doan"] is True
    assert kq[0]["cac_so"] == []


def test_so_hieu_khop_dung_thu_tu_doan_trich_trong_prompt():
    """Số hiển thị phải là thứ tự đoạn trích TRONG PROMPT - đúng con số LLM đã nhìn thấy.
    Lệch một bậc là trích dẫn trỏ sai nguồn mà không có lỗi nào báo ra."""
    chunks = _bo_chunk()
    kq = loc_theo_tham_chieu(chunks, "Chỉ dẫn [4].")
    assert kq[0]["cac_so"] == [4]
    assert (kq[0]["nguon"], kq[0]["trang"]) == (chunks[3]["nguon"], chunks[3]["trang"])


# ======================================================================
# Dạng gộp nhiều số trong MỘT cặp ngoặc: "[3,4,5]"
# ======================================================================
# Lỗi có thật, lộ ra trên một lần chạy thật: câu trả lời dẫn "[6]" và "[3,4,5]", regex cũ
# (\[(\d+)\]) chỉ khớp "[6]" nên BA NGUỒN biến mất khỏi danh sách mà không có lỗi nào báo ra.
# Nguy hiểm hơn phần hiển thị: cau_theo_trich_dan() dùng chung mẫu này để chấm Citation
# accuracy, nên mọi ý dẫn theo dạng gộp đều bị bỏ khỏi phép chấm - thước đo âm thầm bỏ sót
# đúng những câu trả lời dẫn nhiều nguồn cho một ý.

def test_bat_duoc_dang_gop_nhieu_so_trong_mot_ngoac():
    from rag.citation import _cac_so_tham_chieu
    assert _cac_so_tham_chieu("IoT là ... [6]. Môi trường IoT [3,4,5]") == [6, 3, 4, 5]
    assert _cac_so_tham_chieu("Cảm biến [3, 4]") == [3, 4]
    assert _cac_so_tham_chieu("a [3;4]") == [3, 4]
    assert _cac_so_tham_chieu("a [1][2]") == [1, 2]


def test_moc_bang_hinh_khong_bi_nham_la_so_trich_dan():
    """Nội dung đoạn trích có mốc [BẢNG]/[HÌNH]; nhận nhầm chúng thành số sẽ hỏng cả phép lọc."""
    from rag.citation import _cac_so_tham_chieu
    assert _cac_so_tham_chieu("[BẢNG] nội dung [HÌNH]") == []


def test_dang_gop_hien_du_moi_nguon():
    kq = loc_theo_tham_chieu(_bo_chunk(), "Ý một [1]. Các thành phần [2,4].", so_toi_da=5)
    assert [t["cac_so"] for t in kq] == [[1], [2], [4]]


def test_dang_gop_duoc_cham_citation_accuracy():
    """Từng số trong "[3,4]" phải được ghép với câu đã dẫn nó, để chấm được từng cặp."""
    assert cau_theo_trich_dan("Ý một [3,4]. Ý hai [6].") == {
        3: ["Ý một [3,4]."],
        4: ["Ý một [3,4]."],
        6: ["Ý hai [6]."],
    }


# ======================================================================
# Bỏ số [n] khỏi văn bản HIỂN THỊ (dữ liệu gốc giữ nguyên)
# ======================================================================

def test_bo_so_trich_dan_khoi_cau_tra_loi():
    from rag.citation import bo_so_trich_dan
    assert bo_so_trich_dan("IoT là mạng lưới thiết bị [6].") == "IoT là mạng lưới thiết bị."
    assert bo_so_trich_dan("Môi trường IoT [3,4,5]") == "Môi trường IoT"
    assert bo_so_trich_dan("Cảm biến [3, 4] và đám mây [5].") == "Cảm biến và đám mây."


def test_bo_so_khong_dung_toi_moc_bang_hinh():
    """Đoạn trích hiển thị có thể còn mốc [BẢNG]/[HÌNH]; nuốt mất chúng là mất thông tin."""
    from rag.citation import bo_so_trich_dan
    assert bo_so_trich_dan("[BẢNG] Cột A | Cột B") == "[BẢNG] Cột A | Cột B"


def test_bo_so_giu_lai_ngoac_dang_do_khi_stream():
    """Khi đang stream, mảnh cuối có thể cắt ngang giữa "[3,4]". Đẩy nguyên mảnh dở ra màn
    hình thì người dùng thấy một mẩu ngoặc nhấp nháy rồi biến mất - cùng lý do khiến việc bóc
    thẻ <think> phải là máy trạng thái chứ không phải regex trên chuỗi hoàn chỉnh (§5.42)."""
    from rag.citation import bo_so_trich_dan
    assert bo_so_trich_dan("Nội dung đang chạy [3,") == "Nội dung đang chạy"
    assert bo_so_trich_dan("Nội dung đang chạy [") == "Nội dung đang chạy"


def test_du_lieu_goc_van_con_so_de_loc_nguon_va_cham_diem():
    """bo_so_trich_dan CHỈ dùng ở tầng hiển thị. Bỏ số khỏi dữ liệu sẽ phá đúng cơ chế quyết
    định nguồn nào được hiện, và phá luôn phép chấm Citation accuracy."""
    from rag.citation import bo_so_trich_dan
    goc = "Ý một [1]. Ý hai [2,4]."
    assert bo_so_trich_dan(goc) == "Ý một. Ý hai."
    assert [t["cac_so"] for t in loc_theo_tham_chieu(_bo_chunk(), goc)] == [[1], [2], [4]]
    assert set(cau_theo_trich_dan(goc)) == {1, 2, 4}
