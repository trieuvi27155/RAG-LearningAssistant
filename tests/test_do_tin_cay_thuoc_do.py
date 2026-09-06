"""Test cho phần THƯỚC ĐO tự kiểm tra chính nó, và cho việc chunk bảng lớn."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import do_bam_ngu_canh, do_bam_ngu_canh_thap_nhat
from rag.chunking import _cat_bang_giu_tieu_de, chia_chunk
from rag.citation import cau_theo_trich_dan
from rag.document_loader import MOC_BANG_DONG, MOC_BANG_MO, _ty_le_dinh_chu


_SO_LAN_LAP_CHO_CO_TRANG = 40


def test_van_ban_binh_thuong_khong_bi_coi_la_dinh_chu():
    binh_thuong = (
        "The bias-variance decomposition breaks down the expected squared loss into "
        "the squared bias and the variance of the model. "
    )
    assert _ty_le_dinh_chu(binh_thuong * _SO_LAN_LAP_CHO_CO_TRANG) < 0.05


def test_van_ban_bi_nuot_khoang_trang_bi_phat_hien():
    dinh = "whichareknownasthenormalequationsfortheleastsquaresproblem "
    assert _ty_le_dinh_chu(dinh * _SO_LAN_LAP_CHO_CO_TRANG) > 0.9


def test_cong_thuc_toan_khong_bi_bao_dong_gia():
    """Công thức viết liền là chuyện bình thường - chỉ đếm CHỮ CÁI nên số và ký hiệu toán
    không được phép làm bật cảnh báo, nếu không mọi trang sách kỹ thuật đều bị đọc lại."""
    cong_thuc = "w = (X^T X)^-1 X^T y  với  sigma^2 = 1/N * sum((y_i - f(x_i))^2) "
    assert _ty_le_dinh_chu(cong_thuc * _SO_LAN_LAP_CHO_CO_TRANG) < 0.05


_NGU_CANH_DINH = (
    "Thebias-variancedecompositionbreaksdowntheexpectedsquaredlossinto"
    "thesquaredbiasandthevariance."
)


def test_bam_ngu_canh_van_bat_duoc_du_ngu_canh_bi_dinh_chu():
    """Đây chính là ca khiến giám khảo 4B chấm 0.0 cho một câu trả lời đúng: nếu phép đo
    này cũng trượt thì không còn gì bác lại được điểm sai đó."""
    cau_tra_loi = (
        "The bias-variance decomposition breaks down the expected squared loss into the "
        "squared bias and the variance [1]."
    )
    assert do_bam_ngu_canh(cau_tra_loi, _NGU_CANH_DINH) > 0.5


def test_cau_tra_loi_bia_thi_khong_bam_ngu_canh():
    assert do_bam_ngu_canh("Paris is the capital city of France.", _NGU_CANH_DINH) == 0.0


def test_dung_mot_nua_bi_muc_thap_nhat_keo_ve_khong():
    """Phải dùng mức của CÂU TỆ NHẤT chứ không phải trung bình: câu trả lời nửa đúng nửa
    bịa vẫn cho trung bình khá cao, đủ để bật cờ nghi ngờ OAN cho giám khảo đang chấm đúng."""
    ngu_canh = "Nhà nước có ba đặc điểm: tính giai cấp, quyền lực công cộng và chủ quyền."
    nua_bia = (
        "Nhà nước có ba đặc điểm: tính giai cấp, quyền lực công cộng và chủ quyền [1]. "
        "Nhà nước Việt Nam được thành lập vào năm 1945 sau Cách mạng Tháng Tám [1]."
    )
    assert do_bam_ngu_canh(nua_bia, ngu_canh) > 0.3
    assert do_bam_ngu_canh_thap_nhat(nua_bia, ngu_canh) == 0.0


def test_cau_noi_ngan_khong_keo_muc_thap_nhat_xuong():
    """"Cụ thể như sau:" không mang nội dung để đối chiếu - tính vào thì mọi câu trả lời có
    gạch đầu dòng đều bị mức thấp nhất bằng 0, phép đo mất hết tác dụng."""
    ngu_canh = "Nhà nước có ba đặc điểm: tính giai cấp, quyền lực công cộng và chủ quyền."
    co_cau_noi = (
        "Cụ thể như sau: "
        "Nhà nước có ba đặc điểm: tính giai cấp, quyền lực công cộng và chủ quyền [1]."
    )
    assert do_bam_ngu_canh_thap_nhat(co_cau_noi, ngu_canh) > 0.5


def _bang_markdown(so_hang: int) -> str:
    dong = ["| Họ và tên | Mã số sinh viên | Nội dung được giao |", "| --- | --- | --- |"]
    dong += [f"| Sinh viên {i} | 235101{i:04d} | Nghiên cứu phần {i} |" for i in range(so_hang)]
    return MOC_BANG_MO + "\n" + "\n".join(dong) + "\n" + MOC_BANG_DONG


def test_moi_manh_cua_bang_lon_deu_giu_dong_tieu_de():
    """Mất dòng tiêu đề là mất đúng thứ khiến bảng là bảng: các ô còn lại thành những giá
    trị trôi nổi không biết thuộc cột nào."""
    cac_manh = _cat_bang_giu_tieu_de(_bang_markdown(40), dem=lambda t: len(t.split()), tran=40)

    assert len(cac_manh) > 1, "bảng này phải bị cắt thì test mới có ý nghĩa"
    for noi_dung, la_bang in cac_manh:
        assert la_bang
        assert "Họ và tên | Mã số sinh viên | Nội dung được giao" in noi_dung
        assert noi_dung.startswith(MOC_BANG_MO) and noi_dung.endswith(MOC_BANG_DONG)


def test_khong_mat_hang_nao_khi_cat():
    cac_manh = _cat_bang_giu_tieu_de(_bang_markdown(40), dem=lambda t: len(t.split()), tran=40)
    gop = "\n".join(m for m, _ in cac_manh)
    for i in range(40):
        assert f"Sinh viên {i} " in gop


def test_o_van_xuoi_dai_tro_ve_dang_van_xuoi_khong_con_dau_gach_dung():
    """Biểu mẫu Word hay có ô "Giới thiệu ý tưởng" chứa cả bài văn. Ép nó ở dạng bảng thì
    mỗi mảnh cắt ra mang theo mấy dấu "|" lạc lõng và bị gắn nhãn "bảng" sai."""
    van_dai = "Trong thời đại kinh tế số " * 60
    khoi = (
        MOC_BANG_MO + "\n| Mục | Nội dung |\n| --- | --- |\n"
        f"| 5. | {van_dai} |\n" + MOC_BANG_DONG
    )
    cac_manh = _cat_bang_giu_tieu_de(khoi, dem=lambda t: len(t.split()), tran=40)

    van_xuoi = [m for m, la_bang in cac_manh if not la_bang]
    assert van_xuoi, "ô văn xuôi dài phải được trả về dạng văn xuôi"
    assert "|" not in van_xuoi[0]


def test_chia_chunk_gan_nhan_dung_cho_bang_va_cho_van_xuoi_trong_o():
    trang = [{
        "nguon": "bieu_mau.docx", "trang": 1,
        "noidung": _bang_markdown(3) + "\n\nĐoạn văn xuôi bình thường nằm ngoài bảng, đủ dài "
                   "để không bị lọc bỏ vì quá ngắn.",
    }]
    cac_chunk = chia_chunk(trang, dem_token_fn=lambda t: len(t.split()))
    cac_loai = {c["loai_noi_dung"] for c in cac_chunk}
    assert "bang" in cac_loai and "van_ban" in cac_loai


def test_cau_noi_doan_trich_khong_lien_quan_khong_bi_tinh_la_trich_dan():
    """Hỏi "đoạn [3] có chứng minh điều này không" cho một câu đang nói "[3] KHÔNG chứa gì
    cả" là đặt sai câu hỏi - giám khảo tất nhiên chấm 0 và kéo Citation accuracy xuống oan."""
    cau_tra_loi = (
        "Đề tài thuộc lĩnh vực Khoa học máy tính theo [1]. "
        "Các phần còn lại [3], [4] không liên quan đến đề tài đang xét."
    )
    theo_so = cau_theo_trich_dan(cau_tra_loi)
    assert set(theo_so) == {1}


def test_cau_khang_dinh_binh_thuong_van_duoc_tinh():
    """Bộ lọc trên phải HẸP: một câu có chữ "không" nhưng vẫn là khẳng định dựa vào nguồn
    thì tuyệt đối không được bỏ qua, nếu không phép đo sẽ tự bỏ sót lỗi thật."""
    cau_tra_loi = "Đề tài không thuộc lĩnh vực Y sinh mà thuộc Khoa học máy tính theo [2]."
    assert set(cau_theo_trich_dan(cau_tra_loi)) == {2}


def test_cau_tu_choi_khong_dan_nguon_thi_bi_loai_khoi_trung_binh():
    """Câu từ chối KHÔNG được kèm nguồn - vừa nói "không có thông tin" vừa chỉ vào một trang
    cụ thể là tự mâu thuẫn. Nên đây không phải lỗi, phải loại khỏi trung bình."""
    import config
    from evaluation.metrics import do_chinh_xac_trich_dan

    ket_qua = do_chinh_xac_trich_dan(
        config.CAU_TU_CHOI["vi"], [{"nguon": "a.pdf", "trang": 1, "noidung": "nội dung"}]
    )
    assert ket_qua["diem"] is None


def test_cau_tra_loi_that_ma_khong_dan_nguon_phai_bi_tinh_0():
    """Đây là lỗi trích dẫn NẶNG NHẤT: người đọc không có cách nào kiểm chứng điều vừa đọc."""
    from evaluation.metrics import do_chinh_xac_trich_dan

    ket_qua = do_chinh_xac_trich_dan(
        "Công nghệ thông tin.", [{"nguon": "a.pdf", "trang": 1, "noidung": "nội dung"}]
    )
    assert ket_qua["diem"] == 0.0


def test_diem_ngoai_thang_bi_loai_khoi_phep_lay_trung_vi(monkeypatch):
    """Điểm giám khảo nằm ngoài thang 0-1 phải bị loại khỏi phép lấy trung vị, vì chỉ vài câu
    như vậy đã đủ kéo lệch hẳn kết quả trung bình."""
    from evaluation import metrics

    cac_diem = iter([100.0, 0.0, 0.0])

    def judge_gia(prompt):
        d = next(cac_diem)
        return {"diem": d, "ly_do": "x", "hop_le": 0.0 <= d <= 1.0}

    monkeypatch.setattr(metrics, "_goi_judge", judge_gia)

    ket_qua = metrics._goi_judge_on_dinh("prompt bất kỳ", so_lan=3)
    assert ket_qua["diem"] == 0.0
    assert ket_qua["so_lan_bi_loai"] == 1
    assert ket_qua["dao_dong_judge"] == 0.0, "dao động phải tính trên mẫu HỢP LỆ, không kể 100.0"


def test_khong_con_diem_hop_le_nao_thi_kep_ve_khoang_va_van_chay_tiep(monkeypatch):
    """Hỏng hết thì phải kẹp để cả lần đánh giá không sập giữa chừng - nhưng đó là dấu hiệu
    model hiểu sai thang điểm một cách hệ thống, nên phải ghi log ở mức error."""
    from evaluation import metrics

    monkeypatch.setattr(
        metrics, "_goi_judge", lambda prompt: {"diem": 100.0, "ly_do": "x", "hop_le": False}
    )
    ket_qua = metrics._goi_judge_on_dinh("prompt bất kỳ", so_lan=3)
    assert ket_qua["diem"] == 1.0


def test_phep_do_cham_mot_lan_cung_duoc_kep(monkeypatch):
    """answer_relevance và từng cặp trích dẫn chỉ chấm 1 lần nên không có cơ chế bỏ phiếu -
    lựa chọn duy nhất còn lại là kẹp, nhưng tuyệt đối không được để lọt 100.0 ra ngoài."""
    from evaluation import metrics

    monkeypatch.setattr(
        metrics, "_goi_judge", lambda prompt: {"diem": 100.0, "ly_do": "x", "hop_le": False}
    )
    assert metrics._goi_judge_kep("prompt")["diem"] == 1.0
    assert metrics.answer_relevance("hỏi gì đó", "trả lời gì đó")["diem"] == 1.0
