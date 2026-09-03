"""Test cho việc viết lại câu hỏi NỐI TIẾP theo ngữ cảnh hội thoại.

Ba nhóm điều cần khoá lại, theo đúng thứ tự quan trọng:

1. CHI PHÍ. Toàn bộ thiết kế hai tầng chỉ có nghĩa nếu tầng đắt hầu như không chạy. Test
   đếm thẳng số lần gọi model, chứ không tin vào việc "chắc là nó không gọi".
2. AN TOÀN. Mọi đường thất bại của tầng 2 (model hỏng, model trả rác, model trả lời thay vì
   viết lại) đều phải lùi về ĐÚNG câu gốc - tức về đúng hành vi của bản chưa có tính năng này.
3. ĐỘ CHÍNH XÁC của tầng nhận diện, đo trên một bộ ca có nhãn thay vì kiểm từng ca lẻ.

Không dùng model thật ở đâu cả: hành vi cần kiểm là logic điều phối, không phải chất lượng
model viết lại (thứ đó đo bằng evaluation/kiem_dinh_viet_lai.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from rag.tiep_noi_hoi_thoai import (
    chuan_bi_truy_van,
    la_cau_hoi_tiep_noi,
    truy_van_ngu_canh,
    viet_lai_cau_hoi,
)


class ClientGia:
    """Thay ollama.Client: trả về chuỗi định sẵn và ĐẾM số lần bị gọi."""

    def __init__(self, tra_ve="", loi=None):
        self.tra_ve = tra_ve
        self.loi = loi
        self.so_lan_goi = 0

    def chat(self, **kwargs):
        self.so_lan_goi += 1
        if self.loi:
            raise self.loi
        return {"message": {"content": self.tra_ve}}


LICH_SU = [
    {"role": "user", "content": "Vi phạm pháp luật gồm những dấu hiệu nào?"},
    {"role": "assistant", "content": "Có bốn dấu hiệu: hành vi trái pháp luật, có lỗi, "
                                     "do chủ thể có năng lực trách nhiệm thực hiện, và xâm "
                                     "hại quan hệ xã hội được pháp luật bảo vệ."},
]


# ======================================================================
# Tầng 1: nhận diện câu nối tiếp (tất định)
# ======================================================================

# Bộ ca có nhãn. Để ở dạng bảng thay vì viết mỗi ca một hàm test, vì thứ đáng biết là TỈ LỆ
# đúng trên cả bộ - một ca lẻ sai không nói lên gì, còn bộ này tụt thì mới là hồi quy thật.
CAC_CA_TIEP_NOI = [
    ("Thế còn dấu hiệu thứ hai thì sao?", True),
    ("Vậy còn trường hợp không có lỗi?", True),
    ("Cái đó khác gì với vi phạm hành chính?", True),
    ("Giải thích thêm đi", True),
    ("Cho ví dụ", True),
    ("Tại sao vậy?", True),
    ("What about the second one?", True),
    ("Tell me more", True),
    ("Còn lại thì sao?", True),
    ("Yếu tố thứ ba là gì?", True),
]

CAC_CA_DOC_LAP = [
    ("Vi phạm pháp luật gồm những dấu hiệu nào?", False),
    ("Nhà nước có những đặc điểm gì?", False),
    ("Trình bày khái niệm quy phạm pháp luật.", False),
    ("What is the Kullback-Leibler divergence?", False),
    ("Mã số sinh viên 2351010039 thuộc về ai?", False),
    ("Tài liệu Bishop nói gì về hồi quy tuyến tính?", False),
    ("How is the late return fee calculated?", False),
]


@pytest.mark.parametrize("cau_hoi,mong_doi", CAC_CA_TIEP_NOI + CAC_CA_DOC_LAP)
def test_nhan_dien_cau_noi_tiep(cau_hoi, mong_doi):
    assert la_cau_hoi_tiep_noi(cau_hoi, LICH_SU) is mong_doi


def test_khong_co_lich_su_thi_khong_bao_gio_la_noi_tiep():
    """Câu đầu tiên của phiên không thể nối tiếp cái gì, dù nó trông giống đến đâu.

    Viết lại nó cũng vô nghĩa - không có ngữ cảnh nào để lấy chủ đề ra."""
    assert la_cau_hoi_tiep_noi("Thế còn cái kia?", []) is False
    assert la_cau_hoi_tiep_noi("Thế còn cái kia?", None) is False


def test_lich_su_chi_co_cau_tra_loi_cung_khong_tinh():
    """Phải có ít nhất một lượt NGƯỜI DÙNG hỏi trước đó thì mới có cái để nối tiếp."""
    chi_tra_loi = [{"role": "assistant", "content": "Xin chào"}]
    assert la_cau_hoi_tiep_noi("Thế còn cái kia?", chi_tra_loi) is False


# ======================================================================
# Chi phí: tầng đắt không được chạy khi không cần
# ======================================================================

def test_cau_hoi_doc_lap_di_thang_khong_ghep_gi():
    """Câu tự đứng được không cần ngữ cảnh, và cũng không được gánh thêm chi phí nào."""
    client = ClientGia(tra_ve="không nên được dùng")
    ket_qua = chuan_bi_truy_van("Nhà nước có những đặc điểm gì?", LICH_SU, client=client)
    assert client.so_lan_goi == 0
    assert ket_qua["cau_hoi_chinh"] == "Nhà nước có những đặc điểm gì?"
    assert ket_qua["ngu_canh_llm"] == ""
    assert ket_qua["la_tiep_noi"] is False


def test_duong_mac_dinh_ghep_ngu_canh_KHONG_goi_model():
    """Đường mặc định phải hoàn toàn TẤT ĐỊNH.

    Đây là lý do chính khiến hướng ghép ngữ cảnh được chọn thay cho query rewriting (§5.58):
    một bước đứng chắn trước toàn bộ truy xuất mà lại không tất định thì mọi metric phía sau
    mất khả năng so sánh giữa hai lần chạy (§5.46)."""
    client = ClientGia(tra_ve="không nên được dùng")
    ket_qua = chuan_bi_truy_van("Thế còn dấu hiệu thứ hai?", LICH_SU, client=client)
    assert client.so_lan_goi == 0
    assert ket_qua["la_tiep_noi"] is True
    assert ket_qua["da_viet_lai"] is False
    # Truy vấn chính mang đủ từ khoá chủ đề mà câu gốc không hề có.
    assert "Vi phạm pháp luật" in ket_qua["cau_hoi_chinh"]
    assert "dấu hiệu thứ hai" in ket_qua["cau_hoi_chinh"]
    # Câu gốc giữ nguyên: giao diện hiện lại đúng thứ người dùng gõ, và truy_xuat() dùng nó
    # làm một nhánh riêng trong RRF.
    assert ket_qua["cau_hoi_goc"] == "Thế còn dấu hiệu thứ hai?"


def test_ngu_canh_cho_llm_chi_chua_cau_hoi_khong_chua_cau_tra_loi():
    """Câu trả lời trước là lời của CHÍNH MODEL, không phải tài liệu. Đưa nó vào prompt như
    một nguồn là mở đúng cánh cửa mà cả hệ thống tồn tại để đóng - model sẽ trích dẫn lại
    lời của mình như thể đó là căn cứ trong tài liệu."""
    ket_qua = chuan_bi_truy_van("Thế còn dấu hiệu thứ hai?", LICH_SU)
    ngu_canh = ket_qua["ngu_canh_llm"]
    assert "Vi phạm pháp luật gồm những dấu hiệu nào?" in ngu_canh
    assert "hành vi trái pháp luật" not in ngu_canh
    assert "KHÔNG PHẢI nguồn thông tin" in ngu_canh


def test_tat_ghep_ngu_canh_thi_ve_dung_hanh_vi_single_turn(monkeypatch):
    monkeypatch.setattr(config, "BAT_TRUY_VAN_NGU_CANH", False)
    ket_qua = chuan_bi_truy_van("Thế còn dấu hiệu thứ hai?", LICH_SU)
    assert ket_qua["cau_hoi_chinh"] == "Thế còn dấu hiệu thứ hai?"
    assert ket_qua["ngu_canh_llm"] == ""


def test_duong_llm_mac_dinh_TAT(monkeypatch):
    """Mặc định tắt là một KẾT QUẢ ÂM TÍNH ĐO ĐƯỢC (§5.58), không phải chưa làm xong."""
    assert config.BAT_VIET_LAI_CAU_HOI is False


def test_bat_duong_llm_thi_ban_viet_lai_thay_lam_truy_van_chinh(monkeypatch):
    monkeypatch.setattr(config, "BAT_VIET_LAI_CAU_HOI", True)
    client = ClientGia(tra_ve="Dấu hiệu thứ hai của vi phạm pháp luật là gì?")
    ket_qua = chuan_bi_truy_van("Thế còn dấu hiệu thứ hai?", LICH_SU, client=client)
    assert client.so_lan_goi == 1
    assert ket_qua["da_viet_lai"] is True
    assert ket_qua["cau_hoi_chinh"] == "Dấu hiệu thứ hai của vi phạm pháp luật là gì?"
    # Bản ghép ngữ cảnh không bị vứt đi mà lùi xuống thành một nhánh phụ - hai cách hiểu
    # cùng được thử, đúng việc RRF sinh ra để làm.
    assert any("Vi phạm pháp luật" in v for v in ket_qua["cac_truy_van_phu"])


# ======================================================================
# An toàn: mọi đường hỏng đều lùi về câu gốc
# ======================================================================

GOC = "Thế còn dấu hiệu thứ hai?"


def test_model_chet_thi_giu_nguyen_cau_goc():
    """Ollama chưa chạy là chuyện rất hay gặp. Một tính năng phụ trợ không được phép biến
    nó thành lỗi của cả lượt hỏi - người dùng sẽ gặp thông báo tử tế ở bước sinh câu trả lời,
    nơi có hướng dẫn xử lý đầy đủ."""
    assert viet_lai_cau_hoi(GOC, LICH_SU, ClientGia(loi=ConnectionError("chưa bật"))) == GOC


def test_model_tra_rac_thi_giu_nguyen_cau_goc():
    assert viet_lai_cau_hoi(GOC, LICH_SU, ClientGia(tra_ve="")) == GOC
    assert viet_lai_cau_hoi(GOC, LICH_SU, ClientGia(tra_ve="-----")) == GOC


def test_ban_viet_lai_qua_dai_bi_loai():
    """Bản viết lại dài bất thường gần như luôn là model đã tự trả lời câu hỏi, hoặc nhồi
    cả đoạn ngữ cảnh vào. Lúc đó câu gốc an toàn hơn."""
    qua_dai = "Dấu hiệu thứ hai là có lỗi, " + "và điều này có nghĩa là " * 20 + "?"
    assert viet_lai_cau_hoi(GOC, LICH_SU, ClientGia(tra_ve=qua_dai)) == GOC


def test_ban_viet_lai_ngan_hon_cau_goc_bi_loai():
    """Viết lại phải LÀM RÕ THÊM. Ngắn đi nghĩa là đã mất thông tin, không phải làm rõ."""
    assert viet_lai_cau_hoi(GOC, LICH_SU, ClientGia(tra_ve="Dấu hiệu?")) == GOC


def test_model_tra_loi_thay_vi_viet_lai_bi_loai():
    """Ca đã gặp thật với model nhỏ: nó bỏ qua chỉ dẫn và trả lời luôn câu hỏi.

    Nếu nhận bản này thì hệ thống sẽ đi truy xuất bằng một câu KHẲNG ĐỊNH, và tệ hơn, câu
    khẳng định đó do chính model bịa ra chứ không lấy từ tài liệu."""
    tra_loi = ("Dấu hiệu thứ hai của vi phạm pháp luật là yếu tố lỗi của chủ thể thực hiện "
               "hành vi trái pháp luật đó")
    assert viet_lai_cau_hoi(GOC, LICH_SU, ClientGia(tra_ve=tra_loi)) == GOC


def test_boc_duoc_nhan_va_ngoac_kep_model_hay_them():
    client = ClientGia(tra_ve='Câu hỏi độc lập: "Dấu hiệu thứ hai của vi phạm pháp luật là gì?"')
    assert viet_lai_cau_hoi(GOC, LICH_SU, client) == (
        "Dấu hiệu thứ hai của vi phạm pháp luật là gì?"
    )


def test_ban_viet_lai_trung_cau_goc_khong_bi_bao_la_da_viet_lai():
    """Prompt dặn model chép nguyên văn khi câu đã độc lập. Báo "đã viết lại" cho một thay
    đổi vô hình chỉ làm nhiễu giao diện."""
    client = ClientGia(tra_ve="  Thế   còn dấu hiệu thứ hai?  ")
    ket_qua = chuan_bi_truy_van(GOC, LICH_SU, client=client)
    assert ket_qua["da_viet_lai"] is False


# ======================================================================
# Ngữ cảnh đưa vào prompt
# ======================================================================

def test_chi_lay_vai_luot_gan_nhat_va_cat_ngan_cau_tra_loi(monkeypatch):
    """Hai chốt chặn chi phí/chất lượng của prompt viết lại, kiểm bằng cách soi đúng chuỗi
    prompt đã gửi đi."""
    monkeypatch.setattr(config, "SO_LUOT_NGU_CANH", 1)
    monkeypatch.setattr(config, "DO_DAI_TRA_LOI_TRONG_NGU_CANH", 20)

    lich_su_dai = [
        {"role": "user", "content": "CÂU HỎI RẤT CŨ không được đưa vào"},
        {"role": "assistant", "content": "trả lời rất cũ"},
        {"role": "user", "content": "Vi phạm pháp luật gồm những dấu hiệu nào?"},
        {"role": "assistant", "content": "A" * 500},
    ]

    class ClientBatPrompt(ClientGia):
        def chat(self, **kwargs):
            self.prompt = kwargs["messages"][0]["content"]
            return super().chat(**kwargs)

    client = ClientBatPrompt(tra_ve="Dấu hiệu thứ hai của vi phạm pháp luật là gì?")
    viet_lai_cau_hoi(GOC, lich_su_dai, client)

    assert "CÂU HỎI RẤT CŨ" not in client.prompt
    assert "Vi phạm pháp luật gồm những dấu hiệu nào?" in client.prompt
    assert "A" * 21 not in client.prompt  # câu trả lời cũ đã bị cắt còn 20 ký tự


# ======================================================================
# Ghép ngữ cảnh: chi tiết quyết định chất lượng truy vấn
# ======================================================================

def test_chi_ghep_CAU_HOI_truoc_khong_ghep_cau_tra_loi():
    """Câu trả lời của lượt trước dài hơn câu hỏi hàng chục lần. Ghép nó vào thì vector truy
    vấn bị chính nội dung câu trả lời cũ lấn át - và nội dung đó đã nằm sẵn trong tài liệu,
    nên nhánh này sẽ chỉ kéo về đúng những đoạn vừa dùng ở lượt trước, không tìm được phần
    MỚI mà người dùng đang hỏi."""
    ghep = truy_van_ngu_canh("Thế còn dấu hiệu thứ hai?", LICH_SU)
    assert "Vi phạm pháp luật gồm những dấu hiệu nào?" in ghep
    assert "hành vi trái pháp luật" not in ghep


def test_cau_hoi_hien_tai_dat_o_CUOI():
    """Câu hiện tại là thứ cần được nhấn; phần ghép thêm chỉ đóng vai trò nêu chủ đề."""
    ghep = truy_van_ngu_canh("Thế còn dấu hiệu thứ hai?", LICH_SU)
    assert ghep.endswith("Thế còn dấu hiệu thứ hai?")


def test_ghep_theo_dung_thu_tu_thoi_gian(monkeypatch):
    monkeypatch.setattr(config, "SO_LUOT_NGU_CANH", 3)
    lich_su = [
        {"role": "user", "content": "CÂU MỘT"},
        {"role": "assistant", "content": "trả lời một"},
        {"role": "user", "content": "CÂU HAI"},
        {"role": "assistant", "content": "trả lời hai"},
    ]
    assert truy_van_ngu_canh("CÂU BA", lich_su) == "CÂU MỘT CÂU HAI CÂU BA"


def test_chi_lay_so_luot_gan_nhat(monkeypatch):
    """Nhồi thêm lịch sử không tốt hơn: truy vấn càng dài thì chủ đề của câu hỏi hiện tại
    càng bị các chủ đề cũ pha loãng."""
    monkeypatch.setattr(config, "SO_LUOT_NGU_CANH", 1)
    lich_su = [
        {"role": "user", "content": "CÂU RẤT CŨ"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "CÂU LIỀN TRƯỚC"},
        {"role": "assistant", "content": "..."},
    ]
    ghep = truy_van_ngu_canh("CÂU MỚI", lich_su)
    assert "CÂU RẤT CŨ" not in ghep
    assert ghep == "CÂU LIỀN TRƯỚC CÂU MỚI"
