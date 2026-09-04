"""Test cho ngân sách THÍCH ỨNG lúc truy vấn: rerank, num_predict, nén ngữ cảnh.

Bối cảnh: bản trước cấp ngân sách TỐI ĐA cho mọi câu hỏi - 30 ứng viên cross-encoder, trần
sinh 12000 token, cửa sổ ngữ cảnh tính theo mức dự phòng lớn nhất. Câu "Overfitting là gì?"
không dùng hết phần nào trong số đó nhưng vẫn phải chờ nó.

Ràng buộc quan trọng nhất mà bộ test này canh giữ nằm ở chiều NGƯỢC LẠI với tối ưu: khi ngữ
cảnh vượt trần cửa sổ, hệ thống KHÔNG được hạ num_ctx. Hạ num_ctx không làm prompt ngắn đi,
nó chỉ chuyển quyền quyết định cắt chỗ nào từ ta sang Ollama - mà Ollama luôn cắt từ ĐẦU
phần ngữ cảnh, tức xoá đúng đoạn trích liên quan nhất. Đó là một bug đã xảy ra thật trong
project này (xem config.OLLAMA_NUM_CTX) và tuyệt đối không được tái lập dưới danh nghĩa
"tối ưu tốc độ".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from rag.rag_pipeline import (
    _tinh_num_ctx,
    _uoc_luong_so_token,
    la_cau_hoi_phuc_tap,
    nen_ngu_canh,
    ngan_sach_token_ngu_canh,
)


# ======================================================================
# 1. Nhận diện câu hỏi đơn giản / phức tạp
# ======================================================================

@pytest.mark.parametrize(
    "cau_hoi, phuc_tap",
    [
        ("Overfitting là gì?", False),
        ("Định nghĩa cây quyết định", False),
        ("What is KNN?", False),
        ("So sánh KNN với Naive Bayes", True),
        ("Liệt kê các bước chuẩn bị dữ liệu", True),
        ("Vì sao cần chuẩn hoá dữ liệu trước khi gom cụm?", True),
        ("Compare K-means and DBSCAN", True),
        ("Phân biệt học có giám sát và học không giám sát", True),
        (
            "Trình bày quy trình khai phá dữ liệu từ bước thu thập cho tới bước đánh giá "
            "mô hình và triển khai",
            True,
        ),
    ],
)
def test_nhan_dien_do_phuc_tap(cau_hoi, phuc_tap):
    assert la_cau_hoi_phuc_tap(cau_hoi) is phuc_tap


def test_cau_hoi_kiem_chung_luon_duoc_cap_ngan_sach_day_du():
    """Câu kiểm chứng bật cả chế độ suy luận và buộc trích nguyên văn căn cứ (§5.29) -
    cắt ngân sách của nó là cắt đúng loại câu hỏi tốn kém nhất về mặt lập luận."""
    assert la_cau_hoi_phuc_tap("Đúng hay sai: KNN là thuật toán học không giám sát?")


def test_tat_thich_ung_thi_moi_cau_hoi_deu_duoc_ngan_sach_day_du(monkeypatch):
    """Đường lui về hành vi cũ, để đo đối chứng."""
    monkeypatch.setattr(config, "BAT_NGAN_SACH_THICH_UNG", False)
    assert la_cau_hoi_phuc_tap("Overfitting là gì?")


# ======================================================================
# 2. num_ctx: KHÔNG BAO GIỜ bị hạ để tiết kiệm
# ======================================================================

def test_num_ctx_khong_bao_gio_thap_hon_cau_hinh():
    for so_token in (10, 500, 5000, 50000):
        for num_predict in (100, 3000, 12000):
            assert _tinh_num_ctx(so_token, num_predict) >= config.OLLAMA_NUM_CTX


def test_num_ctx_van_no_ra_khi_prompt_dai():
    nho = _tinh_num_ctx(100, config.OLLAMA_NUM_PREDICT)
    lon = _tinh_num_ctx(config.OLLAMA_NUM_CTX * 2, config.OLLAMA_NUM_PREDICT)
    assert lon > nho
    assert lon <= max(config.OLLAMA_NUM_CTX_TOI_DA, config.OLLAMA_NUM_CTX)


def test_num_predict_nho_khong_lam_num_ctx_lon_hon():
    """Trần sinh thấp chỉ được phép GIỮ NGUYÊN hoặc GIẢM cỡ cửa sổ, không bao giờ tăng."""
    for so_token in (1000, 8000, 20000):
        assert _tinh_num_ctx(so_token, 3000) <= _tinh_num_ctx(so_token, 12000)


def test_num_predict_lon_hon_du_phong_thi_giu_nguyen_hanh_vi_cu():
    """num_predict mặc định (12000) lớn hơn OLLAMA_DU_PHONG_TOKEN_SINH (4000), nên với câu
    hỏi phức tạp thì công thức phải cho ra ĐÚNG kết quả của bản chưa có thích ứng."""
    for so_token in (100, 3000, 9000):
        assert _tinh_num_ctx(so_token, config.OLLAMA_NUM_PREDICT) == _tinh_num_ctx(so_token)


# ======================================================================
# 2b. Trần token sinh: KHÔNG được hạ theo độ phức tạp câu hỏi
# ======================================================================

def test_moi_cau_hoi_deu_duoc_tron_ngan_sach_sinh():
    """Hồi quy đã gây lỗi THẬT cho người dùng: câu trả lời đứt giữa chừng ở chữ "và".

    Bản trước hạ num_predict xuống 3000 cho câu hỏi "đơn giản". Nhưng `num_predict` giới hạn
    SUY LUẬN + CÂU TRẢ LỜI cộng lại, mà riêng chuỗi suy luận của qwen3 đã ngốn 2.000-4.000
    token — nên 3000 gần như không chừa gì cho câu trả lời. Đo trên đúng câu hỏi gây lỗi:
    num_predict=3000 cho ra 569 ký tự (đứt), num_predict=12000 cho ra 1012 ký tự (đủ).

    Test này khoá lại: dù câu hỏi ngắn tới đâu, trần sinh vẫn phải là trần đầy đủ.
    """
    import inspect

    from rag import rag_pipeline

    ma_nguon = inspect.getsource(rag_pipeline.RagPipeline.sinh_cau_tra_loi_theo_luong)
    assert "num_predict = config.OLLAMA_NUM_PREDICT" in ma_nguon, (
        "trần sinh phải là hằng số, không được rẽ nhánh theo độ phức tạp câu hỏi"
    )
    assert not hasattr(config, "NUM_PREDICT_CAU_HOI_DON_GIAN"), (
        "tham số này đã bị gỡ bỏ vì nó cắt cụt câu trả lời - đừng thêm lại"
    )


def test_ngan_sach_sinh_luon_du_cho_ca_suy_luan_lan_cau_tra_loi():
    """Trần sinh phải lớn hơn hẳn chuỗi suy luận dài nhất đã quan sát được (7.232 token).

    Con số 7.232 không phải phòng xa: nó đo được trên một câu hỏi bình thường. Độ dài suy
    luận KHÔNG tương quan với độ dài câu hỏi, nên trần phải phủ được ca xấu nhất chứ không
    phải ca trung bình.
    """
    SUY_LUAN_DAI_NHAT_DA_THAY = 7232
    assert config.OLLAMA_NUM_PREDICT > SUY_LUAN_DAI_NHAT_DA_THAY * 1.5


# ======================================================================
# 3. Nén ngữ cảnh
# ======================================================================

def _chunk(i: int, so_ky_tu: int) -> dict:
    return {
        "nguon": "bai.pdf",
        "trang": i,
        "noidung": f"doan {i} " + "x" * so_ky_tu,
        "diem_similarity": 1.0 - i * 0.1,
    }


def test_khong_dung_toi_khi_ngu_canh_da_vua_ngan_sach():
    cac_chunk = [_chunk(i, 200) for i in range(4)]
    assert nen_ngu_canh(cac_chunk, 100000) is cac_chunk


def test_bo_tu_doan_xep_hang_thap_nhat_len():
    """Bỏ từ CUỐI là điều kiện để số thứ tự [1], [2]... của các đoạn còn lại không lệch -
    lệch một bậc là trích dẫn trỏ sai nguồn."""
    cac_chunk = [_chunk(i, 2000) for i in range(6)]
    ngan_sach = _uoc_luong_so_token(cac_chunk[0]["noidung"]) * 2 + 100

    giu = nen_ngu_canh(cac_chunk, ngan_sach)
    assert 0 < len(giu) < len(cac_chunk)
    assert [c["trang"] for c in giu] == list(range(len(giu))), "phải giữ đúng tiền tố đầu danh sách"


def test_van_gui_duoc_mot_doan_da_cat_khi_ngan_sach_qua_hep():
    """Thà đưa nửa đầu đoạn tốt nhất còn hơn không đưa gì - không đoạn nào thì LLM từ chối."""
    cac_chunk = [_chunk(0, 20000)]
    giu = nen_ngu_canh(cac_chunk, 300)

    assert len(giu) == 1
    assert len(giu[0]["noidung"]) < len(cac_chunk[0]["noidung"])
    assert "đoạn bị cắt bớt" in giu[0]["noidung"]


def test_nen_khong_sua_danh_sach_goc():
    """Danh sách gốc còn dùng để hiển thị trích dẫn cho người đọc - phải giữ nguyên văn."""
    cac_chunk = [_chunk(0, 20000)]
    goc = cac_chunk[0]["noidung"]
    nen_ngu_canh(cac_chunk, 300)
    assert cac_chunk[0]["noidung"] == goc


def test_tat_nen_thi_giu_nguyen_moi_doan(monkeypatch):
    monkeypatch.setattr(config, "BAT_NEN_NGU_CANH", False)
    cac_chunk = [_chunk(i, 5000) for i in range(6)]
    assert nen_ngu_canh(cac_chunk, 100) is cac_chunk


def test_ngan_sach_ngu_canh_tru_du_phan_co_dinh():
    tran = max(config.OLLAMA_NUM_CTX_TOI_DA, config.OLLAMA_NUM_CTX)
    ngan_sach = ngan_sach_token_ngu_canh(config.OLLAMA_NUM_PREDICT, 1000)
    assert ngan_sach == tran - config.OLLAMA_DU_PHONG_TOKEN_SINH - 1000
    # Phần cố định lớn hơn -> ngân sách cho đoạn trích nhỏ đi tương ứng, không âm thầm bù trừ.
    assert ngan_sach_token_ngu_canh(config.OLLAMA_NUM_PREDICT, 5000) == ngan_sach - 4000
