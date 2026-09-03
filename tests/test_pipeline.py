"""Test cho phần lõi của luồng Query: dựng đoạn trích, chặn ngưỡng, nhận diện câu hỏi
kiểm chứng, chọn trích dẫn hiển thị.

Cố ý KHÔNG dùng model embedding thật: các hành vi cần kiểm ở đây (mở rộng ngữ cảnh tới
đâu, có chặn được câu hỏi lạc đề không, trần số đoạn mỗi trang) đều là logic thuần tuý,
không phụ thuộc chất lượng model. Thay model bằng vector dựng tay giúp test chạy trong
mili giây và cho phép đặt điểm similarity về đúng con số muốn kiểm, thay vì phải chiều
theo con số model tình cờ trả ra.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import config
from rag.citation import loc_theo_tham_chieu
from rag.rag_pipeline import (
    RagPipeline,
    _noi_lien_mach,
    _phat_hien_ngon_ngu,
    la_cau_hoi_kiem_chung,
)
from rag.vector_store import VectorStore

SO_CHIEU = 4


class EmbeddingGia:
    """Thay cho EmbeddingService: luôn trả về đúng 1 vector câu hỏi đã định sẵn."""

    def __init__(self, vector_cau_hoi):
        self.vector = np.array([vector_cau_hoi], dtype="float32")

    def encode_cau_hoi(self, texts):
        return self.vector


def _vector_don_vi(chi_so_truc: int) -> list:
    v = [0.0] * SO_CHIEU
    v[chi_so_truc] = 1.0
    return v


def _tao_store(cac_chunk, cac_vector) -> VectorStore:
    store = VectorStore(dimension=SO_CHIEU)
    store.them(np.array(cac_vector, dtype="float32"), cac_chunk)
    return store


def _chunk(nguon: str, trang: int, vi_tri: int, noidung: str) -> dict:
    return {"chunk_id": f"{nguon}-{trang}-{vi_tri}", "nguon": nguon, "trang": trang,
            "vi_tri": vi_tri, "noidung": noidung}


@pytest.fixture(autouse=True)
def tat_tim_kiem_tu_khoa(monkeypatch):
    """Các test dưới đây kiểm phần dựng đoạn trích / ngưỡng, muốn thứ hạng do đúng vector
    quyết định. BM25 được kiểm riêng ở test_lexical_search.py."""
    monkeypatch.setattr(config, "TRONG_SO_BM25", 0.0)


# ======================================================================
# Dựng đoạn trích quanh chunk khớp ("small-to-big")
# ======================================================================

def test_doan_trich_lay_dung_chunk_khop_lam_neo():
    """doan_khop phải là chunk thật sự khớp câu hỏi - đây là thứ citation.py hiển thị,
    và là chỗ bản cũ làm sai (cắt 400 ký tự đầu của cả trang đã gộp)."""
    cac_chunk = [_chunk("a.pdf", 1, i, f"Đoạn số {i} nói về chủ đề {i}. " * 4) for i in range(6)]
    cac_vector = [_vector_don_vi(1) for _ in cac_chunk]
    cac_vector[3] = _vector_don_vi(0)  # chỉ chunk số 3 khớp câu hỏi

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    ket_qua = pipeline.truy_xuat("câu hỏi bất kỳ", top_k=1)

    assert len(ket_qua) == 1
    assert ket_qua[0]["doan_khop"] == cac_chunk[3]["noidung"]
    assert ket_qua[0]["trang"] == 1


def test_doan_trich_mo_rong_sang_chunk_lien_ke_cung_trang():
    """Lý do ban đầu của việc gộp: một câu/đoạn liệt kê bị ranh giới chunk cắt ngang phải
    được nối lại, nếu không câu trả lời sẽ cụt đúng chỗ bị cắt."""
    cac_chunk = [
        _chunk("a.pdf", 1, 0, "Phần mở đầu của trang."),
        _chunk("a.pdf", 1, 1, "Các đặc trưng của nhà nước bao gồm: chủ quyền quốc gia và"),
        _chunk("a.pdf", 1, 2, "quyền ban hành pháp luật, quyền thu thuế."),
    ]
    cac_vector = [_vector_don_vi(1), _vector_don_vi(0), _vector_don_vi(1)]

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    noidung = pipeline.truy_xuat("đặc trưng của nhà nước", top_k=1)[0]["noidung"]

    # Chunk số 2 không tự nó đủ giống câu hỏi để lọt top, nhưng phải được kéo vào cùng.
    assert "quyền thu thuế" in noidung
    assert "chủ quyền quốc gia" in noidung


def test_doan_trich_khong_nuot_ca_trang_khi_trang_qua_dai(monkeypatch):
    """Đây là lỗi gốc với tài liệu dài: gộp nguyên trang ~2000 ký tự khiến ngữ cảnh loãng.
    Ngân sách ký tự phải chặn được điều đó."""
    monkeypatch.setattr(config, "NGAN_SACH_KY_TU_MOI_DOAN", 200)
    cac_chunk = [_chunk("a.pdf", 1, i, f"Nội dung khá dài của đoạn {i}. " * 5) for i in range(10)]
    cac_vector = [_vector_don_vi(1)] * 10
    cac_vector[5] = _vector_don_vi(0)

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    noidung = pipeline.truy_xuat("câu hỏi", top_k=1)[0]["noidung"]

    do_dai_ca_trang = sum(len(c["noidung"]) for c in cac_chunk)
    assert len(noidung) < do_dai_ca_trang / 2


def test_trang_ngan_van_duoc_lay_tron_ven():
    """Tài liệu ngắn (slide, trang thưa chữ) phải giữ nguyên hành vi cũ - cả trang vẫn lọt
    trong ngân sách, không được vì tối ưu cho tài liệu dài mà làm hỏng tài liệu ngắn."""
    cac_chunk = [_chunk("slide.pptx", 2, i, f"Ý thứ {i} của slide.") for i in range(3)]
    cac_vector = [_vector_don_vi(1), _vector_don_vi(0), _vector_don_vi(1)]

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    noidung = pipeline.truy_xuat("ý gì", top_k=1)[0]["noidung"]

    for c in cac_chunk:
        assert c["noidung"] in noidung


def test_tran_so_doan_moi_trang(monkeypatch):
    """Với tài liệu dài, các chunk liền kề cùng 1 trang có điểm sát nhau nên rất dễ chiếm
    sạch TOP_K suất, đẩy hết các trang liên quan khác ra ngoài.

    Điều kiện để trần được áp: ứng viên phải TRẢI RA đủ nhiều trang (>= top_k trang) - xem
    test kế tiếp cho nhánh ngược lại."""
    monkeypatch.setattr(config, "SO_DOAN_TOI_DA_MOI_TRANG", 2)
    monkeypatch.setattr(config, "NGAN_SACH_KY_TU_MOI_DOAN", 60)
    monkeypatch.setattr(config, "MO_RONG_QUA_RANH_GIOI_TRANG", False)
    cac_chunk = [_chunk("a.pdf", 1, i, f"Đoạn {i} cùng một trang duy nhất.") for i in range(8)]
    cac_chunk += [_chunk("a.pdf", t, 0, f"Nội dung của trang thứ {t}.") for t in range(2, 8)]
    cac_vector = [_vector_don_vi(0)] * len(cac_chunk)

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    ket_qua = pipeline.truy_xuat("câu hỏi", top_k=5)

    assert sum(1 for d in ket_qua if d["trang"] == 1) <= 2
    assert any(d["trang"] != 1 for d in ket_qua), "trang khác vẫn phải có suất"


def test_tran_moi_trang_duoc_noi_khi_moi_thu_nam_tren_mot_trang(monkeypatch):
    """Trần chỉ có ý nghĩa khi CÓ nhiều trang để phân bổ.

    Với câu hỏi mà toàn bộ câu trả lời nằm gọn trong một trang (một mục định nghĩa, một
    bảng tiêu chí, một quy trình), trần cứng chỉ cho lấy 2 đoạn từ đúng trang chứa câu trả
    lời và đẩy các suất còn lại cho những trang kém liên quan - ngữ cảnh vừa THIẾU phần
    đúng vừa LOÃNG vì phần sai."""
    monkeypatch.setattr(config, "SO_DOAN_TOI_DA_MOI_TRANG", 2)
    monkeypatch.setattr(config, "NGAN_SACH_KY_TU_MOI_DOAN", 60)
    monkeypatch.setattr(config, "MO_RONG_QUA_RANH_GIOI_TRANG", False)
    cac_chunk = [_chunk("a.pdf", 1, i, f"Đoạn {i} cùng một trang duy nhất.") for i in range(8)]
    cac_chunk.append(_chunk("a.pdf", 2, 0, "Nội dung của trang thứ hai."))
    cac_vector = [_vector_don_vi(0)] * len(cac_chunk)

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    ket_qua = pipeline.truy_xuat("câu hỏi", top_k=5)

    assert sum(1 for d in ket_qua if d["trang"] == 1) > 2, (
        "chỉ có 2 trang ứng viên mà vẫn áp trần thì 3 suất còn lại bỏ không"
    )


# ======================================================================
# Mở rộng đoạn trích QUA ranh giới trang
# ======================================================================

def test_doan_trich_noi_lai_dinh_nghia_bi_ranh_gioi_trang_cat_doi():
    """Lỗi chỉ lộ ra trên PDF văn bản chảy liên tục: một định nghĩa bắt đầu cuối trang 12
    và kết thúc đầu trang 13 thì chunk neo nằm ở cuối trang 12, mở rộng sang phải chạm hết
    mảng của trang rồi dừng - phần còn thiếu KHÔNG BAO GIỜ được nối lại.

    Corpus cũ nhiều slide (mỗi slide tự đóng) nên không ai thấy; corpus mới nhiều PDF văn
    xuôi thì lộ ngay."""
    cac_chunk = [
        _chunk("giaotrinh.pdf", 12, 0, "Phần đầu trang mười hai, nói chuyện khác."),
        _chunk("giaotrinh.pdf", 12, 1, "Quy phạm pháp luật là quy tắc xử sự chung do"),
        _chunk("giaotrinh.pdf", 13, 0, "nhà nước ban hành và bảo đảm thực hiện."),
    ]
    cac_vector = [_vector_don_vi(1), _vector_don_vi(0), _vector_don_vi(1)]

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    doan = pipeline.truy_xuat("quy phạm pháp luật là gì", top_k=1)[0]

    assert "nhà nước ban hành" in doan["noidung"], "phần nằm ở trang sau phải được nối lại"
    assert doan["trang"] == 12, "trích dẫn vẫn phải ghi trang của chunk NEO"
    assert doan["cac_trang"] == [12, 13]


def test_mo_rong_khong_vuot_qua_so_trang_cho_phep(monkeypatch):
    """Chốt chặn cho SLIDE: không có nó, một slide thưa chữ sẽ hút thêm vài slide xung
    quanh cho đầy ngân sách ký tự - tức tái tạo lại đúng lỗi "ngữ cảnh loãng" mà việc bỏ
    cách gộp-nguyên-trang đã sửa."""
    monkeypatch.setattr(config, "SO_TRANG_TOI_DA_MO_RONG", 1)
    cac_chunk = [_chunk("slide.pptx", t, 0, f"Nội dung ngắn của slide số {t}.")
                 for t in range(1, 6)]
    cac_vector = [_vector_don_vi(1)] * 5
    cac_vector[2] = _vector_don_vi(0)  # neo ở slide 3

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    doan = pipeline.truy_xuat("hỏi gì đó", top_k=1)[0]

    assert doan["cac_trang"] == [2, 3, 4]
    assert "slide số 1" not in doan["noidung"]
    assert "slide số 5" not in doan["noidung"]


def test_tat_mo_rong_xuyen_trang_thi_ve_dung_hanh_vi_cu(monkeypatch):
    """Công tắc phải thật sự tắt được - đây là đường lui khi corpus toàn slide và việc mở
    rộng xuyên trang đo ra là có hại."""
    monkeypatch.setattr(config, "MO_RONG_QUA_RANH_GIOI_TRANG", False)
    cac_chunk = [
        _chunk("giaotrinh.pdf", 12, 0, "Quy phạm pháp luật là quy tắc xử sự chung do"),
        _chunk("giaotrinh.pdf", 13, 0, "nhà nước ban hành và bảo đảm thực hiện."),
    ]
    pipeline = RagPipeline(
        EmbeddingGia(_vector_don_vi(0)),
        _tao_store(cac_chunk, [_vector_don_vi(0), _vector_don_vi(1)]),
    )
    doan = pipeline.truy_xuat("quy phạm pháp luật là gì", top_k=1)[0]

    assert "nhà nước ban hành" not in doan["noidung"]
    assert doan["cac_trang"] == [12]


# ======================================================================
# Ngưỡng TƯƠNG ĐỐI (so với đoạn tốt nhất của chính lượt đó)
# ======================================================================

def test_nguong_tuong_doi_loai_doan_kem_hon_han_doan_tot_nhat(monkeypatch):
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)
    monkeypatch.setattr(config, "TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT", 0.78)
    monkeypatch.setattr(config, "MO_RONG_QUA_RANH_GIOI_TRANG", False)
    cac_chunk = [_chunk("a.pdf", t, 0, f"Đoạn nội dung của trang {t}, đủ dài để giữ lại.")
                 for t in (1, 2)]
    # cos = 1.0 và 0.6 (0.6 < 1.0 * 0.78) -> đoạn thứ hai phải bị loại.
    cac_vector = [_vector_don_vi(0), [0.6, 0.8, 0.0, 0.0]]

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    ket_qua = pipeline.truy_xuat("câu hỏi", top_k=5)

    assert [d["trang"] for d in ket_qua] == [1]


def test_nguong_tuong_doi_giu_lai_ca_nhom_khi_diem_thap_deu(monkeypatch):
    """Đây là lý do tồn tại của tầng lọc này. Cosine của E5 không phải thang đo tuyệt đối:
    một corpus mới (nhiều bảng, nhiều công thức) dịch cả phân bố xuống, và ngưỡng cố định
    hiệu chỉnh trên corpus cũ sẽ cắt sạch dù các đoạn vẫn liên quan NGANG NHAU."""
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.50)
    monkeypatch.setattr(config, "TY_LE_GIU_SO_VOI_DIEM_CAO_NHAT", 0.78)
    monkeypatch.setattr(config, "MO_RONG_QUA_RANH_GIOI_TRANG", False)
    cac_chunk = [_chunk("a.pdf", t, 0, f"Đoạn nội dung của trang {t}, đủ dài để giữ lại.")
                 for t in (1, 2, 3)]
    # Cả ba đều ~0.6-0.64: dưới ngưỡng tuyệt đối CŨ (0.70) nhưng sát nhau, nên phải giữ cả.
    cac_vector = [[0.64, 0.768, 0.0, 0.0], [0.62, 0.785, 0.0, 0.0], [0.60, 0.80, 0.0, 0.0]]

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    ket_qua = pipeline.truy_xuat("câu hỏi", top_k=5)

    assert len(ket_qua) == 3, "cả nhóm liên quan ngang nhau, không được cắt bớt"


# ======================================================================
# Ngưỡng chặn "không có thông tin"
# ======================================================================

def test_cau_hoi_lac_de_bi_chan_truoc_khi_goi_llm(monkeypatch):
    """Đường dẫn chính khiến hệ thống "xác nhận" một khẳng định sai: tài liệu không hề nói
    về chuyện đó, nhưng vẫn đưa vài đoạn lạc đề cho LLM rồi để nó ghép ra câu nghe hợp lý."""
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.5)
    cac_chunk = [_chunk("a.pdf", 1, 0, "Nội dung về pháp luật đại cương.")]
    pipeline = RagPipeline(
        EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, [_vector_don_vi(1)])
    )

    assert pipeline.truy_xuat("chủ đề hoàn toàn khác") == []
    # Không có đoạn nào -> trả lời từ chối NGAY, không gọi Ollama (nên test này không cần
    # Ollama đang chạy).
    assert pipeline.sinh_cau_tra_loi("chủ đề hoàn toàn khác", []) == (
        "Không tìm thấy thông tin trong tài liệu."
    )


def test_index_rong_tra_ve_rong():
    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), VectorStore(dimension=SO_CHIEU))
    assert pipeline.truy_xuat("bất kỳ") == []


# ======================================================================
# Nhận diện câu hỏi kiểm chứng khẳng định
# ======================================================================

@pytest.mark.parametrize(
    "cau_hoi",
    [
        "Pháp luật ra đời trước nhà nước, đúng không?",
        "Có phải nhà nước do giai cấp thống trị lập ra không?",
        "Theo tôi thì pháp luật và đạo đức là một.",
        "Tôi nhớ là có 5 hình thức nhà nước, phải không?",
        "Is it true that law existed before the state?",
        "The state creates law, right?",
    ],
)
def test_nhan_dien_cau_hoi_kiem_chung(cau_hoi):
    assert la_cau_hoi_kiem_chung(cau_hoi)


@pytest.mark.parametrize(
    "cau_hoi",
    [
        "Nhà nước là gì?",
        "Trình bày các đặc trưng của pháp luật.",
        "What are the characteristics of law?",
    ],
)
def test_cau_hoi_thuong_khong_bi_nham_thanh_kiem_chung(cau_hoi):
    assert not la_cau_hoi_kiem_chung(cau_hoi)


# ======================================================================
# Nối chunk chồng lấn + chọn trích dẫn hiển thị
# ======================================================================

def test_noi_lien_mach_bo_phan_lap_do_overlap():
    truoc = "Nhà nước là tổ chức quyền lực chính trị đặc biệt của xã hội có giai cấp."
    sau = "đặc biệt của xã hội có giai cấp. Nhà nước có chủ quyền quốc gia."
    ket_qua = _noi_lien_mach(truoc, sau)

    assert ket_qua.count("đặc biệt của xã hội có giai cấp") == 1
    assert ket_qua.endswith("Nhà nước có chủ quyền quốc gia.")


def test_noi_lien_mach_khong_chong_lap_thi_giu_nguyen_ca_hai():
    ket_qua = _noi_lien_mach("Câu thứ nhất.", "Câu thứ hai.")
    assert "Câu thứ nhất." in ket_qua and "Câu thứ hai." in ket_qua


def test_trich_dan_bam_theo_so_ma_cau_tra_loi_tham_chieu():
    """Bản cũ luôn hiển thị đoạn có điểm similarity cao nhất, không phải đoạn LLM đã dùng -
    nguyên nhân khiến trích dẫn lệch với nội dung câu trả lời."""
    cac_chunk = [
        {"nguon": "a.pdf", "trang": 10, "noidung": "x", "doan_khop": "A", "diem_similarity": 0.9},
        {"nguon": "a.pdf", "trang": 20, "noidung": "y", "doan_khop": "B", "diem_similarity": 0.8},
        {"nguon": "b.pdf", "trang": 30, "noidung": "z", "doan_khop": "C", "diem_similarity": 0.7},
    ]
    trich_dan = loc_theo_tham_chieu(cac_chunk, "Theo [3] thì nội dung là ...")

    assert len(trich_dan) == 1
    assert (trich_dan[0]["nguon"], trich_dan[0]["trang"]) == ("b.pdf", 30)
    assert trich_dan[0]["doan_trich"] == "C"


def test_trich_dan_ghi_du_khoang_trang_khi_doan_trich_vuot_ranh_gioi_trang():
    """Đoạn trích mở rộng sang trang liền kề thì câu trả lời có thể dựa vào nội dung ở trang
    bên cạnh. Trích dẫn chỉ ghi trang neo là trỏ người đọc tới chỗ KHÔNG chứa hết căn cứ -
    đúng loại sai lệch mà §5.54 đã phải sửa một lần."""
    cac_chunk = [
        {"nguon": "giaotrinh.pdf", "trang": 12, "cac_trang": [12, 13], "noidung": "x",
         "doan_khop": "Quy phạm pháp luật là", "diem_similarity": 0.9},
    ]
    trich_dan = loc_theo_tham_chieu(cac_chunk, "Nội dung theo [1].")

    assert trich_dan[0]["trang"] == 12, "trang NEO vẫn là trang neo"
    assert trich_dan[0]["cac_trang"] == [12, 13]


def test_trich_dan_khong_vuot_trang_thi_khoang_trang_chi_co_mot_trang():
    cac_chunk = [
        {"nguon": "a.pdf", "trang": 10, "noidung": "x", "doan_khop": "A", "diem_similarity": 0.9},
    ]
    trich_dan = loc_theo_tham_chieu(cac_chunk, "Nội dung theo [1].")
    assert trich_dan[0]["cac_trang"] == [10]


def test_trich_dan_lui_ve_nguon_lien_quan_nhat_khi_khong_co_tham_chieu():
    cac_chunk = [
        {"nguon": "a.pdf", "trang": 10, "noidung": "x", "doan_khop": "A", "diem_similarity": 0.9},
        {"nguon": "b.pdf", "trang": 30, "noidung": "z", "doan_khop": "C", "diem_similarity": 0.7},
    ]
    trich_dan = loc_theo_tham_chieu(cac_chunk, "Câu trả lời không gắn số nào.")

    assert len(trich_dan) == 1
    assert trich_dan[0]["nguon"] == "a.pdf"


def test_trich_dan_bat_duoc_ca_khi_llm_trich_theo_so_trang():
    """System prompt cấm trích kiểu "theo Slide 109" nhưng model nhỏ vẫn thỉnh thoảng làm
    vậy - bắt lại được thì trích dẫn vẫn đúng, thay vì lùi về đoán đoạn điểm cao nhất."""
    cac_chunk = [
        {"nguon": "a.pdf", "trang": 10, "noidung": "x", "doan_khop": "A", "diem_similarity": 0.9},
        {"nguon": "a.pdf", "trang": 109, "noidung": "y", "doan_khop": "B", "diem_similarity": 0.8},
    ]
    trich_dan = loc_theo_tham_chieu(cac_chunk, "Theo Slide 109 thì vi phạm pháp luật gồm...")

    assert len(trich_dan) == 1
    assert trich_dan[0]["trang"] == 109


def test_cau_tu_choi_khong_kem_trich_dan():
    """Vừa nói "không tìm thấy thông tin trong tài liệu" vừa chỉ vào một trang cụ thể là tự
    mâu thuẫn, và khiến người đọc tưởng trang đó có liên quan."""
    cac_chunk = [
        {"nguon": "a.pdf", "trang": 9, "noidung": "x", "doan_khop": "A", "diem_similarity": 0.7},
    ]
    for cau_tu_choi in config.CAU_TU_CHOI.values():
        assert loc_theo_tham_chieu(cac_chunk, cau_tu_choi) == []


def test_trich_dan_gop_cac_doan_cung_trang_khi_hien_thi():
    """Truy xuất cho phép tối đa 2 đoạn cùng 1 trang, nhưng người đọc chỉ cần biết trang."""
    cac_chunk = [
        {"nguon": "a.pdf", "trang": 10, "noidung": "x", "doan_khop": "A", "diem_similarity": 0.9},
        {"nguon": "a.pdf", "trang": 10, "noidung": "y", "doan_khop": "B", "diem_similarity": 0.8},
    ]
    assert len(loc_theo_tham_chieu(cac_chunk, "Theo [1] và [2] ...")) == 1


# ======================================================================
# Ngưỡng từ chối dựa trên điểm rerank (Phase 5)
# ======================================================================

class RerankerGiaTheoDiem:
    """Reranker giả trả về đúng bộ điểm đã định sẵn."""

    def __init__(self, cac_diem):
        self.cac_diem = cac_diem

    def xep_hang(self, cau_hoi, cac_doan):
        return np.array(self.cac_diem[: len(cac_doan)], dtype="float32")


def _pipeline_co_rerank(cac_diem_rerank, monkeypatch):
    cac_chunk = [_chunk("a.pdf", 1, i, f"Đoạn nội dung số {i} trong tài liệu.") for i in range(3)]
    cac_vector = [_vector_don_vi(0)] * 3
    store = _tao_store(cac_chunk, cac_vector)
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)
    return RagPipeline(
        EmbeddingGia(_vector_don_vi(0)), store,
        reranker_service=RerankerGiaTheoDiem(cac_diem_rerank),
    )


def test_diem_rerank_qua_thap_thi_tu_choi(monkeypatch):
    """Đo thực tế: câu lạc đề cho điểm rerank gần 0 tuyệt đối, trong khi câu đúng chủ đề
    (kể cả hỏi tiếng Anh trên tài liệu tiếng Việt) từ 0.019 trở lên - cosine KHÔNG tách
    được hai nhóm này, rerank thì tách được."""
    monkeypatch.setattr(config, "NGUONG_DIEM_RERANK_TOI_THIEU", 0.005)
    pipeline = _pipeline_co_rerank([0.001, 0.0005, 0.0], monkeypatch)
    assert pipeline.truy_xuat("câu hỏi lạc đề") == []


def test_diem_rerank_du_cao_thi_van_tra_ve(monkeypatch):
    monkeypatch.setattr(config, "NGUONG_DIEM_RERANK_TOI_THIEU", 0.005)
    pipeline = _pipeline_co_rerank([0.02, 0.01, 0.001], monkeypatch)
    assert pipeline.truy_xuat("câu hỏi đúng chủ đề") != []


def test_khong_rerank_thi_nguong_rerank_tu_bo_qua(monkeypatch):
    """Tắt rerank phải lùi về đúng hành vi cũ, không được từ chối oan mọi câu hỏi."""
    monkeypatch.setattr(config, "NGUONG_DIEM_RERANK_TOI_THIEU", 0.005)
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)
    cac_chunk = [_chunk("a.pdf", 1, 0, "Một đoạn nội dung đủ dài để giữ lại trong index.")]
    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, [_vector_don_vi(0)]))
    assert pipeline.diem_rerank_cao_nhat is None
    assert pipeline.truy_xuat("bất kỳ") != []


def test_diem_rerank_khong_sot_lai_giua_hai_luot(monkeypatch):
    """Điểm của lượt trước còn sót lại sẽ khiến câu hỏi này bị phán xét bằng điểm của câu
    hỏi khác - sai âm thầm, rất khó lần ra."""
    monkeypatch.setattr(config, "NGUONG_DIEM_RERANK_TOI_THIEU", 0.005)
    pipeline = _pipeline_co_rerank([0.9, 0.8, 0.7], monkeypatch)
    pipeline.truy_xuat("câu hỏi tốt")
    assert pipeline.diem_rerank_cao_nhat is not None

    pipeline.vector_store = VectorStore(dimension=SO_CHIEU)  # index rỗng -> thoát sớm
    pipeline.truy_xuat("câu khác")
    assert pipeline.diem_rerank_cao_nhat is None, "phải xoá điểm của lượt trước"


# ======================================================================
# Nhận diện ngôn ngữ (yêu cầu song ngữ: hỏi tiếng nào trả lời tiếng đó)
# ======================================================================

@pytest.mark.parametrize("cau_hoi", [
    "What does criminal law regulate?",
    "What is criminal liability?",
    "What are the characteristics of the state?",
    "What is optical flow?",
    "How does the EM algorithm work?",
])
def test_nhan_dien_dung_tieng_anh(cau_hoi):
    """Bản trước coi MỌI thứ không phải tiếng Anh là tiếng Việt, nên câu tiếng Anh bị
    langdetect chấm nhầm sang một thứ tiếng Latin họ gần sẽ bị trả lời bằng tiếng Việt.
    Đo thực tế: "What does criminal law regulate?" được chấm là tiếng Catalan (0.71) còn
    tiếng Anh chỉ 0.29 - câu hỏi tiếng Anh rõ ràng mà trả lời sai ngôn ngữ."""
    assert _phat_hien_ngon_ngu(cau_hoi) == "en"


@pytest.mark.parametrize("cau_hoi", [
    "Vi phạm pháp luật gồm những dấu hiệu nào?",
    "Nhà nước có những đặc điểm gì?",
    "Trách nhiệm hình sự là gì?",
])
def test_nhan_dien_dung_tieng_viet_co_dau(cau_hoi):
    assert _phat_hien_ngon_ngu(cau_hoi) == "vi"


@pytest.mark.parametrize("cau_hoi", [
    "SIFT la gi",
    "Luat hinh su dieu chinh gi",
    "nha nuoc co dac diem gi",
])
def test_nhan_dien_tieng_viet_khong_dau(cau_hoi):
    """Người Việt hay gõ không dấu. langdetect chấm những câu này thành tl/it - nếu không
    bắt được thì người hỏi tiếng Việt nhận câu trả lời tiếng Anh."""
    assert _phat_hien_ngon_ngu(cau_hoi) == "vi"


def test_cau_rong_hoac_vo_nghia_mac_dinh_tieng_viet():
    """Tiếng Việt là ngôn ngữ chính của hệ thống nên là lựa chọn mặc định an toàn."""
    for cau_hoi in ("", "   ", "123456", "???"):
        assert _phat_hien_ngon_ngu(cau_hoi) == "vi"


def test_nhan_dien_ngon_ngu_tat_dinh():
    """langdetect lấy mẫu ngẫu nhiên nên mặc định KHÔNG tất định - đo thực tế: cùng một câu
    chạy 8 lần cho 8 kết quả khác nhau, có lần danh sách không hề chứa 'en'. Với hệ song
    ngữ, nghĩa là cùng một câu hỏi lúc được trả lời tiếng Anh lúc tiếng Việt."""
    for cau_hoi in ("What does criminal law regulate?", "Nhà nước là gì?", "SIFT la gi"):
        cac_lan = {_phat_hien_ngon_ngu(cau_hoi) for _ in range(8)}
        assert len(cac_lan) == 1, f"'{cau_hoi}' cho kết quả khác nhau giữa các lần: {cac_lan}"


def test_tran_so_doan_anh(monkeypatch):
    """Mô tả ảnh do model vision sinh ra khá dài, nên với tài liệu nhiều hình chúng lấn át
    các trang văn bản đúng. Đo thực tế trên corpus thật: không có trần này thì Recall@K tụt
    0.96 -> 0.92, riêng nhóm slide tiếng Anh tụt 0.97 -> 0.86."""
    monkeypatch.setattr(config, "SO_DOAN_ANH_TOI_DA", 1)
    monkeypatch.setattr(config, "NGUONG_DIEM_TOI_THIEU", 0.0)

    cac_chunk = []
    for i in range(4):  # 4 bản ghi ảnh, tất cả đều khớp câu hỏi
        c = _chunk("a.pdf", 10 + i, 0, f"[HÌNH] Mô tả ảnh số {i} do model vision sinh ra.")
        c["loai_noi_dung"] = "anh"
        cac_chunk.append(c)
    for i in range(3):  # và 3 đoạn văn bản
        cac_chunk.append(_chunk("b.pdf", 20 + i, 0, f"Đoạn văn bản số {i} trong tài liệu."))
    cac_vector = [_vector_don_vi(0)] * len(cac_chunk)

    pipeline = RagPipeline(EmbeddingGia(_vector_don_vi(0)), _tao_store(cac_chunk, cac_vector))
    ket_qua = pipeline.truy_xuat("câu hỏi", top_k=6)

    so_anh = sum(1 for d in ket_qua if d.get("loai_noi_dung") == "anh")
    assert so_anh <= 1, f"ảnh lấn át: {so_anh} đoạn ảnh trong kết quả"
    assert any(d.get("loai_noi_dung") != "anh" for d in ket_qua), "văn bản phải còn suất"
