"""Test cơ bản cho module chunking - chỉ 3-5 test theo đúng phạm vi đồ án
(không cần bộ test suite pytest đầy đủ cho mọi module)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.chunking import DO_DAI_CHUNK_TOI_THIEU, chia_chunk, dem_token, kich_thuoc_chunk_an_toan, tao_splitter


def test_dem_token_khop_voi_tiktoken_truc_tiep():
    """dem_token() (bộ đếm dự phòng) phải cho kết quả giống hệt việc gọi tiktoken trực tiếp -
    đảm bảo không có lỗi wrapping khi đưa vào length_function của splitter."""
    import tiktoken

    enc = tiktoken.get_encoding(config.TIKTOKEN_ENCODING)
    text = "Trí tuệ nhân tạo là một lĩnh vực quan trọng của khoa học máy tính."
    assert dem_token(text) == len(enc.encode(text))


def test_splitter_duoc_cau_hinh_dung_theo_config():
    splitter = tao_splitter()
    assert splitter._chunk_size == config.CHUNK_SIZE_TOKENS
    assert splitter._chunk_overlap == min(config.CHUNK_OVERLAP_TOKENS, config.CHUNK_SIZE_TOKENS // 3)


def test_kich_thuoc_chunk_tu_ha_theo_gioi_han_cua_model():
    """Nội dung vượt max_seq_length bị embedding model cắt bỏ ÂM THẦM lúc encode - phần bị
    cắt sẽ không nằm trong vector nên vĩnh viễn không tìm thấy được. Vì vậy chunk phải tự
    nhỏ theo model, không phụ thuộc vào việc có ai nhớ sửa config hay không."""
    assert kich_thuoc_chunk_an_toan(64) == 64 - config.BIEN_AN_TOAN_TOKEN
    assert kich_thuoc_chunk_an_toan(4096) == config.CHUNK_SIZE_TOKENS
    assert kich_thuoc_chunk_an_toan(None) == config.CHUNK_SIZE_TOKENS


def test_chunk_khong_vuot_qua_kich_thuoc_cau_hinh():
    """Mỗi chunk sinh ra không được vượt quá kích thước cấu hình - đây là ràng buộc quan
    trọng nhất vì embedding model sẽ cắt bớt (và mất thông tin) nếu chunk quá dài."""
    noi_dung_dai = "Đây là một câu ví dụ để kiểm tra chức năng chunking. " * 80
    cac_trang = [{"nguon": "test.pdf", "trang": 1, "noidung": noi_dung_dai}]
    cac_chunk = chia_chunk(cac_trang)

    assert len(cac_chunk) > 1  # nội dung đủ dài nên phải bị chia thành nhiều chunk
    for chunk in cac_chunk:
        assert dem_token(chunk["noidung"]) <= config.CHUNK_SIZE_TOKENS


def test_chunk_dung_ham_dem_token_duoc_truyen_vao():
    """Kích thước chunk chỉ có ý nghĩa khi đo bằng tokenizer của chính model sẽ encode nó -
    đo bằng tiktoken cho ra số token gấp ~1.9 lần trên tiếng Việt, khiến chunk nhỏ hơn hẳn
    dự định và nội dung bị băm vụn."""
    dem_theo_tu = lambda text: len(text.split())  # noqa: E731 - bộ đếm giả, chỉ dùng trong test
    cac_trang = [{"nguon": "test.pdf", "trang": 1, "noidung": "từ " * 400}]
    cac_chunk = chia_chunk(cac_trang, dem_token_fn=dem_theo_tu)

    for chunk in cac_chunk:
        assert dem_theo_tu(chunk["noidung"]) <= config.CHUNK_SIZE_TOKENS


def test_chunk_giu_metadata_va_co_chunk_id_duy_nhat():
    """Mỗi chunk phải giữ đúng nguon/trang của trang gốc (để citation.py trích dẫn đúng),
    và chunk_id phải duy nhất giữa các chunk."""
    cac_trang = [
        {"nguon": "a.pdf", "trang": 1, "noidung": "Nội dung trang một. " * 30},
        {"nguon": "a.pdf", "trang": 2, "noidung": "Nội dung trang hai. " * 30},
    ]
    cac_chunk = chia_chunk(cac_trang)

    assert all(c["nguon"] == "a.pdf" for c in cac_chunk)
    assert any(c["trang"] == 1 for c in cac_chunk)
    assert any(c["trang"] == 2 for c in cac_chunk)

    cac_id = [c["chunk_id"] for c in cac_chunk]
    assert len(cac_id) == len(set(cac_id))


def test_vi_tri_tang_dan_trong_tung_trang():
    """rag_pipeline sắp xếp theo vi_tri để mở rộng ngữ cảnh sang chunk liền kề đúng thứ tự
    gốc - nếu vi_tri không phản ánh đúng thứ tự thì đoạn trích sẽ bị xáo nội dung."""
    cac_trang = [{"nguon": "a.pdf", "trang": 1, "noidung": "Câu văn mẫu để chia nhỏ. " * 60}]
    vi_tri = [c["vi_tri"] for c in chia_chunk(cac_trang)]
    assert vi_tri == sorted(vi_tri)


def test_bo_qua_chunk_qua_ngan():
    """Mẩu vụn do tách thô (số thứ tự lạc dòng, dấu ngoặc đơn lẻ) không mang đủ ngữ nghĩa
    để khớp câu hỏi nào, nhưng vẫn chiếm 1 vector và vẫn có thể lọt top -> chỉ gây nhiễu."""
    cac_trang = [{"nguon": "a.pdf", "trang": 1, "noidung": "1."}]
    assert chia_chunk(cac_trang) == []
    assert DO_DAI_CHUNK_TOI_THIEU > len("1.")


def test_chunk_tra_ve_rong_voi_trang_khong_co_noi_dung():
    cac_trang = [{"nguon": "a.pdf", "trang": 1, "noidung": ""}]
    assert chia_chunk(cac_trang) == []


def test_bang_duoc_phep_dai_hon_van_xuoi():
    """Cắt nhỏ bảng phá đúng thứ khiến nó là bảng (dòng tiêu đề cột), mà không đổi lại được
    gì về độ chính xác truy xuất. Ràng buộc cứng duy nhất với bảng là giới hạn của model.

    Đo trên bộ tài liệu thật: 15 bảng bị cắt vì vượt 160 token, phần lớn dài 164-476 token -
    vẫn nằm gọn trong giới hạn 496 của model, tức bị cắt oan hoàn toàn."""
    from rag.document_loader import MOC_BANG_DONG, MOC_BANG_MO

    # Bảng ~300 "token" theo bộ đếm giả: vượt CHUNK_SIZE_TOKENS nhưng vừa giới hạn model.
    hang = "| cot mot | cot hai | cot ba |\n"
    bang = f"{MOC_BANG_MO}\n" + hang * 100 + f"{MOC_BANG_DONG}"
    cac_trang = [{"nguon": "a.pdf", "trang": 1, "noidung": bang}]

    dem_theo_tu = lambda t: len(t.split())  # noqa: E731
    cac_chunk = chia_chunk(cac_trang, dem_token_fn=dem_theo_tu, max_seq_length=4000)

    cac_chunk_bang = [c for c in cac_chunk if c.get("loai_noi_dung") == "bang"]
    assert len(cac_chunk_bang) == 1, "bảng vừa giới hạn model phải nằm trọn 1 chunk"
    assert dem_theo_tu(cac_chunk_bang[0]["noidung"]) > config.CHUNK_SIZE_TOKENS


def test_bang_vuot_gioi_han_model_van_bi_cat():
    """Vượt giới hạn model thì buộc phải cắt - nội dung vượt quá sẽ bị model cắt âm thầm
    lúc encode, tức mất hẳn khỏi index. Cắt được vẫn hơn mất."""
    from rag.document_loader import MOC_BANG_DONG, MOC_BANG_MO

    hang = "| cot mot | cot hai | cot ba |\n"
    bang = f"{MOC_BANG_MO}\n" + hang * 500 + f"{MOC_BANG_DONG}"
    cac_trang = [{"nguon": "a.pdf", "trang": 1, "noidung": bang}]

    dem_theo_tu = lambda t: len(t.split())  # noqa: E731
    cac_chunk = chia_chunk(cac_trang, dem_token_fn=dem_theo_tu, max_seq_length=200)

    assert len(cac_chunk) > 1
    for c in cac_chunk:
        assert dem_theo_tu(c["noidung"]) <= 200
