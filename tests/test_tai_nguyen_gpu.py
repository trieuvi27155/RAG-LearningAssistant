"""Test cho lớp nhận biết phần cứng và quản lý VRAM theo giai đoạn.

RÀNG BUỘC QUAN TRỌNG NHẤT mà bộ test này canh giữ: **mọi thứ phải chạy được trên máy KHÔNG
có GPU**. Đó không phải trường hợp biên hiếm gặp mà là môi trường mặc định của người chấm đồ
án, và của bất kỳ ai chạy thử trên laptop. Một hàm quản lý GPU ném lỗi trên máy không GPU sẽ
làm sập cả luồng build index vì một tính năng lẽ ra chỉ là tối ưu.

Vì vậy các test dưới đây ép `co_cuda()` trả False để chạy đúng nhánh "không có GPU", thay vì
phụ thuộc vào việc máy chạy test có card hay không - nếu phụ thuộc thì trên máy có GPU nhánh
đó sẽ KHÔNG BAO GIỜ được kiểm, tức lỗi chỉ lộ ra ở máy người khác.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from rag import tai_nguyen_gpu


@pytest.fixture
def khong_gpu(monkeypatch):
    monkeypatch.setattr(tai_nguyen_gpu, "co_cuda", lambda: False)


@pytest.fixture
def gia_lap_gpu(monkeypatch):
    """Giả lập một GPU với lượng VRAM còn trống tuỳ ý, không cần card thật.

    Trả về một hàm đặt số GB còn trống, để từng test mô tả đúng tình huống nó quan tâm
    (card rộng rãi / card đang chật / card gần đầy).
    """
    monkeypatch.setattr(tai_nguyen_gpu, "co_cuda", lambda: True)

    def dat(con_trong_gb: float, tong_gb: float = 8.0):
        monkeypatch.setattr(
            tai_nguyen_gpu, "vram", lambda: (1.0, tong_gb, con_trong_gb)
        )

    dat(6.0)
    return dat


# ======================================================================
# 1. Chọn thiết bị
# ======================================================================

def test_khong_co_gpu_thi_moi_vai_tro_deu_chay_cpu(khong_gpu, monkeypatch):
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "auto")
    monkeypatch.setattr(config, "THIET_BI_RERANK", "auto")
    assert tai_nguyen_gpu.thiet_bi("embedding") == "cpu"
    assert tai_nguyen_gpu.thiet_bi("rerank") == "cpu"


def test_card_lon_thi_auto_chon_cuda_cho_ca_hai(gia_lap_gpu, monkeypatch):
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "auto")
    monkeypatch.setattr(config, "THIET_BI_RERANK", "auto")
    monkeypatch.setattr(config, "VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB", 10.0)
    gia_lap_gpu(20.0, tong_gb=24.0)
    assert tai_nguyen_gpu.thiet_bi("embedding") == "cuda"
    assert tai_nguyen_gpu.thiet_bi("rerank") == "cuda"


def test_card_nho_thi_embedding_mac_dinh_o_cpu_con_rerank_van_gpu(gia_lap_gpu, monkeypatch):
    """Trạng thái mặc định phải là trạng thái của giai đoạn HAY GẶP NHẤT, tức query.

    Phiên làm việc điển hình không bắt đầu bằng "Đọc tài liệu" mà bằng việc mở app lên hỏi
    ngay trên index đã có — lúc đó bước chuyển giai đoạn chưa hề chạy. Mặc định sai ở đây
    khiến embedding tranh VRAM với LLM suốt cả phiên: đo được truy xuất 7,4s thay vì 0,35s.
    """
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "auto")
    monkeypatch.setattr(config, "THIET_BI_RERANK", "auto")
    monkeypatch.setattr(config, "VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB", 10.0)
    gia_lap_gpu(6.0, tong_gb=8.0)
    assert tai_nguyen_gpu.thiet_bi("embedding") == "cpu"
    assert tai_nguyen_gpu.thiet_bi("rerank") == "cuda", "rerank nằm trên đường đi mọi câu hỏi"


def test_cau_hinh_ep_thiet_bi_thi_khong_bi_tu_do_de(gia_lap_gpu, monkeypatch):
    """Ép "cpu" phải được tôn trọng KỂ CẢ khi máy có GPU - đây là đường lui để đo đối chứng,
    và để nhường GPU cho việc khác."""
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "cpu")
    monkeypatch.setattr(config, "THIET_BI_RERANK", "cuda")
    assert tai_nguyen_gpu.thiet_bi("embedding") == "cpu"
    assert tai_nguyen_gpu.thiet_bi("rerank") == "cuda"


def test_hai_vai_tro_tach_roi_nhau(gia_lap_gpu, monkeypatch):
    """Trên card nhỏ, cấu hình hợp lý có thể là embedding GPU còn rerank CPU."""
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "cuda")
    monkeypatch.setattr(config, "THIET_BI_RERANK", "cpu")
    assert tai_nguyen_gpu.thiet_bi("embedding") != tai_nguyen_gpu.thiet_bi("rerank")


# ======================================================================
# 2. Batch size suy từ VRAM
# ======================================================================

def test_tren_cpu_thi_giu_nguyen_batch_cau_hinh(khong_gpu, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 64)
    assert tai_nguyen_gpu.kich_thuoc_lo_embedding() == 64


def test_vram_rong_rai_thi_dung_tron_batch_cau_hinh(gia_lap_gpu, monkeypatch):
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "auto")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 64)
    gia_lap_gpu(6.0)
    assert tai_nguyen_gpu.kich_thuoc_lo_embedding() == 64


def test_vram_chat_thi_ha_batch_xuong(gia_lap_gpu, monkeypatch):
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "auto")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 64)

    gia_lap_gpu(2.5)  # giữa hai ngưỡng
    assert tai_nguyen_gpu.kich_thuoc_lo_embedding() == 32
    gia_lap_gpu(0.6)  # gần đầy
    assert tai_nguyen_gpu.kich_thuoc_lo_embedding() == 16


def test_batch_khong_bao_gio_vuot_tran_cau_hinh(gia_lap_gpu, monkeypatch):
    """EMBEDDING_BATCH_SIZE là TRẦN, không phải giá trị đích: người dùng hạ xuống thì phải
    được tôn trọng dù VRAM có rộng tới đâu."""
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "auto")
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 8)
    for con_trong in (0.5, 2.5, 24.0):
        gia_lap_gpu(con_trong)
        assert tai_nguyen_gpu.kich_thuoc_lo_embedding() == 8


# ======================================================================
# 3. Số worker suy từ phần cứng
# ======================================================================

def test_so_worker_luon_it_nhat_mot(khong_gpu, monkeypatch):
    """Trả 0 worker sẽ làm ThreadPoolExecutor ném lỗi và giết cả lần build."""
    monkeypatch.setattr(config, "SO_WORKER_VISION", 0)
    monkeypatch.setattr(tai_nguyen_gpu, "so_nhan_cpu", lambda: 1)
    assert tai_nguyen_gpu.so_worker_vision() >= 1


def test_so_worker_khong_vuot_tran_cau_hinh(gia_lap_gpu, monkeypatch):
    monkeypatch.setattr(config, "SO_WORKER_VISION", 2)
    monkeypatch.setattr(tai_nguyen_gpu, "so_nhan_cpu", lambda: 64)
    gia_lap_gpu(24.0)
    assert tai_nguyen_gpu.so_worker_vision() == 2


def test_may_it_nhan_cpu_thi_it_worker(khong_gpu, monkeypatch):
    monkeypatch.setattr(config, "SO_WORKER_VISION", 8)
    monkeypatch.setattr(tai_nguyen_gpu, "so_nhan_cpu", lambda: 2)
    assert tai_nguyen_gpu.so_worker_vision() == 1


def test_vram_it_thi_ha_so_worker(gia_lap_gpu, monkeypatch):
    """Ràng buộc VRAM là thứ khiến cùng một cấu hình cho câu trả lời khác nhau trên hai máy."""
    monkeypatch.setattr(config, "SO_WORKER_VISION", 8)
    monkeypatch.setattr(config, "VRAM_MOI_WORKER_VISION_GB", 1.5)
    monkeypatch.setattr(tai_nguyen_gpu, "so_nhan_cpu", lambda: 32)

    gia_lap_gpu(6.0)
    assert tai_nguyen_gpu.so_worker_vision() == 4
    gia_lap_gpu(1.6)
    assert tai_nguyen_gpu.so_worker_vision() == 1


# ======================================================================
# 4. Không có GPU thì mọi thứ phải là KHÔNG-LÀM-GÌ, không được ném lỗi
# ======================================================================

def test_cac_ham_gpu_khong_nem_loi_tren_may_khong_gpu(khong_gpu):
    assert tai_nguyen_gpu.vram() is None
    assert tai_nguyen_gpu.tong_vram_gb() == 0.0
    assert tai_nguyen_gpu.vram_con_trong_gb() == 0.0
    tai_nguyen_gpu.don_bo_nho_cuda()   # không được ném
    tai_nguyen_gpu.ghi_log_vram("thử")  # không được ném
    tai_nguyen_gpu.ket_thuc_ingestion()  # không được ném


def test_mo_ta_phan_cung_noi_ro_khi_dang_chay_cpu(khong_gpu):
    """Câu này là thứ duy nhất cho người dùng biết họ đang chạy chậm gấp chục lần mà không
    hề có lỗi nào báo ra."""
    mo_ta = tai_nguyen_gpu.mo_ta_phan_cung()
    assert "KHÔNG dùng GPU" in mo_ta
    assert "CPU" in mo_ta


def test_khong_nha_model_khi_tuy_chon_tat(monkeypatch):
    monkeypatch.setattr(config, "NHA_MODEL_SAU_INGESTION", False)
    assert tai_nguyen_gpu.nha_model_ollama("model-bat-ky") is False


def test_ollama_khong_chay_thi_nha_model_that_bai_am_tham(monkeypatch):
    """Ollama tắt không phải lý do làm hỏng một lần build vừa chạy xong."""
    monkeypatch.setattr(config, "NHA_MODEL_SAU_INGESTION", True)

    class ClientHong:
        def generate(self, **_):
            raise ConnectionError("Ollama chưa chạy")

    assert tai_nguyen_gpu.nha_model_ollama("qwen2.5vl:3b", ClientHong()) is False


def test_tat_quan_ly_vram_thi_khong_nha_gi_ca(gia_lap_gpu, monkeypatch):
    da_goi = []
    monkeypatch.setattr(config, "BAT_QUAN_LY_VRAM", False)
    monkeypatch.setattr(tai_nguyen_gpu, "nha_model_ollama",
                        lambda *a, **k: da_goi.append(a) or True)
    tai_nguyen_gpu.ket_thuc_ingestion()
    assert da_goi == []


# ======================================================================
# 5. Chuyển embedding giữa GPU và CPU theo giai đoạn
# ======================================================================

class _EmbeddingGia:
    """Ghi lại các lần bị yêu cầu đổi thiết bị, để test kiểm được THỨ TỰ và SỐ LẦN."""

    def __init__(self, thiet_bi="cuda"):
        self.thiet_bi = thiet_bi
        self.lich_su = []

    def chuyen_thiet_bi(self, moi):
        self.lich_su.append(moi)
        da_doi = moi != self.thiet_bi
        self.thiet_bi = moi
        return da_doi


def test_card_nho_thi_day_embedding_xuong_cpu_sau_ingestion(gia_lap_gpu, monkeypatch):
    """Trên card 8 GB, LLM + reranker + embedding = 8,07 GB nên phải bớt một."""
    monkeypatch.setattr(config, "BAT_QUAN_LY_VRAM", True)
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "auto")
    monkeypatch.setattr(config, "VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB", 10.0)
    monkeypatch.setattr(tai_nguyen_gpu, "nha_model_ollama", lambda *a, **k: True)
    monkeypatch.setattr(tai_nguyen_gpu, "don_bo_nho_cuda", lambda: None)
    gia_lap_gpu(1.0, tong_gb=8.0)

    emb = _EmbeddingGia("cuda")
    tai_nguyen_gpu.ket_thuc_ingestion(embedding_service=emb)
    assert emb.lich_su == ["cpu"]


def test_card_lon_thi_giu_embedding_tren_gpu(gia_lap_gpu, monkeypatch):
    """Card đủ rộng thì không có lý do gì phải đánh đổi."""
    monkeypatch.setattr(config, "BAT_QUAN_LY_VRAM", True)
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "auto")
    monkeypatch.setattr(config, "VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB", 10.0)
    monkeypatch.setattr(tai_nguyen_gpu, "nha_model_ollama", lambda *a, **k: True)
    monkeypatch.setattr(tai_nguyen_gpu, "don_bo_nho_cuda", lambda: None)
    gia_lap_gpu(20.0, tong_gb=24.0)

    emb = _EmbeddingGia("cuda")
    tai_nguyen_gpu.ket_thuc_ingestion(embedding_service=emb)
    assert emb.lich_su == []


def test_ep_thiet_bi_bang_cau_hinh_thi_khong_bi_tu_dong_chuyen(gia_lap_gpu, monkeypatch):
    """Người dùng ép "cuda" là một quyết định, không phải gợi ý - đừng lặng lẽ lật lại."""
    monkeypatch.setattr(config, "BAT_QUAN_LY_VRAM", True)
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "cuda")
    monkeypatch.setattr(config, "VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB", 10.0)
    monkeypatch.setattr(tai_nguyen_gpu, "nha_model_ollama", lambda *a, **k: True)
    monkeypatch.setattr(tai_nguyen_gpu, "don_bo_nho_cuda", lambda: None)
    gia_lap_gpu(1.0, tong_gb=8.0)

    emb = _EmbeddingGia("cuda")
    tai_nguyen_gpu.ket_thuc_ingestion(embedding_service=emb)
    tai_nguyen_gpu.bat_dau_ingestion(emb)
    assert emb.lich_su == []


def test_lan_build_sau_dua_embedding_tro_lai_gpu(gia_lap_gpu, monkeypatch):
    """Không đưa lại thì mọi lần build sau đều chạy CPU và mất khoản 13,3× ở chỗ nó đáng giá
    nhất - đây đúng là loại hồi quy không gây lỗi, chỉ âm thầm chậm đi."""
    monkeypatch.setattr(config, "BAT_QUAN_LY_VRAM", True)
    monkeypatch.setattr(config, "THIET_BI_EMBEDDING", "auto")
    gia_lap_gpu(1.0, tong_gb=8.0)

    emb = _EmbeddingGia("cpu")
    tai_nguyen_gpu.bat_dau_ingestion(emb)
    assert emb.lich_su == ["cuda"]


def test_khong_co_gpu_thi_khong_dong_gi_toi_embedding(khong_gpu, monkeypatch):
    monkeypatch.setattr(config, "BAT_QUAN_LY_VRAM", True)
    emb = _EmbeddingGia("cpu")
    tai_nguyen_gpu.bat_dau_ingestion(emb)
    tai_nguyen_gpu.ket_thuc_ingestion(embedding_service=emb)
    assert emb.lich_su == []


def test_ket_thuc_ingestion_nha_dung_model_vision(gia_lap_gpu, monkeypatch):
    da_nha = []
    monkeypatch.setattr(config, "BAT_QUAN_LY_VRAM", True)
    monkeypatch.setattr(tai_nguyen_gpu, "don_bo_nho_cuda", lambda: None)
    monkeypatch.setattr(tai_nguyen_gpu, "nha_model_ollama",
                        lambda ten, client=None: da_nha.append(ten) or True)
    tai_nguyen_gpu.ket_thuc_ingestion()
    assert da_nha == [config.VISION_MODEL_NAME], (
        "chỉ được nhả model vision - nhả cả embedding thì mọi câu hỏi sau đó phải nạp lại"
    )
