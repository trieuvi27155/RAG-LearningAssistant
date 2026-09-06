"""Nhận biết phần cứng và quản lý GPU theo GIAI ĐOẠN - tự thích ứng với máy đang chạy."""

import logging
import os
from typing import Optional, Tuple

import config

logger = logging.getLogger(__name__)

_co_cuda: Optional[bool] = None


def co_cuda() -> bool:
    """Máy này có GPU dùng được cho PyTorch không."""
    global _co_cuda
    if _co_cuda is None:
        try:
            import torch

            _co_cuda = bool(torch.cuda.is_available())
        except Exception as loi:  # noqa: BLE001
            logger.info("Không dùng được CUDA (%s) - chạy trên CPU.", type(loi).__name__)
            _co_cuda = False
    return _co_cuda


def thiet_bi(vai_tro: str) -> str:
    """Thiết bị nên dùng cho một vai trò: "cuda" hoặc "cpu"."""
    cau_hinh = {
        "embedding": config.THIET_BI_EMBEDDING,
        "rerank": config.THIET_BI_RERANK,
    }[vai_tro]
    if cau_hinh != "auto":
        return cau_hinh
    if not co_cuda():
        return "cpu"
    if vai_tro == "embedding" and not du_cho_giu_embedding_tren_gpu():
        return "cpu"
    return "cuda"


def vram() -> Optional[Tuple[float, float, float]]:
    """(PyTorch đang giữ, tổng VRAM, VRAM còn trống) tính bằng GB. None nếu không có GPU."""
    if not co_cuda():
        return None
    import torch

    trong, tong = torch.cuda.mem_get_info(0)
    return (
        torch.cuda.memory_reserved(0) / (1 << 30),
        tong / (1 << 30),
        trong / (1 << 30),
    )


def tong_vram_gb() -> float:
    """Tổng VRAM của GPU (GB), 0.0 nếu không có GPU."""
    so_lieu = vram()
    return so_lieu[1] if so_lieu else 0.0


def vram_con_trong_gb() -> float:
    """VRAM còn trống ngay lúc này (GB), 0.0 nếu không có GPU."""
    so_lieu = vram()
    return so_lieu[2] if so_lieu else 0.0


def so_nhan_cpu() -> int:
    return os.cpu_count() or 2


def mo_ta_phan_cung() -> str:
    """Một dòng mô tả phần cứng đang được dùng, cho log và giao diện."""
    if not co_cuda():
        return (
            f"Phần cứng: CPU {so_nhan_cpu()} nhân · KHÔNG dùng GPU "
            "(PyTorch bản CPU-only, hoặc máy không có GPU NVIDIA dùng được)"
        )
    import torch

    return (
        f"Phần cứng: {torch.cuda.get_device_name(0)} ({tong_vram_gb():.1f} GB VRAM) · "
        f"CPU {so_nhan_cpu()} nhân · embedding={thiet_bi('embedding')} "
        f"· rerank={thiet_bi('rerank')} · batch embedding={kich_thuoc_lo_embedding()}"
    )


def kich_thuoc_lo_embedding() -> int:
    """Batch size encode, suy từ VRAM CÒN TRỐNG. Trên CPU thì giữ nguyên giá trị cấu hình."""
    tran = config.EMBEDDING_BATCH_SIZE
    if not co_cuda():
        return tran
    trong = vram_con_trong_gb()
    if trong >= config.VRAM_DU_CHO_LO_LON_GB:
        return tran
    if trong >= config.VRAM_DU_CHO_LO_VUA_GB:
        return min(tran, 32)
    logger.info(
        "VRAM còn trống chỉ %.1f GB - hạ batch embedding xuống 16 để tránh tràn.", trong
    )
    return min(tran, 16)


def so_worker_vision() -> int:
    """Số luồng gọi model vision/OCR song song, suy từ phần cứng đang có."""
    tran = max(1, config.SO_WORKER_VISION)
    theo_cpu = max(1, so_nhan_cpu() // 2)
    if not co_cuda():
        return min(tran, theo_cpu)
    theo_vram = max(1, int(vram_con_trong_gb() // config.VRAM_MOI_WORKER_VISION_GB))
    ket_qua = min(tran, theo_cpu, theo_vram)
    if ket_qua < tran:
        logger.info(
            "Số worker vision: %d (trần cấu hình %d, theo CPU %d, theo VRAM còn trống %d).",
            ket_qua, tran, theo_cpu, theo_vram,
        )
    return ket_qua


def ghi_log_vram(nhan: str) -> None:
    """Ghi tình trạng VRAM kèm nhãn giai đoạn. Không có GPU thì im lặng bỏ qua."""
    so_lieu = vram()
    if so_lieu is None:
        return
    torch_giu, tong, trong = so_lieu
    logger.info(
        "VRAM [%s]: PyTorch giữ %.2f GB · còn trống %.2f/%.2f GB.",
        nhan, torch_giu, trong, tong,
    )


def don_bo_nho_cuda() -> None:
    """Trả lại cho driver phần VRAM PyTorch đã cấp phát nhưng không còn dùng."""
    if not co_cuda():
        return
    import torch

    torch.cuda.empty_cache()


def nha_model_ollama(ten_model: str, client=None) -> bool:
    """Bảo Ollama nhả một model khỏi VRAM NGAY, thay vì đợi hết thời gian giữ mặc định."""
    if not config.NHA_MODEL_SAU_INGESTION:
        return False
    try:
        import ollama

        client = client or ollama.Client(host=config.OLLAMA_HOST)
        client.generate(model=ten_model, prompt="", keep_alive=0)
    except Exception as loi:  # noqa: BLE001
        logger.debug("Không nhả được model '%s' khỏi VRAM (%s).", ten_model, type(loi).__name__)
        return False
    logger.info("Đã yêu cầu Ollama nhả model '%s' khỏi VRAM.", ten_model)
    return True


def du_cho_giu_embedding_tren_gpu() -> bool:
    """Card này có đủ chỗ để GIỮ embedding trên GPU trong lúc truy vấn không?"""
    if not co_cuda():
        return False
    return tong_vram_gb() >= config.VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB


def bat_dau_ingestion(embedding_service=None) -> None:
    """Vào giai đoạn INGESTION: đưa embedding trở lại GPU nếu máy có."""
    if not config.BAT_QUAN_LY_VRAM or embedding_service is None or not co_cuda():
        return
    if config.THIET_BI_EMBEDDING == "auto":
        embedding_service.chuyen_thiet_bi("cuda")


def ket_thuc_ingestion(client=None, embedding_service=None) -> None:
    """Chuyển từ giai đoạn INGESTION sang QUERY: nhả những gì query không cần."""
    if not config.BAT_QUAN_LY_VRAM or not co_cuda():
        return
    ghi_log_vram("kết thúc ingestion")
    nha_model_ollama(config.VISION_MODEL_NAME, client)
    if embedding_service is not None and config.THIET_BI_EMBEDDING == "auto":
        if not du_cho_giu_embedding_tren_gpu():
            logger.info(
                "VRAM %.1f GB không đủ cho cả LLM + reranker + embedding cùng lúc - chuyển "
                "embedding sang CPU cho giai đoạn truy vấn (mất ~15ms mỗi câu, trả lại ~1,1 GB).",
                tong_vram_gb(),
            )
            embedding_service.chuyen_thiet_bi("cpu")
    don_bo_nho_cuda()
    ghi_log_vram("sau khi nhả vision")
