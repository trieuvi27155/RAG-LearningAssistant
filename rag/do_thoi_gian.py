"""Đo thời gian từng bước của luồng Ingestion - để tối ưu bằng SỐ ĐO, không bằng phỏng đoán."""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

_bo_dem: Dict[str, Tuple[int, float]] = {}
_khoa = threading.Lock()
_moc_bat_dau = time.perf_counter()


def dat_lai() -> None:
    """Xoá mọi số đo và bấm lại đồng hồ tổng. Gọi ở đầu mỗi lần build index."""
    global _moc_bat_dau
    with _khoa:
        _bo_dem.clear()
        _moc_bat_dau = time.perf_counter()


def ghi_nhan(ten_buoc: str, so_giay: float, so_lan: int = 1) -> None:
    """Cộng thêm một lần chạy vào bộ đếm của `ten_buoc`."""
    with _khoa:
        lan, tong = _bo_dem.get(ten_buoc, (0, 0.0))
        _bo_dem[ten_buoc] = (lan + so_lan, tong + so_giay)


@contextmanager
def do(ten_buoc: str):
    """Bọc quanh một bước cần đo: `with do("ocr_trang"): ...`"""
    moc = time.perf_counter()
    try:
        yield
    finally:
        ghi_nhan(ten_buoc, time.perf_counter() - moc)


def so_lieu() -> Dict[str, Tuple[int, float]]:
    """Bản sao số đo hiện tại - cho test và cho các chỗ muốn tự định dạng lại."""
    with _khoa:
        return dict(_bo_dem)


def tong_giay() -> float:
    """Số giây trôi qua kể từ dat_lai() - mẫu số để tính phần trăm."""
    return time.perf_counter() - _moc_bat_dau


def bao_cao(tieu_de: str = "PROFILING INGESTION") -> str:
    """Bảng tổng kết dạng text, sắp theo tổng thời gian giảm dần."""
    du_lieu = so_lieu()
    if not du_lieu:
        return f"{tieu_de}: chưa có số đo nào."
    tong = max(tong_giay(), 1e-9)
    dong = [
        f"{tieu_de} (tổng {tong:.1f}s)",
        f"{'BƯỚC':<32}{'SỐ LẦN':>9}{'TỔNG (s)':>11}{'TB (ms)':>10}{'%':>7}",
        "-" * 69,
    ]
    for ten, (lan, giay) in sorted(du_lieu.items(), key=lambda kv: -kv[1][1]):
        tb_ms = giay / lan * 1000 if lan else 0.0
        dong.append(f"{ten:<32}{lan:>9}{giay:>11.1f}{tb_ms:>10.1f}{giay / tong * 100:>6.1f}%")
    return "\n".join(dong)


def ghi_bao_cao(tieu_de: str = "PROFILING INGESTION") -> None:
    """Ghi bảng tổng kết ra log (INFO). Không tự dat_lai() - việc đó thuộc về chỗ gọi."""
    logger.info("\n%s", bao_cao(tieu_de))
