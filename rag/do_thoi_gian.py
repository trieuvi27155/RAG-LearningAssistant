"""Đo thời gian từng bước của luồng Ingestion - để tối ưu bằng SỐ ĐO, không bằng phỏng đoán.

VÌ SAO CẦN, khi log đã có sẵn dấu thời gian:
Log nói được "trang 412 bị OCR lúc 09:31:07", nhưng không nói được "OCR chiếm 68% tổng thời
gian build". Hai câu đó dẫn tới hai quyết định tối ưu hoàn toàn khác nhau. Trước khi có bảng
tổng kết này, câu hỏi "chỗ nào đang chậm nhất" chỉ trả lời được bằng cách ngồi đọc hàng nghìn
dòng log rồi trừ dấu thời gian bằng tay - tức là không ai làm, và mọi tối ưu đều thành đoán.

CÁCH DÙNG:
    from rag import do_thoi_gian

    do_thoi_gian.dat_lai()                       # đầu một lần build
    with do_thoi_gian.do("doc_trang_pdf"):       # bọc quanh bước cần đo
        ...
    logger.info("\n%s", do_thoi_gian.bao_cao())  # cuối lần build

THIẾT KẾ:
  - Bộ đếm là biến MODULE (một tiến trình = một lần build), không phải tham số truyền tay
    qua 5 tầng hàm. Truyền tay sẽ khiến mọi hàm đọc tài liệu phải mang thêm một tham số
    chẳng liên quan gì tới việc đọc tài liệu.
  - Có Lock vì Vision/OCR chạy trên nhiều thread (xem config.SO_WORKER_VISION); cộng dồn
    float từ nhiều thread mà không khoá thì số đo sai lệch âm thầm - đúng loại lỗi khiến
    người ta mất niềm tin vào chính công cụ đo của mình.
  - Ghi nhận cả SỐ LẦN gọi chứ không chỉ tổng giây: "OCR 900 giây" và "OCR 900 giây / 3 lần"
    là hai câu chuyện khác nhau (bước chậm vs. bước gọi quá nhiều lần).
  - Không bao giờ nuốt exception: khối `with` vẫn tính giờ cho lần chạy hỏng rồi để lỗi bay
    tiếp, vì thời gian đã tiêu vào một bước hỏng vẫn là thời gian đã tiêu.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ten_buoc -> (số lần chạy, tổng số giây)
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
    """Bảng tổng kết dạng text, sắp theo tổng thời gian giảm dần.

    Cột "%" tính trên tổng thời gian THẬT của cả lần build, nên tổng các dòng có thể nhỏ hơn
    100% (phần không được bọc `do()`) hoặc LỚN hơn 100% (các bước lồng nhau, hoặc chạy song
    song trên nhiều thread). Cả hai đều là thông tin có ích chứ không phải lỗi: chênh lệch
    lớn giữa tổng cột và 100% chính là dấu hiệu còn một bước tốn kém chưa được đo.
    """
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
