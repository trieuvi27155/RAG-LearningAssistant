"""Đo số worker TỐI ƯU cho OCR/Vision trên phần cứng đang có, kèm GPU utilization và VRAM.

VÌ SAO CẦN ĐO CHỨ KHÔNG ĐOÁN: "nhiều worker hơn thì nhanh hơn" là trực giác sai với GPU.
Mọi yêu cầu đều xếp hàng qua MỘT máy chủ Ollama và một GPU; mở thêm luồng chỉ giúp khi phần
chờ (mạng, nạp/xả giữa các lượt) còn đáng kể so với phần tính toán. Vượt quá điểm đó thì
thêm worker chỉ làm VRAM phình lên và có thể khiến Ollama nạp/nhả model liên tục - chậm hơn
hẳn so với chạy ít worker.

Con số tối ưu KHÁC NHAU TRÊN MỖI MÁY (dung lượng VRAM, số nhân CPU, model vision đang dùng),
nên script này để chạy lại trên máy của bạn chứ không phải để đọc lại kết quả của máy khác.

CÁCH CHẠY:
    python evaluation/do_worker_gpu.py                      # 1, 2, 4 worker · 6 trang
    python evaluation/do_worker_gpu.py --so-trang 12        # nhiều trang hơn, số ổn định hơn
    python evaluation/do_worker_gpu.py --worker 1,2,4,8     # tự chọn mức cần thử
    python evaluation/do_worker_gpu.py --file TaiLieuTest/x.pdf

ĐỌC KẾT QUẢ: chọn mức worker nhỏ nhất còn cho tốc độ gần với mức tốt nhất, rồi đặt vào
`SO_WORKER_VISION` trong `.env`. Không chọn mức nhanh nhất bằng mọi giá: mức đó thường ăn
sát VRAM, và hệ thống còn phải chia GPU với LLM lẫn embedding ở giai đoạn sau.
"""

import argparse
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfplumber

import config
from rag import bo_nho_dem, tai_nguyen_gpu
from rag.document_loader import _doc_mot_trang_pdf, _ocr_cac_trang
from rag.vision_caption import trang_can_ocr

FILE_MAC_DINH = "TaiLieuTest/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"


class TheoDoiGpu:
    """Lấy mẫu GPU utilization và VRAM trong lúc phép đo chạy, bằng nvidia-smi.

    Dùng nvidia-smi thay vì torch: phần việc đang đo nằm trong tiến trình OLLAMA, không phải
    trong tiến trình Python này - `torch.cuda.memory_allocated()` sẽ báo gần như 0 và khiến
    người đọc kết luận sai rằng GPU đang rảnh. nvidia-smi nhìn thấy toàn bộ card.

    Không có nvidia-smi (máy không GPU, hoặc GPU không phải NVIDIA) thì mọi thứ trả None -
    phép đo thời gian vẫn chạy bình thường, chỉ thiếu hai cột phụ.
    """

    def __init__(self, chu_ky_giay: float = 0.25):
        self.chu_ky_giay = chu_ky_giay
        self.co_san = shutil.which("nvidia-smi") is not None
        self.util: list = []
        self.vram_mb: list = []
        self._dung = threading.Event()
        self._luong = None

    def _lay_mau(self) -> None:
        while not self._dung.is_set():
            try:
                ket_qua = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                u, m = ket_qua.stdout.strip().splitlines()[0].split(",")
                self.util.append(float(u))
                self.vram_mb.append(float(m))
            except Exception:  # noqa: BLE001 - mất một mẫu không đáng làm hỏng phép đo
                pass
            self._dung.wait(self.chu_ky_giay)

    def __enter__(self):
        if self.co_san:
            self._luong = threading.Thread(target=self._lay_mau, daemon=True)
            self._luong.start()
        return self

    def __exit__(self, *_):
        self._dung.set()
        if self._luong:
            self._luong.join(timeout=2)

    def tom_tat(self):
        """(GPU util trung bình %, GPU util đỉnh %, VRAM đỉnh GB) hoặc None."""
        if not self.util:
            return None
        return (
            statistics.mean(self.util),
            max(self.util),
            max(self.vram_mb) / 1024,
        )


def _tim_trang_can_ocr(duong_dan: Path, so_trang_can: int, bo_qua_dau: int = 300):
    """Tìm những trang THẬT SỰ cần OCR, để phép đo chạy trên đúng loại việc nó mô tả.

    Bỏ qua phần đầu sách: bìa, mục lục và lời nói đầu thường đọc được bình thường nên không
    kích hoạt OCR, đo trên chúng sẽ ra một con số không nói lên điều gì.
    """
    can_ocr = []
    with pdfplumber.open(duong_dan) as pdf:
        for so_trang, trang in enumerate(pdf.pages, start=1):
            if so_trang < bo_qua_dau:
                trang.flush_cache()
                continue
            text = _doc_mot_trang_pdf(trang, duong_dan.name, so_trang, None)
            if trang_can_ocr(text, len(trang.images)):
                can_ocr.append(so_trang)
            trang.flush_cache()
            if len(can_ocr) >= so_trang_can:
                break
    return can_ocr


def chay(duong_dan: Path, cac_muc_worker, so_trang: int) -> None:
    print(tai_nguyen_gpu.mo_ta_phan_cung())
    print(f"Tài liệu: {duong_dan.name}\n")

    can_ocr = _tim_trang_can_ocr(duong_dan, so_trang)
    if not can_ocr:
        print("Không tìm được trang nào cần OCR trong tài liệu này - thử --file khác.")
        return
    print(f"{len(can_ocr)} trang cần OCR: {can_ocr}\n")

    bam = bo_nho_dem.bam_file(duong_dan)
    goc_cache = config.CACHE_DIR
    ket_qua = []

    with pdfplumber.open(duong_dan) as pdf:
        for so_worker in cac_muc_worker:
            # MỖI mức worker phải bắt đầu từ cache RỖNG, nếu không mức thứ hai trở đi chỉ đo
            # tốc độ đọc file cache và cho ra một bảng vô nghĩa.
            config.CACHE_DIR = goc_cache / f"_do_worker_{so_worker}"
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            bo_nho_dem.kho_ocr = bo_nho_dem.KhoDem("ocr", ".txt")
            # Ép cứng số worker: mục đích ở đây là ĐO từng mức, không phải để hệ thống tự
            # chọn giúp (việc tự chọn dựa trên chính bảng số này).
            goc_so_worker = tai_nguyen_gpu.so_worker_vision
            tai_nguyen_gpu.so_worker_vision = lambda n=so_worker: n
            try:
                with TheoDoiGpu() as theo_doi:
                    moc = time.perf_counter()
                    doc_duoc = _ocr_cac_trang(pdf, can_ocr, duong_dan.name, bam)
                    giay = time.perf_counter() - moc
                gpu = theo_doi.tom_tat()
            finally:
                tai_nguyen_gpu.so_worker_vision = goc_so_worker
            ket_qua.append((so_worker, giay, len(doc_duoc), gpu))
            print(f"  worker={so_worker}: {giay:.1f}s ({giay / len(can_ocr):.1f}s/trang)",
                  flush=True)

    config.CACHE_DIR = goc_cache
    for thu_muc in goc_cache.glob("_do_worker_*"):
        shutil.rmtree(thu_muc, ignore_errors=True)

    print(f"\n{'WORKER':>7}{'TỔNG (s)':>11}{'s/TRANG':>10}{'NHANH HƠN':>11}"
          f"{'GPU TB %':>10}{'GPU ĐỈNH %':>12}{'VRAM ĐỈNH GB':>14}")
    print("-" * 75)
    goc_giay = ket_qua[0][1]
    for so_worker, giay, doc_duoc, gpu in ket_qua:
        tb, dinh, vram = gpu if gpu else (float("nan"),) * 3
        print(f"{so_worker:>7}{giay:>11.1f}{giay / len(can_ocr):>10.1f}"
              f"{goc_giay / giay:>10.2f}x{tb:>10.0f}{dinh:>12.0f}{vram:>14.2f}")

    tot_nhat = min(ket_qua, key=lambda r: r[1])
    # Chọn mức NHỎ NHẤT còn nằm trong 5% của mức nhanh nhất: 5% chênh lệch nằm trong khoảng
    # nhiễu giữa các lần chạy, nên không đáng đổi lấy VRAM và rủi ro tranh chấp tài nguyên.
    de_xuat = min(r[0] for r in ket_qua if r[1] <= tot_nhat[1] * 1.05)
    print(f"\nĐỀ XUẤT: SO_WORKER_VISION={de_xuat} "
          f"(mức nhỏ nhất còn trong 5% của mức nhanh nhất là {tot_nhat[0]}).")
    print("Đặt vào .env; số liệu này chỉ đúng cho máy vừa chạy phép đo.")


def main() -> None:
    bo_phan_tich = argparse.ArgumentParser(description=__doc__)
    bo_phan_tich.add_argument("--file", default=FILE_MAC_DINH)
    bo_phan_tich.add_argument("--worker", default="1,2,4")
    bo_phan_tich.add_argument("--so-trang", type=int, default=6)
    tham_so = bo_phan_tich.parse_args()

    duong_dan = Path(tham_so.file)
    if not duong_dan.exists():
        sys.exit(f"Không thấy file '{duong_dan}'. Dùng --file để chỉ tài liệu khác.")
    chay(duong_dan, [int(x) for x in tham_so.worker.split(",")], tham_so.so_trang)


if __name__ == "__main__":
    main()
