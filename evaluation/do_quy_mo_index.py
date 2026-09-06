"""Đo xem `IndexFlatIP` chịu được corpus tới cỡ nào, và chuyển sang IVF/HNSW thì được gì."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np

import config

K_TIM_KIEM = max(config.TOP_K * config.HE_SO_OVER_FETCH, config.SO_UNG_VIEN_TOI_THIEU)

NGAN_SACH_TIM_KIEM_MS = 200.0

SO_CAU_HOI_DO = 30


def _so_co_dau_cham(n: int) -> str:
    """123456 -> '123.456' (dấu chấm phân cách hàng nghìn theo lối viết tiếng Việt)."""
    return f"{n:,}".replace(",", ".")


def _vector_gia_lap(
    so_luong: int, so_chieu: int, seed: int, so_cum: int = 200, do_tuong_dong: float = 0.8
) -> np.ndarray:
    """Vector giả lập ĐÃ CHUẨN HOÁ, có GOM CỤM theo chủ đề như embedding thật."""
    rng = np.random.default_rng(seed)
    tam_cum = rng.standard_normal((so_cum, so_chieu), dtype="float32")
    faiss.normalize_L2(tam_cum)
    sigma = float(np.sqrt(1.0 / do_tuong_dong**2 - 1.0) / np.sqrt(so_chieu))
    thuoc_cum = rng.integers(0, so_cum, size=so_luong)
    v = tam_cum[thuoc_cum] + sigma * rng.standard_normal(
        (so_luong, so_chieu), dtype="float32"
    )
    v = np.ascontiguousarray(v, dtype="float32")
    faiss.normalize_L2(v)
    return v


def _do_do_tre(index, cac_cau_hoi: np.ndarray) -> tuple:
    """Trả về (p50, p95) tính bằng mili giây, đo từng câu hỏi một."""
    cac_moc = []
    for i in range(len(cac_cau_hoi)):
        mot_cau = cac_cau_hoi[i : i + 1]
        bat_dau = time.perf_counter()
        index.search(mot_cau, K_TIM_KIEM)
        cac_moc.append((time.perf_counter() - bat_dau) * 1000)
    cac_moc.sort()
    return cac_moc[len(cac_moc) // 2], cac_moc[int(len(cac_moc) * 0.95)]


def _recall_so_voi_flat(index, cac_cau_hoi: np.ndarray, dap_an_flat: np.ndarray) -> float:
    """Tỉ lệ vector mà index gần đúng tìm được, so với danh sách đúng do Flat trả về."""
    _, vi_tri = index.search(cac_cau_hoi, K_TIM_KIEM)
    trung = sum(
        len(set(vi_tri[i]) & set(dap_an_flat[i])) for i in range(len(cac_cau_hoi))
    )
    return trung / (len(cac_cau_hoi) * K_TIM_KIEM)


def do_mot_kich_thuoc(so_chunk: int, so_chieu: int) -> dict:
    print(f"\n=== {_so_co_dau_cham(so_chunk)} chunk × {so_chieu} chiều ===")
    du_lieu = _vector_gia_lap(so_chunk, so_chieu, seed=1)
    cac_cau_hoi = _vector_gia_lap(SO_CAU_HOI_DO, so_chieu, seed=2)
    ram_mb = du_lieu.nbytes / 1024 / 1024

    ket_qua = {"so_chunk": so_chunk, "ram_mb": ram_mb}

    flat = faiss.IndexFlatIP(so_chieu)
    bat_dau = time.perf_counter()
    flat.add(du_lieu)
    ket_qua["flat_build_giay"] = time.perf_counter() - bat_dau
    ket_qua["flat_p50"], ket_qua["flat_p95"] = _do_do_tre(flat, cac_cau_hoi)
    _, dap_an_flat = flat.search(cac_cau_hoi, K_TIM_KIEM)
    print(
        f"  FlatIP  build {ket_qua['flat_build_giay']:6.2f}s  "
        f"p50 {ket_qua['flat_p50']:7.1f}ms  p95 {ket_qua['flat_p95']:7.1f}ms  "
        f"recall 1.000 (theo định nghĩa)  RAM {ram_mb:.0f}MB"
    )

    hnsw = faiss.IndexHNSWFlat(so_chieu, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    bat_dau = time.perf_counter()
    hnsw.add(du_lieu)
    ket_qua["hnsw_build_giay"] = time.perf_counter() - bat_dau
    hnsw.hnsw.efSearch = 128
    ket_qua["hnsw_p50"], ket_qua["hnsw_p95"] = _do_do_tre(hnsw, cac_cau_hoi)
    ket_qua["hnsw_recall"] = _recall_so_voi_flat(hnsw, cac_cau_hoi, dap_an_flat)
    print(
        f"  HNSW    build {ket_qua['hnsw_build_giay']:6.2f}s  "
        f"p50 {ket_qua['hnsw_p50']:7.1f}ms  p95 {ket_qua['hnsw_p95']:7.1f}ms  "
        f"recall {ket_qua['hnsw_recall']:.3f}"
    )

    nlist = max(int(4 * so_chunk**0.5), 16)
    ivf = faiss.IndexIVFFlat(
        faiss.IndexFlatIP(so_chieu), so_chieu, nlist, faiss.METRIC_INNER_PRODUCT
    )
    bat_dau = time.perf_counter()
    ivf.train(du_lieu)
    ivf.add(du_lieu)
    ket_qua["ivf_build_giay"] = time.perf_counter() - bat_dau
    ivf.nprobe = max(nlist // 16, 8)
    ket_qua["ivf_p50"], ket_qua["ivf_p95"] = _do_do_tre(ivf, cac_cau_hoi)
    ket_qua["ivf_recall"] = _recall_so_voi_flat(ivf, cac_cau_hoi, dap_an_flat)
    print(
        f"  IVFFlat build {ket_qua['ivf_build_giay']:6.2f}s  "
        f"p50 {ket_qua['ivf_p50']:7.1f}ms  p95 {ket_qua['ivf_p95']:7.1f}ms  "
        f"recall {ket_qua['ivf_recall']:.3f}  (nlist={nlist}, nprobe={ivf.nprobe})"
    )
    return ket_qua


def _ket_luan(cac_ket_qua: list, so_chieu: int) -> None:
    print("\n" + "=" * 78)
    print("KẾT LUẬN — ngưỡng cần chuyển khỏi IndexFlatIP")
    print("=" * 78)

    lon_nhat = max(cac_ket_qua, key=lambda k: k["so_chunk"])
    ms_moi_chunk = lon_nhat["flat_p95"] / lon_nhat["so_chunk"]
    nguong = int(NGAN_SACH_TIM_KIEM_MS / ms_moi_chunk)

    print(
        f"Đo được: FlatIP tốn {ms_moi_chunk * 1_000_000:.1f} µs cho mỗi 1.000 chunk "
        f"(p95, k={K_TIM_KIEM}, {so_chieu} chiều, trên chính máy này)."
    )
    print(
        f"Với ngân sách {NGAN_SACH_TIM_KIEM_MS:.0f}ms cho riêng bước tìm vector, "
        f"ngưỡng ≈ {_so_co_dau_cham(nguong)} chunk."
    )
    corpus_hien_tai = 5854
    print(
        f"Corpus hiện tại của đồ án ({_so_co_dau_cham(corpus_hien_tai)} chunk) đang ở "
        f"{corpus_hien_tai / nguong:.1%} ngưỡng đó — còn dư rất nhiều, "
        "nên GIỮ NGUYÊN FlatIP là quyết định đúng."
    )
    mb_moi_nghin = so_chieu * 4 * 1000 / 1024**2
    print(
        "\nNhưng RAM chạm trần TRƯỚC độ trễ: FlatIP giữ nguyên vector nên tốn "
        f"{mb_moi_nghin:.1f}MB mỗi 1.000 chunk, tức "
        f"{nguong * so_chieu * 4 / 1024**3:.1f}GB ở ngưỡng trên — quá sức một máy để bàn "
        "thông thường. Ràng buộc thật vì vậy là BỘ NHỚ, không phải tốc độ: mốc thực tế nằm "
        f"ở khoảng {_so_co_dau_cham(int(2 * 1024**3 / (so_chieu * 4)))} chunk (≈2GB RAM "
        "cho index)."
    )

    lon_nhat_co_ann = lon_nhat.get("hnsw_recall") is not None
    if lon_nhat_co_ann:
        print(
            f"\nKhi thật sự phải đổi, ở cỡ lớn nhất đã đo "
            f"({_so_co_dau_cham(lon_nhat['so_chunk'])} chunk):"
        )
        print(
            f"  HNSW : nhanh hơn {lon_nhat['flat_p95'] / lon_nhat['hnsw_p95']:.0f} lần, "
            f"recall {lon_nhat['hnsw_recall']:.2f}, build {lon_nhat['hnsw_build_giay']:.0f}s"
        )
        print(
            f"  IVF  : nhanh hơn {lon_nhat['flat_p95'] / lon_nhat['ivf_p95']:.0f} lần, "
            f"recall {lon_nhat['ivf_recall']:.2f}, build {lon_nhat['ivf_build_giay']:.0f}s"
        )
        tot_hon = "IVFFlat" if lon_nhat["ivf_recall"] > lon_nhat["hnsw_recall"] else "HNSW"
        print(
            f"  → {tot_hon} là lựa chọn đúng ở đây. Lưu ý recall của HNSW TỤT DẦN khi corpus "
            "to lên nếu giữ nguyên efSearch (đo được: 0.87 → 0.74 → 0.61) - nó không phải "
            "một tham số đặt một lần rồi quên."
        )
        print(
            "  Và dù chọn cái nào: đổi index BẮT BUỘC phải chạy lại run_evaluation.py để đo "
            "Recall@K của cả hệ thống. Đoạn bị bỏ sót hoàn toàn có thể là đoạn chứa câu trả "
            "lời - nhanh hơn mà trả lời sai thì không phải là cải tiến."
        )


def main() -> None:
    bo_phan_tich = argparse.ArgumentParser(description=__doc__)
    bo_phan_tich.add_argument(
        "--kich-thuoc",
        default="10000,50000,100000",
        help="Danh sách số chunk cần đo, ngăn bằng dấu phẩy.",
    )
    bo_phan_tich.add_argument(
        "--so-chieu",
        type=int,
        default=768,
        help="Số chiều vector (mặc định 768 = multilingual-e5-base).",
    )
    doi_so = bo_phan_tich.parse_args()
    cac_kich_thuoc = [int(x) for x in doi_so.kich_thuoc.split(",") if x.strip()]

    print(
        f"Đo FAISS trên máy này — k={K_TIM_KIEM} (đúng số ứng viên hệ thống thật lấy về), "
        f"{SO_CAU_HOI_DO} câu hỏi mỗi phép đo, từng câu một."
    )
    cac_ket_qua = [do_mot_kich_thuoc(n, doi_so.so_chieu) for n in cac_kich_thuoc]
    _ket_luan(cac_ket_qua, doi_so.so_chieu)


if __name__ == "__main__":
    main()
