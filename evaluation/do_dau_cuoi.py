"""Đo ĐẦU-CUỐI: từ lúc đưa tài liệu vào tới lúc hỏi được, và từ lúc hỏi tới lúc có câu trả lời.

VÌ SAO PHẢI ĐO ĐẦU-CUỐI CHỨ KHÔNG CỘNG CÁC PHẦN LẠI: mọi phép đo lẻ trong project này (đọc
tài liệu, OCR, rerank, sinh câu trả lời) đều đo một bước với model đã nạp sẵn và cache đã ấm.
Người dùng thì không sống trong điều kiện đó - họ trả cả tiền nạp model, tiền khởi tạo CUDA,
tiền ghi index xuống đĩa, tiền chờ Ollama nạp LLM ở câu hỏi đầu tiên. Cộng các phần lẻ lại sẽ
ra một con số nhỏ hơn thực tế, và tối ưu theo con số đó là tối ưu nhầm chỗ.

HAI PHÉP ĐO, TÁCH RIÊNG VÌ CHÚNG NÓI VỀ HAI TRẢI NGHIỆM KHÁC NHAU:

  A. INGESTION: đưa tài liệu vào -> index sẵn sàng. Đây là thứ người dùng chờ MỘT LẦN.
     Đo cả hai kịch bản, vì chúng lệch nhau rất xa và cả hai đều có thật:
       - cache RỖNG  : lần đầu nạp một corpus mới.
       - cache ĐẦY   : bấm "Đọc tài liệu" lần nữa, hoặc thêm 1 file vào corpus đã có.

  B. QUERY: gửi câu hỏi -> câu trả lời hoàn chỉnh. Đây là thứ người dùng chờ MỖI LẦN, nên nó
     mới là con số quyết định cảm nhận. Tách riêng "câu hỏi đầu tiên" khỏi các câu sau: câu
     đầu còn gánh thời gian Ollama nạp LLM vào VRAM, và trộn nó vào trung bình sẽ bôi một
     chi phí một-lần lên mọi câu hỏi.

CÁCH CHẠY:
    python evaluation/do_dau_cuoi.py                       # cả hai phép đo
    python evaluation/do_dau_cuoi.py --chi ingestion
    python evaluation/do_dau_cuoi.py --chi query
    python evaluation/do_dau_cuoi.py --thu-muc TaiLieuTest --so-file 5
    python evaluation/do_dau_cuoi.py --khong-llm           # chỉ đo truy xuất, không gọi LLM

LƯU Ý: script KHÔNG đụng tới `data/raw`, `data/faiss_index` hay `data/cache` của bạn - nó
dựng index riêng trong thư mục tạm rồi xoá đi.
"""

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag import bo_nho_dem, do_thoi_gian, tai_nguyen_gpu

CAU_HOI_MAC_DINH = [
    "Biến trong Python được khai báo như thế nào?",
    "Sự khác nhau giữa list và tuple là gì?",
    "Vòng lặp for hoạt động ra sao?",
    "Hàm được định nghĩa bằng từ khoá nào?",
]


def _tom_tat(cac_giay):
    if not cac_giay:
        return "—"
    if len(cac_giay) == 1:
        return f"{cac_giay[0]:.2f}s"
    return (f"trung vị {statistics.median(cac_giay):.2f}s "
            f"(min {min(cac_giay):.2f} · max {max(cac_giay):.2f})")


def _bang_profiling(tieu_de: str) -> None:
    so_lieu = do_thoi_gian.so_lieu()
    if not so_lieu:
        return
    print(f"\n  {tieu_de}")
    for ten, (lan, giay) in sorted(so_lieu.items(), key=lambda kv: -kv[1][1]):
        print(f"    {ten:<32}{lan:>6} lần{giay:>9.2f}s{giay / lan * 1000:>10.0f} ms/lần")


# ============================================================
# A. INGESTION ĐẦU-CUỐI
# ============================================================
def do_ingestion(cac_file, thu_muc_lam_viec: Path, embedding_service):
    """Đo trọn luồng: đọc -> chunk -> embed -> ghi index, với cache rỗng rồi cache đầy."""
    from rag.chunking import chia_chunk
    from rag.document_loader import doc_nhieu_file
    from rag.vector_store import VectorStore

    duong_dan_index = dict(
        index_path=thu_muc_lam_viec / "index.faiss",
        metadata_path=thu_muc_lam_viec / "metadata.pkl",
        info_path=thu_muc_lam_viec / "index_info.json",
    )

    ket_qua = {}
    for nhan in ("cache rỗng", "cache đầy"):
        do_thoi_gian.dat_lai()
        moc = time.perf_counter()
        # Vào giai đoạn INGESTION giống hệt app.py: đưa embedding lên GPU nếu máy có. Thiếu
        # bước này thì phép đo chạy với embedding ở CPU (trạng thái mặc định của giai đoạn
        # QUERY) và cho ra một con số không mô tả đúng thứ người dùng gặp.
        tai_nguyen_gpu.bat_dau_ingestion(embedding_service)
        cac_trang = doc_nhieu_file(cac_file)
        with do_thoi_gian.do("chunking"):
            cac_chunk = chia_chunk(
                cac_trang,
                dem_token_fn=embedding_service.lay_ham_dem_token(),
                max_seq_length=embedding_service.max_seq_length,
            )
        with do_thoi_gian.do("embedding"):
            vectors = bo_nho_dem.encode_co_cache(
                embedding_service, [c["noidung"] for c in cac_chunk]
            )
        with do_thoi_gian.do("ghi_index"):
            store = VectorStore(dimension=embedding_service.dimension)
            store.them(vectors, cac_chunk)
            store.luu(**duong_dan_index)
        # Truyền embedding_service để phép đo đi ĐÚNG đường mà app.py đi: trên card chật,
        # bước này còn đẩy embedding xuống CPU nhường VRAM cho reranker và LLM. Bỏ tham số
        # thì phép đo query bên dưới chạy trong một điều kiện VRAM khác với thực tế.
        tai_nguyen_gpu.ket_thuc_ingestion(embedding_service=embedding_service)
        giay = time.perf_counter() - moc

        ket_qua[nhan] = (giay, len(cac_chunk), len(cac_trang))
        print(f"\n  {nhan:12s}: {giay:7.2f}s  ({len(cac_trang)} bản ghi · {len(cac_chunk)} chunk)")
        _bang_profiling("chi tiết:")
    return ket_qua, duong_dan_index


# ============================================================
# B. QUERY ĐẦU-CUỐI
# ============================================================
def do_query(pipeline, cac_cau_hoi, goi_llm: bool):
    """Đo từ lúc gửi câu hỏi tới lúc có câu trả lời hoàn chỉnh."""
    dong = []
    for i, cau_hoi in enumerate(cac_cau_hoi, start=1):
        do_thoi_gian.dat_lai()
        moc = time.perf_counter()
        if goi_llm:
            ket_qua = pipeline.hoi_dap(cau_hoi, doi_chieu=False)
            tong = time.perf_counter() - moc
            do_tre = ket_qua["do_tre"]
            dong.append({
                "cau_hoi": cau_hoi,
                "tong": tong,
                "truy_xuat": do_tre["truy_xuat"],
                "chu_dau_tien": do_tre.get("chu_dau_tien"),
                "so_doan": len(ket_qua["cac_chunk_nguon"]),
                "chi_tiet": do_thoi_gian.so_lieu(),
            })
        else:
            cac_chunk = pipeline.truy_xuat(cau_hoi)
            tong = time.perf_counter() - moc
            dong.append({
                "cau_hoi": cau_hoi, "tong": tong, "truy_xuat": tong,
                "chu_dau_tien": None, "so_doan": len(cac_chunk),
                "chi_tiet": do_thoi_gian.so_lieu(),
            })
        nhan = "câu ĐẦU (gồm nạp LLM)" if i == 1 else f"câu {i}"
        print(f"  {nhan:24s} tổng {tong:6.2f}s · truy xuất {dong[-1]['truy_xuat']:5.2f}s"
              f" · {dong[-1]['so_doan']} đoạn", flush=True)
    return dong


def _in_tong_ket_query(dong, goi_llm: bool) -> None:
    if not dong:
        return
    sau_cau_dau = dong[1:] or dong
    print(f"\n  Câu hỏi ĐẦU TIÊN : {dong[0]['tong']:.2f}s  (gồm cả thời gian Ollama nạp LLM)")
    print(f"  Các câu sau      : {_tom_tat([d['tong'] for d in sau_cau_dau])}")
    print(f"  - truy xuất      : {_tom_tat([d['truy_xuat'] for d in sau_cau_dau])}")
    if goi_llm:
        cac_chu = [d["chu_dau_tien"] for d in sau_cau_dau if d["chu_dau_tien"]]
        print(f"  - tới chữ đầu    : {_tom_tat(cac_chu)}")

    # Gộp chi tiết các bước truy xuất của những câu SAU câu đầu.
    gop = {}
    for d in sau_cau_dau:
        for ten, (lan, giay) in d["chi_tiet"].items():
            l, g = gop.get(ten, (0, 0.0))
            gop[ten] = (l + lan, g + giay)
    if gop:
        print("\n  Chi tiết bước truy xuất (trung bình mỗi câu):")
        for ten, (lan, giay) in sorted(gop.items(), key=lambda kv: -kv[1][1]):
            print(f"    {ten:<32}{giay / len(sau_cau_dau):>8.3f}s")


# ============================================================
def main() -> None:
    bo = argparse.ArgumentParser(description=__doc__)
    bo.add_argument("--thu-muc", default="TaiLieuTest")
    bo.add_argument("--so-file", type=int, default=4)
    bo.add_argument("--chi", choices=["ingestion", "query"], default=None)
    bo.add_argument("--khong-llm", action="store_true")
    bo.add_argument("--cau-hoi", default=None, help="file JSON chứa danh sách câu hỏi")
    tham_so = bo.parse_args()

    from rag.embedding import EmbeddingService
    from rag.rag_pipeline import RagPipeline
    from rag.reranker import tao_reranker_neu_bat
    from rag.vector_store import VectorStore

    print("=" * 78)
    print("ĐO ĐẦU-CUỐI")
    print("=" * 78)
    print(tai_nguyen_gpu.mo_ta_phan_cung())

    thu_muc = Path(tham_so.thu_muc)
    cac_file = [
        d for d in sorted(thu_muc.glob("*"))
        if d.suffix.lower() in (".pdf", ".pptx", ".docx")
    ][: tham_so.so_file]
    if not cac_file:
        sys.exit(f"Không có tài liệu nào trong '{thu_muc}'.")
    print(f"Corpus đo: {len(cac_file)} tài liệu từ '{thu_muc}'")

    # Cache và index riêng cho phép đo - KHÔNG đụng dữ liệu thật của người dùng.
    with tempfile.TemporaryDirectory() as tam:
        tam = Path(tam)
        config.CACHE_DIR = tam / "cache"
        config.IMAGES_DIR = tam / "images"
        for d in (config.CACHE_DIR, config.IMAGES_DIR):
            d.mkdir(parents=True, exist_ok=True)
        bo_nho_dem.kho_tai_lieu = bo_nho_dem.KhoDem("tai_lieu", ".json")
        bo_nho_dem.kho_ocr = bo_nho_dem.KhoDem("ocr", ".txt")
        bo_nho_dem.kho_vision = bo_nho_dem.KhoDem("vision", ".txt")

        moc_nap = time.perf_counter()
        embedding_service = EmbeddingService()
        giay_nap_embedding = time.perf_counter() - moc_nap
        print(f"Nạp model embedding: {giay_nap_embedding:.2f}s")

        duong_dan_index = None
        if tham_so.chi != "query":
            print("\n" + "-" * 78)
            print("A. INGESTION ĐẦU-CUỐI (đưa tài liệu vào -> index sẵn sàng)")
            print("-" * 78)
            _, duong_dan_index = do_ingestion(cac_file, tam, embedding_service)

        if tham_so.chi == "ingestion":
            return

        print("\n" + "-" * 78)
        print("B. QUERY ĐẦU-CUỐI (gửi câu hỏi -> câu trả lời hoàn chỉnh)")
        print("-" * 78)
        if duong_dan_index is None:
            if not config.FAISS_INDEX_FILE.exists():
                sys.exit("Chưa có index để đo query. Bỏ --chi query, hoặc build index trước.")
            store = VectorStore.tai()
        else:
            store = VectorStore.tai(**duong_dan_index)
        print(f"Index: {store.so_luong_vector} chunk")

        moc_nap = time.perf_counter()
        reranker = tao_reranker_neu_bat()
        print(f"Nạp model rerank: {time.perf_counter() - moc_nap:.2f}s "
              f"({'không nạp' if reranker is None else reranker.thiet_bi})")

        cac_cau_hoi = CAU_HOI_MAC_DINH
        if tham_so.cau_hoi:
            du_lieu = json.loads(Path(tham_so.cau_hoi).read_text(encoding="utf-8"))
            cac_cau_hoi = [c["cau_hoi"] if isinstance(c, dict) else c for c in du_lieu][:6]

        pipeline = RagPipeline(embedding_service, store, reranker_service=reranker)
        goi_llm = not tham_so.khong_llm
        print(f"({len(cac_cau_hoi)} câu hỏi · {'CÓ' if goi_llm else 'KHÔNG'} gọi LLM)\n")
        dong = do_query(pipeline, cac_cau_hoi, goi_llm)
        _in_tong_ket_query(dong, goi_llm)

    print("\n" + "=" * 78)
    print("Số liệu chỉ đúng cho máy vừa chạy phép đo. Chạy lại sau mỗi thay đổi cấu hình.")


if __name__ == "__main__":
    main()
