"""Chạy đánh giá hệ thống RAG trên bộ câu hỏi test (evaluation/test_questions.json)."""

import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluation.metrics import (
    answer_relevance,
    do_chinh_xac_trich_dan,
    faithfulness,
    nghich_dao_thu_hang,
    precision_tai_k,
    recall_tai_k,
    thu_hang_dung_dau_tien,
)
from rag.embedding import EmbeddingService
from rag.rag_pipeline import RagPipeline
from rag.reranker import tao_reranker_neu_bat
from rag.vector_store import VectorStore

CAC_COT_DIEM = [
    "precision_at_k",
    "recall_at_k",
    "faithfulness",
    "answer_relevance",
    "citation_accuracy",
]
CAC_COT_SO = CAC_COT_DIEM + ["do_tre_giay"]

DUONG_DAN_CSV_MAC_DINH = config.EVAL_DIR / "ket_qua_danh_gia.csv"
DUONG_DAN_CSV_TRUOC = config.EVAL_DIR / "ket_qua_danh_gia_truoc.csv"
DUONG_DAN_CSV_HELD_OUT = config.EVAL_DIR / "ket_qua_danh_gia_held_out.csv"


def nap_bo_cau_hoi(duong_dan: Path = None) -> List[Dict]:
    duong_dan = duong_dan or config.TEST_QUESTIONS_FILE
    with open(duong_dan, "r", encoding="utf-8") as f:
        return json.load(f)


def _trung_binh(cac_muc: List[Dict], cot: str) -> float:
    """Trung bình 1 cột, bỏ qua mục thiếu cột đó (CSV cũ có thể chưa có cột mới)."""
    gia_tri = [float(m[cot]) for m in cac_muc if m.get(cot) not in (None, "")]
    return sum(gia_tri) / len(gia_tri) if gia_tri else 0.0


def _dang_ngo(kq: Dict) -> bool:
    """Câu này có bị chính thước đo Faithfulness tự đánh dấu là chấm đáng ngờ không."""
    return str(kq.get("faithfulness_dang_ngo", "")) in ("1", "True", "true")


def _in_bang_ket_qua(ket_qua_tung_cau: List[Dict]) -> None:
    print(f"{'Câu hỏi':<45} {'P@K':>6} {'R@K':>6} {'Faith':>7} {'Relev':>7} {'Trích':>7} {'Giây':>7}")
    print("-" * 92)
    for kq in ket_qua_tung_cau:
        cau_hoi = kq["cau_hoi"]
        cau_hoi_rut_gon = (cau_hoi[:42] + "...") if len(cau_hoi) > 45 else cau_hoi
        trich = kq.get("citation_accuracy")
        trich_hien = f"{float(trich):>7.2f}" if trich not in (None, "") else f"{'-':>7}"
        faith_hien = f"{kq['faithfulness']:>6.2f}" + ("!" if _dang_ngo(kq) else " ")
        print(
            f"{cau_hoi_rut_gon:<45} {kq['precision_at_k']:>6.2f} {kq['recall_at_k']:>6.2f} "
            f"{faith_hien} {kq['answer_relevance']:>7.2f} {trich_hien} "
            f"{kq['do_tre_giay']:>7.1f}"
        )

    print("-" * 92)
    print(
        f"{'TRUNG BÌNH':<45} "
        f"{_trung_binh(ket_qua_tung_cau, 'precision_at_k'):>6.2f} "
        f"{_trung_binh(ket_qua_tung_cau, 'recall_at_k'):>6.2f} "
        f"{_trung_binh(ket_qua_tung_cau, 'faithfulness'):>7.2f} "
        f"{_trung_binh(ket_qua_tung_cau, 'answer_relevance'):>7.2f} "
        f"{_trung_binh(ket_qua_tung_cau, 'citation_accuracy'):>7.2f} "
        f"{_trung_binh(ket_qua_tung_cau, 'do_tre_giay'):>7.1f}"
    )

    cac_cau_dao_dong = [
        kq for kq in ket_qua_tung_cau if float(kq.get("faithfulness_dao_dong") or 0) > 0
    ]
    if cac_cau_dao_dong:
        print()
        print(
            f"Độ ổn định của giám khảo: {len(cac_cau_dao_dong)}/{len(ket_qua_tung_cau)} câu "
            f"có điểm khác nhau giữa {config.SO_LAN_CHAM_FAITHFULNESS} lần chấm "
            f"(dao động lớn nhất "
            f"{max(float(kq['faithfulness_dao_dong']) for kq in cac_cau_dao_dong):.2f}). "
            "Bảng trên đã lấy TRUNG VỊ nên con số so sánh được giữa các lần chạy."
        )

    cac_cau_ngo = [kq for kq in ket_qua_tung_cau if _dang_ngo(kq)]
    if cac_cau_ngo:
        con_lai = [kq for kq in ket_qua_tung_cau if not _dang_ngo(kq)]
        print()
        print(
            f"⚠ {len(cac_cau_ngo)}/{len(ket_qua_tung_cau)} câu có điểm Faithfulness ĐÁNG NGỜ "
            f"(đánh dấu '!'): giám khảo chấm ≤ {config.NGUONG_DIEM_JUDGE_THAP} trong khi MỌI "
            f"câu của câu trả lời đều có ≥ {config.NGUONG_BAM_NGU_CANH_DE_NGHI_NGO:.0%} cụm "
            "từ nguyên văn trong ngữ cảnh - hai điều này không thể cùng đúng."
        )
        print(
            f"  Faithfulness nếu LOẠI các câu đó: "
            f"{_trung_binh(con_lai, 'faithfulness'):.2f} "
            f"(so với {_trung_binh(ket_qua_tung_cau, 'faithfulness'):.2f} khi tính cả)"
        )
        for kq in cac_cau_ngo:
            print(
                f"  - {kq['cau_hoi'][:60]}: chấm {float(kq['faithfulness']):.2f}, "
                f"câu bám ít nhất {float(kq.get('faithfulness_bam_ngu_canh', 0)):.0%}"
            )
        print("  → Đọc tay các câu này trước khi dùng con số Faithfulness làm căn cứ kết luận.")


def _in_bang_theo_nhom(ket_qua_tung_cau: List[Dict], khoa: str, nhan: str) -> None:
    """In điểm trung bình tách theo 1 trường phân nhóm (loai_tai_lieu / loai_cau_hoi)."""
    theo_nhom: Dict[str, List[Dict]] = defaultdict(list)
    for kq in ket_qua_tung_cau:
        if kq.get(khoa):
            theo_nhom[kq[khoa]].append(kq)
    if not theo_nhom:
        return

    print(f"\n=== Tách theo {nhan} ===")
    print(
        f"{nhan:<22} {'Số câu':>7} {'P@K':>6} {'R@K':>6} {'Faith':>7} {'Relev':>7} "
        f"{'Trích':>7} {'Giây':>7}"
    )
    print("-" * 78)
    for ten_nhom in sorted(theo_nhom):
        nhom = theo_nhom[ten_nhom]
        print(
            f"{ten_nhom:<22} {len(nhom):>7} "
            f"{_trung_binh(nhom, 'precision_at_k'):>6.2f} {_trung_binh(nhom, 'recall_at_k'):>6.2f} "
            f"{_trung_binh(nhom, 'faithfulness'):>7.2f} {_trung_binh(nhom, 'answer_relevance'):>7.2f} "
            f"{_trung_binh(nhom, 'citation_accuracy'):>7.2f} {_trung_binh(nhom, 'do_tre_giay'):>7.1f}"
        )


def doc_csv_ket_qua(duong_dan: Path) -> List[Dict]:
    with open(duong_dan, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def so_sanh_voi_ban_truoc(csv_moi: Path = None, csv_cu: Path = None) -> None:
    """In chênh lệch từng metric giữa 2 lần chạy - biến câu hỏi "thay đổi vừa rồi có thực
    sự cải thiện không" thành một con số, thay vì so 2 bảng bằng mắt."""
    csv_moi = csv_moi or DUONG_DAN_CSV_MAC_DINH
    csv_cu = csv_cu or DUONG_DAN_CSV_TRUOC
    if not Path(csv_cu).exists():
        print(f"\n(Chưa có '{Path(csv_cu).name}' để so sánh - đây là lần chạy đầu tiên.)")
        return

    moi, cu = doc_csv_ket_qua(csv_moi), doc_csv_ket_qua(csv_cu)

    cau_moi = {m.get("cau_hoi", "") for m in moi}
    cau_cu = {m.get("cau_hoi", "") for m in cu}
    chung = cau_moi & cau_cu
    if not chung:
        print(
            f"\n(BỎ QUA so sánh: '{Path(csv_cu).name}' dùng bộ câu hỏi hoàn toàn khác "
            f"({len(cau_cu)} câu, không câu nào trùng) - so sánh sẽ vô nghĩa. "
            "Xoá file đó đi nếu nó là kết quả của một bộ tài liệu cũ.)"
        )
        return
    if len(chung) < max(len(cau_moi), len(cau_cu)) * 0.8:
        print(
            f"\nCẢNH BÁO: chỉ {len(chung)} câu hỏi trùng nhau giữa 2 lần chạy "
            f"(mới {len(cau_moi)}, cũ {len(cau_cu)}) - số liệu so sánh bên dưới kém tin cậy."
        )

    print(f"\n=== So sánh với lần chạy trước ({Path(csv_cu).name}) ===")
    print(f"{'Metric':<20} {'Trước':>8} {'Sau':>8} {'Chênh lệch':>12}")
    print("-" * 52)
    for cot in CAC_COT_SO:
        truoc, sau = _trung_binh(cu, cot), _trung_binh(moi, cot)
        if truoc == 0.0 and sau == 0.0:
            continue
        chenh = sau - truoc
        tot_len = cot != "do_tre_giay"
        dau = "" if abs(chenh) < 1e-9 else ("✔" if (chenh > 0) == tot_len else "✘")
        print(f"{cot:<20} {truoc:>8.2f} {sau:>8.2f} {chenh:>+11.2f} {dau}")


def _xuat_csv(ket_qua_tung_cau: List[Dict], duong_dan: Path = None) -> None:
    duong_dan = duong_dan or DUONG_DAN_CSV_MAC_DINH
    duong_dan_truoc = (
        DUONG_DAN_CSV_TRUOC
        if Path(duong_dan) == DUONG_DAN_CSV_MAC_DINH
        else Path(duong_dan).with_name(Path(duong_dan).stem + "_truoc.csv")
    )
    if Path(duong_dan).exists():
        shutil.copy2(duong_dan, duong_dan_truoc)

    with open(duong_dan, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(ket_qua_tung_cau[0].keys()))
        writer.writeheader()
        writer.writerows(ket_qua_tung_cau)
    print(f"\nĐã lưu kết quả chi tiết tại: {duong_dan}")


def chay_danh_gia_nhanh(
    gioi_han: Optional[int] = None, duong_dan_cau_hoi: Path = None
) -> Optional[List[Dict]]:
    """Đo CHỈ phần truy xuất (Precision@K / Recall@K), không gọi LLM lần nào."""
    cac_cau_hoi = nap_bo_cau_hoi(duong_dan_cau_hoi)
    if gioi_han:
        cac_cau_hoi = cac_cau_hoi[:gioi_han]
    if not (config.FAISS_INDEX_FILE.exists() and config.METADATA_MAPPING_FILE.exists()):
        print("Chưa có FAISS index. Hãy build index trước.")
        return None

    embedding_service = EmbeddingService()
    vector_store = VectorStore.tai()
    ly_do = vector_store.ly_do_khong_tuong_thich()
    if ly_do:
        print(f"DỪNG: {ly_do}")
        return None
    reranker_service = tao_reranker_neu_bat()
    print(f"Rerank: {'BẬT (' + config.RERANKER_MODEL_NAME + ')' if reranker_service else 'TẮT'}")
    pipeline = RagPipeline(embedding_service, vector_store, reranker_service=reranker_service)

    ket_qua = []
    for muc in cac_cau_hoi:
        bat_dau = time.perf_counter()
        cac_chunk = pipeline.truy_xuat(muc["cau_hoi"])
        do_tre = time.perf_counter() - bat_dau
        ket_qua.append(
            {
                "cau_hoi": muc["cau_hoi"],
                "precision_at_k": precision_tai_k(cac_chunk, muc["cac_trang_dung"]),
                "recall_at_k": recall_tai_k(cac_chunk, muc["cac_trang_dung"]),
                "mrr": nghich_dao_thu_hang(cac_chunk, muc["cac_trang_dung"]),
                "thu_hang_dung": thu_hang_dung_dau_tien(cac_chunk, muc["cac_trang_dung"]),
                "so_doan": len(cac_chunk),
                "do_tre_giay": round(do_tre, 2),
                "loai_tai_lieu": muc.get("loai_tai_lieu", ""),
                "loai_cau_hoi": muc.get("loai_cau_hoi", ""),
            }
        )

    co_dap_an = [k for k in ket_qua if k["loai_cau_hoi"] != "tu_choi"]
    cau_tu_choi = [k for k in ket_qua if k["loai_cau_hoi"] == "tu_choi"]
    print(f"\n{'Câu hỏi':<52} {'P@K':>6} {'R@K':>6} {'Hạng':>5} {'MRR':>6} {'Giây':>6}")
    print("-" * 90)
    for k in ket_qua:
        ten = (k["cau_hoi"][:49] + "...") if len(k["cau_hoi"]) > 52 else k["cau_hoi"]
        hang = k["thu_hang_dung"] or "-"
        print(f"{ten:<52} {k['precision_at_k']:>6.2f} {k['recall_at_k']:>6.2f} "
              f"{hang:>5} {k['mrr']:>6.2f} {k['do_tre_giay']:>6.2f}")
    print("-" * 90)
    print(f"{'TRUNG BÌNH (câu CÓ đáp án)':<52} "
          f"{_trung_binh(co_dap_an, 'precision_at_k'):>6.2f} "
          f"{_trung_binh(co_dap_an, 'recall_at_k'):>6.2f} "
          f"{'':>5} {_trung_binh(co_dap_an, 'mrr'):>6.2f} "
          f"{_trung_binh(ket_qua, 'do_tre_giay'):>6.2f}")
    so_hang_1 = sum(1 for k in co_dap_an if k["thu_hang_dung"] == 1)
    print(f"Đoạn đúng xếp HẠNG 1: {so_hang_1}/{len(co_dap_an)} câu")
    if cau_tu_choi:
        so_chan_duoc = sum(1 for k in cau_tu_choi if k["so_doan"] == 0)
        print(f"Câu lạc đề bị chặn ở tầng truy xuất: {so_chan_duoc}/{len(cau_tu_choi)} "
              f"(phần còn lại do LLM từ chối)")
    _in_bang_theo_nhom(co_dap_an, "loai_tai_lieu", "Loại tài liệu")
    return ket_qua


def do_khoang_cach_held_out(gioi_han: Optional[int] = None) -> None:
    """Chạy CẢ HAI bộ câu hỏi rồi in chênh lệch - con số đo mức OVERFIT của hệ thống."""
    print("=" * 92)
    print("BỘ IN-SAMPLE (đã dùng để hiệu chỉnh mọi tham số)")
    print("=" * 92)
    trong_mau = chay_danh_gia_nhanh(gioi_han, config.TEST_QUESTIONS_FILE)

    print()
    print("=" * 92)
    print("BỘ HELD-OUT (tài liệu chưa từng dùng để chỉnh tham số nào)")
    print("=" * 92)
    ngoai_mau = chay_danh_gia_nhanh(gioi_han, config.TEST_QUESTIONS_HELD_OUT_FILE)

    if not trong_mau or not ngoai_mau:
        return

    def _co_dap_an(kq):
        return [k for k in kq if k["loai_cau_hoi"] != "tu_choi"]

    print()
    print("=" * 92)
    print("KHOẢNG CÁCH IN-SAMPLE vs HELD-OUT (mức overfit của hệ thống)")
    print("=" * 92)
    print(f"{'Metric':<20} {'in-sample':>11} {'held-out':>11} {'khoảng cách':>13}")
    print("-" * 60)
    for cot in ("precision_at_k", "recall_at_k", "mrr"):
        a = _trung_binh(_co_dap_an(trong_mau), cot)
        b = _trung_binh(_co_dap_an(ngoai_mau), cot)
        print(f"{cot:<20} {a:>11.3f} {b:>11.3f} {a - b:>+13.3f}")
    print()
    print(
        "Khoảng cách dương lớn = hệ thống đang bám vào chính bộ tài liệu đã dùng để "
        "hiệu chỉnh."
    )
    print(
        "Sau MỖI thay đổi, báo cáo lại CẢ HAI con số chứ không chỉ con số in-sample."
    )


def chay_danh_gia(
    gioi_han: Optional[int] = None, duong_dan_cau_hoi: Path = None
) -> None:
    cac_cau_hoi = nap_bo_cau_hoi(duong_dan_cau_hoi)
    if not cac_cau_hoi:
        print(
            f"Chưa có câu hỏi nào trong '{config.TEST_QUESTIONS_FILE.name}'.\n"
            "Hãy thêm câu hỏi test theo đúng định dạng (xem README) rồi chạy lại."
        )
        return
    if gioi_han:
        cac_cau_hoi = cac_cau_hoi[:gioi_han]

    if not (config.FAISS_INDEX_FILE.exists() and config.METADATA_MAPPING_FILE.exists()):
        print("Chưa có FAISS index. Hãy build index qua Streamlit UI (app.py) trước rồi chạy lại evaluation.")
        return

    print("Đang tải model embedding và FAISS index...")
    embedding_service = EmbeddingService()
    vector_store = VectorStore.tai()
    ly_do = vector_store.ly_do_khong_tuong_thich()
    if ly_do:
        print(f"DỪNG: {ly_do}\nHãy build lại index (qua app.py) rồi chạy lại evaluation.")
        return
    reranker_service = tao_reranker_neu_bat()
    print(f"Rerank: {'BẬT (' + config.RERANKER_MODEL_NAME + ')' if reranker_service else 'TẮT'}")
    pipeline = RagPipeline(embedding_service, vector_store, reranker_service=reranker_service)

    ket_qua_tung_cau = []
    for i, muc in enumerate(cac_cau_hoi, start=1):
        cau_hoi = muc["cau_hoi"]
        cac_trang_dung = muc["cac_trang_dung"]
        print(f"[{i}/{len(cac_cau_hoi)}] Đang đánh giá: {cau_hoi}")

        bat_dau = time.perf_counter()
        cac_chunk = pipeline.truy_xuat(cau_hoi)
        cau_tra_loi = pipeline.sinh_cau_tra_loi(cau_hoi, cac_chunk)
        do_tre = time.perf_counter() - bat_dau

        diem_faithfulness = faithfulness(cau_tra_loi, cac_chunk)
        diem_relevance = answer_relevance(cau_hoi, cau_tra_loi)
        diem_trich_dan = do_chinh_xac_trich_dan(cau_tra_loi, cac_chunk)

        ket_qua_tung_cau.append(
            {
                "cau_hoi": cau_hoi,
                "cau_tra_loi": cau_tra_loi,
                "precision_at_k": precision_tai_k(cac_chunk, cac_trang_dung),
                "recall_at_k": recall_tai_k(cac_chunk, cac_trang_dung),
                "faithfulness": diem_faithfulness["diem"],
                "faithfulness_dang_ngo": int(diem_faithfulness.get("dang_ngo", False)),
                "faithfulness_dao_dong": round(diem_faithfulness.get("dao_dong_judge", 0.0), 2),
                "faithfulness_bam_ngu_canh": round(
                    diem_faithfulness.get("bam_ngu_canh_thap_nhat", 0.0), 3
                ),
                "answer_relevance": diem_relevance["diem"],
                "citation_accuracy": diem_trich_dan["diem"],
                "so_cap_trich_dan_da_kiem": diem_trich_dan["so_cap_da_kiem"],
                "do_tre_giay": round(do_tre, 2),
                "loai_tai_lieu": muc.get("loai_tai_lieu", ""),
                "loai_cau_hoi": muc.get("loai_cau_hoi", ""),
            }
        )

    print()
    _in_bang_ket_qua(ket_qua_tung_cau)
    _in_bang_theo_nhom(ket_qua_tung_cau, "loai_tai_lieu", "Loại tài liệu")
    _in_bang_theo_nhom(ket_qua_tung_cau, "loai_cau_hoi", "Loại câu hỏi")
    duong_dan_csv = (
        DUONG_DAN_CSV_HELD_OUT
        if duong_dan_cau_hoi and Path(duong_dan_cau_hoi) == config.TEST_QUESTIONS_HELD_OUT_FILE
        else DUONG_DAN_CSV_MAC_DINH
    )
    _xuat_csv(ket_qua_tung_cau, duong_dan_csv)
    so_sanh_voi_ban_truoc(
        duong_dan_csv,
        duong_dan_csv.with_name(duong_dan_csv.stem + "_truoc.csv")
        if duong_dan_csv != DUONG_DAN_CSV_MAC_DINH
        else DUONG_DAN_CSV_TRUOC,
    )


if __name__ == "__main__":
    CAC_CO = {"--nhanh", "--held-out", "--khoang-cach"}
    tham_so = [t for t in sys.argv[1:] if t not in CAC_CO]
    gioi_han_dong_lenh = int(tham_so[0]) if tham_so else None
    bo_cau_hoi = (
        config.TEST_QUESTIONS_HELD_OUT_FILE
        if "--held-out" in sys.argv
        else config.TEST_QUESTIONS_FILE
    )
    if "--khoang-cach" in sys.argv:
        do_khoang_cach_held_out(gioi_han=gioi_han_dong_lenh)
    elif "--nhanh" in sys.argv:
        chay_danh_gia_nhanh(gioi_han=gioi_han_dong_lenh, duong_dan_cau_hoi=bo_cau_hoi)
    else:
        chay_danh_gia(gioi_han=gioi_han_dong_lenh, duong_dan_cau_hoi=bo_cau_hoi)
