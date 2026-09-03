"""Chạy đánh giá hệ thống RAG trên bộ câu hỏi test (evaluation/test_questions.json).

Cách chạy: python evaluation/run_evaluation.py (từ thư mục gốc project).
Yêu cầu: đã build FAISS index (qua Streamlit UI) và đã điền câu hỏi vào test_questions.json.

Ngoài bảng tổng, script còn in bảng TÁCH THEO NHÓM (loại tài liệu / loại câu hỏi) và cho
phép SO SÁNH với lần chạy trước. Lý do: một con số trung bình duy nhất che mất đúng thứ cần
biết - hệ thống có thể tăng điểm trên tài liệu dài nhưng tụt trên tài liệu ngắn, hoặc trả lời
tốt hơn nhưng lại hết biết từ chối câu lạc đề, mà trung bình cộng vẫn "đẹp". Tách nhóm khiến
những đánh đổi đó lộ ra thay vì bị trung bình hoá.
"""

import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Cho phép chạy trực tiếp bằng "python evaluation/run_evaluation.py" (không chỉ
# "python -m evaluation.run_evaluation") bằng cách tự thêm thư mục gốc project vào
# sys.path - nếu không, "import config" ở dưới sẽ báo lỗi ModuleNotFoundError vì
# Python chỉ tự thêm thư mục chứa file evaluation/ vào sys.path, không phải thư mục gốc.
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

# Các cột điểm để in bảng/so sánh. Gom vào 1 chỗ để thêm metric mới (vd citation accuracy ở
# giai đoạn sau) chỉ phải sửa đúng danh sách này, không phải sửa rải rác 3-4 hàm.
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
# Bộ held-out ghi ra FILE RIÊNG. Nếu dùng chung file với bộ in-sample thì lần chạy sau đè lên
# lần trước, và bảng "so sánh với lần chạy trước" sẽ đem hai bộ câu hỏi khác nhau ra so - đúng
# kiểu số liệu trông thuyết phục mà vô nghĩa mà so_sanh_voi_ban_truoc() đã phải thêm cơ chế
# chặn. Tách file là cách chặn rẻ hơn: hai bộ không bao giờ chạm vào nhau.
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
        # citation_accuracy = None nghĩa là câu trả lời không dẫn nguồn nào (câu từ chối) -
        # in "  -  " thay vì 0.00 để không nhìn nhầm thành "trích dẫn sai hết".
        trich = kq.get("citation_accuracy")
        trich_hien = f"{float(trich):>7.2f}" if trich not in (None, "") else f"{'-':>7}"
        # Dấu "!" ngay cạnh điểm Faithfulness: điểm này giám khảo chấm thấp nhưng câu trả
        # lời lại chép gần nguyên văn ngữ cảnh, tức nhiều khả năng giám khảo sai. Đánh dấu
        # tại chỗ để không ai đọc lướt bảng rồi đi tối ưu vào một lỗi không tồn tại.
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
    """In điểm trung bình tách theo 1 trường phân nhóm (loai_tai_lieu / loai_cau_hoi).

    Bỏ qua im lặng nếu bộ câu hỏi chưa khai báo trường đó - giữ tương thích với
    test_questions.json cũ (chỉ có cau_hoi/cac_trang_dung/dap_an_mau).
    """
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
    sự cải thiện không" thành một con số, thay vì so 2 bảng bằng mắt.

    Đây là công cụ then chốt cho nguyên tắc "mỗi thay đổi phải đo trước/sau rồi mới giữ":
    không có nó thì rất dễ giữ lại một thay đổi nghe hợp lý nhưng thực tế làm tệ đi.
    """
    csv_moi = csv_moi or DUONG_DAN_CSV_MAC_DINH
    csv_cu = csv_cu or DUONG_DAN_CSV_TRUOC
    if not Path(csv_cu).exists():
        print(f"\n(Chưa có '{Path(csv_cu).name}' để so sánh - đây là lần chạy đầu tiên.)")
        return

    moi, cu = doc_csv_ket_qua(csv_moi), doc_csv_ket_qua(csv_cu)

    # Chỉ so được khi HAI LẦN CHẠY DÙNG CÙNG BỘ CÂU HỎI. Đã gặp thực tế: file cũ còn sót lại
    # từ lần chạy trên một bộ tài liệu HOÀN TOÀN KHÁC, bảng chênh lệch vẫn in ra bình thường
    # với những con số trông rất thuyết phục nhưng vô nghĩa - đúng loại sai lệch dễ khiến
    # người ta kết luận nhầm về việc thay đổi vừa rồi có ích hay không.
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
        # Với độ trễ thì TĂNG là xấu, với các metric chất lượng thì tăng là tốt - đánh dấu
        # theo đúng hướng để đọc nhanh không bị nhầm dấu.
        tot_len = cot != "do_tre_giay"
        dau = "" if abs(chenh) < 1e-9 else ("✔" if (chenh > 0) == tot_len else "✘")
        print(f"{cot:<20} {truoc:>8.2f} {sau:>8.2f} {chenh:>+11.2f} {dau}")


def _xuat_csv(ket_qua_tung_cau: List[Dict], duong_dan: Path = None) -> None:
    duong_dan = duong_dan or DUONG_DAN_CSV_MAC_DINH
    # Giữ lại kết quả lần trước để so sánh được ngay ở lần chạy kế tiếp mà người dùng không
    # phải nhớ tự copy file thủ công trước mỗi lần chạy. Bản sao đi kèm ĐÚNG file kết quả của
    # nó (thêm hậu tố "_truoc"), không dồn chung một chỗ - nếu không thì chạy bộ held-out sẽ
    # ghi đè bản sao của bộ in-sample và lần so sánh sau đó đem hai bộ khác nhau ra đối chiếu.
    duong_dan_truoc = (
        DUONG_DAN_CSV_TRUOC
        if Path(duong_dan) == DUONG_DAN_CSV_MAC_DINH
        else Path(duong_dan).with_name(Path(duong_dan).stem + "_truoc.csv")
    )
    if Path(duong_dan).exists():
        shutil.copy2(duong_dan, duong_dan_truoc)

    # utf-8-sig (có BOM) để Excel trên Windows mở file mở đúng tiếng Việt có dấu,
    # thay vì hiển thị ký tự lỗi như với utf-8 thường.
    with open(duong_dan, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(ket_qua_tung_cau[0].keys()))
        writer.writeheader()
        writer.writerows(ket_qua_tung_cau)
    print(f"\nĐã lưu kết quả chi tiết tại: {duong_dan}")


def chay_danh_gia_nhanh(
    gioi_han: Optional[int] = None, duong_dan_cau_hoi: Path = None
) -> Optional[List[Dict]]:
    """Đo CHỈ phần truy xuất (Precision@K / Recall@K), không gọi LLM lần nào.

    Vì sao cần chế độ này: bản đầy đủ tốn 3 lượt gọi LLM mỗi câu (1 sinh câu trả lời + 2
    chấm điểm bằng judge), tức 2-3 phút/câu - cả bộ mất từ một tới hai tiếng.
    Nhưng phần lớn thay đổi cần đo (rerank, chunking, trích xuất bảng/ảnh) chỉ ảnh hưởng
    tới TRUY XUẤT, mà Precision@K/Recall@K lại tính được hoàn toàn không cần LLM.

    Tách riêng chế độ nhanh biến vòng lặp "sửa - đo - quyết định" từ hàng giờ xuống vài
    giây. Chạy bản đầy đủ khi cần đo chất lượng câu trả lời (faithfulness/relevance).
    """
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
                # MRR/thứ hạng: metric NHẠY với thứ tự, khác với P@K/R@K vốn chỉ so tập hợp.
                # Đây là metric chính để đánh giá rerank (xem docstring thu_hang_dung_dau_tien).
                "mrr": nghich_dao_thu_hang(cac_chunk, muc["cac_trang_dung"]),
                "thu_hang_dung": thu_hang_dung_dau_tien(cac_chunk, muc["cac_trang_dung"]),
                # Câu từ chối (cac_trang_dung rỗng) thì P@K/R@K luôn = 0 theo định nghĩa,
                # không phản ánh chất lượng. Đo riêng "có chặn được không" bằng số đoạn
                # truy xuất được: 0 đoạn = hệ thống tự biết từ chối, đúng hành vi mong muốn.
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
    """Chạy CẢ HAI bộ câu hỏi rồi in chênh lệch - con số đo mức OVERFIT của hệ thống.

    Vì sao con số này quan trọng hơn từng con số riêng lẻ: mọi hằng số của hệ thống (ngưỡng
    cosine, ngưỡng rerank, trần đoạn mỗi trang, số ứng viên rerank, chunk size) đều được
    chọn bằng cách tối ưu trên chính bộ test_questions.json, tức tuning trên tập test. Điểm
    in-sample vì thế luôn đẹp và không nói được gì về tài liệu mới. Khoảng cách giữa hai bộ
    mới là thứ nói được, và nó phải được BÁO CÁO LẠI SAU MỖI THAY ĐỔI:
      - Khoảng cách thu hẹp -> thay đổi giúp hệ thống tổng quát hơn.
      - Cả hai cùng tăng nhưng khoảng cách giữ nguyên -> thay đổi tốt nhưng không chữa overfit.
      - In-sample tăng còn held-out giảm -> đang tối ưu vào đúng bộ tài liệu cũ, phải bỏ.

    QUY TẮC BẤT DI BẤT DỊCH: không bao giờ chỉnh tham số theo kết quả của bộ held-out. Chỉnh
    một lần là nó thành bộ in-sample thứ hai và con số này mất hết ý nghĩa.
    """
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
    # Index build bằng model embedding khác với model đang dùng thì kết quả truy xuất sai
    # hoàn toàn mà không hề báo lỗi - chạy đánh giá trên đó sẽ ra một bảng số liệu trông
    # bình thường nhưng vô nghĩa, nên phải dừng lại thay vì chỉ cảnh báo.
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

        # Chỉ tính giờ phần TRUY XUẤT + SINH CÂU TRẢ LỜI (không tính phần chấm điểm bằng
        # judge) - đây mới là độ trễ người dùng thật sự chịu khi hỏi trên UI.
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
                # Cờ tự nghi ngờ của chính thước đo: giám khảo chấm thấp NHƯNG câu trả lời
                # chép gần nguyên văn ngữ cảnh - hai điều không thể cùng đúng (xem
                # metrics.faithfulness). Ghi ra CSV để đọc lại được, và để bảng kết quả báo
                # riêng trung bình đã loại các câu này.
                "faithfulness_dang_ngo": int(diem_faithfulness.get("dang_ngo", False)),
                # Biên độ dao động giữa các lần chấm cùng một câu. Khác 0 nghĩa là giám
                # khảo tự mâu thuẫn với chính nó - số này thuộc về ĐỘ TIN CẬY CỦA THƯỚC ĐO,
                # cần có trong báo cáo bên cạnh điểm Faithfulness.
                "faithfulness_dao_dong": round(diem_faithfulness.get("dao_dong_judge", 0.0), 2),
                # Mức bám của CÂU TỆ NHẤT - đây mới là con số quyết định cờ (xem
                # metrics.do_bam_ngu_canh_thap_nhat), nên ghi đúng nó ra CSV.
                "faithfulness_bam_ngu_canh": round(
                    diem_faithfulness.get("bam_ngu_canh_thap_nhat", 0.0), 3
                ),
                "answer_relevance": diem_relevance["diem"],
                # None khi câu trả lời không dẫn nguồn nào (câu từ chối) - _trung_binh() bỏ
                # qua, để câu từ chối hợp lệ không kéo điểm trích dẫn xuống oan.
                "citation_accuracy": diem_trich_dan["diem"],
                "so_cap_trich_dan_da_kiem": diem_trich_dan["so_cap_da_kiem"],
                "do_tre_giay": round(do_tre, 2),
                # .get() chứ không phải [...]: 2 trường này là TUỲ CHỌN, bộ câu hỏi cũ
                # không có chúng vẫn phải chạy được bình thường.
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
    # python evaluation/run_evaluation.py                -> đầy đủ (có gọi LLM, chậm)
    # python evaluation/run_evaluation.py --nhanh        -> chỉ đo truy xuất (không gọi LLM)
    # python evaluation/run_evaluation.py --nhanh 5      -> chỉ đo 5 câu đầu
    # python evaluation/run_evaluation.py --held-out     -> đo trên bộ HELD-OUT
    # python evaluation/run_evaluation.py --khoang-cach  -> đo CẢ HAI bộ + in mức overfit
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
