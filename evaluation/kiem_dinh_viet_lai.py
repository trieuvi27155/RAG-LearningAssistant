"""Đo chất lượng của bước VIẾT LẠI CÂU HỎI NỐI TIẾP, trên bộ ca đã biết trước đáp án."""

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.tiep_noi_hoi_thoai import la_cau_hoi_tiep_noi, viet_lai_cau_hoi

_HOI_THOAI_LUAT = [
    {"role": "user", "content": "Vi phạm pháp luật gồm những dấu hiệu nào?"},
    {"role": "assistant", "content": "Vi phạm pháp luật có bốn dấu hiệu: hành vi trái pháp "
                                     "luật, có lỗi của chủ thể, do chủ thể có năng lực trách "
                                     "nhiệm pháp lý thực hiện, và xâm hại quan hệ xã hội "
                                     "được pháp luật bảo vệ."},
]

_HOI_THOAI_ML = [
    {"role": "user", "content": "What is overfitting in machine learning?"},
    {"role": "assistant", "content": "Overfitting happens when a model learns the training "
                                     "data too closely, including its noise, so it performs "
                                     "poorly on unseen data."},
]

_HOI_THOAI_THU_VIEN = [
    {"role": "user", "content": "Quy trình mượn tài liệu ở thư viện gồm những bước nào?"},
    {"role": "assistant", "content": "Quy trình gồm ba bước: tra cứu trên hệ thống, đăng ký "
                                     "mượn tại quầy, và nhận tài liệu kèm phiếu hẹn trả."},
]

CAC_CA = [
    (_HOI_THOAI_LUAT, "Thế còn dấu hiệu thứ hai thì sao?", True, ["dấu hiệu", "vi phạm pháp luật"]),
    (_HOI_THOAI_LUAT, "Cái đó khác gì với vi phạm hành chính?", True, ["vi phạm pháp luật"]),
    (_HOI_THOAI_LUAT, "Cho ví dụ đi", True, ["vi phạm pháp luật"]),
    (_HOI_THOAI_ML, "What about the second one?", True, ["overfitting"]),
    (_HOI_THOAI_ML, "Tell me more", True, ["overfitting"]),
    (_HOI_THOAI_THU_VIEN, "Vậy còn bước cuối cùng?", True, ["mượn tài liệu"]),
    (_HOI_THOAI_THU_VIEN, "Giải thích thêm về bước đầu tiên", True, ["mượn tài liệu"]),
    (_HOI_THOAI_LUAT, "Nhà nước có những đặc điểm gì?", False, []),
    (_HOI_THOAI_ML, "How is the late return fee calculated?", False, []),
    (_HOI_THOAI_THU_VIEN, "Trình bày khái niệm quy phạm pháp luật.", False, []),
]


def _khong_dau(s: str) -> str:
    """So khớp từ khoá bỏ qua dấu và hoa/thường."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn"
    )


def main() -> None:
    bo_phan_tich = argparse.ArgumentParser(description=__doc__)
    bo_phan_tich.add_argument("--chi-tang-1", action="store_true",
                              help="chỉ đo tầng nhận diện (tất định, không cần Ollama)")
    bo_phan_tich.add_argument(
        "--truy-xuat", metavar="CÂU_HỎI",
        help="đo ảnh hưởng THẬT lên truy xuất, trên index đang có. Truyền vào MỘT câu hỏi "
             "đầy đủ về tài liệu của bạn - nó vừa làm lượt hỏi trước vừa làm chuẩn vàng.")
    doi_so = bo_phan_tich.parse_args()

    if doi_so.truy_xuat:
        _do_anh_huong_truy_xuat(doi_so.truy_xuat)
        return

    print("\nKIỂM ĐỊNH VIẾT LẠI CÂU HỎI NỐI TIẾP")
    print(f"Model viết lại: {config.OLLAMA_MODEL}\n")

    print("TẦNG 1 — NHẬN DIỆN (tất định, không gọi model)")
    print(f"{'Câu hỏi':<48}{'Kỳ vọng':>12}{'Thực tế':>12}{'':>5}")
    print("-" * 78)
    so_dung_t1 = 0
    for hoi_thoai, cau_hoi, la_tiep_noi, _ in CAC_CA:
        thuc_te = la_cau_hoi_tiep_noi(cau_hoi, hoi_thoai)
        dat = thuc_te == la_tiep_noi
        so_dung_t1 += dat
        print(f"{cau_hoi[:46]:<48}{('nối tiếp' if la_tiep_noi else 'đi thẳng'):>12}"
              f"{('nối tiếp' if thuc_te else 'đi thẳng'):>12}{('  ✔' if dat else '  ✘'):>5}")
    print("-" * 78)
    print(f"Nhận diện đúng: {so_dung_t1}/{len(CAC_CA)} ca ({so_dung_t1 / len(CAC_CA):.0%})\n")

    if doi_so.chi_tang_1:
        return

    cac_ca_viet_lai = [c for c in CAC_CA if c[2]]
    print("TẦNG 2 — VIẾT LẠI (cần Ollama đang chạy)")
    print(f"{'Câu gốc':<34}{'Bản viết lại':<40}{'':>5}")
    print("-" * 78)
    so_dung_t2 = 0
    for hoi_thoai, cau_hoi, _, cac_tu_khoa in cac_ca_viet_lai:
        viet_lai = viet_lai_cau_hoi(cau_hoi, hoi_thoai)
        gon = _khong_dau(viet_lai)
        dat = viet_lai != cau_hoi and all(_khong_dau(tu) in gon for tu in cac_tu_khoa)
        so_dung_t2 += dat
        print(f"{cau_hoi[:32]:<34}{viet_lai[:38]:<40}{('  ✔' if dat else '  ✘'):>5}")

    print("-" * 78)
    ty_le = so_dung_t2 / len(cac_ca_viet_lai)
    print(f"Viết lại đủ chủ đề: {so_dung_t2}/{len(cac_ca_viet_lai)} ca ({ty_le:.0%})")

    print()
    if ty_le < 0.7:
        print(
            f"✘ Chỉ {ty_le:.0%} bản viết lại mang đủ chủ đề. Cần xem lại PROMPT_VIET_LAI, "
            "hoặc dùng model lớn hơn. LƯU Ý: mức này KHÔNG có nghĩa hệ thống trả lời sai "
            f"{1 - ty_le:.0%} số câu — các chốt chặn trong viet_lai_cau_hoi() lùi về câu gốc "
            "khi bản viết lại không đạt, nên trường hợp xấu nhất là hành vi của bản chưa có "
            "tính năng này, không phải hành vi sai."
        )
    else:
        print(f"✔ {ty_le:.0%} bản viết lại mang đủ chủ đề — con số nên đặt cạnh tính năng "
              "trong báo cáo.")


CAU_NOI_TIEP_TRUNG_TINH = [
    "Giải thích thêm đi",
    "Cho ví dụ",
    "Tell me more",
    "Cái đó cụ thể là thế nào?",
]


def _do_anh_huong_truy_xuat(cau_hoi_goc: str) -> None:
    """Đo ảnh hưởng THẬT của việc ghép ngữ cảnh lên truy xuất, trên index đang có."""
    from rag.embedding import EmbeddingService
    from rag.rag_pipeline import RagPipeline
    from rag.reranker import RerankerService
    from rag.vector_store import VectorStore

    svc = EmbeddingService()
    store = VectorStore.tai()
    if store.so_luong_vector == 0:
        print("Chưa có index nào. Build index trước (bấm 'Đọc tài liệu' trên UI) rồi chạy lại.")
        return
    pipeline = RagPipeline(svc, store, RerankerService())

    def _khoa(cac_doan):
        return {(d["nguon"], d["trang"]) for d in cac_doan}

    chuan_vang = _khoa(pipeline.truy_xuat(cau_hoi_goc))
    if not chuan_vang:
        print(f"Câu hỏi gốc {cau_hoi_goc!r} không lấy về đoạn nào - nó nằm ngoài phạm vi "
              "tài liệu đang index. Hãy chọn một câu hỏi thật sự về tài liệu của bạn.")
        return

    hoi_thoai = [
        {"role": "user", "content": cau_hoi_goc},
        {"role": "assistant", "content": "(câu trả lời của lượt trước)"},
    ]

    print()
    print(f"ẢNH HƯỞNG LÊN TRUY XUẤT — index {store.so_luong_vector} chunk")
    print(f"Câu hỏi gốc (làm chuẩn vàng): {cau_hoi_goc}")
    print(f"Chuẩn vàng: {len(chuan_vang)} (nguồn, trang) · "
          f"ngưỡng TỪ CHỐI theo rerank: {config.NGUONG_DIEM_RERANK_TOI_THIEU}")
    print()
    print(f"{'Câu nối tiếp':<28}{'trùng chuẩn vàng':>34}{'điểm rerank':>26}")
    print(f"{'':<28}{'không NC':>16}{'có NC':>18}{'không NC':>13}{'có NC':>13}")
    print("-" * 88)

    tong_khong, tong_co, so_sat_nguong = 0, 0, 0
    for cau in CAU_NOI_TIEP_TRUNG_TINH:
        doan_khong = pipeline.truy_xuat(cau, lich_su=None)
        rr_khong = pipeline.diem_rerank_cao_nhat
        doan_co = pipeline.truy_xuat(cau, lich_su=hoi_thoai)
        rr_co = pipeline.diem_rerank_cao_nhat

        trung_khong = len(_khoa(doan_khong) & chuan_vang)
        trung_co = len(_khoa(doan_co) & chuan_vang)
        tong_khong += trung_khong
        tong_co += trung_co
        so_sat_nguong += rr_khong is not None and rr_khong < 0.01

        def _r(x):
            return f"{x:.4f}" if x is not None else "TỪ CHỐI"

        print(f"{cau[:26]:<28}{f'{trung_khong}/{len(chuan_vang)}':>16}"
              f"{f'{trung_co}/{len(chuan_vang)}':>18}{_r(rr_khong):>13}{_r(rr_co):>13}")

    print("-" * 88)
    toi_da = len(chuan_vang) * len(CAU_NOI_TIEP_TRUNG_TINH)
    print(f"Tổng trùng chuẩn vàng: {tong_khong}/{toi_da} (không ngữ cảnh) -> "
          f"{tong_co}/{toi_da} (có ngữ cảnh)")
    if so_sat_nguong:
        print(f"⚠ {so_sat_nguong}/{len(CAU_NOI_TIEP_TRUNG_TINH)} câu nối tiếp KHÔNG có ngữ "
              "cảnh rơi xuống dưới 0.01 điểm rerank — sát vùng bị TỪ CHỐI OAN (§5.29). Đây "
              "là hậu quả nghiêm trọng hơn 'lấy nhầm đoạn': người dùng mất hẳn câu trả lời.")


if __name__ == "__main__":
    main()
