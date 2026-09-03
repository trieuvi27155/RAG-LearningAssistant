"""Đo độ tin cậy của chính CƠ CHẾ PHÁT HIỆN MÂU THUẪN, không phải của hệ thống RAG.

VÌ SAO CÓ FILE NÀY
------------------
Cảnh báo "hai nguồn của bạn nói khác nhau" là loại thông tin người dùng KHÔNG có cách nào
tự kiểm rẻ: họ mở từng file ra thì mỗi file đều nhất quán với chính nó. Nếu cảnh báo đó sai,
người dùng mất niềm tin vào chính tài liệu học tập của mình - và họ sẽ không biết là mình
vừa bị báo nhầm.

Đây đúng là dạng lỗi âm thầm mà cả đồ án tìm cách loại bỏ, nên cơ chế này phải tự khai báo
độ tin cậy của nó, giống hệt cách kiem_dinh_judge.py làm với Faithfulness (§5.43).

Bộ ca dưới đây cố ý gồm CẢ ca phải im lặng - và đó mới là phần khó. Bắt được mâu thuẫn hiển
nhiên thì dễ; khó là KHÔNG báo động trên hai đoạn chỉ bổ sung cho nhau, hoặc cùng một ý diễn
đạt bằng từ khác. Ba trên bảy ca ở đây là ca im lặng, và một trong số đó (bổ sung, khác số)
được dựng riêng để qua được tầng lọc tất định rồi bị tầng LLM bác - tức nó kiểm đúng phần
mà tầng lọc rẻ không làm được.

Con số rút ra từ đây là thứ nên đưa vào báo cáo bên cạnh tính năng: không phải "hệ thống
phát hiện được mâu thuẫn giữa các nguồn" mà "phát hiện được, với X/7 ca đúng trên bộ kiểm
định, trong đó Y/3 ca im lặng đúng".

CÁCH CHẠY
---------
    python evaluation/kiem_dinh_doi_chieu.py
    python evaluation/kiem_dinh_doi_chieu.py --so-lan 3   # đo thêm độ ổn định giữa các lần
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.doi_chieu_nguon import cac_cap_dang_ngo, tim_mau_thuan


def _doan(nguon, trang, noidung):
    return {"nguon": nguon, "trang": trang, "noidung": noidung, "doan_khop": noidung,
            "diem_similarity": 0.9}


# Mỗi ca: (tên, đoạn A, đoạn B, CÓ PHẢI mâu thuẫn thật không).
CAC_CA_KIEM_DINH = [
    (
        "MÂU THUẪN — khác số lượng (số viết bằng chữ)",
        _doan("giaotrinh.pdf", 12, "Nhà nước có năm đặc điểm cơ bản: tính giai cấp, quyền "
                                   "lực công cộng đặc biệt, chủ quyền quốc gia, quyền ban "
                                   "hành pháp luật và quyền thu thuế."),
        _doan("slide.pptx", 3, "Nhà nước có bốn đặc điểm cơ bản: tính giai cấp, quyền lực "
                               "công cộng đặc biệt, chủ quyền quốc gia và quyền ban hành "
                               "pháp luật."),
        True,
    ),
    (
        "MÂU THUẪN — khác số liệu cụ thể",
        _doan("quydinh_2019.pdf", 4, "Phí phạt trả sách quá hạn là 2.000 đồng mỗi ngày cho "
                                     "mỗi đầu sách."),
        _doan("quydinh_2024.pdf", 2, "Phí phạt trả sách quá hạn là 5.000 đồng mỗi ngày cho "
                                     "mỗi đầu sách."),
        True,
    ),
    (
        "MÂU THUẪN — khẳng định đối lập",
        _doan("baigiang.pptx", 8, "Doanh nghiệp tư nhân có tư cách pháp nhân kể từ ngày "
                                  "được cấp giấy chứng nhận đăng ký doanh nghiệp."),
        _doan("luatdoanhnghiep.pdf", 21, "Doanh nghiệp tư nhân không có tư cách pháp nhân "
                                         "vì tài sản của doanh nghiệp không tách bạch với "
                                         "tài sản của chủ doanh nghiệp."),
        True,
    ),
    (
        "MÂU THUẪN — khác định nghĩa cùng một khái niệm",
        _doan("mon_a.pdf", 5, "Overfitting là hiện tượng mô hình học thuộc dữ liệu huấn "
                              "luyện nên hoạt động kém trên dữ liệu mới."),
        _doan("mon_b.pdf", 7, "Overfitting là hiện tượng mô hình quá đơn giản nên không "
                              "nắm được quy luật trong dữ liệu huấn luyện."),
        True,
    ),
    (
        "IM LẶNG — bổ sung cho nhau (dù khác số)",
        _doan("chuong1.pdf", 3, "Điều kiện thứ nhất để được xét học bổng là điểm trung bình "
                                "từ 8.0 trở lên."),
        _doan("chuong2.pdf", 9, "Điều kiện thứ hai để được xét học bổng là điểm rèn luyện "
                                "từ 80 trở lên."),
        False,
    ),
    (
        "IM LẶNG — cùng một ý, diễn đạt khác",
        _doan("giaotrinh.pdf", 30, "Quy phạm pháp luật là quy tắc xử sự chung do nhà nước "
                                   "ban hành và bảo đảm thực hiện."),
        _doan("slide.pptx", 11, "Nhà nước ban hành quy phạm pháp luật - tức các quy tắc xử "
                                "sự chung - và bảo đảm cho chúng được thực hiện."),
        False,
    ),
    (
        "IM LẶNG — một bên chi tiết hơn, không nói ngược",
        _doan("tomtat.pdf", 1, "Vi phạm pháp luật có bốn dấu hiệu."),
        _doan("chitiet.pdf", 14, "Vi phạm pháp luật có bốn dấu hiệu: hành vi trái pháp luật, "
                                 "có lỗi, do chủ thể có năng lực trách nhiệm pháp lý thực "
                                 "hiện, và xâm hại quan hệ xã hội được pháp luật bảo vệ."),
        False,
    ),
]


def main() -> None:
    bo_phan_tich = argparse.ArgumentParser(description=__doc__)
    bo_phan_tich.add_argument("--so-lan", type=int, default=1,
                              help="chạy mỗi ca nhiều lần để đo độ ổn định")
    doi_so = bo_phan_tich.parse_args()

    print(f"\nKIỂM ĐỊNH CƠ CHẾ PHÁT HIỆN MÂU THUẪN — model chấm: {config.JUDGE_MODEL}")
    print(f"Đồng thuận {config.SO_LAN_CHAM_MAU_THUAN} lần chấm · "
          f"ngưỡng mức độ {config.NGUONG_MAU_THUAN}\n")
    print(f"{'Ca kiểm định':<50}{'Kỳ vọng':>10}{'Thực tế':>10}{'Tầng 1':>9}{'':>6}")
    print("-" * 92)

    so_dung = 0
    so_ca_im_lang = sum(1 for *_, that in CAC_CA_KIEM_DINH if not that)
    so_im_lang_dung = 0
    khong_on_dinh = []

    for ten, a, b, la_mau_thuan_that in CAC_CA_KIEM_DINH:
        # Tầng 1 (tất định) đo riêng: một ca mâu thuẫn thật mà tầng 1 đã loại thì tầng LLM
        # không bao giờ được nhìn thấy nó. Đây là chỗ hay hỏng âm thầm nhất của thiết kế hai
        # tầng, nên phải nhìn thấy được trong bảng chứ không trộn vào kết quả cuối.
        qua_tang_1 = bool(cac_cap_dang_ngo([a, b], None))

        cac_lan = []
        for _ in range(max(1, doi_so.so_lan)):
            cac_lan.append(bool(tim_mau_thuan([a, b], embedding_service=None)))
        thuc_te = cac_lan[0]
        if len(set(cac_lan)) > 1:
            khong_on_dinh.append(ten)

        dat = thuc_te == la_mau_thuan_that
        so_dung += dat
        if not la_mau_thuan_that and dat:
            so_im_lang_dung += 1

        print(
            f"{ten:<50}"
            f"{('có' if la_mau_thuan_that else 'im lặng'):>10}"
            f"{('có' if thuc_te else 'im lặng'):>10}"
            f"{('qua' if qua_tang_1 else 'chặn'):>9}"
            f"{('  ✔' if dat else '  ✘'):>6}"
        )

    print("-" * 92)
    tong = len(CAC_CA_KIEM_DINH)
    ty_le = so_dung / tong
    print(f"Đúng: {so_dung}/{tong} ca ({ty_le:.0%})")
    print(f"Trong đó im lặng đúng: {so_im_lang_dung}/{so_ca_im_lang} ca "
          "(đây mới là phần khó — báo động giả tốn niềm tin của người dùng)")
    if doi_so.so_lan > 1:
        if khong_on_dinh:
            print(f"✘ Không ổn định giữa các lần chạy ở {len(khong_on_dinh)} ca: "
                  f"{', '.join(khong_on_dinh)}")
        else:
            print(f"✔ Ổn định qua {doi_so.so_lan} lần chạy — cơ chế đồng thuận "
                  f"{config.SO_LAN_CHAM_MAU_THUAN} lần đang làm đúng việc của nó.")

    print()
    if so_im_lang_dung < so_ca_im_lang:
        print(
            "✘ CẢNH BÁO: cơ chế đang BÁO ĐỘNG GIẢ. Với tính năng này, báo nhầm tệ hơn bỏ "
            "sót — người dùng mất niềm tin vào chính tài liệu của họ mà không có cách nào "
            "kiểm rẻ. Hãy nâng NGUONG_MAU_THUAN hoặc SO_LAN_CHAM_MAU_THUAN trước khi dùng."
        )
    elif ty_le < 0.7:
        print(
            f"✘ Chỉ đúng {ty_le:.0%} số ca. Cơ chế không báo động giả (tốt) nhưng bỏ sót "
            "nhiều — cân nhắc JUDGE_MODEL lớn hơn, hoặc hạ NGUONG_MAU_THUAN và chạy lại "
            "chính script này để xem có sinh báo động giả không."
        )
    else:
        print(
            f"✔ Dùng được: {ty_le:.0%} số ca đúng và không có báo động giả nào. Đây là con "
            "số nên đặt cạnh tính năng trong báo cáo."
        )


if __name__ == "__main__":
    main()
