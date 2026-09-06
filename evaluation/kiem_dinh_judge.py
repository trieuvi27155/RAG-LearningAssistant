"""Đo độ tin cậy của chính THƯỚC ĐO Faithfulness (LLM-as-judge), không phải của hệ thống RAG."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluation.metrics import faithfulness

_NGU_CANH_SACH = (
    "The bias-variance decomposition breaks down the expected squared loss into three "
    "terms: the squared bias, the variance, and the intrinsic noise of the data."
)
_NGU_CANH_DINH_CHU = (
    "Thebias-variancedecompositionbreaksdowntheexpectedsquaredlossintothreeterms:"
    "thesquaredbias,thevariance,andtheintrinsicnoiseofthedata."
)

_NGU_CANH_LUAT = (
    "Nhà nước có ba đặc điểm cơ bản: tính giai cấp, quyền lực công cộng đặc biệt và chủ "
    "quyền quốc gia. Pháp luật ra đời cùng với nhà nước."
)

CAC_CA_KIEM_DINH = [
    (
        "Bám sát ngữ cảnh sạch",
        "The bias-variance decomposition splits the expected squared loss into the squared "
        "bias, the variance and the intrinsic noise [1].",
        _NGU_CANH_SACH,
        (0.8, 1.0),
    ),
    (
        "Bám sát nhưng NGỮ CẢNH BỊ DÍNH CHỮ (ca hỏng đã biết)",
        "The bias-variance decomposition splits the expected squared loss into the squared "
        "bias, the variance and the intrinsic noise [1].",
        _NGU_CANH_DINH_CHU,
        (0.8, 1.0),
    ),
    (
        "Diễn đạt lại bằng lời khác, không chép nguyên văn",
        "Sai số bình phương kỳ vọng được tách làm ba phần: độ chệch bình phương, phương sai "
        "và nhiễu vốn có của dữ liệu [1].",
        _NGU_CANH_SACH,
        (0.7, 1.0),
    ),
    (
        "Đúng một nửa, thêm một ý KHÔNG có trong ngữ cảnh",
        "Nhà nước có ba đặc điểm: tính giai cấp, quyền lực công cộng và chủ quyền quốc gia "
        "[1]. Nhà nước Việt Nam được thành lập năm 1945 [1].",
        _NGU_CANH_LUAT,
        (0.0, 0.7),
    ),
    (
        "Bịa hoàn toàn, không dính dáng ngữ cảnh",
        "Thủ đô của nước Pháp là Paris và thành phố này có rất nhiều bảo tàng nổi tiếng [1].",
        _NGU_CANH_LUAT,
        (0.0, 0.3),
    ),
    (
        "Nói NGƯỢC lại ngữ cảnh",
        "Pháp luật ra đời trước nhà nước và tồn tại độc lập với nhà nước [1].",
        _NGU_CANH_LUAT,
        (0.0, 0.3),
    ),
    (
        "Câu từ chối (quy ước luôn 1.0)",
        config.CAU_TU_CHOI["vi"],
        _NGU_CANH_LUAT,
        (0.9, 1.0),
    ),
]


def _chay_mot_ca(cau_tra_loi: str, ngu_canh: str) -> dict:
    """Gọi đúng hàm faithfulness() mà run_evaluation.py dùng, không viết lại logic."""
    return faithfulness(cau_tra_loi, [{"noidung": ngu_canh}])


def main() -> None:
    bo_phan_tich = argparse.ArgumentParser(description=__doc__)
    bo_phan_tich.add_argument(
        "--so-lan",
        type=int,
        default=1,
        help="Chấm lại mỗi ca bao nhiêu lần (>1 để đo độ ổn định giữa các lần chạy).",
    )
    doi_so = bo_phan_tich.parse_args()

    print(f"Kiểm định giám khảo: {config.JUDGE_MODEL} (temperature=0)")
    print(f"{len(CAC_CA_KIEM_DINH)} ca đã biết trước đáp án, mỗi ca chấm {doi_so.so_lan} lần.\n")
    print(f"{'Ca kiểm định':<50} {'Chấm':>7} {'Cần':>12} {'Kết quả':>9} {'Cờ ngờ':>8}")
    print("-" * 92)

    so_dung = 0
    tong_sai_lech = 0.0
    dao_dong_lon_nhat = 0.0
    ca_dinh_chu_duoc_cuu = None

    for ten, cau_tra_loi, ngu_canh, (thap, cao) in CAC_CA_KIEM_DINH:
        cac_diem, cac_co = [], []
        for _ in range(doi_so.so_lan):
            kq = _chay_mot_ca(cau_tra_loi, ngu_canh)
            cac_diem.append(kq["diem"])
            cac_co.append(kq["dang_ngo"])

        diem = sum(cac_diem) / len(cac_diem)
        dao_dong = max(cac_diem) - min(cac_diem)
        dao_dong_lon_nhat = max(dao_dong_lon_nhat, dao_dong)
        dat = thap <= diem <= cao
        so_dung += dat
        tong_sai_lech += 0.0 if dat else min(abs(diem - thap), abs(diem - cao))
        co_ngo = any(cac_co)

        if "DÍNH CHỮ" in ten:
            ca_dinh_chu_duoc_cuu = dat or co_ngo

        print(
            f"{ten:<50} {diem:>7.2f} {f'{thap:.1f}-{cao:.1f}':>12} "
            f"{('ĐẠT' if dat else 'LỆCH'):>9} {('⚠' if co_ngo else ''):>8}"
        )

    print("-" * 92)
    ty_le = so_dung / len(CAC_CA_KIEM_DINH)
    print(f"Chấm đúng khoảng: {so_dung}/{len(CAC_CA_KIEM_DINH)} ca ({ty_le:.0%})")
    print(f"Sai lệch trung bình khi lệch khoảng: {tong_sai_lech / len(CAC_CA_KIEM_DINH):.3f}")
    if doi_so.so_lan > 1:
        print(
            f"Dao động lớn nhất giữa các lần chấm cùng một ca: {dao_dong_lon_nhat:.2f} "
            "(0.00 nghĩa là temperature=0 thật sự cho kết quả lặp lại được)"
        )

    print()
    if ca_dinh_chu_duoc_cuu:
        print(
            "✔ Ca ngữ cảnh dính chữ đã được xử lý đúng: hoặc giám khảo chấm đúng, hoặc cờ "
            "tự nghi ngờ đã bật. Con số Faithfulness trong báo cáo dùng được."
        )
    else:
        print(
            "✘ CẢNH BÁO: ca ngữ cảnh dính chữ vẫn bị chấm sai MÀ KHÔNG có cờ nào bật. "
            "Đây đúng là kiểu lỗi âm thầm đã gặp ở §5.38 - đừng dùng Faithfulness làm căn "
            "cứ kết luận cho tới khi sửa được (hạ NGUONG_BAM_NGU_CANH_DE_NGHI_NGO, hoặc "
            "dùng judge model lớn hơn qua biến môi trường JUDGE_MODEL)."
        )
    if ty_le < 0.8:
        print(
            f"✘ Giám khảo chỉ đúng {ty_le:.0%} số ca — quá thấp để dùng con số Faithfulness "
            "làm căn cứ so sánh giữa các phiên bản. Cân nhắc JUDGE_MODEL lớn hơn."
        )


if __name__ == "__main__":
    main()
