"""Tìm kiếm theo TỪ KHOÁ (BM25) - nhánh thứ hai của tìm kiếm lai, bổ khuyết cho FAISS.

Vì sao cần thêm nhánh này bên cạnh tìm kiếm vector:
Embedding nén cả một đoạn văn về 1 vector 384 chiều, nên nó giữ được "đại ý" nhưng làm mờ
các chi tiết HIẾM và CỤ THỂ - tên riêng, thuật ngữ chuyên ngành, số hiệu điều luật, con số,
năm. Với tài liệu ngắn thì không sao (ít đoạn cùng chủ đề để nhầm), nhưng với giáo trình
vài trăm trang thì có hàng chục đoạn "cùng chủ đề, khác chi tiết" - lúc đó tìm kiếm thuần
vector rất hay trả về đoạn nói chung chung đúng chủ đề thay vì đúng đoạn chứa chi tiết được
hỏi. BM25 mạnh đúng ở chỗ đó: từ khoá càng hiếm trong corpus thì trọng số IDF càng cao, nên
"Điều 15" hay "quy phạm pháp luật" được ưu tiên đúng chỗ có mặt chúng.

Tự cài BM25 (~60 dòng) thay vì thêm dependency `rank_bm25`: công thức BM25 cố định và ngắn,
việc tự cài cho phép tự quyết cách tách từ tiếng Việt (xem _tach_tu) - vốn là phần quan
trọng hơn nhiều so với bản thân công thức.
"""

import math
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple

# Tham số chuẩn của BM25 (Robertson/Okapi): k1 điều tiết mức "bão hoà" khi 1 từ lặp nhiều
# lần trong cùng đoạn, b điều tiết mức phạt đoạn dài. Dùng giá trị mặc định kinh điển -
# không có lý do gì để tinh chỉnh riêng cho đồ án khi chưa có bộ dữ liệu đủ lớn để đo.
K1 = 1.5
B = 0.75

_MAU_TU = re.compile(r"\w+", re.UNICODE)


def _tach_tu(text: str) -> List[str]:
    """Tách 1 đoạn text thành danh sách "từ" để lập chỉ mục BM25.

    Tiếng Việt viết rời từng ÂM TIẾT ("nhà nước" là 2 âm tiết nhưng 1 từ), nên nếu chỉ lấy
    âm tiết đơn thì những âm tiết cực phổ biến ("nhà", "quy", "pháp") sẽ khớp tràn lan khắp
    tài liệu. Vì vậy lập chỉ mục CẢ âm tiết đơn LẪN cặp âm tiết liền nhau (bigram): cặp
    "pháp_luật", "quy_phạm" hiếm hơn hẳn từng âm tiết riêng lẻ nên IDF cao, giúp khớp đúng
    cụm thuật ngữ mà không cần thư viện tách từ tiếng Việt (underthesea/VnCoreNLP - nặng và
    ngoài phạm vi đồ án).

    Giữ nguyên dấu tiếng Việt (không bỏ dấu): bỏ dấu sẽ gộp nhầm các từ khác nghĩa hẳn nhau
    ("má"/"mà"/"mã" đều thành "ma"), gây nhiễu nhiều hơn lợi.
    """
    # NFC cho khớp với chuẩn hoá đã làm ở document_loader - nếu 2 bên khác dạng Unicode,
    # cùng một từ sẽ bị coi là 2 từ khác nhau và không bao giờ khớp.
    am_tiet = _MAU_TU.findall(unicodedata.normalize("NFC", text).lower())
    bigram = [f"{a}_{b}" for a, b in zip(am_tiet, am_tiet[1:])]
    return am_tiet + bigram


class BM25:
    """Chỉ mục BM25 dựng trong bộ nhớ từ danh sách văn bản.

    Không lưu xuống đĩa cùng FAISS index: dựng lại từ metadata mất vài chục mili giây với
    corpus cỡ đồ án, không đáng để thêm một định dạng file phải bảo trì và đồng bộ.
    """

    def __init__(self, cac_van_ban: List[str]):
        self.so_tai_lieu = len(cac_van_ban)
        self.do_dai: List[int] = []
        # term -> list[(chi_so_tai_lieu, so_lan_xuat_hien)]
        self.chi_muc_nguoc: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for i, van_ban in enumerate(cac_van_ban):
            cac_tu = _tach_tu(van_ban)
            self.do_dai.append(len(cac_tu))
            tan_suat: Dict[str, int] = defaultdict(int)
            for tu in cac_tu:
                tan_suat[tu] += 1
            for tu, so_lan in tan_suat.items():
                self.chi_muc_nguoc[tu].append((i, so_lan))

        self.do_dai_trung_binh = (sum(self.do_dai) / self.so_tai_lieu) if self.so_tai_lieu else 0.0

    def tim_kiem(self, cau_hoi: str, top_n: int) -> List[Tuple[int, float]]:
        """Trả về [(chi_so_tai_lieu, diem_bm25)] sắp xếp giảm dần, tối đa top_n phần tử.

        Chỉ duyệt những tài liệu có chứa ít nhất 1 từ của câu hỏi (nhờ chỉ mục ngược), không
        duyệt toàn bộ corpus - nên chi phí tỉ lệ với độ hiếm của từ khoá, không tỉ lệ với
        kích thước tài liệu.
        """
        if not self.so_tai_lieu or self.do_dai_trung_binh <= 0:
            return []

        diem_tich_luy: Dict[int, float] = defaultdict(float)
        for tu in set(_tach_tu(cau_hoi)):
            danh_sach = self.chi_muc_nguoc.get(tu)
            if not danh_sach:
                continue
            # IDF dạng "probabilistic" có cộng 1 bên trong log để không bao giờ ra số âm
            # với từ xuất hiện ở quá nửa số tài liệu (bản BM25 nguyên gốc có thể ra âm,
            # khiến một từ quá phổ biến lại TRỪ điểm của chính đoạn chứa nó).
            idf = math.log(1 + (self.so_tai_lieu - len(danh_sach) + 0.5) / (len(danh_sach) + 0.5))
            for chi_so, so_lan in danh_sach:
                chuan_hoa_do_dai = 1 - B + B * self.do_dai[chi_so] / self.do_dai_trung_binh
                diem_tich_luy[chi_so] += idf * so_lan * (K1 + 1) / (so_lan + K1 * chuan_hoa_do_dai)

        return sorted(diem_tich_luy.items(), key=lambda x: x[1], reverse=True)[:top_n]
