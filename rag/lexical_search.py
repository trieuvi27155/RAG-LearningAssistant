"""Tìm kiếm theo TỪ KHOÁ (BM25) - nhánh thứ hai của tìm kiếm lai, bổ khuyết cho FAISS."""

import math
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple

K1 = 1.5
B = 0.75

_MAU_TU = re.compile(r"\w+", re.UNICODE)


def _tach_tu(text: str) -> List[str]:
    """Tách 1 đoạn text thành danh sách "từ" để lập chỉ mục BM25."""
    am_tiet = _MAU_TU.findall(unicodedata.normalize("NFC", text).lower())
    bigram = [f"{a}_{b}" for a, b in zip(am_tiet, am_tiet[1:])]
    return am_tiet + bigram


class BM25:
    """Chỉ mục BM25 dựng trong bộ nhớ từ danh sách văn bản."""

    def __init__(self, cac_van_ban: List[str]):
        self.so_tai_lieu = len(cac_van_ban)
        self.do_dai: List[int] = []
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
        """Trả về [(chi_so_tai_lieu, diem_bm25)] sắp xếp giảm dần, tối đa top_n phần tử."""
        if not self.so_tai_lieu or self.do_dai_trung_binh <= 0:
            return []

        diem_tich_luy: Dict[int, float] = defaultdict(float)
        for tu in set(_tach_tu(cau_hoi)):
            danh_sach = self.chi_muc_nguoc.get(tu)
            if not danh_sach:
                continue
            idf = math.log(1 + (self.so_tai_lieu - len(danh_sach) + 0.5) / (len(danh_sach) + 0.5))
            for chi_so, so_lan in danh_sach:
                chuan_hoa_do_dai = 1 - B + B * self.do_dai[chi_so] / self.do_dai_trung_binh
                diem_tich_luy[chi_so] += idf * so_lan * (K1 + 1) / (so_lan + K1 * chuan_hoa_do_dai)

        return sorted(diem_tich_luy.items(), key=lambda x: x[1], reverse=True)[:top_n]
