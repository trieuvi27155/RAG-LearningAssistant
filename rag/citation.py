"""Định dạng đoạn trích nguồn thành trích dẫn dễ hiển thị (tên file + trang/slide + đoạn trích),
lấy trực tiếp từ metadata đã gắn từ Document Loader."""

import re
from typing import Dict, List, Optional

import config
from rag.document_loader import MOC_BANG_DONG, MOC_BANG_MO
from rag.image_extractor import MOC_ANH

DO_DAI_TRICH_DAN = 600

_MAU_THAM_CHIEU_KHOI = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
_MAU_SO_TRONG_KHOI = re.compile(r"\d+")


def _cac_so_tham_chieu(van_ban: str) -> List[int]:
    """Mọi số đoạn trích được dẫn trong đoạn văn bản, kể cả dạng gộp "[3,4,5]"."""
    cac_so = []
    for khoi in _MAU_THAM_CHIEU_KHOI.findall(van_ban or ""):
        cac_so.extend(int(s) for s in _MAU_SO_TRONG_KHOI.findall(khoi))
    return cac_so


_MAU_BO_SO_TRICH_DAN = re.compile(r"[ \t]*\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
_MAU_NGOAC_DANG_DO = re.compile(r"[ \t]*\[[\d,;\s]*$")


def bo_so_trich_dan(van_ban: str) -> str:
    """Bỏ các số [n] khỏi văn bản để HIỂN THỊ, giữ nguyên dữ liệu gốc."""
    s = _MAU_BO_SO_TRICH_DAN.sub("", van_ban or "")
    s = _MAU_NGOAC_DANG_DO.sub("", s)
    s = re.sub(r"[ \t]+([.,;:!?)])", r"\1", s)
    return re.sub(r"[ \t]{2,}", " ", s)
_MAU_SO_TRANG = re.compile(r"(?:trang|slide|page)\s*/?\s*(?:slide\s*)?(\d+)", re.IGNORECASE)
_MAU_TACH_CAU = re.compile(r"(?<=[.!?:])\s+|\n+\s*[-*•]?\s*")

_MAU_CAU_LOAI_TRU_NGUON = re.compile(
    r"không\s+(?:liên\s+quan|đề\s+cập|nhắc\s+(?:tới|đến)|nói\s+(?:tới|đến|về)"
    r"|chứa|có\s+thông\s+tin|cung\s+cấp\s+thông\s+tin)"
    r"|(?:is|are)\s+(?:not\s+relevant|unrelated)"
    r"|do(?:es)?\s+not\s+(?:mention|contain|discuss|cover|relate)"
    r"|no\s+information\s+(?:about|on)",
    re.IGNORECASE,
)


def dinh_dang_trich_dan(cac_chunk: List[Dict]) -> List[Dict]:
    """Chuẩn hoá danh sách đoạn trích nguồn thành list trích dẫn để hiển thị:
    {"nguon", "trang", "doan_trich", "diem_similarity"}."""
    trich_dan = []
    for chunk in cac_chunk:
        doan_trich = chunk.get("doan_khop") or chunk["noidung"]
        for moc in (MOC_BANG_MO, MOC_BANG_DONG, MOC_ANH):
            doan_trich = doan_trich.replace(moc, "")
        doan_trich = doan_trich.strip()
        if len(doan_trich) > DO_DAI_TRICH_DAN:
            doan_trich = doan_trich[:DO_DAI_TRICH_DAN].rstrip() + "..."
        trich_dan.append(
            {
                "nguon": chunk["nguon"],
                "trang": chunk["trang"],
                "cac_trang": chunk.get("cac_trang") or [chunk["trang"]],
                "doan_trich": doan_trich,
                "diem_similarity": chunk.get("diem_similarity"),
                "loai_noi_dung": chunk.get("loai_noi_dung", "van_ban"),
                "duong_dan_anh": chunk.get("duong_dan_anh", ""),
            }
        )
    return trich_dan


def loc_theo_tham_chieu(
    cac_chunk: List[Dict], cau_tra_loi: str, so_toi_da: Optional[int] = None
) -> List[Dict]:
    """Chỉ giữ những nguồn mà câu trả lời THẬT SỰ tham chiếu tới ([1], [2]... trong nội dung)."""
    if not cac_chunk:
        return []
    cau_tra_loi = (cau_tra_loi or "").strip()
    if any(cau_tra_loi.startswith(tu_choi) for tu_choi in config.CAU_TU_CHOI.values()):
        return []
    so_toi_da = so_toi_da or config.SO_TRICH_DAN_HIEN_THI

    trich_dan = dinh_dang_trich_dan(cac_chunk)
    cac_so_tham_chieu = set(_cac_so_tham_chieu(cau_tra_loi))

    duoc_dung = [
        {**t, "so_hieu": i}
        for i, t in enumerate(trich_dan, start=1)
        if i in cac_so_tham_chieu
    ]
    la_suy_doan = False
    if not duoc_dung:
        cac_trang_nhac_toi = {int(s) for s in _MAU_SO_TRANG.findall(cau_tra_loi)}
        duoc_dung = [
            {**t, "so_hieu": i}
            for i, t in enumerate(trich_dan, start=1)
            if t["trang"] in cac_trang_nhac_toi
        ]
    if not duoc_dung:
        duoc_dung = [{**t, "so_hieu": i} for i, t in enumerate(trich_dan[:1], start=1)]
        la_suy_doan = True

    ket_qua, theo_khoa = [], {}
    for t in duoc_dung:
        khoa = (t["nguon"], t["trang"])
        if khoa in theo_khoa:
            theo_khoa[khoa]["cac_so"].append(t["so_hieu"])
            theo_khoa[khoa]["cac_trang"] = sorted(
                set(theo_khoa[khoa]["cac_trang"]) | set(t["cac_trang"]), key=str
            )
            continue
        if len(ket_qua) >= so_toi_da:
            continue
        muc = {**t, "la_suy_doan": la_suy_doan,
               "cac_so": [] if la_suy_doan else [t["so_hieu"]]}
        theo_khoa[khoa] = muc
        ket_qua.append(muc)

    for muc in ket_qua:
        muc["cac_so"] = sorted(set(muc["cac_so"]))

    return ket_qua


_MAU_TU_BAM = re.compile(r"[0-9A-Za-zÀ-ỹ]+")


def do_bam_ngu_canh(cau_tra_loi: str, ngu_canh: str, so_tu_moi_cum: int = 4) -> float:
    """Tỉ lệ cụm 4 từ liên tiếp của câu trả lời xuất hiện NGUYÊN VĂN trong ngữ cảnh."""
    cac_tu = _MAU_TU_BAM.findall(cau_tra_loi.lower())
    if len(cac_tu) < so_tu_moi_cum:
        return 0.0
    ngu_canh_rut_gon = "".join(_MAU_TU_BAM.findall(ngu_canh.lower()))
    cac_cum = [
        "".join(cac_tu[i : i + so_tu_moi_cum])
        for i in range(len(cac_tu) - so_tu_moi_cum + 1)
    ]
    return sum(1 for cum in cac_cum if cum in ngu_canh_rut_gon) / len(cac_cum)


def cau_theo_trich_dan(cau_tra_loi: str) -> Dict[int, List[str]]:
    """Ghép mỗi số trích dẫn [n] với những CÂU trong câu trả lời đã dẫn nó."""
    ket_qua: Dict[int, List[str]] = {}
    for cau in _MAU_TACH_CAU.split(cau_tra_loi or ""):
        cau = cau.strip()
        if not cau or _MAU_CAU_LOAI_TRU_NGUON.search(cau):
            continue
        for so in set(_cac_so_tham_chieu(cau)):
            ket_qua.setdefault(so, []).append(cau)
    return ket_qua


def format_text_trich_dan(cac_chunk: List[Dict]) -> str:
    """Trả về chuỗi text hiển thị nhanh trích dẫn (dùng cho script/terminal, không cần UI)."""
    return "\n".join(
        f"[{i}] {t['nguon']} - trang/slide {t['trang']}: \"{t['doan_trich']}\""
        for i, t in enumerate(dinh_dang_trich_dan(cac_chunk), start=1)
    )
