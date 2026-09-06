"""Chia nội dung từng trang/slide thành các chunk nhỏ hơn (Recursive Character Splitting)."""

import logging
import re
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from rag.document_loader import MOC_BANG_DONG, MOC_BANG_MO

logger = logging.getLogger(__name__)

_encoding = tiktoken.get_encoding(config.TIKTOKEN_ENCODING)

DO_DAI_CHUNK_TOI_THIEU = 25

_MAU_KHOI_BANG = re.compile(
    re.escape(MOC_BANG_MO) + r".*?" + re.escape(MOC_BANG_DONG), re.DOTALL
)


def _tach_khoi_bang(text: str) -> List[Tuple[str, bool]]:
    """Tách nội dung trang thành các đoạn (nội_dung, là_bảng), giữ nguyên thứ tự."""
    cac_phan: List[Tuple[str, bool]] = []
    vi_tri = 0
    for khop in _MAU_KHOI_BANG.finditer(text):
        if khop.start() > vi_tri:
            cac_phan.append((text[vi_tri:khop.start()], False))
        cac_phan.append((khop.group(), True))
        vi_tri = khop.end()
    if vi_tri < len(text):
        cac_phan.append((text[vi_tri:], False))
    return cac_phan or [(text, False)]


def _hang_thanh_van_xuoi(hang: str) -> str:
    """Đổi 1 hàng Markdown trở lại văn xuôi, bỏ dấu | và các ô rỗng."""
    cac_o = [o.strip() for o in hang.strip().strip("|").split("|")]
    return " ".join(o for o in cac_o if o)


def _cat_bang_giu_tieu_de(
    khoi: str, dem: Callable[[str], int], tran: int
) -> List[Tuple[str, bool]]:
    """Cắt một bảng quá lớn thành nhiều mảnh, MỖI MẢNH ĐƯỢC LẶP LẠI DÒNG TIÊU ĐỀ CỘT."""
    ben_trong = khoi.replace(MOC_BANG_MO, "").replace(MOC_BANG_DONG, "").strip()
    cac_dong = [d for d in ben_trong.splitlines() if d.strip()]
    if len(cac_dong) < 3:
        return [(khoi, True)]

    tieu_de, gach, cac_hang = cac_dong[0], cac_dong[1], cac_dong[2:]

    def dong_goi(hang: List[str]) -> str:
        return "\n".join([MOC_BANG_MO, tieu_de, gach, *hang, MOC_BANG_DONG])

    if dem(dong_goi([])) >= tran:
        logger.info(
            "Dòng tiêu đề của bảng đã chiếm hết ngân sách %d token - không lặp lại được, "
            "cắt bảng như văn xuôi.", tran,
        )
        return [(khoi, True)]

    ket_qua: List[Tuple[str, bool]] = []
    dem_hien_tai: List[str] = []
    for hang in cac_hang:
        if dem(dong_goi([hang])) > tran:
            if dem_hien_tai:
                ket_qua.append((dong_goi(dem_hien_tai), True))
                dem_hien_tai = []
            ket_qua.append((_hang_thanh_van_xuoi(hang), False))
            continue
        if dem_hien_tai and dem(dong_goi(dem_hien_tai + [hang])) > tran:
            ket_qua.append((dong_goi(dem_hien_tai), True))
            dem_hien_tai = []
        dem_hien_tai.append(hang)
    if dem_hien_tai:
        ket_qua.append((dong_goi(dem_hien_tai), True))
    return ket_qua


def dem_token(text: str) -> int:
    """Đếm số token xấp xỉ bằng tiktoken - CHỈ dùng khi không có tokenizer của model thật."""
    return len(_encoding.encode(text))


def kich_thuoc_chunk_an_toan(max_seq_length: Optional[int] = None) -> int:
    """Kích thước chunk thực tế sẽ dùng, đã hạ xuống cho vừa giới hạn của embedding model."""
    if not max_seq_length:
        return config.CHUNK_SIZE_TOKENS
    tran = max(max_seq_length - config.BIEN_AN_TOAN_TOKEN, 32)
    if config.CHUNK_SIZE_TOKENS > tran:
        logger.warning(
            "CHUNK_SIZE_TOKENS=%d vượt giới hạn %d token của embedding model - tự hạ xuống %d "
            "để nội dung không bị cắt mất khi encode.",
            config.CHUNK_SIZE_TOKENS,
            max_seq_length,
            tran,
        )
        return tran
    return config.CHUNK_SIZE_TOKENS


def tao_splitter(
    dem_token_fn: Optional[Callable[[str], int]] = None,
    max_seq_length: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """dem_token_fn: hàm đếm token của chính embedding model (EmbeddingService.dem_token).
    Bỏ trống thì lùi về tiktoken - xem cảnh báo ở docstring của dem_token()."""
    chunk_size = kich_thuoc_chunk_an_toan(max_seq_length)
    chunk_overlap = min(config.CHUNK_OVERLAP_TOKENS, chunk_size // 3)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=dem_token_fn or dem_token,
        separators=config.CHUNK_SEPARATORS,
    )


def chia_chunk(
    cac_trang: List[Dict],
    dem_token_fn: Optional[Callable[[str], int]] = None,
    max_seq_length: Optional[int] = None,
) -> List[Dict]:
    """Chia danh sách trang (từ document_loader) thành danh sách chunk."""
    splitter = tao_splitter(dem_token_fn=dem_token_fn, max_seq_length=max_seq_length)
    dem = dem_token_fn or dem_token
    tran_token = kich_thuoc_chunk_an_toan(max_seq_length)
    tran_token_bang = (
        max(tran_token, max_seq_length - config.BIEN_AN_TOAN_TOKEN)
        if max_seq_length
        else tran_token
    )
    cac_chunk = []
    so_chunk_bo_qua = 0
    for trang in cac_trang:
        if trang.get("loai_noi_dung") == "anh":
            cac_chunk.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "nguon": trang["nguon"],
                    "trang": trang["trang"],
                    "vi_tri": 0,
                    "noidung": trang["noidung"],
                    "loai_noi_dung": "anh",
                    "duong_dan_anh": trang.get("duong_dan_anh", ""),
                    "co_chu_thich_vision": trang.get("co_chu_thich_vision", False),
                }
            )
            continue

        vi_tri = 0
        for phan, la_bang in _tach_khoi_bang(trang["noidung"]):
            if la_bang and dem(phan) <= tran_token_bang:
                cac_doan_nho = [(phan, True)]
            elif la_bang:
                logger.info(
                    "Bảng ở '%s' trang %s dài %d token, vượt giới hạn %d của model - cắt "
                    "theo hàng, lặp lại dòng tiêu đề ở từng mảnh.",
                    trang["nguon"], trang["trang"], dem(phan), tran_token_bang,
                )
                cac_doan_nho = []
                for manh, van_la_bang in _cat_bang_giu_tieu_de(phan, dem, tran_token_bang):
                    if not van_la_bang:
                        cac_doan_nho += [(d, False) for d in splitter.split_text(manh)]
                    elif dem(manh) <= tran_token_bang:
                        cac_doan_nho.append((manh, True))
                    else:
                        logger.warning(
                            "Mảnh bảng ở '%s' trang %s vẫn dài %d token sau khi cắt theo "
                            "hàng - buộc phải cắt tiếp (mất quan hệ hàng-cột ở các phần sau).",
                            trang["nguon"], trang["trang"], dem(manh),
                        )
                        cac_doan_nho += [(d, True) for d in splitter.split_text(manh)]
            else:
                cac_doan_nho = [(d, False) for d in splitter.split_text(phan)]

            for doan, doan_la_bang in cac_doan_nho:
                doan = doan.strip()
                if len(doan) < DO_DAI_CHUNK_TOI_THIEU:
                    so_chunk_bo_qua += 1
                    vi_tri += 1
                    continue
                cac_chunk.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "nguon": trang["nguon"],
                        "trang": trang["trang"],
                        "vi_tri": vi_tri,
                        "noidung": doan,
                        "loai_noi_dung": "bang" if doan_la_bang else "van_ban",
                    }
                )
                vi_tri += 1
    if so_chunk_bo_qua:
        logger.info("Đã bỏ qua %d chunk quá ngắn (dưới %d ký tự).", so_chunk_bo_qua, DO_DAI_CHUNK_TOI_THIEU)
    return cac_chunk
