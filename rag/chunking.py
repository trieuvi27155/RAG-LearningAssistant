"""Chia nội dung từng trang/slide thành các chunk nhỏ hơn (Recursive Character Splitting).

Mỗi trang có thể dài hơn nhiều so với giới hạn đầu vào của embedding model, nên cần
chia nhỏ. Mỗi chunk giữ nguyên "nguon"/"trang" của trang gốc (để citation.py trích dẫn
đúng), cộng thêm "chunk_id" duy nhất để dùng làm khóa tra cứu nếu cần sau này.

Kích thước chunk được đo bằng ĐÚNG tokenizer của embedding model (truyền vào qua
dem_token_fn) chứ không phải bộ đếm xấp xỉ - xem giải thích ở config.CHUNK_SIZE_TOKENS.
tiktoken chỉ còn là phương án dự phòng khi không lấy được tokenizer thật.
"""

import logging
import re
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from rag.document_loader import MOC_BANG_DONG, MOC_BANG_MO

logger = logging.getLogger(__name__)

# Khởi tạo 1 lần ở module-level vì load bộ mã hóa của tiktoken có chi phí nhất định,
# không cần tạo lại mỗi lần gọi dem_token().
_encoding = tiktoken.get_encoding(config.TIKTOKEN_ENCODING)

# Chunk ngắn hơn ngưỡng này gần như chắc chắn là rác do tách thô để lại (một số thứ tự
# "1.", một dấu ngoặc lạc dòng, mẩu tiêu đề cụt...): không mang đủ ngữ nghĩa để khớp đúng
# câu hỏi nào, nhưng vẫn chiếm 1 vector trong index và vẫn có thể lọt top với câu hỏi ngắn
# -> chỉ làm nhiễu. Bỏ qua ngay từ bước chunking.
DO_DAI_CHUNK_TOI_THIEU = 25

_MAU_KHOI_BANG = re.compile(
    re.escape(MOC_BANG_MO) + r".*?" + re.escape(MOC_BANG_DONG), re.DOTALL
)


def _tach_khoi_bang(text: str) -> List[Tuple[str, bool]]:
    """Tách nội dung trang thành các đoạn (nội_dung, là_bảng), giữ nguyên thứ tự.

    Cần tách vì bảng và văn xuôi phải được chia chunk theo 2 cách khác hẳn nhau: văn xuôi
    cắt theo câu/đoạn thì vẫn đọc được, còn bảng cắt ngang giữa các hàng thì mất luôn dòng
    tiêu đề cột - các ô còn lại thành những giá trị trôi nổi không biết thuộc cột nào.
    """
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
    """Đổi 1 hàng Markdown trở lại văn xuôi, bỏ dấu | và các ô rỗng.

    Dùng cho ô chứa NGUYÊN MỘT BÀI VĂN (biểu mẫu Word hay có ô "Giới thiệu ý tưởng nghiên
    cứu" dài vài nghìn chữ). Một hàng như thế không mang quan hệ hàng-cột nào để mà giữ -
    ép nó ở dạng bảng chỉ khiến mỗi mảnh cắt ra mang theo mấy dấu "|" lạc lõng ở đầu, và
    bị gắn nhãn "bảng" trong khi thực chất là văn xuôi.
    """
    cac_o = [o.strip() for o in hang.strip().strip("|").split("|")]
    return " ".join(o for o in cac_o if o)


def _cat_bang_giu_tieu_de(
    khoi: str, dem: Callable[[str], int], tran: int
) -> List[Tuple[str, bool]]:
    """Cắt một bảng quá lớn thành nhiều mảnh, MỖI MẢNH ĐƯỢC LẶP LẠI DÒNG TIÊU ĐỀ CỘT.

    Đây là chỗ sửa một mất mát thật đã đo được, không phải phòng xa. Bản trước, bảng vượt
    giới hạn model bị đẩy thẳng vào splitter văn xuôi: mảnh đầu còn dòng tiêu đề, mọi mảnh
    sau chỉ còn các ô trần. Trên biểu mẫu DeCuongNCKH.docx (cả tờ khai là MỘT bảng 6755
    token), hậu quả là ô "LĨNH VỰC: Khoa học máy tính" nằm lẻ loi ở mảnh đầu, còn 60 chunk
    sau toàn văn xuôi mang dấu "|" - hỏi "đề tài thuộc lĩnh vực nào?" thì hệ thống trả lời
    theo phần văn xuôi ("thương mại điện tử") thay vì theo đúng ô LĨNH VỰC.

    Cách làm: gom từng hàng vào mảnh hiện tại chừng nào còn vừa `tran`, mảnh nào cũng mở
    đầu bằng dòng tiêu đề + dòng gạch ngăn nên vẫn là một bảng Markdown hợp lệ, đọc được
    độc lập. Hàng nào một mình đã vượt trần thì đó là ô văn xuôi dài - trả về dạng văn xuôi
    (là_bảng=False) để chỗ gọi cắt bằng splitter thường.

    Trả về [(nội_dung, là_bảng)] theo đúng thứ tự gốc.
    """
    ben_trong = khoi.replace(MOC_BANG_MO, "").replace(MOC_BANG_DONG, "").strip()
    cac_dong = [d for d in ben_trong.splitlines() if d.strip()]
    if len(cac_dong) < 3:
        # Không đủ (tiêu đề + gạch ngăn + ít nhất 1 hàng) để mà lặp tiêu đề.
        return [(khoi, True)]

    tieu_de, gach, cac_hang = cac_dong[0], cac_dong[1], cac_dong[2:]

    def dong_goi(hang: List[str]) -> str:
        return "\n".join([MOC_BANG_MO, tieu_de, gach, *hang, MOC_BANG_DONG])

    # Dòng tiêu đề tự nó đã chiếm gần hết ngân sách -> lặp lại nó ở từng mảnh sẽ đẩy MỌI
    # hàng dữ liệu ra ngoài, biến cả bảng thành một chuỗi văn xuôi rời rạc - tệ hơn hẳn so
    # với việc chấp nhận mất tiêu đề. Xảy ra thật với bảng nhiều cột mà tên cột dài (biểu
    # mẫu hành chính, bảng so sánh nhiều tiêu chí). Trả nguyên khối để chỗ gọi cắt như cũ.
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
            # Ô văn xuôi dài: đẩy nốt phần bảng đang gom rồi trả hàng này về dạng văn xuôi.
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
    """Đếm số token xấp xỉ bằng tiktoken - CHỈ dùng khi không có tokenizer của model thật.

    Lưu ý: với tiếng Việt, tiktoken (cl100k_base) đếm ra số token gấp khoảng 1.9 lần
    tokenizer thật của các model đa ngôn ngữ nền XLM-R (đã đo trên corpus của đồ án), nên
    dùng nó làm thước đo sẽ tạo ra chunk nhỏ hơn dự định rất nhiều.
    """
    return len(_encoding.encode(text))


def kich_thuoc_chunk_an_toan(max_seq_length: Optional[int] = None) -> int:
    """Kích thước chunk thực tế sẽ dùng, đã hạ xuống cho vừa giới hạn của embedding model.

    Nội dung vượt max_seq_length bị model cắt bỏ ÂM THẦM lúc encode (không có lỗi báo ra),
    nghĩa là phần cuối chunk sẽ không hề được đưa vào vector - tra cứu sẽ không bao giờ tìm
    thấy nó. Vì vậy đây là chặn cứng, không phải gợi ý: đổi sang model có giới hạn nhỏ hơn
    thì chunk tự nhỏ theo, không cần ai nhớ sửa config.
    """
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
    Bỏ trống thì lùi về tiktoken - xem cảnh báo ở docstring của dem_token().
    """
    chunk_size = kich_thuoc_chunk_an_toan(max_seq_length)
    # Overlap phải nhỏ hơn hẳn chunk_size, nếu không splitter sẽ lặp gần như toàn bộ nội
    # dung giữa 2 chunk liên tiếp (và với overlap >= chunk_size thì rơi vào vòng lặp vô tận).
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
    """Chia danh sách trang (từ document_loader) thành danh sách chunk.

    cac_trang: list các dict {"nguon", "trang", "noidung"}.
    Trả về: list các dict {"chunk_id", "nguon", "trang", "vi_tri", "noidung"}.
    "vi_tri": thứ tự chunk trong trang gốc (0, 1, 2...) - dùng để rag_pipeline.py mở rộng
    ngữ cảnh sang các chunk liền kề CÙNG TRANG theo đúng thứ tự gốc, tránh xáo trộn nội dung.

    "vi_tri" được đánh số theo thứ tự splitter sinh ra, TRƯỚC khi lọc bỏ chunk quá ngắn -
    nên dãy số có thể khuyết (0, 1, 3, 4...). Đó là chủ ý: rag_pipeline sắp xếp theo vi_tri
    rồi lấy các phần tử liền kề TRONG DANH SÁCH (không dựa vào vi_tri liên tục), nên khuyết
    số không ảnh hưởng, mà lại giữ được đúng thông tin "chunk này đứng trước chunk kia".
    """
    splitter = tao_splitter(dem_token_fn=dem_token_fn, max_seq_length=max_seq_length)
    dem = dem_token_fn or dem_token
    # BẢNG được phép dài hơn văn xuôi, tới sát giới hạn thật của model.
    #
    # CHUNK_SIZE_TOKENS (160) là lựa chọn về ĐỘ CHÍNH XÁC TRUY XUẤT cho văn xuôi: chunk nhỏ
    # thì mỗi vector mô tả một ý gọn, khớp câu hỏi sắc hơn. Nhưng với bảng, cắt nhỏ không
    # đổi lại được gì cả - nó phá đúng thứ khiến bảng là bảng (dòng tiêu đề cột), khiến các
    # ô còn lại thành giá trị trôi nổi không biết thuộc cột nào. Ràng buộc CỨNG duy nhất với
    # bảng là giới hạn của model, vượt qua thì nội dung bị cắt âm thầm lúc encode.
    #
    # Đo trên bộ tài liệu thật: 15 bảng bị cắt vì vượt 160 token, trong đó phần lớn dài
    # 164-476 token - tức vẫn nằm gọn trong giới hạn 496 của model. Chúng bị cắt oan hoàn
    # toàn. Suy trần bảng TỪ CHÍNH MODEL (không thêm tham số cấu hình mới) nên đổi model là
    # trần tự điều chỉnh theo.
    tran_token = kich_thuoc_chunk_an_toan(max_seq_length)
    tran_token_bang = (
        max(tran_token, max_seq_length - config.BIEN_AN_TOAN_TOKEN)
        if max_seq_length
        else tran_token
    )
    cac_chunk = []
    so_chunk_bo_qua = 0
    for trang in cac_trang:
        # Bản ghi ẢNH đi thẳng thành 1 chunk, không qua splitter: nội dung của nó là chú
        # thích ngắn, cắt nhỏ chỉ làm mất liên kết với đường dẫn ảnh đi kèm.
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
                    # Giữ lại cờ này để biết ảnh đã được model vision đọc nội dung hay mới
                    # chỉ có chú thích lân cận - cần cho việc đo đạc (bao nhiêu ảnh thực sự
                    # tìm được theo nội dung) và để UI phân biệt được 2 loại.
                    "co_chu_thich_vision": trang.get("co_chu_thich_vision", False),
                }
            )
            continue

        vi_tri = 0
        for phan, la_bang in _tach_khoi_bang(trang["noidung"]):
            if la_bang and dem(phan) <= tran_token_bang:
                # Bảng vừa giới hạn model -> giữ NGUYÊN KHỐI trong 1 chunk.
                cac_doan_nho = [(phan, True)]
            elif la_bang:
                # Bảng quá lớn: cắt theo HÀNG và lặp lại dòng tiêu đề ở từng mảnh, để mảnh
                # nào cũng còn biết mỗi ô thuộc cột nào (xem _cat_bang_giu_tieu_de).
                logger.info(
                    "Bảng ở '%s' trang %s dài %d token, vượt giới hạn %d của model - cắt "
                    "theo hàng, lặp lại dòng tiêu đề ở từng mảnh.",
                    trang["nguon"], trang["trang"], dem(phan), tran_token_bang,
                )
                cac_doan_nho = []
                for manh, van_la_bang in _cat_bang_giu_tieu_de(phan, dem, tran_token_bang):
                    if not van_la_bang:
                        # Ô văn xuôi dài -> cắt như văn xuôi và gắn nhãn văn xuôi.
                        cac_doan_nho += [(d, False) for d in splitter.split_text(manh)]
                    elif dem(manh) <= tran_token_bang:
                        cac_doan_nho.append((manh, True))
                    else:
                        # Mảnh bảng VẪN vượt trần: bảng chỉ có 1-2 dòng, hoặc dòng tiêu đề
                        # đã chiếm hết ngân sách nên không lặp lại được. Phải cắt tiếp, nếu
                        # không chunk này vượt giới hạn model và bị CẮT ÂM THẦM lúc encode -
                        # đúng loại lỗi mà kich_thuoc_chunk_an_toan() sinh ra để chặn.
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
                        # Cho phép UI/citation phân biệt bảng với văn xuôi khi hiển thị.
                        "loai_noi_dung": "bang" if doan_la_bang else "van_ban",
                    }
                )
                vi_tri += 1
    if so_chunk_bo_qua:
        logger.info("Đã bỏ qua %d chunk quá ngắn (dưới %d ký tự).", so_chunk_bo_qua, DO_DAI_CHUNK_TOI_THIEU)
    return cac_chunk
