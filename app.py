"""Giao diện Streamlit: quản lý tài liệu ở thanh bên, hỏi-đáp có trích dẫn ở khung chính.

BỐ CỤC theo lối các ứng dụng chat quen thuộc (ChatGPT/Claude), đã điều chỉnh cho đúng thứ
hệ thống này thật sự làm:
  - THANH BÊN = nguồn tài liệu (thay cho danh sách hội thoại). Đây là khác biệt cốt lõi giữa
    một trợ lý RAG và một chatbot thường: thứ người dùng cần quản lý không phải các cuộc trò
    chuyện cũ (phiên chat vốn không lưu qua nhiều phiên - xem §5.9) mà là TÀI LIỆU nào đang
    được dùng để trả lời. Đặt nó vào đúng chỗ mắt người tìm đến đầu tiên.
  - KHUNG CHÍNH = một cột hẹp căn giữa (~48rem), trang tự cuộn, ô nhập ghim đáy màn hình.
    Cột hẹp là có chủ đích chứ không phải để trống chỗ: câu trả lời có trích dẫn là văn bản
    để ĐỌC, mà dòng chữ dài quá ~80 ký tự thì mắt bị mỏi khi nhảy dòng.
  - Thu gọn thanh bên bằng nút "«" có sẵn của Streamlit là được toàn màn hình - thay cho nút
    phóng to tự làm ở bản trước (bản đó phải ẩn/giãn cột bằng CSS bám vào class nội bộ, dễ
    vỡ khi Streamlit đổi phiên bản).

Trích dẫn được gắn trực tiếp vào CUỐI mỗi câu trả lời (lưu trong chính
st.session_state.messages, không dùng 1 biến "trích dẫn hiện tại" dùng chung) - tránh lệch
pha giữa câu hỏi và trích dẫn hiển thị (bug đã gặp thực tế: hỏi câu mới nhưng trích dẫn cũ
chưa kịp cập nhật), đồng thời đảm bảo lật lại lịch sử chat vẫn thấy đúng trích dẫn của từng
câu trả lời tương ứng, không chỉ của câu gần nhất. Chỉ hiển thị những nguồn câu trả lời
THẬT SỰ tham chiếu tới (`citation.loc_theo_tham_chieu`, dựa trên số [n] LLM gắn trong câu
trả lời) - không còn mặc định lấy chunk điểm similarity cao nhất như bản trước, vì với tài
liệu dài đoạn điểm cao nhất thường không phải đoạn LLM thật sự dùng để trả lời.

Không có backend riêng - Streamlit gọi thẳng các hàm/class ở rag/*.py (đúng kiến trúc
đã chốt: không cần client-server, vì corpus và lượng truy cập của đồ án còn nhỏ).
"""

import logging
import time

import streamlit as st

import config
from rag import bo_nho_dem, do_thoi_gian
from rag.chunking import chia_chunk
from rag.citation import bo_so_trich_dan, loc_theo_tham_chieu
from rag.document_loader import cac_file_tai_lieu, doc_nhieu_file
from rag.embedding import EmbeddingService
from rag.rag_pipeline import (
    LoiKhongKetNoiDuocOllama,
    RagPipeline,
    kiem_tra_may_chu_llm,
    la_cau_hoi_kiem_chung,
)
from rag.reranker import tao_reranker_neu_bat
from rag.vector_store import VectorStore, so_sanh_bam_tai_lieu

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Hỏi đáp tài liệu học tập (RAG)",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# CSS tuỳ chỉnh. Chỉ động vào những gì theme của Streamlit không làm được: bề rộng cột đọc,
# hình dạng bong bóng chat, ô nhập ghim đáy. Mọi thứ còn lại (màu nền, màu nhấn, màu chữ) để
# theme ở .streamlit/config.toml lo, vì CSS thô bám vào class nội bộ của Streamlit rất dễ vỡ
# khi lên phiên bản mới.
st.markdown(
    """
    <style>
        /* Cột đọc hẹp và căn giữa - dòng chữ dài quá ~80 ký tự làm mắt mỏi khi nhảy dòng. */
        .block-container {
            max-width: 48rem;
            padding-top: 2.2rem;
            padding-bottom: 7rem;   /* chừa chỗ cho ô nhập ghim đáy, không để nó che tin cuối */
        }

        /* Bong bóng câu hỏi: nằm bên phải, nền xám nhạt, bo tròn - giống các app chat quen
           thuộc, để lướt lại lịch sử là phân biệt ngay lượt hỏi với lượt trả lời. Dùng đúng
           data-testid Streamlit render cho avatar user/assistant, không cần can thiệp Python. */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
            background: transparent;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] {
            background-color: #f0f1f3;
            border-radius: 18px;
            padding: 0.6rem 1rem;
            max-width: 85%;
        }
        /* Câu trả lời KHÔNG bọc bong bóng: nó là văn bản dài có tiêu đề, danh sách, bảng,
           trích dẫn - nhốt vào bong bóng chỉ làm hẹp chỗ đọc mà chẳng thêm thông tin gì. */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background: transparent;
        }

        /* Ô nhập: bo tròn + đổ bóng nhẹ để nổi lên khỏi nội dung đang cuộn phía sau. */
        [data-testid="stChatInput"] {
            border-radius: 26px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        }

        /* Thanh bên: viền phải mảnh thay cho đường kẻ đậm mặc định. */
        [data-testid="stSidebar"] {
            border-right: 1px solid #e5e5e5;
        }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

        /* Nút gợi ý câu hỏi ở màn hình trống: viền mảnh, chữ căn trái như một "thẻ" bấm được. */
        .st-key-goi_y button {
            text-align: left;
            justify-content: flex-start;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def lay_embedding_service() -> EmbeddingService:
    # cache_resource: chỉ load model embedding 1 lần cho cả phiên Streamlit (load model
    # tốn vài giây, không nên load lại mỗi lần Streamlit render lại trang).
    return EmbeddingService()


@st.cache_resource
def lay_reranker_service():
    # Cùng lý do như embedding: model reranker nặng ~2GB, nạp 1 lần cho cả phiên.
    # Trả None khi BAT_RERANK=0 - lúc đó model không được nạp chút nào.
    return tao_reranker_neu_bat()


@st.cache_data(ttl=10, show_spinner=False)
def lay_loi_may_chu_llm():
    """Trạng thái Ollama, để cảnh báo NGAY ở thanh bên thay vì đợi người dùng hỏi rồi mới lỗi.

    cache_data với ttl ngắn (không phải cache_resource): Streamlit chạy lại toàn bộ script
    sau MỌI thao tác, gọi thẳng sang Ollama mỗi lần thì thừa; nhưng cache vĩnh viễn lại giữ
    nguyên cảnh báo cũ sau khi người dùng đã bật Ollama lên - đúng lúc họ cần thấy nó biến
    mất để biết mình đã sửa xong. 10 giây là đủ ngắn để tự khỏi, đủ dài để không spam.
    """
    return kiem_tra_may_chu_llm()


def tai_index_da_co():
    """Load FAISS index đã build từ lần trước (nếu có) - tránh phải build lại mỗi lần mở app."""
    if config.FAISS_INDEX_FILE.exists() and config.METADATA_MAPPING_FILE.exists():
        return VectorStore.tai()
    return None


def lay_pipeline(embedding_service: EmbeddingService, store: VectorStore) -> RagPipeline:
    """Giữ lại 1 RagPipeline cho mỗi VectorStore đang dùng, thay vì tạo mới mỗi lần hỏi.

    Không dùng @st.cache_resource được vì VectorStore không hash được (và cũng không nên
    hash - nó thay đổi tại chỗ khi xoá tài liệu). So sánh bằng identity là đủ và đúng: chỉ
    khi store bị THAY THẾ (build lại index) mới cần pipeline mới.
    """
    if st.session_state.get("pipeline_cho_store") is not store:
        st.session_state.pipeline = RagPipeline(
            embedding_service, store, reranker_service=lay_reranker_service()
        )
        st.session_state.pipeline_cho_store = store
    return st.session_state.pipeline


def _store_dung_lai_duoc(store):
    """Index đang có trong phiên có dùng lại được cho một lần build TĂNG DẦN không?

    Ba điều kiện, và cả ba đều nhằm tránh đúng một kiểu hỏng: một index TRỘN hai thế hệ dữ
    liệu. Vector của tài liệu cũ và tài liệu mới phải cùng model, cùng chunk size, cùng các
    tuỳ chọn ăn vào nội dung - nếu không thì chúng nằm ở hai không gian khác nhau mà FAISS
    vẫn cứ so sánh với nhau, cho ra kết quả sai một cách hoàn toàn im lặng.

    ly_do_khong_tuong_thich() vốn đã trả lời đúng câu hỏi đó (nó được viết để cảnh báo người
    dùng khi đổi model mà quên build lại); ở đây chỉ dùng lại nó cho một quyết định tự động.
    """
    if not config.BAT_INDEX_TANG_DAN or store is None or store.so_luong_vector == 0:
        return False
    ly_do = store.ly_do_khong_tuong_thich()
    if ly_do:
        logger.info("Không build tăng dần được, sẽ dựng lại index từ đầu: %s", ly_do)
        return False
    return True


def xay_dung_lai_index(embedding_service: EmbeddingService, store_dang_dung=None):
    """Chạy luồng Ingestion: đọc tài liệu trong data/raw -> chunk -> embed -> lưu index.

    TĂNG DẦN THEO MẶC ĐỊNH (config.BAT_INDEX_TANG_DAN): chỉ những tài liệu MỚI hoặc ĐÃ ĐỔI
    NỘI DUNG mới đi qua toàn bộ luồng; tài liệu không đổi giữ nguyên vector đang có trong
    index, và tài liệu đã bị xoá khỏi thư mục thì vector của nó bị gỡ ra.

    Vì sao so bằng BĂM NỘI DUNG chứ không phải thời điểm sửa file: git checkout, sao chép
    file, đồng bộ cloud đều đổi mtime mà không đổi nội dung - dùng mtime là tự chuốc lấy
    những lần build lại vô nghĩa, đúng thứ đang muốn loại bỏ. Chiều ngược lại cũng có: vài
    công cụ ghi đè file mà giữ nguyên mtime, lúc đó mtime sẽ khiến hệ thống BỎ SÓT thay đổi
    thật - kiểu hỏng tệ hơn hẳn.

    Khi không tăng dần được (đổi model embedding, đổi chunk size, chưa có index), hàm tự lùi
    về dựng lại toàn bộ - vẫn nhanh hơn bản trước rất nhiều nhờ cache theo content hash ở
    rag/bo_nho_dem.py (kết quả đọc tài liệu, OCR, chú thích ảnh và embedding đều dùng lại
    được dù index phải dựng mới).
    """
    do_thoi_gian.dat_lai()
    cac_file = cac_file_tai_lieu(config.RAW_DOCS_DIR)
    if not cac_file:
        st.warning("Chưa có tài liệu nào có nội dung đọc được trong thư mục dữ liệu.")
        return None

    with do_thoi_gian.do("bam_tai_lieu"):
        bam_hien_tai = {d.name: bo_nho_dem.bam_file(d) for d in cac_file}

    if _store_dung_lai_duoc(store_dang_dung):
        store = store_dang_dung
        ten_can_doc, ten_can_xoa, giu_nguyen = so_sanh_bam_tai_lieu(
            store.bam_tai_lieu, bam_hien_tai
        )
        # Tài liệu đã biến mất khỏi thư mục: gỡ vector của nó ra. Không làm bước này thì
        # index vẫn trả lời bằng một tài liệu người dùng tưởng đã xoá.
        for ten in ten_can_xoa:
            so_xoa = store.xoa_theo_nguon(ten)
            logger.info("'%s' không còn trong thư mục - đã gỡ %d chunk khỏi index.", ten, so_xoa)
        # Tài liệu ĐÃ ĐỔI: xoá sạch vector cũ TRƯỚC khi thêm bản mới, nếu không index sẽ
        # chứa cả hai phiên bản và trích dẫn có thể trỏ vào nội dung không còn tồn tại.
        for ten in ten_can_doc:
            store.xoa_theo_nguon(ten)
        can_doc = [d for d in cac_file if d.name in set(ten_can_doc)]
        logger.info(
            "Build TĂNG DẦN: %d/%d tài liệu cần xử lý lại (%d tài liệu giữ nguyên vector cũ).",
            len(can_doc), len(cac_file), len(giu_nguyen),
        )
    else:
        store = VectorStore(dimension=embedding_service.dimension)
        can_doc = cac_file
        logger.info("Build TOÀN BỘ: %d tài liệu.", len(can_doc))

    if can_doc:
        cac_trang = doc_nhieu_file(can_doc)
        # Chia chunk bằng ĐÚNG tokenizer + giới hạn độ dài của model embedding đang dùng,
        # thay vì bộ đếm xấp xỉ dùng chung: kích thước chunk chỉ có ý nghĩa khi đo bằng
        # thước đo của chính model sẽ encode nó (xem config.CHUNK_SIZE_TOKENS).
        with do_thoi_gian.do("chunking"):
            cac_chunk = chia_chunk(
                cac_trang,
                dem_token_fn=embedding_service.lay_ham_dem_token(),
                max_seq_length=embedding_service.max_seq_length,
            )
        with do_thoi_gian.do("embedding"):
            vectors = bo_nho_dem.encode_co_cache(
                embedding_service, [c["noidung"] for c in cac_chunk]
            )
        if len(cac_chunk):
            store.them(vectors, cac_chunk)

        # Chỉ ghi nhận băm cho tài liệu THẬT SỰ có nội dung vào index. Một file đọc hỏng
        # (mật khẩu, tải dở) hay rỗng mà vẫn được ghi là "đã xử lý" sẽ bị bỏ qua ở mọi lần
        # build sau - tức lỗi im lặng vĩnh viễn. Để nó trượt và được thử lại mỗi lần là lựa
        # chọn đúng, và gần như miễn phí vì cache tài liệu bắt ngay ở lượt đọc kế tiếp.
        co_noi_dung = {m["nguon"] for m in cac_trang}
        for duong_dan in can_doc:
            if duong_dan.name in co_noi_dung:
                store.bam_tai_lieu[duong_dan.name] = bam_hien_tai[duong_dan.name]

    if store.so_luong_vector == 0:
        st.warning("Chưa có tài liệu nào có nội dung đọc được trong thư mục dữ liệu.")
        return None

    store.luu()
    if config.BAT_PROFILING_INGESTION:
        do_thoi_gian.ghi_bao_cao("PROFILING INGESTION")
    return store


def _hien_thi_trich_dan(trich_dan: list) -> None:
    """Hiển thị nguồn của 1 câu trả lời: tên file + trang/slide, gọn trên 1 dòng.

    Chỉ liệt kê những nguồn mà câu trả lời THẬT SỰ tham chiếu tới (xem
    citation.loc_theo_tham_chieu) - thường là 1, tối đa config.SO_TRICH_DAN_HIEN_THI. Bản
    trước luôn hiển thị đúng 1 nguồn là đoạn có điểm similarity cao nhất, nhưng với tài
    liệu dài thì đoạn điểm cao nhất thường không phải đoạn được dùng để trả lời, nên trích
    dẫn hay lệch với nội dung câu trả lời.

    CHỈ dẫn ra vị trí (tên file + trang/slide), không hiện lại nguyên văn đoạn đã dùng:
    người đọc tự mở tài liệu gốc mà đối chiếu. Đây là lựa chọn có chủ đích - đoạn trích in
    kèm là bản CẮT NGẮN (citation.DO_DAI_TRICH_DAN) và mất định dạng gốc, nên nó vừa chiếm
    chỗ dưới mỗi câu trả lời vừa dễ khiến người ta dừng lại ở đó thay vì mở tài liệu thật.
    Đoạn trích vẫn được tính và trả về trong dữ liệu (evaluation dùng nó để chấm Citation
    accuracy), chỉ là không vẽ ra màn hình.
    """
    if not trich_dan:
        return

    # Số hiệu đoạn trích ([1], [2]...) KHÔNG hiển thị ở đây, cũng như đã bị gỡ khỏi câu trả
    # lời (rag.citation.bo_so_trich_dan). Chúng vẫn được LLM sinh ra và vẫn được hệ thống đọc
    # - đó chính là căn cứ để biết câu trả lời dùng nguồn nào và để chấm Citation accuracy -
    # nhưng với người đọc thì chúng là thứ tự đoạn trích TRONG PROMPT, một thứ tự họ không
    # nhìn thấy nên cũng không tra ngược được. Hiện ra chỉ thêm nhiễu.
    def _nhan_trang(t: dict) -> str:
        """Ghi ĐỦ khoảng trang mà đoạn trích đã đọc, không chỉ trang neo.

        Đoạn trích được phép mở rộng sang trang liền kề để nối lại một định nghĩa bị ranh
        giới trang cắt đôi. Khi đó câu trả lời có thể dựa vào nội dung nằm ở trang bên cạnh,
        nên chỉ ghi trang neo là trỏ người đọc tới chỗ không chứa hết căn cứ.
        """
        cac_trang = t.get("cac_trang") or [t["trang"]]
        if len(cac_trang) <= 1:
            return f"trang/slide {t['trang']}"
        return f"trang/slide {cac_trang[0]}–{cac_trang[-1]}"

    danh_sach = " · ".join(f"**{t['nguon']}** — {_nhan_trang(t)}" for t in trich_dan)

    if trich_dan[0].get("la_suy_doan"):
        # Model không tự gắn số đoạn trích nào. Bản trước vẫn hiện đoạn điểm cao nhất dưới
        # nhãn "Nguồn", tức trình bày PHỎNG ĐOÁN CỦA HỆ THỐNG như thể là căn cứ mà câu trả
        # lời đã dùng - đúng kiểu trích dẫn gây hiểu lầm nhất. Nói thật thì người đọc biết
        # phải tự kiểm tra kỹ hơn; nói dối thì họ yên tâm nhầm.
        st.caption(
            "⚠️ Câu trả lời **không tự dẫn nguồn** — đây là chỗ liên quan nhất do hệ thống "
            f"chọn, KHÔNG chắc là căn cứ đã dùng: {danh_sach}"
        )
    else:
        st.caption("📎 Nguồn: " + danh_sach)


def _hien_thi_bam_nguon(bam_nguon) -> None:
    """Cho người đọc biết câu trả lời bám sát nguyên văn tài liệu tới mức nào.

    Chỉ hiện khi mức bám CAO - và đó là một quyết định có chủ đích, không phải để giấu số
    xấu. Phép đo này (rag.citation.do_bam_ngu_canh) chỉ nói được MỘT chiều: mức cao là bằng
    chứng mạnh rằng câu trả lời không bịa, còn mức thấp KHÔNG chứng minh điều gì cả - một câu
    trả lời diễn đạt lại bằng lời của mình, điều hoàn toàn hợp lệ và thường là mong muốn,
    cũng cho mức thấp.

    Hiện một dấu hiệu "bám nguồn thấp" vì thế sẽ khiến người đọc nghi ngờ oan đúng những câu
    trả lời viết tốt nhất. Nói khi có bằng chứng, im lặng khi không có - chứ không suy đoán
    theo chiều ngược lại.
    """
    if bam_nguon is None or bam_nguon < config.NGUONG_BAM_NGUON_HIEN_THI:
        return
    st.caption(
        f"✓ {bam_nguon:.0%} nội dung câu trả lời trùng **nguyên văn** với đoạn trích đã dẫn"
    )


def _hien_thi_cach_hieu(truy_van: dict) -> None:
    """Nói ra việc hệ thống đã hiểu câu hỏi này là NỐI TIẾP một câu trước đó.

    KHÔNG được im lặng làm chuyện này. Đây là một PHỎNG ĐOÁN của hệ thống về ý người dùng,
    và trình bày phỏng đoán như thể là sự thật đúng là lỗi mà §5.54 đã phải sửa một lần rồi
    (bản trước hiện đoạn điểm cao nhất dưới nhãn "Nguồn" khi model không tự dẫn nguồn).

    Hiện ra thì người dùng có đường sửa: thấy hệ thống nối nhầm vào câu khác, họ gõ lại câu
    đầy đủ. Giấu đi thì họ nhận một câu trả lời đúng-nhưng-cho-câu-hỏi-khác mà không hiểu
    vì sao - và đó là kiểu hỏng im lặng mà cả hệ thống này tồn tại để loại bỏ.
    """
    if not truy_van or not truy_van.get("la_tiep_noi"):
        return
    # st.caption(
    #     "🔎 Hiểu đây là câu hỏi **nối tiếp**, nên đã tra kèm ngữ cảnh: "
    #     f"*\"{truy_van.get('cau_hoi_chinh', '')}\"* — nối nhầm chỗ thì bạn hỏi lại đầy đủ hơn nhé."
    # )


def _hien_thi_mau_thuan(cac_mau_thuan: list) -> None:
    """Cảnh báo khi hai nguồn được dẫn đang nói ngược nhau.

    Đây là thứ người đọc không tự thấy được: họ mở từng file riêng thì mỗi file đều nhất
    quán với chính nó. Chỉ khi đặt cạnh nhau mới lộ ra, mà đặt cạnh nhau đúng là việc hệ
    thống vừa làm khi gom các đoạn trích vào một câu trả lời.

    Cố ý KHÔNG kết luận nguồn nào đúng: hệ thống không có căn cứ nào để phân xử (không biết
    tài liệu nào mới hơn, không biết môn học nào ưu tiên bản nào). Việc của nó là chỉ ra chỗ
    xung đột và đủ toạ độ để người đọc tự mở ra đối chiếu.
    """
    if not cac_mau_thuan:
        return
    for m in cac_mau_thuan:
        st.warning(
            f"⚠️ **Hai nguồn nói khác nhau** — "
            f"`{m['nguon_a']}` (trang/slide {m['trang_a']}) và "
            f"`{m['nguon_b']}` (trang/slide {m['trang_b']}): {m['noi_dung_xung_dot']}\n\n"
            "Hệ thống không tự phân xử nguồn nào đúng — hãy mở cả hai chỗ trên và tự đối chiếu.",
            icon="⚠️",
        )


def _chay_va_hien_theo_luong(pipeline, cau_hoi: str, nguon_cho_phep, lich_su) -> dict:
    """Chạy hỏi-đáp ở chế độ streaming và vẽ dần kết quả ra màn hình.

    Ba vùng hiển thị, tương ứng ba giai đoạn người dùng thật sự quan tâm:
      1. khung trạng thái  - "đang truy xuất" -> "đã tìm N đoạn, đang soạn trả lời (23s)".
         Cập nhật kèm số giây đang trôi, để cái đang chạy luôn phân biệt được với cái đã treo.
      2. phần suy luận     - đuôi chuỗi suy luận nội bộ của model, nằm THU GỌN trong khung
         trạng thái. Hiện ra để chứng minh hệ thống đang chạy chứ không đứng hình; thu gọn
         vì đây là nháp của model, không phải câu trả lời (và thường bằng tiếng Anh).
      3. câu trả lời       - chữ chạy dần, có con trỏ nhấp nháy ở cuối cho tới khi xong.

    Vẽ lại bị GIÃN CÁCH (config.GIAN_CACH_VE_LAI_GIAY) chứ không vẽ theo từng token: mỗi
    lần vẽ Streamlit đẩy lại cả khối markdown qua websocket, làm theo token thì trình duyệt
    nghẽn mà mắt người cũng không đọc kịp.
    """
    khung_trang_thai = st.status("Đang truy xuất trong tài liệu...", expanded=False)
    o_suy_luan = khung_trang_thai.empty()
    o_cau_tra_loi = st.empty()

    cac_manh_suy_luan, cac_manh_tra_loi = [], []
    ket_qua = None
    moc_bat_dau = time.perf_counter()
    lan_ve_cuoi = 0.0
    nhan_trang_thai = "Đang truy xuất trong tài liệu..."

    def _ve_lai(ep_buoc: bool = False) -> None:
        nonlocal lan_ve_cuoi
        gio = time.perf_counter()
        if not ep_buoc and gio - lan_ve_cuoi < config.GIAN_CACH_VE_LAI_GIAY:
            return
        lan_ve_cuoi = gio
        khung_trang_thai.update(label=f"{nhan_trang_thai} ({gio - moc_bat_dau:.0f}s)")
        if cac_manh_suy_luan:
            duoi = "".join(cac_manh_suy_luan)[-config.SO_KY_TU_SUY_LUAN_HIEN:]
            o_suy_luan.caption("💭 " + " ".join(duoi.split()))
        if cac_manh_tra_loi:
            o_cau_tra_loi.markdown(bo_so_trich_dan("".join(cac_manh_tra_loi)) + " ▌")

    for su_kien in pipeline.hoi_dap_theo_luong(
        cau_hoi, nguon_cho_phep=nguon_cho_phep, lich_su=lich_su
    ):
        loai = su_kien["loai"]
        if loai == "dang_doi_chieu":
            # Bước này chạy SAU khi câu trả lời đã hiện xong nên không làm chậm chữ đầu tiên,
            # nhưng vẫn phải nói ra: khung trạng thái đang ghi "đang soạn câu trả lời" trong
            # khi câu trả lời đã xong rồi thì người dùng tưởng hệ thống bị treo.
            nhan_trang_thai = "Đang đối chiếu chéo các nguồn"
            _ve_lai(ep_buoc=True)
        elif loai == "truy_xuat_xong":
            so_doan = len(su_kien["cac_chunk"])
            if so_doan:
                cac_nguon = sorted({c["nguon"] for c in su_kien["cac_chunk"]})
                mo_ta_nguon = cac_nguon[0] if len(cac_nguon) == 1 else f"{len(cac_nguon)} tài liệu"
                nhan_trang_thai = (
                    f"Đã tìm {so_doan} đoạn liên quan trong {mo_ta_nguon} "
                    f"({su_kien['giay']:.1f}s) — đang soạn câu trả lời"
                )
            else:
                nhan_trang_thai = "Không có đoạn nào đủ liên quan trong tài liệu"
            _ve_lai(ep_buoc=True)
        elif loai == "suy_luan":
            cac_manh_suy_luan.append(su_kien["them"])
            _ve_lai()
        elif loai == "cau_tra_loi":
            cac_manh_tra_loi.append(su_kien["them"])
            _ve_lai()
        elif loai == "xong":
            ket_qua = su_kien["ket_qua"]

    # Vẽ bản cuối: bỏ con trỏ nhấp nháy, đóng khung trạng thái lại.
    o_cau_tra_loi.markdown(bo_so_trich_dan(ket_qua["cau_tra_loi"]))
    o_suy_luan.empty()
    khung_trang_thai.update(
        label=f"Xong sau {ket_qua['do_tre']['tong']:.1f} giây", state="complete", expanded=False
    )
    return ket_qua


def _dat_cau_hoi(cau_hoi: str) -> None:
    """Nhận 1 câu hỏi rồi rerun NGAY - không gọi LLM ở đây.

    Việc gọi LLM (chậm) được tách sang lần chạy kế tiếp, lúc đó mọi widget dễ bấm nhầm đã
    render ở trạng thái disabled rồi. Nếu gọi thẳng tại đây, một cú bấm bất kỳ trong lúc chờ
    sẽ khiến Streamlit HUỶ NGANG lần chạy hiện tại (đang ở giữa lệnh gọi Ollama) để chạy lại
    từ đầu, làm mất câu trả lời đang sinh dở - bug đã gặp thực tế.

    Dùng chung cho cả ô nhập lẫn các nút gợi ý ở màn hình trống, để hai lối vào đi đúng một
    đường mã, không có lối nào lách được cơ chế trên.
    """
    st.session_state.messages.append({"role": "user", "content": cau_hoi})
    st.session_state.cau_hoi_dang_xu_ly = cau_hoi
    st.session_state.dang_xu_ly = True
    st.rerun()


embedding_service = lay_embedding_service()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = tai_index_da_co()

if "messages" not in st.session_state:
    # Chỉ giữ lịch sử chat trong phiên Streamlit hiện tại (session_state mặc định) -
    # đúng phạm vi đã chốt: không cần conversation memory nhiều phiên/session ID phức tạp.
    st.session_state.messages = []

if "uploader_key_n" not in st.session_state:
    # Đổi key của file_uploader sau mỗi lần upload thành công (xem thanh bên) để Streamlit
    # coi đó là 1 widget MỚI, rỗng - tránh bug: st.file_uploader() trả về cùng 1 danh sách
    # file trên MỌI lần rerun cho tới khi người dùng tự bấm "x" xoá khỏi widget, khiến file
    # cũ bị ghi đè lại vào data/raw/ ngay sau khi vừa xoá xong (đây là nguyên nhân gốc của
    # bug "xoá không được" - đã fix ở đây, không phải né tránh).
    st.session_state.uploader_key_n = 0

if "dang_xu_ly" not in st.session_state:
    # True trong lúc đang gọi LLM sinh câu trả lời. Khi True, mọi widget có thể gây rerun
    # (ô nhập, nút xoá/build index, nút hội thoại mới...) đều bị disabled=True - xem
    # _dat_cau_hoi() để biết vì sao.
    st.session_state.dang_xu_ly = False

if "cau_hoi_dang_xu_ly" not in st.session_state:
    st.session_state.cau_hoi_dang_xu_ly = None


# ============================================================
# THANH BÊN: nguồn tài liệu
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.55rem; margin-bottom:1rem;">
            <span style="font-size:1.5rem;">📚</span>
            <span style="font-size:1.05rem; font-weight:700;">Hỏi đáp tài liệu</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋  Hội thoại mới",
        use_container_width=True,
        disabled=st.session_state.dang_xu_ly or not st.session_state.messages,
    ):
        # Chỉ xoá lịch sử hiển thị. Index và tài liệu giữ nguyên - đó là thứ người dùng đã
        # bỏ công chuẩn bị, không được mất chỉ vì muốn hỏi sang chuyện khác.
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Nguồn tài liệu**")
    st.caption("Bỏ tick để loại một tài liệu khỏi phạm vi trả lời.")

    file_upload = st.file_uploader(
        "Thêm tài liệu",
        type=["pdf", "pptx", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key_n}",
        disabled=st.session_state.dang_xu_ly,
    )
    if file_upload:
        for f in file_upload:
            (config.RAW_DOCS_DIR / f.name).write_bytes(f.getvalue())
        st.success(f"Đã lưu {len(file_upload)} tài liệu vào thư mục dữ liệu.")
        st.session_state.uploader_key_n += 1
        st.rerun()

    cac_file = sorted(config.RAW_DOCS_DIR.glob("*"))
    nguon_da_chon = {}
    if cac_file:
        for f in cac_file:
            c_checkbox, c_xoa = st.columns([5, 1], vertical_alignment="center")
            with c_checkbox:
                # value=True chỉ áp dụng lần đầu tạo widget - lần chọn trước của người dùng
                # (nếu có) được Streamlit tự giữ qua key, không bị reset mỗi lần rerun.
                nguon_da_chon[f.name] = st.checkbox(
                    f.name, value=True, key=f"nguon_{f.name}", disabled=st.session_state.dang_xu_ly
                )
            with c_xoa:
                # popover làm bước xác nhận trước khi xoá thật - xoá file là hành động
                # không thể hoàn tác, không nên xảy ra chỉ vì bấm nhầm 1 nút.
                with st.popover("🗑️", disabled=st.session_state.dang_xu_ly):
                    st.write(f"Xóa **{f.name}**?")
                    if st.button("Xác nhận xóa", key=f"xoa_{f.name}", type="primary"):
                        f.unlink()
                        # Xoá NGAY khỏi index đang dùng (không đợi bấm "Đọc tài liệu") - dữ
                        # liệu liên quan tới file vừa xoá phải mất khỏi hệ thống ngay lập tức.
                        if st.session_state.vector_store is not None:
                            so_xoa = st.session_state.vector_store.xoa_theo_nguon(f.name)
                            if so_xoa > 0:
                                if st.session_state.vector_store.so_luong_vector == 0:
                                    st.session_state.vector_store = None
                                    for f_index in (
                                        config.FAISS_INDEX_FILE,
                                        config.METADATA_MAPPING_FILE,
                                        config.INDEX_INFO_FILE,
                                    ):
                                        f_index.unlink(missing_ok=True)
                                else:
                                    st.session_state.vector_store.luu()
                        st.rerun()
    else:
        st.caption("Chưa có tài liệu nào.")

    if cac_file and not any(nguon_da_chon.values()):
        # Bỏ tick hết thì mọi câu hỏi đều nhận về câu từ chối - đúng logic, nhưng nhìn từ
        # phía người dùng thì giống hệt hệ thống bị hỏng. Nói thẳng nguyên nhân ngay tại chỗ
        # gây ra nó, thay vì để họ đi hỏi vài câu rồi tự đoán.
        st.warning("Chưa chọn tài liệu nào — mọi câu hỏi sẽ bị từ chối vì không có gì để tra.")

    st.divider()

    if st.button(
        "Đọc tài liệu", use_container_width=True, disabled=st.session_state.dang_xu_ly
    ):
        with st.spinner("Đang đọc tài liệu, vui lòng chờ trong giây lát!"):
            st.session_state.vector_store = xay_dung_lai_index(
                embedding_service, st.session_state.vector_store
            )
        if st.session_state.vector_store is not None:
            st.success(f"Đã build index với {st.session_state.vector_store.so_luong_vector} chunk.")

    if st.session_state.vector_store is not None:
        st.caption(
            f"📊 {st.session_state.vector_store.so_luong_vector} chunk · "
            f"{config.OLLAMA_MODEL}"
        )
        # Index cũ + model embedding mới = kết quả truy xuất sai HOÀN TOÀN nhưng không hề
        # báo lỗi (2 model có thể cùng số chiều vector nên FAISS vẫn chạy bình thường). Đây
        # là loại hỏng hóc không thể tự nhận ra qua câu trả lời, nên phải cảnh báo rõ ràng.
        ly_do = st.session_state.vector_store.ly_do_khong_tuong_thich()
        if ly_do:
            st.warning(f"⚠️ {ly_do}\n\nHãy bấm **đọc tài liệu** để cập nhật.")

    # BỘ NHỚ ĐỆM INGESTION. Hiện ra vì hai lẽ, đều là chuyện người dùng cần biết chứ không
    # phải chi tiết nội bộ: (1) nó chiếm chỗ thật trên đĩa - ảnh render, chú thích vision và
    # vector embedding của cả corpus; (2) khi ai đó nghi ngờ hệ thống đang trả lời bằng nội
    # dung cũ, họ phải có một cách dứt khoát để loại bỏ giả thuyết đó. Nút xoá ở đây là câu
    # trả lời cho cả hai, và nó an toàn tuyệt đối: mọi thứ trong cache đều tính lại được.
    dung_luong = bo_nho_dem.dung_luong_cache()
    if dung_luong:
        cot_thong_tin, cot_nut = st.columns([2, 1])
        cot_thong_tin.caption(f"💾 Cache đọc tài liệu: {dung_luong / (1 << 20):.0f} MB")
        if cot_nut.button("Xoá cache", use_container_width=True,
                          disabled=st.session_state.dang_xu_ly):
            bo_nho_dem.xoa_cache()
            st.toast("Đã xoá cache. Lần đọc tài liệu tới sẽ xử lý lại từ đầu.")
            st.rerun()

    # Ollama chưa chạy = truy xuất vẫn ra kết quả nhưng không sinh nổi một chữ nào. Nói ngay
    # ở đây, trước khi người dùng gõ câu hỏi đầu tiên: đây là nguyên nhân số một khiến hệ
    # thống "không chạy được" trên một máy mới, và nó không có triệu chứng nào khác.
    loi_llm = lay_loi_may_chu_llm()
    if loi_llm:
        st.warning(f"⚠️ {loi_llm}")


# ============================================================
# KHUNG CHÍNH: hỏi đáp
# ============================================================
if st.session_state.vector_store is None:
    # Màn hình khởi đầu khi chưa có gì để hỏi. Nói thẳng ba bước phải làm, thay vì một dòng
    # thông báo chung chung rồi để người dùng tự mò trong thanh bên.
    st.markdown("### 👋 Bắt đầu")
    st.markdown(
        "Chưa có tài liệu nào. Ba bước để chạy hệ thống:\n\n"
        "1. Mở thanh bên (nút **»** ở góc trên bên trái nếu đang thu gọn)\n"
        "2. Tải lên tài liệu **PDF / PPTX / DOCX**\n"
        "3. Bấm **Đọc tài liệu**"
    )
    st.stop()

# --- Màn hình trống: lời chào + vài câu hỏi gợi ý bấm được ---
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; margin: 3.5rem 0 2rem;">
            <div style="font-size:2.6rem; margin-bottom:0.6rem;">📚</div>
            <div style="font-size:1.45rem; font-weight:700;">Hỏi gì về tài liệu của bạn?</div>
            <div style="color:#6b6b6b; margin-top:0.45rem; font-size:0.95rem;">
                Trả lời bám sát nguồn, kèm trích dẫn tên file và số trang để bạn tự đối chiếu.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Gợi ý bám vào ĐÚNG tài liệu đang có, không phải câu mẫu chung chung: gợi ý mà hỏi về
    # thứ không nằm trong corpus thì lần thử đầu tiên của người dùng nhận về câu từ chối -
    # ấn tượng đầu tệ nhất có thể. Tên file lấy từ chính danh sách nguồn đang bật.
    ten_dang_bat = [ten for ten, chon in nguon_da_chon.items() if chon]
    cac_goi_y = [
        "Tóm tắt những ý chính trong tài liệu.",
        "Liệt kê các khái niệm quan trọng và giải thích ngắn gọn.",
    ]
    if ten_dang_bat:
        cac_goi_y.insert(0, f"Tài liệu “{ten_dang_bat[0]}” nói về những nội dung gì?")

    with st.container(key="goi_y"):
        for i, goi_y in enumerate(cac_goi_y[:3]):
            if st.button(
                goi_y,
                key=f"goi_y_{i}",
                use_container_width=True,
                disabled=st.session_state.dang_xu_ly,
            ):
                _dat_cau_hoi(goi_y)

    # Mách nước về chế độ KIỂM CHỨNG - viết thành chữ chứ không làm thành nút bấm, vì đây là
    # một CÁCH HỎI chứ không phải một câu hỏi cụ thể: người dùng phải tự nêu khẳng định của
    # mình thì mới có gì để đối chiếu. Đây cũng là tính năng phân biệt hệ thống này với một
    # chatbot thường (§5.22), mà người dùng sẽ không tự nghĩ ra là mình hỏi được kiểu đó.
    st.caption(
        "💡 Bạn cũng có thể **đưa ra một khẳng định để kiểm chứng**. Hệ thống sẽ đối chiếu với tài liệu và kết luận "
        "ĐÚNG / SAI / KHÔNG ĐỀ CẬP kèm trích nguyên văn căn cứ, thay vì trả lời thuận theo "
        "giả định của bạn."
    )

# --- Lịch sử hội thoại ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📚"):
        # Lịch sử LƯU bản gốc còn số; chỉ bỏ số ở đúng lúc vẽ ra màn hình.
        st.markdown(bo_so_trich_dan(msg["content"])
                    if msg["role"] == "assistant" else msg["content"])
        if msg["role"] == "assistant":
            _hien_thi_cach_hieu(msg.get("truy_van"))
            _hien_thi_trich_dan(msg.get("trich_dan", []))
            _hien_thi_bam_nguon(msg.get("bam_nguon"))
            _hien_thi_mau_thuan(msg.get("mau_thuan", []))
          

# --- Sinh câu trả lời cho câu hỏi đang chờ ---
# Chạy NGAY TẠI ĐÂY - tức trong cùng lần chạy mà ô nhập và các nút đã render với
# disabled=True, nên không thể bị một cú bấm khác huỷ ngang giữa chừng. Câu hỏi (đã có trong
# session_state.messages) hiện ra từ vòng lặp lịch sử bên trên; chỉ còn phải hiện câu trả lời.
if st.session_state.dang_xu_ly:
    cac_nguon_duoc_chon = {ten for ten, chon in nguon_da_chon.items() if chon}
    nguon_cho_phep = cac_nguon_duoc_chon if cac_nguon_duoc_chon != set(nguon_da_chon) else None
    pipeline = lay_pipeline(embedding_service, st.session_state.vector_store)

    # Bắt lỗi RỘNG quanh cả khối là có chủ đích, không phải cẩu thả. Bất kỳ lỗi nào thoát ra
    # khỏi đây mà không được xử lý đều để lại dang_xu_ly=True: Streamlit vẽ traceback thô,
    # còn ô nhập và MỌI nút vẫn đang ở trạng thái disabled - người dùng không thao tác được
    # gì nữa, kể cả sau khi đã sửa xong nguyên nhân, cho tới lúc tự tải lại trang. Nói cách
    # khác, một lỗi tạm thời ở bước gọi LLM làm hỏng luôn cả phiên làm việc. Nguyên nhân hay
    # gặp nhất chính là máy chủ Ollama chưa chạy ([WinError 10061]).
    # Trong khối này không gọi st.rerun()/st.stop(), nên `except Exception` không nuốt nhầm
    # ngoại lệ điều khiển luồng của Streamlit; st.rerun() được đặt hẳn ra ngoài bên dưới.
    try:
        with st.chat_message("assistant", avatar="📚"):
            # Không còn cho chọn Top-K trên UI nữa - luôn dùng đúng 1 giá trị config.TOP_K để
            # hành vi truy xuất nhất quán, dễ kiểm soát/tái hiện.
            # Lịch sử BỎ tin nhắn cuối - đó chính là câu hỏi đang xử lý, đã được _dat_cau_hoi()
            # thêm vào messages trước khi rerun. Để nguyên thì bước viết lại nhìn thấy câu
            # hỏi hiện tại nằm trong cả "lịch sử" lẫn "câu hỏi mới", và model hay hiểu thành
            # người dùng vừa hỏi lại y hệt lần trước.
            lich_su = st.session_state.messages[:-1]
            if config.BAT_STREAMING:
                ket_qua = _chay_va_hien_theo_luong(
                    pipeline, st.session_state.cau_hoi_dang_xu_ly, nguon_cho_phep, lich_su
                )
            else:
                # Chế độ dự phòng (BAT_STREAMING=0): spinner + trả một cục. Câu hỏi dạng kiểm
                # chứng ("... đúng không?") được bật lại chế độ suy luận của model nên chậm hơn
                # hẳn - nói trước để người dùng không tưởng là hệ thống bị treo.
                dang_kiem_chung = la_cau_hoi_kiem_chung(st.session_state.cau_hoi_dang_xu_ly)
                thong_bao_cho = (
                    "Đang đối chiếu khẳng định với tài liệu..."
                    if dang_kiem_chung
                    else "Đang truy xuất và sinh câu trả lời..."
                )
                with st.spinner(thong_bao_cho):
                    ket_qua = pipeline.hoi_dap(
                        st.session_state.cau_hoi_dang_xu_ly,
                        nguon_cho_phep=nguon_cho_phep,
                        lich_su=lich_su,
                    )
                st.markdown(bo_so_trich_dan(ket_qua["cau_tra_loi"]))

            # Chỉ hiển thị những nguồn câu trả lời thật sự tham chiếu tới, thay vì mặc định lấy
            # đoạn điểm cao nhất (hay lệch với nội dung đã trả lời).
            trich_dan = loc_theo_tham_chieu(ket_qua["cac_chunk_nguon"], ket_qua["cau_tra_loi"])
            _hien_thi_cach_hieu(ket_qua.get("truy_van"))
            _hien_thi_trich_dan(trich_dan)
            _hien_thi_bam_nguon(ket_qua.get("bam_nguon"))
            _hien_thi_mau_thuan(ket_qua.get("mau_thuan", []))
        # Trích dẫn lưu NGAY TRONG tin nhắn - mỗi câu trả lời tự chứa trích dẫn của chính nó,
        # không dùng 1 biến "trích dẫn hiện tại" dùng chung (dễ lệch pha với câu hỏi mới).
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": ket_qua["cau_tra_loi"],
                "trich_dan": trich_dan,
                "do_tre": ket_qua.get("do_tre", {}),
                "bam_nguon": ket_qua.get("bam_nguon"),
                # Lưu ngay trong tin nhắn, cùng lý do với trích dẫn: mỗi câu trả lời tự chứa
                # mọi thứ thuộc về nó, nên vẽ lại lịch sử ở lần rerun sau vẫn ra đúng như lúc
                # vừa sinh, không phụ thuộc trạng thái dùng chung nào.
                "truy_van": ket_qua.get("truy_van"),
                "mau_thuan": ket_qua.get("mau_thuan", []),
            }
        )
    except Exception as loi:
        logger.exception("Lỗi khi sinh câu trả lời cho: %s", st.session_state.cau_hoi_dang_xu_ly)
        # Lỗi kết nối Ollama đã tự mang sẵn hướng dẫn xử lý (xem LoiKhongKetNoiDuocOllama);
        # các lỗi còn lại thì ít ra cũng nêu đúng loại lỗi để còn tra được.
        thong_bao = (
            str(loi)
            if isinstance(loi, LoiKhongKetNoiDuocOllama)
            else f"Không sinh được câu trả lời — {type(loi).__name__}: {loi}"
        )
        # Ghi vào lịch sử chứ không chỉ st.error(): st.error vẽ ở lần chạy này rồi biến mất
        # ngay sau st.rerun() bên dưới, người dùng chưa kịp đọc đã mất. Lưu thành một tin
        # nhắn thì nó nằm lại đúng chỗ câu trả lời lẽ ra phải xuất hiện.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"⚠️ {thong_bao}",
                "trich_dan": [],
                "do_tre": {},
                "bam_nguon": None,
            }
        )
    st.session_state.dang_xu_ly = False
    st.session_state.cau_hoi_dang_xu_ly = None
    st.rerun()

# Ô nhập đặt ở TẦNG NGOÀI CÙNG của script (không nằm trong container/cột nào) - đó là điều
# kiện để Streamlit ghim nó xuống đáy màn hình thay vì chèn inline giữa nội dung.
cau_hoi = st.chat_input(
    "Hỏi về tài liệu của bạn...", disabled=st.session_state.dang_xu_ly
)
if cau_hoi and not st.session_state.dang_xu_ly:
    _dat_cau_hoi(cau_hoi)
