"""Giao diện Streamlit: quản lý tài liệu ở thanh bên, hỏi-đáp có trích dẫn ở khung chính."""

import logging
import time

import streamlit as st

import config
from rag import bo_nho_dem, do_thoi_gian, tai_nguyen_gpu
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
    return EmbeddingService()


@st.cache_resource
def lay_reranker_service():
    return tao_reranker_neu_bat()


@st.cache_data(ttl=10, show_spinner=False)
def lay_loi_may_chu_llm():
    """Trạng thái Ollama, để cảnh báo NGAY ở thanh bên thay vì đợi người dùng hỏi rồi mới lỗi."""
    return kiem_tra_may_chu_llm()


def tai_index_da_co():
    """Load FAISS index đã build từ lần trước (nếu có) - tránh phải build lại mỗi lần mở app."""
    if config.FAISS_INDEX_FILE.exists() and config.METADATA_MAPPING_FILE.exists():
        return VectorStore.tai()
    return None


def lay_pipeline(embedding_service: EmbeddingService, store: VectorStore) -> RagPipeline:
    """Giữ lại 1 RagPipeline cho mỗi VectorStore đang dùng, thay vì tạo mới mỗi lần hỏi."""
    if st.session_state.get("pipeline_cho_store") is not store:
        st.session_state.pipeline = RagPipeline(
            embedding_service, store, reranker_service=lay_reranker_service()
        )
        st.session_state.pipeline_cho_store = store
    return st.session_state.pipeline


def _store_dung_lai_duoc(store):
    """Index đang có trong phiên có dùng lại được cho một lần build TĂNG DẦN không?"""
    if not config.BAT_INDEX_TANG_DAN or store is None or store.so_luong_vector == 0:
        return False
    ly_do = store.ly_do_khong_tuong_thich()
    if ly_do:
        logger.info("Không build tăng dần được, sẽ dựng lại index từ đầu: %s", ly_do)
        return False
    return True


def xay_dung_lai_index(embedding_service: EmbeddingService, store_dang_dung=None):
    """Chạy luồng Ingestion: đọc tài liệu trong data/raw -> chunk -> embed -> lưu index."""
    do_thoi_gian.dat_lai()
    tai_nguyen_gpu.bat_dau_ingestion(embedding_service)
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
        for ten in ten_can_xoa:
            so_xoa = store.xoa_theo_nguon(ten)
            logger.info("'%s' không còn trong thư mục - đã gỡ %d chunk khỏi index.", ten, so_xoa)
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

        co_noi_dung = {m["nguon"] for m in cac_trang}
        for duong_dan in can_doc:
            if duong_dan.name in co_noi_dung:
                store.bam_tai_lieu[duong_dan.name] = bam_hien_tai[duong_dan.name]

    if store.so_luong_vector == 0:
        st.warning("Chưa có tài liệu nào có nội dung đọc được trong thư mục dữ liệu.")
        return None

    store.luu()
    tai_nguyen_gpu.ket_thuc_ingestion(embedding_service=embedding_service)
    if config.BAT_PROFILING_INGESTION:
        do_thoi_gian.ghi_bao_cao("PROFILING INGESTION")
    return store


def _hien_thi_trich_dan(trich_dan: list) -> None:
    """Hiển thị nguồn của 1 câu trả lời: tên file + trang/slide, gọn trên 1 dòng."""
    if not trich_dan:
        return

    def _nhan_trang(t: dict) -> str:
        """Ghi ĐỦ khoảng trang mà đoạn trích đã đọc, không chỉ trang neo."""
        cac_trang = t.get("cac_trang") or [t["trang"]]
        if len(cac_trang) <= 1:
            return f"trang/slide {t['trang']}"
        return f"trang/slide {cac_trang[0]}–{cac_trang[-1]}"

    danh_sach = " · ".join(f"**{t['nguon']}** — {_nhan_trang(t)}" for t in trich_dan)

    if trich_dan[0].get("la_suy_doan"):
        st.caption(
            "⚠️ Câu trả lời **không tự dẫn nguồn** — đây là chỗ liên quan nhất do hệ thống "
            f"chọn, KHÔNG chắc là căn cứ đã dùng: {danh_sach}"
        )
    else:
        st.caption("📎 Nguồn: " + danh_sach)


def _hien_thi_bam_nguon(bam_nguon) -> None:
    """Cho người đọc biết câu trả lời bám sát nguyên văn tài liệu tới mức nào."""
    if bam_nguon is None or bam_nguon < config.NGUONG_BAM_NGUON_HIEN_THI:
        return
    st.caption(
        f"✓ {bam_nguon:.0%} nội dung câu trả lời trùng **nguyên văn** với đoạn trích đã dẫn"
    )


def _hien_thi_cach_hieu(truy_van: dict) -> None:
    """Nói ra việc hệ thống đã hiểu câu hỏi này là NỐI TIẾP một câu trước đó."""
    if not truy_van or not truy_van.get("la_tiep_noi"):
        return


def _hien_thi_mau_thuan(cac_mau_thuan: list) -> None:
    """Cảnh báo khi hai nguồn được dẫn đang nói ngược nhau."""
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
    """Chạy hỏi-đáp ở chế độ streaming và vẽ dần kết quả ra màn hình."""
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

    o_cau_tra_loi.markdown(bo_so_trich_dan(ket_qua["cau_tra_loi"]))
    o_suy_luan.empty()
    khung_trang_thai.update(
        label=f"Xong sau {ket_qua['do_tre']['tong']:.1f} giây", state="complete", expanded=False
    )
    return ket_qua


def _dat_cau_hoi(cau_hoi: str) -> None:
    """Nhận 1 câu hỏi rồi rerun NGAY - không gọi LLM ở đây."""
    st.session_state.messages.append({"role": "user", "content": cau_hoi})
    st.session_state.cau_hoi_dang_xu_ly = cau_hoi
    st.session_state.dang_xu_ly = True
    st.rerun()


embedding_service = lay_embedding_service()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = tai_index_da_co()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key_n" not in st.session_state:
    st.session_state.uploader_key_n = 0

if "dang_xu_ly" not in st.session_state:
    st.session_state.dang_xu_ly = False

if "cau_hoi_dang_xu_ly" not in st.session_state:
    st.session_state.cau_hoi_dang_xu_ly = None


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
                nguon_da_chon[f.name] = st.checkbox(
                    f.name, value=True, key=f"nguon_{f.name}", disabled=st.session_state.dang_xu_ly
                )
            with c_xoa:
                with st.popover("🗑️", disabled=st.session_state.dang_xu_ly):
                    st.write(f"Xóa **{f.name}**?")
                    if st.button("Xác nhận xóa", key=f"xoa_{f.name}", type="primary"):
                        f.unlink()
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
        if tai_nguyen_gpu.co_cuda():
            st.caption(
                f"⚡ GPU: rerank={tai_nguyen_gpu.thiet_bi('rerank')} · "
                f"embedding={tai_nguyen_gpu.thiet_bi('embedding')}"
            )
        else:
            st.caption(
                "🐢 Đang chạy CPU — cài PyTorch bản CUDA sẽ nhanh hơn nhiều nếu máy có GPU NVIDIA"
            )
        ly_do = st.session_state.vector_store.ly_do_khong_tuong_thich()
        if ly_do:
            st.warning(f"⚠️ {ly_do}\n\nHãy bấm **đọc tài liệu** để cập nhật.")

    dung_luong = bo_nho_dem.dung_luong_cache()
    if dung_luong:
        cot_thong_tin, cot_nut = st.columns([2, 1])
        cot_thong_tin.caption(f"💾 Cache đọc tài liệu: {dung_luong / (1 << 20):.0f} MB")
        if cot_nut.button("Xoá cache", use_container_width=True,
                          disabled=st.session_state.dang_xu_ly):
            bo_nho_dem.xoa_cache()
            st.toast("Đã xoá cache. Lần đọc tài liệu tới sẽ xử lý lại từ đầu.")
            st.rerun()

    loi_llm = lay_loi_may_chu_llm()
    if loi_llm:
        st.warning(f"⚠️ {loi_llm}")


if st.session_state.vector_store is None:
    st.markdown("### 👋 Bắt đầu")
    st.markdown(
        "Chưa có tài liệu nào. Ba bước để chạy hệ thống:\n\n"
        "1. Mở thanh bên (nút **»** ở góc trên bên trái nếu đang thu gọn)\n"
        "2. Tải lên tài liệu **PDF / PPTX / DOCX**\n"
        "3. Bấm **Đọc tài liệu**"
    )
    st.stop()

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

    st.caption(
        "💡 Bạn cũng có thể **đưa ra một khẳng định để kiểm chứng**. Hệ thống sẽ đối chiếu với tài liệu và kết luận "
        "ĐÚNG / SAI / KHÔNG ĐỀ CẬP kèm trích nguyên văn căn cứ, thay vì trả lời thuận theo "
        "giả định của bạn."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📚"):
        st.markdown(bo_so_trich_dan(msg["content"])
                    if msg["role"] == "assistant" else msg["content"])
        if msg["role"] == "assistant":
            _hien_thi_cach_hieu(msg.get("truy_van"))
            _hien_thi_trich_dan(msg.get("trich_dan", []))
            _hien_thi_bam_nguon(msg.get("bam_nguon"))
            _hien_thi_mau_thuan(msg.get("mau_thuan", []))
          

if st.session_state.dang_xu_ly:
    cac_nguon_duoc_chon = {ten for ten, chon in nguon_da_chon.items() if chon}
    nguon_cho_phep = cac_nguon_duoc_chon if cac_nguon_duoc_chon != set(nguon_da_chon) else None
    pipeline = lay_pipeline(embedding_service, st.session_state.vector_store)

    try:
        with st.chat_message("assistant", avatar="📚"):
            lich_su = st.session_state.messages[:-1]
            if config.BAT_STREAMING:
                ket_qua = _chay_va_hien_theo_luong(
                    pipeline, st.session_state.cau_hoi_dang_xu_ly, nguon_cho_phep, lich_su
                )
            else:
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

            trich_dan = loc_theo_tham_chieu(ket_qua["cac_chunk_nguon"], ket_qua["cau_tra_loi"])
            _hien_thi_cach_hieu(ket_qua.get("truy_van"))
            _hien_thi_trich_dan(trich_dan)
            _hien_thi_bam_nguon(ket_qua.get("bam_nguon"))
            _hien_thi_mau_thuan(ket_qua.get("mau_thuan", []))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": ket_qua["cau_tra_loi"],
                "trich_dan": trich_dan,
                "do_tre": ket_qua.get("do_tre", {}),
                "bam_nguon": ket_qua.get("bam_nguon"),
                "truy_van": ket_qua.get("truy_van"),
                "mau_thuan": ket_qua.get("mau_thuan", []),
            }
        )
    except Exception as loi:
        logger.exception("Lỗi khi sinh câu trả lời cho: %s", st.session_state.cau_hoi_dang_xu_ly)
        thong_bao = (
            str(loi)
            if isinstance(loi, LoiKhongKetNoiDuocOllama)
            else f"Không sinh được câu trả lời — {type(loi).__name__}: {loi}"
        )
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

cau_hoi = st.chat_input(
    "Hỏi về tài liệu của bạn...", disabled=st.session_state.dang_xu_ly
)
if cau_hoi and not st.session_state.dang_xu_ly:
    _dat_cau_hoi(cau_hoi)
