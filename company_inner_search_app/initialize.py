"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import unicodedata

from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import constants as ct


############################################################
# 2. 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み（ローカル用）
load_dotenv()

# Streamlit Cloud: st.secrets から API Key を環境変数に反映
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# WebBaseLoader 対策（環境によって User-Agent が無いと警告/失敗しやすい）
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (compatible; StreamlitRAGApp/1.0; +https://streamlit.io)"
)


############################################################
# 3. ユーティリティ
############################################################
def _is_writable_dir(path: str) -> bool:
    """ディレクトリが作成・書き込み可能かを軽くチェック"""
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except Exception:
        return False


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    Args:
        s: 調整を行う文字列
    Returns:
        調整を行った文字列
    """
    if type(s) is not str:
        return s

    if sys.platform.startswith("win"):
        s = unicodedata.normalize("NFC", s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s

    return s


############################################################
# 4. 初期化エントリ
############################################################
def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    initialize_session_state()
    initialize_session_id()
    initialize_logger()
    initialize_retriever()


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_logger():
    """
    ログ出力の設定（Streamlit Cloud では /tmp を優先的に使う）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # 既に設定済みなら二重登録を防止
    if logger.hasHandlers():
        return

    # Streamlit Cloud 対策: 書き込み不可なら /tmp/logs にフォールバック
    log_dir = getattr(ct, "LOG_DIR_PATH", "./logs")
    if not _is_writable_dir(log_dir):
        log_dir = "/tmp/logs"
        os.makedirs(log_dir, exist_ok=True)

    log_file = getattr(ct, "LOG_FILE", "app.log")

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, "
        f"session_id={st.session_state.session_id}: %(message)s"
    )

    logger.setLevel(logging.INFO)

    # ファイルログ（失敗したら標準出力ログへ）
    try:
        file_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, log_file),
            when="D",
            encoding="utf8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info(f"Logger initialized. log_dir={log_dir}")


############################################################
# 5. Retriever 構築（Streamlit rerun 対策でキャッシュ）
############################################################
@st.cache_resource(show_spinner=False)
def _build_retriever_cached():
    """
    RAGのRetriever（ベクターストアから検索するオブジェクト）を構築して返す。
    Streamlit の rerun でも毎回作り直さないよう cache_resource を使う。
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        # Cloud/Local どちらでも分かりやすく落とす
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "Set it in Streamlit Secrets (OPENAI_API_KEY) or in your local .env file."
        )

    docs_all = load_data_sources()

    # Windows 対策（metadata含む）
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in list(doc.metadata.keys()):
            doc.metadata[key] = adjust_string(doc.metadata[key])

    embeddings = OpenAIEmbeddings()

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    # Streamlit Cloud のファイル永続化は不安定なので /tmp を使う（任意）
    persist_dir = "/tmp/chroma"
    try:
        os.makedirs(persist_dir, exist_ok=True)
        db = Chroma.from_documents(
            splitted_docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    except Exception as e:
        # persist がダメでもメモリで動かす
        logger.warning(f"Chroma persist_directory failed, fallback to in-memory. error={e}")
        db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    return db.as_retriever(search_kwargs={"k": 3})


def initialize_retriever():
    """
    画面読み込み時にRAGのRetrieverを用意
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if "retriever" in st.session_state:
        return

    try:
        st.session_state.retriever = _build_retriever_cached()
        logger.info("Retriever initialized.")
    except Exception as e:
        logger.exception("Failed to initialize retriever.")
        st.error(f"Retriever initialization failed: {e}")
        # アプリをこの時点で止めたい場合
        st.stop()


############################################################
# 6. データ読み込み
############################################################
def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み
    Returns:
        読み込んだデータソース（Documents）
    """
    docs_all = []
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    # Webページ読み込み
    web_docs_all = []
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        web_docs_all.extend(web_docs)

    docs_all.extend(web_docs_all)
    return docs_all


def recursive_file_check(path, docs_all):
    """
    データソースの読み込み（フォルダなら再帰的に探索）
    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    if not path:
        return

    if os.path.isdir(path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            recursive_file_check(full_path, docs_all)
    else:
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み
    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    file_extension = os.path.splitext(path)[1]
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        loader_cls = ct.SUPPORTED_EXTENSIONS[file_extension]
        loader = loader_cls(path)
        docs = loader.load()
        docs_all.extend(docs)
