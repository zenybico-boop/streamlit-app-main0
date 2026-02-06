"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from dotenv import load_dotenv
import streamlit as st

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.messages import AIMessage  # ★重要：AI側メッセージ用
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み（ローカル用）
load_dotenv()


############################################################
# 関数定義
############################################################
def get_source_icon(source: str):
    """
    メッセージと一緒に表示するアイコンの種類を取得
    """
    if isinstance(source, str) and source.startswith("http"):
        return ct.LINK_SOURCE_ICON
    return ct.DOC_SOURCE_ICON


def build_error_message(message: str):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message: str):
    """
    LLMからの回答取得
    """
    # mode が未設定でも落ちないように
    current_mode = st.session_state.get("mode", ct.ANSWER_MODE_2)

    # Retriever が初期化されていない場合は明確にエラー
    if "retriever" not in st.session_state:
        raise RuntimeError("Retriever is not initialized. Please call initialize() first.")

    # API Key がない場合は分かりやすくエラー
    # （Cloud: st.secrets から initialize.py が設定する想定）
    import os
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in Streamlit Secrets (OPENAI_API_KEY) or local .env."
        )

    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでも理解できる独立入力を生成するプロンプト
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # モードでプロンプトを切り替え
    if current_mode == ct.ANSWER_MODE_1:
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY

    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # History-aware Retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # QA chain + RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 実行
    llm_response = chain.invoke(
        {"input": chat_message, "chat_history": st.session_state.chat_history}
    )

    # ★重要：chat_history には Message オブジェクトを入れる
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=chat_message),
            AIMessage(content=llm_response.get("answer", "")),
        ]
    )

    return llm_response
