"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
import logging

from dotenv import load_dotenv
import streamlit as st

import utils
from initialize import initialize
import components as cn
import constants as ct


############################################################
# 2. 設定関連
############################################################
# ローカル実行時の .env 読み込み（Cloud では initialize.py 側で st.secrets を優先）
load_dotenv()

# ブラウザタブの表示文言を設定
st.set_page_config(page_title=ct.APP_NAME)

# ログ出力を行うためのロガーの設定（initialize() 内で handler が付与される想定）
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 3. 初期化処理
############################################################
try:
    initialize()
except Exception:
    # initialize() が途中で落ちても表示だけは行う
    try:
        logger.exception(ct.INITIALIZE_ERROR_MESSAGE)
    except Exception:
        pass

    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()

# アプリ起動時のログ出力（初回だけ）
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


############################################################
# 4. 初期表示
############################################################
cn.display_app_title()
cn.display_select_mode()          # ここで st.session_state.mode がセットされる想定
cn.display_initial_ai_message()


############################################################
# 5. 会話ログの表示
############################################################
try:
    cn.display_conversation_log()
except Exception as e:
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()


############################################################
# 6. チャット入力の受け付け
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# 7. チャット送信時の処理
############################################################
if chat_message:
    # mode が未設定でも落ちないようにデフォルトを用意
    current_mode = st.session_state.get("mode", ct.ANSWER_MODE_2)

    # ==========================================
    # 7-1. ユーザーメッセージの表示
    # ==========================================
    logger.info({"message": chat_message, "application_mode": current_mode})

    with st.chat_message("user"):
        st.markdown(chat_message)

    # ==========================================
    # 7-2. LLMからの回答取得
    # ==========================================
    with st.spinner(ct.SPINNER_TEXT):
        try:
            llm_response = utils.get_llm_response(chat_message)
        except Exception as e:
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            st.stop()

    # ==========================================
    # 7-3. LLMからの回答表示
    # ==========================================
    with st.chat_message("assistant"):
        try:
            if current_mode == ct.ANSWER_MODE_1:
                content = cn.display_search_llm_response(llm_response)
            else:
                content = cn.display_contact_llm_response(llm_response)

            logger.info({"message": content, "application_mode": current_mode})
        except Exception as e:
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            st.stop()

    # ==========================================
    # 7-4. 会話ログへの追加
    # ==========================================
    st.session_state.messages.append({"role": "user", "content": chat_message})
    st.session_state.messages.append({"role": "assistant", "content": content})
