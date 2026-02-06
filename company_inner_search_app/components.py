"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct


############################################################
# 関数定義
############################################################
def display_app_title():
    """タイトル表示"""
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """回答モードのラジオボタンを表示（mode を安全に初期化）"""
    if "mode" not in st.session_state:
        st.session_state.mode = ct.ANSWER_MODE_2  # デフォルト（好みで MODE_1 にしてOK）

    col1, col2 = st.columns([100, 1])
    with col1:
        st.session_state.mode = st.radio(
            label="",
            options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
            index=0 if st.session_state.mode == ct.ANSWER_MODE_1 else 1,
            label_visibility="collapsed",
        )


def display_initial_ai_message():
    """AIメッセージの初期表示"""
    with st.chat_message("assistant"):
        st.markdown(
            "こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。"
            "上記で利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。"
        )

        st.markdown("**【「社内文書検索」を選択した場合】**")
        st.info("入力内容と関連性が高い社内文書のありかを検索できます。")
        st.code("【入力例】\n社員の育成方針に関するMTGの議事録", wrap_lines=True, language=None)

        st.markdown("**【「社内問い合わせ」を選択した場合】**")
        st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        st.code("【入力例】\n人事部に所属している従業員情報を一覧化して", wrap_lines=True, language=None)


def display_conversation_log():
    """会話ログの一覧表示（過去データが多少崩れていても落ちないように）"""
    messages = st.session_state.get("messages", [])
    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content")

        with st.chat_message(role):
            if role == "user":
                st.markdown(content if isinstance(content, str) else str(content))
                continue

            # assistant の content は dict 想定。崩れている場合はそのまま出す
            if not isinstance(content, dict):
                st.markdown(str(content))
                continue

            mode = content.get("mode")

            # ==========================================
            # 「社内文書検索」モードの表示
            # ==========================================
            if mode == ct.ANSWER_MODE_1:
                if not content.get("no_file_path_flg", False):
                    main_message = content.get("main_message", "")
                    main_file_path = content.get("main_file_path", "")
                    main_page_number = content.get("main_page_number")

                    if main_message:
                        st.markdown(main_message)

                    if main_file_path:
                        icon = utils.get_source_icon(main_file_path)
                        if main_page_number is not None:
                            st.success(f"{main_file_path}（page: {main_page_number}）", icon=icon)
                        else:
                            st.success(f"{main_file_path}", icon=icon)

                    sub_message = content.get("sub_message")
                    sub_choices = content.get("sub_choices", [])

                    if sub_message and sub_choices:
                        st.markdown(sub_message)

                        for sub_choice in sub_choices:
                            src = sub_choice.get("source", "")
                            if not src:
                                continue
                            icon = utils.get_source_icon(src)
                            if "page_number" in sub_choice:
                                st.info(f"{src}（page: {sub_choice['page_number']}）", icon=icon)
                            else:
                                st.info(f"{src}", icon=icon)
                else:
                    # no match のとき
                    st.markdown(content.get("answer", ct.NO_DOC_MATCH_MESSAGE))

            # ==========================================
            # 「社内問い合わせ」モードの表示
            # ==========================================
            else:
                st.markdown(content.get("answer", ""))

                file_info_list = content.get("file_info_list")
                message_label = content.get("message", "情報源")

                if file_info_list:
                    st.divider()
                    st.markdown(f"##### {message_label}")
                    for file_info in file_info_list:
                        icon = utils.get_source_icon(file_info)
                        st.info(file_info, icon=icon)


def display_search_llm_response(llm_response):
    """
    「社内文書検索」モードにおけるLLMレスポンスを表示し、
    会話ログに保存する content(dict) を返す
    """
    content = {"mode": ct.ANSWER_MODE_1}

    # context があり、かつ「該当資料なし」ではない場合
    if llm_response.get("context") and llm_response.get("answer") != ct.NO_DOC_MATCH_ANSWER:
        context_docs = llm_response["context"]

        main_doc = context_docs[0]
        main_file_path = main_doc.metadata.get("source", "")
        main_page_number = main_doc.metadata.get("page")

        main_message = "入力内容に関する情報は、以下のファイルに含まれている可能性があります。"
        st.markdown(main_message)

        if main_file_path:
            icon = utils.get_source_icon(main_file_path)
            if main_page_number is not None:
                st.success(f"{main_file_path}（page: {main_page_number}）", icon=icon)
            else:
                st.success(f"{main_file_path}", icon=icon)

        # サブ候補（重複除去）
        sub_choices = []
        seen = set([main_file_path])

        for doc in context_docs[1:]:
            sub_file_path = doc.metadata.get("source", "")
            if not sub_file_path or sub_file_path in seen:
                continue
            seen.add(sub_file_path)

            if "page" in doc.metadata:
                sub_choices.append({"source": sub_file_path, "page_number": doc.metadata["page"]})
            else:
                sub_choices.append({"source": sub_file_path})

        if sub_choices:
            sub_message = "その他、ファイルありかの候補を提示します。"
            st.markdown(sub_message)
            for sc in sub_choices:
                icon = utils.get_source_icon(sc["source"])
                if "page_number" in sc:
                    st.info(f"{sc['source']}（page: {sc['page_number']}）", icon=icon)
                else:
                    st.info(f"{sc['source']}", icon=icon)

        # 会話ログ保存用
        content["main_message"] = main_message
        content["main_file_path"] = main_file_path
        if main_page_number is not None:
            content["main_page_number"] = main_page_number
        if sub_choices:
            content["sub_message"] = sub_message
            content["sub_choices"] = sub_choices

    else:
        # no match
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)
        content["answer"] = ct.NO_DOC_MATCH_MESSAGE
        content["no_file_path_flg"] = True

    return content


def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示し、
    会話ログに保存する content(dict) を返す
    """
    answer = llm_response.get("answer", "")
    st.markdown(answer)

    content = {"mode": ct.ANSWER_MODE_2, "answer": answer}

    # 文脈が見つかった場合のみ情報源を表示
    if answer != ct.INQUIRY_NO_MATCH_ANSWER:
        st.divider()

        message = "情報源"
        st.markdown(f"##### {message}")

        file_info_list = []
        seen = set()

        for document in llm_response.get("context", []):
            file_path = document.metadata.get("source", "")
            if not file_path or file_path in seen:
                continue
            seen.add(file_path)

            page = document.metadata.get("page")
            file_info = f"{file_path}（page: {page}）" if page is not None else file_path

            icon = utils.get_source_icon(file_path)
            st.info(file_info, icon=icon)

            file_info_list.append(file_info)

        if file_info_list:
            content["message"] = message
            content["file_info_list"] = file_info_list

    return content
