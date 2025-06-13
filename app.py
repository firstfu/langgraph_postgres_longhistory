import os
import uuid
from getpass import getpass
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from config import get_postgres_connection_string


# 讀取API金鑰
def get_api_key():
    key_path = "env.txt"
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            return f.read().strip()
    else:
        return getpass("請輸入OpenAI API金鑰: ")

# 初始化模型
model = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=get_api_key())

# 設定訊息修剪器
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# 定義提示模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你說話像個卡通人物。盡你所能按照語言{language}回答所有問題。"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 定義狀態類型
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    language: str

# 建立工作流程圖
workflow = StateGraph(state_schema=State)

# 定義模型調用函數
def call_model(state: State):
    trimmed = trimmer.invoke(state["messages"])

    print('=================================================')
    print('trimmed:', trimmed)
    print('=================================================')

    prompt = prompt_template.invoke(
        {"messages": trimmed, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

# 設定工作流程
workflow.add_edge(START, "call_model")
workflow.add_node("call_model", call_model)

# 初始化資料庫表格
def setup_database():
    """初始化 PostgreSQL 資料庫表格"""
    try:
        postgres_connection_string = get_postgres_connection_string()
        print("🔧 正在初始化資料庫表格...")

        # 創建 PostgresSaver 實例並設置表格
        with PostgresSaver.from_conn_string(postgres_connection_string) as checkpointer:
            # 這會自動創建必要的表格
            checkpointer.setup()
            print("✅ 資料庫表格初始化完成！")
            return checkpointer

    except Exception as e:
        print(f"❌ 資料庫初始化失敗: {e}")
        raise

# 會話管理函數
def get_or_create_session():
    """獲取或創建會話 ID"""
    session_file = "session.txt"

    # 檢查是否存在之前的會話
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            saved_session_id = f.read().strip()

        print(f"🔍 發現之前的會話 ID: {saved_session_id}")
        choice = input("是否要繼續之前的對話？(y/n，預設為 y): ").strip().lower()

        if choice in ["", "y", "yes", "是", "繼續"]:
            print(f"📖 繼續之前的對話...")
            return saved_session_id
        else:
            print("🆕 開始新的對話...")

    # 創建新的會話 ID
    new_session_id = str(uuid.uuid4())

    # 保存會話 ID 到檔案
    with open(session_file, "w") as f:
        f.write(new_session_id)

    print(f"💾 新會話 ID 已保存: {new_session_id}")
    return new_session_id

def show_conversation_history(app, config):
    """顯示對話歷史"""
    try:
        # 獲取當前狀態來查看歷史訊息
        current_state = app.get_state(config)
        if current_state and current_state.values.get("messages"):
            print("\n📚 對話歷史:")
            print("-" * 30)
            for i, message in enumerate(current_state.values["messages"], 1):
                if message.type == "human":
                    print(f"{i}. 你: {message.content}")
                elif message.type == "ai":
                    print(f"{i}. 🤖 AI: {message.content}")
            print("-" * 30)
        else:
            print("📝 這是一個新的對話")
    except Exception as e:
        print(f"⚠️ 無法載入對話歷史: {e}")

# 主函數
def main():
    try:
        # 獲取 PostgreSQL 連接字串
        postgres_connection_string = get_postgres_connection_string()
        print("🔌 正在連接到 PostgreSQL 資料庫...")

        # 初始化資料庫表格
        setup_database()

        # 建立 PostgreSQL 檢查點保存器
        with PostgresSaver.from_conn_string(postgres_connection_string) as checkpointer:
            # 編譯工作流程，使用 PostgreSQL 保存器
            app = workflow.compile(checkpointer=checkpointer)

            # 獲取或創建會話 ID
            session_id = get_or_create_session()
            print(f"✅ 資料庫連接成功！")
            print(f"🆔 當前會話 ID: {session_id}")

            # 建立符合RunnableConfig類型的配置
            config: RunnableConfig = {"configurable": {"thread_id": session_id}}

            # 顯示對話歷史
            show_conversation_history(app, config)

            print("💬 開始對話（輸入 'exit' 結束，'history' 查看歷史，'new' 開始新對話）")
            print("-" * 50)

            # 對話循環
            while True:
                try:
                    query = input("\n你: ")

                    if query.strip().lower() in ["exit", "退出", "結束"]:
                        break
                    elif query.strip().lower() in ["history", "歷史"]:
                        show_conversation_history(app, config)
                        continue
                    elif query.strip().lower() in ["new", "新對話"]:
                        # 開始新對話
                        session_id = str(uuid.uuid4())
                        with open("session.txt", "w") as f:
                            f.write(session_id)
                        config = {"configurable": {"thread_id": session_id}}
                        print(f"🆕 已開始新對話，會話 ID: {session_id}")
                        continue

                    if not query.strip():
                        print("請輸入有效的問題！")
                        continue

                    # 創建新的用戶訊息
                    input_message = HumanMessage(content=query)

                    # 調用模型 (使用 PostgreSQL 持久化)
                    output = app.invoke(
                        {"messages": [input_message], "language": "中文"},
                        config
                    )

                    # 顯示回應
                    for message in output["messages"]:
                        if message.type == "ai":
                            print(f"🤖 AI: {message.content}")

                except KeyboardInterrupt:
                    print("\n\n👋 程式被中斷，正在退出...")
                    break
                except Exception as e:
                    print(f"❌ 處理訊息時發生錯誤: {e}")
                    continue

    except Exception as e:
        print(f"❌ 資料庫連接失敗: {e}")
        print("請檢查您的 PostgreSQL 設定和連接資訊。")
        print("確保資料庫已創建且用戶有適當的權限。")
        return

    print("\n👋 程式結束，感謝使用！")

if __name__ == "__main__":
    main()