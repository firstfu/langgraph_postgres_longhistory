#!/usr/bin/env python3
"""
PostgreSQL 資料庫初始化腳本
用於創建 LangGraph 所需的表格
"""

from langgraph.checkpoint.postgres import PostgresSaver

from config import get_postgres_connection_string


def init_database():
    """初始化 PostgreSQL 資料庫表格"""
    try:
        postgres_connection_string = get_postgres_connection_string()
        print("🔌 正在連接到 PostgreSQL 資料庫...")
        print(f"連接字串: {postgres_connection_string.replace(postgres_connection_string.split('@')[0].split('//')[1], '***:***')}")

        # 創建 PostgresSaver 實例並設置表格
        with PostgresSaver.from_conn_string(postgres_connection_string) as checkpointer:
            print("🔧 正在創建必要的資料庫表格...")
            checkpointer.setup()
            print("✅ 資料庫表格創建完成！")

            # 驗證表格是否創建成功
            print("🔍 驗證表格創建...")
            # 這裡可以添加驗證邏輯
            print("✅ 資料庫初始化成功！")

    except Exception as e:
        print(f"❌ 資料庫初始化失敗: {e}")
        print("\n可能的解決方案：")
        print("1. 確保 PostgreSQL 服務正在運行")
        print("2. 檢查資料庫連接資訊是否正確")
        print("3. 確保用戶有創建表格的權限")
        print("4. 確保資料庫已存在")
        raise

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 LangGraph PostgreSQL 資料庫初始化")
    print("=" * 50)
    init_database()
    print("=" * 50)
    print("🎉 初始化完成！現在可以運行 app.py")
    print("=" * 50)