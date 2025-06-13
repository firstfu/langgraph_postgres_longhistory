import os

# PostgreSQL 資料庫配置
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5433"),
    "database": os.getenv("POSTGRES_DB", "langgraph_db"),
    "username": os.getenv("POSTGRES_USER", "AdminUser20nn"),
    "password": os.getenv("POSTGRES_PASSWORD", "5!QD13BmL3iFKKCk]0c"),
}

def get_postgres_connection_string():
    """獲取 PostgreSQL 連接字串"""
    return f"postgresql://{POSTGRES_CONFIG['username']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"