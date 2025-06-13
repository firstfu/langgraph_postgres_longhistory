# LangGraph PostgreSQL 聊天機器人

這是一個使用 LangGraph 和 PostgreSQL 持久化的聊天機器人。

## 安裝依賴

```bash
pip install -r requirements.txt
```

## 資料庫設定

### 1. 安裝 PostgreSQL

確保您的系統已安裝 PostgreSQL。

### 2. 創建資料庫

```sql
CREATE DATABASE langgraph_db;
```

### 3. 初始化資料庫表格

執行初始化腳本來創建 LangGraph 所需的表格：

```bash
python init_db.py
```

### 4. 設定環境變數

您可以通過環境變數來配置資料庫連接：

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=langgraph_db
export POSTGRES_USER=your_username
export POSTGRES_PASSWORD=your_password
```

或者直接修改 `config.py` 文件中的預設值。

## API 金鑰設定

將您的 OpenAI API 金鑰放在 `key.txt` 文件中，或者程式會提示您輸入。

## 執行程式

```bash
python llm_env.py
```

## 功能特點

- ✅ 使用 PostgreSQL 進行對話持久化
- ✅ 自動生成會話 ID（不需要手動輸入）
- ✅ 支援對話歷史記錄
- ✅ 卡通人物風格回應
- ✅ 訊息修剪功能

## 使用方式

1. 執行程式後，系統會自動生成一個會話 ID
2. 輸入您的問題
3. 輸入 "exit" 結束對話
4. 所有對話記錄會自動保存到 PostgreSQL 資料庫中
