''' Ref: https://www.youtube.com/watch?v=ntuuLCR8TkA '''
# 安裝嵌入模型：ollama pull mxbai-embed-large
# 安裝聯發科LLM模型：ollama run ycchen/breeze-7b-instruct-v1_0
# 安裝chromadb(向量庫)：pip install chromadb

import streamlit as st  # 導入Streamlit庫，用於建立網頁應用
import ollama  # 導入ollama庫，用於自然語言處理
import chromadb  # 導入chromadb庫，用於數據存儲和查詢
import pandas as pd  # 導入pandas庫，用於數據分析和處理

# 定義一個初始化函數，用於設置Streamlit的會話狀態
def initialize():
    # 檢查'session_state'（會話狀態）中是否已有'already_executed'這個變量
    # 這個變量用來檢查是否已經進行過一次資料庫初始化操作
    if "already_executed" not in st.session_state:
        st.session_state.already_executed = False  # 若不存在，則設置為False

    # 如果'already_executed'為False，表示還未初始化過資料庫
    if not st.session_state.already_executed:
        setup_database()  # 呼叫setup_database函數來進行資料庫的設置和數據加載

# 定義設置資料庫的函數
def setup_database():
    client = chromadb.Client()  # 創建一個chromadb的客戶端，用於與資料庫交互
    file_path = 'QA50.xlsx'  # 指定Excel文件的路徑和名稱
    documents = pd.read_excel(file_path, header=None)  # 使用pandas讀取Excel文件

    # 使用chromadb客戶端創建或獲取名為'demodocs'的集合
    collection = client.get_or_create_collection(name="demodocs")

    # 遍歷從Excel文件中讀取的數據，每一行代表一條記錄
    for index, content in documents.iterrows():
        response = ollama.embeddings(model="mxbai-embed-large", prompt=content[0])  # 通過ollama生成該行文本的嵌入向量
        collection.add(ids=[str(index)], embeddings=[response["embedding"]], documents=[content[0]])  # 將文本和其嵌入向量添加到集合中

    st.session_state.already_executed = True  # 設置'already_executed'為True，表示已完成初始化
    st.session_state.collection = collection  # 將集合保存在會話狀態中，供後續使用    


# 定義創建新chromadb客戶端的函數，每次需要時創建新的連接
def create_chromadb_client():
    return chromadb.Client()  # 返回一個新的chromadb客戶端實例

# 主函數，負責構建UI和處理用戶事件
def main():
    initialize()  # 呼叫初始化函數
    st.title("我的第一個LLM+RAG本地知識問答")  # 在網頁應用中設置標題
    user_input = st.text_area("您想問什麼？", "")  # 創建一個文本區域供用戶輸入問題

    # 如果用戶點擊"送出"按鈕
    if st.button("送出"):
        if user_input:
            handle_user_input(user_input, st.session_state.collection)  # 處理用戶輸入，進行查詢和回答
        else:
            st.warning("請輸入問題！")  # 如果用戶沒有輸入，顯示警告消息

# 定義處理用戶輸入的函數
def handle_user_input(user_input, collection):
    response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")  # 生成用戶輸入的嵌入向量
    results = collection.query(query_embeddings=[response["embedding"]], n_results=3)  # 在集合中查詢最相關的三個文檔
    data = results['documents'][0]  # 獲取最相關的文檔
    output = ollama.generate(
        model="ycchen/breeze-7b-instruct-v1_0",
        prompt=f"Using this data: {data}. Respond to this prompt and use chinese: {user_input}"  # 生成回應
    )
    st.text("回答：")  # 顯示"回答："
    st.write(output['response'])  # 將生成的回應顯示在網頁上

if __name__ == "__main__":
    main()  # 如果直接執行此文件，則執行main函數