# RAG system made by 185661
import re
import json
import faiss
import os
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from flask import Flask, request, Response

# 禁止連網
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

PDF_PATH = "data.pdf"
FAISS_INDEX_PATH = "faiss_index.faiss"

# ========== 1️⃣ 讀取 PDF ==========
def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ========== 2️⃣ 清理文本 ==========
def clean_text_func(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"－|—|-", "—", text)
    return text.strip()

# ========== 3️⃣ 切分章節與條文 ==========
def split_law_text(clean_text, max_len=None):
    chunks = []

    # 偵測 header（只抓法規名稱與修正日期，不包含章節）
    header_pattern = r"(法規名稱[:：].+?修正日期[:：].+?)(?=\s*第\s*[\u4e00-\u9fa5\d]+\s*章)"
    header_match = re.search(header_pattern, clean_text, re.DOTALL)
    if header_match:
        header_text = header_match.group(1).strip()
        chunks.append({
            "chapter": "",
            "article": "",
            "text": header_text
        })
        # 移除 header，只留下章節開始部分
        clean_text = clean_text.replace(header_text, "").strip()

    # 預清理剩餘正文
    clean_text = re.sub(r'([，。；：、])\s*\n\s*', r'\1', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text.strip())

    # 按章節切分
    chapter_pattern = r"(第\s*[\u4e00-\u9fa5\d]+\s*章\s*[^第]*)"
    chapters = re.split(chapter_pattern, clean_text)

    current_chapter = ""

    for part in chapters:
        if re.match(chapter_pattern, part):
            current_chapter = part.strip()
        else:
            content = part.strip()
            if not content:
                continue

            # 按條文切分
            article_pattern = r"(第\s*\d+\s*條)"
            articles = re.split(article_pattern, content)
            current_article = ""

            for article_part in articles:
                if re.match(article_pattern, article_part):
                    current_article = article_part.strip()
                else:
                    text = article_part.strip()
                    if not text:
                        continue

                    # 阿拉伯數字項拆分
                    arabic_split = re.split(r'(?:^|\s)(\d+)\s', text)
                    sub_items = []

                    if len(arabic_split) == 1:
                        sub_items = [arabic_split[0]]
                    else:
                        for i in range(1, len(arabic_split), 2):
                            num = arabic_split[i].strip()
                            content_part = arabic_split[i+1].strip() if i+1 < len(arabic_split) else ""
                            sub_items.append(f"{num} {content_part}")

                    for sub_text in sub_items:
                        sub_text = sub_text.strip()
                        if not sub_text:
                            continue

                        # 超長文本分段（可選）
                        if max_len and len(sub_text) > max_len:
                            for k in range(0, len(sub_text), max_len):
                                chunk_text = sub_text[k:k+max_len]
                                chunks.append({
                                    "chapter": current_chapter,
                                    "article": current_article,
                                    "text": chunk_text
                                })
                        else:
                            chunks.append({
                                "chapter": current_chapter,
                                "article": current_article,
                                "text": sub_text
                            })
    return chunks

# ========== 4️⃣ 建立向量索引 ==========
def build_faiss_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity (normalize embeddings)
    index.add(embeddings)

    faiss.write_index(index, "faiss_index.faiss")
    with open("chunks_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ 已建立索引，共 {len(chunks)} 個 chunks。")

# ========== 5️⃣ 問答 ==========
def ask_llm(question, llm, embedder, search_top_k=10, final_top_k=3, distance_threshold=0.6):
    # 先檢查輸入
    if not isinstance(question, str):
        question = str(question)  # 轉字串
    question = question.strip()
    if not question:
        return "⚠️ 無法回覆該問題 (空問題)", []

    # 載入 FAISS 索引與 metadata
    index = faiss.read_index("faiss_index.faiss")
    with open("chunks_metadata.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 生成問題向量
    q_emb = embedder.encode([question], normalize_embeddings=True)

    # 搜尋相似 chunks
    D, I = index.search(q_emb, search_top_k)  # 結果已依相似度由高至低排序
    top_chunks = [chunks[i]["text"] for i in I[0] if i != -1]
    top_sims = [s for s, i in zip(D[0], I[0]) if i != -1]

    # 相似度過低時回傳提示
    if len(top_sims) == 0 or top_sims[0] < distance_threshold:
        return "⚠️ 無法回覆該問題", []

    # 組合 prompt
    context = "\n\n".join(top_chunks[:final_top_k])
    prompt = (
        "### Instruction:\n"
        "您是一位法律專家，請僅根據下列條文，以正體中文zh-TW回答問題：\n\n"
        f"{context}\n\n"
        f"### 問題：{question}\n"
        "### 回答：（僅回答，不要生成其他問題）"
    )

    response = llm(prompt=prompt, max_tokens=256, temperature=0.3, top_p=0.8)
    answer_text = response["choices"][0]["text"].strip()

    return answer_text, top_chunks[:final_top_k]

# ====================== 6️⃣ CLI & API ======================
def setup():
    embedder = SentenceTransformer("model/bge-small-zh-v1.5")
    llm = Llama(model_path="model/mradermacher_chatglm3-6b-GGUF_chatglm3-6b.Q4_K_M.gguf")

    if not os.path.exists("faiss_index.faiss"):
        raw_text = read_pdf("data.pdf")
        clean_text = clean_text_func(raw_text)
        chunks = split_law_text(clean_text)
        build_faiss_index(chunks, embedder)
    else:
        print("📘 索引已存在，直接進入問答模式。")

    return embedder, llm

def main():
    embedder, llm = setup()

    try:
        while True:
            mode = input("請選擇模式：\n[(c)li/(a)pi/(q)uit]=> ").strip().lower()
            if mode in ["c", "a", "q"]:
                break
            print("⚠️ 請輸入 'c'、'a' 或 'q'。")

        if mode == "q":
            print("退出程式......")
            return

        if mode == "c":
            # CLI 提問
            while True:
                q = input("\n❓ 請輸入性騷擾防治法問題 (離開：q)=> ")
                if q.lower() == "q":
                    break
                result = ask_llm(q, llm, embedder)
                if result is not None:
                    answer, refs = result
                    print("\n🧠 回答=> ", answer)
                    print("\n📚 參考內容：")
                    for t in refs:
                        print("-", t[:80]+"...")
        else:
            # API 模式
            app = Flask(__name__)

            @app.route("/ask", methods=["POST"])
            def ask_api():
                data = request.json
                q = data.get("question", "")
                result = ask_llm(q, llm, embedder)
                if result is not None:
                    answer, refs = result

                    # 使用 json.dumps 並 ensure_ascii=False
                    response_body = json.dumps({
                        "answer": answer,
                        "references": refs
                    }, ensure_ascii=False)

                    return Response(response_body, content_type="application/json; charset=utf-8")

                return Response(
                    json.dumps({"error": "⚠️ 無法回覆該問題"}, ensure_ascii=False),
                    status=400,
                    content_type="application/json; charset=utf-8"
                )

            print("🚀 Flask API 啟動，請訪問 http://127.0.0.1:5000/ask")
            app.run(host="0.0.0.0", port=5000)

    finally:
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
            llm = None

if __name__ == "__main__":
    main()
