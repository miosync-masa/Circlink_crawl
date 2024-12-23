import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from upstash_vector import Index
from datetime import datetime
import psycopg2
import uuid
import json
import re

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Set API key via OpenAI client instance
client = OpenAI(api_key=openai_api_key)

upstash_vector_endpoint = os.getenv("UPSTASH_VECTOR_ENDPOINT")
upstash_vector_token = os.getenv("UPSTASH_VECTOR_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

# Increase chunk size to avoid large metadata errors
Settings.text_splitter = SentenceSplitter(chunk_size=4096)

# Configure UpStash VectorDB
vector_index = Index(url=upstash_vector_endpoint, token=upstash_vector_token)

# Connect to the database
try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
except Exception as e:
    st.error(f"Database connection failed: {e}")

# 現在の日付を作成日時として設定
created_at = datetime.now().strftime("%Y-%m-%d")

# Streamlit page configuration
st.set_page_config(page_title="Wantedly Job Registration", page_icon="📄", layout="wide")
st.title("Wantedly Job Registration with LlamaIndex and UpStash")
st.write("Enter a Wantedly job URL and Job ID to fetch and save job data.")

# Initialize session state
if "job_data" not in st.session_state:
    st.session_state["job_data"] = None
if "tags" not in st.session_state:
    st.session_state["tags"] = None
if "job_id" not in st.session_state:
    st.session_state["job_id"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None

# Input fields
job_url = st.text_input("Enter Wantedly Job URL")
job_id = st.text_input("Enter Job ID")

# ジョブID抽出関数
def extract_job_id_from_url(url):
    match = re.search(r'/(\d+)$', url)
    if match:
        return int(match.group(1))
    raise ValueError("Invalid URL format: Job ID not found")

# Wantedlyジョブ詳細取得関数
def fetch_wantedly_job_details(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch URL: {url}")

    soup = BeautifulSoup(response.text, "html.parser")
    meta_og_title = soup.find("meta", property="og:title")
    main_title = meta_og_title["content"] if meta_og_title else None

    company_name = None
    if main_title and "-" in main_title:
        parts = main_title.split("-")
        possible_company_text = parts[-2].strip()
        if "の採用" in possible_company_text:
            company_name = possible_company_text.split("の採用")[0].strip()
        else:
            company_name = possible_company_text

    sections = soup.find_all("section", class_="ProjectDescription__Section-sc-r2aril-9")
    what_text = why_text = how_text = do_text = None
    for sec in sections:
        title_el = sec.find("h3", class_="ProjectDescription__Title-sc-r2aril-1")
        if not title_el:
            continue
        heading = title_el.get_text(strip=True)
        content_div = sec.find("div", class_="ProjectDescription__DescriptionBase-sc-r2aril-8")
        content = content_div.get_text(strip=True) if content_div else ""
        if heading == "なにをやっているのか":
            what_text = content
        elif heading == "なぜやるのか":
            why_text = content
        elif heading == "どうやっているのか":
            how_text = content
        elif heading == "こんなことやります":
            do_text = content

    members = []
    member_sections = soup.find_all("section", class_="ProjectMemberListLaptop__Section-sc-11asydj-15")
    for ms in member_sections:
        name_tags = ms.find_all("p", class_="ProjectMemberListLaptop__ProjectMemberName-sc-11asydj-13")
        for nt in name_tags:
            member_name = nt.get_text(strip=True)
            members.append(member_name)

    stories = []
    heading_elems = soup.find_all("h3")
    for heading in heading_elems:
        if heading.get_text(strip=True) == "会社の注目のストーリー":
            parent_section = heading.find_parent("section")
            if not parent_section:
                continue
            story_blocks = parent_section.find_all("div", class_="PinnedStoryList__Base-sc-w8czck-5")
            for block in story_blocks:
                title_box = block.find("h3", class_="PinnedStoryList__PostTitleBase-sc-w8czck-10")
                if not title_box:
                    continue
                link_tag = title_box.find("a")
                if not link_tag:
                    continue
                story_title = link_tag.get_text(strip=True)
                story_url = link_tag.get("href", "")
                stories.append({
                    "title": story_title,
                    "url": story_url,
                })

    return {
        "id": extract_job_id_from_url(url),
        "url": url,  # URLを追加
        "main_title": main_title,
        "company_name": company_name,
        "what": what_text,
        "why":  why_text,
        "how":  how_text,
        "do":   do_text,
        "members": members,
        "stories": stories,
    }

# Embedding取得関数
def get_embedding(text, model="text-embedding-3-small"):
    # Call the embeddings endpoint via the client
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# タグ生成関数
def generate_tags(job_data):
    prompt = f"""
    以下の求人情報から関連するタグを生成してください:
    - 主な内容: {job_data['what']}
    - なぜやるのか: {job_data['why']}
    - どうやっているのか: {job_data['how']}
    - こんなことやります: {job_data['do']}

    出力例: ["Python", "リモート", "データ分析", "フルタイム"]
    タグはJSON形式の配列で出力してください。
    """

    # Using the client to create ChatCompletion
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        temperature=0.7
    )

    result_text = response.choices[0].message.content.strip()

    try:
        tags = json.loads(result_text)
        if isinstance(tags, list):
            return tags
        else:
            return [tags]
    except json.JSONDecodeError:
        return [result_text]

# 一意のLlama ID生成関数
def generate_llama_id():
    return str(uuid.uuid4())

# SQLデータベース保存関数        
def save_to_sql_db(job_data, sql_cursor):
    try:
        # LlamaIndex用の一意IDを生成
        llama_id = generate_llama_id()

        # デバッグ出力: タグの確認
        print(f"Generated tags: {job_data['tags']}")

        # SQLに保存
        sql_cursor.execute("""
            INSERT INTO job_metadata (
                id, llama_id, main_title, company_name, url, what, why, how, "do",
                members, stories, tags, is_active, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, NOW()
            )
        """, (
            job_data["id"],          # ジョブID
            llama_id,                # LlamaIndexの一意ID
            job_data["main_title"],  # メインタイトル
            job_data["company_name"],# 会社名
            job_data["url"],         # URL
            job_data["what"],        # なにをやっているのか
            job_data["why"],         # なぜやるのか
            job_data["how"],         # どうやっているのか
            job_data["do"],          # こんなことやります
            json.dumps(job_data["members"]),  # メンバー (JSON形式)
            json.dumps(job_data["stories"]),  # ストーリー (JSON形式)
            json.dumps(job_data["tags"]),     # タグ (JSON形式)
            job_data.get("is_active", True)   # アクティブ状態 (デフォルト: True)
        ))

        # コミットを追加
        sql_cursor.connection.commit()
        print("SQL DB save completed.")

        return llama_id
    except Exception as e:
        print(f"Error saving to SQL DB: {e}")
        raise
        
# UpstashにVectorデータとMetaデータを保存   
def save_to_llama_index_with_embedding(job_data, index, sql_cursor):
    try:
        # SQLDBに保存し、一意のLlama IDを取得
        llama_id = save_to_sql_db(job_data, sql_cursor)
        print(f"Llama ID generated: {llama_id}")

        # 結合されたテキストを生成
        combined_text = f"""
        {job_data['main_title']}
        {job_data['company_name']}
        なにをやっているのか: {job_data['what']}
        なぜやるのか: {job_data['why']}
        どうやっているのか: {job_data['how']}
        こんなことやります: {job_data['do']}
        """
        print("Combined text for embedding:", combined_text)

        # 埋め込み生成
        embedding = get_embedding(combined_text, model="text-embedding-3-small")
        print("Generated embedding:", embedding)

        # タプル形式のデータ
        file_id = str(uuid.uuid4())  # UUID4で一意のIDを生成
        metadata = {
            "llama_id": llama_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        print("Metadata for Upstash:", metadata)

        # Upstash Vectorに保存
        index.upsert(vectors=[(file_id, embedding, metadata)])
        print("Upstash save completed.")

        # 結果を辞書形式で返す
        return {
            "llama_id": llama_id,
            "file_id": file_id,
            "status": "success",
            "message": "Data successfully saved to SQL and Upstash"
        }

    except Exception as e:
        print(f"Error saving to Upstash: {e}")
        raise

# --- Streamlit UI Flow ---
if st.button("Fetch Job Data"):
    # 1. 入力検証
    if not job_url:
        st.error("Please enter a URL.")
    elif not job_id:
        st.error("Please enter a valid Job ID.")
    else:
        try:
            # 2. Wantedlyからデータ取得
            job_data = fetch_wantedly_job_details(job_url)
            
            # 3. LLMを使ったタグ生成
            tags = generate_tags(job_data)
            job_data['tags'] = tags  # Ensure tags are assigned to job_data

            # デバッグ出力: タグの確認
            print(f"Generated tags: {tags}")

            # 4. セッションにデータを格納（一時保存）
            st.session_state["job_data"] = job_data
            st.session_state["tags"] = tags
            st.session_state["job_id"] = job_id

            # 5. プレビュー（HITLプロセス）
            st.subheader("Fetched Job Data (Preview)")
            st.write("### Main Title:", job_data["main_title"])
            st.write("### Company Name:", job_data["company_name"])
            st.write("### What:", job_data["what"])
            st.write("### Why:", job_data["why"])
            st.write("### How:", job_data["how"])
            st.write("### Do:", job_data["do"])
            st.write("### Members:", job_data["members"])
            st.write("### Stories:", job_data["stories"])
            st.write("### Tags (Generated):", tags)

            st.info("Check the preview above. Then click 'Approve and Save' if it looks good.")
        except Exception as e:
            # 例外処理
            st.error(f"Error fetching job data: {e}")

# 6. 保存（HITLの承認後に実行）
if st.session_state["job_data"] and st.session_state["job_id"] and st.button("Approve and Save"):
    try:
        # LlamaIndexとUpstash Vectorへの保存
        result = save_to_llama_index_with_embedding(st.session_state["job_data"], vector_index, cursor)
        st.success(
            f"Job '{st.session_state['job_data']['main_title']}' "
            f"with ID '{st.session_state['job_id']}' "
            "saved successfully to UpStash and SQL!"
        )
        st.write("Save result:", result)
    except Exception as e:
        # 例外処理
        st.error(f"Error saving job data: {e}")
