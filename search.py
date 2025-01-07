import os
import streamlit as st
import requests
from dotenv import load_dotenv
from openai import OpenAI
from llama_parse import LlamaParse 
import psycopg2
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
import json
import logging
from upstash_vector import Index

##############################################################################
# 環境変数の読み込み
##############################################################################
load_dotenv()

LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

DATABASE_URL = os.getenv("DATABASE_URL")
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

UPSTASH_VECTOR_ENDPOINT = os.getenv("UPSTASH_VECTOR_ENDPOINT")
UPSTASH_VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")
UPSTASH_VECTOR_DIMENSION = int(os.getenv("UPSTASH_VECTOR_DIMENSION", "1536"))
UPSTASH_SIMILARITY_FUNCTION = os.getenv("UPSTASH_SIMILARITY_FUNCTION", "COSINE")

# API Key 設定
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
if not LLAMA_PARSE_API_KEY:
    raise ValueError("LLAMA_PARSE_API_KEY is not set. Please check your .env file.")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Set API key via OpenAI client instance
client = OpenAI(api_key=openai_api_key)

# LlamaParseの設定
llama_parser = LlamaParse(
    api_key=LLAMA_PARSE_API_KEY,  # 修正: LLAMA_PARSE_API_KEY を渡す
    result_type="markdown",
    verbose=True
)

# サポートされるファイル形式
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".doc", ".docx", ".xlsx", ".csv", ".json"]

##############################################################################
# STEP①: フリーワード検索条件の入力 + PDFアップロード処理
##############################################################################
def input_search_conditions_and_upload():
    """
    フリーワード検索条件を入力し、PDFファイルをアップロードする処理。
    LlamaParseで解析し、取得したテキストを返す。
    """
    # フリーワード入力
    search_word = st.text_input("検索条件のキーワードを入力してください:", "")

    # PDFファイルのアップロード
    uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type=SUPPORTED_FILE_TYPES)

    # ファイルがアップロードされた場合の処理
    if uploaded_file:
        try:
            # ファイル名と拡張子の取得
            file_name = uploaded_file.name
            file_extension = os.path.splitext(file_name)[1].lower()

            # ファイル形式のバリデーション
            if file_extension not in SUPPORTED_FILE_TYPES:
                raise ValueError(f"Unsupported file type: {file_extension}. Supported types are: {', '.join(SUPPORTED_FILE_TYPES)}")

            # ファイル内容の読み込み
            file_contents = uploaded_file.read()
            if not file_contents:
                raise ValueError("The uploaded file is empty. Please upload a valid file.")

            # アップロード成功メッセージ
            st.success(f"File '{file_name}' uploaded successfully!")

            # LlamaParseで解析
            extra_info = {"file_name": file_name}
            documents = llama_parser.load_data(file_contents, extra_info=extra_info)

            if not documents or len(documents) == 0:
                raise ValueError("No documents were returned by LlamaParse. Ensure the file is valid.")

            # 解析結果からテキストを取得
            parsed_text = documents[0].text  # 最初のドキュメントのテキストを取得
            if not parsed_text.strip():
                raise ValueError("The parsed document is empty. Verify the file content.")

            # 解析結果をプレビュー
            st.write("Parsed Text Preview:", parsed_text[:2500])
            return search_word, parsed_text

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            return search_word, ""

    # ファイルがアップロードされていない場合
    return search_word, ""
    
##############################################################################
# DB接続の定義
##############################################################################
def get_postgres_connection():
    """
    PostgreSQLへの接続を返す関数。
    """
    try:
        engine = create_engine(DATABASE_URL)
        conn = engine.connect()
        return conn
    except Exception as e:
        st.error(f"Postgres接続エラー: {e}")
        return None
        
##############################################################################
# STEP②: 条件フリーワード＋LLAMA_PARSEの内容を用いてGPT-4でタグ生成
##############################################################################
def generate_tags_with_gpt4(free_text, pdf_text):
    """
    GPT-4で要約とタグを生成する関数。
    """
    if not free_text and not pdf_text:
        return {"error": "フリーワードまたはPDFの内容が空です"}

    # プロンプト作成
    prompt = f"""
    あなたは転職活動をサポートする優秀なAIエージェントです。

    以下の情報を基に、転職活動に活用できるタグと要約を生成してください。

    ### 要約
    職務経歴書の情報を基に、候補者のスキル、経験、実績を簡潔にまとめた文章を作成してください。
    - 情報を凝縮し、職務経歴全体の概要を簡潔にまとめること。
    - 候補者の実績やユニークなスキルを強調すること。
    - 転職活動において直接活用できる要素（業務経験、成果、資格など）を明示すること。
    - 読み手が理解しやすい言葉で表現すること。

    ### タグ
    スキル、職種、業界、経験分野、強み、資格、価値観、仕事への取り組み姿勢、希望条件をカテゴリ別にJSON形式の配列としてリストアップしてください。

    タグ生成時の留意点:
    1. タグは具体的で実用的にすること。
    2. 各カテゴリにおいて重複を避け、独立したタグを作成すること。
    3. PDFの情報から、暗黙的な価値観や取り組み姿勢も読み取ること。
    4. 必要な個数を生成し、制限を設けないこと。

    ### 入力情報
    - フリーワード: {free_text}
    - PDF内容: {pdf_text}

    #### 出力形式
    ```json
    {{
        "summary": "ここに要約文を生成",
        "tags": {{
            "skills": ["タグ1", "タグ2"],
            "roles": ["タグ1", "タグ2"],
            "industries": ["タグ1", "タグ2"],
            "experience": ["タグ1", "タグ2"],
            "strengths": ["タグ1", "タグ2"],
            "certifications": ["タグ1", "タグ2"],
            "values": ["タグ1", "タグ2"],
            "work_attitude": ["タグ1", "タグ2"],
            "wish_list": ["タグ1", "タグ2"]
        }}
    }}
    ```
    """

    try:
        # GPT-4o API の呼び出し
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=12000,
            temperature=0.75
        )
        # 結果を取得
        result_text = completion.choices[0].message.content.strip()

        # JSON解析
        try:
            tags = json.loads(result_text)
            if isinstance(tags, list):
                return tags
            else:
                # JSON解析成功だがリストでない場合
                return [tags]
        except json.JSONDecodeError:
            # JSON解析失敗時、テキストをそのままリストで返す
            return [result_text]

    except Exception as e:
        # API呼び出しエラー時
        st.error(f"GPT-4o API呼び出しエラー: {e}")
        return []

##############################################################################
# STEP③: 生成されたタグをEmbedding実施
##############################################################################
def get_embedding(text):
    """
    OpenAI Embedding API (text-embedding-3-small) でテキストをベクトル化するでござる。
    """
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"  # 必要に応じて変更
        )
        vector = response.data[0].embedding
        return vector
    except Exception as e:
        st.error(f"Embedding取得エラー: {e}")
        return []

def get_tags_embedding(tags):
    """
    タグをカンマで結合して1つの文章にまとめ、エンベディングを取得。
    """
    if not tags:
        return []
    # タグをカンマで結合
    combined_text = ", ".join(tags)
    return get_embedding(combined_text)

##############################################################################
# STEP④: VectorDB(Upstash)でセマンティックサーチ
##############################################################################
def upstash_vector_search(query_vector, top_k=3):
    """
    Upstash Vectorのライブラリを使って検索を行い、上位3件のみを返す関数。
    """
    # インデックスの初期化
    index = Index(url=UPSTASH_VECTOR_ENDPOINT, token=UPSTASH_VECTOR_TOKEN)

    try:
        # 検索クエリの実行
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_vectors=False
        )

        # 上位3件を抽出してフォーマット
        matches = []
        for vec in result[:top_k]:  # 上位3件のみをスライス
            matches.append({
                "id": vec.id,
                "score": vec.score,
                "metadata": vec.metadata,
            })
        return matches

    except Exception as e:
        st.error(f"Upstashベクター検索中にエラー: {e}")
        return []

##############################################################################
# STEP⑤: 結果の上位３つ（A,B,C）を取得し、JOB_IDをキーにDBから詳細情報を取得
##############################################################################
def fetch_job_details_by_llama_id(llama_ids):
    """
    llama_ids を用いて PostgreSQL から該当する求人情報を取得する関数
    """
    conn = get_postgres_connection()
    if conn is None:
        return []

    # UUID型に変換
    try:
        uuid_llama_ids = [uuid.UUID(llama_id) for llama_id in llama_ids]
    except ValueError as e:
        st.error(f"Invalid UUID in llama_ids: {e}")
        return []

    results = []
    try:
        sql = text("""
            SELECT * FROM job_metadata
            WHERE llama_id = ANY(:llama_ids)
        """)
        rows = conn.execute(sql, {"llama_ids": uuid_llama_ids}).fetchall()

        # 辞書形式に変換
        results = [dict(row._mapping) for row in rows]  # _mapping を使う
    except Exception as e:
        st.error(f"PostgreSQL取得エラー: {e}")
    finally:
        conn.close()

    return results

##############################################################################
# STEP⑥: 取得した求人情報(A,B,C)と条件フリーワード＋PDF内容を比較し推奨利用を生成
##############################################################################
def generate_recommendations(free_text, pdf_text, job_details_list):
    """
    求人情報と求職者情報+フリーワードから、最終的な推奨文面を生成する関数
    """
    recommendations = []
    for job_detail in job_details_list:
        # 各求人情報に基づいてプロンプトを生成
        content = f"""
        あなたは求人マッチングの専門家です。
        以下の情報を基に、求職者に最適な求人である理由や、どのように役立つかを推奨文としてまとめてください。必要に応じて、マッチ度を簡潔に評価してください。

        求職者の詳細情報:
        求職者の履歴とスキルの概要: {pdf_text[:200]}

        検索条件:
        {free_text}

        求人情報:
        - タイトル: {job_detail.get('main_title', 'タイトル情報なし')}
        - 会社名: {job_detail.get('company_name', '会社名情報なし')}
        - 仕事内容: {job_detail.get('what', '仕事内容情報なし')}
        - なぜ: {job_detail.get('why', '理由情報なし')}
        - どのように: {job_detail.get('how', 'プロセス情報なし')}
        - こんなことやります: {job_detail.get('do', 'こんなことやりますなし')}

        提案を具体的に述べてください。
        """
        try:
            # 推奨文をGPT-4で生成
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=0.9
            )
            recommendation_text = completion.choices[0].message.content.strip()

            # 推奨文をリストに追加
            recommendations.append({
                "job_id": job_detail.get("id", ""),
                "recommendation": recommendation_text
            })
        except Exception as e:
            # エラー時のログと例外処理
            st.error(f"推奨文面生成エラー (Job ID: {job_detail.get('id', '不明')}): {e}")
    return recommendations

###############################################################################
# メインのStreamlitアプリ
##############################################################################
def main():
    st.title("Circlink_jobSearch")

    # フリーワード入力とPDF解析を同時に取得
    free_text, pdf_text = input_search_conditions_and_upload()

    # 「検索実行」ボタン
    if st.button("検索実行"):
        # STEP②: GPT-4でタグ生成
        st.write("タグを生成中...")
        tags = generate_tags_with_gpt4(free_text, pdf_text)
        st.write(f"生成されたタグ: {tags}")

        # STEP③: Embedding 実施
        st.write("タグのEmbeddingを取得中...")
        tags_vector = get_tags_embedding(tags)

        # STEP④: Upstash VectorDBでセマンティックサーチ
        st.write("ベクトル検索を実行中...")
        search_results = upstash_vector_search(tags_vector, top_k=3)
        st.write("検索結果:", search_results)

        if search_results:
            # 上位3つ（A,B,C）から llama_id を取得
            llama_ids = [match["metadata"].get("llama_id") for match in search_results if "metadata" in match]
            st.write(f"上位3件のLLAMA_ID: {llama_ids}")

            # STEP⑤: LLAMA_IDをキーにDB詳細を取得
            st.write("DBから詳細情報を取得中...")
            job_details_list = fetch_job_details_by_llama_id(llama_ids)
            st.write("取得した求人詳細:", job_details_list)

            # STEP⑥: 推奨利用を生成
            st.write("求職者情報と求人情報の比較・推奨文面を生成中...")
            recommendations = generate_recommendations(free_text, pdf_text, job_details_list)

            st.write("最終推奨結果:")
            for rec in recommendations:
                st.subheader(f"LLAMA_ID: {rec['job_id']}")
                st.write(rec["recommendation"])

if __name__ == "__main__":
    main()