import os
import time
import requests
import xml.etree.ElementTree as ET
import trafilatura
import cohere

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# -------------------------------------
# LOAD ENV
# -------------------------------------
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# -------------------------------------
# CONFIG
# -------------------------------------
SITEMAP_URL = "https://physical-ai-textbook-six.vercel.app/sitemap.xml"
COLLECTION_NAME = "physical_ai_book"

# Cohere
EMBED_MODEL = "embed-english-v3.0"
cohere_client = cohere.Client(COHERE_API_KEY)

# Qdrant
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# -------------------------------------
# STEP 1 — Extract URLs from sitemap
# -------------------------------------
def get_all_urls(sitemap_url):
    """Extract all URLs from sitemap.xml"""
    try:
        xml = requests.get(sitemap_url, timeout=10).text
        root = ET.fromstring(xml)
        urls = []

        for child in root:
            loc = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
            if loc is not None:
                urls.append(loc.text)

        print("\n✅ URLs Found:")
        for u in urls:
            print(" -", u)

        return urls

    except Exception as e:
        print(f"❌ Sitemap error: {e}")
        return []

# -------------------------------------
# STEP 2 — Extract page text
# -------------------------------------
def extract_text_from_url(url):
    """Extract clean text using trafilatura"""
    try:
        html = requests.get(url, timeout=10).text
        text = trafilatura.extract(html)

        if not text:
            print(f"⚠️ No content extracted: {url}")

        return text

    except Exception as e:
        print(f"❌ Extraction error ({url}): {e}")
        return None

# -------------------------------------
# STEP 3 — Chunk text
# -------------------------------------
def chunk_text(text, max_chars=1000):
    """Split text into semantic chunks"""
    chunks = []

    while len(text) > max_chars:
        split_pos = text[:max_chars].rfind(". ")
        split_pos = split_pos + 1 if split_pos != -1 else max_chars

        chunks.append(text[:split_pos].strip())
        text = text[split_pos:].strip()

    if text:
        chunks.append(text)

    return chunks

# -------------------------------------
# STEP 4 — Create embeddings (Cohere)
# -------------------------------------
def embed_text(text):
    response = cohere_client.embed(
        model=EMBED_MODEL,
        input_type="search_document",
        texts=[text]
    )
    return response.embeddings[0]

# -------------------------------------
# STEP 5 — Create Qdrant collection
# -------------------------------------
def create_collection():
    print("\n🔧 Creating Qdrant collection...")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,  # Cohere embedding dimension
            distance=Distance.COSINE
        )
    )
    print("✅ Collection ready")

# -------------------------------------
# STEP 6 — Save chunk to Qdrant
# -------------------------------------
def save_chunk(chunk, chunk_id, url):
    vector = embed_text(chunk)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "url": url,
                    "text": chunk,
                    "chunk_id": chunk_id
                }
            )
        ]
    )

# -------------------------------------
# MAIN INGESTION PIPELINE
# -------------------------------------
def ingest_book():
    print("\n🚀 Starting RAG ingestion pipeline")

    urls = get_all_urls(SITEMAP_URL)
    if not urls:
        print("❌ No URLs found. Exiting.")
        return

    create_collection()

    global_id = 1
    total_chunks = 0

    for idx, url in enumerate(urls, 1):
        print(f"\n📄 ({idx}/{len(urls)}) Processing: {url}")

        text = extract_text_from_url(url)
        if not text:
            continue

        chunks = chunk_text(text)
        print(f"   🧩 {len(chunks)} chunks created")

        for chunk in chunks:
            save_chunk(chunk, global_id, url)
            print(f"   ✅ Stored chunk {global_id}")

            global_id += 1
            total_chunks += 1

            # Cohere rate limit safety
            time.sleep(1)

    print("\n" + "=" * 50)
    print("✅ INGESTION COMPLETE")
    print(f"📊 Total chunks stored: {total_chunks}")
    print(f"🔗 Pages processed: {len(urls)}")
    print("=" * 50)

# -------------------------------------
# COLLECTION INFO
# -------------------------------------
def check_collection():
    info = qdrant.get_collection(COLLECTION_NAME)
    print("\n📊 Collection Info")
    print(f"   Name: {COLLECTION_NAME}")
    print(f"   Points: {info.points_count}")
    print(f"   Vector size: {info.config.params.vectors.size}")

# -------------------------------------
# ENTRY POINT
# -------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_collection()
    else:
        ingest_book()
