import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
import tiktoken

def count_tokens(text, model_name="text-embedding-3-small"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

def create_embeddings_from_csv(file_path, id_col, text_col):
    print(f"🔍 İşleniyor: {file_path}")
    try:
        df = pd.read_csv(file_path)
        if text_col not in df.columns:
            print(f"⚠️ '{text_col}' sütunu {file_path} içinde bulunamadı, atlanıyor.")
            return []
        docs = [
            Document(page_content=row[text_col], metadata={"id": row[id_col]})
            for _, row in df.iterrows()
        ]
        return docs
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        return []

# CSV dosyaları ve ilgili sütun adları
csv_files = {
    "acik_riza_metni": ("acik_riza_metni_taslaklari_idli.csv", "ID", "Açık Rıza Metni Taslağı"),
    "aydinlatma_metni": ("aydinlatma_metni_taslaklari_idli_v2.csv", "ID", "Aydınlatma Metni Taslağı"),
    "kvkk_onay_metni": ("kvkk_onay_muvafakatname_taslaklari_idli.csv", "ID", "KVKK Onay Metni Taslağı"),
    "kvkk_muvafakatname": ("kvkk_onay_muvafakatname_taslaklari_idli.csv", "ID", "KVKK Muvafakatname Taslağı")
}

embedding_model = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
os.makedirs("legal_vector_index", exist_ok=True)

for file_key, (file_path, id_col, text_col) in csv_files.items():
    all_docs = create_embeddings_from_csv(file_path, id_col, text_col)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"📄 {file_key}: Toplam belge sayısı (parçalanmış): {len(split_docs)}")

    # Batch işle
    batch = []
    total_tokens = 0
    max_tokens = 280000
    vector_db = None

    for doc in split_docs:
        tokens = count_tokens(doc.page_content)
        if total_tokens + tokens > max_tokens:
            print(f"➡️ Yeni batch başlatılıyor (toplam token: {total_tokens})")
            if vector_db is None:
                vector_db = FAISS.from_documents(batch, embedding_model)
            else:
                vector_db.add_documents(batch)
            batch = []
            total_tokens = 0
        batch.append(doc)
        total_tokens += tokens

    # Son batch’i kaydet
    if batch:
        print(f"✅ Son batch işleniyor (toplam token: {total_tokens})")
        if vector_db is None:
            vector_db = FAISS.from_documents(batch, embedding_model)
        else:
            vector_db.add_documents(batch)

    # Kayıt
    index_path = f"legal_vector_index/{file_key}_vector_index"
    vector_db.save_local(index_path)
    print(f"✅ {file_key} için embedding başarıyla kaydedildi: {index_path}")
