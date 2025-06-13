from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Vektör veritabanını yükle
vector_db = FAISS.load_local("legal_vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def search_documents(query, k=3):
    print(f"🔍 Soru: {query}")
    results = vector_db.similarity_search(query, k=k)
    print("\n📄 En alakalı belgeler:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Belge {i} ---")
        print(doc.page_content[:1000])  # Çok uzun olmasın diye ilk 1000 karakteri yazıyoruz

# Deneme
if __name__ == "__main__":
    query = input("Sormak istediğiniz soruyu yazın: ")
    search_documents(query)
