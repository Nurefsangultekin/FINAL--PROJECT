from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# FAISS veritabanını yükle
vector_db = FAISS.load_local("legal_vector_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# LLM modelini oluştur (gpt-3.5 veya gpt-4 API key'ine göre çalışır)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Soru-Cevap zinciri oluştur (retriever, FAISS içinden belge çeker)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# Kullanıcıdan doğal dilde soru al ve cevabı üret
if __name__ == "__main__":
    query = input("📨 Sorunuzu yazın: ")
    result = qa_chain(query)
    
    print("\n🤖 Chatbot Yanıtı:\n")
    print(result['result'])

    print("\n📄 Kullanılan Belgeler:")
    for doc in result["source_documents"]:
        print("-" * 40)
        print(doc.page_content[:500])  # İlk 500 karakteri yaz
