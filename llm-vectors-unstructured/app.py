import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from query_customer_neo4j import customer_chain

# Ortam değişkenlerini yükle
load_dotenv()

# LLM başlat (API key .env dosyasından alınır)
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")  # ✅ Burada API anahtarı güvenli şekilde okunuyor
)

# Neo4j bağlantısı
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# GraphQA zinciri
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

# Streamlit Arayüz
st.set_page_config(page_title="KVKK Chatbot", page_icon="🧠")
st.title("🧠 LEGRA Knowledge Graph Chatbot")

st.markdown("Kullanıcılar grafik veritabanından veya metin tabanlı sistemden sorgu yapabilir.")

# Sorgu tipi seçimi
query_type = st.radio(
    "Sorgu yapmak istediğiniz alanı seçin:",
    ["Neo4j Graph Tabanlı Sorgu", "Hukuki Metin Tabanlı Sorgu", "Müşteri Bilgisi Tabanlı Sorgu"]
)

# Soru girişi
if query_type == "Neo4j Graph Tabanlı Sorgu" or query_type == "Müşteri Bilgisi Tabanlı Sorgu":
    st.caption("📌 Örnek: 'X A.Ş. sözleşme bitiş tarihi nedir?' veya 'Y firması hangi ürünü kullanıyor?' gibi sorular sorabilirsiniz.")
    user_question = st.text_input("❓ Soru:")

    if user_question:
        if len(user_question.strip().split()) < 3:
            st.warning("Lütfen daha açıklayıcı bir soru yazın. Örneğin: 'X firmasının sözleşme süresi nedir?' gibi.")
        else:
            with st.spinner("Yanıt getiriliyor..."):
                try:
                    if query_type == "Neo4j Graph Tabanlı Sorgu":
                        response = graph_chain.run(user_question)
                    elif query_type == "Müşteri Bilgisi Tabanlı Sorgu":
                        response = customer_chain.run(user_question)
                    st.success("✅ Cevap:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Hata oluştu: {str(e)}")

# Eğer Hukuki Metin Tabanlı ise veri seçimi ve metin önerisi
elif query_type == "Hukuki Metin Tabanlı Sorgu":
    st.info("Aşağıda talep ettiğiniz metinde işleneceği beyan edilecek olan verileri seçerek metin tipi tercihinizi yapınız.")

    st.header("❌ 5 İşlenen Veri Seçerek Metin Önerisi Al")
    st.caption("🔐 Lütfen en fazla 5 veri seçin:")

    veri_secenekleri = ["Ad Soyad", "E-posta", "Telefon", "Adres", "TCKN", "Banka Bilgisi", "IP Adresi", "Konum Bilgisi"]
    secili_veriler = st.multiselect("", veri_secenekleri, max_selections=5)

    metin_tipi = st.selectbox("Metin Tipini Seçin", ["Açık Rıza Metni", "Aydınlatma Metni", "KVKK Onay Metni", "KVKK Muvafakatname"])

    if st.button("📄 Önerilen Metin Göster"):
        if not secili_veriler:
            st.warning("Lütfen en az bir veri seçiniz.")
        else:
            try:
                # metin_tipi'ni dosya anahtarına çevir
                index_map = {
                    "Açık Rıza Metni": "acik_riza_metni",
                    "Aydınlatma Metni": "aydinlatma_metni",
                    "KVKK Onay Metni": "kvkk_onay_metni",
                    "KVKK Muvafakatname": "kvkk_muvafakatname"
                }
                index_key = index_map.get(metin_tipi)

                vector_store = FAISS.load_local(
                    f"legal_vector_index/{index_key}_vector_index",
                    OpenAIEmbeddings(),
                    allow_dangerous_deserialization=True
                )
                vector_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

                example_question = f"{', '.join(secili_veriler)} verileri işlenecek şekilde bir {metin_tipi.lower()} önerisi ver."
                response = vector_chain.run(example_question)

                st.success("✅ Önerilen Metin:")
                st.write(response)

            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")
