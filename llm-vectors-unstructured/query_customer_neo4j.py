from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()

# LLM tanımı
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Neo4j bağlantısı
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Prompt template (Cypher prompt artık bir string değil, PromptTemplate olmalı)
CUSTOMER_PROMPT_TEMPLATE = PromptTemplate.from_template("""
Aşağıdaki graf veri tabanında 'Müşteri' adlı node'lar şu özelliklere sahiptir:
- `Ticari Unvan`
- `Ürün Bilgisi`
- `Sözleşme Başlangıç Tarihi`
- `Sözleşme Bitiş Tarihi`
- `Gizlilik Sözleşmesi`
- `Otomatik Yenileme`
- `SLA Durumu`
- `Hizmet Yenileme Oranı`

Bu alan isimlerinde boşluklar olduğu için sorgularda bu özellikler **backtick (`) ile yazılmalıdır**.
Örnek doğru kullanım:
MATCH (m:Müşteri {{`Ticari Unvan`: 'X A.Ş.'}}) RETURN m.`Ürün Bilgisi`

Kullanıcıdan gelen soruya göre doğru Cypher sorgusunu oluştur.
Soru: {question}
""")

# GraphCypherQAChain oluştur
customer_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=CUSTOMER_PROMPT_TEMPLATE
)
