import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# Ortam değişkenlerini yükle
load_dotenv()

# Neo4j bağlantısı
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# OpenAI modelini başlat
llm = ChatOpenAI(model="gpt-4", temperature=0)

# LangChain Cypher Q&A zinciri
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True  # Bu satır çok kritik
)

# Test sorgusu
question = "Hukuk departmanının işlediği veriler nelerdir?"
response = chain.run(question)

print("Soru:", question)
print("Cevap:", response)
