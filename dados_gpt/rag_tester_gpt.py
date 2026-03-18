# -*- coding: utf-8 -*-
"""
RAG Tester (simples) - CSVs -> índice vetorial -> recuperação -> LLM -> resultados CSV

Este script implementa um pipeline de testes baseado em RAG (Retrieval-Augmented Generation),
que utiliza arquivos CSV como base de dados, gera embeddings vetoriais, realiza busca semântica
e envia consultas para um modelo de linguagem (LLM), registrando os resultados em CSV.

Requisitos de instalação:
    pip install langchain langchain-openai langchain-community chromadb faiss-cpu tiktoken pandas openai sentence-transformers

Variáveis de ambiente necessárias:
    OPENAI_API_KEY=sk-...

Arquivos de entrada esperados:
    empresa.csv, produtos.csv, servicos.csv, questions_padronizadas.csv

Saídas geradas:
    resultados_rag.csv e a pasta chroma_index/ (índice vetorial persistido)
"""

# ---------------------------- Imports padrão ----------------------------
import os, time                    # os → manipulação de variáveis de ambiente / paths; time → medir latência (tempo de execução)
import pandas as pd                # pandas → leitura, manipulação e gravação de dados tabulares (CSV)
from pathlib import Path           # Path → manipulação de diretórios e arquivos de forma multiplataforma
from datetime import datetime      # datetime → registrar timestamp em cada execução

# ---------------------------- Imports do ecossistema LangChain ----------------------------
# As bibliotecas abaixo pertencem ao framework LangChain, que organiza o fluxo de dados entre documentos, embeddings e LLMs.

from langchain_core.documents import Document                       # Representa um documento (texto + metadados)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Divide textos em blocos (chunks) menores, preservando sentido
from langchain_community.embeddings import HuggingFaceEmbeddings     # Gera embeddings locais usando o modelo MiniLM gratuito
from langchain_community.vectorstores import Chroma                  # Banco vetorial ChromaDB (armazenamento de embeddings)
from langchain_openai import ChatOpenAI                              # Conector LangChain para os modelos da OpenAI (GPT-3.5, GPT-4, etc.)
from langchain_core.messages import SystemMessage, HumanMessage      # Estrutura de mensagens trocadas com o LLM



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Lê a chave do ambiente do sistema
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")  # Define o modelo padrão de teste
CHROMA_DIR = "chroma_index"                             # Diretório onde o índice vetorial será armazenado
EMPRESA_CSV, PRODUTOS_CSV, SERVICOS_CSV = "empresa.csv","produtos.csv","servicos.csv"
QUESTIONS_CSV = "questions_padronizadas.csv"            # Arquivo com perguntas padronizadas (id, question)
RESULTS_CSV = "resultados_rag.csv"                      # Arquivo de saída consolidado (respostas e métricas)
TOP_K = int(os.getenv("RAG_TOPK", "26"))                 # Quantidade de trechos mais semelhantes recuperados (Top-K)

# Define o separador dos arquivos CSV
EMPRESA_SEP  = ";"
PRODUTOS_SEP = ";"
SERVICOS_SEP = ";"


def load_docs():
    """
    Lê os CSVs (empresa, produtos, serviços), normaliza os cabeçalhos
    e transforma cada linha em um objeto Document com conteúdo e metadados.

    Retorna: lista de Document
    """
    # Leitura dos arquivos CSV usando pandas
    emp = pd.read_csv(EMPRESA_CSV, sep=EMPRESA_SEP, engine="python", dtype=str).fillna("")
    emp.columns = emp.columns.str.strip().str.lower()
  
    pro = pd.read_csv(PRODUTOS_CSV, sep=PRODUTOS_SEP, engine="python", dtype=str).fillna("")
    pro.columns = pro.columns.str.strip().str.lower()

    ser = pd.read_csv(SERVICOS_CSV, sep=SERVICOS_SEP, engine="python", dtype=str).fillna("")
    ser.columns = ser.columns.str.strip().str.lower()

    docs = []

    # Converte cada linha de "empresa.csv" em um documento textual padronizado
    for _, r in emp.iterrows():
        docs.append(
            Document(
                page_content=f"TIPO: EMPRESA\n{r['campo']}: {r['valor']}",
                metadata={"tipo": "empresa", "fonte": "empresa.csv"}
            )
        )

    # Cada produto é transformado em um documento com campos essenciais (para recuperação semântica)
    for _, r in pro.iterrows():
        txt = (
            f"TIPO: PRODUTO\nproduto_id: {r['produto_id']}\n"
            f"nome: {r['nome']}\npreco: {r['preco']}\n"
            f"estoque: {r['estoque']}\ncategoria: {r['categoria']}"
        )
        docs.append(
            Document(
                page_content=txt,
                metadata={"tipo": "produto", "fonte": "produtos.csv", "id": str(r["produto_id"])}
            )
        )

    # Cada serviço também é convertido em um documento textual
    for _, r in ser.iterrows():
        txt = (
            f"TIPO: SERVICO\nservico_id: {r['servico_id']}\n"
            f"nome: {r['nome']}\npreco: {r['preco']}"
        )
        docs.append(
            Document(
                page_content=txt,
                metadata={"tipo": "servico", "fonte": "servicos.csv", "id": str(r["servico_id"])}
            )
        )

    return docs


def build_or_load_index(docs):
    """
    Cria ou carrega o índice vetorial (ChromaDB).
    Se a pasta já existir, reabre o índice persistido.
    Caso contrário, cria embeddings e salva os vetores no disco.
    """
    # Cria o modelo de embeddings baseado em MiniLM (HuggingFace)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist = Path(CHROMA_DIR)

    # Se o índice já existe, reusa (para evitar recriação e acelerar execuções)
    if persist.exists() and any(persist.iterdir()):
        return Chroma(collection_name="tcc_rag", embedding_function=emb, persist_directory=CHROMA_DIR)

    # Divide os documentos em blocos menores (chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    # Cria o banco vetorial Chroma, adiciona os chunks e salva
    vdb = Chroma(
        collection_name="tcc_rag",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=CHROMA_DIR
    )
    vdb.add_documents(chunks)
    vdb.persist()
    return vdb


def retrieve(vdb, question, k=TOP_K):
    """
    Realiza a busca semântica (Top-K) no índice vetorial.
    Concatena os conteúdos retornados e registra as fontes de origem.
    Retorna: contexto textual e fontes.
    """
    docs = vdb.similarity_search(question, k=k)  # Busca semântica pelos K documentos mais similares
    contexto = "\n\n".join(d.page_content for d in docs)  # Junta todos os textos encontrados
    fontes = "|".join(sorted(set([d.metadata.get("fonte","") for d in docs if d.metadata])))  # Auditoria de fontes
    return contexto, fontes


def ask_llm(contexto, question):
    """
    Envia o contexto + pergunta ao modelo de linguagem (LLM) via API OpenAI.
    - O modelo deve responder SOMENTE com base no contexto recuperado.
    - Caso não haja informação, deve retornar "Não encontrado na base."
    Retorna: resposta do modelo, tempo de execução e metadados de uso (tokens).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("Defina OPENAI_API_KEY=sk-... para a geração.")

    # Cria cliente LLM (ChatOpenAI) – modelo GPT-3.5-turbo por padrão
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY)

    # Monta o prompt com instruções rígidas
    sys = (
        "Você responde SOMENTE com base no CONTEXTO.\n"
        "Se a resposta não estiver no contexto, diga: 'Não encontrado na base.'\n"
        "Se houver valores numéricos, mostre-os exatamente como no contexto."
        "\n\nCONTEXTO:\n" + contexto
    )

    # Mede o tempo de execução (latência)
    t0 = time.time()
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=question)])
    dt = round(time.time() - t0, 3)

    # Extrai o texto da resposta e as métricas de tokens (se disponíveis)
    answer = getattr(resp, "content", str(resp))
    usage = getattr(resp, "usage_metadata", {}) if hasattr(resp, "usage_metadata") else {}
    return answer, dt, usage


def main():
    """
    Função principal que executa todo o pipeline:
    1) Carrega e transforma os CSVs em documentos
    2) Cria ou reabre o índice vetorial (Chroma)
    3) Lê as perguntas padronizadas
    4) Para cada pergunta, realiza busca, gera resposta e mede tempo
    5) Armazena os resultados e métricas em 'resultados_rag.csv'
    """
    print("Carregando base...")
    docs = load_docs()
    print(f"Documentos base: {len(docs)}")

    print("Criando/abrindo índice vetorial (Chroma)...")
    vdb = build_or_load_index(docs)

    # Leitura das perguntas do CSV (colunas: id, question)
    qdf = pd.read_csv(QUESTIONS_CSV, sep=None, engine="python")

    out = []  # Lista para armazenar os resultados
    for _, row in qdf.iterrows():
        qid = row.get("id","")         
        q = row["question"]            

        contexto, fontes = retrieve(vdb, q, k=TOP_K)
        ans, lat, usage = ask_llm(contexto, q)

        # Registra dados de desempenho e uso
        out.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "id": qid,
            "question": q,
            "answer": ans,
            "latency_s": lat,
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "top_k": TOP_K,
            "sources": fontes
        })
        print(f"OK {qid}: {lat}s")

    # Salva resultados em CSV para posterior análise (latência, tokens e fontes)
    pd.DataFrame(out).to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"Resultados salvos em {RESULTS_CSV}")


# Executa a função principal quando o arquivo é executado diretamente
if __name__ == "__main__":
    main()
