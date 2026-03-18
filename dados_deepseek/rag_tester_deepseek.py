# -*- coding: utf-8 -*-
"""
RAG Tester (DeepSeek nativo LangChain) - CSVs -> índice vetorial -> recuperação -> LLM -> resultados CSV

Requisitos:
    pip install -U langchain langchain-deepseek langchain-community langchain-text-splitters \
                   chromadb faiss-cpu tiktoken pandas sentence-transformers

Vars de ambiente:
    DEEPSEEK_API_KEY=sk-..

Arquivos esperados na pasta:
    - empresa.csv           (colunas: campo;valor)
    - produtos.csv          (colunas: produto_id;nome;preco;estoque;categoria)
    - servicos.csv          (colunas: servico_id;nome;preco)
    - questions_padronizadas.csv  (colunas: id,question)  [sep auto]
Saída:
    - resultados_rag.csv
    - pasta chroma_index/ (índice vetorial persistido)
    "Defina antes de executar o script, por exemplo:"
    "PowerShell →  $env:DEEPSEEK_API_KEY='sua_chave_aqui'"
    "ou configure nas variáveis de ambiente do sistema."
"""

import os, time
import pandas as pd
from pathlib import Path
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# LLM DeepSeek (LangChain nativo)
from langchain_deepseek import ChatDeepSeek

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Lê a chave do ambiente do sistema


DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-chat") 
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

CHROMA_DIR = "chroma_index"
EMPRESA_CSV, PRODUTOS_CSV, SERVICOS_CSV = "empresa.csv", "produtos.csv", "servicos.csv"
QUESTIONS_CSV = "questions_padronizadas.csv"
RESULTS_CSV   = "resultados_rag.csv"
TOP_K = int(os.getenv("RAG_TOPK", "26"))

# Separadores dos CSVs
EMPRESA_SEP  = ";"
PRODUTOS_SEP = ";"
SERVICOS_SEP = ";"


def load_docs():
    """Lê CSVs e transforma cada linha em um Document padronizado."""
    emp = pd.read_csv(EMPRESA_CSV, sep=EMPRESA_SEP, engine="python", dtype=str).fillna("")
    emp.columns = emp.columns.str.strip().str.lower()

    pro = pd.read_csv(PRODUTOS_CSV, sep=PRODUTOS_SEP, engine="python", dtype=str).fillna("")
    pro.columns = pro.columns.str.strip().str.lower()

    ser = pd.read_csv(SERVICOS_CSV, sep=SERVICOS_SEP, engine="python", dtype=str).fillna("")
    ser.columns = ser.columns.str.strip().str.lower()

    docs = []

    # Empresa: "campo;valor"
    for _, r in emp.iterrows():
        docs.append(
            Document(
                page_content=f"TIPO: EMPRESA\n{r['campo']}: {r['valor']}",
                metadata={"tipo": "empresa", "fonte": "empresa.csv"}
            )
        )

    # Produtos
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

    # Serviços
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
    """Cria (ou reabre) índice vetorial Chroma com embeddings MiniLM."""
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist = Path(CHROMA_DIR)

    if persist.exists() and any(persist.iterdir()):
        return Chroma(collection_name="tcc_rag", embedding_function=emb, persist_directory=CHROMA_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    vdb = Chroma(
        collection_name="tcc_rag",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=CHROMA_DIR,
    )
    vdb.add_documents(chunks)
    vdb.persist()
    return vdb


def retrieve(vdb, question, k=TOP_K):
    """Busca semântica top-K e concatena contexto + fontes."""
    docs = vdb.similarity_search(question, k=k)
    contexto = "\n\n".join(d.page_content for d in docs)
    fontes = "|".join(sorted({d.metadata.get("fonte", "") for d in docs if d.metadata}))
    return contexto, fontes


def ask_llm(contexto, question):
    """
    Gera resposta com DeepSeek (LangChain).
    - Responder SOMENTE com base no CONTEXTO
    - Se não houver informação, dizer: 'Não encontrado na base.'
    - Manter valores numéricos como no contexto
    Retorna: (answer, dt, usage_dict)
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Defina DEEPSEEK_API_KEY=sk-...")

    llm = ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
    )

    sys = (
        "Você responde SOMENTE com base no CONTEXTO.\n"
        "Se a resposta não estiver no contexto, diga: 'Não encontrado na base.'\n"
        "Se houver valores numéricos, mostre-os exatamente como no contexto."
        "\n\nCONTEXTO:\n" + contexto
    )

    t0 = time.time()
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=question)])
    dt = round(time.time() - t0, 3)

    answer = getattr(resp, "content", str(resp))
    usage = getattr(resp, "usage_metadata", {}) if hasattr(resp, "usage_metadata") else {}
    # normaliza nomenclatura para salvar no CSV
    usage_norm = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    return answer, dt, usage_norm


def main():
    print("Carregando base...")
    docs = load_docs()
    print(f"Documentos base: {len(docs)}")

    print("Índice vetorial (Chroma)...")
    vdb = build_or_load_index(docs)

    qdf = pd.read_csv(QUESTIONS_CSV, sep=None, engine="python")
    out = []

    for _, row in qdf.iterrows():
        qid = row.get("id", "")
        q = row["question"]

        contexto, fontes = retrieve(vdb, q, k=TOP_K)
        ans, lat, usage = ask_llm(contexto, q)

        out.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "engine": "deepseek",
            "model": DEEPSEEK_MODEL,
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

    pd.DataFrame(out).to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"Resultados salvos em {RESULTS_CSV}")


if __name__ == "__main__":
    main()
