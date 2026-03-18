# TCC - Matheus Dias | Sistemas de Informação 2025

## Avaliação de Modelos de Linguagem em Pipelines RAG para Suporte ao Comércio Eletrônico

Este repositório contém os artefatos do Trabalho de Conclusão de Curso (TCC) do curso de Sistemas de Informação, que investiga o desempenho de diferentes modelos de linguagem (LLMs) integrados a um pipeline de **RAG (Retrieval-Augmented Generation)** aplicado a um cenário de e-commerce.

---

## Objetivo

Comparar a qualidade das respostas, latência e consumo de tokens entre diferentes LLMs ao responderem perguntas sobre dados de uma empresa fictícia de tecnologia, utilizando recuperação semântica como base de contexto.

---

## Modelos Avaliados

| Provedor | Modelos |
|----------|---------|
| OpenAI | GPT-3.5 Turbo, GPT-4o Turbo |
| Google | Gemini 2.0 Flash, Gemini 2.5 Flash |
| DeepSeek | DeepSeek Chat |

---

## Arquitetura do Pipeline RAG

```
CSVs (empresa, produtos, serviços)
        ↓
  Documentos LangChain
        ↓
  Embeddings (MiniLM - HuggingFace)
        ↓
  Índice Vetorial (ChromaDB)
        ↓
  Busca Semântica (Top-K)
        ↓
  LLM (GPT / Gemini / DeepSeek)
        ↓
  Resposta + Métricas (CSV)
```

---

## Base de Conhecimento

A base de dados utilizada é fictícia e representa a empresa **Tech Solutions Comércio e Serviços Ltda**, contendo:

- **empresa.csv** — dados institucionais (endereço, CNPJ, horários, políticas, etc.)
- **produtos.csv** — catálogo com preços, estoque e categorias
- **servicos.csv** — serviços oferecidos e valores

---

## Conjunto de Perguntas

Foram definidas **30 perguntas padronizadas** (`questions_padronizadas.csv`) cobrindo:

- Preços de produtos
- Quantidades em estoque
- Informações institucionais (fundação, CNPJ, funcionários, filiais)
- Serviços e preços
- Políticas (troca, entrega, garantia, pagamento)

---

## Estrutura do Repositório

```
TCC.PROJ/
│
├── dados_gpt/                          # Pipeline e resultados para GPT
│   ├── rag_tester_gpt.py               # Script principal RAG (OpenAI)
│   ├── empresa.csv
│   ├── produtos.csv
│   ├── servicos.csv
│   ├── questions_padronizadas.csv
│   ├── resultados_rag.csv
│   └── chroma_index/                   # Índice vetorial persistido
│
├── dados_gemini/                       # Pipeline e resultados para Gemini
│   ├── rag_tester_gemini.py
│   └── ...
│
├── dados_deepseek/                     # Pipeline e resultados para DeepSeek
│   ├── rag_tester_deepseek.py
│   └── ...
│
├── resultados Python gpt/
│   ├── resultados_rag_gpt-3.5-turbo.csv
│   └── resultados_rag_gpt-4.o-turbo.csv
│
├── resultados Python gemini/
│   ├── resultados_rag_gemini_2.0-Flash.csv
│   └── resultados_rag_gemini_2.5-Flash.csv
│
├── resultados Python deepseek/
│   └── resultados_rag_deepseek.csv
│
├── Resultados Tabelados - Python-Postman.xlsx   # Consolidação e análise dos resultados
├── CONTEXTO_LOJA.txt                            # Contexto completo da empresa (JSON)
└── PERGUNTAS TESTES.txt                         # Perguntas utilizadas nos testes
```

---

## Métricas Coletadas

Cada execução registra por pergunta:

| Coluna | Descrição |
|--------|-----------|
| `timestamp` | Data e hora da execução |
| `id` | Identificador da pergunta |
| `question` | Texto da pergunta |
| `answer` | Resposta gerada pelo LLM |
| `latency_s` | Tempo de resposta em segundos |
| `prompt_tokens` | Tokens enviados ao modelo |
| `completion_tokens` | Tokens da resposta gerada |
| `total_tokens` | Total de tokens consumidos |
| `top_k` | Quantidade de chunks recuperados |
| `sources` | Arquivos fonte utilizados na recuperação |

---

## Tecnologias Utilizadas

- **Python 3.10+**
- **LangChain** — orquestração do pipeline RAG
- **ChromaDB** — banco de vetores local
- **HuggingFace Sentence Transformers** — modelo de embeddings (`all-MiniLM-L6-v2`)
- **OpenAI API** — GPT-3.5 Turbo / GPT-4o Turbo
- **Google Generative AI** — Gemini 2.0 Flash / 2.5 Flash
- **DeepSeek API** — DeepSeek Chat
- **Pandas** — manipulação e exportação de dados

---

## Como Executar

### Pré-requisitos

```bash
pip install langchain langchain-openai langchain-community chromadb \
            faiss-cpu tiktoken pandas openai sentence-transformers
```

### Variáveis de ambiente

```bash
# Para GPT
export OPENAI_API_KEY=sk-...
export OPENAI_CHAT_MODEL=gpt-3.5-turbo   # ou gpt-4o

# Para Gemini
export GOOGLE_API_KEY=...

# Para DeepSeek
export DEEPSEEK_API_KEY=...
```

### Execução

```bash
# Dentro de cada pasta (ex: dados_gpt)
cd dados_gpt
python rag_tester_gpt.py
```

Os resultados serão salvos em `resultados_rag.csv` e o índice vetorial em `chroma_index/`.

---

## Autor

**Matheus Dias**
Curso de Sistemas de Informação — 2025
