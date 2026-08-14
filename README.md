# Digital Egypt Assistant | المساعد الرقمي لخدمات مصر

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://digital-egypt-assistant-k4swnbph4rsfqkvtckrejt.streamlit.app/)

🌐 **Live Demo:** [digital-egypt-assistant.streamlit.app](https://digital-egypt-assistant-k4swnbph4rsfqkvtckrejt.streamlit.app/)

An AI-powered **Arabic-language chatbot** that helps citizens navigate Egypt's digital government services. Built with **Streamlit**, **LangChain**, and **RAG (Retrieval-Augmented Generation)** using a ChromaDB vector store and multilingual HuggingFace embeddings.

> اسألني عن أي خدمة رقمية متاحة على بوابة مصر الرقمية


---

## Features

- **Arabic-first** — Fully supports Arabic queries and responses using multilingual embeddings (`intfloat/multilingual-e5-large`)
- **RAG Architecture** — Retrieves relevant government service information from a local ChromaDB vector store before generating answers
- **Multi-LLM Support** — Switch between OpenAI GPT-4o, Together AI LLaMA 3, and Google Gemini from the sidebar; each provider's API key is read from `.env`
- **Session-aware Memory** — Conversation history is maintained across turns using `StreamlitChatMessageHistory`
- **Conversation Summarization** — One-click sidebar button condenses the chat history into a short Arabic digest
- **Quick-access Service Buttons** — One-click buttons for the most commonly queried services (vehicle fines, social insurance, etc.)
- **Streaming Responses** — Answers are streamed chunk-by-chunk for a smooth experience
- **Self-contained** — No external web links; all answers are grounded in the internal knowledge base

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | [Streamlit](https://streamlit.io/) |
| LLM Options | GPT-4o · LLaMA 3.3 70B · Gemini 2.0 Flash |
| LLM Framework | [LangChain](https://www.langchain.com/) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | `intfloat/multilingual-e5-large` (HuggingFace) |
| RAG Chain | `create_retrieval_chain` + `create_stuff_documents_chain` |
| Memory | `StreamlitChatMessageHistory` |

---

## Project Structure

```
Digital-Egypt-Assistant/
├── app.py                          # Entry point: page setup + wiring only
├── config.py                       # AppConfig / VectorStoreConfig (pydantic settings)
├── core/
│   ├── llm_factory.py              # Builds + smoke-tests the chosen LLM provider
│   ├── prompts.py                  # System prompt + ChatPromptTemplate
│   ├── rag_chain.py                # Retrieval + history-aware chain assembly
│   ├── langsmith_config.py         # LangSmith tracing settings (pydantic settings)
│   └── agents/
│       ├── answer_agent.py         # Streams one chat turn, persists it to memory (@traceable)
│       └── summary_agent.py        # Summarizes the running chat history (@traceable)
├── bootstrap/
│   └── vectorstore_provider.py     # Cached embedding model + Chroma vector store
├── ui.py                           # Sidebar + quick-access buttons + chat rendering
├── .env.example                    # Provider + LangSmith API key template
├── .gitignore                      # Keeps .env out of version control
├── requirements.txt                # Python dependencies
├── egypt_digital.pkl                # Pickled data used to build the vector store
├── infloat/                        # ChromaDB persistence directory
│   └── e5116b2d-.../               # Chroma collection files
└── README.md
```

Structure mirrors the layering used in this author's counselling-bot project (config → bootstrap → core → presentation → utils), scaled down for a single-page Streamlit app.

---

## Getting Started

### Prerequisites

- Python 3.9+
- An API key for **at least one** of:
  - [OpenAI](https://platform.openai.com/) — for GPT-4o
  - [Together AI](https://www.together.ai/) — for LLaMA 3
  - [Google AI Studio](https://aistudio.google.com/) — for Gemini

### Installation

```bash
git clone https://github.com/Mahmedorabi/Digital-Egypt-Assistant.git
cd Digital-Egypt-Assistant

# (Optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

# Copy the env template and fill in at least one provider key
cp .env.example .env
```

### Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Optional: LangSmith Tracing

Copy `.env.example` to `.env` and fill in `LANGCHAIN_API_KEY` to trace every
run in [LangSmith](https://smith.langchain.com/) — the retrieval chain's own
LangChain calls are traced automatically, and `core/agents/answer_agent.py` /
`core/agents/summary_agent.py` are wrapped with `@traceable` so each chat turn
and each summarization call shows up as its own named run under the
`LANGCHAIN_PROJECT` project. Leave `LANGCHAIN_API_KEY` blank to run untraced.

---

## How It Works

```
  User Question (Arabic)
         │
         ▼
  ┌─────────────────────────┐
  │  ChromaDB Retriever     │  ← top-10 relevant chunks
  │  (multilingual-e5-large)│
  └──────────┬──────────────┘
             │ context
             ▼
  ┌─────────────────────────┐
  │  ChatPromptTemplate     │
  │  (system + context +    │
  │   chat history + user)  │
  └──────────┬──────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │  LLM (GPT-4o /          │
  │   LLaMA 3 / Gemini)     │
  └──────────┬──────────────┘
             │ streamed answer
             ▼
       Streamlit Chat UI
```

1. **Choose your LLM** — Select the model in the sidebar; its API key is read from `.env`.
2. **Ask in Arabic** — Type a question about any digital government service or click a quick-access button.
3. **RAG retrieval** — The retriever fetches the top-10 relevant document chunks from the vector store.
4. **Answer generation** — The LLM generates a grounded Arabic response based on the retrieved context and chat history.
5. **Streaming display** — The answer is streamed and rendered incrementally.

---

## Supported Services (Examples)

| Service | Example Query |
|---|---|
| Vehicle Fines | استعلام عن مخالفات رخصة مركبة |
| Social Insurance | الاستعلام عن آخر مدة تأمينية |
| Insurance Number | ما هو رقمي التأميني؟ |
| Subscription Periods | مدد الاشتراك والأجور في التأمين |
| Disbursement Inquiry | الاستعلام عن صرف معين |

---

## Important Notes

- **API Keys** — Keys live in `.env` (never commit it — see `.gitignore`), not in the UI. Switching models in the sidebar only works if that provider's key is set.
- **Gemini** — `gemini-2.0-flash` is used; ensure your Google API plan supports it.
- **Vector Store** — The `infloat/` directory contains the pre-built ChromaDB index. Do not delete it.

---

## Emergency Numbers (Built-in Knowledge)

| Number | Service |
|---|---|
| 122 | Police (النجدة) |
| 123 | Ambulance (الإسعاف) |
| 121 | Electricity Inquiries |
| 15999 | Digital Egypt Portal Support |

---

## Dependencies

```
streamlit
langchain
langchain-core
langchain-community
langchain-openai
langchain-together
langchain-google-genai
chromadb
huggingface-hub
sentence-transformers
tiktoken
python-dotenv
uuid
```

Install all:

```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

MIT License — free to use, modify, and deploy.

---

## Acknowledgments

Built by **Mohamed Orabi** as a prototype to enhance accessibility to Egypt's digital government services through conversational AI.

For feedback, visit: [https://digital.gov.eg/feedback](https://digital.gov.eg/feedback)
