# app.py
"""
Deployment-ready FastAPI app (app.py).
- Uses a lifespan handler to initialize vector DB and RAG chain at startup.
- Runs heavy startup work in a background thread to avoid blocking the event loop.
- GET /ask (help), POST /ask (API), GET /health, and root redirect to /ask.
- Fix: POST /ask passes a plain string to rag_chain.invoke (avoids PyString/dict error).
"""

import os
import logging
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

# LangChain / vector/LLM imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proposal-api")

# -------------------------
# Config & paths
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables")

PDF_PATH = "Proposal/Proposal Knowledge Base (1).pdf"  # update if needed
CHROMA_PERSIST_DIR = "chroma_db"

# Globals populated at startup
vector_db: Optional[Chroma] = None
retriever = None
rag_chain = None
llm = None
prompt = None

# -------------------------
# Prompt Template
# -------------------------
prompt_template = """You are a proposal generation assistant for a software development company. 
Your task is to create accurate, professional project proposals based ONLY on the information provided in the retrieved context.

**STRICT RULES:**
1. ONLY use information from the provided context below.
2. DO NOT invent or hallucinate any features, pricing, or details not in the context.
3. If the requested project type doesn't exist in the context, respond: 
   "I don't have information about this type of project in our knowledge base."
4. Match the project type EXACTLY — use the closest match from available project templates.
5. Maintain the exact pricing structure and currency (USD/INR/Rs) as specified in the context.
6. Include all standard sections: Admin Panel, App/Website Features, Technologies, Pricing, and Maintenance.
7. Do not add introductions or anything extra — just keep all points as they are.
8. Do not add numbers or reformat prices.

**RETRIEVED CONTEXT:**
{context}

**USER QUESTION:**
{question}

**OUTPUT FORMAT:**
Generate a proposal with these sections:
- Admin Panel Features (bullet points from context)
- App/Website Features (bullet points from context)
- Technologies (from context)
- Pricing Breakdown (exact amounts and currency from context)
- Maintenance (from context)

**VALIDATION CHECKLIST:**
- All features listed exist in the retrieved context.
- Pricing matches exactly (including currency).
- No invented or assumed features.
- Maintenance costs are accurate.

If any information is missing, explicitly state: 
"This information is not available in the knowledge base."

Generate the proposal now:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="Proposal Generator",
    description="Generate project proposals based on a PDF knowledge base (RAG)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

def format_docs(docs):
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

# -------------------------
# Startup logic (synchronous helper)
# -------------------------
def startup_sync():
    """
    Runs heavy synchronous startup tasks:
    - Load or build Chroma vector DB (persisted).
    - Initialize retriever, llm and rag_chain.
    This function is intended to be run in a background thread from an async lifespan.
    """
    global vector_db, retriever, llm, rag_chain, prompt

    # 1) Ensure PDF exists
    if not os.path.exists(PDF_PATH):
        logger.error("PDF not found at %s", PDF_PATH)
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    logger.info("Creating/Loading embeddings and Chroma DB (may call OpenAI).")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    # Load persisted Chroma if present
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        logger.info("Loading persisted Chroma DB from '%s'...", CHROMA_PERSIST_DIR)
        vector_db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        logger.info("Persisted Chroma DB loaded.")
    else:
        logger.info("Persisted Chroma DB not found — building from PDF.")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        logger.info("Loaded %d pages from PDF", len(docs))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=250,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        splits = text_splitter.split_documents(docs)
        logger.info("Created %d chunks.", len(splits))

        vector_db = Chroma.from_documents(splits, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
        vector_db.persist()
        logger.info("Chroma DB built and persisted to %s", CHROMA_PERSIST_DIR)

    retriever = vector_db.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)

    # Build LCEL pipeline (your original approach)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("RAG chain initialized successfully.")

# -------------------------
# Lifespan handler (preferred over @app.on_event)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run startup_sync in a thread so we don't block the event loop
    try:
        logger.info("Starting up - initializing resources...")
        await asyncio.to_thread(startup_sync)
        logger.info("Startup complete.")
    except Exception as e:
        logger.exception("Startup failed: %s", e)
        # Re-raise to prevent app from starting in a broken state
        raise
    try:
        yield
    finally:
        # Optional: add shutdown cleanup here if needed
        logger.info("Shutting down.")

app.router.lifespan_context = lifespan  # attach the lifespan handler

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
async def root():
    return RedirectResponse(url="/ask")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "proposal-generator"}

@app.get("/ask")
async def ask_get():
    return {
        "message": "POST JSON to /ask with {'question': '...'}",
        "example": {"question": "Create a proposal for a mobile app from the knowledge base."}
    }

@app.post("/ask")
async def ask_post(query: Query):
    if not query.question or len(query.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain not initialized")

    try:
        # ---- FIX: pass the question string (not a dict) to rag_chain.invoke ----
        # rag_chain.invoke expects a plain string input (the question).
        result = await run_in_threadpool(rag_chain.invoke, query.question)
        return {"answer": result}
    except Exception as e:
        logger.exception("Error generating proposal: %s", e)
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")

# -------------------------
# Local runner
# -------------------------
if __name__ == "__main__":
    import uvicorn
    # You can run either:
    #  - python app.py
    #  - or: uvicorn app:app --host 0.0.0.0 --port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
