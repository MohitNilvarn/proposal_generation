"""proposal_api.py

Improved FastAPI app with startup initialization, Chroma persistence,
safe blocking calls from async endpoints, root redirect and GET /ask help page.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv
import os
import logging
from typing import Optional

# LangChain / vector/LLM imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proposal-api")

load_dotenv()

# -------------------------
# Config & paths
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables")

# Local PDF and Chroma persistence directory - change if needed
PDF_PATH = "Proposal/Proposal Knowledge Base (1).pdf"
CHROMA_PERSIST_DIR = "chroma_db"

# Vector / chain globals (populated on startup)
vector_db: Optional[Chroma] = None
retriever = None
rag_chain = None
llm = None
prompt = None

# -------------------------
# Prompt Template (unchanged logic)
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

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="Proposal Generator",
    description="Generate project proposals based on a PDF knowledge base (RAG)",
    version="1.0.0",
)

# CORS - adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class Query(BaseModel):
    question: str

# -------------------------
# Utility helpers
# -------------------------
def format_docs(docs):
    """Concatenate retrieved docs into a single context string."""
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

# -------------------------
# Startup: build / load vector DB and chain
# -------------------------
@app.on_event("startup")
def startup_event():
    global vector_db, retriever, llm, rag_chain, prompt

    # 1) Ensure PDF exists
    if not os.path.exists(PDF_PATH):
        logger.error("PDF not found at %s", PDF_PATH)
        # Raise so that deployments/platforms know startup failed
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    logger.info("Initializing embeddings and vector DB (this may take a while on first run)...")

    # 2) Create embeddings client
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    # 3) If Chroma persisted DB exists, load it. Otherwise build from PDF and persist.
    try:
        if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            logger.info("Loading persisted Chroma DB from '%s'...", CHROMA_PERSIST_DIR)
            vector_db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
            logger.info("Loaded persisted Chroma DB.")
        else:
            # Load PDF
            logger.info("Loading PDF from %s", PDF_PATH)
            loader = PyPDFLoader(PDF_PATH)
            docs = loader.load()
            logger.info("Loaded %d pages from PDF", len(docs))

            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=250,
                separators=["\n\n", "\n", ".", "!", "?"]
            )
            splits = text_splitter.split_documents(docs)
            logger.info("Created %d document chunks.", len(splits))

            # Build Chroma and persist
            logger.info("Building Chroma vector DB (this will call OpenAI embeddings).")
            vector_db = Chroma.from_documents(splits, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
            vector_db.persist()
            logger.info("Chroma DB built and persisted to %s", CHROMA_PERSIST_DIR)

    except Exception as e:
        logger.exception("Failed to create/load Chroma DB: %s", e)
        raise

    # 4) Create retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 8})

    # 5) Initialize LLM (ChatOpenAI)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY
    )

    # 6) Build a simple LCEL-style rag chain mapping (keeps your original pattern)
    # Note: We'll keep the same pipeline shape you had but use it as-is.
    try:
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain initialized.")
    except Exception:
        # If building the LCEL pipeline fails, log error but keep app startup to catch issues early.
        logger.exception("Failed to build RAG chain with LCEL. You may need to adjust pipeline code.")
        raise

# -------------------------
# Routes
# -------------------------
@app.get("/")
async def root():
    """Redirect to /ask (GET). Remove if you don't want automatic redirect."""
    return RedirectResponse(url="/ask")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "proposal-generator"}


@app.get("/ask")
async def ask_get():
    """Human-friendly GET help for the /ask endpoint"""
    return {
        "message": "POST a JSON object to this endpoint with the shape: { 'question': '...' }",
        "example": {"question": "Create a proposal for X project type from the knowledge base."}
    }


@app.post("/ask")
async def ask_post(query: Query):
    """Ask a question and get a proposal based on the knowledge base."""
    if not query.question or len(query.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if rag_chain is None:
        # Defensive: in case startup failed but app is still running
        raise HTTPException(status_code=500, detail="RAG chain not initialized")

    # The LCEL rag_chain expects an input mapping (we'll pass the question as the input).
    # Because rag_chain.invoke may be blocking, run it in a threadpool.
    try:
        # Provide the question as the input. LCEL pipeline will use retriever|format_docs for 'context'.
        result = await run_in_threadpool(rag_chain.invoke, {"question": query.question})
        return {"answer": result}
    except Exception as e:
        logger.exception("Error while generating proposal: %s", e)
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")

# -------------------------
# Local run (uvicorn)
# -------------------------
if __name__ == "__main__":
    # Note: when running with uvicorn in production, the startup event will be triggered.
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
