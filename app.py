from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# PDF Path
PDF_PATH = "Proposal/Proposal Knowledge Base (1).pdf"

# Check if PDF exists
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"Loaded {len(docs)} pages")

# Split documents
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=250,
    separators=["\n\n", "\n", ".", "!", "?"]
)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks")

# Create embeddings
print("Creating embeddings...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key
)

# Create Chroma vector database (stored locally)
print("Building vector database...")
vector_db = Chroma.from_documents(splits, embedding=embeddings)
print("Vector database ready")

# Prompt template
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

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=openai_api_key
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 8}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

# FastAPI app
app = FastAPI(
    title="Proposal Generator",
    description="Generate project proposals based on knowledge base",
    version="1.0.0"
)

# CORS middleware
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

# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "proposal-generator"}

@app.post("/ask")
async def ask(query: Query):
    """Ask a question and get a proposal based on knowledge base"""
    if not query.question or len(query.question.strip()) == 0:
        return {"error": "Question cannot be empty"}
    
    try:
        answer = qa_chain.run(query.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"Error generating proposal: {str(e)}"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)