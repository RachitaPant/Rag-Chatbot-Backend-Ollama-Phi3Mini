from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import glob
from pypdf import PdfReader
from gtts import gTTS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AUDIO_DIR = "audio"

os.makedirs(AUDIO_DIR, exist_ok=True)

def load_docs():
    docs = []
    for filepath in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        try:
            reader = PdfReader(filepath)
            text = "\n".join([
                page.extract_text() for page in reader.pages if page.extract_text()
            ])
            docs.append(Document(page_content=text, metadata={"source": filepath}))
        except Exception as e:
            print(f"Error loading PDF {filepath}: {e}")

    for filepath in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": filepath}))
        except Exception as e:
            print(f"Error loading text file {filepath}: {e}")

    return docs

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
        docs = load_docs()
        if not docs:
            raise ValueError("No documents found in the data directory.")
        print(f"Loaded {len(docs)} documents.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
        splits = splitter.split_documents(docs)
        vectordb = Chroma.from_documents(splits, embeddings, persist_directory=VECTOR_DIR)
    else:
        vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    return vectordb

# Initialize vector store, retriever, chains
try:
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt_template = """You are a helpful assistant that answers questions based solely on the provided documents. 
Use only the information from the documents to generate your response. 
If the documents do not contain the information needed to answer the question, respond with: 
"I don't have the information in the provided context."

Documents:
{context}

Question: {input}

Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

    llm = OllamaLLM(model="phi3:3.8b-mini-128k-instruct-q4_0", timeout=60)

    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
except Exception as e:
    print(f"Error initializing chain: {e}")
    raise

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    try:
        print(f"Received question: {query.question}")
        result = qa_chain.invoke({"input": query.question})
        answer = result.get("answer", "I could not generate an answer.")
        print(f"Answer: {answer}")

        # Generate audio file with Indian female voice (Google India domain)
        audio_file = os.path.join(AUDIO_DIR, "response.mp3")
        tts = gTTS(text=answer, lang="en", tld="co.in")  # <-- Indian English female
        tts.save(audio_file)

        return {
            "answer": answer,
            "sources": [doc.metadata for doc in result.get("context", [])],
            "audio_url": f"/audio/response.mp3"
        }
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

# Serve audio files
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)