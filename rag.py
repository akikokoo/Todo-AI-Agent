# environment variables
from dotenv import load_dotenv

# path
from pathlib import Path

# logging
import logging as log

# langchain & vectorstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.utils.math import cosine_similarity
from langchain_community.vectorstores import Chroma
import chromadb

# modules
from modules import log_time

# others
import numpy as np
from operator import itemgetter

load_dotenv()

log.basicConfig(
    filename="rag_log.log",
    level=log.INFO,
    format="%(asctime)s :: %(levelname)-8s :: %(message)s"
)
log.getLogger("httpx").setLevel(log.WARNING)

class RAG:
    def __init__(self, rag_system_message, file_path=Path(__file__).parent / "files"):
        log.info("----------------------------------------------------------------------------------Initializing RAG class----------------------------------------------------------------------------------")
        self.file_path = file_path
        
        # embedding model
        self.embedding_model = "text-embedding-3-large"
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

        # invoke llm attrs
        self.chat_model = "gpt-4o-mini"
        self.temperature = 0.3        

        # splitting attrs & class
        self.chunk_size = 400 
        self.chunk_overlap = 10
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        self.buffer_size = 3
        # self.text_splitter = SemanticChunker(
        #     embeddings=self.embeddings, 
        #     buffer_size=self.buffer_size
        # )
        
        # vectorstore attrs
        self.top_k = 15
        
        self.system_message = rag_system_message
    
    def is_rag_needed(self, question, threshold=0.4) -> bool:
        similarity_score = self.get_cosine_similarity(question)   
        max_similarity = np.max(similarity_score)
        is_rag_needed = max_similarity > threshold
        # log.info(f"RAG neediness: {is_rag_needed}")
        
        return is_rag_needed
    
    def load_documents(self): # NO NEED FOR IMPROVEMENT
        self.loader = DirectoryLoader(self.file_path, glob='*.txt', loader_cls=TextLoader, loader_kwargs={"encoding":"UTF-8"})
        self.docs = self.loader.load()
        # log.info(f"Loaded {len(self.docs)} documents.")

    @log_time
    def embed_documents(self): # NEED IMPROVEMENTS
        self.docs_embeddings = np.array(
            self.embeddings.embed_documents([doc.page_content for doc in self.docs])
        )
        # log.info(f"Document embeddings: {self.docs_embeddings}")

    def split_documents(self): # NO NEED FOR IMPROVEMENT
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        match self.text_splitter.__class__.__name__:
            case "RecursiveCharacterTextSplitter":
                self.splits = self.text_splitter.split_documents(self.docs)
            case "SemanticChunker":
                self.splits = self.text_splitter.split_documents(self.docs)

    @log_time
    def index_documents(self): # NEED IMPROVEMENTS
        self.vectorstore = Chroma.from_documents(
            documents=self.splits, 
            embedding=self.embeddings, 
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k":self.top_k})
    
    def get_cosine_similarity(self, question): # NO NEED FOR IMPROVEMENT
        question_embedding = self.embeddings.embed_query(question["content"])
        similarity = cosine_similarity([question_embedding], self.docs_embeddings)
        # log.info(f"Similarity: {similarity}")

        return similarity[0]
    
    @log_time
    def invoke_llm(self, question): # NEED IMPROVEMENTS
        prompt = ChatPromptTemplate.from_template(self.system_message)
        llm = ChatOpenAI(model=self.chat_model, temperature=self.temperature)
        chain = (
            {
                "context": itemgetter("question") | self.retriever,
                "question": itemgetter("question") 
            }
            | prompt
            | llm 
        )

        result = chain.invoke({"question":question["content"]})
        result.role = "assistant"
        result.tool_calls = None
        
        return result
    
    @log_time
    def run_rag(self, question):
        log.info("----------------------------------------------------------------------------------RAG PROCESS STARTED----------------------------------------------------------------------------------")
        log.info(f"Text Splitter: {self.text_splitter.__class__.__name__}")
        if isinstance(self.text_splitter, RecursiveCharacterTextSplitter):
            log.info(f"Chunk size: {self.chunk_size}")
            log.info(f"Chunk overlap: {self.chunk_overlap}")
        elif isinstance(self.text_splitter, SemanticChunker):
            log.info(f"Window size: {self.buffer_size}")

        result = None
        self.load_documents()
        self.embed_documents()
        if self.is_rag_needed(question):
            self.split_documents()
            self.index_documents()
            result = self.invoke_llm(question)

        log.info("----------------------------------------------------------------------------------RAG PROCESS FINISHED----------------------------------------------------------------------------------")
        return result
