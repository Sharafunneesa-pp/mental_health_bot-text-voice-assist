import os
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS,Chroma
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import ArxivLoader, WebBaseLoader
from langchain_openai import OpenAIEmbeddings

import bs4
import pathlib

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

class chat_gen():
    def __init__(self):
        self.chat_history = []

    def load_documents(self, data):
        
        # Load PDF documents
        pdf_loader = PyPDFDirectoryLoader(data)
        pdf_documents = pdf_loader.load()
        
        
        
        # Load arXiv papers
        arxiv_loader = ArxivLoader(query="2401.17477", load_max_docs=2)
        arxiv_documents = arxiv_loader.load()
        
        
        # Load web pages
        web_loader = WebBaseLoader(
            web_path=("https://www.helpguide.org/articles/mental-health/social-media-and-mental-health.htm",),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title", "post-content", "post-header")))
        )
        web_documents = web_loader.load()

        #wikipedia
        from langchain_community.document_loaders import WikipediaLoader
        wiki = WikipediaLoader(query="Digital media use and mental health", load_max_docs=2).load()
        
        # Combine all documents
        documents = pdf_documents  + arxiv_documents +wiki+web_documents

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
        docs = text_splitter.split_documents(documents=documents)

        # Create embeddings
        # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
        embeddings=OpenAIEmbeddings()

        # Create vector store
    
        vectorstore=Chroma.from_documents(documents=docs,embedding=embeddings)
        vectorstore=Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="./chroma_db")
        # vectorstore.save_local("faiss_index_datamodel")

        # Load from local storage
        db2 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        db2.persist()
        # persisted_vectorstore = Chroma.load_local("chroma_index_datamodel", embeddings, allow_dangerous_deserialization=True)
        return db2

    def load_model(self):
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=3000)
       



        system_instruction = """You are a compassionate and supportive AI assistant or therapist or a counselor focused on providing mental health support. 
        Respond to users' queries with empathy and understanding ask whether any social media problem facing you, while being concise and encouraging.
        Follow these guidelines:
        Generate the most supportive and helpful response possible, carefully considering all previous generated text in your response before adding new tokens to the response.
        just use the context if added. Use all of the context of this conversation so your response is relevant to the conversation. '
        Make your responses clear and concise, avoiding any verbosity.Use historical interaction data to offer personalized support, avoiding unnecessary external searches unless essential.
         Prioritize concise, user-focused, and self-contained responses.
        1. Acknowledge the user's feelings and validate their emotions.
        2. Ask open-ended questions to encourage the user to share more.
        3. Provide actionable advice or coping strategies.
        4. Offer resources or suggest activities that can help improve their mental health.
        5. Be positive and encouraging, focusing on the user's strengths and potential for improvement.
        6. Maintain a warm and supportive tone throughout the conversation.
        7.If context  is not  available, provide answers based on your knowledge"""

        # Define template with system instruction
        template = (
            f"{system_instruction} "
           "Combine the chat history {chat_history} and follow up question into provide perfect answer for the question and "
           "a standalone question to answer from the {context} or from your knowledge "
             "Follow up question: {question}"
       )



        prompt = PromptTemplate.from_template(template)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.load_documents("data").as_retriever(),
            combine_docs_chain_kwargs={'prompt': prompt},
            chain_type="stuff",
        )
        return chain

    def ask_query(self, query):
        result = self.load_model()({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        return result['answer']

if __name__ == "__main__":
    chat = chat_gen()
    