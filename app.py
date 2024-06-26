from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "https://life-balance-360.vercel.app"}})

os.environ["GOOGLE_API_KEY"] = "AIzaSyC5agUKvQR7gBuutdV0FSo0tpz2MRn8uL4"

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

directory="./ChatBotData"

loader = PyPDFDirectoryLoader(directory)   
documents = loader.load()


text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
pages = loader.load_and_split(text_splitter)

vectordb = Chroma.from_documents(pages, embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
answer:
"""

prompt = PromptTemplate.from_template(template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']
    response = retrieval_chain.invoke({"input": question})
    answer = response["answer"]
    return jsonify({"answer": answer})


# if __name__ == '__main__':
#     app.run(port=5003)
