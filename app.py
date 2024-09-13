import os
import sys
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Directory to store persistent Chroma DB data
persist_directory = "./data"

# Check if Chroma DB already exists, if not, create it
if not os.path.exists(persist_directory):
    documents = []

    # Create a list of documents from all the files in the ./docs folder
    for file in os.listdir("docs"):
        file_path = os.path.join("docs", file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".docx") or file.endswith(".doc"):
            loader = Docx2txtLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        documents.extend(loader.load())

    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)

    # Create a Chroma vector store from documents
    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)

    # Persist the vector store to disk
    vectordb.persist()

else:
    # Load the existing Chroma DB from the persisted data
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

# Create our Conversational Retrieval QA chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []

print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')

# Start conversation loop
while True:
    query = input(f"{green}Prompt: ")
    if query in {"exit", "quit", "q", "f"}:
        print('Exiting')
        sys.exit()
    
    if query.strip() == '':
        continue
    
    # Run query against the QA chain
    result = pdf_qa.invoke({"question": query, "chat_history": chat_history})

    # Print the answer from the model
    print(f"{white}Answer: " + result["answer"])

    # Save chat history
    chat_history.append((query, result["answer"]))
