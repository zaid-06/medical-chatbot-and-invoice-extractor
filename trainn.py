# preprocess.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load PDF and split text
loader = PyPDFLoader("Medical_book.pdf")
text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=250, 
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)   
pages = loader.load_and_split(text_splitter)

# Create vector database
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
persist_directory = "D:\AI-with-langchain\gemini\Models\MedicalModel"
vectordb = Chroma.from_documents(pages, embeddings, persist_directory=persist_directory)
print("Data saved")
