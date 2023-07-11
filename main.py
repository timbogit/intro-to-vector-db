import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
from dotenv import load_dotenv

import pinecone

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)

if __name__ == "__main__":
    print("Hello VectorStore!")

    loader = TextLoader(
        "/Users/tim/Documents/workspace/intro-to-vector-db/mediumblogs/mediumblog1.txt"
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(),
        vectorstore=docsearch,
        chain_type="stuff",
        return_source_documents=True,
    )
    query = "What is a vector databases? Give me a 15 word answer for a beginner."
    result = qa({"query": query})

    print(result)
