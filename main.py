import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import httpx
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.chains.question_answering import load_qa_chain


load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


def update_base_url(request: httpx.Request) -> None:
    if request.url.path == "/chat/completions":
        request.url = request.url.copy_with(path="/v1/chat")
    elif request.url.path == "/embeddings":
        request.url = request.url.copy_with(path="/v1/openai/ada-002/embeddings")


# Initialize HTTP client with event hook to update the base URL
http_client = httpx.Client(
    event_hooks={"request": [update_base_url]}
)


def main():
    st.header("Chat with PDF")
    pdf = st.file_uploader('Upload your PDF', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)


        embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        st.write('Embeddings created')
        # Accept the user query
        query = st.text_input("Ask questions about your PDF file:")
        #st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            # Create the ChatGPT model instance
            llm = ChatOpenAI(
                base_url="https://aalto-openai-apigw.azure-api.net",
                api_key=openai_api_key,
                default_headers={
                    "Ocp-Apim-Subscription-Key": openai_api_key,
                },
                http_client=http_client
            )
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__ == '__main__':
    main()