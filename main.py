__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from dotenv import load_dotenv
# load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

button(username="Seoyeon", floating=True, width=221)

# 제목
st.title("ChatPDF")
st.write("---")

#OpenAI KEY 입력 받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()  # 임시 디렉토리를 만들어서 메모리에 저장함
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages



# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, # 몇 글자 단위로 쪼갤 것인지
        chunk_overlap=200, # 문맥을 위해 앞뒤로 중복을 포함할 문자 수
        length_function=len,  
        is_separator_regex=False,  # 정규 표현식으로 자를지 여부
    )

    texts = text_splitter.split_documents(pages)

    # create the open-source embedding function
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_key)  

    # load it into Chroma (text to embedding model)
    db = Chroma.from_documents(texts, embedding_function)

    #Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('PDF에게 질문하기'):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500, openai_api_key=openai_key)  # 라마 교체 가능
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])