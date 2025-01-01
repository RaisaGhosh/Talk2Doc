import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from openai import OpenAI
from langchain.chains.question_answering import load_qa_chain


#Sidebar
with st.sidebar:
    st.title('CHAT WITH PDFs')
    st.markdown('''
    ## About
    This is a LLM Powered Application which helps you have a conversation with the documents you upload.\n
    Built using:\n
    -Langchain\n
    -HuggingFace\n
    -Streamlit\n
    -LM Studio with Llama 3.2-1b Instruct\n
    
    ''')
    add_vertical_space(5)
    st.write('Built using RAG')


def main():
    # st.write("Hi")
    st.header('Talk2Doc')

    #uploading a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    #for a valid pdf uploaded
    if pdf is not None:
        pdfReader = PdfReader(pdf)

        text = ""
        for page in pdfReader.pages:
            text +=page.extract_text()

        # st.write(text)

        #chunking
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = textSplitter.split_text(text=text)
        # st.write(chunks)/
        
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        
        vectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)

        #Accept User QUery
        query = st.text_input("Ask questions about pdf file....")
        # st.write(query)

        if query:
            docs = vectorStore.similarity_search(query=query,k=3)
            # st.write(docs)

            context = ""

            for d in docs:
                context += d.page_content+" "

            #Feeding docs as context to LLM
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

            completion = client.chat.completions.create(
            model="model-identifier",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            )

            # print(completion.choices[0].message.content)
            st.write(completion.choices[0].message.content)





if __name__ == '__main__':
    main()