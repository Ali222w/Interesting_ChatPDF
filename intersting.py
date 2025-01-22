import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from typing import List
from langchain_core.documents import Document
import re

groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

def extract_number(text):
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else None

def get_page_numbers(docs: List[Document]) -> List[int]:
    page_numbers = []
    for doc in docs:
        if hasattr(doc.metadata, 'page') and doc.metadata['page'] is not None:
            page_numbers.append(doc.metadata['page'])
    return sorted(list(set(page_numbers)))  # Remove duplicates and sort

with st.sidebar:
    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        prompt = ChatPromptTemplate.from_template("""
Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas industry, with a particular focus on content related to the Basrah Gas Company (BGC). 
Your responses must primarily rely on the PDF files uploaded by the user, which contain information specific to the oil and gas sector and BGC's operational procedures. 
If a specific answer cannot be directly found in the PDFs, you are permitted to provide a logical and well-reasoned response based on your internal knowledge. 
Under no circumstances should you use or rely on information from external sources, including the internet.

Guidelines:
1. **Primary Source Referencing:**
- Always reference the specific page number(s) in the uploaded PDFs where relevant information is found. 
The page numbers for this response can be found in this list: {page_numbers}
Please include these page numbers in your response when citing information.
If the PDFs contain partial or related information, integrate it with logical reasoning to provide a comprehensive response. 
Clearly distinguish between PDF-derived content and logical extrapolations to ensure transparency.

2. **Logical Reasoning:**
- When specific answers are unavailable in the PDFs, use your internal knowledge to provide logical, industry-relevant responses. 
Explicitly state when your response is based on reasoning rather than the uploaded materials.

3. **Visual Representation:**
- When users request visual representations (e.g., diagrams, charts, or illustrations), create accurate and relevant visuals based on the uploaded PDF content and logical reasoning. 
Ensure the visuals align precisely with the context provided and are helpful for understanding the topic.
Provide an appropriate photo or diagram in the response if needed to enhance understanding, even if the user does not explicitly request it.

4. **Restricted Data Usage:**
- Avoid using or assuming information from external sources, including the internet or any pre-existing external knowledge that falls outside the uploaded materials or your internal logical reasoning.

5. **Professional and Contextual Responses:**
- Ensure responses remain professional, accurate, and relevant to the oil and gas industry, with particular tailoring for Basrah Gas Company. 
Maintain a helpful, respectful, and clear tone throughout your interactions.

6. **Multilingual Support:**
- Detect the language of the user's input (Arabic or English) and respond in the same language. 
If the input is in Arabic, provide the response in Arabic. If the input is in English, provide the response in English.

Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions: {input}
""")
        
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
            
        if "vectors" not in st.session_state:
            with st.spinner("Loading embeddings... Please wait."):
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )
                embeddings_path = "embeddings"
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    st.sidebar.write("Embeddings loaded successfully :partying_face:")
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None
    else:
        st.error("Please enter both API keys to proceed.")

st.title("Mohammed Al-Yaseen | BGC ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if human_input := st.chat_input("Ask something about the document"):
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        retriever = st.session_state.vectors.as_retriever()
        # First get relevant documents to extract page numbers
        relevant_docs = retriever.get_relevant_documents(human_input)
        page_numbers = get_page_numbers(relevant_docs)
        
        # Create the document chain with the page numbers included in the prompt
        document_chain = create_stuff_documents_chain(
            llm,
            prompt.partial(page_numbers=page_numbers)
        )
        
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            "input": human_input,
            "history": st.session_state.memory.chat_memory.messages
        })
        
        assistant_response = response["answer"]
        
        st.session_state.memory.chat_memory.add_user_message(human_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        with st.expander("Supporting Information"):
            for i, doc in enumerate(response["context"]):
                st.write(f"Page {doc.metadata.get('page', 'Unknown')}")
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        assistant_response = "Error: Unable to load embeddings. Please check the embeddings folder."
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
