import streamlit as st
import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Function to load chat history from file
def load_chat_history():
    try:
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Function to save chat history to file
def save_chat_history(messages):
    with open('chat_history.json', 'w') as f:
        json.dump(messages, f)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Sidebar configuration
with st.sidebar:
    # Add clear history button
    if st.button('Clear Chat History'):
        st.session_state.chat_history = []
        save_chat_history([])
        st.success('Chat history cleared!')

    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Define the chat prompt template
        prompt = ChatPromptTemplate.from_template(
            """
               Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas industry, with a particular focus on content related to the Basrah Gas Company (BGC). Your responses must primarily rely on the PDF files uploaded by the user, which contain information specific to the oil and gas sector and BGC's operational procedures.
When providing responses, you must:

Structure Your Answer:

Begin with a clear, direct answer to the question
Support each point with relevant quotes from the PDFs
After each quote, cite the specific page number in format: [Page X]
List all referenced pages at the end of your response


Source Attribution Format:
CopyAnswer: [Your main response point]

Supporting Evidence:
"[Exact quote from PDF]" [Page X]

Additional Context:
"[Related quote from another section]" [Page Y]

Referenced Pages:
- Page X: [Brief description of content]
- Page Y: [Brief description of content]

Information Integration:

When information spans multiple pages:

Quote relevant sections from each page
Explain how the information connects
Provide a synthesized conclusion


Always indicate which specific parts come from which pages


When Direct Information is Not Available:

Clearly state that the specific information is not found in the documents
Provide logical reasoning based on available related content
Reference any partial matches or related information from the PDFs



Guidelines:

Primary Source Referencing:

Always reference specific page numbers
Include direct quotes as evidence
Integrate partial information with clear reasoning


Logical Reasoning:

Use internal knowledge only when PDFs lack direct answers
Explicitly mark reasoning-based responses
Connect to relevant PDF content when possible


Visual Representation:

Create visuals based only on PDF content
Include page references for visual information
Ensure accuracy in representations


Restricted Data Usage:

Use only uploaded PDF content
Avoid external sources
Rely on internal reasoning when needed


Professional and Contextual Responses:

Maintain oil and gas industry focus
Tailor to BGC context
Keep professional tone


Multilingual Support:

Match user's language choice
Provide responses in Arabic or English as appropriate
Maintain technical accuracy in both languages



Expected Output Format:
CopyMain Answer:
[Clear response to the question]

Evidence:
"[Direct quote]" [Page X]
"[Supporting quote]" [Page Y]

Context Connection:
[Explanation of how quotes relate]

Referenced Pages:
1. Page X - [Content description]
2. Page Y - [Content description]

[Logical conclusion if needed]
Remember:

Every significant point must be supported by specific page references
Include direct quotes when possible
List all referenced pages at the end of each response
Clearly mark any information derived from reasoning rather than direct quotes

Thank you for your accuracy, professionalism, and commitment to providing exceptional assistance tailored to the Basrah Gas Company and the oil and gas industry.
            {context}
            </context>
            Question: {input}
            Previous conversation context: {chat_history}
            """
           
        )

        # Load existing embeddings from files
        if "vectors" not in st.session_state:
            with st.spinner("Loading embeddings... Please wait."):
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )

                try:
                    st.session_state.vectors = FAISS.load_local(
                        "embeddings",
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    st.sidebar.write("Embeddings loaded successfully :partying_face:")
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None

    else:
        st.error("Please enter both API keys to proceed.")

# Main area for chat interface
st.title("Mohammed Al-Yaseen | BGC ChatBot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user queries
if human_input := st.chat_input("Ask something about the document"):
    # Format chat history for context
    chat_history_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in st.session_state.chat_history[-5:]  # Include last 5 messages for context
    ])
    
    # Add timestamp to the message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_message = {
        "role": "user",
        "content": human_input,
        "timestamp": timestamp
    }
    
    st.session_state.chat_history.append(user_message)
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Create and configure the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response from the assistant
        response = retrieval_chain.invoke({
            "input": human_input,
            "chat_history": chat_history_text
        })
        assistant_response = response["answer"]

        # Add timestamp to assistant's response
        assistant_message = {
            "role": "assistant",
            "content": assistant_response,
            "timestamp": timestamp
        }
        
        st.session_state.chat_history.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Save updated chat history
        save_chat_history(st.session_state.chat_history)

        # Display supporting information from documents
        with st.expander("Supporting Information"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        error_message = "Error: Unable to load embeddings. Please check the embeddings folder and ensure the files are correct."
        error_message_with_timestamp = {
            "role": "assistant",
            "content": error_message,
            "timestamp": timestamp
        }
        st.session_state.chat_history.append(error_message_with_timestamp)
        with st.chat_message("assistant"):
            st.markdown(error_message)
        save_chat_history(st.session_state.chat_history)
Last edited just now
