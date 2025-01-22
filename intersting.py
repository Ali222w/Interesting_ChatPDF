 import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Sidebar configuration
with st.sidebar:
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
            """
        )

        # Load existing embeddings from files
        if "vectors" not in st.session_state:
            with st.spinner("Loading embeddings... Please wait."):
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )

                # Load existing FAISS index with safe deserialization
                embeddings_path = "embeddings"  # Path to your embeddings folder
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True  # Only use if you trust the source of the embeddings
                    )
                    st.sidebar.write("Embeddings loaded successfully :partying_face:")
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None

    else:
        st.error("Please enter both API keys to proceed.")

# Main area for chat interface
st.title("Mohammed Al-Yaseen | BGC ChatBot")

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user queries
if human_input := st.chat_input("Ask something about the document"):
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Create and configure the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response from the assistant
        response = retrieval_chain.invoke({"input": human_input})
        assistant_response = response["answer"]

        # Append and display assistant's response
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Display supporting information from documents
        with st.expander("Supporting Information"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        # Error message if vectors aren't loaded
        assistant_response = (
            "Error: Unable to load embeddings. Please check the embeddings folder and ensure the files are correct."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
