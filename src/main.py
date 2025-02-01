import os
import csv
import pandas as pd
import gradio as gr
from retriever import Retriever
from gradio import Interface
from pinecone import Pinecone
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from pdf_loader import load_pdfs
from langchain.agents import initialize_agent
from dotenv import load_dotenv

chat_history = []

retriever = Retriever() 

def handle_query(query, retriever):
    """
    Handles the user's query by detecting the namespace, retrieving relevant chunks, 
    and generating an answer.
    
    Args:
        query (str): The user's question.
        retriever (Retriever): The retriever instance.
    Returns:
        list: The updated chat history in the required format.
    """
     # Initialize retriever functions to make them accesable for the agent
    retriever = Retriever()

    # Embed the query and retrieve relevant document chunks from the detected namespaces
    relevant_results = retriever.retrieve_relevant_chunks(query)

    if not relevant_results:
        answer = f"Es gibt zu Ihrer Frage in den Wahlprogrammen oder auf ihrem YouTube-Kanal keine Positionierung dazu."
    else:
        # Prepare context with traceable metadata
        context = []
        for result in relevant_results:
            filename = result["filename"]
            page_number = result["page_number"]
            score = result["score"]
            text = result["metadata"].get("text", "[Kein Text verfügbar]")  # Fallback if text is missing

            # Add traceable metadata to the context
            context.append(
                f"Quelle: {filename}, Seite: {page_number}, Score: {score:.2f}\n{text}"
            )

        # Join context into a single string
        context_str = "\n\n".join(context)

        print(f"Generated context for LLM:\n{context_str}")
        print(f"Query: {query}")

        # Calling the election-agent
        answer = election_agent(query, retriever)

    return chat_history

def election_agent(query, retriever): 
    
    # Initialize retriever functions to make them accesable for the agent
    retriever = Retriever()
    # Initialize the OpenAI Chat model      
    llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0)

    # setting up the memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=10,
        return_messages=True,
        max_token_limit=1000
    )

    # providing tool kit for the agent
    tools = []

    tools = [
            Tool(
                    name='election Knowledge Base',
                    func=retriever.retrieve_relevant_chunks,
                    description=(
                        'use this tool when answering specific question about the German elections in 2025'
                        'always insist to just answer questions about the elections and the positioning of the partiest about preceise topics'
                        'here you get precise information, that you have to use as your knowledge about the election porgramms'
                        'this is the base what the parties are standing for and have planed for the future'
                        'If the user is asking in German, alwyas answer in German'
                        'IF the user is asking in English, always answer in English' 
                                ),
                    )
    ]
    
    # initialize the agent and setting it up
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        memory=conversational_memory,
        max_iterations=3,
        early_stopping_method='generate',
        agent_kwargs={
                    # "system_message": system_prompt,
                    "memory_key": "chat_history"
                    }
                            )
    
    # Get the agent's response
    print(f"now starts the agent:... thinking")
    response = str(agent.invoke({"input": query}).get("output", "I didn't understand that. Please try again."))
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": response})
    print(f"chat_history with response {chat_history} and {type(chat_history)}")
    print(f" 🤖  chat_history - in agent_function: {chat_history} and type: {type(chat_history)} ")
    print(f" 🤖  response - in agent_function: {response} and type: {type(response)} ")



    return response # ✅ Return as string                            

def is_pdf_processing_needed():
    """
    Check if new PDFs need processing by comparing existing files with the processed log.
    
    Returns:
        bool: True if new PDFs are found, True if everything is already processed.
    """
    pdf_files = {f for f in os.listdir("data") if f.endswith(".pdf")}
    
    if not os.path.exists("processed_election_pdfs.csv"):
        print("[INFO] No processed PDF record found. Processing all PDFs...")
        return True  # No CSV found, process all PDFs

    processed_df = pd.read_csv("processed_election_pdfs.csv")
    processed_files = set(processed_df["filename"])

    # Check if any new PDFs exist
    new_pdfs = pdf_files - processed_files
    if new_pdfs:
        print(f"[INFO] New PDFs detected: {new_pdfs}")
        return True  # Process new PDFs
    else:
        print("[INFO] All PDFs are already processed. Skipping PDF processing.")
        return False  # No new PDFs
    
def launch_ui():

    with gr.Blocks() as interface:
            gr.Markdown("# 🗳️ Election Chatbot Smart-Vote-Consultant")
            gr.Markdown("Welcome! Ask questions about political party policies and get to know how it affects you for real.")
            gr.Markdown("For the following parties I can provide answers out of their elections programm and youtube published video content:")
            
            # ✅ Row of images (Party Icons)
            with gr.Row():
                gr.Image("icons/afd.png", width=50, height=50, interactive=False, container=False, show_download_button=False, show_fullscreen_button=False)
                gr.Image("icons/bsw.png", width=50, height=50, interactive=False, container=False, show_download_button=False, show_fullscreen_button=False)
                gr.Image("icons/cdu.png", width=50, height=50, interactive=False, container=False, show_download_button=False, show_fullscreen_button=False)
                gr.Image("icons/fdp.png", width=50, height=50, interactive=False, container=False, show_download_button=False, show_fullscreen_button=False)
                gr.Image("icons/gruene.png", width=50, height=50, interactive=False, container=False, show_download_button=False, show_fullscreen_button=False)
                gr.Image("icons/linke.png", width=50, height=50, interactive=False, container=False, show_download_button=False, show_fullscreen_button=False)
                gr.Image("icons/piraten.png", width=50, height=50, interactive=False, container=False, show_download_button=False, show_fullscreen_button=False)
                gr.Image("icons/spd.png", width=50, height=50, interactive=False, container=False, show_download_button=False, show_fullscreen_button=False)
                gr.Image("icons/volt.png", width=50, height=50, interactive=False, container=False, show_download_button=False, show_fullscreen_button=False)

                # Chatbot interface (✅ Now inside `gr.Blocks()`)
            chatbot = gr.Chatbot(label="Question History", type='messages')
            print(f"Type chatbot: {chatbot}")

            with gr.Row():
                query = gr.Textbox(
                    label="What do you need to know about German parties’ stances for the Bundestagswahl on 23.02.2025?",
                    placeholder="Type your question here and name always a party for best results... something like:\n"
                                "... Wie stehen die Grünen und die SPD zum Mindestlohn? \n"
                                "... Welche Chance haben Bindertekinder auf Inklusion unter der AFD? \n",
                    lines=10,
                    min_width=50,
                    interactive=True,

                )

            with gr.Row():
                submit_btn = gr.Button("Submit")
                # clear_button = gr.Button("Clear Chat")

            # Connect the buttons to functions (✅ Now inside `gr.Blocks()`)
            # query.submit(handle_query, inputs=query, outputs=chatbot)
            submit_btn.click(lambda q: handle_query(q, retriever), inputs=[query], outputs=[chatbot])
            # clear_button.click([], inputs=None, outputs=[chatbot])  # ✅ Fix clear function

    # Launch the app
    interface.launch(share=True)


def main():
    """
    Initializes the election chatbot, processes PDFs and YouTube videos,
    and starts the Gradio UI.
    """
    from YT_loader import load_yt_videos
    from YT_loader import process_party_videos
    print("\n")
    print("🚀 Initializing Election Consultant... \n ")

    # Load environment variables for all used keys
    load_dotenv()

    # Initialize Langsmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "Election Consultant MVP"
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

    # Ensure retriever is properly initialized
    retriever = Retriever()

    # Step 1: Process PDFs if necessary
    if is_pdf_processing_needed():
        print("📄 Processing new PDFs...")
        pdf_documents = load_pdfs()  
        pdf_chunks = retriever.process_documents_to_chunks(pdf_documents)  
        retriever.load_embed_and_index_documents(pdf_chunks)  
    else:
        print("📄 PDF dataset is up to date.")

    # Step 2: Process YouTube Videos
    print("🎥 Checking YouTube party channel...")
    video_documents = load_yt_videos()
    video_chunks = retriever.process_documents_to_chunks(video_documents)  
    retriever.load_embed_and_index_documents(video_chunks)  

    print("🚀 Data processing and embedding complete!")

    # Launch the Gradio UI and call agent
    launch_ui()

if __name__ == "__main__":
    main()