# Election Program Chatbot

## Overview
This project provides an interactive chatbot that helps users easily access election program information from major German parties. It uses **Retriever-Augmented Generation (RAG)** to deliver direct and clear answers from available sources.

## Features
- **Easy-to-use interface**: Initially built as a web application (Gradio MVP), with future expansion to mobile versions.
- **Direct answers, not just search results**: Uses **RAG** to extract relevant information from election programs.
- **Supports major parties**: Includes **CDU, SPD, Greens, Die Linke, AfD, BSW, Die Piraten, Volt, and FDP**.
- **Multiple data sources**:
  - **PDF documents** (e.g., official election programs)
  - **YouTube transcripts** (for speeches and campaign materials)
  - **Future expansion**: Podcasts
- **Efficient vector search**:
  - **Pinecone** (default) for scalable cloud-based retrieval
- **LLM-powered responses**:
  - Uses **GPT-4o-mini API** for intelligent and multilingual answers (German/English).
  - Optimized prompt engineering for precise output.

## Setup Instructions

### **1. Install Dependencies**
Run the following command to install the required libraries:

```sh
pip install -r requirements.txt
```

### **2. Set Up API Keys**
The application requires API keys for accessing external services. Store them in a `.env` file:

```
YOUTUBE_API_KEY=your_youtube_api_key
OPENAI_API_KEY=your_openai_api_key
```
For Pinecone vector storage, set up:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```

### **3. Run the Application**
Launch the chatbot with:

```sh
python main.py
```

## File Structure

### **1. PDF Loader (`pdf_loader.py`)**
- Extracts text from PDF documents.
- Used to process official election programs.

### **2. YouTube Loader (`YT_loader.py`)**
- Fetches transcripts from political videos.
- Uses YouTube API for metadata extraction.

### **3. Retriever (`retriever.py`)**
- Performs vector-based search on stored election data.
- Uses **Pinecone** for efficient retrieval.

### **4. Main File (`main.py`)**
- Integrates all components into a functional chatbot.
- Calls retrievers and loaders as needed.

## Future Improvements
- Expand to **mobile and voice-based interaction**.
- Include **Twitter and blog content creation** for real-time political insights.
- agent improvement **asking for personal request**, **political characters for each party**

## Contribution
Contributions are welcome! Feel free to submit issues or pull requests.

---
**Author**: Paul Müller
