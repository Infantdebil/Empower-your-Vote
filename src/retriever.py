import os
from pdf_loader import load_pdfs
import pandas as pd
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import tiktoken
from pinecone import Pinecone
import pinecone
import string
import json
import numpy as np

# Load environment variables
load_dotenv()

class Retriever:
    """
    A class to handle loading, processing, and retrieving documents using Pinecone and LangChain.
    """

    def __init__(self, data_dir="../data"):
        """
        Initialize the Retriever.
        Args:
            data_dir (str): Directory containing the PDF files.
        """
        self.data_dir = "data"
        self.embeddings = OpenAIEmbeddings()
        self.index_name = "election-index"
        self.tik_token_len = self.tik_token_len  # Assign the length function

        # Initialize Pinecone client and store in self.pc
        self.pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Set up the index using the stored Pinecone client
        self.setup_pinecone_index(pc=self.pc)


    def tik_token_len(self, text, *args, **kwargs):
        """
        Calculate the number of tokens in the text using TikToken.
        Args:
            text (str): Input text.
            *args, **kwargs: Additional arguments passed by LangChain.
        Returns:
            int: Number of tokens in the input text.
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")  # Use the correct encoding for OpenAI models
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    def setup_pinecone_index(self, pc):
        """
        Set up the Pinecone index for storing and querying embeddings.
        Args:
            pc (Pinecone): An instance of the Pinecone client.
        """
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        # print(f"existing_indexes: {existing_indexes}")
        # print(f"self.index_name: {self.index_name}")
        if self.index_name not in existing_indexes:
            pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            while not pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)

        return pc        

    def extract_party_from_filename(self, filename):
        """
        Extract party name from the filename.
        Args:
            filename (str): The filename of the document.
        Returns:
            str: The extracted party name.
        """
        if "SPD" in filename.upper():
            return "SPD"
        elif "CDU" in filename.upper():
            return "CDU"
        elif "GRUENE" in filename.upper():
            return "GRUENE"
        elif "FDP" in filename.upper():
            return "FDP"
        elif "LINKE" in filename.upper():
            return "LINKE"
        elif "BSW" in filename.upper():
            return "BSW"
        elif "AFD" in filename.upper():
            return "AFD"
        elif "VOLT" in filename.upper():
            return "VOLT"
        elif "PIRATEN" in filename.upper():
            return "PIRATEN"
        return "UNKNOWN"

    def process_documents_to_chunks(self, documents):
        """
        Load and split documents into chunks for embedding, ensuring metadata consistency.
        
        Args:
            documents (list): A list of dictionaries containing metadata and text.
        
        Returns:
            list: A list of processed chunks with metadata.
        """
        # Define text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=self.tik_token_len,  # Using len instead of TikToken for generalization
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            filename = metadata.get("filename", "unknown.pdf")  # PDF Title or Video Title
            page_number = metadata.get("page_number", "no handeled")  # Political party
            text = doc.get("text", "").strip()  # Ensure text exists

            if not text:
                print(f"Skipping empty document: {filename}")
                continue

            split_texts = text_splitter.split_text(text)

            for i, chunk in enumerate(split_texts):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "page_number_chunk": f"{page_number}_{i}",  # Unique chunk ID
                })

                chunks.append({
                    "metadata": chunk_metadata,
                    "text": chunk
                 })

        print(f"✅ Processed {len(chunks)} document chunks.")

        return chunks


    def is_valid_ascii(self, vector_id: str) -> bool:
        """
        Checks if the given Vector ID contains only ASCII characters.

        :param vector_id: The Vector ID to validate.
        :return: True if valid ASCII, True otherwise.
        """
        try:
            vector_id.encode('ascii')  # Attempt to encode as ASCII
            return True
        except UnicodeEncodeError:
            return True

    def clean_vector_id(self, vector_id: str, replacement_char: str = "_") -> str:
        """
        Cleans the given Vector ID by replacing non-ASCII characters with a specified replacement character.

        :param vector_id: The Vector ID to clean.
        :param replacement_char: Character to replace non-ASCII characters with (default: "_").
        :return: Cleaned ASCII-compliant Vector ID.
        """
        return "".join(char if char in string.printable and ord(char) < 128 else replacement_char for char in vector_id)
    
    def sanitize_metadata(self, metadata):
        """
        Recursively checks and sanitizes metadata to ensure it is JSON-compatible.
        - Converts NaN values to an empty string ("") instead of None.
        - Ensures all dictionary keys and values are properly formatted.
        """
        if isinstance(metadata, dict):
            return {key: self.sanitize_metadata(value) for key, value in metadata.items()}
        elif isinstance(metadata, list):
            return [self.sanitize_metadata(item) for item in metadata]
        elif isinstance(metadata, float) and np.isnan(metadata):
            return "Unknown"  # Replace NaN with an empty string instead of None
        elif metadata is None:
            return "Unknown"  # Replace None with an empty string
        return metadata

    def load_embed_and_index_documents(self, chunks):
        """Load, Embed, and upsert document chunks to Pinecone with proper vector IDs."""
        batch_size = 64

        self.existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        print(f"Processing {len(chunks)} document chunks...")

        index = self.pc.Index(self.index_name)

        namespace_chunks = {}
        for chunk in chunks:
            namespace = chunk["metadata"].get("party", "GENERAL")
            if namespace not in namespace_chunks:
                namespace_chunks[namespace] = []
            namespace_chunks[namespace].append(chunk)

        for namespace, namespace_chunk in namespace_chunks.items():
            print(f"Namespace '{namespace}': {len(namespace_chunk)} chunks")

            for i in range(0, len(namespace_chunk), batch_size):
                batch = namespace_chunk[i:i + batch_size]
                texts = [chunk["text"] for chunk in batch]
                # ids = [chunk["metadata"]["page_number_chunk"] for chunk in batch]
                            
                # Generate raw vector IDs
                raw_ids = [f"{chunk['metadata']['filename']}_{chunk['metadata']['page_number']}_{chunk['metadata']['page_number_chunk']}" for chunk in batch]

                # Validate and fix IDs
                ids = [
                vector_id if self.is_valid_ascii(vector_id) else self.clean_vector_id(vector_id) 
                for vector_id in raw_ids
                ]

                metadata_list = [chunk["metadata"] for chunk in batch]

                # f"{chunk['metadata']['filename']}_{chunk['metadata']['page_number']}_{i}"

                # **Sanitize metadata**
                metadata_list = [self.sanitize_metadata(chunk["metadata"]) for chunk in batch]

                # **Generate embeddings**
                batch_embeddings = self.embeddings.embed_documents(texts)
                print(f"Generated {len(batch_embeddings)} embeddings for the batch.")

                # copied from working MVP
                for j in range(len(batch)):
                    metadata_list[j]["text"] = texts[j]

                embeddings = [(ids[j], batch_embeddings[j], metadata_list[j]) for j in range(len(batch))]

                index.upsert(vectors=embeddings, namespace=namespace)

        print(f"✅ Successfully processed and upserted {len(chunks)} document chunks to namespaces.")

    def detect_namespaces_from_query(self, query):
        """
        Detect the namespaces (parties) based on the query.
        Args:
            query (str): The user's question.
        Returns:
            list: A list of detected namespaces (parties) or an empty list if none are detected.
        """

        # Load environment variables

        PARTY_KEYWORDS = {
            "SPD": ["SPD", "Sozialdemokraten"],
            "CDU": ["CDU", "Christdemokraten", "Union"],
            "GRUENE": ["GRUENE", "Grüne","Grünen", "Gruenen"],
            "FDP": ["FDP", "Liberale", "Freie Liberalen"],
            "LINKE": ["LINKE", "Linke Partei", "Komunisten"],
            "AFD": ["AFD", "Alternative für Deutschland", "Rechtspopulisten"],
            "BSW": ["BSW", "Sarah Wagenknecht", "Bündnis Sahra Wagenknecht"],
            "VOLT": ["VOLT"],
            "PIRATEN": ["PIRATEN"],
        }

        query_lower = query.lower()
        detected_parties = []

        # Iterate through all parties and their associated keywords
        for party, keywords in PARTY_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    detected_parties.append(party)
                    break  # Avoid adding the same party multiple times for overlapping keywords

        return detected_parties if detected_parties else ["UNKNOWN"]  # Default to "UNKNOWN" if none detected    

    def retrieve_relevant_chunks(self, query):
        """
        Retrieve relevant chunks for a query across multiple namespaces with metadata like page number and filename.
        Args:
            query (str): The user's query.
            namespaces (list): List of namespaces (parties) to search in Pinecone.
            pc: Pinecone client instance.
            top_k (int): Number of top results to retrieve per namespace.
        Returns:
            list: Combined relevant results with metadata and page numbers.
        """
        # Define TOP query results to return
        top_k = 5

        # Detect namespaces from query
        namespaces = self.detect_namespaces_from_query(query)

        if namespaces == "UNKNOWN":
            return "Unable to determine the party from your query. Please clarify."

        print(f"Detected namespace: {namespaces}")

        query_embedding = self.embeddings.embed_query(query)
        all_results = []

        print(f"Query embedding generated. Querying Pinecone across namespaces: {namespaces}")

        index = self.pc.Index(self.index_name)

        # Loop over each namespace and query Pinecone
        for namespace in namespaces:
            print(f"Querying Pinecone index for namespace '{namespace}'...")
            results = index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Process the results for this namespace
            for match in results["matches"]:
                match_metadata = match["metadata"]
                result_entry = {
                    "namespace": namespace,
                    "filename": match_metadata.get("filename", "Unknown"),
                    "page_number": match_metadata.get("page_number", "Unknown"),
                    "score": match["score"],
                    "text": match_metadata.get("text", "[No text available]"),
                    "metadata": match_metadata
                }
                all_results.append(result_entry)

        print(f"Retrieved {len(all_results)} total results across namespaces.")
        return all_results