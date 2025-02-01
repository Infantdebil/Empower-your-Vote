import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from datetime import datetime
 
def extract_metadata_with_pypdf2(pdf_path):
    """
    Extract metadata such as creation date and modification date from a PDF using PyPDF2.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        dict: Dictionary containing metadata like creation date, modification date, and author.
    """
    reader = PdfReader(pdf_path)
    metadata = reader.metadata

    # Parse and format dates if they exist
    def format_date(date_str):
        try:
            # Extract the date portion from strings like "D:20250111175410+01'00'"
            raw_date = date_str[2:10]  # Extract "20250111"
            formatted_date = datetime.strptime(raw_date, "%Y%m%d").date()  # Convert to YYYY-MM-DD
            return str(formatted_date)
        except Exception:
            return "Unknown"  # Default if parsing fails

    return {
        "creation_date": format_date(metadata.get("/CreationDate", "Unknown")),
        "author": metadata.get("/Author", "Unknown"),
    }

def extract_party_from_filename(filename):
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

def load_pdfs():
    """
    Load PDFs using PyPDFLoader and add metadata (including creation date).
    
    Returns:
        list: A list of dictionaries (instead of LangChain documents) with metadata.
    """
    documents = []
    
    # Define the fixed path to the data folder
    data_folder = os.path.abspath("data/pdf")  
    os.makedirs(data_folder, exist_ok=True)  # Ensure the folder exists

    # Define the processed PDFs tracking file inside `data/`
    processed_pdfs_file = os.path.join(data_folder, "processed_election_pdfs.csv")

    # Load already processed PDFs if file exists
    if os.path.exists(processed_pdfs_file):
        processed_pdfs_df = pd.read_csv(processed_pdfs_file)
        processed_pdfs_set = set(processed_pdfs_df["filename"])  # Set for fast lookup
    else:
        processed_pdfs_set = set()

    new_pdfs_data = []  # To store new processed PDF info

    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)

            # Skip if already processed
            if filename in processed_pdfs_set:
                print(f"Skipping {filename}, already processed.")
                continue  

            # Extract metadata
            metadata = extract_metadata_with_pypdf2(pdf_path)
            party = extract_party_from_filename(filename)

            # Load content with PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            pages_count = 0

            for i, doc in enumerate(loader.load()):
                # Ensure correct page number is assigned
                page_number = doc.metadata.get("page_number", i + 1)  # Default to index if missing
    
                documents.append({
                    "metadata": {
                        "filename": filename,  
                        "page_number": page_number,  # Correct page number
                        "creation_date": metadata.get("creation_date", "unknown"),
                        "author": metadata.get("author", "unknown"),
                        "party": party,
                        "description": "PDF Document",  
                        "source": filename,  # File name instead of URL
                    },
                    "text": doc.page_content  # Store text under "text" key
                })

                pages_count += 1   
            
            print(f"Loading: {filename} with {pages_count} new pages is done.")
        
            # Store processed PDF info
            new_pdfs_data.append({
                "filename": filename,
                "author": metadata.get("author", "Unknown"),
                "creation_date": metadata.get("creation_date", "Unknown"),
                "amount_of_pages": pages_count,
                "party": party
            })

    # Append new data to the processed PDFs file inside `data/`
    if new_pdfs_data:
        new_pdfs_df = pd.DataFrame(new_pdfs_data)
        if os.path.exists(processed_pdfs_file):
            new_pdfs_df.to_csv(processed_pdfs_file, mode='a', header=True, index=True)
        else:
            new_pdfs_df.to_csv(processed_pdfs_file, index=True)

    return documents
