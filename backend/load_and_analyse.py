import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

DATA_DIR = os.path.join(os.path.pardir, 'data', 'documents')

def load_documents_and_print_stats():
    """Load documents from the specified directory and print statistics."""
    # 1. Initialise SimpleDirectoryReader
    print(f"Loading documents from directory: {DATA_DIR}\n")
    reader = SimpleDirectoryReader(DATA_DIR)

    # 2. Load documents
    documents = reader.load_data()

    # 3. Print statistics
    print("--- Document Loading Statistics ---")
    print(f"Number of documents loaded: {len(documents)}\n")

    # 4. Analyse and Print Individual Document Stats
    total_characters = 0

    for doc in documents:
        file_name = doc.metadata.get('file_path', 'N/A').split(os.sep)[-1]
        char_count = len(doc.text)
        total_characters += char_count

        print(f"Document: {file_name}")
        print(f" - Number of characters: {char_count:,}")
        print("-" * 40)

    print(f"Total number of characters across all documents: {total_characters:,}\n")
    print("------------------------------------")

    return documents

if __name__ == "__main__":
    loaded_documents = load_documents_and_print_stats()
    
