import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.load_and_analyse import load_documents_and_print_stats
from app.chunking import SectionAwareChunker

@pytest.fixture(scope="module")
def documents():
    """
    Fixture to load documents once for all tests in this module.
    """
    print("\n--- Loading Documents (Fixture) ---")

    if not os.path.exists("../data/documents"):
        pytest.skip("Data directory 'data/documents' does not exist. Skipping tests.")

    return load_documents_and_print_stats()

def test_node_creation_count(documents):
    """
    Verify that the chunker is actually splitting documents.
    """
    chunker = SectionAwareChunker()
    nodes = chunker.get_nodes_from_documents(documents)

    # Expect more than 60 pages due to splitting
    print(f"Total nodes created: {len(nodes)} from {len(documents)} documents.")
    assert len(nodes) > 60, "Expected more than 60 nodes after chunking."

def test_fracture_cover_metaddata(documents):
    """
    Verify that the 'Fracture cover' table is preserved and tagged correctly.
    """
    chunker = SectionAwareChunker()
    nodes = chunker.get_nodes_from_documents(documents)

    found_fracture_cover = False
    found_correct_amount = False

    for node in nodes:
        # Check if section title was correctly extracted
        if "Fracture cover" in node.metadata.get("section_title", ""):
            found_fracture_cover = True

            if "£6,000" in node.text:
                found_correct_amount = True
                break

    assert found_fracture_cover, "'Fracture cover' section not found in any node."
    assert found_correct_amount, "'£6,000' not found in 'Fracture cover' section."

def test_definitions_preservation(documents):
    """
    Verify that definitions (like 'Incapacity') are captured.
    """
    chunker = SectionAwareChunker()
    nodes = chunker.get_nodes_from_documents(documents)

    found_incapacity_definition = False

    for node in nodes:
        # Looking for the definition of 'Incapacity'
        if "Definitions" in node.metadata.get("section_title", ""):
            if "Incapacity" in node.text:
                found_incapacity_definition = True
                break
    
    assert found_incapacity_definition, "Definition for 'Incapacity' not found in any node."