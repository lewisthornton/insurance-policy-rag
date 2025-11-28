import re
from typing import List, Tuple
from itertools import groupby
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser.interface import NodeParser

class SectionAwareChunker(NodeParser):
    """
    A custom chunker that splits documents by known section headers first,
    attaches the section title as metadata, and then uses SentenceSplitter
    to ensure chunks fit within the token limit.
    """

    # Configuration
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Look for headers
    section_patterns: List[str] = [
        r"^\s*Definitions",
        r"^\s*Core Benefits",
        r"^\s*Optional benefits",
        r"^\s*Making a claim",
        r"^\s*General conditions",
        r"^\s*Eligibility",
        r"^\s*Policy Summary",
        r"^\s*Your cover",
        r"^\s*Fracture cover",
        r"^\s*Global treatment",
        r"^\s*[0-9]+\.\s+[A-Z]", # Numbered sections
    ]

    def _parse_nodes(self, nodes: List[TextNode], show_progress: bool = False, **kwargs) -> List[TextNode]:
        """Required method for LlamaIndex NodeParsers."""
        all_nodes = []

        # Initialise standard splitter for when sections are too long
        base_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # 1. Sort nodes to ensure pages for the same document are together
        nodes.sort(key=lambda n: n.metadata.get("file_path", "unknown"))

        # 2. Group by file_path
        for file_path, file_docs in groupby(nodes, key=lambda n: n.metadata.get("file_path", "unknown")):
            
            # Reset context at the start of each document
            current_section_title = "Introduction/Preamble"

            for doc in file_docs:
                text = doc.text
                # Pass previous page's section title into current page
                sections, last_section_title = self._split_by_headers(text, current_section_title)

                # Update variable for next page
                current_section_title = last_section_title

                for section_title, section_text in sections:
                    if not section_text.strip():
                        continue
                    
                    section_metadata = dict(doc.metadata)
                    section_metadata["section_title"] = section_title

                    section_nodes = base_splitter.get_nodes_from_documents(
                        [Document(text=section_text, metadata=section_metadata)]
                    )

                    all_nodes.extend(section_nodes)

        return all_nodes
    
    def _split_by_headers(self, text: str, initial_section_title: str) -> Tuple[List[Tuple[str, str]], str]:
        """
        Splits text by regex headers. Returns list of (Section Title, Text).
        """
        lines = text.split("\n")
        sections = []

        # Use initial section title
        current_section_title = initial_section_title
        current_section_lines = []

        combined_pattern = "|".join(self.section_patterns)

        for line in lines:
            line = line
            if not line:
                continue

            if re.search(combined_pattern, line, re.IGNORECASE):
                # Save previous section
                if current_section_lines:
                    sections.append((current_section_title, "\n".join(current_section_lines)))
                
                # Start new section
                current_section_title = line.strip()
                current_section_lines = []
            else:
                current_section_lines.append(line)

        # Add the last section
        if current_section_lines:
            sections.append((current_section_title, "\n".join(current_section_lines)))

        return sections, current_section_title