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
        r"Definitions",
        r"Core Benefits",
        r"Co re Benefits",        
        r"Optional benefits",
        r"Making a claim",
        r"General conditions",
        r"Eligibility",
        r"Policy Summary",
        r"^\s*Your cover",         
        r"Fracture cover",
        r"Global treatment",
        r"What types of cover", 
        r"^\s*[0-9]+\.\s+[A-Z]",
    ]

    noise_patterns: List[str] = [
        r"Policy Conditions.*Living Costs Protection", # Catches the fused header
        r"Living Costs Protection.*Policy Conditions",
        r"Policy Summary.*Living Costs Protection",
        r"Aviva Life & Pensions UK Limited",
        r"aviva\.co\.uk",
        r"^[0-9]+$",                    # Lone page numbers (e.g. "9")
        r"^Page\s+[0-9]+$",             # "Page 10"
        r"^AVIVA$",                     # Logo text
        r"^keyfacts$",                  # Common logo text
        r"^Need this in a different format\?$", # Accessibility footer
        r"P\s?a\s?g\s?e\s+\d+$",
        r"^\s*ge\s+\d+$",
        r"^\s*Pa\s*$",
        r"^Contents$",
        r"\.indd",
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
        lines = text.split("\n")
        sections = []
        
        current_title = initial_section_title
        current_lines = []

        combined_pattern = "|".join(self.section_patterns)

        combined_noise_pattern = "|".join(self.noise_patterns)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.search(combined_noise_pattern, line, re.IGNORECASE):
                continue

            # --- UPDATED HEADER DETECTION LOGIC ---
            match = re.search(combined_pattern, line, re.IGNORECASE)
            is_short_enough = len(line) < 100  # Increased slightly to catch longer headers
            is_not_sentence = not line.endswith('.') # Ignores full sentences
            
            # NEW: Filter out Table of Contents / Page numbers
            # If line ends with a digit (e.g. "Page 10", "12"), it's likely a TOC or footer.
            has_trailing_digit = line[-1].isdigit() 
            
            if match and is_short_enough and is_not_sentence and not has_trailing_digit:
                # If we have accumulated text, save it
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                
                # Start new section
                current_title = line
                current_lines = []
            else:
                current_lines.append(line)

        # Append leftovers
        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))
            
        return sections, current_title