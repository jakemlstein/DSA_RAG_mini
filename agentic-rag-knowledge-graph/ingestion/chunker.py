"""
Semantic chunking implementation for intelligent document splitting.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Import flexible providers
try:
    from ..agent.providers import get_embedding_client, get_ingestion_model
except ImportError:
    # For direct execution or testing
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.providers import get_embedding_client, get_ingestion_model

# Initialize clients with flexible providers
embedding_client = get_embedding_client()
ingestion_model = get_ingestion_model()


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 3000
    min_chunk_size: int = 100
    use_semantic_splitting: bool = True
    preserve_structure: bool = True
    chunking_strategy: str = "semantic"  # "semantic", "simple", or "legal"

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")
        if self.chunking_strategy not in ["semantic", "simple", "legal"]:
            raise ValueError(
                "Chunking strategy must be 'semantic', 'simple', or 'legal'"
            )


@dataclass
class DocumentChunk:
    """Represents a document chunk."""

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None

    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count is None:
            # Rough estimation: ~4 characters per token
            self.token_count = len(self.content) // 4


class SemanticChunker:
    """Semantic document chunker using LLM for intelligent splitting."""

    def __init__(self, config: ChunkingConfig):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self.client = embedding_client
        self.model = ingestion_model

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a document into semantically coherent pieces.

        Args:
            content: Document content
            title: Document title
            source: Document source
            metadata: Additional metadata

        Returns:
            List of document chunks
        """
        if not content.strip():
            return []

        base_metadata = {"title": title, "source": source, **(metadata or {})}

        # First, try semantic chunking if enabled
        if self.config.use_semantic_splitting and len(content) > self.config.chunk_size:
            try:
                semantic_chunks = await self._semantic_chunk(content)
                if semantic_chunks:
                    return self._create_chunk_objects(
                        semantic_chunks, content, base_metadata
                    )
            except Exception as e:
                logger.warning(
                    f"Semantic chunking failed, falling back to simple chunking: {e}"
                )

        # Fallback to rule-based chunking
        return self._simple_chunk(content, base_metadata)

    async def _semantic_chunk(self, content: str) -> List[str]:
        """
        Perform semantic chunking using LLM.

        Args:
            content: Content to chunk

        Returns:
            List of chunk boundaries
        """
        # First, split on natural boundaries
        sections = self._split_on_structure(content)

        # Group sections into semantic chunks
        chunks = []
        current_chunk = ""

        for section in sections:
            # Check if adding this section would exceed chunk size
            potential_chunk = (
                current_chunk + "\n\n" + section if current_chunk else section
            )

            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Current chunk is ready, decide if we should split the section
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Handle oversized sections
                if len(section) > self.config.max_chunk_size:
                    # Split the section semantically
                    sub_chunks = await self._split_long_section(section)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = section

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [
            chunk
            for chunk in chunks
            if len(chunk.strip()) >= self.config.min_chunk_size
        ]

    def _split_on_structure(self, content: str) -> List[str]:
        """
        Split content on structural boundaries.

        Args:
            content: Content to split

        Returns:
            List of sections
        """
        # Split on markdown headers, paragraphs, and other structural elements
        patterns = [
            r"\n#{1,6}\s+.+?\n",  # Markdown headers
            r"\n\n+",  # Multiple newlines (paragraph breaks)
            # r"\n[-*+]\s+",  # List items
            # r"\n\d+\.\s+",  # Numbered lists
            r"\n```.*?```\n",  # Code blocks
            r"\n\|\s*.+?\|\s*\n",  # Tables
        ]

        # Split by patterns but keep the separators
        sections = [content]

        for pattern in patterns:
            new_sections = []
            for section in sections:
                parts = re.split(
                    f"({pattern})", section, flags=re.MULTILINE | re.DOTALL
                )
                new_sections.extend([part for part in parts if part.strip()])
            sections = new_sections

        return sections

    async def _split_long_section(self, section: str) -> List[str]:
        """
        Split a long section using LLM for semantic boundaries.

        Args:
            section: Section to split

        Returns:
            List of sub-chunks
        """
        try:
            prompt = f"""
            Split the following text into semantically coherent chunks. Each chunk should:
            1. Be roughly {self.config.chunk_size} characters long
            2. End at natural semantic boundaries
            3. Maintain context and readability
            4. Not exceed {self.config.max_chunk_size} characters
            
            Return only the split text with "---CHUNK---" as separator between chunks.
            
            Text to split:
            {section}
            """

            # Use Pydantic AI for LLM calls
            from pydantic_ai import Agent

            temp_agent = Agent(self.model)

            response = await temp_agent.run(prompt)
            result = response.data
            chunks = [chunk.strip() for chunk in result.split("---CHUNK---")]

            # Validate chunks
            valid_chunks = []
            for chunk in chunks:
                if (
                    self.config.min_chunk_size
                    <= len(chunk)
                    <= self.config.max_chunk_size
                ):
                    valid_chunks.append(chunk)

            return valid_chunks if valid_chunks else self._simple_split(section)

        except Exception as e:
            logger.error(f"LLM chunking failed: {e}")
            return self._simple_split(section)

    def _simple_split(self, text: str) -> List[str]:
        """
        Simple text splitting as fallback.

        Args:
            text: Text to split

        Returns:
            List of chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size

            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break

            # Try to end at a sentence boundary
            chunk_end = end
            for i in range(end, max(start + self.config.min_chunk_size, end - 200), -1):
                if text[i] in ".!?\n":
                    chunk_end = i + 1
                    break

            chunks.append(text[start:chunk_end])
            start = chunk_end - self.config.chunk_overlap

        return chunks

    def _simple_chunk(
        self, content: str, base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Simple rule-based chunking.

        Args:
            content: Content to chunk
            base_metadata: Base metadata for chunks

        Returns:
            List of document chunks
        """
        chunks = self._simple_split(content)
        return self._create_chunk_objects(chunks, content, base_metadata)

    def _create_chunk_objects(
        self, chunks: List[str], original_content: str, base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Create DocumentChunk objects from text chunks.

        Args:
            chunks: List of chunk texts
            original_content: Original document content
            base_metadata: Base metadata

        Returns:
            List of DocumentChunk objects
        """
        chunk_objects = []
        current_pos = 0

        for i, chunk_text in enumerate(chunks):
            # Find the position of this chunk in the original content
            start_pos = original_content.find(chunk_text, current_pos)
            if start_pos == -1:
                # Fallback: estimate position
                start_pos = current_pos

            end_pos = start_pos + len(chunk_text)

            # Create chunk metadata
            chunk_metadata = {
                **base_metadata,
                "chunk_method": (
                    "semantic" if self.config.use_semantic_splitting else "simple"
                ),
                "total_chunks": len(chunks),
            }

            chunk_objects.append(
                DocumentChunk(
                    content=chunk_text.strip(),
                    index=i,
                    start_char=start_pos,
                    end_char=end_pos,
                    metadata=chunk_metadata,
                )
            )

            current_pos = end_pos

        return chunk_objects


class LegalChunker:
    """Specialized chunker for legal documents (laws, regulations, acts).

    Chunks legal documents according to their hierarchical structure:
    - Articles (primary chunking boundary)
    - Paragraphs (within articles)
    - Sub-paragraphs (within paragraphs)
    - Recitals (for preambles)
    """

    def __init__(self, config: ChunkingConfig):
        """Initialize legal chunker."""
        self.config = config

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a legal document by article structure.

        Args:
            content: Document content
            title: Document title (e.g., "EU Digital Services Act")
            source: Document source
            metadata: Additional metadata

        Returns:
            List of document chunks, one per article or section
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "legal",
            "document_type": "legal",
            **(metadata or {}),
        }

        chunks = []

        # First, separate recitals from articles
        recitals, articles_content = self._separate_recitals_and_articles(content)

        # Chunk recitals if present
        if recitals:
            recital_chunks = self._chunk_recitals(recitals, base_metadata)
            chunks.extend(recital_chunks)

        # Chunk articles
        article_chunks = self._chunk_articles(
            articles_content, base_metadata, len(chunks)
        )
        chunks.extend(article_chunks)

        # Update total chunks in metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _separate_recitals_and_articles(self, content: str) -> Tuple[str, str]:
        """
        Separate recitals (preamble) from articles.

        Args:
            content: Full document content

        Returns:
            Tuple of (recitals_text, articles_text)
        """
        # Look for the first article marker
        article_match = re.search(
            r"^##\s*Article\s+\d+", content, re.MULTILINE | re.IGNORECASE
        )

        if article_match:
            recitals = content[: article_match.start()].strip()
            articles = content[article_match.start() :].strip()
            return recitals, articles

        # If no articles found, treat entire content as having articles
        return "", content

    def _chunk_recitals(
        self, recitals: str, base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk recitals section.

        Recitals are grouped into reasonable chunks based on size.

        Args:
            recitals: Recitals text
            base_metadata: Base metadata

        Returns:
            List of recital chunks
        """
        chunks = []

        # Split recitals by individual numbered items: "- (1)", "- (2)", etc.
        recital_pattern = r"^-\s*\(\d+\)"
        individual_recitals = re.split(
            f"({recital_pattern})", recitals, flags=re.MULTILINE
        )

        # Reconstruct recitals with their numbers
        recital_items = []
        i = 1
        while i < len(individual_recitals):
            if re.match(recital_pattern, individual_recitals[i]):
                recital_num = re.search(r"\((\d+)\)", individual_recitals[i])
                if recital_num and i + 1 < len(individual_recitals):
                    recital_items.append(
                        {
                            "number": int(recital_num.group(1)),
                            "content": individual_recitals[i]
                            + individual_recitals[i + 1],
                        }
                    )
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        if not recital_items:
            # Fallback: treat entire recitals as one chunk
            if recitals.strip():
                chunk_metadata = {
                    **base_metadata,
                    "section_type": "recitals",
                }
                chunks.append(
                    DocumentChunk(
                        content=recitals.strip(),
                        index=0,
                        start_char=0,
                        end_char=len(recitals),
                        metadata=chunk_metadata,
                    )
                )
            return chunks

        # Group recitals into chunks based on size
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_char = 0

        for item in recital_items:
            item_size = len(item["content"])

            if current_size + item_size > self.config.chunk_size and current_chunk:
                # Create chunk from accumulated recitals
                chunk_content = "\n\n".join([r["content"] for r in current_chunk])
                recital_numbers = [r["number"] for r in current_chunk]

                chunk_metadata = {
                    **base_metadata,
                    "section_type": "recitals",
                    "recital_numbers": recital_numbers,
                    "recital_range": (
                        f"{recital_numbers[0]}-{recital_numbers[-1]}"
                        if len(recital_numbers) > 1
                        else str(recital_numbers[0])
                    ),
                }

                chunks.append(
                    DocumentChunk(
                        content=chunk_content.strip(),
                        index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                        metadata=chunk_metadata,
                    )
                )

                chunk_index += 1
                start_char += len(chunk_content)
                current_chunk = [item]
                current_size = item_size
            else:
                current_chunk.append(item)
                current_size += item_size

        # Add remaining recitals
        if current_chunk:
            chunk_content = "\n\n".join([r["content"] for r in current_chunk])
            recital_numbers = [r["number"] for r in current_chunk]

            chunk_metadata = {
                **base_metadata,
                "section_type": "recitals",
                "recital_numbers": recital_numbers,
                "recital_range": (
                    f"{recital_numbers[0]}-{recital_numbers[-1]}"
                    if len(recital_numbers) > 1
                    else str(recital_numbers[0])
                ),
            }

            chunks.append(
                DocumentChunk(
                    content=chunk_content.strip(),
                    index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                    metadata=chunk_metadata,
                )
            )

        return chunks

    def _chunk_articles(
        self, articles_content: str, base_metadata: Dict[str, Any], start_index: int
    ) -> List[DocumentChunk]:
        """
        Chunk articles section.

        Each article becomes a separate chunk (unless it's too large).

        Args:
            articles_content: Articles text
            base_metadata: Base metadata
            start_index: Starting index for chunks

        Returns:
            List of article chunks
        """
        chunks = []

        # Find all article positions with article number
        article_pattern = r"^##\s*Article\s+(\d+)\s*$"
        matches = list(
            re.finditer(article_pattern, articles_content, re.MULTILINE | re.IGNORECASE)
        )

        if not matches:
            return chunks

        chunk_index = start_index

        for i, match in enumerate(matches):
            article_number = int(match.group(1))
            article_start = match.start()

            # Find the end of this article (start of next article or end of content)
            if i + 1 < len(matches):
                article_end = matches[i + 1].start()
            else:
                article_end = len(articles_content)

            # Extract full article content
            full_article = articles_content[article_start:article_end].strip()

            # Extract article title (next ## line after the article number)
            lines = full_article.split("\n")
            article_title = ""
            article_body_start = 0

            for j, line in enumerate(
                lines[1:], 1
            ):  # Start from line after "## Article X"
                if line.strip().startswith("##"):
                    # This is the article title line
                    article_title = line.strip().lstrip("#").strip()
                    article_body_start = j + 1
                    break
                elif line.strip():
                    # Content starts without a title
                    article_body_start = j
                    break

            # Get article body (everything after title)
            article_body = (
                "\n".join(lines[article_body_start:])
                if article_body_start < len(lines)
                else ""
            )

            # Parse paragraphs within article body
            paragraphs = self._parse_article_paragraphs(article_body)

            # Check if article is too large
            if len(full_article) > self.config.max_chunk_size:
                # Split large article into multiple chunks by paragraphs
                sub_chunks = self._split_large_article(
                    article_number,
                    article_title,
                    paragraphs,
                    base_metadata,
                    chunk_index,
                    article_start,
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                # Create single chunk for article
                chunk_metadata = {
                    **base_metadata,
                    "section_type": "article",
                    "article_number": article_number,
                    "article_title": article_title,
                    "paragraph_count": len(paragraphs),
                }

                chunks.append(
                    DocumentChunk(
                        content=full_article,
                        index=chunk_index,
                        start_char=article_start,
                        end_char=article_end,
                        metadata=chunk_metadata,
                    )
                )
                chunk_index += 1

        return chunks

    def _parse_article_paragraphs(self, article_content: str) -> List[Dict[str, Any]]:
        """
        Parse paragraphs within an article.

        Args:
            article_content: Article content text

        Returns:
            List of paragraph dictionaries with structure info
        """
        paragraphs = []

        # Match numbered paragraphs: "1. ", "2. ", etc. at start of line
        para_pattern = r"^(\d+)\.\s+"
        lines = article_content.split("\n")

        current_para = None
        current_para_num = None

        for line in lines:
            para_match = re.match(para_pattern, line)
            if para_match:
                # Save previous paragraph
                if current_para is not None:
                    paragraphs.append(
                        {"number": current_para_num, "content": "\n".join(current_para)}
                    )

                # Start new paragraph
                current_para_num = int(para_match.group(1))
                current_para = [line]
            elif current_para is not None:
                current_para.append(line)

        # Save last paragraph
        if current_para is not None:
            paragraphs.append(
                {"number": current_para_num, "content": "\n".join(current_para)}
            )

        return paragraphs

    def _split_large_article(
        self,
        article_number: int,
        article_title: str,
        paragraphs: List[Dict[str, Any]],
        base_metadata: Dict[str, Any],
        start_index: int,
        start_char: int,
    ) -> List[DocumentChunk]:
        """
        Split a large article into multiple chunks by paragraphs.

        Args:
            article_number: Article number
            article_title: Article title
            paragraphs: List of parsed paragraphs
            base_metadata: Base metadata
            start_index: Starting chunk index
            start_char: Starting character position

        Returns:
            List of chunks for the large article
        """
        chunks = []
        current_paras = []
        current_size = 0
        chunk_index = start_index
        current_pos = start_char
        part_number = 1

        for para in paragraphs:
            para_size = len(para["content"])

            if current_size + para_size > self.config.chunk_size and current_paras:
                # Create chunk from accumulated paragraphs
                chunk_content = "\n\n".join([p["content"] for p in current_paras])
                para_numbers = [p["number"] for p in current_paras]

                chunk_metadata = {
                    **base_metadata,
                    "section_type": "article",
                    "article_number": article_number,
                    "article_title": article_title,
                    "article_part": part_number,
                    "paragraph_numbers": para_numbers,
                    "paragraph_range": (
                        f"{para_numbers[0]}-{para_numbers[-1]}"
                        if len(para_numbers) > 1
                        else str(para_numbers[0])
                    ),
                }

                chunks.append(
                    DocumentChunk(
                        content=chunk_content.strip(),
                        index=chunk_index,
                        start_char=current_pos,
                        end_char=current_pos + len(chunk_content),
                        metadata=chunk_metadata,
                    )
                )

                chunk_index += 1
                part_number += 1
                current_pos += len(chunk_content)
                current_paras = [para]
                current_size = para_size
            else:
                current_paras.append(para)
                current_size += para_size

        # Add remaining paragraphs
        if current_paras:
            chunk_content = "\n\n".join([p["content"] for p in current_paras])
            para_numbers = [p["number"] for p in current_paras]

            chunk_metadata = {
                **base_metadata,
                "section_type": "article",
                "article_number": article_number,
                "article_title": article_title,
                "article_part": part_number if part_number > 1 else None,
                "paragraph_numbers": para_numbers,
                "paragraph_range": (
                    f"{para_numbers[0]}-{para_numbers[-1]}"
                    if len(para_numbers) > 1
                    else str(para_numbers[0])
                ),
            }

            chunks.append(
                DocumentChunk(
                    content=chunk_content.strip(),
                    index=chunk_index,
                    start_char=current_pos,
                    end_char=current_pos + len(chunk_content),
                    metadata=chunk_metadata,
                )
            )

        return chunks


class SimpleChunker:
    """Simple non-semantic chunker for faster processing."""

    def __init__(self, config: ChunkingConfig):
        """Initialize simple chunker."""
        self.config = config

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk document using simple rules.

        Args:
            content: Document content
            title: Document title
            source: Document source
            metadata: Additional metadata

        Returns:
            List of document chunks
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "simple",
            **(metadata or {}),
        }

        # Split on paragraphs first
        paragraphs = re.split(r"\n\s*\n", content)
        chunks = []
        current_chunk = ""
        current_pos = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph exceeds chunk size
            potential_chunk = (
                current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            )

            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(
                        self._create_chunk(
                            current_chunk,
                            chunk_index,
                            current_pos,
                            current_pos + len(current_chunk),
                            base_metadata.copy(),
                        )
                    )

                    # Move position, but ensure overlap is respected
                    overlap_start = max(
                        0, len(current_chunk) - self.config.chunk_overlap
                    )
                    current_pos += overlap_start
                    chunk_index += 1

                # Start new chunk with current paragraph
                current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(
                self._create_chunk(
                    current_chunk,
                    chunk_index,
                    current_pos,
                    current_pos + len(current_chunk),
                    base_metadata.copy(),
                )
            )

        # Update total chunks in metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _create_chunk(
        self,
        content: str,
        index: int,
        start_pos: int,
        end_pos: int,
        metadata: Dict[str, Any],
    ) -> DocumentChunk:
        """Create a DocumentChunk object."""
        return DocumentChunk(
            content=content.strip(),
            index=index,
            start_char=start_pos,
            end_char=end_pos,
            metadata=metadata,
        )


# Factory function
def create_chunker(config: ChunkingConfig):
    """
    Create appropriate chunker based on configuration.

    Args:
        config: Chunking configuration

    Returns:
        Chunker instance (SemanticChunker, SimpleChunker, or LegalChunker)
    """
    if config.chunking_strategy == "legal":
        return LegalChunker(config)
    elif config.chunking_strategy == "simple" or not config.use_semantic_splitting:
        return SimpleChunker(config)
    else:
        # Default to semantic chunker
        return SemanticChunker(config)


# Example usage
async def main():
    """Example usage of the chunker."""

    # Example 1: Semantic chunking (default)
    print("=== SEMANTIC CHUNKING EXAMPLE ===")
    semantic_config = ChunkingConfig(
        chunk_size=500,
        chunk_overlap=50,
        use_semantic_splitting=True,
        chunking_strategy="semantic",
    )

    semantic_chunker = create_chunker(semantic_config)

    sample_text = """
    # Big Tech AI Initiatives
    
    ## Google's AI Strategy
    Google has been investing heavily in artificial intelligence research and development.
    Their main focus areas include:
    
    - Large language models (LaMDA, PaLM, Gemini)
    - Computer vision and image recognition
    - Natural language processing
    - AI-powered search improvements
    
    The company's DeepMind division continues to push the boundaries of AI research,
    with breakthrough achievements in protein folding prediction and game playing.
    
    ## Microsoft's Partnership with OpenAI
    Microsoft's strategic partnership with OpenAI has positioned them as a leader
    in the generative AI space. Key developments include:
    
    1. Integration of GPT models into Office 365
    2. Azure OpenAI Service for enterprise customers
    3. Investment in OpenAI's continued research
    """

    chunks = await semantic_chunker.chunk_document(
        content=sample_text, title="Big Tech AI Report", source="example.md"
    )

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk.content)} chars")
        print(f"Content: {chunk.content[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print("---")

    # Example 2: Legal chunking
    print("\n=== LEGAL CHUNKING EXAMPLE ===")
    legal_config = ChunkingConfig(
        chunk_size=2000,
        chunk_overlap=0,  # No overlap for legal documents
        max_chunk_size=5000,
        chunking_strategy="legal",
    )

    legal_chunker = create_chunker(legal_config)

    legal_sample = """
    - (1) This regulation establishes rules for digital services.
    - (2) The scope includes online platforms and search engines.
    
    ## Article 30
    
    ## Traceability of traders
    
    1. Providers of online platforms shall obtain the following information:
    
    - (a) the name, address, telephone number and electronic mail address of the trader;
    - (b) a copy of the identification document;
    - (c) the payment account details of the trader.
    
    2. Upon receiving the information, the provider shall make best efforts to assess reliability.
    
    ## Article 31
    
    ## Compliance by design
    
    1. Providers shall ensure that the online interface is designed appropriately.
    """

    legal_chunks = legal_chunker.chunk_document(
        content=legal_sample, title="Sample Regulation", source="regulation.md"
    )

    for i, chunk in enumerate(legal_chunks):
        print(f"Chunk {i}: {len(chunk.content)} chars")
        print(f"Section: {chunk.metadata.get('section_type', 'unknown')}")
        if chunk.metadata.get("article_number"):
            print(
                f"Article: {chunk.metadata['article_number']} - {chunk.metadata.get('article_title', '')}"
            )
        if chunk.metadata.get("recital_range"):
            print(f"Recitals: {chunk.metadata['recital_range']}")
        print(f"Content: {chunk.content[:150]}...")
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
