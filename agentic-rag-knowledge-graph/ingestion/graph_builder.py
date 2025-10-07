"""
Knowledge graph builder for extracting entities and relationships.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone
import asyncio
import re

from graphiti_core import Graphiti
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import graph utilities
try:
    from ..agent.graph_utils import GraphitiClient
except ImportError:
    # For direct execution or testing
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph_utils import GraphitiClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds knowledge graph from document chunks."""

    def __init__(self):
        """Initialize graph builder."""
        self.graph_client = GraphitiClient()
        self._initialized = False

    async def initialize(self):
        """Initialize graph client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True

    async def close(self):
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False

    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 3,  # Reduced batch size for Graphiti
    ) -> Dict[str, Any]:
        """
        Add document chunks to the knowledge graph.

        Args:
            chunks: List of document chunks
            document_title: Title of the document
            document_source: Source of the document
            document_metadata: Additional metadata
            batch_size: Number of chunks to process in each batch

        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            return {"episodes_created": 0, "errors": []}

        logger.info(
            f"Adding {len(chunks)} chunks to knowledge graph for document: {document_title}"
        )
        logger.info("⚠️ Large chunks will be truncated to avoid Graphiti token limits.")

        # Check for oversized chunks and warn
        oversized_chunks = [
            i for i, chunk in enumerate(chunks) if len(chunk.content) > 6000
        ]
        if oversized_chunks:
            logger.warning(
                f"Found {len(oversized_chunks)} chunks over 6000 chars that will be truncated: {oversized_chunks}"
            )

        episodes_created = 0
        errors = []

        # Process chunks one by one to avoid overwhelming Graphiti
        for i, chunk in enumerate(chunks):
            try:
                # Create episode ID
                episode_id = (
                    f"{document_source}_{chunk.index}_{datetime.now().timestamp()}"
                )

                # Prepare episode content with size limits
                episode_content = self._prepare_episode_content(
                    chunk, document_title, document_metadata
                )

                # Create source description (shorter)
                source_description = (
                    f"Document: {document_title} (Chunk: {chunk.index})"
                )

                # Add episode to graph
                await self.graph_client.add_episode(
                    episode_id=episode_id,
                    content=episode_content,
                    source=source_description,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "document_title": document_title,
                        "document_source": document_source,
                        "chunk_index": chunk.index,
                        "original_length": len(chunk.content),
                        "processed_length": len(episode_content),
                    },
                )

                episodes_created += 1
                logger.info(
                    f"✓ Added episode {episode_id} to knowledge graph ({episodes_created}/{len(chunks)})"
                )

                # Small delay between each episode to reduce API pressure
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)

            except Exception as e:
                error_msg = f"Failed to add chunk {chunk.index} to graph: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                # Continue processing other chunks even if one fails
                continue

        result = {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors,
        }

        logger.info(
            f"Graph building complete: {episodes_created} episodes created, {len(errors)} errors"
        )
        return result

    def _prepare_episode_content(
        self,
        chunk: DocumentChunk,
        document_title: str,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Prepare episode content with minimal context to avoid token limits.

        Args:
            chunk: Document chunk
            document_title: Title of the document
            document_metadata: Additional metadata

        Returns:
            Formatted episode content (optimized for Graphiti)
        """
        # Limit chunk content to avoid Graphiti's 8192 token limit
        # Estimate ~4 chars per token, keep content under 6000 chars to leave room for processing
        max_content_length = 6000

        content = chunk.content
        if len(content) > max_content_length:
            # Truncate content but try to end at a sentence boundary
            truncated = content[:max_content_length]
            last_sentence_end = max(
                truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? ")
            )

            if (
                last_sentence_end > max_content_length * 0.7
            ):  # If we can keep 70% and end cleanly
                content = truncated[: last_sentence_end + 1] + " [TRUNCATED]"
            else:
                content = truncated + "... [TRUNCATED]"

            logger.warning(
                f"Truncated chunk {chunk.index} from {len(chunk.content)} to {len(content)} chars for Graphiti"
            )

        # Add minimal context (just document title for now)
        if document_title and len(content) < max_content_length - 100:
            episode_content = f"[Doc: {document_title[:50]}]\n\n{content}"
        else:
            episode_content = content

        return episode_content

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (4 chars per token)."""
        return len(text) // 4

    def _is_content_too_large(self, content: str, max_tokens: int = 7000) -> bool:
        """Check if content is too large for Graphiti processing."""
        return self._estimate_tokens(content) > max_tokens

    async def extract_entities_from_chunks(
        self,
        chunks: List[DocumentChunk],
        extract_articles: bool = True,
        extract_actors: bool = True,
        extract_concepts: bool = True,
    ) -> List[DocumentChunk]:
        """
        Extract entities from chunks and add to metadata.

        Args:
            chunks: List of document chunks
            extract_articles: Whether to extract article names
            extract_actors: Whether to extract actor names
            extract_concepts: Whether to extract concept names

        Returns:
            Chunks with entity metadata added
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks")

        enriched_chunks = []

        for chunk in chunks:
            entities = {"articles": [], "actors": [], "people": [], "locations": []}

            content = chunk.content

            # Extract articles
            if extract_articles:
                entities["articles"] = self._extract_articles(content)

            # Extract actors
            if extract_actors:
                entities["actors"] = self._extract_actors(content)

            # Extract concepts
            if extract_concepts:
                entities["concepts"] = self._extract_concepts(content)

            # Create enriched chunk
            enriched_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "entities": entities,
                    "entity_extraction_date": datetime.now().isoformat(),
                },
                token_count=chunk.token_count,
            )

            # Preserve embedding if it exists
            if hasattr(chunk, "embedding"):
                enriched_chunk.embedding = chunk.embedding

            enriched_chunks.append(enriched_chunk)

        logger.info("Entity extraction complete")
        return enriched_chunks

    def _extract_articles(self, text: str) -> List[str]:
        """Extract article names from text."""
        # Known articles (extend this list as needed)
        articles = {
            "Article 1",
            "Article 2",
            "Article 3",
            "Article 4",
            "Article 5",
            "Article 6",
            "Article 7",
            "Article 8",
            "Article 9",
            "Article 10",
            "Article 11",
            "Article 12",
            "Article 13",
            "Article 14",
            "Article 15",
            "Article 16",
            "Article 17",
            "Article 18",
            "Article 19",
            "Article 20",
            "Article 21",
            "Article 22",
            "Article 23",
            "Article 24",
            "Article 25",
            "Article 26",
            "Article 27",
            "Article 28",
            "Article 29",
            "Article 30",
            "Article 31",
            "Article 32",
            "Article 33",
            "Article 34",
            "Article 35",
            "Article 36",
            "Article 37",
            "Article 38",
            "Article 39",
            "Article 40",
            "Article 41",
            "Article 42",
            "Article 43",
            "Article 44",
            "Article 45",
            "Article 46",
            "Article 47",
            "Article 48",
            "Article 49",
            "Article 50",
            "Article 51",
            "Article 52",
            "Article 53",
            "Article 54",
            "Article 55",
            "Article 56",
            "Article 57",
            "Article 58",
            "Article 59",
            "Article 60",
            "Article 61",
            "Article 62",
            "Article 63",
            "Article 64",
            "Article 65",
            "Article 66",
            "Article 67",
            "Article 68",
            "Article 69",
            "Article 70",
            "Article 71",
            "Article 72",
            "Article 73",
            "Article 74",
            "Article 75",
            "Article 76",
            "Article 77",
            "Article 78",
            "Article 79",
            "Article 80",
            "Article 81",
            "Article 82",
            "Article 83",
            "Article 84",
            "Article 85",
            "Article 86",
            "Article 87",
            "Article 88",
            "Article 89",
            "Article 90",
            "Article 91",
            "Article 92",
            "Article 93",
            "Article 94",
            "Article 95",
            "Article 96",
            "Article 97",
            "Article 98",
            "Article 99",
            "Article 100",
        }

        found_articles = set()
        text_lower = text.lower()

        for article in articles:
            # Match markdown format: ### **Article N**
            pattern = r"###\s+\*\*" + re.escape(article.lower()) + r"\*\*"
            if re.search(pattern, text_lower):
                found_articles.add(article)

        return list(found_articles)

    def _extract_actors(self, text: str) -> List[str]:
        """Extract actors terms from text."""
        actors = {
            "Digital Services Coordinator",
            "Coordinator",
            "Provider",
            "Data Provider",
            "Alibaba",
            "AliExpress",
            "Amazon Store",
            "Apple AppStore",
            "Booking.com",
            "Facebook",
            "Google Play",
            "Google Maps",
            "Google",
            "Bing",
            "Google Shopping",
            "Instagram",
            "Wikipedia",
            "X",
            "XNXX",
            "XVideos",
            "YouTube",
            "Zalando",
            "VLOPs",
            "Very Large Online Platforms",
            "Very Large Online Search Engines",
            "VLOSEs",
            "Member States",
            "Commission",
            "Board",
            "Applicant Researcher",
            "Principle Researcher",
            "Research Organisation",
            "research organisation",
            "member state",
            "Member State",
            "applicant researcher",
            "principal researcher",
            "vetted researcher",
            "Vettered Researcher",
        }

        found_actors = set()
        text_lower = text.lower()

        for actor in actors:
            if actor.lower() in text_lower:
                found_actors.add(actor)

        return list(found_actors)

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concept from text."""
        # Known concepts (extend this list as needed)
        concepts = {
            "Reasoned Request",
            "reasoned request",
            "data access process",
            "Data Access Process",
            "Data Access Application",
            "data access application",
            "data access modalities",
            "Data Access Modalities",
            "API",
            "application programming interface",
            "Application Programming Interface",
            "online databases",
            "Online Databases",
            "Data Scraping",
            "data scraping",
            "secure processing environment",
            "Secure Processing Environment",
            "Data Access Portal",
            "data access portal",
            "Data Access",
            "data access",
            "Data Access Regulation",
            "data access regulation",
            "DSA data access portal",
            "amendment request",
            "documentary evidence",
            "documentary evidence request",
            "affiliation",
            "research activities",
            "proposed safeguards",
            "identity",
            "contact details",
        }

        found_concepts = set()

        for concept in concepts:
            if concept in text:
                found_concepts.add(concept)

        return list(found_concepts)

    def _extract_locations(self, text: str) -> List[str]:
        """Extract location names from text."""
        locations = {
            "Silicon Valley",
            "San Francisco",
            "Seattle",
            "Austin",
            "New York",
            "Boston",
            "London",
            "Tel Aviv",
            "Singapore",
            "Beijing",
            "Shanghai",
            "Tokyo",
            "Seoul",
            "Bangalore",
            "Mountain View",
            "Cupertino",
            "Redmond",
            "Menlo Park",
        }

        found_locations = set()

        for location in locations:
            if location in text:
                found_locations.add(location)

        return list(found_locations)

    async def clear_graph(self):
        """Clear all data from the knowledge graph."""
        if not self._initialized:
            await self.initialize()

        logger.warning("Clearing knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Knowledge graph cleared")


class SimpleEntityExtractor:
    """Simple rule-based entity extractor as fallback."""

    def __init__(self):
        """Initialize extractor."""
        self.articles_patterns = [
            r"###\s+\*\*Article\s+\d{1,2}\*\*",
            r"###\s+\*\*Recital\s+\(\d{1,2}\)\*\*",
        ]
        self.actors_patterns = [
            r"\b(?:Digital Services Coordinator|Applicant Researcher|Principle Researcher|Data Provider)\b",
            r"\b(?:Research Organisation|Member State|VLOP|VLOSE)\b",
        ]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using patterns."""
        entities = {"actors": [], "articles": []}

        # Extract articles
        for pattern in self.articles_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["articles"].extend(matches)

        # Extract actors
        for pattern in self.actors_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["actors"].extend(matches)

        # Remove duplicates and clean up
        entities["articles"] = list(set(entities["articles"]))
        entities["actors"] = list(set(entities["actors"]))

        return entities


# Factory function
def create_graph_builder() -> GraphBuilder:
    """Create graph builder instance."""
    return GraphBuilder()


# Example usage
async def main():
    """Example usage of the graph builder."""
    from .chunker import ChunkingConfig, create_chunker

    # Create chunker and graph builder
    config = ChunkingConfig(chunk_size=300, use_semantic_splitting=False)
    chunker = create_chunker(config)
    graph_builder = create_graph_builder()

    sample_text = """
    The Digital Services Coordinator is responsible for mediating between the applicant researcher and the data provider."""

    # Chunk the document
    chunks = chunker.chunk_document(
        content=sample_text, title="Role of the DSC", source="example.md"
    )

    print(f"Created {len(chunks)} chunks")

    # Extract entities
    enriched_chunks = await graph_builder.extract_entities_from_chunks(chunks)

    for i, chunk in enumerate(enriched_chunks):
        print(f"Chunk {i}: {chunk.metadata.get('entities', {})}")

    # Add to knowledge graph
    try:
        result = await graph_builder.add_document_to_graph(
            chunks=enriched_chunks,
            document_title="Role of the DSCs",
            document_source="example.md",
            document_metadata={"topic": "EU DSA", "date": "2024"},
        )

        print(f"Graph building result: {result}")

    except Exception as e:
        print(f"Graph building failed: {e}")

    finally:
        await graph_builder.close()


if __name__ == "__main__":
    asyncio.run(main())
