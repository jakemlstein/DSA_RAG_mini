from docling.document_converter import DocumentConverter

source = "/Users/jacein/Documents/RAG_DSA_Staging/agentic-rag-knowledge-graph/DSA_delegated_act.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(
    result.document.export_to_markdown()
)  # output: "## Docling Technical Report[...]"

with open("DSA_delegated_act.md", "w") as f:
    f.write(result.document.export_to_markdown())
