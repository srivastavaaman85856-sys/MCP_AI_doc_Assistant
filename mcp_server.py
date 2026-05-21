from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DocumentMCP", log_level="ERROR")

docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}


# =========================
# TOOL: Read a document
# =========================
@mcp.tool()
def read_doc(doc_name: str) -> str:
    """
    Read a document by filename.
    """
    return docs.get(doc_name, "Document not found")


# =========================
# RESOURCE: All document IDs
# =========================
@mcp.resource("docs://all")
def get_all_docs():
    return list(docs.keys())


# =========================
# RESOURCE: Particular doc
# =========================
@mcp.resource("docs://{doc_name}")
def get_doc(doc_name: str):
    return docs.get(doc_name, "Document not found")

# =========================
# PROMPT: Summarize Document
# =========================
@mcp.prompt()
def summarize_doc(doc_name: str):
    content = docs.get(doc_name, "Document not found")

    return f"""
    Summarize the following document clearly:

    {content}
    """


# =========================
# PROMPT: Rewrite in Markdown
# =========================
@mcp.prompt()
def rewrite_markdown(doc_name: str):
    content = docs.get(doc_name, "Document not found")

    return f"""
    Rewrite the following document in clean markdown format:

    {content}
    """


if __name__ == "__main__":
    mcp.run(transport="stdio")