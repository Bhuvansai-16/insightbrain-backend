# main.py
import os
import re
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Initialize environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TAVILY_API_KEY or not GROQ_API_KEY:
    raise RuntimeError("API keys not set in environment variables")

# Setup FastAPI application
app = FastAPI(title="Research Summarizer with Reasoning + PDF Export")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Data models for API
class Message(BaseModel):
    role: str = Field(..., example="user")
    content: str = Field(...)

class ResearchRequest(BaseModel):
    query: str = Field(..., example="Impact of AI on healthcare")
    history: List[Message] = Field(default_factory=list)

class DeepResearchRequest(BaseModel):
    query: str = Field(..., example="Impact of AI on healthcare")
    history: List[Message] = Field(default_factory=list)

class SummarizeResponse(BaseModel):
    reasoning_summary: str
    fact_consistency: str
    follow_up_questions: List[str]
    sources: List[str]
    report_md: str

class PDFRequest(BaseModel):
    query: str
    report_md: str

# Main functionality
def fetch_top_sources(query: str, k: int) -> List[str]:
    """Return top K URLs from web search"""
    try:
        search = TavilySearchResults(max_results=k, api_key=TAVILY_API_KEY)
        results = search.run(query)
        urls = [r.get("url") for r in results if r.get("url")]
        return urls[:k]
    except Exception as e:
        raise RuntimeError(f"Web search error: {e}")

def generate_llm_response(query: str, sources: List[str], history: List[Dict]) -> Dict:
    """Ask LLM to perform multi-step reasoning, fact-check, and follow-ups"""
    sources_text = "\n".join([f"- {s}" for s in sources])
    history_text = ""
    if history:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        history_text = f"Previous conversation:\n{history_text}\n\n"

    prompt = f"""You are an expert research assistant. Respond ONLY with the exact structured format below. Do not add any extra text, introductions, or conclusions.

Task: Given the user query, history, and top web sources, perform:
1) Multi-step reasoning summary of the topic, considering history
2) Fact consistency check across sources
3) Generate exactly 3 follow-up research questions

User query: {query}
{history_text}Sources:
{sources_text}

Format EXACTLY:
REASONING SUMMARY: [Your multi-step reasoning summary here, 200-300 words]

FACT CONSISTENCY: [Your fact consistency check here, 100-200 words]

FOLLOW_UP_QUESTIONS:
- [First follow-up question]
- [Second follow-up question]
- [Third follow-up question]"""

    llm = ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY)
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    text = text.strip()

    # Regex parsing
    reasoning_summary, fact_consistency, follow_ups = "", "", []
    try:
        rs_match = re.search(r'REASONING SUMMARY:\s*(.*?)\s*FACT CONSISTENCY:', text, re.DOTALL)
        if rs_match:
            reasoning_summary = rs_match.group(1).strip()

        fc_match = re.search(r'FACT CONSISTENCY:\s*(.*?)\s*FOLLOW_UP_QUESTIONS:', text, re.DOTALL)
        if fc_match:
            fact_consistency = fc_match.group(1).strip()

        fu_match = re.search(r'FOLLOW_UP_QUESTIONS:\s*(.*)', text, re.DOTALL)
        if fu_match:
            fu_block = fu_match.group(1).strip()
            lines = [line.strip(' -*•\t0123456789.') for line in fu_block.split('\n') if line.strip()]
            follow_ups = [line for line in lines if line][:3]
    except Exception:
        reasoning_summary = text
        fact_consistency = "Unable to parse fact consistency."
        follow_ups = []

    if not reasoning_summary:
        reasoning_summary = "No reasoning summary generated."
    if not fact_consistency:
        fact_consistency = "No fact consistency check generated."
    if not follow_ups:
        follow_ups = ["What are the main challenges?", "How does it impact costs?", "Future trends?"]

    report = f"# Research Report: {query}\n\n"
    report += "## Sources\n"
    for s in sources:
        report += f"- [{s}]({s})\n"
    report += f"\n## Reasoning Summary\n{reasoning_summary}\n\n"
    report += f"## Fact Consistency\n{fact_consistency}\n\n"
    report += "## Follow-up Questions\n"
    for q in follow_ups:
        report += f"- {q}\n"

    return {
        "reasoning_summary": reasoning_summary,
        "fact_consistency": fact_consistency,
        "follow_up_questions": follow_ups,
        "report_md": report
    }

# ===========================
# ===========================
# PDF Generation
# ===========================
def create_pdf_from_markdown(query: str, md_text: str) -> bytes:
    """Convert Markdown report into a downloadable PDF"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Split and add paragraphs
    for line in md_text.split("\n"):
        if line.startswith("# "):
            story.append(Paragraph(f"<b>{line[2:]}</b>", styles['Title']))
        elif line.startswith("## "):
            story.append(Paragraph(f"<b>{line[3:]}</b>", styles['Heading2']))
        elif line.startswith("- "):
            story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
        else:
            story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ===========================
# API Endpoints
# ===========================
@app.post("/research", response_model=SummarizeResponse)
def research(req: ResearchRequest):
    try:
        sources = fetch_top_sources(req.query, 3)
        if not sources:
            raise HTTPException(status_code=404, detail="No sources found")
        history_list = [{"role": m.role, "content": m.content} for m in req.history]
        llm_output = generate_llm_response(req.query, sources, history_list)
        llm_output["sources"] = sources
        return llm_output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/deep_research", response_model=SummarizeResponse)
def deep_research(req: DeepResearchRequest):
    try:
        sources = fetch_top_sources(req.query, 20)
        if not sources:
            raise HTTPException(status_code=404, detail="No sources found")
        history_list = [{"role": m.role, "content": m.content} for m in req.history]
        llm_output = generate_llm_response(req.query, sources, history_list)
        llm_output["sources"] = sources
        return llm_output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/generate_pdf")
def generate_pdf(req: PDFRequest):
    """Generate and return downloadable PDF"""
    try:
        pdf_bytes = create_pdf_from_markdown(req.query, req.report_md)
        headers = {"Content-Disposition": f"attachment; filename={req.query}_report.pdf"}
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}
