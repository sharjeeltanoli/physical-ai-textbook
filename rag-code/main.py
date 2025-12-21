import logging
import time
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Import the RAGAgent class from your refactored agent.py
from agent import RAGAgent

# -------------------------------------
# CONFIGURATION
# -------------------------------------
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the FastAPI app
app = FastAPI(
    title="Physical AI Textbook RAG API",
    description="An API for querying the Physical AI & Humanoid Robotics textbook using a RAG agent.",
    version="1.0.0"
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This will hold the initialized RAGAgent instance
rag_agent: RAGAgent

# -------------------------------------
# PYDANTIC MODELS
# -------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        description="The user's query.",
        max_length=2000
    )

class MatchedChunk(BaseModel):
    content: str
    url: str
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    matched_chunks: List[MatchedChunk]
    status: str = "success"

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

# -------------------------------------
# API LIFECYCLE EVENTS
# -------------------------------------
@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAGAgent once when the application starts.
    This prevents reloading the model on every request.
    """
    global rag_agent
    logging.info("Application startup: Initializing RAGAgent...")
    try:
        rag_agent = RAGAgent()
        logging.info("RAGAgent initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize RAGAgent: {e}", exc_info=True)
        # Depending on the desired behavior, you might want the app to fail starting
        # if the agent can't be initialized.
        raise RuntimeError("Could not initialize RAGAgent.") from e

# -------------------------------------
# API ENDPOINTS
# -------------------------------------
@app.get("/health", summary="Health Check", tags=["Management"])
async def health_check():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"status": "ok", "message": "API is running"}

@app.post(
    "/ask",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Ask the RAG Agent",
    tags=["RAG"]
)
async def ask_question(request: QueryRequest = Body(...)):
    """
    Receives a query, processes it with the RAG agent, and returns the answer.
    """
    query = request.query
    logging.info(f"Received query: '{query}'")
    start_time = time.time()

    try:
        # Ensure agent is initialized
        if not rag_agent:
            raise HTTPException(status_code=503, detail="RAG Agent is not available.")

        # Call the agent's query method
        result = rag_agent.query_agent(query)
        
        # Create and return the response
        response = QueryResponse(**result)
        return response

    except Exception as e:
        logging.error(f"Error processing query '{query}': {e}", exc_info=True)
        # Return a clean JSON error response
        return HTTPException(
            status_code=500,
            detail={"status": "error", "message": "An internal error occurred while processing the request."}
        )
    finally:
        processing_time = (time.time() - start_time) * 1000
        logging.info(f"Query processed in {processing_time:.2f} ms")