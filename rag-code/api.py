import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import Runner
from agent import agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create thread pool executor
executor = ThreadPoolExecutor(max_workers=4)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    query: str

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_loaded": agent is not None
    }

@app.post("/api/chat")
async def chat_endpoint(chat_query: ChatQuery):
    """
    Endpoint to interact with the chatbot agent.
    """
    try:
        logger.info(f"Received query: {chat_query.query}")
        
        # Check if agent is available
        if agent is None:
            logger.error("Agent not available")
            raise HTTPException(
                status_code=503, 
                detail="Agent not properly configured. Please check agent.py."
            )
        
        # Run the agent in a separate thread to avoid event loop conflicts
        loop = asyncio.get_event_loop()
        
        def run_agent():
            return Runner.run_sync(agent, input=chat_query.query)
        
        result = await loop.run_in_executor(executor, run_agent)
        
        if hasattr(result, 'final_output') and result.final_output:
            logger.info(f"Agent returned answer: {result.final_output}")
            return {"answer": result.final_output}
        else:
            logger.error("Agent did not produce a valid final_output.")
            raise HTTPException(
                status_code=500, 
                detail="Agent failed to produce a response."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"An error occurred during agent execution: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)