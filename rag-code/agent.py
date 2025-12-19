# This script demonstrates a simple RAG (Retrieval-Augmented Generation) agent.
# The agent uses a retrieval tool to find relevant information from a Qdrant vector database
# and then uses a language model to generate an answer based on the retrieved information.

from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
from agents import enable_verbose_stdout_logging
import cohere
from qdrant_client import QdrantClient

# Enable verbose logging to see the agent's thought process.
enable_verbose_stdout_logging()

# Load environment variables from a .env file.
# This is where API keys and other secrets are stored.
load_dotenv()
# Disable tracing for this example. Tracing is a feature for debugging and monitoring agent execution.
set_tracing_disabled(disabled=True)

# Set up the Gemini model using an OpenAI-compatible endpoint.
# The `AsyncOpenAI` client is configured to point to Google's generative language API.
gemini_api_key = os.getenv("GOOGLE_API_KEY")
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# The `OpenAIChatCompletionsModel` is used to interact with the language model.
# We are using the "models/gemini-2.5-flash" model.
model = OpenAIChatCompletionsModel(
    model="models/gemini-2.5-flash",
    openai_client=provider
)

# Initialize the Cohere client for generating embeddings.
# Embeddings are numerical representations of text that capture semantic meaning.
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
# Connect to the Qdrant vector database.
# Qdrant is used to store and retrieve the embeddings.
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def get_embedding(text):
    """
    Get embedding vector from Cohere Embed v3.
    This function takes a text string and returns a vector embedding.
    """
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use "search_query" for queries to the vector database.
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding.

@function_tool
def retrieve(query):
    """
    This tool retrieves relevant text from the Qdrant vector database.
    It takes a user query, generates an embedding for it, and then queries Qdrant
    to find the most similar documents.
    """
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="physical_ai_book",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]

# Create the agent.
# The agent is given a name, instructions, a model to use, and a list of tools.
agent = Agent(
    name="Assistant",
    instructions="""
You are an AI tutor for the Physical AI & Humanoid Robotics textbook.
To answer the user question, first call the tool `retrieve` with the user query.
Use ONLY the returned content from `retrieve` to answer.
If the answer is not in the retrieved content, say "I don't know".
""",
    model=model,
    tools=[retrieve]
)

# Run the agent with a sample input.
# The `Runner.run_sync` function executes the agent and waits for the result.
result = Runner.run_sync(
    agent,
    input="what is physical ai?",
)

# Print the final output from the agent.
print(result.final_output)
