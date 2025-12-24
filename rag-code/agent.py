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
import asyncio

# Enable verbose logging to see the agent's thought process.
enable_verbose_stdout_logging()

# Load environment variables from a .env file.
# This is where API keys and other secrets are stored.
load_dotenv()
# Disable tracing for this example. Tracing is a feature for debugging and monitoring agent execution.
set_tracing_disabled(disabled=True)


gemini_api_key = os.getenv("OPENROUTER_API_KEY")
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://openrouter.ai/api/v1"
)


model = OpenAIChatCompletionsModel(
    model="xiaomi/mimo-v2-flash:free",
    openai_client=provider
)

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
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

Instructions:
1. Call the `retrieve` tool with the user's query
2. Use ONLY the returned content to answer
3. Provide a clear, direct answer without repeating the question
4. Do not include headers like "## What is..." or restate the question
5. If the answer is not in the retrieved content, respond with: 
   "I couldn't find information about that in the course materials. Could you try rephrasing your question or ask about a different topic from the course?"

Response Format:
- Start immediately with the answer
- Be concise and informative
- Use the content naturally without explicitly mentioning "based on the retrieved content"
""",
    model=model,
    tools=[retrieve]
)

if __name__ == '__main__':
    async def main():
        # Run the agent with a sample input.
        # The `Runner.run` function executes the agent and waits for the result.
        result = await Runner.run(
            agent,
            input="what is physical ai?",
        )

        # Print the final output from the agent.
        print(result.final_output)

    asyncio.run(main())
