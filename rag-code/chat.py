# This script provides an interactive chat interface for the RAG agent.
# It allows the user to ask questions and get answers from the agent in a loop.

from agents import Agent, Runner
from agent import agent # Import the agent from agent.py

print("Starting interactive chat with the Physical AI tutor.")
print("Type 'exit' to end the chat.")

while True:
    user_input = input("> ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    # Run the agent with the user's input.
    result = Runner.run_sync(
        agent,
        input=user_input,
    )

    # Print the final output from the agent.
    print(result.final_output)