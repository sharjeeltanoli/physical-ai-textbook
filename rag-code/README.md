# RAG-Code Project

This project demonstrates a Retrieval-Augmented Generation (RAG) system for code-related tasks. It leverages a large language model (LLM) and a retrieval mechanism to provide accurate and context-aware responses.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/rag-code.git
    cd rag-code
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file in the project root and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    ```

## Usage

To run the application, execute:

```bash
python main.py
```

Follow the prompts in the console to interact with the RAG system.
