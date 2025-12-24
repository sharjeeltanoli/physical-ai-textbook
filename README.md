# Physical AI & Humanoid Robotics Textbook

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for the "Physical AI & Humanoid Robotics" open-source textbook. This course is designed to bridge the gap between digital intelligence and physical embodiment, exploring how AI systems can perceive, understand, and interact with the real world through robotic hardware.

## About The Project

This project is an online, open-source textbook that provides a comprehensive curriculum on Physical AI and Humanoid Robotics. The content is built with Docusaurus, a modern static website generator, making it easy to read, navigate, and contribute to.

The textbook also features a sophisticated backend service powered by a Retrieval-Augmented Generation (RAG) pipeline. This allows for intelligent search and chat capabilities, enabling users to ask questions and get answers directly from the textbook's content.

### Key Features

- **Comprehensive Curriculum:** Covers a wide range of topics, from the fundamentals of ROS 2 and Gazebo simulation to advanced concepts like NVIDIA Isaac and Vision-Language-Action (VLA) models.
- **Interactive Learning:** Combines theoretical foundations with hands-on tutorials, practical code examples, and end-of-module projects.
- **RAG-Powered Backend:** A Python-based API using FastAPI, Cohere for text embeddings, and Qdrant as a vector database to provide a powerful search and chat experience.
- **Bilingual Support:** The textbook is available in both English and Urdu.
- **Open Source:** The entire project is open-source and available on GitHub, welcoming contributions from the community.

## Tech Stack

This project is built with a modern tech stack for both the frontend and backend components.

### Frontend (Docusaurus)

- **[Docusaurus](https://docusaurus.io/):** A static-site generator for building optimized documentation websites quickly.
- **[React.js](https://react.dev/):** A JavaScript library for building user interfaces.
- **[TypeScript](https://www.typescriptlang.org/):** A typed superset of JavaScript that compiles to plain JavaScript.
- **[MDX](https://mdxjs.com/):** Allows you to use JSX in your Markdown content.

### Backend (RAG API)

- **[Python](https://www.python.org/):** The primary language for the backend API and data processing pipeline.
- **[FastAPI](https://fastapi.tiangolo.com/):** A modern, fast (high-performance) web framework for building APIs with Python.
- **[Uvicorn](https://www.uvicorn.org/):** An ASGI server for running the FastAPI application.
- **[Cohere](https://cohere.com/):** Used to generate high-quality text embeddings for the RAG pipeline.
- **[Qdrant](https://qdrant.tech/):** A vector database for storing and searching through the textbook's content embeddings.
- **[Trafilatura](https://trafilatura.readthedocs.io/):** A web scraping library used to extract content from the sitemap.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- **Node.js:** Version 20.0 or higher.
- **Python:** Version 3.12 or higher.

### Installation & Usage

#### 1. Docusaurus Frontend

The frontend is the Docusaurus website that serves the textbook content.

```bash
# 1. Navigate to the Docusaurus directory
cd physical-ai-textbook/docusaurus

# 2. Install NPM packages
npm install

# 3. Run the development server
npm start
```

The website will be available at `http://localhost:3000`.

#### 2. RAG Backend

The backend is a Python-based API that provides search and chat functionality.

```bash
# 1. Navigate to the rag-code directory
cd rag-code

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Set up environment variables
#    Create a .env file and add your API keys:
#    COHERE_API_KEY="YOUR_COHERE_KEY"
#    QDRANT_URL="YOUR_QDRANT_URL"
#    QDRANT_API_KEY="YOUR_QDRANT_KEY"

# 5. Run the data ingestion pipeline
#    This will scrape the textbook content, create embeddings, and store them in Qdrant.
python main.py

# 6. Run the FastAPI server
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Sharjeel Tanoli - [@sharjeeltanoli](https://www.linkedin.com/in/muhammad-sharjeel-10591b254/)

Project Link: [https://github.com/sharjeeltanoli/physical-ai-textbook](https://github.com/sharjeeltanoli/physical-ai-textbook)
