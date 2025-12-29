# Project Constitution: Physical AI & Humanoid Robotics Textbook

This document outlines the core principles, architectural decisions, and development conventions for the "Physical AI & Humanoid Robotics" open-source textbook project. Its purpose is to ensure consistency, clarity, and quality throughout the development lifecycle.

## 1. Core Mission & Vision

- **Primary Goal**: To create a comprehensive, high-quality, and freely accessible online textbook that bridges the gap between digital intelligence and physical embodiment.
- **Vision**: To be the go-to educational resource for students, researchers, and hobbyists interested in Physical AI and humanoid robotics, fostering an interactive and engaging learning experience.

## 2. Guiding Principles

- **Open Source First**: The project is fundamentally open source. Contributions are encouraged, and all content and code should be developed with community collaboration in mind.
- **Quality & Accuracy**: All educational content must be technically accurate, up-to-date, and well-written. Code must be clean, efficient, and well-documented.
- **User-Centric Design**: The user experience is paramount. The textbook must be easy to navigate, and features like the RAG chatbot and personalization should be intuitive and genuinely useful.
- **Modularity & Scalability**: The architecture, both for the content and the code, should be modular to allow for easy updates, extensions, and maintenance.

## 3. Architectural & Technical Stack

This project is composed of two main components: a Docusaurus frontend and a Python backend.

### Frontend

- **Framework**: **Docusaurus** is the official static-site generator. All content pages and the overall site structure must be built within this framework.
- **Language**: **TypeScript** is the standard for all new frontend code to ensure type safety and maintainability.
- **Styling**: Use CSS Modules for component-level styling and a global `custom.css` for site-wide theme adjustments.
- **Content**: Content is written in **MDX**, allowing for the use of React components within Markdown.

### Backend (RAG API)

- **Framework**: **FastAPI** is the chosen framework for building the backend API due to its performance and automatic documentation generation.
- **Language**: **Python 3.12+** is the standard for all backend code.
- **Vector Database**: **Qdrant** is the designated vector store for the RAG pipeline.
- **Embeddings**: Text embeddings are generated using the **Cohere** API.
- **Relational Database**: **Neon Serverless Postgres** is used for storing user data, profiles, and chat history.

### Authentication

- **Provider**: User authentication and profile management must be implemented using **Better-Auth**.

## 4. Development Conventions

### Code Style

- **Python**: Adhere to **PEP 8** standards. Use a linter and formatter like `black` or `ruff` to maintain consistency.
- **TypeScript/JavaScript**: Follow the conventions established by the Docusaurus and React communities. Use a linter like ESLint and a formatter like Prettier.
- **Naming**: Use clear, descriptive names for variables, functions, and classes. Follow standard conventions (e.g., `PascalCase` for components, `camelCase` for functions/variables in TS, `snake_case` for Python).

### Content Structure

- **Organization**: The textbook is structured into 4 main modules, which are then broken down into weekly chapters, as defined in the `project.spec.md`.
- **Bilingual Support**: The textbook supports both **English (en)** and **Urdu (ur)**. All content must have a corresponding translation.
- **File Naming**: Markdown files for content should be named descriptively (e.g., `intro-ros2.md`).

### Version Control

- **Branching**: All new features and significant changes must be developed in a feature branch, following the naming convention `feature/<feature-name>`.
- **Commits**: Write clear, concise, and informative commit messages.
- **Pull Requests**: All changes must be submitted via a Pull Request (PR). PR descriptions should clearly explain the "why" and "what" of the changes.

## 5. Key Artifacts & Documentation

- **Project Specification (`.speckit/project.spec.md`)**: This is the "source of truth" for all functional requirements, user stories, and technical specifications. All development work should align with this document.
- **README.md**: The main `README.md` should provide a high-level overview of the project and instructions for getting started.
- **API Documentation**: The FastAPI backend will automatically generate interactive API documentation (e.g., at `/docs`), which should be kept up-to-date.
- **Inline Comments**: Use inline comments sparingly, focusing on the *why* of a complex piece of logic, not the *what*.

This constitution is a living document. It can be updated as the project evolves, but any changes should be discussed and agreed upon by the core contributors.