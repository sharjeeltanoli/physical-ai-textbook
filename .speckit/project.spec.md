# Feature Specification: Physical AI & Humanoid Robotics Textbook

**Feature Branch**: `book-ai-project`
**Created**: 2025-12-06
**Status**: Draft
**Input**: User description for a comprehensive project specification for a Physical AI & Humanoid Robotics textbook.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Access Course Content (Priority: P1)

A user, upon visiting the book website, can navigate through the 13-week course content, viewing chapters within the 4 main modules.

**Why this priority**: Core functionality for a textbook; without it, the book serves no purpose.

**Independent Test**: Can be fully tested by browsing the generated Docusaurus site and verifying all chapter content is accessible and organized.

**Acceptance Scenarios**:

1. **Given** a user is on the book website, **When** they click on a module, **Then** they see a list of chapters for that module.
2. **Given** a user is viewing a list of chapters, **When** they click on a chapter, **Then** they see the full content of that chapter.

---

### User Story 2 - Interact with RAG Chatbot (Priority: P1)

A user can interact with an embedded RAG chatbot to ask questions about the book's content, receiving relevant and accurate answers.

**Why this priority**: A key distinguishing feature that enhances the learning experience.

**Independent Test**: Can be fully tested by asking questions related to various chapters and verifying the chatbot provides accurate, contextually relevant responses.

**Acceptance Scenarios**:

1. **Given** a user is on any chapter page, **When** they open the chatbot and ask a question about the chapter content, **Then** the chatbot provides a relevant answer.
2. **Given** a user selects text within a chapter, **When** they initiate a query based on the selection, **Then** the chatbot uses the selected text as context for the query.

---

### User Story 3 - Personalized Learning Experience (Priority: P2)

A user can sign up, create a profile, and receive personalized content recommendations or variations per chapter based on their background. They can also toggle Urdu translations.

**Why this priority**: Enhances engagement and accessibility, making the book more valuable to a wider audience.

**Independent Test**: Can be tested by creating different user profiles, observing content personalization, and verifying the Urdu translation toggle functions correctly per chapter.

**Acceptance Scenarios**:

1. **Given** a new user visits the site, **When** they sign up and complete their background profiling, **Then** subsequent chapter views show personalized content elements.
2. **Given** a user is viewing a chapter, **When** they activate the Urdu translation toggle, **Then** the chapter content is displayed in Urdu.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The book content MUST be structured into a 13-week course.
- **FR-002**: The course MUST be divided into 4 main modules: ROS 2, Gazebo & Unity, NVIDIA Isaac, and VLA.
- **FR-003**: Chapters MUST cover Physical AI fundamentals, ROS 2, simulation, humanoid development, and conversational robotics.
- **FR-004**: The book website MUST be built using Docusaurus.
- **FR-005**: The RAG chatbot backend MUST be implemented using FastAPI.
- **FR-006**: The chatbot functionality MUST leverage OpenAI Agents/ChatKit SDK.
- **FR-007**: The database for the project MUST be Neon Serverless Postgres.
- **FR-008**: The vector database for RAG MUST be Qdrant Cloud (free tier).
- **FR-009**: User authentication MUST be handled by Better-Auth.
- **FR-010**: An embedded RAG chatbot MUST be available on all book pages.
- **FR-011**: Users MUST be able to initiate chatbot queries based on selected text.
- **FR-012**: The system MUST support user signup with background profiling.
- **FR-013**: The system MUST provide content personalization per chapter based on user profiles.
- **FR-014**: The system MUST include an Urdu translation toggle per chapter.

### Key Entities *(include if feature involves data)*

- **User**: Represents a reader of the textbook, with attributes for authentication details, background profile, and personalization preferences.
- **Chapter**: Represents a unit of course content, including its text, code examples, images, and associated metadata for modules, weeks, and personalization variants, and translations.
- **Chatbot Interaction**: Records user queries, selected text context, and chatbot responses for potential feedback and improvement.

## Technical Specification

### 1. Book Architecture
- **Platform**: Docusaurus 3.x for static site generation.
- **Content Format**: All chapters and documentation written in Markdown.
- **Theme**: Custom Docusaurus theme with integrated dark mode toggle.
- **Responsiveness**: Fully responsive design for desktop, tablet, and mobile.
- **Search**: Built-in Docusaurus search functionality (e.g., DocSearch or local search).
- **Navigation**: Left-hand sidebar navigation structured by 4 main modules (ROS 2, Gazebo & Unity, NVIDIA Isaac, VLA).
- **Code Highlighting**: Support for syntax highlighting in Python, C++, YAML, and XML code blocks.

### 2. Content Structure
The book will be structured as a 13-week course, divided into 4 main modules with specific chapter topics:

**Module 1: ROS 2 (Weeks 3-5)**
- ROS 2 fundamentals: architecture, nodes, topics, services, actions.
- Hands-on examples with `rclpy` (Python client library).
- Creating and understanding URDF (Unified Robot Description Format) for robot modeling.

**Module 2: Gazebo & Unity (Weeks 6-7)**
- Principles of physics simulation in robotics.
- Simulation of common robot sensors: LIDAR, cameras, IMUs.
- Working with URDF and SDF (Simulation Description Format) for robot and environment modeling.

**Module 3: NVIDIA Isaac (Weeks 8-10)**
- Introduction to NVIDIA Isaac Sim and Isaac SDK.
- AI for robot perception (e.g., object detection, segmentation) and manipulation.
- Concepts and techniques for Sim-to-Real transfer.

**Module 4: VLA (Weeks 11-13)**
- Voice-to-Action systems using technologies like OpenAI Whisper.
- Integration of Large Language Models (LLMs) for high-level robot planning and decision-making.
- Developing conversational robotics interfaces.

### 3. RAG Chatbot System

#### Backend (FastAPI)
- **Framework**: FastAPI for high performance and ease of API development.
- **OpenAI Integration**: Utilize OpenAI Agents/ChatKit SDK for managing conversation flow, tool use, and agent orchestration.
- **Endpoints**:
  - `POST /chat`: Main chat endpoint.
    - **Request Body**: `{ "user_id": "string", "message": "string", "chapter_context": "string (optional)" }`
    - **Response Body**: `{ "response": "string", "conversation_id": "string" }`
  - `POST /query-selection`: Endpoint for text selection-based queries.
    - **Request Body**: `{ "user_id": "string", "selected_text": "string", "chapter_context": "string" }`
    - **Response Body**: `{ "response": "string", "conversation_id": "string" }`
  - `GET /health`: Health check endpoint.
    - **Response Body**: `{ "status": "healthy" }`
- **Component Interactions**:
  1. Frontend sends user query to FastAPI.
  2. FastAPI routes query to OpenAI Agents, providing book chapter context (from Qdrant) and potentially selected text.
  3. OpenAI Agent processes query, performs RAG lookup (via Qdrant), and generates a response.
  4. FastAPI stores conversation history in Neon Postgres and returns response to frontend.

#### Database (Neon Serverless Postgres)
- **Purpose**: Store user profiles, authentication details, and chat conversation history.
- **Schema**:
  - `users` table:
    - `id` (UUID, PK)
    - `email` (VARCHAR, UNIQUE)
    - `password_hash` (VARCHAR)
    - `name` (VARCHAR)
    - `created_at` (TIMESTAMP)
    - `updated_at` (TIMESTAMP)
  - `user_backgrounds` table:
    - `user_id` (UUID, FK to users.id)
    - `software_experience` (ENUM: 'beginner', 'intermediate', 'advanced')
    - `hardware_experience` (ENUM: 'none', 'basic', 'experienced')
    - `learning_style` (VARCHAR)
  - `conversations` table:
    - `id` (UUID, PK)
    - `user_id` (UUID, FK to users.id)
    - `chapter_id` (VARCHAR, references book chapter)
    - `started_at` (TIMESTAMP)
  - `messages` table:
    - `id` (UUID, PK)
    - `conversation_id` (UUID, FK to conversations.id)
    - `sender` (ENUM: 'user', 'chatbot')
    - `content` (TEXT)
    - `timestamp` (TIMESTAMP)

#### Vector Store (Qdrant Cloud - free tier)
- **Purpose**: Store vector embeddings of book content for RAG.
- **Collections**:
  - `book_chapters`:
    - Stores embedded chunks of each chapter.
    - Metadata: `chapter_id`, `module`, `week`, `page_num`, `text_content`.
  - `code_examples`:
    - Stores embedded code snippets from chapters.
    - Metadata: `chapter_id`, `example_id`, `language`, `code_content`, `explanation`.
- **Embedding Model**: `text-embedding-3-small` (OpenAI).
- **Ingestion**: A separate script will process all Markdown chapters and code examples, chunk them, embed them, and upload to Qdrant.

#### Chatbot Features
- **Full Book Content RAG**: Chatbot can answer questions based on the entire textbook content.
- **Selected Text Query Support**: Users can highlight text in the book and use it as direct context for a chatbot query.
- **Context-Aware Responses**: Responses will prioritize and integrate information from the current chapter and user-selected text.
- **Conversation History**: Users can view and continue past conversations.

### 4. Authentication System
- **Provider**: Better-Auth for robust authentication management.
- **Signup Form**:
  - Fields: Name, Email, Password.
  - Additional profiling questions:
    - Software Background: Radio buttons (Beginner, Intermediate, Advanced)
    - Hardware Experience: Radio buttons (None, Basic, Experienced)
    - Preferred Learning Style: Text input (e.g., "visual learner", "hands-on projects").
- **Session Management**: JWT (JSON Web Token) based sessions for stateless authentication.
- **Protected Routes**: Implement middleware to protect API endpoints and Docusaurus content requiring user authentication and personalization.

### 5. Personalization System
- **UI Trigger**: A prominently displayed button at the start of each chapter, labeled "Personalize for Me".
- **API Endpoint**: `POST /personalize-content`
  - **Request Body**: `{ "user_id": "string", "chapter_id": "string" }`
  - **Response Body**: `{ "personalized_content": "string" }` (Markdown format)
- **Content Adjustment Logic**: Based on user's `software_experience`, `hardware_experience`, and `learning_style` stored in `user_backgrounds` table.
  - **Beginners**: Content adjusted to include more foundational explanations, step-by-step guides, and simpler code examples.
  - **Advanced**: Content adjusted to skip introductory material, delve deeper into complex topics, advanced optimizations, and challenge questions.
  - **Real-time Transformation**: Utilize a powerful LLM (e.g., GPT-4 or similar) to dynamically rewrite/augment chapter content based on personalization rules and user profile.
- **Cache**: Personalized content will be cached per user per chapter in the database to avoid redundant LLM calls.

### 6. Translation System
- **UI Trigger**: A button at the start of each chapter, labeled "Translate to Urdu".
- **API Endpoint**: `POST /translate-content`
  - **Request Body**: `{ "user_id": "string", "chapter_id": "string", "target_language": "string (e.g., 'ur')" }`
  - **Response Body**: `{ "translated_content": "string" }` (Markdown format)
- **Caching**: Translated content will be cached in a `translations` table in Neon Postgres.
  - `translations` table:
    - `id` (UUID, PK)
    - `chapter_id` (VARCHAR, FK to book chapter)
    - `language` (VARCHAR, e.g., 'ur')
    - `original_hash` (VARCHAR, hash of original content for invalidation)
    - `translated_content` (TEXT)
    - `created_at` (TIMESTAMP)
- **Toggle Functionality**: Frontend toggle to switch between original English and cached Urdu translation.
- **Preservation**: The translation LLM will be instructed to preserve code blocks, technical terms (unless a specific Urdu equivalent is provided), and markdown formatting.

### 7. Deployment
- **Frontend (Docusaurus)**: Hosted on GitHub Pages or Vercel for static site hosting.
- **Backend (FastAPI)**: Deployed on Railway, Render, or Vercel Serverless functions.
- **Database (Postgres)**: Neon Serverless Postgres.
- **Vector DB**: Qdrant Cloud.
- **CI/CD**: GitHub Actions for automated testing, building, and deployment of both frontend and backend.

### 8. File Structure
```
.
├── .speckit/
│   └── project.spec.md
├── docs/
│   ├── chapters/
│   │   ├── module1-ros2/
│   │   │   ├── week3-intro-ros2.md
│   │   │   └── ...
│   │   ├── module2-gazebo-unity/
│   │   │   └── ...
│   │   ├── module3-nvidia-isaac/
│   │   │   └── ...
│   │   └── module4-vla/
│   │       └── ...
│   ├── images/
│   │   └── ...
│   └── code-examples/
│       └── ...
├── chatbot/
│   ├── main.py (FastAPI app)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── openai_agent.py
│   │   ├── qdrant_client.py
│   │   └── db_client.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py (SQLAlchemy models)
│   ├── config.py
│   └── requirements.txt
├── scripts/
│   ├── ingest_qdrant.py (for embedding book content)
│   ├── setup_db.py
│   └── ...
├── .github/
│   └── workflows/
│       ├── frontend-ci-cd.yml
│       └── backend-ci-cd.yml
├── .env.example
├── README.md
├── docusaurus.config.js
└── package.json
```

### 9. Configuration Files
- **`.env`**: For API keys (OpenAI, Better-Auth, Qdrant), database connection strings (Neon), and other sensitive environment variables.
- **`docusaurus.config.js`**: Frontend configuration, theme settings, navigation structure.
- **`package.json`**: Frontend dependencies.
- **`requirements.txt`**: Backend Python dependencies.
- **GitHub Actions YAMLs**: CI/CD pipeline definitions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 13 weeks of course content are accessible via the Docusaurus website.
- **SC-002**: The RAG chatbot provides answers with an average relevance score of 4/5 or higher, as rated by internal testers.
- **SC-003**: User signup and profiling process can be completed within 2 minutes.
- **SC-004**: Content personalization displays unique content for at least 3 distinct user profiles across 5 chapters.
- **SC-005**: Urdu translation toggle correctly translates 95% of chapter text when activated.
- **SC-006**: The Docusaurus site achieves a Lighthouse performance score of 90+ on desktop.
