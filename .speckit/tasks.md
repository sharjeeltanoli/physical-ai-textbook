# Tasks: Physical AI & Humanoid Robotics Textbook

**Input**: Design documents from `/.speckit/`
**Prerequisites**: plan.md (required), project.spec.md (required for user stories)

**Tests**: Not explicitly requested, so integrated into implementation for verification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Frontend**: `physical-ai-textbook/docusaurus/`
- **Backend**: `rag-code/`
- Paths shown below assume project structure from the file system.

---

## Phase 1: Project Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure.

- [ ] T001 Initialize git repository (Completed by user)
- [ ] T002 Set up Docusaurus project in `physical-ai-textbook/` (✅ Done)
- [ ] T003 Configure basic Docusaurus project structure (sidebar, routes) in `physical-ai-textbook/docusaurus/docusaurus.config.ts`
- [ ] T004 Set up Python development environment for chatbot backend in `rag-code/requirements.txt`
- [ ] T005 Create initial `README.md` and basic content (✅ Done)

---

## Phase 2: Book Content Creation (Core Content)

**Purpose**: Develop the full 13-week course content.

**Goal**: All chapter content, code examples, images are drafted and ready for review.

### Book Outline and Chapters

- [ ] T006 Verify book outline and placeholder files for all modules in `physical-ai-textbook/docusaurus/docs/`
- [ ] T007 [P] Write Module 1 content in `physical-ai-textbook/docusaurus/docs/module1/`
- [ ] T008 [P] Write Module 2 content in `physical-ai-textbook/docusaurus/docs/module2/`
- [ ] T009 [P] Write Module 3 content in `physical-ai-textbook/docusaurus/docs/module3/`
- [ ] T010 [P] Write Module 4 content in `physical-ai-textbook/docusaurus/docs/module4/`
- [ ] T011 Create code examples for each chapter and place in `physical-ai-textbook/docusaurus/src/components/` or link from `static/`
- [ ] T012 Add diagrams and images for each chapter in `physical-ai-textbook/docusaurus/static/img/`
- [ ] T013 Review and edit all content for clarity and accuracy.

---

## Phase 3: RAG Chatbot Backend (US2: Interact with RAG Chatbot)

**Goal**: A functional FastAPI backend for the RAG chatbot, ingesting book content and providing context-aware responses.

**Independent Test**: Can be tested by sending sample queries to the FastAPI endpoints and verifying relevant responses.

### Setup and Database

- [ ] T025 [US2] Set up FastAPI project structure in `rag-code/main.py` and `rag-code/api.py`
- [ ] T026 [US2] Configure Neon Postgres database connection in `rag-code/` (e.g., in a settings file).
- [ ] T027 [P] [US2] Create `users` database schema.
- [ ] T028 [P] [US2] Create `conversations` and `messages` database schemas.
- [ ] T029 [US2] Set up Qdrant Cloud vector store client in `rag-code/retrieve.py`

### Content Ingestion and Embeddings

- [ ] T030 [US2] Implement book content ingestion pipeline script in `scripts/ingest_qdrant.py`
- [ ] T031 [US2] Create embeddings for all chapters using `scripts/ingest_qdrant.py` and store in Qdrant `book_chapters` collection
- [ ] T032 [US2] Create embeddings for all code examples using `scripts/ingest_qdrant.py` and store in Qdrant `code_examples` collection

### Chatbot Endpoints and Features

- [ ] T033 [US2] Build `POST /chat` endpoint in `rag-code/main.py` or `rag-code/api.py`.
- [ ] T034 [US2] Implement Cohere API integration for chat logic in `rag-code/agent.py`
- [ ] T035 [US2] Implement text selection query feature endpoint `POST /query-selection` in `rag-code/main.py` or `rag-code/api.py`.
- [ ] T036 [US2] Add conversation history management.
- [ ] T037 [US2] Test RAG responses by querying the `/chat` and `/query-selection` endpoints

---

## Phase 4: Authentication System (US3: Personalized Learning Experience)

**Goal**: A secure user authentication system integrated with the chatbot backend, allowing user signup and profiling.

**Independent Test**: Can be tested by signing up new users, logging in, and verifying access to protected routes.

- [ ] T038 [US3] Integrate Better-Auth library into FastAPI backend in `rag-code/main.py`.
- [ ] T039 [US3] Create signup endpoint with background questions in `rag-code/api.py` and define `user_backgrounds` schema.
- [ ] T040 [US3] Create login endpoint in `rag-code/api.py`.
- [ ] T041 [US3] Implement JWT token handling for sessions in `rag-code/api.py` and middleware.
- [ ] T042 [US3] Set up protected routes for personalized features in `rag-code/api.py`.
- [ ] T043 [US3] Test authentication flow: signup, login, access protected resources.

---

## Phase 5: Personalization Feature (US3: Personalized Learning Experience)

**Goal**: Implement dynamic content personalization per chapter based on user profiles.

**Independent Test**: Can be tested by creating different user profiles, activating personalization, and verifying content changes.

- [ ] T044 [US3] Create personalization API endpoint `POST /personalize-content` in `rag-code/api.py`
- [ ] T045 [US3] Implement LLM-based content adjustment logic in `rag-code/agent.py` using user background data.
- [ ] T046 [US3] Cache personalized content in Neon Postgres.
- [ ] T047 [US3] Test personalization with different user backgrounds (beginner/advanced) via API.

---

## Phase 6: Translation Feature (US3: Personalized Learning Experience)

**Goal**: Provide on-demand Urdu translation for chapter content.

**Independent Test**: Can be tested by activating the translation toggle and verifying Urdu content.

- [ ] T048 [US3] Create translation API endpoint `POST /translate-content` in `rag-code/api.py`
- [ ] T049 [US3] Implement Urdu translation using Cohere API in `rag-code/agent.py`
- [ ] T050 [US3] Cache translations in a `translations` table in Neon Postgres.
- [ ] T051 [US3] Handle preservation of code blocks and technical terms during translation in `rag-code/agent.py`
- [ ] T052 [US3] Test translation quality by calling API and reviewing output.

---

## Phase 7: Frontend Integration (US1, US2, US3)

**Goal**: Integrate all backend features into the Docusaurus frontend and ensure a responsive user experience.

**Independent Test**: Can be tested by interacting with the live Docusaurus site and verifying all features work.

- [ ] T053 [US1] Embed Docusaurus chatbot UI component (`ChatWidget.tsx`) in custom theme layout `physical-ai-textbook/docusaurus/src/theme/Root.tsx`.
- [ ] T054 [US2] Ensure chat interface `physical-ai-textbook/docusaurus/src/components/ChatWidget.tsx` has input and message display.
- [ ] T055 [US2] Add text selection handler in Docusaurus to trigger chatbot queries.
- [ ] T056 [US3] Integrate authentication UI (`AuthDisplay.tsx`) into Docusaurus in `physical-ai-textbook/docusaurus/src/pages/auth.tsx`.
- [ ] T057 [US3] Add "Personalize for Me" button to chapter layouts.
- [ ] T058 [US3] Add "Translate to Urdu" toggle button (`TranslateButton`) to chapter layouts.
- [ ] T059 [P] Style all new frontend components using Docusaurus styling conventions.
- [ ] T060 Ensure responsive design for all new UI elements and existing book content across devices.

---

## Phase 8: Deployment & Testing

**Goal**: The project is deployed, thoroughly tested, and ready for demonstration.

- [ ] T061 Deploy Docusaurus frontend to Vercel.
- [ ] T062 Deploy FastAPI backend to a serverless platform.
- [ ] T063 Configure environment variables for all deployed services.
- [ ] T064 Set up CI/CD pipeline for frontend.
- [ ] T065 Set up CI/CD pipeline for backend.
- [ ] T066 Perform end-to-end testing across all integrated features.
- [ ] T067 Conduct performance optimization and load testing.
- [ ] T068 Create a demo video (under 90 seconds) showcasing core features.