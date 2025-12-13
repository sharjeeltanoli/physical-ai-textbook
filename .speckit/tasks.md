# Tasks: Physical AI & Humanoid Robotics Textbook

**Input**: Design documents from `/specs/book-ai-project/`
**Prerequisites**: plan.md (required - currently missing, will need to be created), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Not explicitly requested, so integrated into implementation for verification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume project structure from project.spec.md

---

## Phase 1: Project Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure.

- [ ] T001 Initialize git repository (Completed by user)
- [ ] T002 Set up Docusaurus project in the root directory
- [ ] T003 Configure basic Docusaurus project structure (sidebar, routes) in `docusaurus.config.js`
- [ ] T004 Set up Python development environment for chatbot backend in `chatbot/requirements.txt`
- [ ] T005 Create initial `README.md` and basic content (Completed by user)

---

## Phase 2: Book Content Creation (Core Content)

**Purpose**: Develop the full 13-week course content.

**Goal**: All chapter content, code examples, images are drafted and ready for review.

### Book Outline and Chapters

- [ ] T006 Create book outline and placeholder files for all 13 weeks in `docs/chapters/`
- [ ] T007 [P] Write Module 1: ROS 2 - Chapter: ROS 2 Architecture in `docs/chapters/module1-ros2/week3-ros2-architecture.md`
- [ ] T008 [P] Write Module 1: ROS 2 - Chapter: Nodes, Topics, Services in `docs/chapters/module1-ros2/week4-nodes-topics-services.md`
- [ ] T009 [P] Write Module 1: ROS 2 - Chapter: Python with rclpy in `docs/chapters/module1-ros2/week5-python-rclpy.md`
- [ ] T010 [P] Write Module 1: ROS 2 - Chapter: URDF Basics in `docs/chapters/module1-ros2/week5-urdf-basics.md`
- [ ] T011 [P] Write Module 2: Gazebo & Unity - Chapter: Gazebo Setup in `docs/chapters/module2-gazebo-unity/week6-gazebo-setup.md`
- [ ] T012 [P] Write Module 2: Gazebo & Unity - Chapter: Physics Simulation in `docs/chapters/module2-gazebo-unity/week6-physics-simulation.md`
- [ ] T013 [P] Write Module 2: Gazebo & Unity - Chapter: Sensor Simulation in `docs/chapters/module2-gazebo-unity/week7-sensor-simulation.md`
- [ ] T014 [P] Write Module 2: Gazebo & Unity - Chapter: Unity Integration in `docs/chapters/module2-gazebo-unity/week7-unity-integration.md`
- [ ] T015 [P] Write Module 3: NVIDIA Isaac - Chapter: Isaac Sim Introduction in `docs/chapters/module3-nvidia-isaac/week8-isaac-sim-intro.md`
- [ ] T016 [P] Write Module 3: NVIDIA Isaac - Chapter: Isaac SDK in `docs/chapters/module3-nvidia-isaac/week9-isaac-sdk.md`
- [ ] T017 [P] Write Module 3: NVIDIA Isaac - Chapter: AI Perception in `docs/chapters/module3-nvidia-isaac/week10-ai-perception.md`
- [ ] T018 [P] Write Module 3: NVIDIA Isaac - Chapter: Sim-to-Real Transfer in `docs/chapters/module3-nvidia-isaac/week10-sim-to-real-transfer.md`
- [ ] T019 [P] Write Module 4: VLA - Chapter: Voice-to-Action in `docs/chapters/module4-vla/week11-voice-to-action.md`
- [ ] T020 [P] Write Module 4: VLA - Chapter: LLM Planning in `docs/chapters/module4-vla/week12-llm-planning.md`
- [ ] T021 [P] Write Module 4: VLA - Chapter: Conversational Robotics in `docs/chapters/module4-vla/week13-conversational-robotics.md`
- [ ] T022 Create code examples for each chapter in `docs/code-examples/`
- [ ] T023 Add diagrams and images for each chapter in `docs/images/`
- [ ] T024 Review and edit all content for clarity and accuracy across `docs/chapters/`, `docs/code-examples/`, `docs/images/`

---

## Phase 3: RAG Chatbot Backend (US2: Interact with RAG Chatbot)

**Goal**: A functional FastAPI backend for the RAG chatbot, ingesting book content and providing context-aware responses.

**Independent Test**: Can be tested by sending sample queries to the FastAPI endpoints and verifying relevant responses.

### Setup and Database

- [ ] T025 [US2] Set up FastAPI project structure in `chatbot/main.py` and `chatbot/api/`
- [ ] T026 [US2] Configure Neon Postgres database connection in `chatbot/config.py`
- [ ] T027 [P] [US2] Create `users` database schema in `chatbot/models/database.py`
- [ ] T028 [P] [US2] Create `conversations` and `messages` database schemas in `chatbot/models/database.py`
- [ ] T029 [US2] Set up Qdrant Cloud vector store client in `chatbot/services/qdrant_client.py`

### Content Ingestion and Embeddings

- [ ] T030 [US2] Implement book content ingestion pipeline script in `scripts/ingest_qdrant.py`
- [ ] T031 [US2] Create embeddings for all chapters using `scripts/ingest_qdrant.py` and store in Qdrant `book_chapters` collection
- [ ] T032 [US2] Create embeddings for all code examples using `scripts/ingest_qdrant.py` and store in Qdrant `code_examples` collection

### Chatbot Endpoints and Features

- [ ] T033 [US2] Build `POST /chat` endpoint in `chatbot/api/routes.py`
- [ ] T034 [US2] Implement OpenAI Agents/ChatKit SDK integration for chat logic in `chatbot/services/openai_agent.py`
- [ ] T035 [US2] Implement text selection query feature endpoint `POST /query-selection` in `chatbot/api/routes.py`
- [ ] T036 [US2] Add conversation history management in `chatbot/services/db_client.py`
- [ ] T037 [US2] Test RAG responses by querying the `/chat` and `/query-selection` endpoints

---

## Phase 4: Authentication System (US3: Personalized Learning Experience)

**Goal**: A secure user authentication system integrated with the chatbot backend, allowing user signup and profiling.

**Independent Test**: Can be tested by signing up new users, logging in, and verifying access to protected routes.

- [ ] T038 [US3] Integrate Better-Auth library into FastAPI backend in `chatbot/main.py` and `chatbot/config.py`
- [ ] T039 [US3] Create signup endpoint with background questions in `chatbot/api/routes.py` and update `chatbot/models/database.py` for `user_backgrounds` schema
- [ ] T040 [US3] Create login endpoint in `chatbot/api/routes.py`
- [ ] T041 [US3] Implement JWT token handling for sessions in `chatbot/api/routes.py` and middleware
- [ ] T042 [US3] Set up protected routes for personalized features in `chatbot/api/routes.py`
- [ ] T043 [US3] Test authentication flow: signup, login, access protected resources

---

## Phase 5: Personalization Feature (US3: Personalized Learning Experience)

**Goal**: Implement dynamic content personalization per chapter based on user profiles.

**Independent Test**: Can be tested by creating different user profiles, activating personalization, and verifying content changes.

- [ ] T044 [US3] Create personalization API endpoint `POST /personalize-content` in `chatbot/api/routes.py`
- [ ] T045 [US3] Implement LLM-based content adjustment logic in `chatbot/services/openai_agent.py` using user background data
- [ ] T046 [US3] Cache personalized content in Neon Postgres (new table `personalized_content` or extend `chapters` table) in `chatbot/models/database.py`
- [ ] T047 [US3] Test personalization with different user backgrounds (beginner/advanced) via API

---

## Phase 6: Translation Feature (US3: Personalized Learning Experience)

**Goal**: Provide on-demand Urdu translation for chapter content.

**Independent Test**: Can be tested by activating the translation toggle and verifying Urdu content.

- [ ] T048 [US3] Create translation API endpoint `POST /translate-content` in `chatbot/api/routes.py`
- [ ] T049 [US3] Implement Urdu translation using LLM in `chatbot/services/openai_agent.py`
- [ ] T050 [US3] Cache translations in a `translations` table in `chatbot/models/database.py`
- [ ] T051 [US3] Handle preservation of code blocks and technical terms during translation in `chatbot/services/openai_agent.py`
- [ ] T052 [US3] Test translation quality by calling API and reviewing output

---

## Phase 7: Frontend Integration (US1, US2, US3)

**Goal**: Integrate all backend features into the Docusaurus frontend and ensure a responsive user experience.

**Independent Test**: Can be tested by interacting with the live Docusaurus site and verifying all features (chat, auth, personalization, translation) work.

- [ ] T053 [US1] Embed Docusaurus chatbot UI component (placeholder) in custom theme layout `src/theme/Layout/index.js`
- [ ] T054 [US2] Create chat interface component with input and message display in `src/components/Chatbot/`
- [ ] T055 [US2] Add text selection handler in Docusaurus to trigger chatbot queries in `src/theme/DocItem/Content/index.js`
- [ ] T056 [US3] Integrate authentication UI (signup/login forms) into Docusaurus in `src/components/Auth/`
- [ ] T057 [US3] Add "Personalize for Me" button to chapter layouts in `src/theme/DocItem/Content/index.js`
- [ ] T058 [US3] Add "Translate to Urdu" toggle button to chapter layouts in `src/theme/DocItem/Content/index.js`
- [ ] T059 [P] Style all new frontend components using Docusaurus styling conventions
- [ ] T060 Ensure responsive design for all new UI elements and existing book content across devices

---

## Phase 8: Deployment & Testing

**Goal**: The project is deployed, thoroughly tested, and ready for demonstration.

- [ ] T061 Deploy Docusaurus frontend to GitHub Pages/Vercel
- [ ] T062 Deploy FastAPI backend to Railway/Render/Vercel serverless
- [ ] T063 Configure environment variables for all deployed services (`.env` files, CI/CD secrets)
- [ ] T064 Set up CI/CD pipeline for frontend (GitHub Actions `frontend-ci-cd.yml`)
- [ ] T065 Set up CI/CD pipeline for backend (GitHub Actions `backend-ci-cd.yml`)
- [ ] T066 Perform end-to-end testing across all integrated features
- [ ] T067 Conduct performance optimization and load testing for backend and frontend
- [ ] T068 Create a demo video (under 90 seconds) showcasing core features

---

## Phase 9: Bonus Features

**Goal**: Enhance project with advanced agent capabilities.

- [ ] T069 Create reusable Claude Code Subagents for specific tasks (e.g., content generation, code review)
- [ ] T070 Develop Agent Skills for specific project workflows
- [ ] T071 Document subagents and skills usage in `docs/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately.
- **Book Content Creation (Phase 2)**: Can largely run in parallel with Setup, but content will be integrated into Docusaurus setup.
- **RAG Chatbot Backend (Phase 3)**: Depends on Phase 1 setup for Python environment.
- **Authentication System (Phase 4)**: Depends on Phase 3 for FastAPI backend.
- **Personalization Feature (Phase 5)**: Depends on Phase 4 (user profiles) and Phase 3 (LLM interaction).
- **Translation Feature (Phase 6)**: Depends on Phase 3 (LLM interaction) and database for caching.
- **Frontend Integration (Phase 7)**: Depends on completion of Phase 2 (content), Phase 3 (chatbot APIs), Phase 4 (auth APIs), Phase 5 (personalization APIs), Phase 6 (translation APIs).
- **Deployment & Testing (Phase 8)**: Depends on completion of all previous phases.
- **Bonus Features (Phase 9)**: Can be worked on in parallel with later phases, or after MVP.

### User Story Dependencies

- **User Story 1 (P1: Access Course Content)**: Primarily Phase 1 & 2, then Frontend integration (Phase 7).
- **User Story 2 (P1: Interact with RAG Chatbot)**: Primarily Phase 3, then Frontend integration (Phase 7).
- **User Story 3 (P2: Personalized Learning Experience)**: Primarily Phase 4, 5, 6, then Frontend integration (Phase 7).

### Within Each User Story

- Tasks related to models should precede services.
- Services should precede API endpoints.
- Backend tasks should precede frontend integration.

### Parallel Opportunities

- Many tasks within Phase 2 (Content Creation) are highly parallelizable (T007-T021).
- Tasks within Phase 3 (RAG Chatbot Backend) related to database schemas (T027, T028) can be parallelized.
- Frontend styling (T059) can be done in parallel with other frontend integration tasks.
- Setting up CI/CD for frontend and backend (T064, T065) can be parallelized.
- Different user stories can be worked on in parallel by different team members once foundational backend pieces are in place.

---

## Implementation Strategy

### MVP First (Prioritizing Core Experience)

1.  Complete Phase 1: Project Setup.
2.  Complete foundational elements of Phase 2: Book Content Creation (at least a few chapters).
3.  Complete Phase 3: RAG Chatbot Backend (core functionality).
4.  Complete core elements of Phase 7: Frontend Integration for US1 (content access) and US2 (chatbot).
5.  **STOP and VALIDATE**: Test core book access and chatbot functionality independently.
6.  Deploy/demo if ready.

### Incremental Delivery

1.  Complete Setup + Foundational Content + Core Chatbot → Foundation ready.
2.  Add User Story 1 (Access Course Content) → Test independently → Deploy/Demo (MVP!).
3.  Add User Story 2 (Interact with RAG Chatbot) → Test independently → Deploy/Demo.
4.  Add User Story 3 (Personalization + Translation) → Test independently → Deploy/Demo.
5.  Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers:

1.  Team completes Setup + Foundational together.
2.  Once Foundational is done:
    *   Developer A: Focus on RAG Chatbot Backend (Phase 3).
    *   Developer B: Focus on Authentication, Personalization, Translation Backend (Phase 4, 5, 6).
    *   Developer C: Focus on Frontend Integration (Phase 7) once backend APIs are defined.
3.  Stories complete and integrate independently.

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
