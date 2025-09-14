🧠 Psych Support LLM Chatbot
The Psych Support LLM Chatbot is a holistic psychology support application designed to provide a comprehensive blend of AI-driven interaction, structured self-assessment, wellness features, and direct access to human counseling. This system is architected with modern technologies to be a complete ecosystem for users, counselors, and administrators, ensuring scalability, security, and extensibility.

📸 Showcase
📚 Table of Contents
📋 Key Features

🛠️ Technology Stack

⚙️ Setup and Deployment (Windows)

🚀 Accessing the Services

🧑‍💻 How to Use

📋 Key Features
The application provides distinct functionalities based on user roles:

👤 For Users
💬 AI Chat: Engage with an AI that provides contextual answers from a knowledge base.

📝 Daily Questionnaires: Complete adaptive daily questionnaires to track mood and well-being.

🧘 Wellness Tools: Access guided meditations, calming audio, and curated yoga videos.

🎬 Video Library: Watch curated or personally assigned videos.

🌐 Community Forum: Participate in public forum discussions.

🗓️ Counselor Appointments: Book and manage appointments with counselors.

🔒 Privacy Control: Grant or revoke counselor access to your data.

👨‍⚕️ For Counselors
📊 User Monitoring: View chat histories and questionnaire dashboards of users who grant access.

✅ Appointment Management: Approve, reject, and manage appointments.

📚 Content Curation: Upload documents to enhance the chatbot's knowledge base.

🔒 Private Forum: Participate in a private forum visible only to other counselors.

👑 For Administrators
👥 Full User Management: View, search, and manage all user accounts.

⚙️ System Utilities: Manage the entire VectorDB and knowledge base.

⬆️ Counselor Promotion: Promote standard users to the counselor role.

🎥 Global Content Management: Add or delete videos from the global library.

📧 Email Tools: Trigger automated reminder and summary emails to users.

🛠️ Technology Stack
The application is built using a modern and reliable architecture:

Category

Technology

Description

Frontend

Streamlit

For building the interactive user interface.

Backend

FastAPI

For creating a high-performance Python API.

Database

PostgreSQL

For storing user data, appointments, etc.

Vector DB

Qdrant

For Retrieval-Augmented Generation (RAG).

Authentication

JWT (JSON Web Tokens)

For secure session management.

LLM Providers

GROQ & OLLAMA-Mistral

For fast cloud and reliable local model options.

Mailing

MailHog

For handling and viewing development emails.

Deployment

Docker

For containerizing and isolating services.

⚙️ Setup and Deployment (Windows)
This guide explains how to set up and run the chatbot on a Windows machine.

Prerequisites
Python 3.x

Docker Desktop

1. Python Environment Setup
First, set up a dedicated Python virtual environment.

Open Command Prompt or PowerShell and navigate to the project folder:

cd Interview\llm-challenge

Create a virtual environment:

python -m venv venv

Activate the environment:

venv\Scripts\activate

Install all required packages:

pip install -r requirements.txt

2. Configure Environment Variables
Place the .env file you received via email into the backend directory: Interview\llm-challenge\backend\.env

3. Start Docker Containers
Run the Docker startup script:

start-llm.sh

Verify that the containers are running:

docker ps

4. Set Up the Local LLM (Mistral)
Run the following commands to pull and run the Mistral model inside the OLLAMA container:

docker exec -it interview_ollama ollama pull mistral
docker exec -it interview_ollama ollama run mistral

5. Run the Application
Start the Backend

In a new terminal, navigate to the backend folder:

cd Interview\llm-challenge\backend

Run the FastAPI server:

python -m uvicorn main:app --reload

Wait for the confirmation message Application startup complete before proceeding.

Start the Frontend

Open another new terminal.

Navigate to the frontend folder:

cd Interview\llm-challenge\frontend

Run the Streamlit application:

streamlit run app.py

🚀 Accessing the Services
Once all steps are complete, the services will be available at the following URLs:

Frontend Application: The Streamlit app will open automatically in your browser.

Backend API: http://127.0.0.1:8000

MailHog (Email Viewer): http://localhost:8025/

🧑‍💻 How to Use
First-Time Users
Registration: New users must register with a valid email address.

Login: After registering, you can log in with your credentials.

Initial Delay: The first load may be slow as the VectorDB and embedder models initialize.

Admin Access
To log in as an administrator, use the following credentials directly on the login page:

Email: admin

Password: admin