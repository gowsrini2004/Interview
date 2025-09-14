# Psych Support — LLM Chatbot  

The **Psych Support — LLM Chatbot** is a complete mental health support system that combines **LLM-based AI chat**, **RAG-based knowledge retrieval**, and **human counselor consultation** into a single platform.  

This is not just a chatbot — it is a **psychological support dashboard** where:  
- **Users** can interact with an AI, complete wellness questionnaires, practice meditation & yoga, watch videos, join forums, and book consultations with counselors.  
- **Counselors** can monitor user progress (with user consent), assign videos, manage appointments, and host counselor-only forums.  
- **Admins** have full control over users, content, and system utilities.  

---

## 🏗️ Architecture & Tech Stack  

- **Frontend** → Streamlit (simple, responsive UI with sidebar navigation).  
- **Backend** → FastAPI (Python, provides APIs, authentication, mailing).  
- **Database** → PostgreSQL (stores login details, users, appointments, forum topics).  
- **Vector Database** → Qdrant (stores embeddings for Retrieval-Augmented Generation).  
- **Authentication** → JWT tokens for secure session management.  
- **Mailing** → MailHog (used for email notifications & reminders).  
- **LLM Providers**:  
  - **GROQ API** (fast, online, API-based).  
  - **OLLAMA–Mistral** (local containerized model, slower but offline).  
- **Deployment** → Docker (`base.yml` and `llm.yml` compose services).  

---

## 📑 Features  

### 🔹 Admin Module  
- **User Management** → View, search, clear chats/questionnaires, reset passwords, delete users.  
- **Utilities** → Upload or clear documents in VectorDB for chatbot context.  
- **Email Tool** → Send reminders, daily scores, or monthly summaries to users.  
- **Video Library** → Add/delete videos accessible by all users.  
- **Manage Counselors** → Promote/demote users, edit counselor details, or delete counselors.  
- **Forum** → View/create/delete topics.  

### 🔹 User Module  
- **Chatbot** → Context-aware chat with RAG sources and out-of-context warnings.  
- **Questionnaire** → Daily wellness questions, automated scoring, and dashboards.  
- **Videos** → Access global videos (admin/counselor uploads) and assigned videos.  
- **Meditate** → Guided meditation steps (AI-generated) with audio + meditation room.  
- **Yoga** → Curated videos by type (basic, advanced, surya namaskara, pranayama, flexibility).  
- **Forum** → Create/join discussions accessible to all.  
- **Consultation** → Book appointments with counselors, view today/future/closed appointments.  

### 🔹 Counselor Module  
- **Utility** → Upload documents to VectorDB for users, choose LLM provider, streaming toggle.  
- **User Chats & Questionnaires** → View only if user grants permission; can request access.  
- **Appointments** → Accept/reject requests, add meeting links, create appointments directly, close sessions with reports.  
- **Video Gallery** → Add videos globally or assign videos to specific users.  
- **Shared Tools** → Access Chat, Meditate, and Yoga features like normal users.  
- **Forum** → Create topics in public or counselor-only forums.  

---

## 📬 Appointment Notifications  

- For every scheduled appointment (user- or counselor-created):  
  - Automatic reminder emails are sent **24 hours before** and **12 hours before** the appointment.  
- Both **user and counselor** receive reminders.  

---

## ⚙️ Setup & Deployment Guide (Windows)  

Follow these steps to set up the application on a Windows machine.  

---

### 1. Python Environment Setup  

```bash
cd Interview\llm-challenge
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt