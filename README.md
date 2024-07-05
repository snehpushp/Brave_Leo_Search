# Brave_Leo_Search

## Overview
Brave_Leo_Search enhances the capabilities of Brave Leo AI by integrating search functionality with Groq as the language model (LLM). The project uses SearXNG and Firecrawl to power search operations, which are currently containerized via Docker.

The existing sequential search mechanism is under review for optimization. Plans are underway to adopt an agentic approach using Crew AI to improve search efficiency and response times.

## Prerequisites
- Docker
- Python 3.9+
- Uvicorn

## Installation

1. **Clone the Repository:**
   Begin by cloning this repository and the necessary submodules for SearXNG and Firecrawl.

   ```bash
   git clone --recursive <repository-url>
   ```

2. **Environment Setup:**
   Copy the `.env.example` file to `.env` and fill in the necessary environment variables.

   ```bash
   cp .env.example .env
   ```

3. **Running the Server:**
   Use Uvicorn to run the server. This will host the application on your local machine at port 8080.

   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
   ```

4. **Configure Endpoint:**
   Configure your client to use the server endpoint:

   ```
   http://localhost:8080/openai/chat
   ```

   Note: The model name parameter can be adjusted as required.

## Usage

Once the server is running, connect your client to the provided endpoint to start interacting with the Brave Leo AI with enhanced search capabilities.

## Future Updates
Further updates will focus on streamlining the search process and refining the installation steps once the new search methodology is implemented.

Thank you for your interest in Brave_Leo_Search!