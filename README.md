# Address Parsing API

This repository contains an API that parses and sorts address components using vector search and language model inference. The API is built with FastAPI and leverages modern technologies to process and interpret address data, returning structured JSON output along with the processing time.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation and Running](#installation-and-running)
  - [Running Locally](#running-locally)
  - [Running with Docker Compose](#running-with-docker-compose)
- [Project Structure](#project-structure)
- [Based On](#based-on)
- [License](#license)

## Features

- **Address Parsing:** Extracts regional and emirate information along with addressee details (e.g., name, phone, email).
- **Vector Search:** Utilizes a vector store (Chroma) built from document embeddings to enhance inference context.
- **Embedding Service:** Integrates with the Ollama embeddings service to generate text embeddings.
- **Processing Time Measurement:** Returns the total time taken for processing each request.
- **API Based:** Built with FastAPI and deployable via Docker Compose for ease of scaling and cloud deployment.

## Technologies Used

- **Python 3.9+**
- **FastAPI:** For building the RESTful API.
- **Uvicorn:** ASGI server to run the FastAPI app.
- **Pydantic:** For request and response data validation.
- **Python Dotenv:** For loading environment variables.
- **LangChain, langchain-community, langchain-core:** For text splitting, prompt chaining, and language model interactions.
- **aiXplain:** For language model management and inference.
- **Chromadb:** Vector database for storing document embeddings.
- **Ollama:** Embeddings service (provided as a Docker container).
- **Docker & Docker Compose:** For containerizing and orchestrating the API and Ollama service.

## Prerequisites

- **Docker** and **Docker Compose** installed on your system.
- An API key for aiXplain.
- The Ollama service Docker image is used to provide the embeddings service.

## Installation and Running

### Running Locally

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/address-parser-api.git
   cd address-parser-api
   ```

2. **Set Up Environment Variables:**

   Create a `.env` file inside the app directory with the following content (update with your actual API key):

   ```env
   TEAM_API_KEY=YOUR_API_KEY
   OLLAMA_BASE_URL=http://localhost:11434
   ```

3. **Install Dependencies:**

   It is recommended to use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the FastAPI App:**

   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at http://localhost:8000.

### Running with Docker Compose

This project includes a Docker Compose setup to run both the API and the Ollama embeddings service together.

1. **Build and Start the Containers:**

   From the project root (where docker-compose.yml is located), run:

   ```bash
   docker-compose up --build
   ```

2. **Access the API:**

   The FastAPI API will be available at http://localhost:8000.

   The API is configured to connect to the Ollama service on port 11434 using the environment variable `OLLAMA_BASE_URL` set to `http://ollama:11434`.

## Project Structure

```
address-parser-api/
├── app/
│   ├── .env              # Environment variables file
│   ├── main.py           # FastAPI application code
│   └── output.txt        # Data file for building the vector store
├── requirements.txt      # Python dependencies
├── Dockerfile            # Dockerfile for the API container
├── Dockerfile.ollama     # Custom Dockerfile for the Ollama service container
├── entrypoint.sh         # Entrypoint script for the Ollama container (waits for service, pulls model, etc.)
└── docker-compose.yml    # Docker Compose configuration file
```

## Based On

This project is based on the work of Mohammad Qazi and his repository [Address-Parsing](https://github.com/Areeb2735/Address-Parsing/). Their work provided inspiration and foundational code for integrating vector search and language model inference, which has been adapted and extended in this project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.