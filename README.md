# Sri Lankan Tea Industry Assistant

## Overview

This project provides an AI-powered chatbot that answers questions about the Sri Lankan tea industry. It's built using Google Generative AI, LangChain, Pinecone, and Gradio, and deployed on Hugging Face Spaces.

## Features

- **Knowledge Base:** Leverages a Pinecone vector database to store and retrieve information about the Sri Lankan tea industry.
- **AI Assistant:** Employs Google Generative AI (Gemini) as the language model for understanding and answering user queries.
- **User-Friendly Interface:** Presents a visually appealing and easy-to-use chatbot interface powered by Gradio.
- **Ceylon Tea Theme:** Incorporates a Ceylon tea-inspired visual theme for an immersive experience.

## Getting Started

1.  **Clone the Repository:** Clone this repository to your local machine.
2.  **Install Dependencies:** Install the required libraries using `pip install -r requirements.txt`.
3.  **Set Up Secrets:** Create a Hugging Face Space and configure the following secrets:
    -   `GOOGLE_API_KEY`: Your Google AI Studio API key
    -   `PINECONE_API_KEY`: Your Pinecone API key
    -   `PINECONE_ENVIRONMENT`: Your Pinecone environment
    -   `PINECONE_INDEX_NAME`: Your Pinecone index name
    -   `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
4.  **Run the App:** Execute `python app.py` to launch the Gradio interface locally.
5.  **Deploy to Hugging Face:** Push your code to your Hugging Face repository to deploy the app on Hugging Face Spaces.

## Usage

-   Enter your questions about the Sri Lankan tea industry in the chatbot input field.
-   The AI assistant will analyze your query and provide relevant answers based on the information stored in the Pinecone vector database.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.

## License

This project is licensed under the [MIT License](LICENSE).