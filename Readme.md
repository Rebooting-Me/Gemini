# Gemini (*beta*)

Gemini is a JavaScript project based on its namesake LLM that uses Google's Generative AI to generate content based on input context. It uses the `@google/generative-ai` package to interact with the Google Generative AI API.

### Setup

1. Clone the repository.
2. Install the dependencies with `npm install`.
3. Set up your [Google Generative AI API](https://makersuite.google.com/) key in a `.env` file in the root of the project:

```
GEMINI_API_KEY=your_api_key_here

Please replace your_api_key_here with your actual API key.
```

### Usage
Run the main script with node main.js.

### Functionality
The `main.js` script performs the following steps:

1. Configures the generation parameters.
2. Uses the `embedding-001` model to generate embeddings for a set of texts parsed from a PDF file.
3. Stores the generated embeddings in `new.json`(*for now*).
4. Loads the stored embeddings and embeds a question.
5. Uses the `gemini-pro` model to find the best passage that answers the question from the stored embeddings.
   
### Dependencies
- `@google/generative-ai`: To interact with the Google Generative AI API.
- `dotenv`: To load environment variables from a .env file.
- `fs`: To read and write files.
- `mathjs`: To perform mathematical operations.
- `pdf-parse`: To parse PDF files.
  
### Contributing
Contributions are welcome. Please open an issue or submit a pull request.

### License
This project is licensed under the ISC license.