// Import necessary modules
import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";
import fs from "fs";
import { dot } from "mathjs";
import parsePdf from "./file.js";

// Load environment variables
dotenv.config();

// Initialize Google Generative AI with API key
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Define configuration for generation
const generationConfig = {
  stopSequences: ["chup"],
  maxOutputTokens: 4096,
  temperature: 0.8,
  topP: 0.1,
  topK: 16,
};

// Get generative model from Google Generative AI
const newModel = genAI.getGenerativeModel({
  model: "gemini-pro",
  generationConfig,
});

// Function to generate context embeddings
async function generateContextEmbeddings(texts, fileName, model) {
  // Map over texts and generate embeddings
  const passages = await Promise.all(
    texts.map(async (text) => {
      const { embedding } = await model.embedContent(text);
      return {
        text,
        embedding,
      };
    })
  );

  // Write embeddings to file
  try {
    fs.writeFileSync(`./contexts/${fileName}.json`, JSON.stringify(passages));
    console.log("File written successfully\n");
  } catch (err) {
    console.error("Writing file failed\n", err);
  }
}

// Function to load stored embeddings
async function loadStoredEmbeddings(storedFile) {
  let storedEmbeddings;
  try {
    storedEmbeddings = await JSON.parse(
      fs.readFileSync(`./contexts/${storedFile}.json`)
    );
    console.log("Stored embeddings loaded successfully");
  } catch (err) {
    console.error("Stored embeddings failed to load:", err);
    return;
  }
  return storedEmbeddings;
}

// Function to find the best passage
async function FindBestPassage(question, storedEmbeddings, model) {
  // Embed the question
  const questionResult = await model.embedContent(question, {
    taskType: "QUERY",
  });
  const questionEmbedding = questionResult.embedding;

  // Calculate dot products of question and stored embeddings
  const dotProducts = storedEmbeddings.map((storedEmbedding) => {
    if (Array.isArray(storedEmbedding.embedding.values)) {
      return dot(questionEmbedding.values, storedEmbedding.embedding.values);
    } else {
      return -Infinity;
    }
  });

  // Find the index of the maximum dot product
  const maxIndex = dotProducts.indexOf(Math.max(...dotProducts));

  // Construct the prompt
  var prompt = `QUESTION: ${question} PASSAGE: ${storedEmbeddings[maxIndex].text} ANSWER:`;
  const threshold = 0.5;

  // Generate answer if dot product is above threshold
  if (dotProducts[maxIndex] > threshold) {
    const answer = await newModel.generateContent(prompt);
    console.log("Answer:", answer);

    // Return the answer if it exists
    if (
      answer.response &&
      answer.response.candidates &&
      answer.response.candidates.length > 0
    ) {
      return answer.response.candidates[0].content.parts[0].text;
    }
  } else {
    // Return a default message if no answer is found
    return "I'm sorry, I couldn't find a relevant answer to your question.";
  }
}

// Main function to run the program
async function run() {
  // Get the embedding model
  const model = genAI.getGenerativeModel({
    model: "embedding-001",
    generationConfig,
  });

  // Parse the PDF and generate embeddings
  const texts = await parsePdf();
  let contextFile = "test";
  await generateContextEmbeddings(texts, contextFile, model);

  // Load stored embeddings
  let storedFile = "test";
  const storedEmbeddings = await loadStoredEmbeddings(storedFile);

  // Check if stored embeddings is an array
  if (!Array.isArray(storedEmbeddings)) {
    console.error("storedEmbeddings is not an array:", storedEmbeddings);
    return;
  }

  // Define the question and find the best passage
  const question = "What are the research objectives of the article?";
  const bestPassage = await FindBestPassage(question, storedEmbeddings, model);
  console.log("Best passage:", bestPassage);
}

// Run the main function
run();