import {GoogleGenerativeAI} from "@google/generative-ai";
import dotenv from "dotenv";
import fs from "fs";
import {dot} from "mathjs";
import parsePdf from "./file.js";
dotenv.config();

// Access your API key as an environment variable (see "Set up your API key" above)
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

async function run() {
  // Set up the configuration for the generation
  const generationConfig = {
    stopSequences: ["chup"],
    maxOutputTokens: 4096,
    temperature: 0.8,
    topP: 0.1,
    topK: 16,
  };

  // For embeddings, use the embedding-001 model
  const model = genAI.getGenerativeModel({
    model: "embedding-001",
    generationConfig,
  });

  const texts = await parsePdf();
  // console.log(texts);

  let contextFile = "test"; //name of the context file to be embedded
  let storedFile = "test"; //name of the stored context file to ask question from

  async function generateContextEmbeddings(texts, fileName) {
    const passages = await Promise.all(
      texts.map(async (text) => {
        const { embedding } = await model.embedContent(text);
        return {
          text,
          embedding,
        };
      })
    );
    // Store the array of embeddings
    try {
      fs.writeFileSync(`./contexts/${fileName}.json`, JSON.stringify(passages));
      console.log("File written successfully\n");
    } catch (err) {
      console.error("Writing file failed\n", err);
    }
  }
  await generateContextEmbeddings(texts, contextFile);

  const newModel = genAI.getGenerativeModel({
    model: "gemini-pro",
    generationConfig,
  });
  // Load the stored embeddings
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

  if (!Array.isArray(storedEmbeddings)) {
    console.error("storedEmbeddings is not an array:", storedEmbeddings);
    return;
  }

  async function FindBestPassage(question, storedEmbeddings, model) {
    // Embed the question
    const questionResult = await model.embedContent(question, {
      taskType: "QUERY",
    });
    const questionEmbedding = questionResult.embedding;

    // Calculate the dot product for each stored passage
    const dotProducts = storedEmbeddings.map((storedEmbedding) => {
      if (Array.isArray(storedEmbedding.embedding.values)) {
        return dot(questionEmbedding.values, storedEmbedding.embedding.values);
      } else {
        return -Infinity;
      }
    });

    // Find the index of the maximum dot product
    const maxIndex = dotProducts.indexOf(Math.max(...dotProducts));

    // If the similarity is above a certain threshold, generate an answer using the Gemini model
    console.log("Max index:", maxIndex);
    // console.log("Object at max index:", storedEmbeddings[maxIndex]);

    var prompt = `QUESTION: ${question} PASSAGE: ${storedEmbeddings[maxIndex].text} ANSWER:`;
    const threshold = 0.5; // Adjust this value according to your needs

    console.log("Dot product:", dotProducts[maxIndex]);
    console.log("Threshold:", threshold);
    console.log("Condition met:", dotProducts[maxIndex] > threshold);

    if (dotProducts[maxIndex] > threshold) {
      const answer = await newModel.generateContent(prompt);
      console.log("Answer:", answer);

      // Check that answer.response.candidates is defined and has at least one element
      if (
        answer.response &&
        answer.response.candidates &&
        answer.response.candidates.length > 0
      ) {
        // Log the first candidate
        console.log(
          "First candidate:",
          answer.response.candidates[0].content.parts[0].text
        );
        // Return the generated answer
        return answer.response.candidates[0].content.parts[0].text;
      }
    } else {
      return "I'm sorry, I couldn't find a relevant answer to your question.";
    }
  }

  // Embed the question
  const question = "What are the research objectives of the article?";
  const bestPassage = await FindBestPassage(question, storedEmbeddings, model);
  console.log("Best passage:", bestPassage);
}

run();
