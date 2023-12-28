import { createRequire } from "module";
const require = createRequire(import.meta.url);

const fs = require("fs");
const pdfParse = require("pdf-parse");

let dataBuffer = fs.readFileSync("User.pdf");

// Function to parse the PDF
function parsePdf() {
  return pdfParse(dataBuffer).then(function(data) {
    // Split the text into chunks
    return data.text.split(); // Adjust the separator as needed
  });
}

parsePdf().then(function(text) {
  console.log(text);
});

export default parsePdf;