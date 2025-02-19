const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const express = require('express');
const cors = require('cors');
const multer = require('multer');

const app = express();
app.use(cors());

// Set up multer for file handling
const upload = multer({ dest: 'uploads/' });

async function sendAudioFile(filePath) {
    const formData = new FormData();
    formData.append('audio', fs.createReadStream(filePath));

    try {
        const response = await axios.post('http://127.0.0.1:5000/process-audio', formData, {
            headers: formData.getHeaders(),
        });
        console.log('Flask response:', response.data);
        return response.data; // Return the response data from the Flask API
    } catch (error) {
        console.error('Error occurred:', error.message); // Log the error to see what went wrong
        throw new Error('Failed to process the audio'); // Rethrow error to be handled in the main endpoint
    }
}

app.post('/findAccuracy', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).send({ error: 'No audio file uploaded' });
        }
        const filePath = req.file.path;
        const result = await sendAudioFile(filePath);

        // Log the result to verify its structure
        console.log("Received from Flask:", result);
        console.log('Uploaded file path:', req.file.path);

        const responseData = { emotion: result.accuracy, confidence : 0.55};
        console.log(responseData)


        res.send(responseData);
        console.log('Attempting to delete file at:', filePath);
        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath); // Clean up uploaded file
            console.log("File deleted successfully");
        } else {
            console.log("File not found, unable to delete");
        }
    } catch (error) {
        console.error("Error processing the audio:", error.message); // Log any error that occurs
        res.status(500).send({ error: error.message });
    }
});

app.listen(3000, () => console.log('Server running on port 3000'));
