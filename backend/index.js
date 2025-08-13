const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const axios = require('axios');

dotenv.config();
const app = express();

app.use(cors());
app.use(express.json());

// Forward request to AI Service
app.post('/forecast', async (req, res) => {
    try {
        const aiResponse = await axios.post(
            `${process.env.FLASK_API_URL}/forecast`,
            req.body
        );
        res.json(aiResponse.data);
    } catch (error) {
        console.error("Error calling AI Service:", error.message);
        res.status(500).json({ error: "Failed to get forecast from AI service" });
    }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Backend running on port ${PORT}`);
});
