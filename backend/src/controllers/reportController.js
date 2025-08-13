// backend/src/controllers/reportController.js
import axios from "axios";

const FLASK_API = process.env.FLASK_API_URL || "http://localhost:5002";

export const getReport = async (req, res) => {
  try {
    const { idea, starting_capital, forecast_years, sector } = req.body;

    const response = await axios.post(
      `${FLASK_API}/idea`,
      { idea, starting_capital, forecast_years, sector },
      { headers: { "Content-Type": "application/json" } }
    );

    // forward Flask response
    res.json(response.data);
  } catch (err) {
    console.error("Error in getReport:", err?.response?.data || err.message);
    const status = err?.response?.status || 500;
    res.status(status).json({ error: "Failed to get report from AI service" });
  }
};
