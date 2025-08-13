// backend/src/server.js
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import reportRoutes from "./routes/reportRoutes.js";

dotenv.config();
const app = express();

app.use(cors());
app.use(express.json());

// mount the router -> POST http://localhost:5001/api/report
app.use("/api/report", reportRoutes);

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`Backend API running on http://localhost:${PORT}`);
});
