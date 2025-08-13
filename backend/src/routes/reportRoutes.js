// backend/src/routes/reportRoutes.js
import express from "express";
import { getReport } from "../controllers/reportController.js";

const router = express.Router();

// POST /api/report
router.post("/", getReport);

export default router;
