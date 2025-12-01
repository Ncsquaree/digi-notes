const express = require('express');
const router = express.Router();
const aiController = require('../controllers/ai.controller');

// POST /api/ai/process-note
router.post('/process-note', aiController.processNote);

// Health
router.get('/health', (req, res) => res.json({ ok: true }));

module.exports = router;

