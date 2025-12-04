const express = require('express');
const router = express.Router();

const authRoutes = require('./auth');
const notesRoutes = require('./notes');
const subjectsRoutes = require('./subjects');
const chaptersRoutes = require('./chapters');
const flashcardsRoutes = require('./flashcards');
const aiRoutes = require('./ai');
const dashboardRoutes = require('./dashboard');
const s3Routes = require('./s3');
const studySessionsRoutes = require('./studySessions');

router.use('/auth', authRoutes);
router.use('/notes', notesRoutes);
router.use('/subjects', subjectsRoutes);
router.use('/chapters', chaptersRoutes);
router.use('/flashcards', flashcardsRoutes);
router.use('/ai', aiRoutes);
router.use('/dashboard', dashboardRoutes);
router.use('/s3', s3Routes);
router.use('/study-sessions', studySessionsRoutes);

router.use((req, res) => {
  res.status(404).json({ success: false, error: { message: 'Route not found', requestId: req.id } });
});

module.exports = router;
