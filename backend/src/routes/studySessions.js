const express = require('express');
const router = express.Router();
const authenticate = require('../middleware/authenticate');
const { validate } = require('../middleware/validation');
const { listStudySessionsValidation } = require('../validators/studySession.validators');
const studySessionController = require('../controllers/studySession.controller');

/**
 * @openapi
 * tags:
 *   - name: StudySessions
 *     description: Study session and scheduling endpoints
 */
router.use(authenticate);

/**
 * @openapi
 * /api/study-sessions:
 *   get:
 *     tags:
 *       - StudySessions
 *     summary: List study sessions for user
 *     security:
 *       - bearerAuth: []
 */
router.get('/', validate(listStudySessionsValidation), studySessionController.listStudySessions);

module.exports = router;
