const express = require('express');
const router = express.Router();
const authenticate = require('../middleware/authenticate');
const { validate } = require('../middleware/validation');
const {
	listNotesValidation,
	getNoteValidation,
	createNoteValidation,
	updateNoteValidation,
	deleteNoteValidation,
	processNoteValidation,
} = require('../validators/notes.validators');
const notesController = require('../controllers/notes.controller');

/**
 * @openapi
 * tags:
 *   - name: Notes
 *     description: Note management endpoints
 */
// All note routes require authentication
router.use(authenticate);

/**
 * @openapi
 * /api/notes:
 *   get:
 *     tags:
 *       - Notes
 *     summary: List notes for the current user
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: OK
 */
router.get('/', validate(listNotesValidation), notesController.listNotes);

/**
 * @openapi
 * /api/notes/{id}:
 *   get:
 *     tags:
 *       - Notes
 *     summary: Get a note by id
 *     parameters:
 *       - name: id
 *         in: path
 *         required: true
 *         schema:
 *           type: string
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: OK
 */
router.get('/:id', validate(getNoteValidation), notesController.getNote);

/**
 * @openapi
 * /api/notes:
 *   post:
 *     tags:
 *       - Notes
 *     summary: Create a new note
 *     requestBody:
 *       required: true
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       201:
 *         description: Created
 */
router.post('/', validate(createNoteValidation), notesController.createNote);

/**
 * @openapi
 * /api/notes/{id}:
 *   put:
 *     tags:
 *       - Notes
 *     summary: Update a note
 *     security:
 *       - bearerAuth: []
 */
router.put('/:id', validate(updateNoteValidation), notesController.updateNote);

/**
 * @openapi
 * /api/notes/{id}:
 *   delete:
 *     tags:
 *       - Notes
 *     summary: Delete a note
 *     security:
 *       - bearerAuth: []
 */
router.delete('/:id', validate(deleteNoteValidation), notesController.deleteNote);

/**
 * @openapi
 * /api/notes/{id}/process:
 *   post:
 *     tags:
 *       - Notes
 *     summary: Start AI processing for a note (background)
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       202:
 *         description: Accepted
 */
router.post('/:id/process', validate(processNoteValidation), notesController.processNote);

/**
 * @openapi
 * /api/notes/{id}/status:
 *   get:
 *     tags:
 *       - Notes
 *     summary: Get processing status for a note
 *     security:
 *       - bearerAuth: []
 */
router.get('/:id/status', validate(getNoteValidation), notesController.getProcessingStatus);

module.exports = router;
