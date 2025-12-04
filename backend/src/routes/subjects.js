const express = require('express');
const router = express.Router();
const authenticate = require('../middleware/authenticate');
const { validate } = require('../middleware/validation');
const {
	listSubjectsValidation,
	getSubjectValidation,
	createSubjectValidation,
	updateSubjectValidation,
	deleteSubjectValidation,
} = require('../validators/subjects.validators');

/**
 * @openapi
 * tags:
 *   - name: Subjects
 *     description: Subject and course management
 */
// All subject routes require authentication
router.use(authenticate);

const subjectController = require('../controllers/subject.controller');

/**
 * @openapi
 * /api/subjects:
 *   get:
 *     tags:
 *       - Subjects
 *     summary: List subjects for the user
 *     security:
 *       - bearerAuth: []
 */
router.get('/', validate(listSubjectsValidation), subjectController.listSubjects);

/**
 * @openapi
 * /api/subjects/{id}:
 *   get:
 *     tags:
 *       - Subjects
 *     summary: Get a subject
 */
router.get('/:id', validate(getSubjectValidation), subjectController.getSubject);

/**
 * @openapi
 * /api/subjects:
 *   post:
 *     tags:
 *       - Subjects
 *     summary: Create a subject
 */
router.post('/', validate(createSubjectValidation), subjectController.createSubject);

/**
 * @openapi
 * /api/subjects/{id}:
 *   put:
 *     tags:
 *       - Subjects
 *     summary: Update a subject
 */
router.put('/:id', validate(updateSubjectValidation), subjectController.updateSubject);

/**
 * @openapi
 * /api/subjects/{id}:
 *   delete:
 *     tags:
 *       - Subjects
 *     summary: Delete a subject
 */
router.delete('/:id', validate(deleteSubjectValidation), subjectController.deleteSubject);

/**
 * @openapi
 * /api/subjects/{id}/chapters:
 *   get:
 *     tags:
 *       - Subjects
 *     summary: Get chapters for a subject
 */
router.get('/:id/chapters', validate(getSubjectValidation), subjectController.getSubjectChapters);

/**
 * @openapi
 * /api/subjects/{id}/notes:
 *   get:
 *     tags:
 *       - Subjects
 *     summary: Get notes for a subject
 */
router.get('/:id/notes', validate(getSubjectValidation), subjectController.getSubjectNotes);

module.exports = router;
