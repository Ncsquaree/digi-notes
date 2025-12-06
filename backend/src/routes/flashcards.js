const express = require('express');
const router = express.Router();
const authenticate = require('../middleware/authenticate');
const { validate } = require('../middleware/validation');
const {
	listFlashcardsValidation,
	dueFlashcardsValidation,
	getFlashcardValidation,
	createFlashcardValidation,
	updateFlashcardValidation,
	deleteFlashcardValidation,
	reviewFlashcardValidation,
} = require('../validators/flashcards.validators');

const flashcardController = require('../controllers/flashcard.controller');

/**
 * @openapi
 * tags:
 *   - name: Flashcards
 *     description: Flashcard management and review
 */
// All flashcard routes require authentication
router.use(authenticate);

/**
 * @openapi
 * /api/flashcards:
 *   get:
 *     tags:
 *       - Flashcards
 *     summary: List flashcards
 *     security:
 *       - bearerAuth: []
 */
router.get('/', validate(listFlashcardsValidation), flashcardController.listFlashcards);

/**
 * @openapi
 * /api/flashcards/due:
 *   get:
 *     tags:
 *       - Flashcards
 *     summary: Get due flashcards
 */
router.get('/due', validate(dueFlashcardsValidation), flashcardController.getDueFlashcards);

/**
 * Stats route must be defined before the param-based routes so `/stats` is not
 * incorrectly routed to `/:id`.
 */
router.get('/stats', flashcardController.getFlashcardStats);

/**
 * @openapi
 * /api/flashcards/{id}:
 *   get:
 *     tags:
 *       - Flashcards
 *     summary: Get a flashcard
 */
router.get('/:id', validate(getFlashcardValidation), flashcardController.getFlashcard);

/**
 * @openapi
 * /api/flashcards:
 *   post:
 *     tags:
 *       - Flashcards
 *     summary: Create a flashcard
 */
router.post('/', validate(createFlashcardValidation), flashcardController.createFlashcard);

/**
 * @openapi
 * /api/flashcards/{id}:
 *   put:
 *     tags:
 *       - Flashcards
 *     summary: Update a flashcard
 */
router.put('/:id', validate(updateFlashcardValidation), flashcardController.updateFlashcard);

/**
 * @openapi
 * /api/flashcards/{id}:
 *   delete:
 *     tags:
 *       - Flashcards
 *     summary: Delete a flashcard
 */
router.delete('/:id', validate(deleteFlashcardValidation), flashcardController.deleteFlashcard);

/**
 * @openapi
 * /api/flashcards/{id}/review:
 *   post:
 *     tags:
 *       - Flashcards
 *     summary: Review a flashcard (spaced repetition)
 */
router.post('/:id/review', validate(reviewFlashcardValidation), flashcardController.reviewFlashcard);

/**
 * @openapi
 * /api/flashcards/stats:
 *   get:
 *     tags:
 *       - Flashcards
 *     summary: Flashcard statistics
 */
router.get('/stats', flashcardController.getFlashcardStats);

module.exports = router;
