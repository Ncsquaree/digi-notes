const express = require('express');
const router = express.Router();
const authenticate = require('../middleware/authenticate');
const dashboardController = require('../controllers/dashboard.controller');

/**
 * @openapi
 * tags:
 *   - name: Dashboard
 *     description: Aggregated stats and metrics
 */
// Protect all dashboard routes
router.use(authenticate);

/**
 * @openapi
 * /api/dashboard:
 *   get:
 *     tags:
 *       - Dashboard
 *     summary: Get dashboard stats
 *     security:
 *       - bearerAuth: []
 */
router.get('/', dashboardController.getDashboardStats);

module.exports = router;
