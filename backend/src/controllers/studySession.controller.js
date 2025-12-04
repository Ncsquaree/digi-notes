const StudySession = require('../models').StudySession;
const logger = require('../utils/logger');

module.exports = {
  async listStudySessions(req, res, next) {
    try {
      const userId = req.user.id;
      const page = req.query.page ? Number(req.query.page) : 1;
      const limit = req.query.limit ? Number(req.query.limit) : 20;
      const startDate = req.query.startDate || null;
      const endDate = req.query.endDate || null;

      // compute offset from page and limit
      const offset = (page - 1) * limit;
      const opts = { limit, offset, startDate, endDate };
      const sessions = await StudySession.findByUserId(userId, opts);

      // sessions is an array of rows. If you need a total count, add an explicit count query.
      const total = Array.isArray(sessions) ? sessions.length : 0;
      const pagination = { page, limit, total };

      logger.info('study_sessions_list', { userId, page, limit, offset });
      return res.json({ success: true, sessions, pagination });
    } catch (err) {
      next(err);
    }
  }
};
