const { User } = require('../models');
const db = require('../config/database');
const { NotFoundError, ConflictError } = require('../utils/errors');
const logger = require('../utils/logger');

class UserService {
  static async getUserById(userId) {
    const user = await User.findById(userId);
    if (!user) throw new NotFoundError('User not found');
    return user;
  }

  static async getUserByEmail(email) {
    return User.findByEmail(email);
  }

  static async createUser({ email, password_hash, first_name, last_name }) {
    const existing = await User.findByEmail(email);
    if (existing) throw new ConflictError('Email already registered');
    const user = await User.create({ email, password_hash, first_name, last_name });
    logger.info('User created', { userId: user.id });
    return user;
  }

  static async updateUser(userId, fields) {
    const user = await User.findById(userId);
    if (!user) throw new NotFoundError('User not found');
    const updated = await User.update(userId, fields);
    return updated;
  }

  static async deleteUser(userId) {
    const user = await User.findById(userId);
    if (!user) throw new NotFoundError('User not found');
    await User.delete(userId);
    return true;
  }

  static async listUsers(opts = {}) {
    return User.list(opts);
  }

  static async updateLastLogin(userId) {
    await User.updateLastLogin(userId);
  }
}

module.exports = UserService;
