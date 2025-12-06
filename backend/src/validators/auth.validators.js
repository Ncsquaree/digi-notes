const { body } = require('express-validator');
const { isEmail, isString, isOptionalString } = require('../middleware/validation');

/**
 * Auth validation schemas
 * Uses environment-driven password policy flags where applicable.
 */
const PASSWORD_MIN_LENGTH = Number(process.env.PASSWORD_MIN_LENGTH || 8);
const REQUIRE_UPPER = process.env.PASSWORD_REQUIRE_UPPERCASE === 'true';
const REQUIRE_LOWER = process.env.PASSWORD_REQUIRE_LOWERCASE === 'true';
const REQUIRE_NUMBER = process.env.PASSWORD_REQUIRE_NUMBERS === 'true';
const REQUIRE_SPECIAL = process.env.PASSWORD_REQUIRE_SPECIAL === 'true';

const passwordPolicyChecks = [];
passwordPolicyChecks.push(body('password').isLength({ min: PASSWORD_MIN_LENGTH, max: 128 }).withMessage(`Password must be at least ${PASSWORD_MIN_LENGTH} characters`));
if (REQUIRE_UPPER) passwordPolicyChecks.push(body('password').matches(/[A-Z]/).withMessage('Password must contain an uppercase letter'));
if (REQUIRE_LOWER) passwordPolicyChecks.push(body('password').matches(/[a-z]/).withMessage('Password must contain a lowercase letter'));
if (REQUIRE_NUMBER) passwordPolicyChecks.push(body('password').matches(/[0-9]/).withMessage('Password must contain a number'));
if (REQUIRE_SPECIAL) passwordPolicyChecks.push(body('password').matches(/[!@#\$%\^&\*\(\)_\+\-=`~\[\]{};:'"\\|,<.>\/\?]/).withMessage('Password must contain a special character'));

const registerValidation = [
  isEmail('email'),
  ...passwordPolicyChecks,
  isOptionalString('firstName', 100),
  isOptionalString('lastName', 100),
];

const loginValidation = [
  isEmail('email'),
  body('password').exists().withMessage('password is required'),
];

const refreshTokenValidation = [
  body('refreshToken').exists().isString().withMessage('refreshToken is required'),
];

const logoutValidation = [
  body('refreshToken').exists().isString().withMessage('refreshToken is required'),
];

module.exports = {
  registerValidation,
  loginValidation,
  refreshTokenValidation,
  logoutValidation,
};
