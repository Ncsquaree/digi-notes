const db = require('../config/db');

exports.register = async (req, res) => {
	// Placeholder - in production use bcrypt and proper validation
	const { email, name } = req.body;
	try {
		const result = await db.query('INSERT INTO users (email, name, created_at) VALUES ($1,$2,now()) RETURNING id', [email, name]);
		res.json({ id: result.rows[0].id });
	} catch (err) {
		console.error(err);
		res.status(500).json({ error: 'Register failed' });
	}
};

exports.login = async (req, res) => {
	// Placeholder - implement JWT / Cognito
	res.json({ token: 'dev-token', user: { id: 1, name: 'Dev User' } });
};

