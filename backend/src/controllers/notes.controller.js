const db = require('../config/db');

exports.createNote = async (req, res) => {
	// Minimal placeholder: persist metadata
	const { user_id, title, content } = req.body;
	try {
		const result = await db.query(
			'INSERT INTO notes (user_id, title, content, created_at) VALUES ($1,$2,$3,now()) RETURNING id',
			[user_id, title, content]
		);
		res.json({ id: result.rows[0].id });
	} catch (err) {
		console.error(err);
		res.status(500).json({ error: 'Create note failed' });
	}
};

exports.getNote = async (req, res) => {
	const id = req.params.id;
	try {
		const result = await db.query('SELECT * FROM notes WHERE id=$1', [id]);
		res.json(result.rows[0] || {});
	} catch (err) {
		console.error(err);
		res.status(500).json({ error: 'Get note failed' });
	}
};

exports.listNotes = async (req, res) => {
	try {
		const result = await db.query('SELECT * FROM notes ORDER BY created_at DESC LIMIT 100');
		res.json(result.rows);
	} catch (err) {
		console.error(err);
		res.status(500).json({ error: 'List notes failed' });
	}
};

