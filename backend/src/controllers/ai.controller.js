const axios = require('axios');

exports.processNote = async (req, res) => {
	try {
		const { s3Url, userId } = req.body;
		// Proxy to AI service (FastAPI) — URL should be in env
		const aiUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000/process-note';
		const resp = await axios.post(aiUrl, { s3Url, userId });
		res.json(resp.data);
	} catch (err) {
		console.error(err);
		res.status(500).json({ error: 'AI processing failed' });
	}
};

