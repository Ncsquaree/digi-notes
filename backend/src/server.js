const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const aiRoutes = require('./routes/ai.routes');
const notesRoutes = require('./routes/notes.routes');
const userRoutes = require('./routes/user.routes');

dotenv.config();
const app = express();
app.use(cors());
app.use(express.json());

app.use('/api/ai', aiRoutes);
app.use('/api/notes', notesRoutes);
app.use('/api/users', userRoutes);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
	console.log(`Backend server running on port ${PORT}`);
});

module.exports = app;

