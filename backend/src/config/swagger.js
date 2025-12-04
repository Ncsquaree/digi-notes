const swaggerJSDoc = require('swagger-jsdoc');

const swaggerDefinition = {
  openapi: '3.0.0',
  info: {
    title: 'Digi Notes Backend API',
    version: '1.0.0',
    description: 'Backend API for Digi Notes platform — authentication, notes, flashcards, dashboards, and AI proxies.'
  },
  servers: [
    {
      url: process.env.BACKEND_URL || `http://localhost:${process.env.PORT || 5000}`,
      description: 'Local or container backend server'
    }
  ],
  components: {
    securitySchemes: {
      bearerAuth: {
        type: 'http',
        scheme: 'bearer',
        bearerFormat: 'JWT'
      }
    },
    schemas: {
      ErrorResponse: {
        type: 'object',
        properties: {
          success: { type: 'boolean' },
          error: { type: 'object', properties: { message: { type: 'string' } } },
          requestId: { type: 'string' }
        }
      },
      AuthTokens: {
        type: 'object',
        properties: {
          accessToken: { type: 'string' },
          refreshToken: { type: 'string' }
        }
      }
    }
  },
  security: [{ bearerAuth: [] }]
};

const options = {
  swaggerDefinition,
  apis: [
    './src/routes/*.js',
    './src/controllers/*.js'
  ]
};

const swaggerSpec = swaggerJSDoc(options);

module.exports = { swaggerSpec };
