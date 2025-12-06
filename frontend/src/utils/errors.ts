export interface ValidationError {
  field: string;
  message: string;
}

export function parseApiError(err: any): { message: string; validationErrors: ValidationError[] } {
  const response = err?.response?.data;
  const defaultMessage = err?.message || 'An error occurred';
  // Guard: ensure response is an object (not string/array/primitive)
  if (!response || typeof response !== 'object' || Array.isArray(response)) {
    return { message: defaultMessage, validationErrors: [] };
  }

  const errorObj = response.error || response;
  const message = errorObj.message || response.message || defaultMessage;
  const validationErrors = errorObj.validationErrors || errorObj.validation_errors || [];

  return { message, validationErrors };
}

export function formatValidationErrors(errors: ValidationError[]): string {
  if (!errors || errors.length === 0) return '';
  return errors.map((e) => `• ${e.message}`).join('\n');
}
