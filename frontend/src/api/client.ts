import axios, { AxiosError } from 'axios';

const baseURL = (import.meta.env.VITE_API_URL as string) || '';

export const apiClient = axios.create({
  baseURL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

export function setAuthTokens(tokens: { accessToken: string; refreshToken?: string }) {
  if (tokens.accessToken) localStorage.setItem('accessToken', tokens.accessToken);
  if (tokens.refreshToken) localStorage.setItem('refreshToken', tokens.refreshToken);
}

export function clearAuthTokens() {
  localStorage.removeItem('accessToken');
  localStorage.removeItem('refreshToken');
}

// Request interceptor: attach access token
apiClient.interceptors.request.use((config) => {
  try {
    const token = localStorage.getItem('accessToken');
    if (token) {
      config.headers = config.headers || {};
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      config.headers.Authorization = `Bearer ${token}`;
    }
  } catch (e) {
    // ignore
  }
  return config;
});

// Response interceptor: try a single refresh on 401
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError & { config?: any }) => {
    const originalRequest = error?.config;

    if (
      error?.response?.status === 401 &&
      originalRequest &&
      !originalRequest._retry &&
      !originalRequest.url?.includes('/auth/refresh-token')
    ) {
      originalRequest._retry = true;
      const refreshToken = localStorage.getItem('refreshToken');
      if (!refreshToken) {
        clearAuthTokens();
        window.location.href = '/login';
        return Promise.reject(error);
      }

      try {
        const resp = await apiClient.post('/auth/refresh-token', { refreshToken });
        const data = resp.data || {};
        const accessToken = data.accessToken ?? data.tokens?.accessToken;
        const newRefreshToken = data.refreshToken ?? data.tokens?.refreshToken ?? refreshToken;

        if (accessToken) {
          setAuthTokens({ accessToken, refreshToken: newRefreshToken });
          apiClient.defaults.headers.common['Authorization'] = `Bearer ${accessToken}`;
          if (originalRequest.headers) originalRequest.headers['Authorization'] = `Bearer ${accessToken}`;
          return apiClient(originalRequest);
        }
      } catch (e) {
        clearAuthTokens();
        window.location.href = '/login';
        return Promise.reject(e);
      }
    }

    return Promise.reject(error);
  }
);

export default apiClient;
