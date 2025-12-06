import React, { createContext, useContext, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient, { setAuthTokens, clearAuthTokens } from '../api/client';
import { parseApiError } from '../utils/errors';

type User = {
  id: string;
  email: string;
  name?: string;
  [key: string]: any;
};

type AuthContextType = {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, name?: string) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Helper: normalize user-like objects by ensuring `name` exists when possible
function normalizeUser(user: any) {
  if (!user || typeof user !== 'object') return user;
  if (user.name) return user;
  const fn = user.firstName || user.first_name || '';
  const ln = user.lastName || user.last_name || '';
  const combined = [fn, ln].filter(Boolean).join(' ').trim();
  if (combined) {
    // Mutate shallowly to preserve references used elsewhere
    // but return the object for clarity
    user.name = combined;
  }
  return user;
}

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const init = async () => {
      // Support both 'accessToken' and legacy 'token' localStorage keys
      const token = localStorage.getItem('accessToken') || localStorage.getItem('token');
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const resp = await apiClient.get('/auth/me');
        // Backend returns { success: true, data: { user: {...} } }
        // Unwrap the common `data` wrapper first for compatibility
        const payload = resp.data?.data ?? resp.data;
        const userData = normalizeUser(payload?.user ?? payload);
        setUser(userData);
      } catch (e) {
        clearAuthTokens();
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    init();
  }, []);

  const login = async (email: string, password: string) => {
    try {
      const resp = await apiClient.post('/auth/login', { email, password });
      // Backend returns { success: true, data: { user, accessToken, refreshToken } }
      const payload = resp.data?.data ?? resp.data;
      const accessToken = payload?.accessToken ?? payload?.tokens?.accessToken;
      const refreshToken = payload?.refreshToken ?? payload?.tokens?.refreshToken;
      const userData = normalizeUser(payload?.user ?? payload);

      if (accessToken) setAuthTokens({ accessToken, refreshToken });
      setUser(userData);
    } catch (err: any) {
      // Re-throw the original axios error so pages can parse it with parseApiError
      throw err;
    }
  };

  const signup = async (email: string, password: string, name?: string) => {
    try {
      // Map frontend `name` to backend `firstName` to satisfy strict validation
      const resp = await apiClient.post('/auth/register', { email, password, firstName: name });
      // Backend returns { success: true, data: { user, accessToken, refreshToken } }
      const payload = resp.data?.data ?? resp.data;
      const accessToken = payload?.accessToken ?? payload?.tokens?.accessToken;
      const refreshToken = payload?.refreshToken ?? payload?.tokens?.refreshToken;
      const userData = normalizeUser(payload?.user ?? payload);

      if (accessToken) setAuthTokens({ accessToken, refreshToken });
      setUser(userData);
    } catch (err: any) {
      // Re-throw the original axios error so pages can parse it with parseApiError
      throw err;
    }
  };

  const logout = async () => {
    try {
      const refreshToken = localStorage.getItem('refreshToken');
      await apiClient.post('/auth/logout', { refreshToken });
    } catch (e) {
      // ignore errors
    }
    clearAuthTokens();
    setUser(null);
    navigate('/login');
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};
