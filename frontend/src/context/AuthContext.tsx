import React, { createContext, useContext, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient, { setAuthTokens, clearAuthTokens } from '../api/client';

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

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const init = async () => {
      const token = localStorage.getItem('accessToken');
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const resp = await apiClient.get('/auth/me');
        setUser(resp.data.user ?? resp.data);
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
    const resp = await apiClient.post('/auth/login', { email, password });
    const data = resp.data || {};
    const accessToken = data.accessToken ?? data.tokens?.accessToken;
    const refreshToken = data.refreshToken ?? data.tokens?.refreshToken;
    const userData = data.user ?? data;

    if (accessToken) setAuthTokens({ accessToken, refreshToken });
    setUser(userData);
  };

  const signup = async (email: string, password: string, name?: string) => {
    const resp = await apiClient.post('/auth/register', { email, password, name });
    const data = resp.data || {};
    const accessToken = data.accessToken ?? data.tokens?.accessToken;
    const refreshToken = data.refreshToken ?? data.tokens?.refreshToken;
    const userData = data.user ?? data;

    if (accessToken) setAuthTokens({ accessToken, refreshToken });
    setUser(userData);
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
