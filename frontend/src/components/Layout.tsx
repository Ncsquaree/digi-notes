import { ReactNode } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiBookOpen, FiHome, FiLayers, FiTool, FiArchive, FiLogOut } from 'react-icons/fi';
import { useAuth } from '../context/AuthContext';

export default function Layout({ children }: { children: ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();

  const links = [
    { to: '/dashboard', label: 'Dashboard', icon: <FiHome /> },
    { to: '/subjects', label: 'Subjects', icon: <FiBookOpen /> },
    { to: '/templates', label: 'Templates', icon: <FiLayers /> },
    { to: '/tools', label: 'Tools', icon: <FiTool /> },
    { to: '/library', label: 'Library', icon: <FiArchive /> }
  ];

  const { logout } = useAuth();

  async function handleLogout() {
    await logout();
  }

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-[#050509] via-[#05030F] to-black">
      <aside className="w-60 border-r border-white/10 bg-white/5 backdrop-blur-xl flex flex-col">
        <div className="px-6 py-5 border-b border-white/10">
          <div className="flex items-center gap-2">
            <div className="h-9 w-9 rounded-2xl bg-violet-500/80 flex items-center justify-center text-xl font-bold">
              N
            </div>
            <div>
              <div className="font-semibold tracking-wide">Notexa</div>
              <div className="text-xs text-gray-400">Study OS</div>
            </div>
          </div>
        </div>
        <nav className="flex-1 px-3 py-4 flex flex-col gap-1">
          {links.map((link) => {
            const active = location.pathname === link.to;
            return (
              <Link key={link.to} to={link.to}>
                <motion.div
                  whileHover={{ x: 4 }}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm cursor-pointer ${
                    active
                      ? 'bg-violet-500 text-white shadow-lg shadow-violet-500/30'
                      : 'text-gray-300 hover:bg-white/5'
                  }`}
                >
                  <span className="text-lg">{link.icon}</span>
                  <span>{link.label}</span>
                </motion.div>
              </Link>
            );
          })}
        </nav>
        <button
          onClick={handleLogout}
          className="m-3 mt-auto flex items-center gap-2 px-3 py-2 rounded-xl text-sm text-gray-300 hover:bg-white/5"
        >
          <FiLogOut className="text-lg" />
          Logout
        </button>
      </aside>
      <main className="flex-1">
        <div className="h-16 border-b border-white/10 flex items-center justify-between px-8 bg-black/40 backdrop-blur-xl">
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Workspace</div>
            <div className="text-lg font-semibold">Notexa Dashboard</div>
          </div>
        </div>
        <div className="p-8">{children}</div>
      </main>
    </div>
  );
}
