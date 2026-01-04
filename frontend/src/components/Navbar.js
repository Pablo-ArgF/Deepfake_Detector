import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="flex items-center justify-between flex-wrap p-4 bg-gray-800/80 text-gray-100 shadow-lg border-b border-gray-700 backdrop-blur-md sticky top-0 z-50">
      {/* Logo and Mobile-only Title */}
      <div className="flex items-center flex-shrink-0 mr-6">
        <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <img
            className="w-12 h-12"
            src="/favicon-dark.svg"
            alt="Deepfake Detector Logo"
          />
          <span className="md:hidden text-xl font-black bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            DeepFake
          </span>
        </Link>
      </div>

      {/* Centered Desktop Title */}
      <div className="hidden md:flex absolute left-1/2 transform -translate-x-1/2">
        <Link to="/" className="text-2xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-blue-200 to-purple-500 hover:scale-105 transition-transform">
          DeepFake Detection
        </Link>
      </div>

      {/* Navigation Links */}
      <div className="flex items-center gap-6">
        <Link
          to="/about"
          className="text-gray-300 hover:text-white font-bold transition-all text-sm uppercase tracking-widest hover:border-b-2 border-blue-500 pb-1"
        >
          About
        </Link>
        <Link
          to="/models"
          className="text-gray-300 hover:text-white font-bold transition-all text-sm uppercase tracking-widest hover:border-b-2 border-purple-500 pb-1"
        >
          Models
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
