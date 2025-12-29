import React from 'react';

const Navbar = ({ setView }) => {
  return (
    <nav className="flex items-center justify-between flex-wrap p-6 bg-gray-800 text-gray-100 shadow-lg border-b border-gray-700">
      <div
        className="flex items-center flex-shrink-0 mr-6 cursor-pointer hover:opacity-80 transition-opacity"
        onClick={() => setView('BodyView')}
      >
        <img
          className="w-48 md:w-64"
          src="./Logo Universidad de Oviedo.png"
          alt="Logo Universidad de Oviedo"
        />
        <span className="hidden md:block ml-4 text-xl font-bold tracking-tight">
          DeepFake Detection
        </span>
      </div>

      <div className="block lg:hidden">
        {/* Mobile menu button could go here if needed */}
      </div>

      <div className="w-full block flex-grow lg:flex lg:items-center lg:w-auto">
        <div className="text-sm lg:flex-grow flex justify-end gap-8">
          <button
            onClick={() => setView('About')}
            className="block mt-4 lg:inline-block lg:mt-0 text-gray-300 hover:text-white font-semibold transition-colors text-lg"
          >
            About
          </button>
          <button
            onClick={() => setView('ModelDetails')}
            className="block mt-4 lg:inline-block lg:mt-0 text-gray-300 hover:text-white font-semibold transition-colors text-lg"
          >
            Model
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
