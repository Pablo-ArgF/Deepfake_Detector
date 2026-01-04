import './App.css';
import Navbar from './components/Navbar.js';
import BodyView from './components/BodyView.js';
import ModelDetails from './components/ModelDetails.js';
import About from './components/About.js';
import { Routes, Route } from 'react-router-dom';

function App() {
  return (
    <div className="App">
      <Navbar />
      <Routes>
        <Route path="/" element={<BodyView />} />
        <Route path="/about" element={<About />} />
        <Route path="/models" element={<ModelDetails />} />
        {/* Analysis routes will be handled within BodyView or as separate pages */}
        <Route path="/:uuid/:type" element={<BodyView />} />
      </Routes>
    </div>
  );
}

export default App;
