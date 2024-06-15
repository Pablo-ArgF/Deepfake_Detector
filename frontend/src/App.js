import './App.css';
import Navbar from './components/Navbar.js';
import BodyView from './components/BodyView.js';
import ModelDetails from './components/ModelDetails.js';
import About from './components/About.js';
import {useState } from 'react';

function App() {
  //usestates for changing the main view between bodyView / ModelDetails / About
  const [view, setView] = useState('BodyView');
  return (
    <div className="App">
      {/*Navbar receibes the setView useState to update it*/}
      <Navbar setView={setView}/>
      {
        view.match('BodyView') ? <BodyView/> : view.match('About') ? <About/> : <ModelDetails/>
      }
      
    </div>
  );
}

export default App;
