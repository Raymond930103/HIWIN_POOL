import React, { useState } from 'react';
import Login from './Login';
import Register from './Register';
import Dashboard from './Dashboard';
import MatchPlay from './MatchPlay';

export default function App() {
  const [token, setToken] = useState(null);
  const [matchId, setMatchId] = useState(null);
  const [mode, setMode] = useState('login');

  if (!token) {
    if (mode === 'login') {
      return <Login onLogin={setToken} onSwitch={() => setMode('register')} />;
    }
    return <Register onRegister={setToken} onSwitch={() => setMode('login')} />;
  }
  if (!matchId) return <Dashboard token={token} onStart={setMatchId} />;
  return <MatchPlay token={token} />;
}
