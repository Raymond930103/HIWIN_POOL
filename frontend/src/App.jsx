import React, { useState } from 'react';
import Login from './Login';
import Dashboard from './Dashboard';
import MatchPlay from './MatchPlay';

export default function App() {
  const [token, setToken] = useState(null);
  const [matchId, setMatchId] = useState(null);

  if (!token) return <Login onLogin={setToken} />;
  if (!matchId) return <Dashboard token={token} onStart={setMatchId} />;
  return <MatchPlay token={token} />;
}
