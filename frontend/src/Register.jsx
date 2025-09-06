import React, { useState } from 'react';
import { register as apiRegister, login } from './api';

export default function Register({ onRegister, onSwitch }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState(null);

  const submit = async e => {
    e.preventDefault();
    try {
      await apiRegister({ email, password, display_name: displayName });
      const res = await login(email, password);
      onRegister(res.data.access_token);
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed');
    }
  };

  return (
    <form onSubmit={submit}>
      {error && <p>{error}</p>}
      <input value={email} onChange={e => setEmail(e.target.value)} placeholder="email" />
      <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="password" />
      <input value={displayName} onChange={e => setDisplayName(e.target.value)} placeholder="display name" />
      <button type="submit">Register</button>
      <button type="button" onClick={onSwitch}>Back to Login</button>
    </form>
  );
}
