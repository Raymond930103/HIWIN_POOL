import React, { useState } from 'react';
import { login } from './api';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const submit = async e => {
    e.preventDefault();
    const res = await login(email, password);
    onLogin(res.data.access_token);
  };

  return (
    <form onSubmit={submit}>
      <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="email" />
      <input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="password" />
      <button type="submit">Login</button>
    </form>
  );
}
