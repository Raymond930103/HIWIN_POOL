import React from 'react';
import { startMatch } from './api';

export default function Dashboard({ token, onStart }) {
  const create = async () => {
    const res = await startMatch(token, { mode: '9', difficulty: 'easy' });
    onStart(res.data.id);
  };
  return <button onClick={create}>Start Match</button>;
}
