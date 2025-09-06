import React, { useState } from 'react';
import { shoot } from './api';

export default function MatchPlay({ token }) {
  const [angle, setAngle] = useState(45);
  const [power, setPower] = useState(0.8);

  const send = () => {
    shoot(token, angle, power);
  };

  return (
    <div>
      <div>Angle: <input type="number" value={angle} onChange={e=>setAngle(parseFloat(e.target.value))} /></div>
      <div>Power: <input type="number" step="0.1" value={power} onChange={e=>setPower(parseFloat(e.target.value))} /></div>
      <button onClick={send}>Shoot</button>
    </div>
  );
}
