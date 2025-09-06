import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

export function login(email, password) {
  return api.post('/auth/login', { email, password });
}

export function register(data) {
  return api.post('/auth/register', data);
}

export function startMatch(token, data) {
  return api.post('/matches', data, { headers: { Authorization: `Bearer ${token}` }});
}

export function shoot(token, angle, power, spin=0) {
  return api.post('/robot/shoot', { angle, power, spin }, { headers: { Authorization: `Bearer ${token}` }});
}

export default api;
