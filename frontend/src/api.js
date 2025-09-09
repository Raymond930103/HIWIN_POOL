import axios from "axios";

const instance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
});

export default {
  async login({ email, password }) {
    const params = new URLSearchParams();
    params.append("username", email);
    params.append("password", password);
    const res = await instance.post("/auth/login", params);
    return res.data;
  },
  async runPipeline(token) {
    const res = await instance.post(
      "/pipeline/run",
      {},
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return res.data;
  },
  async getPipelineStatus(token, jobId) {
    const res = await instance.get("/pipeline/status", {
      params: { job_id: jobId },
      headers: { Authorization: `Bearer ${token}` },
    });
    return res.data;
  },
  async stopPipeline(token, jobId) {
    const res = await instance.post(
      "/pipeline/stop",
      { job_id: jobId },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return res.data;
  },
  async healthz() {
    const res = await instance.get("/healthz");
    return res.data;
  },
};
