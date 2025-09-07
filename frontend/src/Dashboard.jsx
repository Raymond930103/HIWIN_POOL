import React, { useEffect, useRef, useState } from "react";
import api from "./api";

export default function Dashboard({ token, onLogout }) {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState("idle");
  const [lastLog, setLastLog] = useState("");
  const [healthy, setHealthy] = useState(false);
  const logRef = useRef(null);

  const run = async () => {
    try {
      const data = await api.runPipeline(token);
      setJobId(data.job_id);
      setStatus(data.status);
    } catch (err) {
      if (err.response && err.response.status === 409) {
        setJobId(err.response.data.job_id);
        setStatus(err.response.data.status);
      } else if (err.response && (err.response.status === 401 || err.response.status === 403)) {
        onLogout();
      }
    }
  };

  const poll = async (id) => {
    try {
      const data = await api.getPipelineStatus(token, id);
      setStatus(data.status);
      setLastLog(data.last_log || "");
    } catch (err) {
      if (err.response && (err.response.status === 401 || err.response.status === 403)) {
        onLogout();
      }
    }
  };

  const stop = async () => {
    try {
      const data = await api.stopPipeline(token, jobId);
      setStatus(data.status);
    } catch (err) {
      if (err.response && (err.response.status === 401 || err.response.status === 403)) {
        onLogout();
      }
    }
  };

  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        await api.healthz();
        setHealthy(true);
      } catch {
        setHealthy(false);
      }
      if (jobId) {
        await poll(jobId);
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [jobId, token]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [lastLog]);

  return (
    <div>
      <h2>Dashboard</h2>
      <p>
        Backend: <span style={{ color: healthy ? "green" : "red" }}>●</span>
      </p>
      {status !== "running" ? (
        <button onClick={run}>Run Pipeline</button>
      ) : (
        <button onClick={stop}>Stop</button>
      )}
      <p>Status: {status}</p>
      <pre
        ref={logRef}
        style={{
          height: "200px",
          overflow: "auto",
          background: "#eee",
        }}
      >
        {lastLog}
      </pre>
    </div>
  );
}
