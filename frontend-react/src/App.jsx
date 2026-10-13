import React, { useState, useEffect } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [serverHealth, setServerHealth] = useState({ online: false, data: null });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Form State
  const [formData, setFormData] = useState({
    gender: 'Female',
    SeniorCitizen: 0,
    Partner: 'No',
    Dependents: 'No',
    tenure: 12,
    PhoneService: 'Yes',
    MultipleLines: 'No',
    InternetService: 'Fiber optic',
    OnlineSecurity: 'No',
    OnlineBackup: 'No',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'No',
    StreamingMovies: 'No',
    Contract: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 70.0,
    TotalCharges: 840.0,
  });

  // Batch CSV State
  const [csvFile, setCsvFile] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState(null);
  const [batchSuccess, setBatchSuccess] = useState(false);

  // Check health on mount
  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setServerHealth({ online: true, data });
      } else {
        setServerHealth({ online: false, data: null });
      }
    } catch {
      setServerHealth({ online: false, data: null });
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    const parsedValue =
      name === 'SeniorCitizen' || name === 'tenure'
        ? parseInt(value, 10)
        : name === 'MonthlyCharges'
        ? parseFloat(value)
        : value;

    setFormData((prev) => {
      const updated = { ...prev, [name]: parsedValue };
      // Dynamically recalculate TotalCharges if tenure or MonthlyCharges changes
      if (name === 'tenure' || name === 'MonthlyCharges') {
        const tenureVal = name === 'tenure' ? parsedValue : prev.tenure;
        const monthlyVal = name === 'MonthlyCharges' ? parsedValue : prev.MonthlyCharges;
        updated.TotalCharges = parseFloat((tenureVal * monthlyVal).toFixed(2));
      }
      return updated;
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Prediction failed');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCsvUpload = async (e) => {
    e.preventDefault();
    if (!csvFile) return;

    setBatchLoading(true);
    setBatchError(null);
    setBatchSuccess(false);

    try {
      const body = new FormData();
      body.append('file', csvFile);

      const res = await fetch(`${API_BASE}/predict/csv`, {
        method: 'POST',
        body,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Batch prediction failed');
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'scored_customers_churn.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setBatchSuccess(true);
    } catch (err) {
      setBatchError(err.message);
    } finally {
      setBatchLoading(false);
    }
  };

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <h1>📊 Customer Churn AI Intelligence</h1>
        </div>
        <div>
          {serverHealth.online ? (
            <span className="badge badge-success">
              🟢 FastAPI Engine: Online (v{serverHealth.data?.version})
            </span>
          ) : (
            <span className="badge badge-danger">🔴 FastAPI Engine: Disconnected</span>
          )}
        </div>
      </header>

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab-btn ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          Single Customer Predictor
        </button>
        <button
          className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          Batch CSV Scorer
        </button>
      </div>

      {activeTab === 'single' ? (
        <div className="grid-main">
          {/* Input Form */}
          <div className="card">
            <h2 className="card-title">Customer Information</h2>
            <form onSubmit={handleSubmit} className="form-grid">
              <div className="form-group">
                <label>Gender</label>
                <select className="form-control" name="gender" value={formData.gender} onChange={handleChange}>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <div className="form-group">
                <label>Senior Citizen</label>
                <select className="form-control" name="SeniorCitizen" value={formData.SeniorCitizen} onChange={handleChange}>
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Partner</label>
                <select className="form-control" name="Partner" value={formData.Partner} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Dependents</label>
                <select className="form-control" name="Dependents" value={formData.Dependents} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Tenure ({formData.tenure} months)</label>
                <input
                  type="range"
                  className="form-control"
                  name="tenure"
                  min="0"
                  max="72"
                  value={formData.tenure}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group">
                <label>Contract Term</label>
                <select className="form-control" name="Contract" value={formData.Contract} onChange={handleChange}>
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>

              <div className="form-group">
                <label>Phone Service</label>
                <select className="form-control" name="PhoneService" value={formData.PhoneService} onChange={handleChange}>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Internet Service</label>
                <select className="form-control" name="InternetService" value={formData.InternetService} onChange={handleChange}>
                  <option value="DSL">DSL</option>
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Tech Support</label>
                <select className="form-control" name="TechSupport" value={formData.TechSupport} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div className="form-group">
                <label>Online Security</label>
                <select className="form-control" name="OnlineSecurity" value={formData.OnlineSecurity} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div className="form-group">
                <label>Payment Method</label>
                <select className="form-control" name="PaymentMethod" value={formData.PaymentMethod} onChange={handleChange}>
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
              </div>

              <div className="form-group">
                <label>Paperless Billing</label>
                <select className="form-control" name="PaperlessBilling" value={formData.PaperlessBilling} onChange={handleChange}>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Monthly Charges ($)</label>
                <input
                  type="number"
                  step="0.01"
                  className="form-control"
                  name="MonthlyCharges"
                  value={formData.MonthlyCharges}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group">
                <label>Total Charges ($)</label>
                <input
                  type="number"
                  step="0.01"
                  className="form-control"
                  name="TotalCharges"
                  value={formData.TotalCharges}
                  onChange={handleChange}
                />
              </div>

              <button type="submit" className="btn-primary" disabled={loading || !serverHealth.online}>
                {loading ? 'Evaluating Model...' : 'Calculate Churn Probability'}
              </button>
            </form>
          </div>

          {/* Results Panel */}
          <div className="card">
            <h2 className="card-title">Inference Results</h2>
            {error && <div className="badge badge-danger" style={{ marginBottom: '1rem', width: '100%' }}>{error}</div>}

            {result ? (
              <div className="result-box">
                <div className="gauge-container">
                  <span
                    className="badge"
                    style={{
                      backgroundColor:
                        result.risk_level === 'High'
                          ? 'rgba(239, 68, 68, 0.2)'
                          : result.risk_level === 'Medium'
                          ? 'rgba(245, 158, 11, 0.2)'
                          : 'rgba(16, 185, 129, 0.2)',
                      color:
                        result.risk_level === 'High'
                          ? '#ef4444'
                          : result.risk_level === 'Medium'
                          ? '#f59e0b'
                          : '#10b981',
                      border: `1px solid ${
                        result.risk_level === 'High'
                          ? '#ef4444'
                          : result.risk_level === 'Medium'
                          ? '#f59e0b'
                          : '#10b981'
                      }`,
                    }}
                  >
                    {result.risk_level} Churn Risk
                  </span>
                  <div
                    className="probability-metric"
                    style={{
                      color:
                        result.risk_level === 'High'
                          ? '#ef4444'
                          : result.risk_level === 'Medium'
                          ? '#f59e0b'
                          : '#10b981',
                    }}
                  >
                    {(result.churn_probability * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Model Confidence: {result.confidence}</div>

                  <div className="progress-track">
                    <div
                      className="progress-bar"
                      style={{
                        width: `${result.churn_probability * 100}%`,
                        backgroundColor:
                          result.risk_level === 'High'
                            ? '#ef4444'
                            : result.risk_level === 'Medium'
                            ? '#f59e0b'
                            : '#10b981',
                      }}
                    />
                  </div>
                </div>

                <div className="recommendation-card">
                  <h4>💡 Proactive Retention Strategy</h4>
                  <p>{result.key_recommendation}</p>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '3rem 1rem', color: '#64748b' }}>
                <p>Select customer features and click Calculate to view real-time churn risk and retention advice.</p>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Batch Tab */
        <div className="card">
          <h2 className="card-title">Batch Customer CSV Inference</h2>
          <p style={{ color: '#94a3b8', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
            Upload customer call records or customer database exports. The API will score each record and return an enriched CSV file.
          </p>

          <form onSubmit={handleCsvUpload} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxWidth: '500px' }}>
            <input
              type="file"
              accept=".csv"
              className="form-control"
              onChange={(e) => setCsvFile(e.target.files[0])}
            />

            <button type="submit" className="btn-primary" style={{ gridColumn: 'unset' }} disabled={batchLoading || !csvFile}>
              {batchLoading ? 'Scoring Batch Records...' : 'Score CSV & Download Report'}
            </button>
          </form>

          {batchSuccess && (
            <div className="badge badge-success" style={{ marginTop: '1rem' }}>
              ✅ Batch scoring complete! Download started automatically.
            </div>
          )}
          {batchError && (
            <div className="badge badge-danger" style={{ marginTop: '1rem' }}>
              ❌ {batchError}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
