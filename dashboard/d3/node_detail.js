// Load graph data locally for node metrics
let graphData = null;

async function loadGraph() {
  if (graphData) return graphData; // cache after first load
  const response = await fetch("data/all/graph_internal.json");
  graphData = await response.json();
  return graphData;
}

// Supabase client config (for emails)
const SUPABASE_URL = "https://klzdfxpuxqgfdwvgrbvr.supabase.co";
const SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtsemRmeHB1eHFnZmR3dmdyYnZyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDIxNjEzMywiZXhwIjoyMDc1NzkyMTMzfQ.ngiFc2MVovqgsbGrLY4z0sqsM8mvYkSfxkQ-ayVZ4pg";
const supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);

// HTML elements
const nodeBox = document.getElementById("node-info");
const emailBox = document.getElementById("email-list");
const params = new URLSearchParams(window.location.search);
const nodeId = params.get("id");

if (!nodeId) {
  nodeBox.innerHTML = "<h2>No node ID provided!</h2>";
} else {
  renderNode(nodeId);
  loadEmails(nodeId);  // Supabase fetch
}

// Load node info from JSON 
async function renderNode(id) {
  const data = await loadGraph();
  const node = data.nodes.find(n => n.id === id);

  if (!node) {
    nodeBox.innerHTML = `<h2>Node not found in local graph data.</h2>`;
    return;
  }

  nodeBox.innerHTML = `
    <h2>${node.id}</h2>

    <h4 class="mt-3 mb-2 text-uppercase fw-bold">General Information</h4>
    <p><span class="label">Aliases:</span> ${(node.aliases && node.aliases.length > 0) ? node.aliases.join(", ") : "N/A"}</p>
    <p><span class="label">Community:</span> ${node.community ?? "N/A"}</p>

    <p><span class="label">Risk Label:</span> ${node.risk_label ?? "None"}</p>
    <p><span class="label">Risk Intensity:</span> ${(node.risk_intensity ?? 0).toFixed(3)}</p>

    <p><span class="label">Sentiment Label:</span> ${node.sentiment_label ?? "N/A"}</p>
    <p><span class="label">Sentiment Intensity:</span> ${(node.sentiment_intensity ?? 0).toFixed(3)}</p>

    <p><span class="label">Emotion Label:</span> ${node.emotion_label ?? "N/A"}</p>
    <p><span class="label">Emotion Intensity:</span> ${(node.emotion_intensity ?? 0).toFixed(3)}</p>

    <p><span class="label">Total Emails:</span> ${node.total_emails ?? 0}</p>
    <p><span class="label">Risk Emails:</span> ${node.risk_emails ?? 0}</p>

    <p><span class="label">First Date:</span> ${node.first_date ? new Date(node.first_date).toLocaleString() : "N/A"}</p>
    <p><span class="label">Last Date:</span> ${node.last_date ? new Date(node.last_date).toLocaleString() : "N/A"}</p>
    <p><span class="label">Years Active:</span> ${(node.years && node.years.length > 0) ? node.years.join(", ") : "N/A"}</p>

    <h4 class="mt-4 mb-2 text-uppercase fw-bold">Network Metrics</h4>
    <p><span class="label">Degree:</span> ${node.degree ?? 0}</p>
    <p><span class="label">In-Degree:</span> ${node.in_degree ?? 0}</p>
    <p><span class="label">Out-Degree:</span> ${node.out_degree ?? 0}</p>
    <p><span class="label">Weighted Degree:</span> ${(node.w_degree ?? 0).toFixed(2)}</p>
    <p><span class="label">Pagerank:</span> ${(node.pagerank ?? 0).toExponential(2)}</p>
    <p><span class="label">Clustering Coefficient:</span> ${(node.clustering_coef ?? 0).toFixed(3)}</p>
    <p><span class="label">K-Core:</span> ${node.kcore ?? 0}</p>
  `;

}

// Fetch emails dynamically from Supabase
async function loadEmails(personId) {
  try {
    let { data, error } = await supabase
      .from("emails_clean_9902")
      .select("email_id, from_norm, to_norm, cc_norm, bcc_norm, dt_utc, subject, body_raw, path, risk_label, final_score, sentiment_label, sentiment_score, emotion_label, emotion_score")
      .eq("from_norm", personId)
      .order("dt_utc", { ascending: false })

    if (error) throw error;
    else {
    data = data
      .filter(d => (d.body_raw?.split(/\s+/).length ?? 0) > 50) // keep only long bodies

      .sort((a, b) => {
        const nonRiskLabel = "This email appears routine, compliant, and shows no indication of risk or wrongdoing.";

        const aIsRoutine = a.risk_label === nonRiskLabel;
        const bIsRoutine = b.risk_label === nonRiskLabel;
        if (aIsRoutine !== bIsRoutine) return aIsRoutine ? 1 : -1;


        const riskDiff = (b.final_score ?? 0) - (a.final_score ?? 0);
        if (riskDiff !== 0) return riskDiff;


        // const dateDiff = new Date(b.dt_utc) - new Date(a.dt_utc);
        // if (dateDiff !== 0) return dateDiff;
      });


  }


    if (!data || data.length === 0) {
      emailBox.innerHTML = `<h3>No emails found for this person.</h3>`;
      return;
    }

    const extractFolder = (path) => {
      if (!path) return "N/A";
      const match = path.match(/\\([^\\]+)\\/);
      return match ? match[1].toLowerCase() : "N/A";
    };

    // Build email cards
    let html = `<h3>Emails related to ${personId}</h3><div class="scrollable">`;
    for (const e of data) {
      const folder = extractFolder(e.path);
      html += `
        <div class="email-card">
          <h4>${e.subject || "(No Subject)"}</h4>
          <small><b>Email ID:</b> ${e.email_id || "N/A"}</small><br>
          <small><b>Risk label:</b> ${e.risk_label || "N/A"}</small><br>
          <small><b>Risk score:</b> ${e.final_score || "N/A"}</small><br>
          <small><b>Sentiment label:</b> ${e.sentiment_label || "N/A"}</small><br>
          <small><b>Sentiment score:</b> ${e.sentiment_score || "N/A"}</small><br>
          <small><b>Emotion label:</b> ${e.emotion_label || "N/A"}</small><br>
          <small><b>Emotion score:</b> ${e.emotion_score || "N/A"}</small><br>
          <small><b>Date:</b> ${e.dt_utc || "N/A"}</small><br>
          <small><b>Folder:</b> ${folder}</small><br>
          <small><b>From:</b> ${formatList(e.from_norm)}</small><br>
          <small><b>To:</b> ${formatList(e.to_norm)}</small><br>
          <small><b>CC:</b> ${formatList(e.cc_norm)}</small><br>
          <small><b>BCC:</b> ${formatList(e.bcc_norm)}</small>
          <div class="email-body">${sanitizeBody(e.body_raw)}</div>
        </div>`;
    }
    html += "</div>";
    emailBox.innerHTML = html;

  } catch (err) {
    console.error("Error loading emails:", err);
    emailBox.innerHTML = `<h3>Error loading emails.</h3>`;
  }
}

function formatList(json) {
  if (!json) return "—";
  try {
    const arr = typeof json === "string" ? JSON.parse(json) : json;
    return arr.length ? arr.join(", ") : "—";
  } catch {
    return String(json);
  }
}

function sanitizeBody(body) {
  if (!body) return "";
  return body.replace(/\n{3,}/g, "\n\n");
}

// Local search (from JSON)
const searchInput = document.getElementById("nodeSearch");
const resultsBox = document.getElementById("searchResults");

searchInput.addEventListener("input", async () => {
  const query = searchInput.value.trim().toLowerCase();
  resultsBox.innerHTML = "";
  if (!query) return;

  const data = await loadGraph();
  const matches = data.nodes.filter(n =>
    n.id.toLowerCase().includes(query) ||
    (Array.isArray(n.aliases) && n.aliases.some(a => a.toLowerCase().includes(query)))
  ).slice(0, 25);

  if (matches.length === 0) {
    resultsBox.innerHTML = "<div class='dropdown-item'>(No results)</div>";
    return;
  }

  matches.forEach(m => {
    const option = document.createElement("div");
    option.className = "dropdown-item";
    option.textContent = m.id;
    option.onclick = () => {
      window.location.href = `node_detail.html?id=${encodeURIComponent(m.id)}`;
    };
    resultsBox.appendChild(option);
  });
});
