export function updateCommunityStats(nodes) {
  if (!nodes.length) {
    d3.select("#statsBox").html("No nodes match current filters.");
    return;
  }

  const communityGroups = d3.groups(nodes, d => d.community);
  const comm = document.getElementById("communityFilter").value;
  const totalNodes = nodes.length;

  const labelShort = {
    "This email appears routine, compliant, and shows no indication of risk or wrongdoing.": "Non-risky",
    "This email contains evidence of accounting fraud or deceptive financial activity.": "Fraud",
    "This email discusses legal violations or compliance failures.": "Compliance",
    "This email indicates poor management, process failures, or operational missteps.": "Operational",
    "This email could damage the company's reputation or public image if disclosed.": "Reputational",
    "This email reveals secret collaboration or collusion between individuals or groups.": "Collusion",
    "This email shows manipulation of data, reports, or market positions.": "Manipulation"
  };



  let text = `<b>Community Metrics</b><br>`;

  if (comm === "all") {
    // Show global averages
    const riskStats = d3.rollups(
      nodes,
      v => ({
        mean: d3.mean(v, d => d.risk_intensity ?? 0),
        count: v.length
      }),
      d => d.risk_label
    ).sort((a, b) => b[1].mean - a[1].mean);  

    // Show global averages
    const sentimentStats = d3.rollups(
      nodes,
      v => ({
        mean: d3.mean(v, d => d.sentiment_intensity ?? 0),
        count: v.length
      }),
      d => d.sentiment_label
    ).sort((a, b) => b[1].mean - a[1].mean);  


    // Show global averages
    const emotionStats = d3.rollups(
      nodes,
      v => ({
        mean: d3.mean(v, d => d.emotion_intensity ?? 0),
        count: v.length
      }),
      d => d.emotion_label
    ).sort((a, b) => b[1].mean - a[1].mean);  

    const meanDeg = d3.mean(nodes, d => d.degree ?? 0);
    text += `Communities: ${communityGroups.length}<br>
             Nodes: ${nodes.length}<br>
             Avg Degree: ${meanDeg.toFixed(2)} <br>
             <br>
            <b>Risk by Category:</b><br>
             `;
    for (const [label, stats] of riskStats) {
      // if(label == "None") {
      //   continue;
      // }
      const width = Math.min(stats.mean * 100, 100);
      const pct = ((stats.count / totalNodes) * 100).toFixed(1); // % of total nodes
      text += `
        <div style="margin:4px 0">
          <div class="text-capitalize">${labelShort[label] ?? label}: ${stats.count} (${pct}%)</div>
          <div style="
            height:6px;
            width:${width}%;
            background:#e74c3c;
            border-radius:4px;
          "></div>
          <small>${stats.mean.toFixed(2)}</small>
        </div>`;
    }

    text+= `<br><b>Sentiment by Category:</b><br>`;
    for (const [label, stats] of sentimentStats) {
      // if(label == "None") {
      //   continue;
      // }
      const width = Math.min(stats.mean * 100, 100);
      const pct = ((stats.count / totalNodes) * 100).toFixed(1); // % of total nodes
      text += `
        <div style="margin:4px 0">
          <div class="text-capitalize">${label}: ${stats.count} (${pct}%)</div>
          <div style="
            height:6px;
            width:${width}%;
            background:#e74c3c;
            border-radius:4px;
          "></div>
          <small>${stats.mean.toFixed(2)}</small>
        </div>`;
    }

    text+= `<br><b>Emotion by Category:</b><br>`;
    for (const [label, stats] of emotionStats) {
      // if(label == "None") {
      //   continue;
      // }
      const width = Math.min(stats.mean * 100, 100);
      const pct = ((stats.count / totalNodes) * 100).toFixed(1); // % of total nodes
      text += `
        <div style="margin:4px 0">
          <div class="text-capitalize">${label}: ${stats.count} (${pct}%)</div>
          <div style="
            height:6px;
            width:${width}%;
            background:#e74c3c;
            border-radius:4px;
          "></div>
          <small>${stats.mean.toFixed(2)}</small>
        </div>`;
    }
             
  } else {
   // Show global averages
    const riskStats = d3.rollups(
      nodes,
      v => ({
        mean: d3.mean(v, d => d.risk_intensity ?? 0),
        count: v.length
      }),
      d => d.risk_label
    ).sort((a, b) => b[1].mean - a[1].mean);  

    // Show global averages
    const sentimentStats = d3.rollups(
      nodes,
      v => ({
        mean: d3.mean(v, d => d.sentiment_intensity ?? 0),
        count: v.length
      }),
      d => d.sentiment_label
    ).sort((a, b) => b[1].mean - a[1].mean);  


    // Show global averages
    const emotionStats = d3.rollups(
      nodes,
      v => ({
        mean: d3.mean(v, d => d.emotion_intensity ?? 0),
        count: v.length
      }),
      d => d.emotion_label
    ).sort((a, b) => b[1].mean - a[1].mean);  

    const meanDeg = d3.mean(nodes, d => d.degree ?? 0);
    text += `Communities: ${communityGroups.length}<br>
             Nodes: ${nodes.length}<br>
             Avg Degree: ${meanDeg.toFixed(2)} <br>
             <br>
            <b>Risk by Category:</b><br>
             `;
    for (const [label, stats] of riskStats) {
      // if(label == "None") {
      //   continue;
      // }
      const width = Math.min(stats.mean * 100, 100);
      const pct = ((stats.count / totalNodes) * 100).toFixed(1); // % of total nodes
      text += `
        <div style="margin:4px 0">
          <div class="text-capitalize">${labelShort[label] ?? label}: ${stats.count} (${pct}%)</div>
          <div style="
            height:6px;
            width:${width}%;
            background:#e74c3c;
            border-radius:4px;
          "></div>
          <small>${stats.mean.toFixed(2)}</small>
        </div>`;
    }

    text+= `<br><b>Sentiment by Category:</b><br>`;
    for (const [label, stats] of sentimentStats) {
      // if(label == "None") {
      //   continue;
      // }
      const width = Math.min(stats.mean * 100, 100);
      const pct = ((stats.count / totalNodes) * 100).toFixed(1); // % of total nodes
      text += `
        <div style="margin:4px 0">
          <div class="text-capitalize">${label}: ${stats.count} (${pct}%)</div>
          <div style="
            height:6px;
            width:${width}%;
            background:#e74c3c;
            border-radius:4px;
          "></div>
          <small>${stats.mean.toFixed(2)}</small>
        </div>`;
    }

    text+= `<br><b>Emotion by Category:</b><br>`;
    for (const [label, stats] of emotionStats) {
      // if(label == "None") {
      //   continue;
      // }
      const width = Math.min(stats.mean * 100, 100);
      const pct = ((stats.count / totalNodes) * 100).toFixed(1); // % of total nodes
      text += `
        <div style="margin:4px 0">
          <div class="text-capitalize">${label}: ${stats.count} (${pct}%)</div>
          <div style="
            height:6px;
            width:${width}%;
            background:#e74c3c;
            border-radius:4px;
          "></div>
          <small>${stats.mean.toFixed(2)}</small>
        </div>`;
    }
  }

  d3.select("#statsBox").html(text);
}
