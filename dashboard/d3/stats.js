export function updateCommunityStats(nodes) {
  if (!nodes.length) {
    d3.select("#statsBox").html("No nodes match current filters.");
    return;
  }

  const communityGroups = d3.groups(nodes, d => d.community);
  const comm = document.getElementById("communityFilter").value;

  let text = `<b>Community Metrics</b><br>`;

  if (comm === "all") {
    // Show global averages
    const meanRisk = d3.mean(nodes, d => d.avg_risk ?? 0);
    const meanSent = d3.mean(nodes, d => d.avg_sentiment ?? 0);
    const meanDeg = d3.mean(nodes, d => d.degree ?? 0);
    text += `Communities: ${communityGroups.length}<br>
             Nodes: ${nodes.length}<br>
             Avg Risk: ${meanRisk.toFixed(2)}<br>
             Avg Sentiment: ${meanSent.toFixed(2)}<br>
             Avg Degree: ${meanDeg.toFixed(2)}`;
  } else {
    // Show single community metrics
    const meanRisk = d3.mean(nodes, d => d.avg_risk ?? 0);
    const meanSent = d3.mean(nodes, d => d.avg_sentiment ?? 0);
    const meanDeg = d3.mean(nodes, d => d.degree ?? 0);
    text += `<b>${comm}</b><br>
             Nodes: ${nodes.length}<br>
             Avg Risk: ${meanRisk.toFixed(2)}<br>
             Avg Sentiment: ${meanSent.toFixed(2)}<br>
             Avg Degree: ${meanDeg.toFixed(2)}`;
  }

  d3.select("#statsBox").html(text);
}
