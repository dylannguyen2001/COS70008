// Create tooltip div (global, single instance)
const tooltip = d3.select("body")
  .append("div")
  .attr("class", "tooltip-basic")
  .style("position", "absolute")
  .style("visibility", "hidden")
  .style("background", "rgba(0,0,0,0.9)")
  .style("color", "#fff")
  .style("padding", "8px 12px")
  .style("border-radius", "6px")
  .style("font-size", "13px")
  .style("pointer-events", "none")
  .style("line-height", "1.4em")
  .style("max-width", "320px");

// Node tooltip
export function nodeMouseOver(event, d) {
  tooltip
    .style("visibility", "visible")
    .html(`
      <div><b>${d.id}</b></div>
      <div><b>Community:</b> ${d.community ?? "N/A"}</div>

      <div><b>Risk Label:</b> ${d.risk_label ?? "None"}</div>
      <div><b>Risk Score:</b> ${(d.risk_intensity ?? 0).toFixed(3)}</div>

      <div><b>Sentiment Label:</b> ${d.sentiment_label ?? "N/A"}</div>
      <div><b>Sentiment Score:</b> ${(d.sentiment_intensity ?? 0).toFixed(3)}</div>

      <div><b>Emotion Label:</b> ${d.emotion_label ?? "N/A"}</div>
      <div><b>Emotion Score:</b> ${(d.emotion_intensity ?? 0).toFixed(3)}</div>

      <div><b>Total Emails:</b> ${d.total_emails ?? 0}</div>
      <div><b>Degree:</b> ${d.degree ?? 0}</div>
    `);
}

export function nodeMouseMove(event) {
  tooltip
    .style("left", (event.pageX + 12) + "px")
    .style("top", (event.pageY - 20) + "px");
}

export function nodeMouseOut() {
  tooltip.style("visibility", "hidden");
}

// Edge tooltip
export function edgeMouseOver(event, d) {
  tooltip
    .style("visibility", "visible")
    .html(`
      <div><b>Source:</b> ${d.source.id ?? d.source}</div>
      <div><b>Target:</b> ${d.target.id ?? d.target}</div>
      <div><b>Direction:</b> ${d.source.id ?? d.source} -> ${d.target.id ?? d.target}</div>
      <div><b>Years in contact:</b> ${d.years}</div>
      <hr style="margin: 4px 0; border: 0; border-top: 1px solid #444;">
      <div><b>Interaction Count:</b> ${d.weight ?? 1}</div>
    `);
}

export function edgeMouseMove(event) {
  tooltip
    .style("left", (event.pageX + 12) + "px")
    .style("top", (event.pageY - 20) + "px");
}

export function edgeMouseOut() {
  tooltip.style("visibility", "hidden");
}
