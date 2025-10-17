import { updateCommunityStats } from "./stats.js";
import { nodeMouseOver, nodeMouseMove, nodeMouseOut, edgeMouseOver, edgeMouseMove, edgeMouseOut } from "./hover_info.js";
import { drag } from "./graph_interaction.js";

export function applyFilters(allNodes, allEdges, container, simulation, commColors, riskColor, intraColor, interColor) {

  // Initialize filter bindings
  d3.select("#nodeLimit").on("input", function () {
    d3.select("#nodeLimitValue").text(this.value);
    updateGraph();
  });

  d3.select("#communityFilter").on("change", updateGraph);

  d3.select("#riskFilter").on("input", function () {
    d3.select("#riskValue").text(this.value);
    updateGraph();
  });

  d3.select("#sentimentFilter").on("input", function () {
    d3.select("#sentimentValue").text(this.value);
    updateGraph();
  });

  // Build the community dropdown dynamically (top 10 only)
  const commSelect = document.getElementById("communityFilter");
  const commCounts = d3.rollups(
    allNodes.filter(d => d.community != null && !isNaN(d.community)),
    v => v.length,
    d => d.community
  ).sort((a, b) => b[1] - a[1]);
  const topCommunities = commCounts.slice(0, 10);
  commSelect.innerHTML = `<option value="all">All (${allNodes.length})</option>`;
  topCommunities.forEach(([cid, count]) => {
    commSelect.innerHTML += `<option value="${cid}">Community ${cid} (${count})</option>`;
  });

  // Update graph visualization
  function updateGraph() {
    const limit = +document.getElementById("nodeLimit").value;
    const commVal = document.getElementById("communityFilter").value;
    const minRisk = +document.getElementById("riskFilter").value;
    const minSentiment = +document.getElementById("sentimentFilter").value;

    const visibleEdges = allEdges;
    const activeNodeIds = new Set();
    visibleEdges.forEach(e => {
      activeNodeIds.add(e.source.id || e.source);
      activeNodeIds.add(e.target.id || e.target);
    });

    let activeNodes = allNodes.filter(d => activeNodeIds.has(d.id));

    // Apply community + risk + sentiment filters
    activeNodes = activeNodes.filter(d => {
      const passesRisk = (d.risk_intensity ?? 0) >= minRisk;
      const passesSent = (d.sentiment_intensity ?? 0) >= minSentiment;

      if (commVal === "all") return passesRisk && passesSent;
      if (commVal === "nonnull")
        return d.community != null && passesRisk && passesSent;
      return d.community === +commVal && passesRisk && passesSent;
    });

    activeNodes = activeNodes.slice(0, limit);
    const nodeIds = new Set(activeNodes.map(d => d.id));

    const filteredEdges = visibleEdges.filter(e => {
      const src = e.source.id || e.source;
      const tgt = e.target.id || e.target;
      return nodeIds.has(src) && nodeIds.has(tgt);
    });

    simulation.nodes(activeNodes);
    simulation.force("link").links(filteredEdges);
    simulation.alpha(1).restart();

    container.selectAll(".link").remove();
    container.selectAll(".node").remove();

    const link = container.selectAll(".link")
      .data(filteredEdges)
      .enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke", d => {
        const score = d.sentiment_score || 0;
        return d.source.community === d.target.community
          ? intraColor((score + 1) / 2)   
          : interColor((score + 1) / 2);
      })
      .attr("stroke-width", d => 0.4 + Math.log1p(d.weight || 1) * 0.5)
      .attr("opacity", d => {
        const w = Math.min(Math.log1p(d.weight || 1) / 5, 1); 
        return d.source.community === d.target.community
          ? 0.3 + 0.7 * w     // stronger visibility within community
          : 0.05 + 0.4 * w;   // less visible across communities
      })
      .on("mouseover", edgeMouseOver)
      .on("mousemove", edgeMouseMove)
      .on("mouseout", edgeMouseOut);



    const node = container.selectAll(".node")
      .data(activeNodes)
      .enter()
      .append("circle")
      .attr("class", "node")
      .attr("r", d => 10 + 8000 * (d.pagerank))
      .attr("fill", d => commColors(d.community ?? 0))
      .attr("fill-opacity", d => 0.4 + 0.6 * (d.sentiment_intensity ?? 0))
      .attr("stroke", d => riskColor(d.risk_intensity ?? 0))
      .attr("stroke-width", d => 0.5 + 2.5 * (d.risk_intensity ?? 0))
      .call(drag(simulation))
      .on("mouseover", nodeMouseOver)
      .on("mousemove", nodeMouseMove)
      .on("mouseout", nodeMouseOut)
      .on("click", (event, d) => {
        window.location.href = `node_detail.html?id=${encodeURIComponent(d.id)}`;
      });

    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
    });

    updateCommunityStats(activeNodes);
  }

  updateGraph();
}
