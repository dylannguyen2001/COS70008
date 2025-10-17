import { createSimulation, drag } from "./graph_interaction.js";
import { nodeMouseOver, nodeMouseMove, nodeMouseOut, edgeMouseOver, edgeMouseMove, edgeMouseOut } from "./hover_info.js";
import { applyFilters } from "./filters.js";
import { updateCommunityStats } from "./stats.js";

const width = window.innerWidth;
const height = window.innerHeight;

const svg = d3.select("svg").attr("width", width).attr("height", height);
const container = svg.append("g");

svg.call(
  d3.zoom()
    .scaleExtent([0.1, 6])
    .on("zoom", (event) => container.attr("transform", event.transform))
);


const statsBox = d3.select("#statsBox");

// =====================================================
// Dynamic Graph Loader (Based on time filter)
// =====================================================
function loadGraph(year = "all") {
  const filePath = year === "all"
    ? "data/all/graph_internal.json"
    : `data/${year}/graph_internal.json`;

  statsBox.text(`Loading ${year} graph...`);

  d3.json(filePath).then(data => {
    statsBox.text(`Loaded ${year}: ${data.nodes.length} nodes, ${data.edges.length} edges`);
    renderGraph(data, year);
  }).catch(err => {
    console.error("Failed to load graph:", err);
    statsBox.text(`Error loading ${year} data`);
  });
}

// =====================================================
// Render Graph Function
// =====================================================
function renderGraph(data, year) {
  container.selectAll("*").remove(); // clear previous graph

  const allNodes = data.nodes;
  const allEdges = data.edges;

  console.log(`Rendering ${year}: ${allNodes.length} nodes, ${allEdges.length} edges`);

  const communitySelect = document.getElementById("communityFilter");
  const counts = d3.rollups(
    allNodes.filter(d => d.community != null),
    v => v.length,
    d => d.community
  ).sort((a, b) => a[0] - b[0]);

  communitySelect.innerHTML = `<option value="all">All (${allNodes.length})</option>`;
  communitySelect.innerHTML += `<option value="nonnull">All Non-null (${counts.reduce((a, [_, c]) => a + c, 0)})</option>`;
  counts.forEach(([id, count]) => {
    communitySelect.innerHTML += `<option value="${id}">Community ${id} (${count})</option>`;
  });

  // Coloring based on component
  const riskColor = d3.scaleSequential(d3.interpolateRdYlGn).domain([1, 0]);
  const commExtent = d3.extent(allNodes, d => d.community ?? 0);
  const commColors = d3.scaleSequential(d3.interpolateTurbo).domain(commExtent);
  const sizeScale = d3.scaleSqrt()
    .domain(d3.extent(allNodes, d => d.pagerank || 0.001))
    .range([0.05, 0.15]);

  // Build node map & edges
  const nodeMap = new Map(allNodes.map(d => [d.id, d]));
  const edges = allEdges
    .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
    .map(e => ({
      source: nodeMap.get(e.source),
      target: nodeMap.get(e.target),
      weight: e.weight ?? 1,
      sentiment_score: e.sentiment_score ?? 0
    }));
  const intraColor = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);  // within-community
  const interColor = d3.scaleSequential(d3.interpolateGreys).domain([0, 1]);  // cross-community
    

  // Initial positions
  allNodes.forEach(d => {
    d.x = width / 2 + (Math.random() - 0.5) * 800;
    d.y = height / 2 + (Math.random() - 0.5) * 800;
  });

  // Simulation
  const simulation = createSimulation(allNodes, edges, width, height, sizeScale)
    .force("link", d3.forceLink(edges)
      .id(d => d.id)
      .distance(d => 300 / Math.sqrt(d.weight || 1)))
    .force("charge", d3.forceManyBody().strength(-250))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .alpha(1)
    .alphaDecay(0.02)
    .restart();

  // Draw edges
  const link = container.selectAll(".link")
    .data(allEdges)
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
        ? 0.3 + 0.8 * w     // stronger visibility within community
        : 0.05 + 0.3 * w;   // less visible across communities
    })
    .on("mouseover", edgeMouseOver)
    .on("mousemove", edgeMouseMove)
    .on("mouseout", edgeMouseOut);

  // === Draw nodes ===
  const node = container.selectAll(".node")
    .data(allNodes)
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

  // Filters (work on loaded dataset)
  applyFilters(allNodes, allEdges, container, simulation, commColors, riskColor, intraColor, interColor);
  updateCommunityStats(allNodes);
}

// Year Dropdown Listener
document.getElementById("yearSelect").addEventListener("change", (e) => {
  const year = e.target.value;
  loadGraph(year);
});

loadGraph("all");
