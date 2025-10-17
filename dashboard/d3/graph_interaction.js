// Handles simulation and dragging logic for Enron graph

export function createSimulation(nodes, edges, width, height, sizeScale) {
  return d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id(d => d.id).distance(55)) 
    .force("charge", d3.forceManyBody().strength(-180))           
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(d => sizeScale(d.total_emails || 1) + 2))
}

export function drag(sim) {
  function dragstarted(event, d) {
    if (!event.active) sim.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }

  function dragended(event, d) {
    if (!event.active) sim.alphaTarget(0);
    if (event.sourceEvent.shiftKey) {
      d.fx = null;
      d.fy = null;
    }
  }

  return d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}
