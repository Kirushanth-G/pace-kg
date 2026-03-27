import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

const RELATION_COLORS = {
  isPrerequisiteOf: "#f59e0b",
  isDefinedAs: "#2563eb",
  isExampleOf: "#14b8a6",
  contrastedWith: "#ef4444",
  appliedIn: "#0ea5e9",
  isPartOf: "#22c55e",
  causeOf: "#dc2626",
  isGeneralizationOf: "#7c3aed",
  relatedConcept: "#64748b",
};

function deriveNodeDetails(nodeName, nodes, edges) {
  const node = nodes.find((entry) => entry.name === nodeName);
  const relatedEdge = edges.find(
    (edge) => edge.subject === nodeName || edge.object === nodeName
  );

  if (!node) {
    return null;
  }

  return {
    name: node.name,
    sourceSlide: relatedEdge?.slide_id || node.slide_ids?.[0] || "N/A",
    evidence: relatedEdge?.evidence || "No evidence sentence available.",
    keyTerms:
      node.aliases && node.aliases.length
        ? Array.from(new Set([node.name, ...node.aliases]))
        : [node.name],
  };
}

function GraphView({
  nodes,
  edges,
  selectedSlideId,
  selectedNode,
  onSelectNode,
}) {
  const svgRef = useRef(null);
  const [relationFilter, setRelationFilter] = useState("all");
  const [minWeight, setMinWeight] = useState(0);
  const [slideFilter, setSlideFilter] = useState("all");

  useEffect(() => {
    if (selectedSlideId) {
      setSlideFilter(selectedSlideId);
    }
  }, [selectedSlideId]);

  const relationTypes = useMemo(
    () => Array.from(new Set(edges.map((edge) => edge.relation))).sort(),
    [edges]
  );

  const slideIds = useMemo(
    () =>
      Array.from(
        new Set(
          nodes.flatMap((node) => node.slide_ids || []).filter((slideId) => slideId)
        )
      ).sort(),
    [nodes]
  );

  const filteredData = useMemo(() => {
    const filteredNodes = nodes.filter((node) => {
      const passesWeight = Number(node.final_weight || 0) >= minWeight;
      const matchesSlide =
        slideFilter === "all" ||
        (node.slide_ids || []).some((slideId) => slideId === slideFilter);
      return passesWeight && matchesSlide;
    });

    const filteredNodeNames = new Set(filteredNodes.map((node) => node.name));

    const filteredEdges = edges.filter((edge) => {
      const matchesRelation =
        relationFilter === "all" || edge.relation === relationFilter;
      const matchesSlide = slideFilter === "all" || edge.slide_id === slideFilter;
      return (
        matchesRelation &&
        matchesSlide &&
        filteredNodeNames.has(edge.subject) &&
        filteredNodeNames.has(edge.object)
      );
    });

    const connectedNodeNames = new Set();
    for (const edge of filteredEdges) {
      connectedNodeNames.add(edge.subject);
      connectedNodeNames.add(edge.object);
    }

    const graphNodes = filteredNodes.filter((node) => connectedNodeNames.has(node.name));

    return {
      nodes: graphNodes,
      edges: filteredEdges,
    };
  }, [nodes, edges, relationFilter, minWeight, slideFilter]);

  const selectedDetails = useMemo(
    () => deriveNodeDetails(selectedNode, filteredData.nodes, filteredData.edges),
    [selectedNode, filteredData]
  );

  useEffect(() => {
    const width = 880;
    const height = 560;
    const svg = d3.select(svgRef.current);

    svg.selectAll("*").remove();

    if (!filteredData.nodes.length) {
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", height / 2)
        .attr("text-anchor", "middle")
        .attr("fill", "#64748b")
        .text("No graph data matches current filters.");
      return undefined;
    }

    const simulationNodes = filteredData.nodes.map((node) => ({ ...node }));
    const simulationEdges = filteredData.edges.map((edge) => ({
      ...edge,
      source: edge.subject,
      target: edge.object,
    }));

    const simulation = d3
      .forceSimulation(simulationNodes)
      .force(
        "link",
        d3
          .forceLink(simulationEdges)
          .id((node) => node.name)
          .distance(120)
      )
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((node) => 14 + (node.final_weight || 0) * 20));

    const graphRoot = svg.append("g");

    const link = graphRoot
      .append("g")
      .attr("stroke-opacity", 0.85)
      .selectAll("line")
      .data(simulationEdges)
      .join("line")
      .attr("stroke-width", 1.8)
      .attr("stroke", (edge) => RELATION_COLORS[edge.relation] || "#334155");

    const edgeLabels = graphRoot
      .append("g")
      .selectAll("text")
      .data(simulationEdges)
      .join("text")
      .attr("font-size", 10)
      .attr("fill", "#475569")
      .attr("text-anchor", "middle")
      .text((edge) => edge.relation);

    const node = graphRoot
      .append("g")
      .selectAll("circle")
      .data(simulationNodes)
      .join("circle")
      .attr("r", (entry) => 9 + (entry.final_weight || 0) * 18)
      .attr("fill", (entry) =>
        entry.slide_ids?.includes(slideFilter) ? "#f97316" : "#0f766e"
      )
      .attr("stroke", (entry) => (entry.name === selectedNode ? "#f8fafc" : "#083344"))
      .attr("stroke-width", (entry) => (entry.name === selectedNode ? 3 : 1.5))
      .on("click", (_, entry) => {
        onSelectNode(entry.name);
      })
      .call(
        d3
          .drag()
          .on("start", (event, entry) => {
            if (!event.active) {
              simulation.alphaTarget(0.3).restart();
            }
            entry.fx = entry.x;
            entry.fy = entry.y;
          })
          .on("drag", (event, entry) => {
            entry.fx = event.x;
            entry.fy = event.y;
          })
          .on("end", (event, entry) => {
            if (!event.active) {
              simulation.alphaTarget(0);
            }
            entry.fx = null;
            entry.fy = null;
          })
      );

    const labels = graphRoot
      .append("g")
      .selectAll("text")
      .data(simulationNodes)
      .join("text")
      .attr("class", "node-label")
      .attr("font-size", 11)
      .attr("dx", 12)
      .attr("dy", 4)
      .text((entry) => entry.name);

    svg.call(
      d3
        .zoom()
        .scaleExtent([0.5, 2.3])
        .on("zoom", (event) => {
          graphRoot.attr("transform", event.transform);
        })
    );

    simulation.on("tick", () => {
      link
        .attr("x1", (entry) => entry.source.x)
        .attr("y1", (entry) => entry.source.y)
        .attr("x2", (entry) => entry.target.x)
        .attr("y2", (entry) => entry.target.y);

      edgeLabels
        .attr("x", (entry) => (entry.source.x + entry.target.x) / 2)
        .attr("y", (entry) => (entry.source.y + entry.target.y) / 2);

      node.attr("cx", (entry) => entry.x).attr("cy", (entry) => entry.y);
      labels.attr("x", (entry) => entry.x).attr("y", (entry) => entry.y);
    });

    return () => {
      simulation.stop();
    };
  }, [filteredData, onSelectNode, selectedNode, slideFilter]);

  return (
    <section className="panel graph-panel">
      <div className="panel-header">
        <h2>Interactive Knowledge Graph</h2>
      </div>

      <div className="filters-row">
        <label>
          Slide
          <select value={slideFilter} onChange={(event) => setSlideFilter(event.target.value)}>
            <option value="all">All slides</option>
            {slideIds.map((slideId) => (
              <option key={slideId} value={slideId}>
                {slideId}
              </option>
            ))}
          </select>
        </label>

        <label>
          Relation
          <select
            value={relationFilter}
            onChange={(event) => setRelationFilter(event.target.value)}
          >
            <option value="all">All relations</option>
            {relationTypes.map((relation) => (
              <option key={relation} value={relation}>
                {relation}
              </option>
            ))}
          </select>
        </label>

        <label>
          Min Weight
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={minWeight}
            onChange={(event) => setMinWeight(Number(event.target.value))}
          />
          <span>{minWeight.toFixed(2)}</span>
        </label>
      </div>

      <svg ref={svgRef} viewBox="0 0 880 560" role="img" aria-label="Knowledge graph" />

      <div className="node-details">
        {selectedDetails ? (
          <>
            <h3>{selectedDetails.name}</h3>
            <p>
              <strong>Source slide:</strong> {selectedDetails.sourceSlide}
            </p>
            <p>
              <strong>Evidence:</strong> {selectedDetails.evidence}
            </p>
            <div className="chip-row">
              {selectedDetails.keyTerms.map((term) => (
                <span key={term} className="chip">
                  {term}
                </span>
              ))}
            </div>
          </>
        ) : (
          <p className="muted">
            Click a concept node to inspect source slide, evidence sentence, and key terms.
          </p>
        )}
      </div>
    </section>
  );
}

export default GraphView;
