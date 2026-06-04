const routeColors = [
  "#157a72",
  "#c27803",
  "#b84b5f",
  "#6f5bb6",
  "#2f6f9f",
  "#4f8a43",
  "#b15d2e",
  "#465b6b"
];

const sampleHistory = [68.60, 68.45, 68.45, 68.41, 68.21, 68.21, 68.21, 68.21, 68.21, 68.21];

const sampleRoutes = [
  [0, 11, 4, 68, 87, 79, 49, 74, 89, 12, 10, 23, 64, 93, 60, 0],
  [0, 25, 16, 58, 83, 18, 100, 31, 6, 42, 22, 0],
  [0, 55, 43, 24, 90, 63, 44, 81, 75, 52, 13, 0],
  [0, 59, 15, 14, 45, 72, 99, 38, 65, 8, 47, 0],
  [0, 62, 35, 29, 30, 80, 32, 46, 2, 73, 51, 54, 5, 78, 33, 0],
  [0, 17, 3, 21, 66, 37, 76, 69, 40, 9, 1, 56, 95, 77, 96, 0],
  [0, 71, 27, 88, 50, 94, 67, 61, 85, 36, 48, 34, 7, 86, 92, 0],
  [0, 84, 53, 20, 91, 26, 41, 98, 57, 82, 97, 19, 28, 70, 39, 0]
];

const baseConfig = {
  numAnts: 100,
  iterations: 10,
  alpha: 0.5,
  beta: 0.5,
  rho: 0.05,
  q0: 0.9,
  depotId: 0,
  nDeliveries: 100,
  nVehicles: 8,
  depot: [45.062786, 7.678686],
  vehicleCosts: [10, 5, 5, 5, 10, 10, 10, 10],
  vehicleCapacity: [50, 10, 10, 10, 50, 50, 50, 50]
};

const els = {
  routeCanvas: document.getElementById("routeCanvas"),
  costChart: document.getElementById("costChart"),
  tooltip: document.getElementById("nodeTooltip"),
  runBtn: document.getElementById("runBtn"),
  resetBtn: document.getElementById("resetBtn"),
  consoleLog: document.getElementById("consoleLog"),
  routeSummary: document.getElementById("routeSummary"),
  routeLegend: document.getElementById("routeLegend"),
  mapTitle: document.getElementById("mapTitle"),
  iterationMetric: document.getElementById("iterationMetric"),
  costMetric: document.getElementById("costMetric"),
  routeMetric: document.getElementById("routeMetric"),
  servedMetric: document.getElementById("servedMetric"),
  sampleModeBtn: document.getElementById("sampleModeBtn"),
  liveModeBtn: document.getElementById("liveModeBtn"),
  liveSettings: document.getElementById("liveSettings"),
  antsInput: document.getElementById("antsInput"),
  antsOutput: document.getElementById("antsOutput"),
  iterationsInput: document.getElementById("iterationsInput"),
  iterationsOutput: document.getElementById("iterationsOutput"),
  seedInput: document.getElementById("seedInput")
};

let mode = "sample";
let canvasPositions = [];
let activeResult = makeSampleResult();
let activeIteration = sampleHistory.length;
let routeReveal = 1;
let isRunning = false;

function mulberry32(seed) {
  return function nextRandom() {
    let t = seed += 0x6d2b79f5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateNetwork(seed = 42) {
  const rng = mulberry32(Number(seed) || 42);
  const points = [
    {
      id: 0,
      lat: baseConfig.depot[0],
      lng: baseConfig.depot[1],
      vol: 0,
      crowdCost: 0,
      timeWindowMin: 0,
      timeWindowMax: Number.POSITIVE_INFINITY,
      type: "depot"
    }
  ];

  for (let id = 1; id <= baseConfig.nDeliveries; id += 1) {
    const lat = baseConfig.depot[0] + (rng() * 0.2 - 0.1);
    const lng = baseConfig.depot[1] + (rng() * 0.2 - 0.1);
    const distToDepot = Math.hypot(lat - baseConfig.depot[0], lng - baseConfig.depot[1]);
    points.push({
      id,
      lat,
      lng,
      vol: 1,
      crowdCost: 2,
      timeWindowMin: distToDepot * 10,
      timeWindowMax: distToDepot * 10 + 100000,
      type: "customer"
    });
  }

  const distanceMatrix = points.map((from) =>
    points.map((to) => Math.hypot(from.lat - to.lat, from.lng - to.lng))
  );

  const vehicles = Array.from({ length: baseConfig.nVehicles }, (_, index) => ({
    id: index + 1,
    capacity: baseConfig.vehicleCapacity[index],
    cost: baseConfig.vehicleCosts[index]
  }));

  return { points, distanceMatrix, vehicles };
}

function makeSampleResult() {
  return {
    label: "Kết quả mẫu",
    env: generateNetwork(42),
    history: sampleHistory.slice(),
    best: {
      cost: 68.21,
      routes: sampleRoutes.map((route) => route.slice()),
      served: 100
    }
  };
}

function makeConfigFromControls() {
  return {
    ...baseConfig,
    numAnts: Number(els.antsInput.value),
    iterations: Number(els.iterationsInput.value),
    seed: Number(els.seedInput.value) || 42
  };
}

function randomChoice(items, rng) {
  return items[Math.floor(rng() * items.length)];
}

function weightedChoice(items, weights, rng) {
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    return randomChoice(items, rng);
  }

  let threshold = rng() * total;
  for (let i = 0; i < items.length; i += 1) {
    threshold -= weights[i];
    if (threshold <= 0) {
      return items[i];
    }
  }
  return items[items.length - 1];
}

function runAco(config) {
  const env = generateNetwork(config.seed);
  const rng = mulberry32((Number(config.seed) || 42) + 1009);
  const nodeCount = config.nDeliveries + 1;
  const pheromone = Array.from({ length: nodeCount }, () => Array(nodeCount).fill(1));
  let best = null;
  const history = [];

  for (let iteration = 1; iteration <= config.iterations; iteration += 1) {
    const solutions = [];
    for (let ant = 0; ant < config.numAnts; ant += 1) {
      solutions.push(findSolution(env, pheromone, config, rng));
    }

    const candidate = solutions.reduce((currentBest, solution) =>
      solution.cost < currentBest.cost ? solution : currentBest
    );

    if (!best || candidate.cost < best.cost) {
      best = candidate;
    }

    history.push(Number(best.cost.toFixed(2)));
    updatePheromone(pheromone, best, config);
  }

  return {
    label: "Chạy mới",
    env,
    history,
    best: {
      cost: best.cost,
      routes: best.routes,
      served: best.served
    }
  };
}

function findSolution(env, pheromone, config, rng) {
  const capacities = env.vehicles.map((vehicle) => vehicle.capacity);
  const tourTimes = env.vehicles.map(() => 0);
  const totalPathCost = env.vehicles.map(() => 0);
  const nodesLeft = new Set(Array.from({ length: config.nDeliveries }, (_, index) => index + 1));
  const routes = [];

  function getAvailableNodes(currentNode, vehicleIndex) {
    const available = [];
    for (const node of nodesLeft) {
      const point = env.points[node];
      const projectedTime = tourTimes[vehicleIndex] +
        env.distanceMatrix[currentNode][node] +
        point.crowdCost;
      if (projectedTime <= point.timeWindowMax && capacities[vehicleIndex] >= point.vol) {
        available.push(node);
      }
    }
    return available;
  }

  function selectNextDelivery(currentNode, vehicleIndex) {
    const availableNodes = getAvailableNodes(currentNode, vehicleIndex);
    if (!availableNodes.length) {
      return null;
    }

    const scores = availableNodes.map((node) => {
      const distance = Math.max(env.distanceMatrix[currentNode][node], 1e-9);
      const tau = Math.pow(pheromone[currentNode][node], config.alpha);
      const eta = Math.pow(1 / distance, config.beta);
      return tau * eta;
    });

    if (rng() <= config.q0) {
      let bestIndex = 0;
      for (let i = 1; i < scores.length; i += 1) {
        if (scores[i] > scores[bestIndex]) {
          bestIndex = i;
        }
      }
      return availableNodes[bestIndex];
    }

    return weightedChoice(availableNodes, scores, rng);
  }

  function moveToDelivery(vehicleIndex, currentNode, nextNode) {
    routes[vehicleIndex].push(nextNode);
    let crowdCost = 0;
    if (nextNode !== config.depotId) {
      nodesLeft.delete(nextNode);
      crowdCost = env.points[nextNode].crowdCost;
    }
    capacities[vehicleIndex] -= env.points[nextNode].vol;
    tourTimes[vehicleIndex] += env.distanceMatrix[currentNode][nextNode] + crowdCost;
    totalPathCost[vehicleIndex] += env.distanceMatrix[currentNode][nextNode];
  }

  for (let vehicleIndex = 0; vehicleIndex < env.vehicles.length; vehicleIndex += 1) {
    routes.push([config.depotId]);
    const firstOptions = getAvailableNodes(config.depotId, vehicleIndex);
    if (firstOptions.length) {
      const firstDelivery = randomChoice(firstOptions, rng);
      moveToDelivery(vehicleIndex, config.depotId, firstDelivery);
      totalPathCost[vehicleIndex] += env.vehicles[vehicleIndex].cost;
    }
  }

  while (nodesLeft.size) {
    let movedThisTurn = false;
    for (let vehicleIndex = 0; vehicleIndex < env.vehicles.length; vehicleIndex += 1) {
      const route = routes[vehicleIndex];
      const currentNode = route[route.length - 1];

      if (route.length > 2 && currentNode === config.depotId) {
        continue;
      }

      const nextDelivery = selectNextDelivery(currentNode, vehicleIndex);
      if (nextDelivery === null) {
        if (currentNode !== config.depotId) {
          moveToDelivery(vehicleIndex, currentNode, config.depotId);
          movedThisTurn = true;
        }
      } else {
        moveToDelivery(vehicleIndex, currentNode, nextDelivery);
        movedThisTurn = true;
      }
    }

    if (!movedThisTurn) {
      break;
    }
  }

  for (let vehicleIndex = 0; vehicleIndex < env.vehicles.length; vehicleIndex += 1) {
    const route = routes[vehicleIndex];
    const currentNode = route[route.length - 1];
    if (currentNode !== config.depotId) {
      moveToDelivery(vehicleIndex, currentNode, config.depotId);
    }
  }

  return {
    routes,
    cost: totalPathCost.reduce((sum, value) => sum + value, 0),
    served: config.nDeliveries - nodesLeft.size
  };
}

function updatePheromone(pheromone, bestSolution, config) {
  const nodeCount = pheromone.length;
  for (let from = 0; from < nodeCount; from += 1) {
    for (let to = from + 1; to < nodeCount; to += 1) {
      pheromone[from][to] = Math.max((1 - config.rho) * pheromone[from][to], 1e-10);
    }
  }

  const pheromoneIncrease = 1 / bestSolution.cost;
  for (const route of bestSolution.routes) {
    for (let index = 0; index < route.length - 1; index += 1) {
      const from = route[index];
      const to = route[index + 1];
      pheromone[from][to] += config.rho * pheromoneIncrease;
    }
  }
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.max(window.devicePixelRatio || 1, 1);
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { width: rect.width, height: rect.height, ctx };
}

function getPointPositions(env, width, height) {
  const padding = Math.max(34, Math.min(width, height) * 0.08);
  const lats = env.points.map((point) => point.lat);
  const lngs = env.points.map((point) => point.lng);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs);
  const maxLng = Math.max(...lngs);
  const latSpan = Math.max(maxLat - minLat, 1e-9);
  const lngSpan = Math.max(maxLng - minLng, 1e-9);

  return env.points.map((point) => ({
    id: point.id,
    x: padding + ((point.lng - minLng) / lngSpan) * (width - padding * 2),
    y: height - padding - ((point.lat - minLat) / latSpan) * (height - padding * 2),
    point
  }));
}

function drawRouteMap() {
  const { width, height, ctx } = resizeCanvas(els.routeCanvas);
  if (!width || !height) {
    return;
  }

  ctx.clearRect(0, 0, width, height);
  canvasPositions = getPointPositions(activeResult.env, width, height);
  const byId = new Map(canvasPositions.map((entry) => [entry.id, entry]));

  drawMapGrid(ctx, width, height);
  drawRoutes(ctx, byId, activeResult.best.routes, routeReveal);
  drawNodes(ctx, canvasPositions, activeResult.best.routes);
}

function drawMapGrid(ctx, width, height) {
  ctx.save();
  ctx.strokeStyle = "rgba(31, 37, 40, 0.08)";
  ctx.lineWidth = 1;
  for (let x = 48; x < width; x += 96) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 48; y < height; y += 96) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawRoutes(ctx, byId, routes, reveal) {
  routes.forEach((route, routeIndex) => {
    const points = route.map((node) => byId.get(node)).filter(Boolean);
    const color = routeColors[routeIndex % routeColors.length];
    ctx.save();
    ctx.lineWidth = 2.7;
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.75;
    drawPartialPolyline(ctx, points, reveal);
    ctx.restore();
  });
}

function drawPartialPolyline(ctx, points, reveal) {
  if (points.length < 2 || reveal <= 0) {
    return;
  }

  const segmentLengths = [];
  let totalLength = 0;
  for (let i = 0; i < points.length - 1; i += 1) {
    const length = Math.hypot(points[i + 1].x - points[i].x, points[i + 1].y - points[i].y);
    segmentLengths.push(length);
    totalLength += length;
  }

  let remaining = totalLength * Math.min(1, Math.max(0, reveal));
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 0; i < segmentLengths.length; i += 1) {
    const from = points[i];
    const to = points[i + 1];
    const segmentLength = segmentLengths[i];
    if (remaining >= segmentLength) {
      ctx.lineTo(to.x, to.y);
      remaining -= segmentLength;
    } else {
      const ratio = segmentLength ? remaining / segmentLength : 0;
      ctx.lineTo(
        from.x + (to.x - from.x) * ratio,
        from.y + (to.y - from.y) * ratio
      );
      break;
    }
  }
  ctx.stroke();
}

function drawNodes(ctx, positions, routes) {
  const routeMembership = new Map();
  routes.forEach((route, routeIndex) => {
    route.forEach((node) => {
      if (node !== 0 && !routeMembership.has(node)) {
        routeMembership.set(node, routeIndex);
      }
    });
  });

  for (const entry of positions) {
    if (entry.id === 0) {
      continue;
    }
    const routeIndex = routeMembership.get(entry.id) ?? 0;
    ctx.save();
    ctx.beginPath();
    ctx.fillStyle = routeColors[routeIndex % routeColors.length];
    ctx.globalAlpha = 0.9;
    ctx.arc(entry.x, entry.y, 3.7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.restore();
  }

  const depot = positions.find((entry) => entry.id === 0);
  if (depot) {
    ctx.save();
    ctx.translate(depot.x, depot.y);
    ctx.fillStyle = "#1f2528";
    ctx.strokeStyle = "#fffdf7";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(0, -12);
    ctx.lineTo(12, 0);
    ctx.lineTo(0, 12);
    ctx.lineTo(-12, 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = "700 11px Segoe UI, Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("0", 0, 0);
    ctx.restore();
  }
}

function drawCostChart(history, limit = history.length) {
  const { width, height, ctx } = resizeCanvas(els.costChart);
  ctx.clearRect(0, 0, width, height);
  const visible = history.slice(0, Math.max(0, limit));
  const pad = { left: 42, right: 16, top: 16, bottom: 28 };

  ctx.save();
  ctx.strokeStyle = "rgba(31, 37, 40, 0.16)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();

  ctx.fillStyle = "#687076";
  ctx.font = "12px Segoe UI, Arial";
  ctx.fillText("cost", 8, 20);
  ctx.fillText("iteration", width - 72, height - 8);

  if (!visible.length) {
    ctx.fillStyle = "#687076";
    ctx.fillText("Đang chờ chạy...", pad.left + 12, pad.top + 28);
    ctx.restore();
    return;
  }

  const minCost = Math.min(...history) - 0.08;
  const maxCost = Math.max(...history) + 0.08;
  const xSpan = Math.max(history.length - 1, 1);
  const ySpan = Math.max(maxCost - minCost, 0.01);

  function xAt(index) {
    return pad.left + (index / xSpan) * (width - pad.left - pad.right);
  }

  function yAt(cost) {
    return height - pad.bottom - ((cost - minCost) / ySpan) * (height - pad.top - pad.bottom);
  }

  ctx.strokeStyle = "rgba(21, 122, 114, 0.24)";
  for (let i = 0; i < history.length; i += 1) {
    const x = xAt(i);
    ctx.beginPath();
    ctx.moveTo(x, height - pad.bottom);
    ctx.lineTo(x, height - pad.bottom + 4);
    ctx.stroke();
  }

  ctx.strokeStyle = "#157a72";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  visible.forEach((cost, index) => {
    const x = xAt(index);
    const y = yAt(cost);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  visible.forEach((cost, index) => {
    const x = xAt(index);
    const y = yAt(cost);
    ctx.beginPath();
    ctx.fillStyle = index === visible.length - 1 ? "#c27803" : "#157a72";
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  const lastCost = visible[visible.length - 1];
  ctx.fillStyle = "#1f2528";
  ctx.font = "700 12px Segoe UI, Arial";
  ctx.fillText(lastCost.toFixed(2), Math.min(width - 58, xAt(visible.length - 1) + 8), yAt(lastCost) - 8);
  ctx.restore();
}

function updateLegend(routes) {
  els.routeLegend.innerHTML = "";
  routes.forEach((route, index) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = routeColors[index % routeColors.length];
    const label = document.createElement("span");
    label.textContent = `Xe ${index + 1}: ${Math.max(0, route.length - 2)} điểm`;
    item.append(swatch, label);
    els.routeLegend.append(item);
  });
}

function countServed(routes) {
  const served = new Set();
  routes.forEach((route) => route.forEach((node) => {
    if (node !== 0) {
      served.add(node);
    }
  }));
  return served.size;
}

function routeDistance(route, env) {
  let distance = 0;
  for (let index = 0; index < route.length - 1; index += 1) {
    distance += env.distanceMatrix[route[index]][route[index + 1]];
  }
  return distance;
}

function updateRouteSummary() {
  els.routeSummary.innerHTML = "";
  activeResult.best.routes.forEach((route, index) => {
    const card = document.createElement("article");
    card.className = "route-card";
    const dot = document.createElement("span");
    dot.className = "route-dot";
    dot.style.background = routeColors[index % routeColors.length];
    const body = document.createElement("div");
    const customerCount = Math.max(0, route.length - 2);
    const distance = routeDistance(route, activeResult.env);
    const fixedCost = activeResult.env.vehicles[index]?.cost ?? 0;
    const title = document.createElement("strong");
    title.textContent = `Xe ${index + 1} · ${customerCount} khách · khoảng cách ${distance.toFixed(2)} · phí xe ${fixedCost}`;
    const path = document.createElement("div");
    path.className = "route-path";
    path.textContent = route.join(" → ");
    body.append(title, path);
    card.append(dot, body);
    els.routeSummary.append(card);
  });
}

function buildConsole(history, routes, finalCost, limit = history.length, done = true) {
  const lines = ["Bắt đầu chạy Thuật toán ACO..."];
  history.slice(0, limit).forEach((cost, index) => {
    lines.push(`Best Solution in Iteration ${index + 1}/${history.length} = ${cost.toFixed(2)}`);
  });
  if (done && limit >= history.length) {
    lines.push("");
    lines.push("--- FINAL RESULT ---");
    lines.push(`Best Solution Cost: ${finalCost.toFixed(2)}`);
    lines.push("Best Solution Routes:");
    lines.push(JSON.stringify(routes));
  }

  els.consoleLog.innerHTML = "";
  lines.forEach((line) => {
    const div = document.createElement("div");
    div.className = "console-line";
    div.textContent = line;
    els.consoleLog.append(div);
  });
  els.consoleLog.scrollTop = els.consoleLog.scrollHeight;
}

function updateMetrics(iterationLimit = activeResult.history.length) {
  const currentHistory = activeResult.history.slice(0, Math.max(0, iterationLimit));
  const currentCost = currentHistory.length
    ? currentHistory[currentHistory.length - 1]
    : activeResult.best.cost;
  els.iterationMetric.textContent = `${Math.min(iterationLimit, activeResult.history.length)}/${activeResult.history.length}`;
  els.costMetric.textContent = Number(currentCost).toFixed(2);
  els.routeMetric.textContent = String(activeResult.best.routes.length);
  els.servedMetric.textContent = String(countServed(activeResult.best.routes));
}

function updateTitle() {
  if (mode === "sample") {
    els.mapTitle.textContent = "Kết quả mẫu: Best Solution Cost 68.21";
    return;
  }
  els.mapTitle.textContent = `Chạy mới: Best Solution Cost ${activeResult.best.cost.toFixed(2)}`;
}

function setPhase(phase) {
  document.querySelectorAll(".method-step").forEach((step) => {
    step.classList.toggle("active", step.dataset.phase === phase);
  });
}

function renderAll() {
  updateTitle();
  updateLegend(activeResult.best.routes);
  updateMetrics(activeIteration);
  updateRouteSummary();
  buildConsole(
    activeResult.history,
    activeResult.best.routes,
    activeResult.best.cost,
    activeIteration,
    activeIteration >= activeResult.history.length
  );
  drawCostChart(activeResult.history, activeIteration);
  drawRouteMap();
}

async function playResult(result) {
  if (isRunning) {
    return;
  }

  isRunning = true;
  els.runBtn.disabled = true;
  activeResult = result;
  activeIteration = 0;
  routeReveal = 0;
  setPhase("graph");
  renderAll();
  await wait(360);

  for (let index = 1; index <= activeResult.history.length; index += 1) {
    activeIteration = index;
    setPhase(index % 3 === 1 ? "ants" : index % 3 === 2 ? "best" : "pheromone");
    routeReveal = Math.min(0.2 + index / activeResult.history.length * 0.8, 1);
    renderAll();
    await wait(320);
  }

  setPhase("pheromone");
  await animateRouteReveal();
  activeIteration = activeResult.history.length;
  routeReveal = 1;
  renderAll();
  els.runBtn.disabled = false;
  isRunning = false;
}

async function animateRouteReveal() {
  const frames = 24;
  for (let frame = 0; frame <= frames; frame += 1) {
    routeReveal = frame / frames;
    drawRouteMap();
    await wait(18);
  }
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function setMode(nextMode) {
  if (isRunning) {
    return;
  }
  mode = nextMode;
  els.sampleModeBtn.classList.toggle("active", mode === "sample");
  els.liveModeBtn.classList.toggle("active", mode === "live");
  els.liveSettings.classList.toggle("visible", mode === "live");
  activeResult = mode === "sample" ? makeSampleResult() : runAco(makeConfigFromControls());
  activeIteration = activeResult.history.length;
  routeReveal = 1;
  setPhase("graph");
  renderAll();
}

function updateRangeOutputs() {
  els.antsOutput.textContent = els.antsInput.value;
  els.iterationsOutput.textContent = els.iterationsInput.value;
}

function nearestNode(clientX, clientY) {
  const rect = els.routeCanvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  let nearest = null;
  let nearestDistance = 12;
  for (const entry of canvasPositions) {
    const distance = Math.hypot(entry.x - x, entry.y - y);
    if (distance < nearestDistance) {
      nearest = entry;
      nearestDistance = distance;
    }
  }
  return { nearest, x, y };
}

function setupEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      document.querySelectorAll(".tab").forEach((other) => {
        const active = other === tab;
        other.classList.toggle("active", active);
        other.setAttribute("aria-selected", String(active));
      });
      document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.toggle("active", pane.id === `tab-${target}`);
      });
      drawCostChart(activeResult.history, activeIteration);
    });
  });

  els.sampleModeBtn.addEventListener("click", () => setMode("sample"));
  els.liveModeBtn.addEventListener("click", () => setMode("live"));

  els.runBtn.addEventListener("click", async () => {
    if (mode === "sample") {
      await playResult(makeSampleResult());
      return;
    }

    buildConsole([], [], 0, 0, false);
    els.consoleLog.firstChild.textContent = "Đang chạy ACO mới bằng tham số hiện tại...";
    await wait(30);
    const result = runAco(makeConfigFromControls());
    await playResult(result);
  });

  els.resetBtn.addEventListener("click", () => {
    if (!isRunning) {
      setMode(mode);
    }
  });

  [els.antsInput, els.iterationsInput].forEach((input) => {
    input.addEventListener("input", updateRangeOutputs);
    input.addEventListener("change", () => {
      if (mode === "live" && !isRunning) {
        setMode("live");
      }
    });
  });

  els.seedInput.addEventListener("change", () => {
    if (mode === "live" && !isRunning) {
      setMode("live");
    }
  });

  els.routeCanvas.addEventListener("mousemove", (event) => {
    const { nearest, x, y } = nearestNode(event.clientX, event.clientY);
    if (!nearest) {
      els.tooltip.hidden = true;
      return;
    }
    const routeIndex = activeResult.best.routes.findIndex((route) => route.includes(nearest.id));
    els.tooltip.hidden = false;
    els.tooltip.style.left = `${Math.min(x + 14, els.routeCanvas.clientWidth - 205)}px`;
    els.tooltip.style.top = `${Math.max(12, y - 16)}px`;
    const point = nearest.point;
    const label = nearest.id === 0
      ? "Depot 0"
      : `Khách ${nearest.id} · Xe ${routeIndex + 1}`;
    els.tooltip.innerHTML = `<strong>${label}</strong><br>lat ${point.lat.toFixed(5)}, lng ${point.lng.toFixed(5)}<br>vol ${point.vol}, crowd cost ${point.crowdCost}`;
  });

  els.routeCanvas.addEventListener("mouseleave", () => {
    els.tooltip.hidden = true;
  });

  window.addEventListener("resize", () => {
    drawRouteMap();
    drawCostChart(activeResult.history, activeIteration);
  });
}

updateRangeOutputs();
setupEvents();
renderAll();
