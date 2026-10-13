import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const countEl = document.getElementById("count");
const statusEl = document.getElementById("status");
const detailsEl = document.getElementById("details");
const startBtn = document.getElementById("start");

// Landmark indices: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
const WRIST = 0;
const FINGERS = [
  // [name, mcp, pip, tip]
  ["index", 5, 6, 8],
  ["middle", 9, 10, 12],
  ["ring", 13, 14, 16],
  ["pinky", 17, 18, 20],
];

let handLandmarker;
let lastVideoTime = -1;

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Angle (degrees) at joint b formed by a-b-c. 180 = perfectly straight.
function angle(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const mag = Math.hypot(v1.x, v1.y, v1.z) * Math.hypot(v2.x, v2.y, v2.z);
  return (Math.acos(Math.min(1, Math.max(-1, dot / mag))) * 180) / Math.PI;
}

// Uses 3D world landmarks (meters, hand-centered) so it works at any hand rotation.
function fingersUp(lm) {
  const up = [];

  // Thumb: straight-ish AND tip pointing away from the palm (far from pinky base)
  const thumbStraight = angle(lm[2], lm[3], lm[4]) > 150;
  const thumbOut = dist(lm[4], lm[17]) > dist(lm[2], lm[17]);
  if (thumbStraight && thumbOut) up.push("thumb");

  // Other fingers: tip is farther from the wrist than the middle knuckle
  for (const [name, mcp, pip, tip] of FINGERS) {
    const extended = dist(lm[WRIST], lm[tip]) > dist(lm[WRIST], lm[pip]);
    const straight = angle(lm[mcp], lm[pip], lm[tip]) > 140;
    if (extended && straight) up.push(name);
  }
  return up;
}

async function init() {
  statusEl.textContent = "Loading hand model…";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

async function startCamera() {
  startBtn.disabled = true;
  try {
    if (!handLandmarker) await init();
    statusEl.textContent = "Requesting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    createBoxes();
    statusEl.textContent = "Tracking. Pinch thumb + index on a box to grab it, open to drop.";
    startBtn.hidden = true;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err.message}`;
    startBtn.disabled = false;
  }
}

function loop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(video, performance.now());
    render(result);
  }
  requestAnimationFrame(loop);
}

// ---- Draggable boxes -------------------------------------------------------
// Boxes live in canvas pixel coordinates (same space as the landmarks), so they
// get mirrored together with the video and line up with your hand on screen.

const BOX_COLORS = ["#f87171", "#60a5fa", "#facc15", "#c084fc"];
let boxes = [];

// Pinch uses hysteresis so it doesn't flicker at the threshold.
// Values are a fraction of hand size (wrist to middle-finger knuckle).
const PINCH_START = 0.3;
const PINCH_END = 0.45;
const SMOOTHING = 0.5; // 0 = no smoothing, closer to 1 = smoother but laggier

// Per-hand state, keyed by MediaPipe's handedness label so two hands don't swap.
const hands = new Map(); // key -> { pinching, box, offsetX, offsetY, x, y }

function createBoxes() {
  const size = Math.round(canvas.height * 0.18);
  const gap = (canvas.width - BOX_COLORS.length * size) / (BOX_COLORS.length + 1);
  boxes = BOX_COLORS.map((color, i) => ({
    x: gap + i * (size + gap),
    y: canvas.height * 0.15,
    w: size,
    h: size,
    color,
  }));
}

function boxAt(x, y) {
  // Search from the top of the stack down (last drawn = on top)
  for (let i = boxes.length - 1; i >= 0; i--) {
    const b = boxes[i];
    if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) return b;
  }
  return null;
}

function toPx(p) {
  return { x: p.x * canvas.width, y: p.y * canvas.height };
}

// lm = screen landmarks (for where the cursor is), world = 3D landmarks in meters
// (for whether you're pinching). Measuring the pinch in 3D means turning or
// tilting your hand doesn't change the result, which it did with screen pixels.
function updatePinch(key, lm, world) {
  const thumb = toPx(lm[4]);
  const index = toPx(lm[8]);
  const handSize = dist(world[WRIST], world[9]);
  const ratio = dist(world[4], world[8]) / handSize;
  const mid = { x: (thumb.x + index.x) / 2, y: (thumb.y + index.y) / 2 };

  let s = hands.get(key);
  if (!s) {
    s = { pinching: false, box: null, offsetX: 0, offsetY: 0, ...mid };
    hands.set(key, s);
  }
  s.seen = true;
  s.ratio = ratio;
  s.x += (mid.x - s.x) * (1 - SMOOTHING);
  s.y += (mid.y - s.y) * (1 - SMOOTHING);

  if (!s.pinching && ratio < PINCH_START) {
    s.pinching = true;
    const box = boxAt(s.x, s.y);
    // Don't steal a box the other hand is already holding
    if (box && ![...hands.values()].some((h) => h !== s && h.box === box)) {
      s.box = box;
      s.offsetX = s.x - box.x;
      s.offsetY = s.y - box.y;
      boxes.splice(boxes.indexOf(box), 1);
      boxes.push(box); // bring to front
    }
  } else if (s.pinching && ratio > PINCH_END) {
    s.pinching = false;
    s.box = null;
  }

  if (s.box) {
    s.box.x = clamp(s.x - s.offsetX, 0, canvas.width - s.box.w);
    s.box.y = clamp(s.y - s.offsetY, 0, canvas.height - s.box.h);
  }
  return s;
}

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

function drawBoxes() {
  const held = new Set([...hands.values()].map((h) => h.box));
  for (const b of boxes) {
    ctx.fillStyle = b.color;
    ctx.globalAlpha = held.has(b) ? 1 : 0.75;
    ctx.fillRect(b.x, b.y, b.w, b.h);
    ctx.globalAlpha = 1;
    if (held.has(b)) {
      ctx.lineWidth = 6;
      ctx.strokeStyle = "#ffffff";
      ctx.strokeRect(b.x, b.y, b.w, b.h);
    }
  }
}

function drawCursor(s) {
  ctx.beginPath();
  ctx.arc(s.x, s.y, s.pinching ? 10 : 16, 0, Math.PI * 2);
  ctx.fillStyle = s.pinching ? "#ffffff" : "rgba(255,255,255,0.25)";
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();
}

// ---- Frame rendering -------------------------------------------------------

function render(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const draw = new DrawingUtils(ctx);

  for (const s of hands.values()) s.seen = false;
  const usedKeys = new Set();
  const states = result.landmarks.map((lm, i) => {
    let key = result.handednesses[i]?.[0]?.categoryName ?? String(i);
    if (usedKeys.has(key)) key += i; // both hands occasionally get the same label
    usedKeys.add(key);
    return updatePinch(key, lm, result.worldLandmarks[i]);
  });
  // A hand that left the frame drops whatever it was holding
  for (const [key, s] of hands) if (!s.seen) hands.delete(key);

  drawBoxes();

  if (!result.landmarks.length) {
    countEl.textContent = "–";
    detailsEl.textContent = "No hand detected";
    return;
  }

  let total = 0;
  const parts = [];

  result.landmarks.forEach((lm, i) => {
    draw.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#4ade80", lineWidth: 4 });
    draw.drawLandmarks(lm, { color: "#ffffff", radius: 4 });
    drawCursor(states[i]);

    const up = fingersUp(result.worldLandmarks[i]);
    total += up.length;

    const s = states[i];
    const pinch =
      ` — pinch ${s.ratio.toFixed(2)}` + (s.box ? " (holding box)" : s.pinching ? " (pinching)" : "");
    parts.push(`Hand ${i + 1}: ${up.length}${up.length ? ` (${up.join(", ")})` : ""}${pinch}`);
  });

  countEl.textContent = total;
  detailsEl.textContent = parts.join("  ·  ");
}

startBtn.addEventListener("click", startCamera);
