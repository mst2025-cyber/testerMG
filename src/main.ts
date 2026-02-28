import './style.css';
import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult
} from '@mediapipe/tasks-vision';
import * as THREE from 'three';

// --- Configuration ---
// --- DOM elements ---
const videoElement = document.getElementById('webcam') as HTMLVideoElement;
const statusText = document.getElementById('status-text') as HTMLElement;
const emotionDisplay = document.getElementById('emotion-display') as HTMLElement;
const detailDisplay = document.getElementById('detail-display') as HTMLElement;
const scaleSlider = document.getElementById('scale-slider') as HTMLInputElement;
const scaleValueDisplay = document.getElementById('scale-value') as HTMLElement;
const aspectSlider = document.getElementById('aspect-slider') as HTMLInputElement;
const aspectValueDisplay = document.getElementById('aspect-value') as HTMLElement;

// --- State Variables ---
let faceLandmarker: FaceLandmarker;
let lastVideoTime = -1;
let currentScale = 4.0;
let currentAspect = 1.0;

// --- Three.js Setup ---
const canvas = document.getElementById('three-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

// Face points visualization (simple points)
const geometry = new THREE.BufferGeometry();
const material = new THREE.PointsMaterial({ color: 0x00f2fe, size: 0.016 });
const points = new THREE.Points(geometry, material);
scene.add(points);

// Update scale when slider moves
scaleSlider.addEventListener('input', () => {
  currentScale = parseFloat(scaleSlider.value);
  scaleValueDisplay.innerText = `${currentScale.toFixed(1)}x`;
  // Adjust point size based on scale (base size 0.004 * currentScale)
  material.size = 0.004 * currentScale;
});

// Update aspect ratio stretch
aspectSlider.addEventListener('input', () => {
  currentAspect = parseFloat(aspectSlider.value);
  aspectValueDisplay.innerText = `${currentAspect.toFixed(2)}x`;
});

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);
const pointLight = new THREE.PointLight(0xffffff, 1);
pointLight.position.set(5, 5, 5);
scene.add(pointLight);

function handleResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', handleResize);

// --- Real-time Processing ---
async function setupFaceLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU" // Will try GPU first, MediaPipe handles fallback internally usually, but explicitly using GPU can fail.
    },
    outputFaceBlendshapes: true,
    runningMode: "VIDEO",
    numFaces: 1
  });
  statusText.innerText = "AI CORE 稼働中";
}

async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
    audio: false
  });
  videoElement.srcObject = stream;
  videoElement.addEventListener('loadeddata', predictWebcam);
}

function updateUI(result: FaceLandmarkerResult) {
  if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
    const blendshapes = result.faceBlendshapes[0].categories;

    // Create UI map
    const scores: Record<string, number> = {
      'Happy': 0, 'Sad': 0, 'Angry': 0, 'Surprised': 0, 'Blink': 0, 'MouthOpen': 0
    };

    blendshapes.forEach(shape => {
      // Map MediaPipe blendshapes to human readable emotions
      if (shape.categoryName === 'eyeBlinkLeft' || shape.categoryName === 'eyeBlinkRight') {
        scores['Blink'] = Math.max(scores['Blink'], shape.score);
      }
      if (shape.categoryName === 'jawOpen') {
        scores['MouthOpen'] = shape.score;
      }
      if (shape.categoryName === 'mouthSmileLeft' || shape.categoryName === 'mouthSmileRight') {
        scores['Happy'] = Math.max(scores['Happy'], shape.score);
      }
      if (shape.categoryName === 'browDownLeft' || shape.categoryName === 'browDownRight') {
        scores['Angry'] = Math.max(scores['Angry'], shape.score);
      }
      if (shape.categoryName === 'browInnerUp') {
        scores['Surprised'] = shape.score;
      }
      if (shape.categoryName === 'mouthFrownLeft' || shape.categoryName === 'mouthFrownRight') {
        scores['Sad'] = Math.max(scores['Sad'], shape.score);
      }
    });

    renderBars(scores);
  }
}

function renderBars(scores: Record<string, number>) {
  emotionDisplay.innerHTML = '';
  detailDisplay.innerHTML = '';

  const mainEmotions = ['Happy', 'Sad', 'Angry', 'Surprised'];
  const details = ['Blink', 'MouthOpen'];

  mainEmotions.forEach(emo => {
    emotionDisplay.appendChild(createProgressBar(emo, scores[emo]));
  });

  details.forEach(det => {
    detailDisplay.appendChild(createProgressBar(det, scores[det]));
  });
}

function createProgressBar(name: string, score: number) {
  const container = document.createElement('div');
  container.className = 'emotion-item';
  container.innerHTML = `
    <div class="emotion-name">${name} (${(score * 100).toFixed(0)}%)</div>
    <div class="progress-bar">
      <div class="progress-fill" style="width: ${score * 100}%"></div>
    </div>
  `;
  return container;
}

function update3DScene(result: FaceLandmarkerResult) {
  if (result.faceLandmarks && result.faceLandmarks.length > 0) {
    const landmarks = result.faceLandmarks[0];
    const positions = new Float32Array(landmarks.length * 3);

    landmarks.forEach((landmark, i) => {
      // Landmarks are in normalized coordinates [0, 1]
      // Project them into Three.js space using adjustable scale and aspect
      positions[i * 3] = (landmark.x - 0.5) * currentScale * currentAspect;
      positions[i * 3 + 1] = -(landmark.y - 0.5) * currentScale;
      positions[i * 3 + 2] = -landmark.z * currentScale; // z is depth
    });

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.attributes.position.needsUpdate = true;
  }
  renderer.render(scene, camera);
}

async function predictWebcam() {
  if (videoElement.currentTime !== lastVideoTime) {
    lastVideoTime = videoElement.currentTime;
    const startTimeMs = performance.now();
    const result = faceLandmarker.detectForVideo(videoElement, startTimeMs);

    updateUI(result);
    update3DScene(result);
  }
  requestAnimationFrame(predictWebcam);
}

// Initialize
async function init() {
  console.log("Initializing app...");
  try {
    console.log("Setting up Face Landmarker...");
    await setupFaceLandmarker();
    console.log("Face Landmarker setup complete.");

    console.log("Setting up Webcam...");
    await setupWebcam();
    console.log("Webcam setup complete.");

    console.log("App initialization finished successfully.");
  } catch (error) {
    console.error("Initialization failed:", error);
    statusText.innerText = "エラー: " + (error as Error).message;
  }
}

init();
