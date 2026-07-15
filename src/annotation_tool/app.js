/**
 * Antigravity Annotation Toolbox — Interactive Frontend Controller
 * Coordinates API requests, UI updates, keyboard shortcuts, playbacks, and interactive canvas manipulations.
 */

// Application State
const state = {
    videos: [],
    csvs: [],
    currentVideo: '',
    currentCsv: '',
    videoDetails: {
        width: 1920,
        height: 1080,
        total_frames: 0,
        fps: 30.0
    },
    currentFrame: 0,
    isPlaying: false,
    playInterval: null,
    playbackSpeed: 1.0,
    
    // Frame detections data: { frame_index: [ { track_id, class_name, x1, y1, x2, y2, confidence }, ... ] }
    frames: {},
    hasUnsavedChanges: false,
    
    // UI Interaction states
    selectedBoxIndex: -1, // Index within state.frames[currentFrame]
    hoveredBoxIndex: -1,  // Index within state.frames[currentFrame]
    activeHandle: null,    // 'tl', 'tr', 'bl', 'br', or 'move'
    dragStart: { x: 0, y: 0 },
    isDrawing: false,
    newBoxStart: { x: 0, y: 0 },
    newBoxEnd: { x: 0, y: 0 },
    editingMode: 'select', // 'select' or 'draw'
    
    // Cache for loaded frame images
    imageCache: new Map()
};

// Class Color Palette (matching style.css variables)
const CLASS_COLORS = {
    person: '#10b981',
    car: '#3b82f6',
    motorcycle: '#f97316',
    bus: '#d946ef',
    truck: '#eab308'
};

// Handle size for resizing bounding boxes (in canvas pixels)
const HANDLE_SIZE = 8;

// DOM Elements
const el = {
    videoSelect: document.getElementById('video-select'),
    csvSelect: document.getElementById('csv-select'),
    activeCsvName: document.getElementById('active-csv-name'),
    saveStatus: document.getElementById('save-status'),
    saveStatusText: document.getElementById('save-status-text'),
    
    // Sidebar properties
    propTrackId: document.getElementById('property-track-id'),
    btnApplyId: document.getElementById('btn-apply-id'),
    propPropagateId: document.getElementById('property-propagate-id'),
    propClass: document.getElementById('property-class'),
    
    // Sidebar detections
    detectionCount: document.getElementById('detection-count'),
    detectionsList: document.getElementById('detections-list'),
    
    // Viewport
    noVideoOverlay: document.getElementById('no-video-overlay'),
    canvas: document.getElementById('annotation-canvas'),
    
    // Timeline
    currentFrameText: document.getElementById('current-frame-text'),
    frameRange: document.getElementById('frame-range'),
    totalFramesText: document.getElementById('total-frames-text'),
    
    // Playback Buttons
    btnFirst: document.getElementById('btn-first'),
    btnPrev10: document.getElementById('btn-prev10'),
    btnPrev: document.getElementById('btn-prev'),
    btnPlay: document.getElementById('btn-play'),
    playIcon: document.getElementById('play-icon'),
    pauseIcon: document.getElementById('pause-icon'),
    btnNext: document.getElementById('btn-next'),
    btnNext10: document.getElementById('btn-next10'),
    btnLast: document.getElementById('btn-last'),
    
    // Action Buttons
    btnClonePrev: document.getElementById('btn-clone-prev'),
    jumpFrameInput: document.getElementById('jump-frame-input'),
    speedSelect: document.getElementById('speed-select'),
    fpsInput: document.getElementById('fps-input'),
    btnSave: document.getElementById('btn-save'),
    
    // Custom elements
    btnModeSelect: document.getElementById('btn-mode-select'),
    btnModeDraw: document.getElementById('btn-mode-draw'),
    csvPathInput: document.getElementById('csv-path-input'),
    btnLoadCustomCsv: document.getElementById('btn-load-custom-csv')
};

// Canvas 2D context
const ctx = el.canvas.getContext('2d');
let currentFrameImage = null; // Store currently drawn frame image object

// Initialize Application
async function init() {
    setupEventListeners();
    await fetchWorkspaceFiles();
    updateUIState();
}

// Fetch lists of videos and CSVs from the server
async function fetchWorkspaceFiles() {
    try {
        const [videosRes, csvsRes] = await Promise.all([
            fetch('/api/videos'),
            fetch('/api/csvs')
        ]);
        state.videos = await videosRes.json();
        state.csvs = await csvsRes.json();

        // Populate dropdowns
        el.videoSelect.innerHTML = '<option value="" disabled selected>Select a video...</option>';
        state.videos.forEach(v => {
            el.videoSelect.innerHTML += `<option value="${v}">${v}</option>`;
        });

        el.csvSelect.innerHTML = '<option value="" disabled selected>Select tracks CSV...</option>';
        state.csvs.forEach(c => {
            el.csvSelect.innerHTML += `<option value="${c}">${c}</option>`;
        });
    } catch (e) {
        console.error('Failed to load workspace files:', e);
        showNotification('Error loading files list from server.', 'error');
    }
}

// Set up event listeners
function setupEventListeners() {
    // Dropdowns
    el.videoSelect.addEventListener('change', handleVideoChange);
    el.csvSelect.addEventListener('change', handleCsvChange);

    // Sidebar Properties
    el.btnApplyId.addEventListener('click', applyTrackIdChange);
    el.propClass.addEventListener('change', handleClassChange);

    // Timeline Scrubber
    el.frameRange.addEventListener('input', (e) => {
        seekToFrame(parseInt(e.target.value));
    });

    // Playback control button clicks
    el.btnFirst.addEventListener('click', () => seekToFrame(0));
    el.btnPrev10.addEventListener('click', () => seekToFrame(state.currentFrame - 10));
    el.btnPrev.addEventListener('click', () => seekToFrame(state.currentFrame - 1));
    el.btnPlay.addEventListener('click', togglePlayback);
    el.btnNext.addEventListener('click', () => seekToFrame(state.currentFrame + 1));
    el.btnNext10.addEventListener('click', () => seekToFrame(state.currentFrame + 10));
    el.btnLast.addEventListener('click', () => seekToFrame(state.videoDetails.total_frames - 1));

    // Mode toggling event listeners
    el.btnModeSelect.addEventListener('click', () => setEditingMode('select'));
    el.btnModeDraw.addEventListener('click', () => setEditingMode('draw'));
    el.btnLoadCustomCsv.addEventListener('click', handleCustomCsvLoad);
    el.csvPathInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleCustomCsvLoad();
    });

    // Inputs & Action Button clicks
    el.btnClonePrev.addEventListener('click', clonePreviousFrameBoxes);
    el.jumpFrameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const frame = parseInt(el.jumpFrameInput.value);
            if (!isNaN(frame)) seekToFrame(frame);
            el.jumpFrameInput.value = '';
            el.jumpFrameInput.blur();
        }
    });
    el.speedSelect.addEventListener('change', (e) => {
        state.playbackSpeed = parseFloat(e.target.value);
        if (state.isPlaying) {
            // Restart playback interval with new speed
            stopPlayback();
            startPlayback();
        }
    });
    el.fpsInput.addEventListener('change', (e) => {
        let fps = parseInt(e.target.value);
        if (isNaN(fps) || fps <= 0) fps = 30;
        state.videoDetails.fps = fps;
        el.fpsInput.value = fps;
    });
    el.btnSave.addEventListener('click', saveCSVTracks);

    // Window Resize - redraw canvas
    window.addEventListener('resize', drawFrame);

    // Canvas Events for box selection/movement/resizing/drawing
    el.canvas.addEventListener('mousedown', handleCanvasMouseDown);
    el.canvas.addEventListener('mousemove', handleCanvasMouseMove);
    el.canvas.addEventListener('mouseup', handleCanvasMouseUp);
    el.canvas.addEventListener('mouseleave', () => {
        state.hoveredBoxIndex = -1;
        state.activeHandle = null;
        drawFrame();
    });

    // Prevent context menu on double click / drag
    el.canvas.addEventListener('contextmenu', e => e.preventDefault());

    // Keyboard Shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

// =========================================================================
// Event Handlers (Files and Setup)
// =========================================================================

async function handleVideoChange(e) {
    const video = e.target.value;
    if (!video) return;

    state.currentVideo = video;
    state.imageCache.clear(); // Clear cached frames for previous video

    try {
        // Fetch video details
        const detailsRes = await fetch(`/api/video-details?video=${encodeURIComponent(video)}`);
        state.videoDetails = await detailsRes.json();
        
        // Update timeline range
        el.frameRange.max = state.videoDetails.total_frames - 1;
        el.totalFramesText.textContent = state.videoDetails.total_frames;
        el.fpsInput.value = Math.round(state.videoDetails.fps);

        // Hide overlay if both video and CSV are loaded
        checkWorkspaceLoaded();

        // Load first frame
        seekToFrame(0);
    } catch (err) {
        console.error('Error opening video:', err);
        showNotification('Error loading video details.', 'error');
    }
}

async function handleCsvChange(e) {
    const csv = e.target.value;
    if (!csv) return;

    // Check if current files have unsaved changes
    if (state.hasUnsavedChanges) {
        if (!confirm('You have unsaved changes. Are you sure you want to load another CSV tracks file? Changes will be lost.')) {
            el.csvSelect.value = state.currentCsv;
            return;
        }
    }

    state.currentCsv = csv;
    el.activeCsvName.textContent = csv.split('/').pop();

    try {
        const csvRes = await fetch(`/api/csv-data?csv=${encodeURIComponent(csv)}`);
        state.frames = await csvRes.json();
        
        setUnsavedChanges(false);
        checkWorkspaceLoaded();
        
        // Refresh frame view
        seekToFrame(state.currentFrame);
    } catch (err) {
        console.error('Error loading CSV data:', err);
        showNotification('Error parsing CSV track data.', 'error');
    }
}

function checkWorkspaceLoaded() {
    if (state.currentVideo && state.currentCsv) {
        el.noVideoOverlay.classList.add('hidden');
    } else {
        el.noVideoOverlay.classList.remove('hidden');
    }
}

function setUnsavedChanges(hasChanges) {
    state.hasUnsavedChanges = hasChanges;
    if (hasChanges) {
        el.saveStatus.classList.remove('saved');
        el.saveStatus.classList.add('unsaved');
        el.saveStatusText.textContent = 'Unsaved changes';
        el.btnSave.disabled = false;
    } else {
        el.saveStatus.classList.remove('unsaved');
        el.saveStatus.classList.add('saved');
        el.saveStatusText.textContent = 'Saved to disk';
        el.btnSave.disabled = true;
    }
}

// =========================================================================
// API & Video Frame Loading
// =========================================================================

// Seek to a specific frame number
async function seekToFrame(frameIdx) {
    if (frameIdx < 0) frameIdx = 0;
    if (state.videoDetails.total_frames > 0 && frameIdx >= state.videoDetails.total_frames) {
        frameIdx = state.videoDetails.total_frames - 1;
    }

    state.currentFrame = frameIdx;
    state.selectedBoxIndex = -1;
    state.hoveredBoxIndex = -1;
    
    // Update UI elements
    el.frameRange.value = frameIdx;
    el.currentFrameText.textContent = frameIdx;

    updateUIState();

    // Fetch frame image
    const img = await loadFrameImage(state.currentVideo, frameIdx);
    currentFrameImage = img;

    // Draw frame and boxes
    drawFrame();
    
    // Populate Sidebar detections list
    populateDetectionsList();
}

// Load a video frame image (with caching)
function loadFrameImage(videoPath, frameIdx) {
    const cacheKey = `${videoPath}_frame_${frameIdx}`;
    if (state.imageCache.has(cacheKey)) {
        return Promise.resolve(state.imageCache.get(cacheKey));
    }

    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            // Keep cache size bounded (e.g. max 150 frames)
            if (state.imageCache.size > 150) {
                const firstKey = state.imageCache.keys().next().value;
                state.imageCache.delete(firstKey);
            }
            state.imageCache.set(cacheKey, img);
            resolve(img);
        };
        img.onerror = () => {
            console.error(`Failed to load frame image for frame ${frameIdx}`);
            resolve(null); // Fallback to drawing a black frame
        };
        img.src = `/api/frame?video=${encodeURIComponent(videoPath)}&frame=${frameIdx}`;
    });
}

// =========================================================================
// Playback Engine
// =========================================================================

function togglePlayback() {
    if (state.isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (!state.currentVideo) return;
    state.isPlaying = true;
    el.playIcon.classList.add('hidden');
    el.pauseIcon.classList.remove('hidden');

    const intervalMs = Math.round(1000 / (state.videoDetails.fps * state.playbackSpeed));
    state.playInterval = setInterval(() => {
        if (state.currentFrame >= state.videoDetails.total_frames - 1) {
            stopPlayback();
        } else {
            seekToFrame(state.currentFrame + 1);
        }
    }, intervalMs);
}

function stopPlayback() {
    state.isPlaying = false;
    el.playIcon.classList.remove('hidden');
    el.pauseIcon.classList.add('hidden');
    if (state.playInterval) {
        clearInterval(state.playInterval);
        state.playInterval = null;
    }
}

// =========================================================================
// Bounding Box Render & Canvas Draw
// =========================================================================

// Redraw frame image and bounding boxes onto the canvas
function drawFrame() {
    if (!state.currentVideo) return;

    // Resize canvas based on viewport container size, preserving video aspect ratio
    const container = document.querySelector('.viewport-container');
    const containerWidth = container.clientWidth - 20;
    const containerHeight = container.clientHeight - 20;
    
    const videoWidth = state.videoDetails.width || 1920;
    const videoHeight = state.videoDetails.height || 1080;
    const videoAspect = videoWidth / videoHeight;
    const containerAspect = containerWidth / containerHeight;

    let canvasWidth, canvasHeight;
    if (containerAspect > videoAspect) {
        // Limited by height
        canvasHeight = containerHeight;
        canvasWidth = containerHeight * videoAspect;
    } else {
        // Limited by width
        canvasWidth = containerWidth;
        canvasHeight = containerWidth / videoAspect;
    }

    // Set canvas dimensions (styled via CSS max-width/height to display properly)
    if (el.canvas.width !== canvasWidth || el.canvas.height !== canvasHeight) {
        el.canvas.width = canvasWidth;
        el.canvas.height = canvasHeight;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // 1. Draw video frame image
    if (currentFrameImage) {
        ctx.drawImage(currentFrameImage, 0, 0, canvasWidth, canvasHeight);
    } else {
        // Black placeholder if image not loaded
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    }

    // 2. Render bounding boxes
    const boxes = state.frames[state.currentFrame] || [];
    boxes.forEach((box, index) => {
        drawBoundingBox(box, index);
    });

    // 3. Render box currently being drawn
    if (state.isDrawing) {
        const start = imageToCanvas(state.newBoxStart.x, state.newBoxStart.y);
        const end = imageToCanvas(state.newBoxEnd.x, state.newBoxEnd.y);
        
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
        ctx.setLineDash([]);
        
        // Badge
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px Plus Jakarta Sans';
        ctx.fillText('DRAWING...', start.x, start.y - 4);
    }
}

// Convert image pixels to canvas display pixels
function imageToCanvas(x, y) {
    const scaleX = el.canvas.width / state.videoDetails.width;
    const scaleY = el.canvas.height / state.videoDetails.height;
    return { x: x * scaleX, y: y * scaleY };
}

// Convert canvas display pixels to original image pixels
function canvasToImage(cx, cy) {
    const scaleX = state.videoDetails.width / el.canvas.width;
    const scaleY = state.videoDetails.height / el.canvas.height;
    return { x: cx * scaleX, y: cy * scaleY };
}

// Draw a single bounding box on the canvas
function drawBoundingBox(box, index) {
    const color = CLASS_COLORS[box.class_name] || '#ffffff';
    const isSelected = (index === state.selectedBoxIndex);
    const isHovered = (index === state.hoveredBoxIndex);

    // Get display coordinates
    const p1 = imageToCanvas(box.x1, box.y1);
    const p2 = imageToCanvas(box.x2, box.y2);
    const w = p2.x - p1.x;
    const h = p2.y - p1.y;

    // Draw box border
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 4 : (isHovered ? 3 : 2);
    
    // Add cyan glow highlight if selected
    if (isSelected) {
        ctx.shadowColor = 'rgba(0, 229, 255, 0.6)';
        ctx.shadowBlur = 10;
        // Draw double border
        ctx.strokeRect(p1.x, p1.y, w, h);
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 1;
    }
    
    ctx.strokeRect(p1.x, p1.y, w, h);
    ctx.shadowBlur = 0; // Reset shadow

    // Draw ID and Class Badge Label above the box
    const label = `ID:${box.track_id} ${box.class_name}`;
    ctx.font = 'bold 11px Plus Jakarta Sans, Outfit';
    const textMetrics = ctx.measureText(label);
    const textWidth = textMetrics.width;
    
    ctx.fillStyle = color;
    ctx.fillRect(p1.x - 1, p1.y - 18, textWidth + 12, 18);

    ctx.fillStyle = '#ffffff';
    // Draw text centered in the badge
    ctx.fillText(label, p1.x + 6, p1.y - 5);

    // Draw Resize Handles if selected
    if (isSelected) {
        ctx.fillStyle = '#00e5ff';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;

        const handles = [
            { x: p1.x, y: p1.y }, // tl
            { x: p2.x, y: p1.y }, // tr
            { x: p1.x, y: p2.y }, // bl
            { x: p2.x, y: p2.y }  // br
        ];

        handles.forEach(hnd => {
            ctx.fillRect(hnd.x - HANDLE_SIZE/2, hnd.y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
            ctx.strokeRect(hnd.x - HANDLE_SIZE/2, hnd.y - HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE);
        });
    }
}

// =========================================================================
// Interactive Canvas Bounding Box Controls
// =========================================================================

function handleCanvasMouseDown(e) {
    if (!state.currentVideo || !state.currentCsv) return;

    // Get canvas relative mouse coordinates
    const rect = el.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Convert mouse position to original image coordinates
    const imgPos = canvasToImage(mx, my);

    // Left click interaction
    if (e.button === 0) {
        // Active Draw Mode handles click-and-drag drawing directly
        if (state.editingMode === 'draw') {
            state.isDrawing = true;
            state.newBoxStart = { x: imgPos.x, y: imgPos.y };
            state.newBoxEnd = { x: imgPos.x, y: imgPos.y };
            return;
        }

        // 1. Check if we clicked a resize handle of the SELECTED box
        if (state.selectedBoxIndex !== -1) {
            const box = (state.frames[state.currentFrame] || [])[state.selectedBoxIndex];
            const handle = getClickedHandle(box, mx, my);
            if (handle) {
                state.activeHandle = handle;
                state.dragStart = { x: imgPos.x, y: imgPos.y };
                return;
            }
        }

        // 2. Check if we clicked inside any box to select/move it
        const clickedBoxIdx = getBoxAtPosition(imgPos.x, imgPos.y);
        if (clickedBoxIdx !== -1) {
            selectBox(clickedBoxIdx);
            state.activeHandle = 'move';
            state.dragStart = { x: imgPos.x, y: imgPos.y };
            return;
        }

        // 3. Clicked empty space: deselect or start drawing a new box (Shift key fallback)
        selectBox(-1);
        if (e.shiftKey) {
            state.isDrawing = true;
            state.newBoxStart = { x: imgPos.x, y: imgPos.y };
            state.newBoxEnd = { x: imgPos.x, y: imgPos.y };
        }
    }
}

function handleCanvasMouseMove(e) {
    if (!state.currentVideo) return;

    const rect = el.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const imgPos = canvasToImage(mx, my);

    // 1. If actively drawing a new box
    if (state.isDrawing) {
        state.newBoxEnd = { x: imgPos.x, y: imgPos.y };
        drawFrame();
        return;
    }

    // 2. If actively dragging or resizing a box
    if (state.activeHandle && state.selectedBoxIndex !== -1) {
        const boxes = state.frames[state.currentFrame] || [];
        const box = boxes[state.selectedBoxIndex];
        
        const dx = imgPos.x - state.dragStart.x;
        const dy = imgPos.y - state.dragStart.y;
        
        if (state.activeHandle === 'move') {
            // Move entire box
            const w = box.x2 - box.x1;
            const h = box.y2 - box.y1;
            box.x1 += dx;
            box.y1 += dy;
            box.x2 = box.x1 + w;
            box.y2 = box.y1 + h;
        } else {
            // Resize box corner handles
            if (state.activeHandle === 'tl') {
                box.x1 = Math.min(imgPos.x, box.x2 - 5);
                box.y1 = Math.min(imgPos.y, box.y2 - 5);
            } else if (state.activeHandle === 'tr') {
                box.x2 = Math.max(imgPos.x, box.x1 + 5);
                box.y1 = Math.min(imgPos.y, box.y2 - 5);
            } else if (state.activeHandle === 'bl') {
                box.x1 = Math.min(imgPos.x, box.x2 - 5);
                box.y2 = Math.max(imgPos.y, box.y1 + 5);
            } else if (state.activeHandle === 'br') {
                box.x2 = Math.max(imgPos.x, box.x1 + 5);
                box.y2 = Math.max(imgPos.y, box.y1 + 5);
            }
        }
        
        state.dragStart = { x: imgPos.x, y: imgPos.y };
        setUnsavedChanges(true);
        drawFrame();
        return;
    }

    // 3. Hover cursors and selection checks
    if (state.editingMode === 'draw') {
        el.canvas.style.cursor = 'crosshair';
        return;
    }

    const boxes = state.frames[state.currentFrame] || [];
    
    // Check handles of selected box first
    if (state.selectedBoxIndex !== -1) {
        const box = boxes[state.selectedBoxIndex];
        const handle = getClickedHandle(box, mx, my);
        if (handle) {
            if (handle === 'tl' || handle === 'br') el.canvas.style.cursor = 'nwse-resize';
            else if (handle === 'tr' || handle === 'bl') el.canvas.style.cursor = 'nesw-resize';
            return;
        }
    }

    // Hover box border/interior check
    const hoverIdx = getBoxAtPosition(imgPos.x, imgPos.y);
    if (hoverIdx !== -1) {
        state.hoveredBoxIndex = hoverIdx;
        el.canvas.style.cursor = 'move';
    } else {
        state.hoveredBoxIndex = -1;
        el.canvas.style.cursor = e.shiftKey ? 'crosshair' : 'default';
    }

    drawFrame();
}

function handleCanvasMouseUp(e) {
    if (state.isDrawing) {
        // Complete drawing a new box
        state.isDrawing = false;
        
        const x1 = Math.min(state.newBoxStart.x, state.newBoxEnd.x);
        const y1 = Math.min(state.newBoxStart.y, state.newBoxEnd.y);
        const x2 = Math.max(state.newBoxStart.x, state.newBoxEnd.x);
        const y2 = Math.max(state.newBoxStart.y, state.newBoxEnd.y);
        
        // Only add if size is reasonable
        if ((x2 - x1) > 8 && (y2 - y1) > 8) {
            addNewBoundingBox(x1, y1, x2, y2);
        } else {
            drawFrame();
        }

        // Return back to select mode for subsequent actions
        setEditingMode('select');
    }

    state.activeHandle = null;
}

// Find if mouse is clicking inside a box (prioritizing smaller boxes for overlap handling)
function getBoxAtPosition(x, y) {
    const boxes = state.frames[state.currentFrame] || [];
    let bestIdx = -1;
    let smallestArea = Infinity;

    boxes.forEach((box, index) => {
        if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2) {
            const area = (box.x2 - box.x1) * (box.y2 - box.y1);
            if (area < smallestArea) {
                smallestArea = area;
                bestIdx = index;
            }
        }
    });

    return bestIdx;
}

// Find if mouse is clicking on a resize handle of the selected box
function getClickedHandle(box, mx, my) {
    const p1 = imageToCanvas(box.x1, box.y1);
    const p2 = imageToCanvas(box.x2, box.y2);

    const handles = {
        tl: { x: p1.x, y: p1.y },
        tr: { x: p2.x, y: p1.y },
        bl: { x: p1.x, y: p2.y },
        br: { x: p2.x, y: p2.y }
    };

    for (const [name, pos] of Object.entries(handles)) {
        if (Math.abs(mx - pos.x) <= HANDLE_SIZE / 2 + 2 && Math.abs(my - pos.y) <= HANDLE_SIZE / 2 + 2) {
            return name;
        }
    }
    return null;
}

// Select a bounding box and populate property controls
function selectBox(index) {
    state.selectedBoxIndex = index;
    
    // Highlight sidebar list
    const items = el.detectionsList.querySelectorAll('li');
    items.forEach(li => li.classList.remove('selected'));
    
    if (index === -1) {
        el.propTrackId.value = '';
        el.propTrackId.disabled = true;
        el.btnApplyId.disabled = true;
        el.propClass.disabled = true;
        el.propClass.value = 'person';
    } else {
        const box = (state.frames[state.currentFrame] || [])[index];
        el.propTrackId.value = box.track_id;
        el.propTrackId.disabled = false;
        el.btnApplyId.disabled = false;
        el.propClass.disabled = false;
        el.propClass.value = box.class_name;
        
        // Highlight active sidebar row
        const targetLi = el.detectionsList.querySelector(`li[data-index="${index}"]`);
        if (targetLi) {
            targetLi.classList.add('selected');
            targetLi.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    }

    drawFrame();
}

// Generate next available ID to initialize a new box
function getNextAvailableId() {
    let maxId = 0;
    // Scan current frame
    const boxes = state.frames[state.currentFrame] || [];
    boxes.forEach(b => {
        if (b.track_id > maxId) maxId = b.track_id;
    });

    // Scan previous frame if available
    if (state.currentFrame > 0) {
        const prevBoxes = state.frames[state.currentFrame - 1] || [];
        prevBoxes.forEach(b => {
            if (b.track_id > maxId) maxId = b.track_id;
        });
    }

    return maxId + 1;
}

// Add a brand new manually drawn box
function addNewBoundingBox(x1, y1, x2, y2) {
    const boxes = state.frames[state.currentFrame] || [];
    const nextId = getNextAvailableId();

    const newBox = {
        frame: state.currentFrame,
        track_id: nextId,
        class_name: 'car', // Default class
        x1: Math.round(x1),
        y1: Math.round(y1),
        x2: Math.round(x2),
        y2: Math.round(y2),
        confidence: 1.0,
        center_x: Math.round((x1 + x2) / 2),
        center_y: Math.round((y1 + y2) / 2)
    };

    boxes.push(newBox);
    state.frames[state.currentFrame] = boxes;
    
    setUnsavedChanges(true);
    seekToFrame(state.currentFrame); // Refresh view and select it
    selectBox(boxes.length - 1);
}

// Delete the selected box
function deleteSelectedBox() {
    if (state.selectedBoxIndex === -1) return;
    
    const boxes = state.frames[state.currentFrame] || [];
    boxes.splice(state.selectedBoxIndex, 1);
    state.frames[state.currentFrame] = boxes;

    setUnsavedChanges(true);
    selectBox(-1);
    seekToFrame(state.currentFrame);
    showNotification('Box deleted.', 'info');
}

// =========================================================================
// Sidebar Operations (Apply ID, Propagate Changes)
// =========================================================================

function applyTrackIdChange() {
    if (state.selectedBoxIndex === -1) return;

    const newId = parseInt(el.propTrackId.value);
    if (isNaN(newId) || newId < 0) {
        alert('Invalid Track ID. Must be a positive integer.');
        return;
    }

    const boxes = state.frames[state.currentFrame] || [];
    const box = boxes[state.selectedBoxIndex];
    const oldId = box.track_id;

    if (oldId === newId) return;

    const propagate = el.propPropagateId.checked;
    if (propagate) {
        // Change track ID across all subsequent frames
        let count = 0;
        const totalF = state.videoDetails.total_frames;
        
        for (let f = state.currentFrame; f < totalF; f++) {
            const frameBoxes = state.frames[f] || [];
            frameBoxes.forEach(b => {
                if (b.track_id === oldId) {
                    b.track_id = newId;
                    count++;
                }
            });
        }
        showNotification(`Propagated Track ID change in ${count} instances.`, 'success');
    } else {
        box.track_id = newId;
        showNotification('Track ID applied to current frame only.', 'info');
    }

    setUnsavedChanges(true);
    seekToFrame(state.currentFrame); // Refresh lists
}

function handleClassChange(e) {
    if (state.selectedBoxIndex === -1) return;
    
    const newClass = e.target.value;
    const boxes = state.frames[state.currentFrame] || [];
    const box = boxes[state.selectedBoxIndex];

    box.class_name = newClass;
    setUnsavedChanges(true);
    drawFrame();
    populateDetectionsList();
}

// =========================================================================
// Advanced Controls & Actions (Clone, Save, Shortcuts)
// =========================================================================

// Clone boxes from frame t-1 to frame t
function clonePreviousFrameBoxes() {
    if (state.currentFrame <= 0) return;
    
    const prevBoxes = state.frames[state.currentFrame - 1] || [];
    if (prevBoxes.length === 0) {
        showNotification('No boxes exist in the previous frame to clone.', 'warning');
        return;
    }

    if (!confirm(`Clone all ${prevBoxes.length} boxes from frame ${state.currentFrame - 1}? This will overwrite boxes in the current frame.`)) {
        return;
    }

    // Deep copy boxes
    const cloned = prevBoxes.map(b => ({
        ...b,
        frame: state.currentFrame,
        // Reset speed or other properties that must be recalculated
        velocity_ms: 0.0
    }));

    state.frames[state.currentFrame] = cloned;
    setUnsavedChanges(true);
    seekToFrame(state.currentFrame);
    showNotification(`Cloned ${cloned.length} boxes successfully.`, 'success');
}

// Save CSV Track coordinates to disk
async function saveCSVTracks() {
    if (!state.currentCsv) return;

    el.btnSave.disabled = true;
    el.btnSave.textContent = 'Saving changes...';

    try {
        const response = await fetch(`/api/save-csv?csv=${encodeURIComponent(state.currentCsv)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                frames: state.frames,
                fps: state.videoDetails.fps
            })
        });

        const result = await response.json();
        if (result.status === 'success') {
            setUnsavedChanges(false);
            showNotification(`Saved tracks to disk (${result.rows_written} rows written).`, 'success');
            // Re-fetch CSV data to ensure velocities/coordinates are synchronized with recalculated coordinates
            const csvRes = await fetch(`/api/csv-data?csv=${encodeURIComponent(state.currentCsv)}`);
            state.frames = await csvRes.json();
            seekToFrame(state.currentFrame);
        } else {
            throw new Error(result.message || 'Save failed');
        }
    } catch (e) {
        console.error('Error saving CSV tracks:', e);
        showNotification('Failed to save CSV changes.', 'error');
        el.btnSave.disabled = false;
    } finally {
        el.btnSave.innerHTML = `<svg class="btn-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M17 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm-5 16c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3zm3-10H5V5h10v4z"/></svg> Save CSV Changes [S]`;
    }
}

// Sidebar list populations
function populateDetectionsList() {
    const boxes = state.frames[state.currentFrame] || [];
    el.detectionCount.textContent = boxes.length;

    if (boxes.length === 0) {
        el.detectionsList.innerHTML = '<li class="empty-list-msg">No detections in this frame.</li>';
        return;
    }

    el.detectionsList.innerHTML = '';
    boxes.forEach((box, index) => {
        const badgeClass = box.class_name;
        const li = document.createElement('li');
        li.dataset.index = index;
        li.className = (index === state.selectedBoxIndex) ? 'selected' : '';
        
        li.innerHTML = `
            <span>ID #${box.track_id}</span>
            <span class="badge ${badgeClass}">${box.class_name}</span>
        `;
        
        li.addEventListener('click', () => selectBox(index));
        el.detectionsList.appendChild(li);
    });
}

// Update UI elements based on state change
function updateUIState() {
    // Properties sidebar buttons/selects
    const isBoxSelected = (state.selectedBoxIndex !== -1);
    el.propTrackId.disabled = !isBoxSelected;
    el.btnApplyId.disabled = !isBoxSelected;
    el.propClass.disabled = !isBoxSelected;

    // Timeline control bounds
    const isLoaded = (state.currentVideo && state.currentCsv);
    el.frameRange.disabled = !isLoaded;
    el.btnFirst.disabled = !isLoaded;
    el.btnPrev10.disabled = !isLoaded;
    el.btnPrev.disabled = !isLoaded;
    el.btnPlay.disabled = !isLoaded;
    el.btnNext.disabled = !isLoaded;
    el.btnNext10.disabled = !isLoaded;
    el.btnLast.disabled = !isLoaded;
    el.btnClonePrev.disabled = !isLoaded || state.currentFrame === 0;
}

// Show a temporary notification badge in the UI
function showNotification(message, type = 'info') {
    // Create notification element
    const notif = document.createElement('div');
    notif.className = `notification ${type}`;
    notif.textContent = message;
    
    // Position notification fixed on top right
    Object.assign(notif.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        backgroundColor: type === 'success' ? '#10b981' : (type === 'error' ? '#ef4444' : '#3b82f6'),
        color: '#ffffff',
        padding: '10px 20px',
        borderRadius: '6px',
        fontFamily: 'Plus Jakarta Sans',
        fontSize: '0.85rem',
        fontWeight: '600',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        zIndex: '1000',
        opacity: '0',
        transform: 'translateY(-20px)',
        transition: 'all 0.3s ease'
    });
    
    document.body.appendChild(notif);
    
    // Trigger transition
    setTimeout(() => {
        notif.style.opacity = '1';
        notif.style.transform = 'translateY(0)';
    }, 10);
    
    // Disappear after 3 seconds
    setTimeout(() => {
        notif.style.opacity = '0';
        notif.style.transform = 'translateY(-20px)';
        setTimeout(() => notif.remove(), 300);
    }, 3000);
}

// Keyboard shortcuts mappings
function handleKeyboardShortcuts(e) {
    // Don't trigger if focus is in an input field (to allow typing frame number or IDs)
    if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') {
        return;
    }

    switch (e.key) {
        case 'd':
        case 'ArrowRight':
        case '+':
            e.preventDefault();
            seekToFrame(state.currentFrame + 1);
            break;
        case 'a':
        case 'ArrowLeft':
        case '-':
            e.preventDefault();
            seekToFrame(state.currentFrame - 1);
            break;
        case 'f':
        case 'ArrowUp':
            e.preventDefault();
            seekToFrame(state.currentFrame + 10);
            break;
        case 'r':
        case 'ArrowDown':
            e.preventDefault();
            seekToFrame(state.currentFrame - 10);
            break;
        case ' ':
        case 'p':
            e.preventDefault();
            togglePlayback();
            break;
        case 'v':
            e.preventDefault();
            clonePreviousFrameBoxes();
            break;
        case 's':
            e.preventDefault();
            saveCSVTracks();
            break;
        case 'Delete':
        case 'Backspace':
            e.preventDefault();
            deleteSelectedBox();
            break;
    }
}

// Set drawing/selecting mode
function setEditingMode(mode) {
    state.editingMode = mode;
    if (mode === 'select') {
        el.btnModeSelect.classList.add('active');
        el.btnModeDraw.classList.remove('active');
        el.canvas.style.cursor = 'default';
    } else {
        el.btnModeSelect.classList.remove('active');
        el.btnModeDraw.classList.add('active');
        el.canvas.style.cursor = 'crosshair';
        selectBox(-1); // Clear selection when drawing
    }
    drawFrame();
}

// Load a custom CSV file specified via path input
async function handleCustomCsvLoad() {
    const csv = el.csvPathInput.value.trim();
    if (!csv) {
        alert('Please enter a valid CSV path.');
        return;
    }

    if (state.hasUnsavedChanges) {
        if (!confirm('You have unsaved changes. Are you sure you want to load another CSV tracks file? Changes will be lost.')) {
            return;
        }
    }

    state.currentCsv = csv;
    el.activeCsvName.textContent = csv.split('/').pop();

    try {
        const csvRes = await fetch(`/api/csv-data?csv=${encodeURIComponent(csv)}`);
        if (!csvRes.ok) throw new Error('CSV not found');
        state.frames = await csvRes.json();
        
        setUnsavedChanges(false);
        checkWorkspaceLoaded();
        
        // Refresh frame view
        seekToFrame(state.currentFrame);
        showNotification('CSV loaded successfully!', 'success');
    } catch (err) {
        console.error('Error loading custom CSV data:', err);
        showNotification('Error loading CSV. Verify the file path is correct.', 'error');
    }
}

// Run Startup
window.addEventListener('DOMContentLoaded', init);
