// --------------------------------------------------
// Motio observation engine, shared by caregiver.html.
//
// Runs the camera / video, MediaPipe hands, the per-hand buffers
// sent to /observe/update, object frames sent to /observe/frame,
// and caregiver questions sent to /assistant/ask.
//
// It also mirrors everything to researcher.html over a
// BroadcastChannel. That channel only connects pages open in the
// same browser on the same computer; nothing extra is sent to the
// server. Camera frames are mirrored only while a researcher page
// is open.
// --------------------------------------------------

import {
    SEQUENCE_LENGTH,
    MEDIAPIPE_URL,
    createHandLandmarker,
    makeFeatures,
    toPixelLandmarks,
    HandTracker,
    detectionsFromResult,
    startCamera,
    cameraErrorMessage
} from "./hand-tracking.js?v=4";


export const HAND_COLORS = ["#2e9e6b", "#3d7bd9"];
export const OBJECT_COLOR = "#f2a541";

export const CHANNEL_NAME = "motio-observer";

// The hand model is scored this often, per visible hand.
const TICK_MS = 250;

// A small frame goes to the local object detector this often.
const OBJECT_MS = 500;

// The latest boxes stay until newer ones arrive, at most this long.
const OBJECT_STALE_MS = 4000;

// Mirrored preview for the researcher page.
const MIRROR_MS = 200;
const MIRROR_WIDTH = 480;

// The researcher page says hello this often; frames stop when it closes.
const RESEARCHER_TIMEOUT_MS = 5000;


async function postJson(path, body) {

    const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(data.detail || `${path} ${response.status}`);
    }

    return data;
}


export class Observer {

    constructor({ video, canvas, overlay = "friendly" }) {

        this.video = video;
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");

        // "friendly": thin lines, object names only. "off": nothing drawn.
        this.overlay = overlay;

        this.sessionId =
            crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random();

        this.listeners = {};
        this.detectObjects = ["localhost", "127.0.0.1"].includes(location.hostname);

        this.ageMonths = null;
        this.config = null;

        this.handLandmarker = null;
        this.drawingUtils = null;
        this.HandLandmarker = null;

        this.tracker = new HandTracker(2);

        this.hands = this.tracker.slots.map(slot => ({
            slot,
            featureBuffer: [],
            landmarkBuffer: [],
            handedness: "Unknown",
            landmarks: null
        }));

        this.running = false;
        this.lastVideoTime = -1;
        this.lastTick = 0;
        this.tickInFlight = false;

        this.objectState = null;
        this.objectInFlight = false;
        this.lastObjectSend = 0;

        this.lastHandResult = null;

        this.frameCanvas = document.createElement("canvas");
        this.mirrorCanvas = document.createElement("canvas");
        this.lastMirror = 0;
        this.researcherSeen = 0;

        this.channel = "BroadcastChannel" in window ? new BroadcastChannel(CHANNEL_NAME) : null;

        if (this.channel) {
            this.channel.onmessage = event => this._onChannel(event.data);
        }

        video.onseeking = () => this._resetBuffers();
    }

    // ------------------------------------------------
    // Events for the page (and the researcher mirror)
    // ------------------------------------------------

    on(type, handler) {

        (this.listeners[type] = this.listeners[type] || []).push(handler);
    }

    _emit(type, data, mirror = true) {

        for (const handler of this.listeners[type] || []) {
            handler(data);
        }

        if (mirror) {
            this._mirror(type, data);
        }
    }

    _mirror(type, data) {

        if (this.channel) {
            this.channel.postMessage({ type, data, sessionId: this.sessionId, t: Date.now() });
        }
    }

    _onChannel(message) {

        if (message.type === "hello") {

            this.researcherSeen = performance.now();

            this._mirror("session", {
                sessionId: this.sessionId,
                ageMonths: this.ageMonths,
                running: this.running,
                detectObjects: this.detectObjects,
                threshold: this.config ? this.config.threshold : null
            });
        }
    }

    get researcherOpen() {

        return performance.now() - this.researcherSeen < RESEARCHER_TIMEOUT_MS;
    }

    // ------------------------------------------------
    // Setup
    // ------------------------------------------------

    async init() {

        this._emit("status", { text: "Getting ready...", kind: "info" });

        this.config = await (await fetch("/config")).json();

        const vision = await import(MEDIAPIPE_URL);

        this.HandLandmarker = vision.HandLandmarker;

        this.handLandmarker =
            await createHandLandmarker(vision.HandLandmarker, vision.FilesetResolver);

        this.drawingUtils = new vision.DrawingUtils(this.ctx);

        this._emit("status", { text: "Ready", kind: "ready" });
    }

    async startCamera(which) {

        this.video.removeAttribute("src");
        this.video.controls = false;

        try {

            const camera = await startCamera(this.video, which);

            this._begin();
            this._emit("source", { kind: "camera", note: camera.note, deviceId: camera.deviceId });

            return camera;

        } catch (error) {

            const text = cameraErrorMessage(error);

            this._emit("status", { text, kind: "error" });

            throw new Error(text);
        }
    }

    useVideoFile(file) {

        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
            this.video.srcObject = null;
        }

        this.video.src = URL.createObjectURL(file);
        this.video.controls = true;
        this.video.autoplay = false;

        this._begin();
        this._emit("source", { kind: "video", note: `Video: ${file.name}` });
    }

    stop() {

        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
            this.video.srcObject = null;
        }

        this.video.pause();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this._emit("source", { kind: "stopped", note: "" });
    }

    _begin() {

        if (!this.running) {
            this.running = true;
            requestAnimationFrame(() => this._loop());
        }
    }

    setDetectObjects(on) {

        this.detectObjects = on;

        if (!on) {
            this.objectState = null;
            this._emit("objects", this._objectSummary());
        }
    }

    // ------------------------------------------------
    // Caregiver assistant
    // ------------------------------------------------

    async setAge(months) {

        this.ageMonths = months;

        await postJson("/observe/profile", {
            session_id: this.sessionId,
            child_age_months: months
        });

        this._emit("age", { months });
    }

    async ask(question) {

        this._emit("ask", { question });

        try {

            const reply = await postJson("/assistant/ask", {
                session_id: this.sessionId,
                question
            });

            this._emit("answer", { question, reply });

            return reply;

        } catch (error) {

            this._emit("answer_error", { question, error: error.message });

            throw error;
        }
    }

    async newConversation() {

        await postJson("/assistant/reset", { session_id: this.sessionId });

        this._emit("conversation_reset", {});
    }

    // ------------------------------------------------
    // Per-frame loop
    // ------------------------------------------------

    _resetBuffers() {

        for (const hand of this.hands) {
            hand.featureBuffer = [];
            hand.landmarkBuffer = [];
        }
    }

    _loop() {

        const video = this.video;

        if (
            this.handLandmarker
            && video.readyState >= 2
            && video.currentTime !== this.lastVideoTime
        ) {

            this.lastVideoTime = video.currentTime;

            this.canvas.width = video.videoWidth;
            this.canvas.height = video.videoHeight;

            const result = this.handLandmarker.detectForVideo(video, performance.now());

            const { matches, lost } = this.tracker.update(detectionsFromResult(result));

            for (const slot of lost) {

                const hand = this.hands[slot.id];

                hand.featureBuffer = [];
                hand.landmarkBuffer = [];
                hand.landmarks = null;
            }

            const seen = new Set();

            for (const { slot, detection } of matches) {

                const hand = this.hands[slot.id];

                seen.add(slot.id);

                hand.handedness = detection.handedness;
                hand.landmarks = detection.landmarks;

                hand.featureBuffer.push(
                    makeFeatures(
                        detection.landmarks,
                        detection.handedness,
                        this.config.joint_confidence_fill,
                        video.videoWidth,
                        video.videoHeight
                    )
                );

                hand.landmarkBuffer.push(
                    toPixelLandmarks(detection.landmarks, video.videoWidth, video.videoHeight)
                );

                if (hand.featureBuffer.length > SEQUENCE_LENGTH) {
                    hand.featureBuffer.shift();
                    hand.landmarkBuffer.shift();
                }
            }

            for (const hand of this.hands) {
                if (!seen.has(hand.slot.id)) {
                    hand.landmarks = null;
                }
            }

            this._draw();

            this._emit("frame_hands", { count: matches.length }, false);
        }

        const now = performance.now();

        if (!video.paused) {

            if (now - this.lastTick >= TICK_MS) {
                this.lastTick = now;
                this._sendTick();
            }

            if (this.detectObjects && now - this.lastObjectSend >= OBJECT_MS) {
                this.lastObjectSend = now;
                this._sendFrame();
            }

            if (this.researcherOpen && now - this.lastMirror >= MIRROR_MS) {
                this.lastMirror = now;
                this._mirrorFrame();
            }
        }

        requestAnimationFrame(() => this._loop());
    }

    _freshObjects() {

        if (
            !this.detectObjects
            || !this.objectState
            || performance.now() - this.objectState.receivedAt > OBJECT_STALE_MS
        ) {
            return [];
        }

        return this.objectState.objects;
    }

    _draw(ctx = this.ctx, width = this.canvas.width, height = this.canvas.height, style = this.overlay) {

        ctx.clearRect(0, 0, width, height);

        if (style === "off") {
            return;
        }

        const detailed = style === "detailed";

        ctx.lineWidth = Math.max(2, width / (detailed ? 400 : 500));
        ctx.font = `600 ${Math.max(13, Math.round(width / 55))}px -apple-system, sans-serif`;

        for (const o of this._freshObjects()) {

            const [x1, y1, x2, y2] = [o.box[0] * width, o.box[1] * height, o.box[2] * width, o.box[3] * height];
            const text = detailed ? `${o.label} ${(o.confidence * 100).toFixed(0)}%` : o.label;

            ctx.strokeStyle = OBJECT_COLOR;
            ctx.setLineDash(detailed ? [] : [8, 6]);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.setLineDash([]);

            const textWidth = ctx.measureText(text).width + 12;
            const textHeight = parseInt(ctx.font.split(" ")[1], 10) + 8;
            const ty = y1 > textHeight ? y1 - textHeight : y1;

            ctx.fillStyle = OBJECT_COLOR;
            ctx.beginPath();
            ctx.roundRect ? ctx.roundRect(x1, ty, textWidth, textHeight, 6) : ctx.rect(x1, ty, textWidth, textHeight);
            ctx.fill();
            ctx.fillStyle = "#1d1a14";
            ctx.fillText(text, x1 + 6, ty + textHeight - 7);
        }

        for (const hand of this.hands) {

            if (!hand.landmarks) {
                continue;
            }

            const color = HAND_COLORS[hand.slot.id];
            const points = hand.landmarks.map(p => [p.x * width, p.y * height]);

            ctx.strokeStyle = color;
            ctx.lineWidth = Math.max(2, width / (detailed ? 320 : 450));

            for (const { start, end } of this.HandLandmarker.HAND_CONNECTIONS) {
                ctx.beginPath();
                ctx.moveTo(...points[start]);
                ctx.lineTo(...points[end]);
                ctx.stroke();
            }

            ctx.fillStyle = detailed ? "#ff3b30" : color;

            for (const [x, y] of points) {
                ctx.beginPath();
                ctx.arc(x, y, Math.max(2, width / (detailed ? 300 : 420)), 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    // ------------------------------------------------
    // Server calls
    // ------------------------------------------------

    async _sendTick() {

        if (this.tickInFlight) {
            return;
        }

        const ready = this.hands.filter(h => h.featureBuffer.length === SEQUENCE_LENGTH);

        this.tickInFlight = true;

        try {

            const result = await postJson("/observe/update", {
                session_id: this.sessionId,
                t: performance.now() / 1000,
                video_size: [this.video.videoWidth, this.video.videoHeight],
                hands: ready.map(h => ({
                    slot: h.slot.id,
                    handedness: h.handedness,
                    sequence: h.featureBuffer,
                    landmarks: h.landmarkBuffer
                }))
            });

            // Hands still warming up (buffer not full) are reported too,
            // so pages can say so.
            result.warming = this.hands
                .filter(h => h.landmarks && h.featureBuffer.length < SEQUENCE_LENGTH)
                .map(h => ({ slot: h.slot.id, frames: h.featureBuffer.length, of: SEQUENCE_LENGTH }));

            this.lastHandResult = result;
            this._emit("hands", result);

        } catch (error) {

            console.error(error);
            this._emit("status", { text: "Lost connection to the Motio server.", kind: "error" });

        } finally {

            this.tickInFlight = false;
        }
    }

    _objectSummary() {

        return {
            enabled: this.detectObjects,
            objects: this._freshObjects(),
            ms: this.objectState ? this.objectState.ms : null,
            device: this.objectState ? this.objectState.device : null,
            roundTripMs: this.objectState ? this.objectState.roundTripMs : null,
            loading: this.detectObjects && !this.objectState
        };
    }

    async _sendFrame() {

        if (this.objectInFlight || !this.video.videoWidth) {
            return;
        }

        this.objectInFlight = true;

        try {

            const scale = Math.min(1, 640 / this.video.videoWidth);

            this.frameCanvas.width = Math.round(this.video.videoWidth * scale);
            this.frameCanvas.height = Math.round(this.video.videoHeight * scale);
            this.frameCanvas
                .getContext("2d")
                .drawImage(this.video, 0, 0, this.frameCanvas.width, this.frameCanvas.height);

            const started = performance.now();

            const blob = await new Promise(resolve =>
                this.frameCanvas.toBlob(resolve, "image/jpeg", 0.8));

            const response = await fetch(
                `/observe/frame?session_id=${encodeURIComponent(this.sessionId)}`
                + `&t=${performance.now() / 1000}`,
                { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob }
            );

            if (!response.ok) {
                throw new Error(`${response.status} ${await response.text()}`);
            }

            this.objectState = await response.json();
            this.objectState.receivedAt = performance.now();
            this.objectState.roundTripMs = Math.round(this.objectState.receivedAt - started);

            this._emit("objects", this._objectSummary());

        } catch (error) {

            console.error(error);
            this._emit("objects", { ...this._objectSummary(), error: error.message });

        } finally {

            this.objectInFlight = false;
        }
    }

    _mirrorFrame() {

        // Video + detailed overlay, downscaled, for researcher.html only.
        const video = this.video;

        if (!video.videoWidth) {
            return;
        }

        const width = MIRROR_WIDTH;
        const height = Math.round(video.videoHeight * width / video.videoWidth);

        this.mirrorCanvas.width = width;
        this.mirrorCanvas.height = height;

        const ctx = this.mirrorCanvas.getContext("2d");

        ctx.drawImage(video, 0, 0, width, height);

        const overlay = document.createElement("canvas");
        overlay.width = width;
        overlay.height = height;

        this._draw(overlay.getContext("2d"), width, height, "detailed");

        ctx.drawImage(overlay, 0, 0);

        this._mirror("frame", { src: this.mirrorCanvas.toDataURL("image/jpeg", 0.65) });
    }
}
