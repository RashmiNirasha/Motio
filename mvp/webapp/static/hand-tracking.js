// --------------------------------------------------
// Shared hand helpers for live-model.html
// and record-pincer.html
// --------------------------------------------------


export const SEQUENCE_LENGTH = 20;


export const MEDIAPIPE_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0";

export const HAND_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";


export async function createHandLandmarker(
    HandLandmarker,
    FilesetResolver
) {

    const vision =
        await FilesetResolver.forVisionTasks(
            MEDIAPIPE_URL + "/wasm"
        );

    return HandLandmarker.createFromOptions(
        vision,
        {
            baseOptions: {
                modelAssetPath: HAND_MODEL_URL,
                delegate: "CPU"
            },

            runningMode: "VIDEO",

            // Track both hands in the frame.
            numHands: 2,

            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        }
    );
}


// --------------------------------------------------
// Convert MediaPipe landmarks
// into SAME 69 feature structure
// used during ChildPlay training
// --------------------------------------------------

// ChildPlay normalised each hand by its pose detector's box, which is
// on average 2.47x larger than the extent of the hand's keypoints
// (median over the ChildPlay val split). MediaPipe has no such box, so
// the landmark extent is scaled by the same factor to match training.
const CHILDPLAY_BOX_PADDING = 2.47;

export function makeFeatures(
    landmarks,
    handedness,
    confidenceFill,
    videoWidth = 1,
    videoHeight = 1
) {

    // Work in pixels, like ChildPlay. MediaPipe's x and y are
    // normalised by width and height separately, which would
    // stretch the hand shape on a non-square camera.
    const W = videoWidth || 1;
    const H = videoHeight || 1;

    const xs = landmarks.map(p => p.x * W);
    const ys = landmarks.map(p => p.y * H);

    const wristX = xs[0];
    const wristY = ys[0];

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const boxW = Math.max(maxX - minX, 1) * CHILDPLAY_BOX_PADDING;
    const boxH = Math.max(maxY - minY, 1) * CHILDPLAY_BOX_PADDING;

    const scale = Math.max(boxW, boxH, 1);

    const isLeft =
        handedness
            .toLowerCase()
            .includes("left");

    const features = [];

    // -----------------------------
    // 21 × [relative x,y,confidence]
    // -----------------------------

    for (let i = 0; i < 21; i++) {

        let dx = (xs[i] - wristX) / scale;
        let dy = (ys[i] - wristY) / scale;

        // Same canonicalization used during training. (The live
        // model is trained with mirroring, so a wrong handedness
        // label does little harm.)
        if (isLeft) {
            dx *= -1;
        }

        features.push(dx);
        features.push(dy);

        // Neutral confidence value
        // from ChildPlay training mean.
        features.push(confidenceFill[i]);
    }

    // -----------------------------
    // Global features (image-normalised, as in training)
    // -----------------------------

    features.push(wristX / W);
    features.push(wristY / H);
    features.push((minX + maxX) / 2 / W);
    features.push((minY + maxY) / 2 / H);
    features.push(boxW / W);
    features.push(boxH / H);

    return features;
}


// --------------------------------------------------
// Landmarks for the pincer pipeline.
//
// MediaPipe x/y are normalized separately by width
// and height; scale back to pixels so distances and
// angles are not stretched by the aspect ratio.
// --------------------------------------------------

export function toPixelLandmarks(
    landmarks,
    videoWidth,
    videoHeight
) {

    return landmarks.map(p => [
        p.x * videoWidth,
        p.y * videoHeight,
        p.z * videoWidth
    ]);
}


// --------------------------------------------------
// Keep each physical hand in the same slot
// from frame to frame, by nearest wrist position.
//
// MediaPipe's result order can swap between frames,
// and handedness is unreliable when two hands of the
// same side (e.g. child + caregiver) are visible.
// --------------------------------------------------

// Larger wrist jumps (in normalized image units)
// between frames are treated as a different hand.
const MAX_WRIST_JUMP = 0.25;


export class HandTracker {

    constructor(maxHands = 2) {

        this.slots = Array.from(
            { length: maxHands },
            (_, id) => ({ id, active: false, wrist: null })
        );
    }


    // detections: [{ landmarks, handedness }]
    //
    // Returns:
    //   matches: [{ slot, detection }]
    //   lost:    slots whose hand disappeared this frame
    update(detections) {

        const pairs = [];

        detections.forEach((detection, d) => {

            const wrist = detection.landmarks[0];

            for (const slot of this.slots) {

                if (!slot.active) {
                    continue;
                }

                const distance = Math.hypot(
                    wrist.x - slot.wrist.x,
                    wrist.y - slot.wrist.y
                );

                if (distance <= MAX_WRIST_JUMP) {
                    pairs.push({ d, slot, distance });
                }
            }
        });

        pairs.sort((a, b) => a.distance - b.distance);

        const matches = [];
        const usedDetections = new Set();
        const usedSlots = new Set();

        for (const { d, slot } of pairs) {

            if (usedDetections.has(d) || usedSlots.has(slot.id)) {
                continue;
            }

            usedDetections.add(d);
            usedSlots.add(slot.id);
            matches.push({ slot, detection: detections[d] });
        }

        // Hands that no longer match end their track.
        const lost = this.slots.filter(
            slot => slot.active && !usedSlots.has(slot.id)
        );

        for (const slot of lost) {
            slot.active = false;
        }

        // New hands take any free slot.
        detections.forEach((detection, d) => {

            if (usedDetections.has(d)) {
                return;
            }

            const free = this.slots.find(
                slot => !slot.active && !usedSlots.has(slot.id)
            );

            if (free) {
                usedSlots.add(free.id);
                matches.push({ slot: free, detection });
            }
        });

        for (const { slot, detection } of matches) {
            slot.active = true;
            slot.wrist = detection.landmarks[0];
        }

        return { matches, lost };
    }
}


export function detectionsFromResult(result) {

    return (result.landmarks || []).map((landmarks, i) => ({
        landmarks,
        handedness:
            result.handedness
            && result.handedness[i]
            && result.handedness[i][0]
                ? result.handedness[i][0].categoryName
                : "Unknown"
    }));
}


// Start a camera. `which` is "user" (front), "environment" (back)
// or a deviceId from the camera list.
//
// A device without a back camera (most laptops) cannot satisfy
// "environment" strictly, so fall back to any camera and say so
// in `note`, instead of failing.
export async function startCamera(video, which) {

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {

        throw new DOMException(
            "Camera access needs HTTPS or localhost",
            "InsecureContext"
        );
    }

    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }

    const size = {
        width: { ideal: 1280 },
        height: { ideal: 720 }
    };

    const attempts =
        which === "user" || which === "environment"
            ? [
                { facingMode: { exact: which } },
                { facingMode: which },
                {}
            ]
            : [
                { deviceId: { exact: which } },
                {}
            ];

    let stream = null;
    let lastError = null;

    for (const attempt of attempts) {

        try {

            stream = await navigator.mediaDevices.getUserMedia({
                video: { ...attempt, ...size },
                audio: false
            });

            break;

        } catch (error) {

            lastError = error;

            // Permission problems and a busy camera won't be
            // fixed by relaxing the constraints.
            if (
                error.name === "NotAllowedError"
                || error.name === "SecurityError"
                || error.name === "NotReadableError"
            ) {
                break;
            }
        }
    }

    if (!stream) {
        throw lastError;
    }

    video.srcObject = stream;

    await video.play();

    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings ? track.getSettings() : {};

    const wanted =
        which === "user" || which === "environment" ? which : null;

    const gotBack =
        settings.facingMode === "environment"
        || /back|rear|environment/i.test(track.label);

    let note = `Camera: ${track.label || "default camera"}`;

    if (wanted === "environment" && !gotBack) {
        note = "No back camera found on this device, using: "
            + (track.label || "default camera");
    }

    return { note, label: track.label, deviceId: settings.deviceId };
}


// Add one entry per camera to a source <select>. Labels are only
// available after camera permission has been granted.
export async function addCameraOptions(select, activeDeviceId) {

    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        return;
    }

    const devices = await navigator.mediaDevices.enumerateDevices();

    for (const option of [...select.options]) {
        if (option.dataset.camera) {
            option.remove();
        }
    }

    const anchor = [...select.options].find(o => o.value === "file");

    devices
        .filter(d => d.kind === "videoinput" && d.deviceId)
        .forEach((device, i) => {

            const option = document.createElement("option");

            option.value = device.deviceId;
            option.dataset.camera = "1";
            option.textContent = device.label || `Camera ${i + 1}`;

            select.insertBefore(option, anchor || null);

            if (device.deviceId === activeDeviceId) {
                select.value = device.deviceId;
            }
        });
}


export function cameraErrorMessage(error) {

    const name = error && error.name;

    if (name === "NotAllowedError" || name === "SecurityError") {
        return "Camera permission was denied. Allow camera access for this "
            + "site in the browser settings, then press Start again.";
    }

    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        return "No camera was found on this device.";
    }

    if (name === "NotReadableError" || name === "TrackStartError") {
        return "The camera is in use by another app or browser tab. "
            + "Close it there, then press Start again.";
    }

    if (name === "OverconstrainedError") {
        return "That camera isn't available. Choose another one in the list.";
    }

    if (name === "InsecureContext") {
        return "Camera access needs HTTPS or localhost. Open the page via "
            + "http://127.0.0.1:8000 or the Cloudflare HTTPS address.";
    }

    return "Could not start the camera: "
        + ((error && (error.message || error.name)) || error);
}
