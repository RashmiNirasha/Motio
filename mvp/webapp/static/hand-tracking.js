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

export function makeFeatures(
    landmarks,
    handedness,
    confidenceFill
) {

    const wrist = landmarks[0];

    // Bounding box from landmarks.
    //
    // ChildPlay originally used detector bbox.
    // This is the closest available live equivalent.

    const xs = landmarks.map(p => p.x);
    const ys = landmarks.map(p => p.y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const width = Math.max(maxX - minX, 0.000001);
    const height = Math.max(maxY - minY, 0.000001);

    const scale = Math.max(width, height, 0.000001);

    const isLeft =
        handedness
            .toLowerCase()
            .includes("left");

    const features = [];

    // -----------------------------
    // 21 × [relative x,y,confidence]
    // -----------------------------

    for (let i = 0; i < 21; i++) {

        let dx = (landmarks[i].x - wrist.x) / scale;
        let dy = (landmarks[i].y - wrist.y) / scale;

        // Same canonicalization
        // used during training.
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
    // Global features
    // -----------------------------

    features.push(wrist.x);
    features.push(wrist.y);
    features.push((minX + maxX) / 2);
    features.push((minY + maxY) / 2);
    features.push(width);
    features.push(height);

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


export async function startCamera(video, facingMode) {

    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
    }

    const stream =
        await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

    video.srcObject = stream;

    await video.play();
}
