"use client";
import { useEffect, useRef, useState, useCallback } from "react";

export interface GestureData {
  enabled: boolean;
  isTracking: boolean;
  handCount: number;         // 0, 1, or 2
  rotationX: number;         // Target pitch angle
  rotationY: number;         // Target yaw angle
  velocityX: number;         // Swiping angular velocity Pitch (spin rate)
  velocityY: number;         // Swiping angular velocity Yaw (spin rate)
  zoom: number;              // Camera FOV (20 to 75)
  isFist: boolean;           // Magnetic brake gesture
  gestureType: "IDLE" | "SINGLE_HAND" | "DUAL_HAND" | "SWIPE_SPIN" | "FIST_BRAKE";
  landmarks: Array<{ x: number; y: number; z: number }> | null;
  secondLandmarks?: Array<{ x: number; y: number; z: number }> | null;
}

export function useHandGesture(enabled: boolean) {
  const [gestureData, setGestureData] = useState<GestureData>({
    enabled: false,
    isTracking: false,
    handCount: 0,
    rotationX: 0,
    rotationY: 0,
    velocityX: 0,
    velocityY: 0,
    zoom: 50,
    isFist: false,
    gestureType: "IDLE",
    landmarks: null,
    secondLandmarks: null,
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const landmarkerRef = useRef<any>(null);

  // Smooth lerp & physics tracking
  const currentRotX = useRef(0);
  const currentRotY = useRef(0);
  const currentZoom = useRef(50);
  const prevHandPos = useRef<{ x: number; y: number; t: number } | null>(null);
  const currentVelX = useRef(0);
  const currentVelY = useRef(0);

  const initLandmarker = useCallback(async () => {
    try {
      const vision = await import("@mediapipe/tasks-vision");
      const { HandLandmarker, FilesetResolver } = vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm"
      );

      const options: {
        baseOptions: {
          modelAssetPath: string;
          delegate: "GPU" | "CPU";
        };
        runningMode: "VIDEO";
        numHands: number;
      } = {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO" as const,
        numHands: 2,
      };

      try {
        landmarkerRef.current = await HandLandmarker.createFromOptions(filesetResolver, options);
      } catch (gpuErr) {
        options.baseOptions.delegate = "CPU";
        landmarkerRef.current = await HandLandmarker.createFromOptions(filesetResolver, options);
      }

      console.log("⚡ [MediaPipe WASM] Multi-Hand Landmarker (numHands: 2) initialized!");
    } catch (err) {
      console.warn("[MediaPipe WASM Warning] Could not initialize HandLandmarker:", err);
    }
  }, []);

  useEffect(() => {
    if (enabled) {
      initLandmarker();
    }
  }, [enabled, initLandmarker]);

  // Video stream & frame detection loop
  useEffect(() => {
    if (!enabled) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
      setGestureData({
        enabled: false,
        isTracking: false,
        handCount: 0,
        rotationX: 0,
        rotationY: 0,
        velocityX: 0,
        velocityY: 0,
        zoom: 50,
        isFist: false,
        gestureType: "IDLE",
        landmarks: null,
        secondLandmarks: null,
      });
      return;
    }

    const video = document.createElement("video");
    video.autoplay = true;
    video.playsInline = true;
    video.muted = true;
    videoRef.current = video;

    let lastVideoTime = -1;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        }).catch(() => navigator.mediaDevices.getUserMedia({ video: true }));

        video.srcObject = stream;
        video.onloadedmetadata = () => {
          video.play().catch(() => {});
          detectFrame();
        };
      } catch (err) {
        console.warn("[HandGesture] Camera access denied or unavailable:", err);
      }
    };

    startCamera();

    const detectFrame = () => {
      if (!enabled) return;

      if (landmarkerRef.current && video.readyState >= 2) {
        lastVideoTime = video.currentTime;
        const now = performance.now();

        try {
          const results = landmarkerRef.current.detectForVideo(video, now);

          if (results.landmarks && results.landmarks.length > 0) {
            const hands = results.landmarks;
            const handCount = hands.length;
            const primaryHand = hands[0];
            const secondaryHand = handCount > 1 ? hands[1] : null;

            // Wrist (0), Thumb tip (4), Index tip (8), Middle MCP (9), Pinky tip (20)
            const wrist = primaryHand[0];
            const thumbTip = primaryHand[4];
            const indexTip = primaryHand[8];
            const middleMcp = primaryHand[9];
            const pinkyTip = primaryHand[20];

            // 1. Fist / Brake Detection (Fingertip distances to wrist)
            const fingerTips = [primaryHand[4], primaryHand[8], primaryHand[12], primaryHand[16], primaryHand[20]];
            const avgDistToWrist = fingerTips.reduce((sum, pt) => sum + Math.hypot(pt.x - wrist.x, pt.y - wrist.y), 0) / 5;
            const isFist = avgDistToWrist < 0.22;

            // 2. Velocity / Momentum Calculation (Hand Motion Swipe Speed)
            const currX = middleMcp.x;
            const currY = middleMcp.y;
            let velX = 0;
            let velY = 0;

            if (prevHandPos.current) {
              const dt = Math.max(0.001, (now - prevHandPos.current.t) / 1000);
              const dx = currX - prevHandPos.current.x;
              const dy = currY - prevHandPos.current.y;
              velX = dx / dt; // Instantaneous horizontal hand speed
              velY = dy / dt; // Instantaneous vertical hand speed
            }
            prevHandPos.current = { x: currX, y: currY, t: now };

            // Smooth angular velocity (momentum)
            if (Math.abs(velX) > 0.4 || Math.abs(velY) > 0.4) {
              currentVelY.current = currentVelY.current * 0.7 + velX * 0.3 * 0.15; // Swiping horizontal turns globe Y
              currentVelX.current = currentVelX.current * 0.7 + velY * 0.3 * 0.15; // Swiping vertical turns globe X
            } else {
              currentVelY.current *= 0.95; // Dampen velocity over time
              currentVelX.current *= 0.95;
            }

            // 3. Rotation (Primary hand tilt / placement)
            const deltaX = (middleMcp.x - 0.5) * 2;
            const deltaY = (middleMcp.y - 0.5) * 2;
            const targetRotY = deltaX * Math.PI * 0.8;
            const targetRotX = deltaY * Math.PI * 0.8;

            // 4. Zoom Calculation (Single Hand Pinch vs Dual Hand Stretch)
            let targetZoom = currentZoom.current;
            let gestureType: GestureData["gestureType"] = isFist ? "FIST_BRAKE" : handCount > 1 ? "DUAL_HAND" : "SINGLE_HAND";

            if (handCount > 1 && secondaryHand) {
              // Dual-Hand Zoom (Distance between wrist of hand 1 and hand 2)
              const dualDist = Math.hypot(primaryHand[0].x - secondaryHand[0].x, primaryHand[0].y - secondaryHand[0].y);
              // Map distance (0.15 to 0.7) to Camera FOV (25° zoomed in to 75° zoomed out)
              targetZoom = Math.min(75, Math.max(20, 25 + (1 - Math.min(1, dualDist / 0.6)) * 50));
            } else {
              // Single Hand Pinch Zoom (Thumb tip to Index tip)
              const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y, thumbTip.z - indexTip.z);
              targetZoom = Math.min(75, Math.max(20, 30 + (1 - Math.min(1, pinchDist / 0.2)) * 40));
            }

            if (Math.abs(currentVelX.current) > 0.02 || Math.abs(currentVelY.current) > 0.02) {
              gestureType = "SWIPE_SPIN";
            }

            // Smooth Lerp
            currentRotX.current += (targetRotX - currentRotX.current) * 0.15;
            currentRotY.current += (targetRotY - currentRotY.current) * 0.15;
            currentZoom.current += (targetZoom - currentZoom.current) * 0.15;

            setGestureData({
              enabled: true,
              isTracking: true,
              handCount,
              rotationX: currentRotX.current,
              rotationY: currentRotY.current,
              velocityX: isFist ? 0 : currentVelX.current,
              velocityY: isFist ? 0 : currentVelY.current,
              zoom: currentZoom.current,
              isFist,
              gestureType,
              landmarks: primaryHand,
              secondLandmarks: secondaryHand,
            });
          } else {
            prevHandPos.current = null;
            currentVelX.current *= 0.95;
            currentVelY.current *= 0.95;
            setGestureData((prev) => ({
              ...prev,
              isTracking: false,
              handCount: 0,
              velocityX: prev.velocityX * 0.95,
              velocityY: prev.velocityY * 0.95,
              landmarks: null,
              secondLandmarks: null,
            }));
          }
        } catch (e) {
          // Detection error fallback
        }
      }

      animFrameRef.current = requestAnimationFrame(detectFrame);
    };

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (video.srcObject) {
        (video.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
      }
    };
  }, [enabled]);

  return gestureData;
}
