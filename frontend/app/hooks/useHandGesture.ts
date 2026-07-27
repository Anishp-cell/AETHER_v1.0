"use client";
import { useEffect, useRef, useState, useCallback } from "react";

export interface GestureData {
  enabled: boolean;
  isTracking: boolean;
  rotationX: number; // Pitch delta
  rotationY: number; // Yaw delta
  zoom: number;      // FOV (30 to 65)
  landmarks: Array<{ x: number; y: number; z: number }> | null;
}

export function useHandGesture(enabled: boolean) {
  const [gestureData, setGestureData] = useState<GestureData>({
    enabled: false,
    isTracking: false,
    rotationX: 0,
    rotationY: 0,
    zoom: 50,
    landmarks: null,
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const landmarkerRef = useRef<any>(null);

  // Smooth lerp state
  const currentRotX = useRef(0);
  const currentRotY = useRef(0);
  const currentZoom = useRef(50);

  const initLandmarker = useCallback(async () => {
    try {
      const vision = await import("@mediapipe/tasks-vision");
      const { HandLandmarker, FilesetResolver } = vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm"
      );

      landmarkerRef.current = await HandLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });

      console.log("⚡ [MediaPipe WASM] HandLandmarker initialized successfully!");
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
        rotationX: 0,
        rotationY: 0,
        zoom: 50,
        landmarks: null,
      });
      return;
    }

    const video = document.createElement("video");
    video.autoplay = true;
    video.playsInline = true;
    video.muted = true;
    videoRef.current = video;

    let lastVideoTime = -1;

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          video.play();
          detectFrame();
        };
      })
      .catch((err) => {
        console.warn("[HandGesture] Camera access denied or unavailable:", err);
      });

    const detectFrame = () => {
      if (!enabled) return;

      if (video.currentTime !== lastVideoTime && landmarkerRef.current && video.readyState >= 2) {
        lastVideoTime = video.currentTime;
        try {
          const results = landmarkerRef.current.detectForVideo(video, performance.now());

          if (results.landmarks && results.landmarks.length > 0) {
            const hand = results.landmarks[0]; // 21 3D points

            // Wrist (0), Index tip (8), Thumb tip (4), Middle MCP (9)
            const wrist = hand[0];
            const thumbTip = hand[4];
            const indexTip = hand[8];
            const middleMcp = hand[9];

            // 1. Rotation (Hand position relative to frame center: 0.5, 0.5)
            const deltaX = (middleMcp.x - 0.5) * 2; // -1 to 1
            const deltaY = (middleMcp.y - 0.5) * 2; // -1 to 1

            const targetRotY = deltaX * Math.PI * 0.8;
            const targetRotX = deltaY * Math.PI * 0.8;

            // 2. Pinch Zoom (Distance between thumb tip and index tip)
            const pinchDist = Math.hypot(
              thumbTip.x - indexTip.x,
              thumbTip.y - indexTip.y,
              thumbTip.z - indexTip.z
            );

            // Map pinch distance (0.02 - 0.25) to Camera FOV (30° zoomed in to 70° zoomed out)
            const targetZoom = Math.min(70, Math.max(25, 30 + (1 - Math.min(1, pinchDist / 0.2)) * 40));

            // Smooth Lerp (α = 0.15)
            currentRotX.current += (targetRotX - currentRotX.current) * 0.15;
            currentRotY.current += (targetRotY - currentRotY.current) * 0.15;
            currentZoom.current += (targetZoom - currentZoom.current) * 0.15;

            setGestureData({
              enabled: true,
              isTracking: true,
              rotationX: currentRotX.current,
              rotationY: currentRotY.current,
              zoom: currentZoom.current,
              landmarks: hand,
            });
          } else {
            setGestureData((prev) => ({ ...prev, isTracking: false, landmarks: null }));
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
