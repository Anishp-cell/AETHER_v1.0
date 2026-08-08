"use client";
import { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, MeshDistortMaterial } from "@react-three/drei";
import * as THREE from "three";
import { GestureData } from "../hooks/useHandGesture";

// ── COLOR THEMES ─────────────────────────────────────────────────────────────
const STATUS_COLORS: Record<string, { core: string; wire: string }> = {
  booting:   { core: "#475569", wire: "#708090" },
  idle:      { core: "#b87300", wire: "#ffaa00" }, // Premium amber gold
  listening: { core: "#009973", wire: "#00e6ac" }, // Tactical emerald cyan
  thinking:  { core: "#b33c00", wire: "#ff4d00" }, // Processing hot orange
  speaking:  { core: "#6600cc", wire: "#9900ff" }, // Neural speech purple
  muted:     { core: "#991b1b", wire: "#ef4444" }, // Secure lock crimson red
};

// ── SUB-COMPONENT: Camera Gesture Controller ──
function CameraGestureController({ gestureData }: { gestureData?: GestureData }) {
  const { camera } = useThree();

  useFrame(() => {
    if (gestureData && gestureData.enabled && gestureData.isTracking) {
      // Smoothly interpolate FOV for pinch zoom
      const targetFov = gestureData.zoom;
      if (camera instanceof THREE.PerspectiveCamera) {
        camera.fov += (targetFov - camera.fov) * 0.1;
        camera.updateProjectionMatrix();
      }
    }
  });

  return null;
}

// ── SUB-COMPONENT: Glowing Core ──
function InnerCore({ status }: { status: string }) {
  const materialRef = useRef<THREE.MeshBasicMaterial>(null!);
  const targetColor = useMemo(() => new THREE.Color(STATUS_COLORS[status]?.core || "#b87300"), [status]);

  useFrame(() => {
    if (materialRef.current) {
      materialRef.current.color.lerp(targetColor, 0.05);
    }
  });

  return (
    <mesh>
      <icosahedronGeometry args={[0.4, 3]} />
      <meshBasicMaterial ref={materialRef} transparent opacity={0.6} blending={THREE.AdditiveBlending} />
    </mesh>
  );
}

// ── UNIFIED HOLOGRAM PHYSICS GROUP ──────────────────────────────────────────
function HologramPhysicsGroup({
  status,
  energy = 0,
  gestureData,
}: {
  status: string;
  energy?: number;
  gestureData?: GestureData;
}) {
  const groupRef = useRef<THREE.Group>(null!);

  // Physics momentum state
  const velX = useRef(0);
  const velY = useRef(0);
  const angleX = useRef(0);
  const angleY = useRef(0);
  const tiltX = useRef(0);
  const tiltY = useRef(0);

  useFrame((_, delta) => {
    if (!groupRef.current) return;

    if (gestureData && gestureData.enabled) {
      if (gestureData.isFist) {
        // Fist Magnetic Brake: halt rotation instantly
        velX.current = 0;
        velY.current = 0;
      } else {
        // Impart swiping velocity momentum
        if (Math.abs(gestureData.velocityX) > 0.001) {
          velY.current += gestureData.velocityX * 0.08;
        }
        if (Math.abs(gestureData.velocityY) > 0.001) {
          velX.current += gestureData.velocityY * 0.08;
        }
      }

      // Smooth hand position tracking / tilt
      if (gestureData.isTracking) {
        tiltX.current += (gestureData.rotationX * 0.8 - tiltX.current) * 0.1;
        tiltY.current += (gestureData.rotationY * 0.8 - tiltY.current) * 0.1;
      } else {
        // When hand leaves frame, smoothly return tilt to neutral while keeping continuous momentum spin
        tiltX.current *= 0.95;
        tiltY.current *= 0.95;
      }

      // Realistic aerodynamic rotational friction (slow coast down over ~12 seconds)
      velX.current *= 0.988;
      velY.current *= 0.988;

      // Base idle orbit + velocity momentum
      angleX.current += velX.current + 0.001;
      angleY.current += velY.current + 0.003;

      groupRef.current.rotation.x = angleX.current + tiltX.current;
      groupRef.current.rotation.y = angleY.current + tiltY.current;
    } else {
      // Natural idle rotation
      groupRef.current.rotation.x += 0.002;
      groupRef.current.rotation.y += 0.005;
    }
  });

  return (
    <group ref={groupRef}>
      <InnerCore status={status} />
      <WireframeShell radius={1.2} status={status} energy={energy} />
      <WireframeShell radius={1.6} status={status} energy={energy} />
      <WireframeShell radius={2.0} status={status} energy={energy} />
      <DataDust status={status} />
    </group>
  );
}

// ── SUB-COMPONENT: Wireframe Fluid Shell ──
function WireframeShell({
  radius,
  status,
  energy = 0,
}: {
  radius: number;
  status: string;
  energy?: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const targetColor = useMemo(() => new THREE.Color(STATUS_COLORS[status]?.wire || "#e69900"), [status]);

  const normalizedEnergy = Math.min(1.0, energy / 1500);
  const distortAmount = status === "listening" ? 0.15 + normalizedEnergy * 0.4 : status === "speaking" ? 0.35 : 0.1;
  const distortSpeed = status === "speaking" ? 6 : status === "listening" ? 2 + normalizedEnergy * 4 : 1;

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[radius, 32, 32]} />
      <MeshDistortMaterial
        color={targetColor}
        wireframe
        transparent
        opacity={0.3}
        blending={THREE.AdditiveBlending}
        distort={distortAmount}
        speed={distortSpeed}
      />
    </mesh>
  );
}

// ── SUB-COMPONENT: Data Particles ──
function DataDust({ status }: { status: string }) {
  const pointsRef = useRef<THREE.Points>(null!);
  const materialRef = useRef<THREE.PointsMaterial>(null!);
  const targetColor = useMemo(() => new THREE.Color(STATUS_COLORS[status]?.core || "#b87300"), [status]);
  const count = 300;

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 2.0 + Math.random() * 2.0;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      arr[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      arr[i * 3 + 2] = r * Math.cos(phi);
    }
    return arr;
  }, []);

  useFrame(({ clock }) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = clock.getElapsedTime() * 0.03;
      pointsRef.current.rotation.z = clock.getElapsedTime() * 0.01;
    }
    if (materialRef.current) materialRef.current.color.lerp(targetColor, 0.05);
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial ref={materialRef} size={0.015} transparent opacity={0.4} blending={THREE.AdditiveBlending} />
    </points>
  );
}

// ── MAIN EXPORT ─────────────────────────────────────────────────────────────
export default function HologramOrb({
  status,
  energy = 0,
  isMuted,
  gestureData,
}: {
  status: string;
  energy?: number;
  isMuted?: boolean;
  gestureData?: GestureData;
}) {
  const effectiveStatus = isMuted ? "muted" : status;
  const isGestureActive = Boolean(gestureData?.enabled && gestureData?.isTracking);

  return (
    <div className="relative flex flex-col items-center justify-center h-full w-full cursor-move">
      {/* Gesture HUD Status Indicator */}
      {gestureData?.enabled && (
        <div className="absolute top-4 left-4 z-20 flex items-center gap-2 px-3 py-1.5 bg-black/60 border border-cyan-500/40 rounded-lg backdrop-blur-md">
          <div className={`w-2 h-2 rounded-full ${isGestureActive ? "bg-cyan-400 animate-ping" : "bg-yellow-400"}`} />
          <span className="text-[10px] font-mono text-cyan-300 uppercase tracking-wider">
            {isGestureActive ? "🖐️ GESTURE TRACKING ACTIVE" : "🖐️ WAVE HAND TO CONTROL ORB"}
          </span>
        </div>
      )}

      <Canvas camera={{ position: [0, 0, 5], fov: 50 }} className="!absolute inset-0 z-0">
        <ambientLight intensity={0.5} />

        <CameraGestureController gestureData={gestureData} />

        {/* Unified Physics Momentum Group */}
        <HologramPhysicsGroup status={effectiveStatus} energy={energy} gestureData={gestureData} />

        {/* Orbit Controls (autoRotate disabled when gesture enabled so physics are never overridden) */}
        <OrbitControls enableZoom={false} enablePan={false} autoRotate={!gestureData?.enabled} autoRotateSpeed={0.5} />
      </Canvas>
    </div>
  );
}
