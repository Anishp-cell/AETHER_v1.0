"use client";
import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// ── COLOR THEMES ─────────────────────────────────────────────────────────────
const STATUS_COLORS: Record<string, { core: string; wire: string }> = {
  booting:   { core: "#555555", wire: "#333333" },
  idle:      { core: "#b87300", wire: "#e69900" }, // Deeper, more authentic amber/gold
  listening: { core: "#00b386", wire: "#00e6ac" },
  thinking:  { core: "#b33c00", wire: "#e64d00" },
  speaking:  { core: "#7a00cc", wire: "#9900ff" },
  muted:     { core: "#550000", wire: "#880000" },
};

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

// ── SUB-COMPONENT: Wireframe Shell ──
function WireframeShell({ radius, speedX, speedY, status, detail = 2 }: { radius: number; speedX: number; speedY: number; status: string; detail?: number }) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const materialRef = useRef<THREE.MeshBasicMaterial>(null!);
  const targetColor = useMemo(() => new THREE.Color(STATUS_COLORS[status]?.wire || "#e69900"), [status]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (meshRef.current) {
      meshRef.current.rotation.x = t * speedX;
      meshRef.current.rotation.y = t * speedY;
      
      if (status === "speaking") {
        const scale = 1.0 + Math.sin(t * 15) * 0.02;
        meshRef.current.scale.setScalar(scale);
      } else {
        meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
      }
    }
    if (materialRef.current) materialRef.current.color.lerp(targetColor, 0.05);
  });

  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[radius, detail]} />
      <meshBasicMaterial ref={materialRef} wireframe transparent opacity={0.3} blending={THREE.AdditiveBlending} />
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
export default function HologramOrb({ status, isMuted }: { status: string; isMuted?: boolean }) {
  const effectiveStatus = isMuted ? "muted" : status;
  return (
    <div className="relative flex flex-col items-center justify-center h-full w-full cursor-move">
      <Canvas camera={{ position: [0, 0, 4.5], fov: 50 }} className="!absolute inset-0 z-0">
        <ambientLight intensity={0.5} />
        
        {/* Core layers */}
        <InnerCore status={effectiveStatus} />
        
        {/* Concentric Wireframe Shells */}
        <WireframeShell radius={1.2} speedX={0.2} speedY={0.3} status={effectiveStatus} detail={3} />
        <WireframeShell radius={1.6} speedX={-0.1} speedY={0.4} status={effectiveStatus} detail={2} />
        <WireframeShell radius={2.0} speedX={0.05} speedY={-0.2} status={effectiveStatus} detail={3} />
        
        {/* Particle Cloud */}
        <DataDust status={effectiveStatus} />

        {/* Mouse Interaction */}
        <OrbitControls enableZoom={false} enablePan={false} autoRotate={true} autoRotateSpeed={0.5} />
      </Canvas>
    </div>
  );
}

