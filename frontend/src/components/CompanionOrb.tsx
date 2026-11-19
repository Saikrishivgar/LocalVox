"use client";

import React, { Component, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { STATE_THEMES } from "@/lib/stateTheme";
import type { PipelineState } from "@/lib/types";

interface CompanionOrbProps {
  pipelineState: PipelineState;
  rms: number;
  bars?: number[];
  onClick?: () => void;
}

class OrbErrorBoundary extends Component<
  { fallback: React.ReactNode; children: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

const ORIGIN = new THREE.Vector3(0, 0, 0);

function bandEnergy(bars: number[], start: number, end: number): number {
  if (!bars.length) return 0;
  let sum = 0;
  let count = 0;
  const hi = Math.min(bars.length, end);
  for (let i = Math.max(0, start); i < hi; i += 1) {
    sum += bars[i] || 0;
    count += 1;
  }
  return count > 0 ? sum / count : 0;
}

function OrbMesh({
  pipelineState,
  rms,
  bars,
}: {
  pipelineState: PipelineState;
  rms: number;
  bars: number[];
}) {
  const groupRef = useRef<THREE.Group>(null);
  const coreRef = useRef<THREE.Mesh>(null);
  const cageRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const ring2Ref = useRef<THREE.Mesh>(null);
  const particleRef = useRef<THREE.Points>(null);

  const smoothRms = useRef(0);
  const smoothLow = useRef(0);
  const smoothMid = useRef(0);
  const smoothHigh = useRef(0);
  const basePositionsRef = useRef<Float32Array | null>(null);
  const wasDeformedRef = useRef(false);

  const theme = STATE_THEMES[pipelineState];
  const primaryColor = useMemo(
    () => new THREE.Color(theme.accentHex),
    [theme.accentHex]
  );
  const secondaryColor = useMemo(
    () => new THREE.Color(theme.secondaryHex),
    [theme.secondaryHex]
  );

  const particlePositions = useMemo(() => {
    const count = 280;
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      const phi = Math.acos(1 - (2 * (i + 0.5)) / count);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      const radius = 1.66 + (i % 6) * 0.045;
      arr[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      arr[i * 3 + 2] = radius * Math.cos(phi);
    }
    return arr;
  }, []);

  useFrame((state, delta) => {
    const clampedDelta = Math.min(0.05, delta);
    const targetRms = Math.min(1, Math.max(0, rms * 4.8));
    const targetLow = Math.min(1, bandEnergy(bars, 0, 8));
    const targetMid = Math.min(1, bandEnergy(bars, 8, 16));
    const targetHigh = Math.min(1, bandEnergy(bars, 16, 24));

    const lerpRate = Math.min(1, clampedDelta * 16);
    smoothRms.current = THREE.MathUtils.lerp(
      smoothRms.current,
      targetRms,
      lerpRate
    );
    smoothLow.current = THREE.MathUtils.lerp(
      smoothLow.current,
      targetLow,
      lerpRate
    );
    smoothMid.current = THREE.MathUtils.lerp(
      smoothMid.current,
      targetMid,
      lerpRate
    );
    smoothHigh.current = THREE.MathUtils.lerp(
      smoothHigh.current,
      targetHigh,
      lerpRate
    );

    const activeAmp = smoothRms.current;
    const low = smoothLow.current;
    const mid = smoothMid.current;
    const high = smoothHigh.current;

    const t = state.clock.elapsedTime;
    const isSleeping = pipelineState === "SLEEPING";
    const isWakeDetected = pipelineState === "WAKE_DETECTED";
    const isThinking =
      pipelineState === "PROCESSING" || pipelineState === "GENERATING";
    const isSpeaking = pipelineState === "SPEAKING";
    const isListening = pipelineState === "LISTENING";
    const isVoiceReactive =
      !isSleeping && (isSpeaking || isListening || activeAmp > 0.015);

    const spinSpeed = isWakeDetected
      ? 1.55
      : isThinking
        ? 1.05
        : isSpeaking
          ? 0.65 + activeAmp * 0.75
          : isListening
            ? 0.4 + activeAmp * 0.4
            : isSleeping
              ? 0.08
              : 0.14;

    const pulseScale = isWakeDetected
      ? 1.14 + Math.sin(t * 14.0) * 0.055
      : isSleeping
        ? 0.96 + Math.sin(t * 0.9) * 0.014
        : 1.0 +
          Math.sin(t * (isThinking ? 3.4 : 1.6)) * 0.018 +
          (isVoiceReactive ? activeAmp * 0.14 : 0);

    // Subtle physical vibration proportional strictly to live audio energy (disabled in standby).
    if (groupRef.current) {
      if (!isSleeping && activeAmp > 0.025) {
        const vib = activeAmp * 0.028;
        groupRef.current.position.x = Math.sin(t * 48.0) * vib;
        groupRef.current.position.y = Math.cos(t * 53.0) * vib * 0.85;
        groupRef.current.position.z = Math.sin(t * 39.0) * vib * 0.5;
      } else {
        groupRef.current.position.lerp(ORIGIN, 0.2);
      }
    }

    if (coreRef.current) {
      const geom = coreRef.current.geometry;
      const posAttr = geom.getAttribute("position") as THREE.BufferAttribute;
      if (posAttr) {
        if (!basePositionsRef.current) {
          basePositionsRef.current = new Float32Array(
            posAttr.array as Float32Array
          );
        }
        const base = basePositionsRef.current;
        const effectiveAmp = isSleeping ? 0 : activeAmp;
        if (effectiveAmp > 0.012 || wasDeformedRef.current) {
          const drive = Math.min(1, effectiveAmp * 1.45);
          for (let i = 0; i < posAttr.count; i += 1) {
            const bx = base[i * 3];
            const by = base[i * 3 + 1];
            const bz = base[i * 3 + 2];
            if (drive > 0.012) {
              const w1 =
                Math.sin(bx * 4.5 + t * 14.0) *
                Math.cos(by * 4.5 - t * 11.0) *
                (0.06 + low * 0.14);
              const w2 =
                Math.sin(bz * 7.5 + t * 23.0) * (0.04 + (mid + high) * 0.09);
              const disp = 1.0 + (w1 + w2) * drive;
              posAttr.setXYZ(i, bx * disp, by * disp, bz * disp);
            } else {
              posAttr.setXYZ(i, bx, by, bz);
            }
          }
          posAttr.needsUpdate = true;
          geom.computeVertexNormals();
          wasDeformedRef.current = drive > 0.012;
        }
      }

      coreRef.current.rotation.y += clampedDelta * spinSpeed;
      coreRef.current.rotation.x = Math.sin(t * 0.5) * 0.24;
      coreRef.current.scale.setScalar(pulseScale);
      const mat = coreRef.current.material as THREE.MeshPhysicalMaterial;
      mat.color.lerp(primaryColor, Math.min(1, clampedDelta * 6));
      mat.emissive.lerp(secondaryColor, Math.min(1, clampedDelta * 6));
      mat.emissiveIntensity = isWakeDetected
        ? 1.45
        : isSleeping
          ? 0.36
          : 0.58 + activeAmp * 1.15 + (isThinking ? 0.38 : 0);
    }

    if (cageRef.current) {
      cageRef.current.rotation.y -= clampedDelta * spinSpeed * 0.7;
      cageRef.current.rotation.z += clampedDelta * (0.2 + activeAmp * 0.45);
      cageRef.current.scale.setScalar(
        pulseScale * (isWakeDetected ? 1.28 : 1.16 + mid * 0.14 + activeAmp * 0.08)
      );
      const cageMat = cageRef.current.material as THREE.MeshBasicMaterial;
      cageMat.color.lerp(primaryColor, Math.min(1, clampedDelta * 6));
      cageMat.opacity = isWakeDetected
        ? 0.52
        : isSleeping
          ? 0.14
          : 0.22 + activeAmp * 0.28;
    }

    if (ringRef.current) {
      ringRef.current.rotation.z +=
        clampedDelta * (isThinking ? 1.05 : 0.28 + activeAmp * 0.8);
      ringRef.current.rotation.x =
        Math.PI / 2.4 + Math.sin(t * 0.8) * (0.16 + activeAmp * 0.14);
      ringRef.current.scale.setScalar(1.06 + low * 0.14 + activeAmp * 0.16);
      const ringMat = ringRef.current.material as THREE.MeshBasicMaterial;
      ringMat.color.lerp(primaryColor, Math.min(1, clampedDelta * 6));
      ringMat.opacity = 0.38 + activeAmp * 0.42;
    }

    if (ring2Ref.current) {
      ring2Ref.current.rotation.z -=
        clampedDelta * (isThinking ? 0.85 : 0.22 + activeAmp * 0.65);
      ring2Ref.current.rotation.y =
        Math.PI / 3.2 + Math.cos(t * 0.65) * (0.14 + activeAmp * 0.16);
      ring2Ref.current.scale.setScalar(1.12 + high * 0.16 + activeAmp * 0.14);
      const ring2Mat = ring2Ref.current.material as THREE.MeshBasicMaterial;
      ring2Mat.color.lerp(primaryColor, Math.min(1, clampedDelta * 6));
      ring2Mat.opacity = 0.22 + activeAmp * 0.36;
    }

    if (particleRef.current) {
      particleRef.current.rotation.y +=
        clampedDelta * (0.14 + spinSpeed * 0.32);
      particleRef.current.scale.setScalar(1 + activeAmp * 0.18);
      const ptsMat = particleRef.current.material as THREE.PointsMaterial;
      ptsMat.color.lerp(primaryColor, Math.min(1, clampedDelta * 6));
      ptsMat.opacity = 0.5 + activeAmp * 0.4;
    }
  });

  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.48} />
      <pointLight position={[3, 4, 4]} intensity={2.4} color="#ffffff" />
      <pointLight
        position={[-3, -2, -3]}
        intensity={1.7}
        color={theme.accentHex}
      />

      {/* Core audio-reactive crystalline sphere */}
      <mesh ref={coreRef}>
        <icosahedronGeometry args={[1.08, 4]} />
        <meshPhysicalMaterial
          color={theme.accentHex}
          emissive={theme.secondaryHex}
          emissiveIntensity={0.68}
          roughness={0.2}
          metalness={0.3}
          clearcoat={0.95}
          clearcoatRoughness={0.1}
        />
      </mesh>

      {/* Precision geodesic energy cage */}
      <mesh ref={cageRef}>
        <icosahedronGeometry args={[1.1, 2]} />
        <meshBasicMaterial
          color={theme.accentHex}
          wireframe
          transparent
          opacity={0.24}
        />
      </mesh>

      {/* Primary orbital harmonic ring */}
      <mesh ref={ringRef}>
        <torusGeometry args={[1.48, 0.014, 16, 96]} />
        <meshBasicMaterial
          color={theme.accentHex}
          transparent
          opacity={0.42}
        />
      </mesh>

      {/* Secondary orthogonal acoustic ring */}
      <mesh ref={ring2Ref}>
        <torusGeometry args={[1.56, 0.009, 16, 96]} />
        <meshBasicMaterial
          color={theme.accentHex}
          transparent
          opacity={0.24}
        />
      </mesh>

      {/* Fibonacci particle halo */}
      <points ref={particleRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[particlePositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.026}
          color={theme.accentHex}
          transparent
          opacity={0.55}
          sizeAttenuation
        />
      </points>
    </group>
  );
}

function FallbackOrb({
  pipelineState,
  rms,
}: {
  pipelineState: PipelineState;
  rms: number;
}) {
  const theme = STATE_THEMES[pipelineState];
  const scale = 1 + Math.min(0.22, rms * 0.6);
  return (
    <div className="relative flex h-full w-full items-center justify-center">
      <div
        className="h-56 w-56 rounded-full transition-transform duration-100"
        style={{
          transform: `scale(${scale})`,
          background: `radial-gradient(circle at 32% 30%, #ffffff55, ${theme.accentHex} 45%, ${theme.secondaryHex} 90%)`,
          boxShadow: `0 0 80px ${theme.glowRgba}`,
        }}
      />
    </div>
  );
}

export function CompanionOrb({
  pipelineState,
  rms,
  bars = [],
  onClick,
}: CompanionOrbProps) {
  const [mounted, setMounted] = useState(false);
  const [webglReady, setWebglReady] = useState(false);
  const theme = STATE_THEMES[pipelineState];

  useEffect(() => {
    setMounted(true);
    try {
      const canvas = document.createElement("canvas");
      const gl =
        canvas.getContext("webgl2") || canvas.getContext("webgl");
      setWebglReady(Boolean(gl));
    } catch {
      setWebglReady(false);
    }
  }, []);

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`AI companion orb, state ${theme.label}`}
      className="group relative flex h-64 w-64 cursor-pointer items-center justify-center rounded-full focus:outline-none sm:h-72 sm:w-72 lg:h-[310px] lg:w-[310px] xl:h-[336px] xl:w-[336px]"
    >
      {/* Reactive ambient radial aura */}
      <div
        className="pointer-events-none absolute inset-4 rounded-full blur-3xl transition-all duration-300"
        style={{
          background: theme.glowRgba,
          transform: `scale(${1 + Math.min(0.24, rms * 0.85)})`,
        }}
      />

      <div className="relative h-full w-full">
        {mounted && webglReady ? (
          <OrbErrorBoundary
            fallback={<FallbackOrb pipelineState={pipelineState} rms={rms} />}
          >
            <Canvas
              camera={{ position: [0, 0, 3.45], fov: 44 }}
              dpr={[1, 2]}
              gl={{ alpha: true, antialias: true }}
            >
              <OrbMesh pipelineState={pipelineState} rms={rms} bars={bars} />
            </Canvas>
          </OrbErrorBoundary>
        ) : (
          <FallbackOrb pipelineState={pipelineState} rms={rms} />
        )}
      </div>
    </button>
  );
}
