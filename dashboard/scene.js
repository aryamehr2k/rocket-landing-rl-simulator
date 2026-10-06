// The 3D view of a live flight: vehicle, thrust plume, trail, pad, landing circle, target height and wind.
// Simulator frame: x and y along the ground, z up. One group rotated -90 degrees about x maps it into
// three.js (y up), so positions and attitude quaternions from the server are used unchanged inside it.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { cssVar } from './dom.js';

const MAX_TRAIL_POINTS = 20000;
const TRAIL_STEP_M = 0.03;
const PLUME_AT_FULL_THRUST_M = 0.9;
const PLUME_MIN_M = 0.005;
const WIND_ARROW_M_PER_MPS = 0.3;
const WIND_ARROW_MIN_MPS = 0.1;
const WIND_ARROW_HEIGHT_M = 0.9;  // above the centre of gravity, clear of the nose
const LOCATOR_SIZE_PX = 9;
const FOLLOW_OFFSET = new THREE.Vector3(2.4, 1.0, 2.8);  // three.js frame, from the vehicle
const OVERVIEW_MIN_HEIGHT_M = 10;
const GROUND_SIZE_M = 400;
const GRID_SIZE_M = 100;
const GRID_CELLS = 20;
const PAD_RADIUS_M = 0.6;
const RING_HALF_WIDTH_M = 0.06;
const TARGET_DISC_RADIUS_M = 3;
const CIRCLE_SEGMENTS = 128;
const Z_AXIS = new THREE.Vector3(0, 0, 1);
const UP = new THREE.Vector3(0, 1, 0);
const DEG = Math.PI / 180;

// Thrust direction in the body frame for gimbal deflections (pitch, yaw) in radians, unit length.
function thrustDirection(pitch, yaw) {
  return new THREE.Vector3(-Math.tan(pitch), -Math.tan(yaw), 1).normalize();
}

// Cylinders and cones are built along +y; turn them onto the body z axis (tail to nose) and centre
// them at body z = cg - station (stations are measured from the nose tip).
function alongBody(geometry, centreStation, v) {
  return geometry.rotateX(Math.PI / 2).translate(0, 0, v.cg - centreStation);
}

function strut(from, to, radius, material) {
  const start = new THREE.Vector3(...from), end = new THREE.Vector3(...to);
  const direction = end.clone().sub(start);
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, direction.length(), 6), material);
  mesh.quaternion.setFromUnitVectors(UP, direction.clone().normalize());
  mesh.position.copy(start).addScaledVector(direction, 0.5);
  return mesh;
}

function circleLine(radius, z, material) {
  const points = [];
  for (let i = 0; i <= CIRCLE_SEGMENTS; i++) {
    const a = 2 * Math.PI * i / CIRCLE_SEGMENTS;
    points.push(new THREE.Vector3(radius * Math.cos(a), radius * Math.sin(a), z));
  }
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material);
}

function translucent(color, opacity) {
  return new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false });
}

export class FlightScene {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);
    this.scene.add(new THREE.HemisphereLight(0xeef3ff, 0x404040, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(30, 60, 20);
    this.scene.add(sun);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 3000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.follow = true;
    this.mission = null;
    this.vehicleInfo = null;
    this.vehicle = new THREE.Group();
    this.world.add(this.vehicle);
    this.locator = this.buildLocator();
    this.world.add(this.locator);
    this.markers = new THREE.Group();
    this.world.add(this.markers);
    this.buildTrail();
    this.windArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0xeb6834, 0.18, 0.1);
    this.windArrow.visible = false;
    this.world.add(this.windArrow);
    this.world.add(new THREE.AxesHelper(1));  // red +x, green +y, blue +z (up), at the pad
    this.applyTheme();
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.applyTheme());
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.placeCamera();
    this.renderer.setAnimationLoop(() => this.render());
  }

  applyTheme() {
    this.scene.background = new THREE.Color(cssVar('--scene-bg'));
    if (this.ground) this.world.remove(this.ground, this.grid);
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(GROUND_SIZE_M, GROUND_SIZE_M),
                                 new THREE.MeshStandardMaterial({ color: cssVar('--scene-ground'), roughness: 1 }));
    this.grid = new THREE.GridHelper(GRID_SIZE_M, GRID_CELLS, cssVar('--scene-grid'), cssVar('--scene-grid'));
    this.grid.rotation.x = Math.PI / 2;  // GridHelper lies in its local XZ plane; move it to the ground plane XY
    this.grid.position.z = 0.003;
    this.world.add(this.ground, this.grid);
  }

  // A dot of fixed screen size on the vehicle, so it stays visible from far away in the overview.
  buildLocator() {
    const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    const locator = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0xeb6834, size: LOCATOR_SIZE_PX, sizeAttenuation: false }));
    locator.visible = false;
    return locator;
  }

  buildTrail() {
    this.trailPositions = new Float32Array(MAX_TRAIL_POINTS * 3);
    this.trailCount = 0;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.trailPositions, 3));
    geometry.setDrawRange(0, 0);
    this.trail = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0x2a78d6 }));
    this.trail.frustumCulled = false;
    this.world.add(this.trail);
  }

  // A new flight: vehicle geometry and mission markers from the server's setup event.
  setup(info) {
    this.vehicleInfo = info.vehicle;
    this.mission = info.mission;
    this.buildVehicle(info.vehicle);
    this.buildMarkers(info.mission, info.vehicle);
    this.trailCount = 0;
    this.trail.geometry.setDrawRange(0, 0);
    this.vehicle.position.set(0, 0, info.vehicle.pad_cg_height);
    this.vehicle.quaternion.identity();
    this.locator.position.copy(this.vehicle.position);
    this.windArrow.visible = false;
    this.placeCamera();
  }

  buildVehicle(v) {
    this.vehicle.traverse((o) => o.geometry?.dispose());
    this.vehicle.clear();
    const body = new THREE.MeshStandardMaterial({ color: 0xe8ebef, roughness: 0.55 });
    const nose = new THREE.MeshStandardMaterial({ color: 0xeb6834, roughness: 0.5 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x3a414c, roughness: 0.7 });
    const r = v.diameter / 2;
    const noseLength = Math.min(2.5 * v.diameter, v.length / 3);
    this.vehicle.add(new THREE.Mesh(alongBody(new THREE.ConeGeometry(r, noseLength, 32), noseLength / 2, v), nose));
    this.vehicle.add(new THREE.Mesh(alongBody(new THREE.CylinderGeometry(r, r, v.length - noseLength, 32),
                                              (noseLength + v.length) / 2, v), body));
    const hinge = v.cg - 0.85 * v.length;
    const footZ = v.cg - v.length - v.leg_height;
    for (let k = 0; k < v.leg_count; k++) {
      const a = 2 * Math.PI * k / v.leg_count + Math.PI / 4;
      const foot = [v.leg_span / 2 * Math.cos(a), v.leg_span / 2 * Math.sin(a), footZ];
      this.vehicle.add(strut([r * Math.cos(a), r * Math.sin(a), hinge], foot, 0.006, dark));
      const pad = new THREE.Mesh(new THREE.SphereGeometry(0.014, 10, 8), dark);
      pad.position.set(...foot);
      this.vehicle.add(pad);
    }
    this.gimbal = new THREE.Group();  // at the pivot; local +z is the thrust direction, the exhaust goes to -z
    this.gimbal.position.z = v.cg - v.pivot;
    const nozzleLength = 0.5 * v.diameter;
    this.gimbal.add(new THREE.Mesh(new THREE.CylinderGeometry(0.3 * v.diameter, 0.42 * v.diameter, nozzleLength, 20)
      .rotateX(Math.PI / 2).translate(0, 0, -nozzleLength / 2), dark));
    this.plume = new THREE.Mesh(new THREE.ConeGeometry(0.36 * v.diameter, 1, 20, 1, true)
      .rotateX(-Math.PI / 2).translate(0, 0, -0.5), translucent(0xffa040, 0.8));
    this.plume.visible = false;
    this.gimbal.add(this.plume);
    this.vehicle.add(this.gimbal);
  }

  buildMarkers(mission, v) {
    this.markers.traverse((o) => o.geometry?.dispose());
    this.markers.clear();
    const aqua = cssVar('--series-3');
    const pad = new THREE.Mesh(new THREE.CircleGeometry(PAD_RADIUS_M, 48), new THREE.MeshStandardMaterial({ color: 0x8a9099, roughness: 0.9 }));
    pad.position.z = 0.006;
    const r = mission.landing_radius;
    const ring = new THREE.Mesh(new THREE.RingGeometry(r - RING_HALF_WIDTH_M, r + RING_HALF_WIDTH_M, CIRCLE_SEGMENTS),
                                translucent(aqua, 0.9));
    ring.position.z = 0.01;
    const target = mission.target_altitude, tolerance = mission.altitude_tolerance;
    const disc = new THREE.Mesh(new THREE.CircleGeometry(TARGET_DISC_RADIUS_M, 64), translucent(aqua, 0.16));
    disc.position.z = target;
    const band = new THREE.Mesh(new THREE.CylinderGeometry(TARGET_DISC_RADIUS_M, TARGET_DISC_RADIUS_M, 2 * tolerance, 64, 1, true)
      .rotateX(Math.PI / 2), translucent(aqua, 0.07));
    band.position.z = target;
    const edge = new THREE.LineBasicMaterial({ color: aqua, transparent: true, opacity: 0.7 });
    this.markers.add(pad, ring, disc, band, circleLine(TARGET_DISC_RADIUS_M, target - tolerance, edge),
                     circleLine(TARGET_DISC_RADIUS_M, target + tolerance, edge));
    this.overviewHeight = Math.max(target, OVERVIEW_MIN_HEIGHT_M) + v.length;
  }

  update(frame) {
    const [x, y, z] = frame.position;
    const [qw, qx, qy, qz] = frame.quaternion;
    const before = this.vehicleWorldPosition();
    this.vehicle.position.set(x, y, z);
    this.vehicle.quaternion.set(qx, qy, qz, qw).normalize();
    if (this.gimbal) {
      const [pitch, yaw] = frame.gimbal_deg;
      this.gimbal.quaternion.setFromUnitVectors(Z_AXIS, thrustDirection(pitch * DEG, yaw * DEG));
      const length = PLUME_AT_FULL_THRUST_M * Math.max(0, frame.thrust) / (this.vehicleInfo?.max_thrust || 1);
      this.plume.visible = length > PLUME_MIN_M;
      this.plume.scale.z = Math.max(length, PLUME_MIN_M);
    }
    this.locator.position.set(x, y, z);
    this.addTrailPoint(x, y, z);
    this.showWind(frame.wind, x, y, z);
    if (this.follow) {  // move camera and target together so the chosen view angle is kept
      const after = this.vehicleWorldPosition();
      this.camera.position.add(after.clone().sub(before));
      this.controls.target.add(after.sub(before));
    }
  }

  addTrailPoint(x, y, z) {
    const n = this.trailCount;
    if (n >= MAX_TRAIL_POINTS) return;
    if (n > 0) {
      const p = this.trailPositions;
      if (Math.hypot(x - p[3 * n - 3], y - p[3 * n - 2], z - p[3 * n - 1]) < TRAIL_STEP_M) return;
    }
    this.trailPositions.set([x, y, z], 3 * n);
    this.trailCount = n + 1;
    this.trail.geometry.attributes.position.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, this.trailCount);
  }

  showWind(wind, x, y, z) {
    const speed = Math.hypot(wind[0], wind[1]);
    this.windArrow.visible = speed >= WIND_ARROW_MIN_MPS;
    if (!this.windArrow.visible) return;
    this.windArrow.position.set(x, y, z + WIND_ARROW_HEIGHT_M);
    this.windArrow.setDirection(new THREE.Vector3(wind[0] / speed, wind[1] / speed, 0));
    this.windArrow.setLength(speed * WIND_ARROW_M_PER_MPS, 0.18, 0.1);
  }

  vehicleWorldPosition() {
    return this.vehicle.getWorldPosition(new THREE.Vector3());
  }

  setFollow(on) {
    this.follow = on;
    this.locator.visible = !on;
    this.placeCamera();
  }

  placeCamera() {
    if (this.follow) {
      const at = this.vehicleWorldPosition();
      this.controls.target.copy(at);
      this.camera.position.copy(at).add(FOLLOW_OFFSET);
    } else {
      const h = this.overviewHeight || OVERVIEW_MIN_HEIGHT_M;
      this.controls.target.set(0, 0.45 * h, 0);
      this.camera.position.set(1.1 * h, 0.75 * h, 1.3 * h);
    }
    this.controls.update();
  }

  resize() {
    const width = this.container.clientWidth, height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  render() {
    if (!this.container.offsetParent) return;  // the Fly tab is hidden
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
