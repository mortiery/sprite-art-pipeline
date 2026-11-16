/**
 * rpgchar.js - load characters produced by `python -m pipeline.character` into Phaser 3/4.
 *
 * A character folder contains  <name>.json (manifest), <name>_walk.png (3x4 frames of 48x48,
 * rows down/left/right/up, RPG Maker MV layout), <name>_walk8.png (smooth walk: 9 columns x 4 rows,
 * column 0 = standing, columns 1-8 = one gait cycle) and <name>_faces.png (4x2 faces of 144x144).
 *
 *   import { fetchManifests, preloadCharacter, createAnimations, addCharacter, faceFrame } from './rpgchar.js';
 *   const manifests = await fetchManifests(['../characters/rin/rin.json']);     // before new Phaser.Game(...)
 *   preload() { manifests.forEach(m => preloadCharacter(this, m)); }
 *   create()  { manifests.forEach(m => createAnimations(this, m));
 *               const hero = addCharacter(this, manifests[0], 100, 100);
 *               hero.walk('left'); hero.stop(); }
 */

/** Fetch manifests and remember the folder each one came from (image paths are relative to it). */
export async function fetchManifests(urls) {
  return Promise.all(urls.map(async (url) => {
    const m = await (await fetch(url)).json();
    m.baseUrl = url.slice(0, url.lastIndexOf('/') + 1);
    return m;
  }));
}

/**
 * Which walk sheet to use: the smooth one (`walkAnim`: column 0 = standing, then N frames of one gait
 * cycle) when the manifest has it, else the 3-frame RPG Maker sheet (`walk`). Pass smooth = false to
 * force the RPG Maker sheet.
 */
function walkSet(m, smooth = true) {
  return (smooth && m.walkAnim) ? { set: m.walkAnim, key: `${m.name}-anim`, tag: 'anim' }
                                : { set: m.walk, key: `${m.name}-walk`, tag: 'walk' };
}

/** Call from Scene.preload(). Texture keys: `${name}-walk`, `${name}-anim`, `${name}-faces`. */
export function preloadCharacter(scene, m) {
  for (const [key, set] of [[`${m.name}-walk`, m.walk], [`${m.name}-anim`, m.walkAnim]]) {
    if (set) scene.load.spritesheet(key, m.baseUrl + set.image, { frameWidth: set.frameWidth, frameHeight: set.frameHeight });
  }
  if (m.faces) {
    scene.load.spritesheet(`${m.name}-faces`, m.baseUrl + m.faces.image,
      { frameWidth: m.faces.frameWidth, frameHeight: m.faces.frameHeight });
  }
}

/**
 * Call from Scene.create(). Animation keys: `${name}-walk-down|left|right|up` and `${name}-idle-...`
 * (smooth sheet when available), plus `${name}-rm-walk-...` / `${name}-rm-idle-...` for the 3-frame sheet.
 */
export function createAnimations(scene, m) {
  const make = (prefix, set, texture) => {
    for (const [key, a] of Object.entries(set.animations)) {
      const full = `${m.name}-${prefix}${key}`;
      if (scene.anims.exists(full)) continue;
      scene.anims.create({
        key: full,
        frames: scene.anims.generateFrameNumbers(texture, { frames: a.frames }),
        frameRate: a.frameRate,
        repeat: a.repeat,
      });
    }
  };
  if (m.walk) make('rm-', m.walk, `${m.name}-walk`);
  const w = walkSet(m);
  if (w.set) make('', w.set, w.key);
}

/** Adds a sprite with walk(dir) / stop() helpers. Origin is at the feet. smooth = false -> 3-frame sheet. */
export function addCharacter(scene, m, x, y, scale = 1, smooth = true) {
  const w = walkSet(m, smooth);
  const prefix = w.tag === 'anim' ? '' : 'rm-';
  const s = scene.add.sprite(x, y, w.key, w.set.animations['idle-down'].frames[0]).setOrigin(0.5, 1).setScale(scale);
  s.charName = m.name;
  s.facing = 'down';
  s.walk = (dir) => { s.facing = dir; s.anims.play(`${m.name}-${prefix}walk-${dir}`, true); return s; };
  s.stop = (dir) => { if (dir) s.facing = dir; s.anims.play(`${m.name}-${prefix}idle-${s.facing}`, true); return s; };
  return s;
}

/** Frame index of an emotion inside the `${name}-faces` sheet (falls back to the first face). */
export function faceFrame(m, emotion) {
  return (m.faces && m.faces.frames && m.faces.frames[emotion] !== undefined) ? m.faces.frames[emotion] : 0;
}
