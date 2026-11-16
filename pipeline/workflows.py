"""API-format workflow builders. Every function returns a `Graph` ready for `Comfy().run()`.

Settings are taken from the official ComfyUI templates shipped with ComfyUI 0.45
(image_anima_base_v1, image_flux3_klein_distilled_t2i, image_flux3_klein_distilled_image_edit).
"""
from __future__ import annotations

from .comfy import Graph, NodeRef

ANIMA_NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"


def _loras(g: Graph, model: NodeRef, loras: list[tuple[str, float]] | None) -> NodeRef:
    for name, strength in loras or []:
        model = g.add("LoraLoaderModelOnly", model=model, lora_name=name, strength_model=strength)
    return model


def _save(g: Graph, image: NodeRef, prefix: str, alpha_bg_removal: bool = False) -> None:
    """SaveImage; optionally cut out the subject with the core BiRefNet node first (RGBA PNG)."""
    if alpha_bg_removal:
        bg = g.add("LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")
        mask = g.add("RemoveBackground", bg_removal_model=bg, image=image)
        mask = g.add("InvertMask", mask=mask)  # JoinImageWithAlpha treats the mask as "transparent where 1"
        image = g.add("JoinImageWithAlpha", image=image, alpha=mask)
    g.add("SaveImage", images=image, filename_prefix=prefix)


def cutout(image_name: str, prefix: str = "ccr/cutout") -> Graph:
    """Background removal for an existing image (name in ComfyUI's input folder) -> RGBA PNG."""
    g = Graph()
    _save(g, g.add("LoadImage", image=image_name), prefix, True)
    return g


# ---------------------------------------------------------------- Anima ----
def anima_t2i(prompt: str, *, negative: str = ANIMA_NEG, width: int = 832, height: int = 1216, seed: int = 0,
              unet: str = "anima-turbo-v1.1.safetensors", steps: int = 8, cfg: float = 1.0,
              sampler: str = "er_sde", scheduler: str = "simple", loras: list[tuple[str, float]] | None = None,
              batch: int = 1, prefix: str = "ccr/anima", cutout: bool = False) -> Graph:
    """Anima (Cosmos-Predict2-based anime model). Turbo: 8 steps / cfg 1. Base: 30 steps / cfg 4."""
    g = Graph()
    model = g.add("UNETLoader", unet_name=unet, weight_dtype="default")
    model = _loras(g, model, loras)
    clip = g.add("CLIPLoader", clip_name="qwen_3_06b_base.safetensors", type="stable_diffusion")
    vae = g.add("VAELoader", vae_name="qwen_image_vae.safetensors")
    pos = g.add("CLIPTextEncode", clip=clip, text=prompt)
    neg = g.add("CLIPTextEncode", clip=clip, text=negative)
    latent = g.add("EmptyLatentImage", width=width, height=height, batch_size=batch)
    samples = g.add("KSampler", model=model, positive=pos, negative=neg, latent_image=latent, seed=seed,
                    steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler, denoise=1.0)
    img = g.add("VAEDecode", samples=samples, vae=vae)
    _save(g, img, prefix, cutout)
    return g


# ----------------------------------------------------------- FLUX 3 klein --
FLUX3_DISTILLED = "flux3-image-klein-int8-convrot.safetensors"       # 16 steps, cfg 1
FLUX3_BASE = "flux3-image-klein-base-int8-convrot.safetensors"        # 50 steps, cfg 3.5 (t2i) / 1-3.5 (edit)


def _flux3_loaders(g: Graph, loras=None, unet: str = FLUX3_DISTILLED):
    model = g.add("UNETLoader", unet_name=unet, weight_dtype="default")
    model = _loras(g, model, loras)
    clip = g.add("CLIPLoader", clip_name="qwen3vl_4b_int8_convrot.safetensors", type="flux3")
    vae = g.add("VAELoader", vae_name="flux2-vae.safetensors")
    return model, clip, vae


def flux3_t2i(prompt: str, *, width: int = 832, height: int = 1216, seed: int = 0, steps: int = 15,
              cfg: float = 1.0, sampler: str = "er_sde", scheduler: str = "simple", negative: str = "",
              loras=None, batch: int = 1, prefix: str = "ccr/flux3", cutout: bool = False) -> Graph:
    g = Graph()
    model, clip, vae = _flux3_loaders(g, loras)
    pos = g.add("CLIPTextEncode", clip=clip, text=prompt)
    neg = g.add("CLIPTextEncode", clip=clip, text=negative)
    latent = g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch)
    samples = g.add("KSampler", model=model, positive=pos, negative=neg, latent_image=latent, seed=seed,
                    steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler, denoise=1.0)
    img = g.add("VAEDecode", samples=samples, vae=vae)
    _save(g, img, prefix, cutout)
    return g


def flux3_edit(prompt: str, refs: list[str], *, width: int | None = None, height: int | None = None,
               seed: int = 0, steps: int = 16, cfg: float = 1.0, sampler: str = "euler", scheduler: str = "simple",
               ref_megapixels: float = 1.0, loras=None, batch: int = 1, prefix: str = "ccr/flux3edit",
               cutout: bool = False, unet: str = FLUX3_DISTILLED) -> Graph:
    """Reference-image edit. `refs` are names in ComfyUI's input folder (use Comfy.upload_image).

    The i-th reference is addressed in the prompt as <ref_image_i>. Write the prompt as a full
    caption of the desired result. If width/height are omitted, the output takes the size of ref 0.
    """
    g = Graph()
    model, clip, vae = _flux3_loaders(g, loras, unet)
    pos = g.add("CLIPTextEncode", clip=clip, text=prompt)
    # distilled template: zeroed conditioning; base template: empty-text negative (matters once cfg > 1)
    neg = g.add("ConditioningZeroOut", conditioning=pos) if unet == FLUX3_DISTILLED else g.add("CLIPTextEncode", clip=clip, text="")
    first_latent = None
    for name in refs:
        img = g.add("LoadImage", image=name)
        img = g.add("ImageScaleToTotalPixels", image=img, upscale_method="lanczos", megapixels=ref_megapixels,
                    resolution_steps=16)
        lat = g.add("VAEEncode", pixels=img, vae=vae)
        pos = g.add("ReferenceLatent", conditioning=pos, latent=lat)
        neg = g.add("ReferenceLatent", conditioning=neg, latent=lat)
        first_latent = first_latent or lat
    if width and height:
        latent = g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch)
    else:
        latent = g.add("LatentMultiply", samples=first_latent, multiplier=0.0)  # same trick as ReferenceMultiLatent
    samples = g.add("KSampler", model=model, positive=pos, negative=neg, latent_image=latent, seed=seed,
                    steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler, denoise=1.0)
    out = g.add("VAEDecode", samples=samples, vae=vae)
    _save(g, out, prefix, cutout)
    return g


# ------------------------------------------------- FLUX.2 klein 4B (base) --
SVNTAX_SHEET_PROMPT = ("Create a pixel art spritesheet of the character in the image. The spritesheet is a 4 by 4 grid of four "
                       "rows of frames - first row is 3 walking frames facing down and 1 frame both arms raised, second row is 3 "
                       "walking frames facing left and 1 frame jumping left, third row is 3 walking frames facing right and 1 frame "
                       "jumping right, fourth row is 3 walking frames back view facing up and 1 frame lying on floor.")


def flux2_klein_base_edit(prompt: str, refs: list[str], *, width: int = 512, height: int = 512, seed: int = 0,
                          steps: int = 20, cfg: float = 5.0, loras=None, ref_megapixels: float = 1.0,
                          prefix: str = "ccr/flux2klein") -> Graph:
    """FLUX.2 klein 4B *base* edit graph (official template), used here with the svntax sprite-sheet LoRA."""
    g = Graph()
    model = g.add("UNETLoader", unet_name="flux-2-klein-base-4b-fp8.safetensors", weight_dtype="default")
    model = _loras(g, model, loras)
    clip = g.add("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2")
    vae = g.add("VAELoader", vae_name="flux2-vae.safetensors")
    pos = g.add("CLIPTextEncode", clip=clip, text=prompt)
    neg = g.add("CLIPTextEncode", clip=clip, text="")
    for name in refs:
        img = g.add("LoadImage", image=name)
        img = g.add("ImageScaleToTotalPixels", image=img, upscale_method="lanczos", megapixels=ref_megapixels, resolution_steps=16)
        lat = g.add("VAEEncode", pixels=img, vae=vae)
        pos = g.add("ReferenceLatent", conditioning=pos, latent=lat)
        neg = g.add("ReferenceLatent", conditioning=neg, latent=lat)
    latent = g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    guider = g.add("CFGGuider", model=model, positive=pos, negative=neg, cfg=cfg)
    sigmas = g.add("Flux2Scheduler", steps=steps, width=width, height=height)
    noise = g.add("RandomNoise", noise_seed=seed)
    sampler = g.add("KSamplerSelect", sampler_name="euler")
    samples = g.add("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)
    out = g.add("VAEDecode", samples=samples, vae=vae)
    g.add("SaveImage", images=out, filename_prefix=prefix)
    return g


# --------------------------------------------------------- Qwen-Image 2.1 --
QWEN21_TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"


def qwen21_edit(prompt: str, refs: list[str], *, width: int = 1024, height: int = 1024, seed: int = 0,
                resolution: int = 1024, turbo: bool = False, steps: int = 25, cfg: float = 1.0,
                prefix: str = "ccr/qwen21") -> Graph:
    """Qwen-Image 2.1 edit (official templates). base: 25 steps euler/simple; turbo: 8 fixed sigmas. cfg 1.
    NOTE: Qwen Research licence = non-commercial. One reference: call it "the image"; several: <image1>, <image2>."""
    g = Graph()
    name = "qwen_image_2.1_turbo_int8_convrot.safetensors" if turbo else "qwen_image_2.1_int8_convrot.safetensors"
    model = g.add("UNETLoader", unet_name=name, weight_dtype="default")
    model = g.add("QwenImage21Cache", model=model, device="auto", dtype="default")
    clip = g.add("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image")
    vae = g.add("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors")
    imgs = {f"images.image_{i + 1}": g.add("LoadImage", image=n) for i, n in enumerate(refs)}
    enc = g.add("TextEncodeQwenImage21", clip=clip, vae=vae, prompt=prompt, negative_prompt="", resolution=resolution, **imgs)
    latent = g.add("EmptyLatentImage", width=width, height=height, batch_size=1)
    if turbo:
        guider = g.add("CFGGuider", model=model, positive=enc, negative=enc[1], cfg=1.0)
        sigmas = g.add("ManualSigmas", sigmas=QWEN21_TURBO_SIGMAS)
        noise = g.add("RandomNoise", noise_seed=seed)
        sampler = g.add("KSamplerSelect", sampler_name="euler")
        samples = g.add("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)
    else:
        samples = g.add("KSampler", model=model, positive=enc, negative=enc[1], latent_image=latent, seed=seed, steps=steps,
                        cfg=cfg, sampler_name="euler", scheduler="simple", denoise=1.0)
    out = g.add("VAEDecode", samples=samples, vae=vae)
    g.add("SaveImage", images=out, filename_prefix=prefix)
    return g


def qwen21_turbo_edit(prompt: str, refs: list[str], **kw) -> Graph:
    return qwen21_edit(prompt, refs, turbo=True, **kw)


# ------------------------------------------------------------- Wan 2.2 ----
# Official templates video_wan2_2_14B_i2v / video_wan2_2_14B_flf2v (ComfyUI 0.45). Two expert models: the
# high-noise one runs the first half of the steps (layout, motion), the low-noise one the rest (detail).
WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
           "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
WAN_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
WAN_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
WAN_LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
WAN_LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"


def wan22_i2v(prompt: str, start: str, *, end: str | None = None, width: int = 480, height: int = 480,
              length: int = 81, seed: int = 0, fast: bool = False, steps: int | None = None, cfg: float | None = None,
              shift: float | None = None, split: int | None = None, negative: str = WAN_NEG,
              sampler: str = "euler", scheduler: str = "simple", loras: list[tuple[str, float]] | None = None,
              prefix: str = "ccr/wan") -> Graph:
    """Wan 2.2 14B image-to-video (sampling only; decode the saved latent with `wan_decode`). `start` (and optional `end`) are image names in ComfyUI's input folder.
    With `end` the clip is pinned at both ends (first-last-frame); start == end gives a loop.
    quality: 20 steps, cfg 3.5 (4 with an end frame), shift 5 (8 with an end frame), experts split at half.
    fast=True: lightx2v 4-step LoRAs, cfg 1, split at 2.
    `length` must be 4n+1 (16 fps)."""
    g = Graph()
    steps = steps or (4 if fast else 20)
    cfg = cfg if cfg is not None else (1.0 if fast else (4.0 if end else 3.5))
    shift = shift if shift is not None else (5.0 if fast or not end else 8.0)
    split = split if split is not None else steps // 2
    clip = g.add("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan")
    vae = g.add("VAELoader", vae_name="wan_2.1_vae.safetensors")
    pos = g.add("CLIPTextEncode", clip=clip, text=prompt)
    neg = g.add("CLIPTextEncode", clip=clip, text=negative)
    models = []
    for unet, lora in ((WAN_HIGH, WAN_LORA_HIGH), (WAN_LOW, WAN_LORA_LOW)):
        m = g.add("UNETLoader", unet_name=unet, weight_dtype="default")
        m = _loras(g, m, ([(lora, 1.0)] if fast else []) + list(loras or []))
        models.append(g.add("ModelSamplingSD3", model=m, shift=shift))
    img = g.add("LoadImage", image=start)
    if end:
        cond = g.add("WanFirstLastFrameToVideo", positive=pos, negative=neg, vae=vae, width=width, height=height,
                     length=length, batch_size=1, start_image=img, end_image=g.add("LoadImage", image=end))
    else:
        cond = g.add("WanImageToVideo", positive=pos, negative=neg, vae=vae, width=width, height=height,
                     length=length, batch_size=1, start_image=img)
    common = dict(positive=cond, negative=cond[1], steps=steps, cfg=cfg, sampler_name=sampler, scheduler=scheduler)
    hi = g.add("KSamplerAdvanced", model=models[0], latent_image=cond[2], add_noise="enable", noise_seed=seed,
               start_at_step=0, end_at_step=split, return_with_leftover_noise="enable", **common)
    lo = g.add("KSamplerAdvanced", model=models[1], latent_image=hi, add_noise="disable", noise_seed=0,
               start_at_step=split, end_at_step=10000, return_with_leftover_noise="disable", **common)
    # Sampling and decoding are two jobs: with the 14B model still in VRAM the video VAE runs out of memory on a
    # 16 GB card, so the latent is saved, the client unloads the models (POST /free) and `wan_decode` follows.
    g.add("SaveLatent", samples=lo, filename_prefix=prefix + "_latent")
    return g


def wan_decode(latent: str, *, fps: float = 16, prefix: str = "ccr/wan", preview: bool = True) -> Graph:
    """Decode a latent saved by `wan22_i2v` ("<subfolder>/<file>.latent [output]") into PNG frames
    (+ an animated WEBP preview)."""
    g = Graph()
    vae = g.add("VAELoader", vae_name="wan_2.1_vae.safetensors")
    frames = g.add("VAEDecode", samples=g.add("LoadLatent", latent=latent), vae=vae)
    g.add("SaveImage", images=frames, filename_prefix=prefix)
    if preview:
        g.add("SaveAnimatedWEBP", images=frames, filename_prefix=prefix + "_anim", fps=fps, lossless=False,
              quality=90, method="default")
    return g
