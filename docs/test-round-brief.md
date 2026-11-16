# Brief for a fresh-agent test round

Give this to a subagent that has never seen the project (general-purpose, background). Replace the character
paragraph and the seed. Used for round 1 (Tobi) and round 2 (Kaela), see experiment 18.

---

You are a game-dev agent. Your task: create a new playable character for a small 2D RPG made with Phaser, using
the asset pipeline in /home/tashi/Projects/claude-comfyui-research (a local ComfyUI image/video pipeline). You
have not seen this project before. Learn how to use it ONLY from its documentation: start with README.md, then
docs/03-agent-guide.md (the guide written for agents like you). Do not read docs/01 or docs/02 or the pipeline
source code unless the guide fails you, and say so if you had to.

The character (write the tags, description and props yourself, as the guide explains):
- Name: <name>
- <one paragraph: look, outfit, colours, carried items and on which side>
- Use --seed <n>.

Rules:
- Use the pipeline exactly as documented (python3 -m pipeline.character ...). Do NOT edit any file under
  pipeline/, docs/, phaser-demo/ or other characters' folders. Only characters/<name>/ and characters/index.json
  may change, and only through the pipeline.
- The pipeline stops at two review gates and asks you to look at review images. Do those reviews yourself,
  honestly and carefully: open the images with your Read tool (it shows images) and check every checklist item.
  Only approve what really passes. If something is wrong, use the documented fixes and look again. At most 3
  takes per stage; if it still fails, approve the best one and record what is wrong.
- Generation is slow (minutes per step, roughly 30-45 minutes in total). Run long commands with
  run_in_background and wait for them; do not start the same command twice. ComfyUI runs at
  http://127.0.0.1:8188. If ComfyUI crashes, follow what the guide says.
- When the character is done, do the final checks the guide describes and confirm the character appears in
  characters/index.json.

Your final report (factual and specific):
1. Every command you ran, in order, with a one-line result each.
2. For each review gate: what you saw, what you rejected and why, what you approved.
3. Final quality as you judge it from the output images (per walk direction and for the faces), plus the
   "problems" list the pipeline printed, and whether you agree with each problem.
4. Documentation feedback: every point where the README or the agent guide was unclear, wrong, missing
   something, or where you had to guess. Quote the passage. Also report anything the pipeline did that
   surprised you. This part matters most - be critical.
