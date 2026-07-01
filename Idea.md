I have been exploring a robotics architecture that departs from the current paradigm of Large Action Models (LAMs). Rather than using a Large Language Model to directly generate executable action plans, my proposal is to use the LLM as a **skill synthesis engine**.

The central idea is that the LLM should infer **which capabilities are required** to accomplish a task, instead of deciding **which actions should be executed**. These capabilities become reusable Reinforcement Learning skills that are organized within a mathematically structured workflow.

As an initial proof of concept, I have started developing an experimental framework:

**Action-Workflow**
https://github.com/KELEBA-LAM/Action-Workflow

The current prototype focuses on workflow generation, but I envision extending it far beyond symbolic planning.

The complete architecture can be summarized as follows:

Natural Language Goal
→ Multimodal Perception (vision, audio, language, proprioception, tactile sensing)
→ Shared World Model
→ LLM Reasoning
→ RL Skill Synthesis
→ Formal Skill Composition
→ Executable Workflow
→ RL Policies
→ Robot Controllers
→ Physical Execution
→ Multimodal Experience
→ Skill Distillation
→ Continuous Skill Library Evolution

Unlike current LAM architectures, the LLM would not generate a sequence of actions. Instead, it would generate a **graph of abstract skills**, each corresponding to a capability that can be learned, reused, refined, or replaced.

I believe these skills can naturally be modeled as morphisms in a category:

State A ── Skill₁ ──► State B
State B ── Skill₂ ──► State C

whose composition yields

Skill₂ ∘ Skill₁

This provides a principled mathematical foundation for workflow construction, allowing formal reasoning about composability, safety constraints, invariants, and correctness.

More importantly, I envision a hierarchy of functors connecting multiple abstraction levels:

Goal Category
→ Skill Category
→ Policy Category
→ Motor Command Category

Such mappings would preserve semantic and safety properties throughout the compilation process, from high-level intentions down to low-level robot control.

The learning process would also be fundamentally multimodal.

Each robot execution would generate a complete multimodal trajectory including:

* RGB and depth vision
* Audio
* Natural language interactions
* Proprioception
* Tactile feedback
* Motor trajectories
* RL rewards
* Internal world-state evolution

These trajectories would then be distilled to automatically:

* improve existing skills,
* create entirely new skills,
* specialize policies,
* generalize reusable behaviors,
* enrich a continuously evolving procedural memory.

I see recent work such as SkillRL

https://github.com/aiming-lab/SkillRL

as an important building block toward this direction. However, my ambition is to integrate RL skill learning into a broader cognitive architecture where multimodal perception, symbolic reasoning, category-theoretic composition, reinforcement learning, and lifelong procedural memory become parts of a single unified system.

Another key aspect is an **Any-to-Any cross-attention multimodal model**, allowing every modality (vision, audio, language, touch, proprioception, and action) to influence every other modality during reasoning and learning. Instead of treating multimodal perception as a simple Vision-Language interface, the robot would build a unified latent representation of its interaction with the world.

Conceptually, I view the LLM less as a planner and more as a **compiler of behavioral abstractions**. Reinforcement learning then becomes the optimization mechanism that grounds these abstractions into executable motor policies, while multimodal experience continuously expands and refines the skill library.

Ultimately, I believe this could evolve into a general cognitive architecture for robotics, where language, perception, reasoning, learning, procedural memory, and formal verification coexist within the same computational framework.
