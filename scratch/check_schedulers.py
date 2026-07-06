import torch
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler

# Create DPMSolverMultistepScheduler
scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="sample",
)
scheduler.set_timesteps(5)

# Simulate 5 steps with a constant model output
noisy = torch.randn(1, 64, 7)
model_output = torch.full((1, 64, 7), 0.14) # Simulate model predicting exactly 0.14

for timestep in scheduler.timesteps:
    noisy = scheduler.step(model_output, timestep, noisy).prev_sample

print("DPMSolverMultistepScheduler Final noisy mean:", noisy.mean().item())

# Create DDPMScheduler
scheduler_ddpm = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="sample",
    clip_sample=False
)
scheduler_ddpm.set_timesteps(5)

noisy_ddpm = torch.randn(1, 64, 7)
for timestep in scheduler_ddpm.timesteps:
    noisy_ddpm = scheduler_ddpm.step(model_output, timestep, noisy_ddpm).prev_sample
    
print("DDPMScheduler Final noisy mean:", noisy_ddpm.mean().item())
