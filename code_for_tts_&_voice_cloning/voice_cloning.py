from TTS.api import TTS
import torch
import functools

# Save the original torch.load function
original_torch_load = torch.load
# Create a wrapper that forces weights_only=False
#to bypass the security check at source of the checkpoint
@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

try:
    # Initialize the TTS model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts.to(device)
    
    # Generate speech by cloning a voice
    tts.tts_to_file(
        text="Alright It's February now and why are you still waiting to have that better relationship with your wife .",
        file_path="output_xtts3.wav",
        speaker_wav= r"C:\Users\SAI HITESH KOTA\Desktop\pythonproject\user_voice3.wav",  # Update this path
        language="en"
    )
finally:
    # Restore the original torch.load function
    torch.load = original_torch_load