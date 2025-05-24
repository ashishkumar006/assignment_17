import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
import gradio as gr
from pathlib import Path
import numpy as np
import os
import gc
import psutil
from PIL import Image

# Detect if running in Colab
IN_COLAB = 'COLAB_GPU' in os.environ

# Custom cyan loss implementation - FIXED FOR LATENT SPACE
def cyan_loss(step: int, timestep: torch.FloatTensor, latents: torch.Tensor) -> torch.Tensor:
    """Custom cyan effect callback that works properly in SD's latent space."""
    try:
        # Input validation
        if not isinstance(step, int) or step < 0:
            return latents
            
        if not torch.is_tensor(latents):
            return latents
            
        if not torch.is_tensor(timestep):
            return latents

        # Only apply effect in later steps for stability
        if step < 30:
            return latents
            
        device = latents.device
        
        # Ensure we have 4-channel latent space
        if latents.shape[1] != 4:
            return latents
        
        # PROPER LATENT SPACE MANIPULATION FOR CYAN BIAS
        # Instead of treating channels as RGB, we apply a subtle transformation
        # that encourages cyan-like features in the final decoded image
        
        # Create a cyan-encouraging transformation matrix
        # This is based on how SD's VAE decoder interprets latent channels
        cyan_transform = torch.tensor([
            [0.8, 0.1, 0.1, 0.0],   # Channel 0: reduce slightly
            [0.1, 1.2, 0.1, 0.0],   # Channel 1: boost (often correlates with green-blue)
            [0.1, 0.1, 1.3, 0.0],   # Channel 2: boost more (often correlates with blue)  
            [0.0, 0.0, 0.0, 1.0]    # Channel 3: keep unchanged (structure)
        ], device=device, dtype=latents.dtype)
        
        # Apply transformation
        batch_size, channels, height, width = latents.shape
        latents_flat = latents.view(batch_size, channels, -1)  # Flatten spatial dims
        
        # Apply the transformation
        transformed_flat = torch.matmul(cyan_transform, latents_flat)
        cyan_latents = transformed_flat.view(batch_size, channels, height, width)
        
        # Progressive application with moderate strength
        progress = min((step - 30) / 15.0, 1.0)  # Ramp up over 15 steps
        strength = 0.25 * progress  # Much gentler effect
        
        # Blend original and transformed latents
        modified = latents * (1 - strength) + cyan_latents * strength
        
        # Ensure values stay in reasonable range for stability
        modified = torch.clamp(modified, -4.0, 4.0)
        
        if step % 10 == 0:  # Less frequent logging
            print(f"Step {step}: Applying cyan transformation with strength {strength:.3f}")
        
        return modified
        
    except Exception as e:
        print(f"Error in cyan_loss: {str(e)}")
        return latents

# Style configurations with enhanced names
STYLES = {
    "✨ Dreamy Portrait": {
        "embedding_path": "buhu.bin",
        "seed": 42,
        "token": "<buhu>"
    },    
    "🌆 Neon Cyberpunk": {
        "embedding_path": "cyberpunk.bin",
        "seed": 123,
        "token": "<cyberpunk>"
    },
    "🎋 Asian Anime": {
        "embedding_path": "hanfu_animestyle.bin",
        "seed": 456,
        "token": "<hanfu>"
    },
    "⭐ Pokemon Style": {
        "embedding_path": "pokemon.bin",
        "seed": 321,
        "token": "<pokemon>"
    },
    "🏛️ Solomon Temple": {
        "embedding_path": "solomon temple.bin",
        "seed": 888,
        "token": "<solomon>"
    }
}

def generate_image(prompt, style_name, use_cyan_loss=False, seed=None):
    """Generate an image with the specified style"""
    print("\n🎨 Starting image generation...")
    print(f"Prompt: {prompt}")
    print(f"Style: {style_name}")
    print(f"Use cyan loss: {use_cyan_loss}")
    print(f"Seed: {seed}")
    
    # Initial memory optimization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimize_memory(device)
    
    if not prompt or prompt.strip() == "":
        raise ValueError("Prompt cannot be empty")
    
    if seed is None:
        seed = STYLES[style_name]["seed"]
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Initialize the pipeline with optimized settings        print("\n📦 Loading Stable Diffusion model...")
        model_id = "CompVis/stable-diffusion-v1-4"
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir="./models"
        ).to(device)
        
        # Enhanced memory optimizations with safer configuration
        pipeline.enable_attention_slicing(slice_size="max")
        if device == "cuda":
            pipeline.enable_vae_slicing()
            pipeline.enable_model_cpu_offload()  # Use model CPU offload instead of sequential
        print("✓ Advanced memory optimizations enabled")
        
        # Load style embedding
        style_info = STYLES[style_name]
        embedding_path = Path(style_info['embedding_path'])
        token = style_info['token']
        
        print(f"\n🔄 Loading style: {style_name}")
        print(f"Path: {embedding_path.absolute()}")
        
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        # Load embedding
        learned_embeds = torch.load(embedding_path, map_location=device)
        if not learned_embeds or not isinstance(learned_embeds, dict):
            raise ValueError("Invalid embedding file format")
        
        # Get embedding vector
        key = list(learned_embeds.keys())[0]
        embedding_vector = learned_embeds[key]
        if not isinstance(embedding_vector, torch.Tensor):
            raise ValueError("Embedding data is not a tensor")
        
        # Add token to tokenizer
        special_tokens_dict = {'additional_special_tokens': [token]}
        num_added = pipeline.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added > 0:
            pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
        
        # Get token ID and verify
        token_id = pipeline.tokenizer.convert_tokens_to_ids(token)
        if token_id == pipeline.tokenizer.unk_token_id:
            raise ValueError(f"Token {token} not properly added to tokenizer")
        
        # Handle embedding dimensions
        required_dim = pipeline.text_encoder.get_input_embeddings().weight.data[0].shape[0]
        if embedding_vector.shape[0] != required_dim:
            print(f"⚠️ Reshaping embedding from {embedding_vector.shape[0]} to {required_dim}")
            if embedding_vector.shape[0] > required_dim:
                embedding_vector = embedding_vector[:required_dim]
            else:
                padding = torch.zeros(required_dim - embedding_vector.shape[0], device=device)
                embedding_vector = torch.cat([embedding_vector, padding])
        
        # Store embedding
        pipeline.text_encoder.get_input_embeddings().weight.data[token_id] = embedding_vector
        
        # Verify token works
        test_prompt = f"test {token} prompt"
        input_ids = pipeline.tokenizer.encode(test_prompt, return_tensors="pt")
        if token_id not in input_ids[0]:
            raise ValueError(f"Token verification failed - {token} not found in test prompt")
        
        print(f"✓ Style {style_name} loaded successfully")
        
        # Prepare prompt with style
        styled_prompt = f"{token} {prompt}"
        print(f"📝 Using prompt: {styled_prompt}")
        
        # Generate image with optimized parameters
        print("🎨 Generating image...")
        generator = torch.Generator(device=device).manual_seed(seed)
        
        generation_params = {
            "prompt": styled_prompt,
            "generator": generator,
            "num_inference_steps": 45,  # Slightly reduced for better speed/quality balance
            "guidance_scale": 8.5,      # Increased for better prompt adherence
            "height": 512,              # Ensure consistent dimensions
            "width": 512
        }
        
        if use_cyan_loss:
            def callback_wrapper(pipe, step_index, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]
                # Clear CUDA cache periodically to prevent memory buildup
                if step_index % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"latents": cyan_loss(step_index, timestep, latents)}
            
            generation_params["callback_on_step_end"] = callback_wrapper
            
        # Generate with optimized memory handling
        with torch.inference_mode():
            image = pipeline(**generation_params).images[0]
            
        # Clean up CUDA cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("✨ Image generated successfully")
        return image
        
    except Exception as e:
        error_msg = f"""
❌ Error during image generation:
Type: {type(e).__name__}
Message: {str(e)}
Style: {style_name}
Device: {device}
Memory: {torch.cuda.memory_summary() if torch.cuda.is_available() else 'No GPU'}
"""
        print(error_msg)
        raise Exception(f"Image generation failed: {str(e)}") from e

# Memory optimization function
def optimize_memory(device, previous_image=None):
    """Clean up memory and caches"""
    try:
        if previous_image and isinstance(previous_image, Image.Image):
            previous_image.close()
        
        if device == "cuda":
            # GPU cleanup
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # CPU cleanup
        gc.collect()
        
        # Log memory status
        process = psutil.Process()
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        if device == "cuda":
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Memory optimization error: {str(e)}")

# Gradio interface
def create_interface():
    with gr.Blocks() as app:
        gr.Markdown("# Artistic Style Transfer with Cyan Loss")
        gr.Markdown("*Generate AI art with Stable Diffusion and optional cyan color effect*")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Enter your prompt", placeholder="A serene landscape...")
                style = gr.Dropdown(
                    choices=list(STYLES.keys()),
                    value="✨ Dreamy Portrait",
                    label="Select Style"
                )
                use_cyan = gr.Checkbox(label="Apply Cyan Loss Effect", value=False)
                seed = gr.Number(label="Seed (optional)", value=None)
                btn = gr.Button("Generate")
            
            with gr.Column():
                output = gr.Image(label="Generated Image", type="pil")
                status = gr.Markdown("System Status")
                error_output = gr.Markdown("Ready to generate images...")
                
                info = gr.Markdown("""
## Quick Instructions:
1. Enter a detailed prompt describing your desired image
2. Select one of the 5 artistic styles
3. Toggle "Apply Cyan Loss Effect" for a blue-green tinted aesthetic
4. Optionally set a seed number for reproducible results
5. Click "Generate" and wait for your image
                """)

        def wrapped_generate(*args):
            prompt = args[0] if args else None
            style_name = args[1] if len(args) > 1 else None
            
            # Validate inputs
            if not prompt or prompt.strip() == "":
                error_msg = "⚠️ Please enter a prompt. The prompt cannot be empty."
                print(error_msg)
                status.value = "❌ Generation failed - No prompt provided"
                error_output.value = error_msg
                return None
                
            if not style_name or style_name not in STYLES:
                error_msg = "⚠️ Please select a valid style."
                print(error_msg)
                status.value = "❌ Generation failed - Invalid style"
                error_output.value = error_msg
                return None
            
            status.value = "⏳ Generating image... Please wait..."
            error_output.value = ""
            
            try:
                print("\nStarting generation...")
                print(f"Input args: {args}")
                
                result = generate_image(*args)
                
                if result is None:
                    raise Exception("Generation returned no image")
                
                print("Generation completed, updating UI...")
                status.value = "✅ Image generated successfully!"
                error_output.value = ""
                return result
                
            except Exception as e:
                error_msg = f"""
🚨 Error Details:
Type: {type(e).__name__}
Message: {str(e)}

Current directory: {Path.cwd()}
Available files in sd-concepts-library:
{list(Path('sd-concepts-library').glob('*')) if Path('sd-concepts-library').exists() else 'Directory not found'}

💡 Troubleshooting:
1. Check if style files exist in the correct location
2. Make sure you have enough GPU memory
3. Try a simpler prompt
"""
                print("\nError occurred:")
                print(error_msg)
                status.value = "❌ Generation failed"
                error_output.value = error_msg
                return None
        
        btn.click(
            fn=wrapped_generate,
            inputs=[prompt, style, use_cyan, seed],
            outputs=[output]
        )
    
    return app

if __name__ == "__main__":
    try:
        print("Starting Gradio interface...")
        print("\nGPU Status:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        app = create_interface()
        # Launch with simpler configuration
        app.launch(share=True)  # This will create both local and public URLs
        
    except Exception as e:
        print(f"Error starting the app: {str(e)}")