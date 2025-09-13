import os
import torch
import random
import numpy as np
from diffusers import StableDiffusion3Pipeline

# Function to set seed for all random operations
def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return generator

# Local path to Stable Diffusion 3
model_directory = "/root/autodl-tmp/Stable-Diffusion-3-Medium"

# Stable Diffusion 3 pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(model_directory, torch_dtype=torch.float16).to("cuda")

# Basic parameters
seed = 1  # [1 2]
generator = seed_everywhere(seed)

# Prompts
prompts = [
    'A graceful cat sitting in a warm and story-rich environment, highlighting its silky fur.',  # 0
    'A beautifully detailed dog with expressive eyes and a unique coat stands in a scenic natural setting.',  # 1
    'A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, with a serene garden visible through the window.',  # 2
    'A majestic orange cat with a crown, sitting on a throne in a medieval castle, surrounded by intricate tapestries and candlelight.',  # 3
    'A gray cat with green eyes, sitting on a wooden porch, with soft sunlight highlighting its fur and a blurred garden in the background.',  # 4
    'A white cat with round, expressive eyes, sitting on a leather armchair in a cozy library filled with books and warm lighting.',  # 5
    'A black cat sitting on a stack of old books in a dusty attic, its multicolored fur contrasting with the vintage surroundings, with beams of sunlight filtering through a small window.',  # 6
    'A fluffy gray-and-white cat with golden eyes sits curled up on a cozy knitted blanket by a crackling fireplace.',  # 7
    'A brown cat with soft fur and green eyes sits calmly on a rustic wooden table in a sunlit kitchen.',  # 8
    'A black-and-white cat with curious eyes sits on a wooden porch, surrounded by autumn leaves and soft sunlight.',  # 9
    'A sleek brown cat with golden eyes sits on a polished grand piano, its fur reflecting the soft light of the room.',  # 10
    'A sleek black cat with yellow eyes sits on a cobblestone street at dusk, its fur glowing under the light of a streetlamp.',  # 11
    'A majestic dog with striking blue eyes and a muscular build stands alert on a rocky cliff edge, its thick, wavy fur glowing in the golden hour sunlight.',  # 12
    'A graceful dog with silky, well-groomed fur and deep, soulful eyes sits calmly in a sunlit meadow, its alert ears perked up and a subtle smile hinting at its friendly nature.',  # 13
    'A lively dog with a lean, athletic physique dashes through a field of tall grass, its wagging tail and bright, inquisitive eyes capturing pure joy as sunlight streams through the blades.',  # 14
    'A contented dog with soft, fluffy fur and gentle, half-closed eyes lies comfortably on a cozy couch.',  # 15
    'A curious dog with a finely detailed coat marked by subtle brindle patterns carefully sniffs the ground in an autumn forest.',  # 16
    'A dog with a sleek, black coat and bright, alert eyes runs through a shallow stream, water splashing around its paws, with sunlight reflecting off the ripples.',  # 17
    'A dog with a short, brindle coat and a strong jawline sits by a campfire, its eyes reflecting the flickering flames and ears twitching at the sound of crackling wood.',  # 18
    'A dog with a thick, double-layered coat stands in a snowy field, its breath visible in the cold air and snowflakes clinging to its fur, looking intently at something ahead.',  # 19
    'A dog with a curly, white coat and a pink nose plays in a field of wildflowers, its tongue out and tail wagging energetically, surrounded by vibrant colors.',  # 20
    'A lively dog with a glossy, golden coat and a slightly tilted head looks up with curious eyes, its ears perked and nose twitching, standing in a sunlit garden filled with vibrant flowers.',  # 21
    ]

# Get all images
for index1, public_prompt in enumerate(prompts):
    for index2, personal_prompt in enumerate(prompts):
        for common_step in range(0,28):
            image1 = pipe(prompt=public_prompt, common_step=common_step, prompt_unchanged=True, generator=generator, skip=True).images[0]
            image2 = pipe(prompt=personal_prompt, common_step=common_step, prompt_unchanged=False, generator=generator).images[0]
            file_name = "Public" + str(index1) + "_Personal" + str(index2) + "_CommonStep" + str(common_step) + ".png"
            folder_name = "./Results"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            image2.save(os.path.join(folder_name, file_name))
