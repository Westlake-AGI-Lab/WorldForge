SCENE_PROMPTS = {
    "horn": (
        "A bullet time effect video in a 3D photography style, where the entire museum exhibit is completely frozen in a single moment of time. A massive Triceratops skull is captured in a perfectly static, fixed display, its fossilized texture and imposing horns utterly motionless, as if suspended in time itself. The background, including the blurred outlines of other dinosaur exhibits, the structural elements of the museum, and the subtly textured floor, is absolutely motionless like a frozen three-dimensional image. The sole source of movement is the camera itself, which moves smoothly and stably in a gentle arc around the skull, capturing this prehistoric relic from a continuously shifting perspective to fully showcase the time-stopped setting."
    ),
    "null": (
        "A bullet time effect video in a 3D photography style."
    ),
    "smoke": (
        "Cinematic style. A heavyset Asian man in a striking plaid suit and dark sunglasses stands beside a light blue and a white taxi. He is lighting a cigarette with a silver lighter. A brief puff of smoke curls upwards. Once lit, he performs a swift, flamboyant flick of both hands outward to his sides, then smoothly places both hands into his suit pockets. In the background, near a building entrance with doorways and columns, several figures can be seen, including individuals who appear to be staff in uniform. The camera lens moves in a slow, steady arc around him."
    ),
  
    "truck": (
        "In a bullet time effect video with a 3D photography style, the entire urban street scene is completely frozen in a single moment of time. A vintage truck is captured in a perfectly static, silent state on a wide concrete sidewalk. Its light blue cab and chassis show a weathered patina, while the brown wooden planks of its cargo bed are held in absolute stillness; every detail, from the chipped paint to the texture of the wood grain, is rendered with sharp, unmoving clarity. The entire background is like a frozen three-dimensional image: the leaves on the city trees are perfectly still, with no hint of a breeze, and the surrounding street furniture, modern buildings, and even the manhole cover on the pavement are all locked in this silent, motionless moment. The only sense of dynamism comes from the implied camera, which moves smoothly and stably in a gentle arc around the scene, capturing this time-stopped moment from a continuously shifting perspective to fully showcase its bullet time setting."
    ),
    "Oil_painting":(
        "Oil painting photography in a bullet time effect video, this oil painting of Socrates' death is absolutely frozen in a single moment, every element suspended in time. Socrates sits motionless on his bed, one arm raised in a statically frozen gesture, his fingers unmoving, the other arm extended towards the hemlock, his hand and fingers also completely frozen. The figures around him are depicted in various frozen postures of sorrow and contemplation, their eyes fixed and unblinking, their arms and legs held in static poses. Every gesture, every expression, every limb – including all fingers and eyes – is utterly frozen, creating a completely fixed tableau within the scene of the oil painting. The texture of the paint, the unmoving folds of clothing, and the sharp, frozen shadows all reinforce the absolute stillness. The only dynamism in the video will be the slow, steady camera movement across this completely frozen scene."
    ),
    "fast":(
        "Realistic style. On a paved road flanked by dense green trees and a guardrail on the left, a red van is speeding forward, moving rapidly away from the lens. Following closely behind the red van is a silver car, maintaining a high speed. The camera moves backward quickly, retreating from the vehicles, while simultaneously and slowly rising upwards to transition into a high-angle overhead view, revealing more of the road and the surrounding forest environment."
        ),
    }

def get_prompt(scene_name):
    """Get prompt for the specified scene name. Returns default if not found."""
    if scene_name in SCENE_PROMPTS:
        return SCENE_PROMPTS[scene_name]
    else:
        print(f"Warning: Scene '{scene_name}' not found, using default prompt")
        return SCENE_PROMPTS["null"]

def list_available_scenes():
    """List all available scene names."""
    return list(SCENE_PROMPTS.keys()) 
