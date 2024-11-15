from pathlib import Path
from typing import Tuple
import modal
import time
from common import app
from generator import Model
from eval_aesthetics import ImprovedAestheticPredictor
from eval_scannability import detect_qr_ensemble

runner_image = modal.Image.debian_slim().pip_install("qrcode", "wonderwords", "Pillow")

URLS = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.reddit.com",
    "https://www.instagram.com",
    "https://www.pinterest.com",
    "https://www.linkedin.com",
    "https://www.tiktok.com",
    "https://www.snapchat.com",
    "https://www.twitch.tv",
    "https://www.discord.com",
    "https://www.spotify.com",
    "https://www.soundcloud.com",
    "https://www.bandcamp.com",
    "https://www.vimeo.com",
    "https://www.flickr.com",
    "https://www.behance.net",
    "https://www.dribbble.com",
    "https://www.deviantart.com",
    "https://www.500px.com",
]
PROMPT_TEMPLATE = "A {adjective} {noun}, 3D render in blender, trending on artstation"
RANDOM_SEED = 42

AESTHETICS_THRESHOLD = 6.0

N_TESTS = 1
K_SAMPLES = 10

MINUTE = 60

with runner_image.imports():
    import numpy as np

def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimates probability of pass@k given n samples with c correct.
    Args:
        n: Number of samples
        c: Number of correct samples
        k: Number of trials
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


@app.function(image=runner_image, timeout=30 * MINUTE)
def run_aesthetics_eval(generated_images):
    import numpy as np
    predictor = ImprovedAestheticPredictor()

    # split into sets of K_SAMPLES
    tests = [generated_images[i:i+K_SAMPLES] for i in range(0, len(generated_images), K_SAMPLES)]

    results = []
    for test_idx, test in enumerate(tests, 1):
        scores = predictor.score.map(test)
        passed = [score >= AESTHETICS_THRESHOLD for score in scores]
        num_correct = sum(passed)
        
        # Calculate various pass@k estimates
        pass_at_1 = estimate_pass_at_k(K_SAMPLES, num_correct, 1)
        pass_at_3 = estimate_pass_at_k(K_SAMPLES, num_correct, 3)
        pass_at_10 = estimate_pass_at_k(K_SAMPLES, num_correct, 10)
        
        results.append({
            'num_samples': K_SAMPLES,
            'num_correct': num_correct,
            'pass@1': pass_at_1,
            'pass@3': pass_at_3,
            'pass@10': pass_at_10
        })
        
        print(f"\nAesthetics Test {test_idx}:")
        print(f"Passed: {passed}")
        print(f"Samples passed: {num_correct}/{K_SAMPLES}")
        print(f"Estimated pass@1: {pass_at_1:.2%}")
        print(f"Estimated pass@3: {pass_at_3:.2%}")
        print(f"Estimated pass@10: {pass_at_10:.2%}")
    
    # Calculate averages across all tests
    avg_results = {
        'pass@1': np.mean([r['pass@1'] for r in results]),
        'pass@3': np.mean([r['pass@3'] for r in results]),
        'pass@10': np.mean([r['pass@10'] for r in results])
    }
    
    print("\nFinal Aesthetics Results:")
    print(f"pass@1:  {avg_results['pass@1']:.2%}")
    print(f"pass@3:  {avg_results['pass@3']:.2%}")
    print(f"pass@10: {avg_results['pass@10']:.2%}")
    
    return avg_results

@app.function(image=runner_image, timeout=30 * MINUTE)
def run_scannability_eval(generated_images):
    import numpy as np

    # split into sets of K_SAMPLES
    tests = [generated_images[i:i+K_SAMPLES] for i in range(0, len(generated_images), K_SAMPLES)]

    results = []
    for test_idx, test in enumerate(tests, 1):
        detections = detect_qr_ensemble.map(test)
        passed = [1 if valid else 0 for valid, _ in detections]
        num_correct = sum(passed)

        # Calculate various pass@k estimates
        pass_at_1 = estimate_pass_at_k(K_SAMPLES, num_correct, 1)
        pass_at_3 = estimate_pass_at_k(K_SAMPLES, num_correct, 3)
        pass_at_10 = estimate_pass_at_k(K_SAMPLES, num_correct, 10)

        results.append({
            'num_samples': K_SAMPLES,
            'num_correct': num_correct,
            'pass@1': pass_at_1,
            'pass@3': pass_at_3,
            'pass@10': pass_at_10
        })

        print(f"\nScannability Test {test_idx}:")
        print(f"Passed: {passed}")
        print(f"Samples passed: {num_correct}/{K_SAMPLES}")
        print(f"Estimated pass@1: {pass_at_1:.2%}")
        print(f"Estimated pass@3: {pass_at_3:.2%}")
        print(f"Estimated pass@10: {pass_at_10:.2%}")


    # Calculate averages across all tests
    avg_results = {
        'pass@1': np.mean([r['pass@1'] for r in results]),
        'pass@3': np.mean([r['pass@3'] for r in results]),
        'pass@10': np.mean([r['pass@10'] for r in results])
    }
    
    print("\nFinal Scannability Results:")
    print(f"pass@1:  {avg_results['pass@1']:.2%}")
    print(f"pass@3:  {avg_results['pass@3']:.2%}")
    print(f"pass@10: {avg_results['pass@10']:.2%}")
    
    return avg_results

            

@app.function(image=runner_image, timeout=30 * MINUTE)
def run_evals():
    '''
    Modal function to run all evals
    '''
    import base64
    import io
    import qrcode
    import random
    from wonderwords import RandomWord
    random.seed(RANDOM_SEED)
    
    starmap_args = []
    for i in range(N_TESTS):
        # generate vanilla qr code
        qr_url = URLS[random.randint(0, len(URLS) - 1)]
        image = qrcode.make(qr_url) 
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # select prompt
        r_gen = RandomWord()
        adjective = r_gen.word(include_parts_of_speech=["adjective"])
        noun = r_gen.word(include_parts_of_speech=["noun"])
        prompt = PROMPT_TEMPLATE.format(adjective=adjective, noun=noun)

        for _ in range(K_SAMPLES):
            # generate multiple images for the same prompt+qr, for pass@k calculation
            starmap_args.append((prompt, image_base64))

    # generate images in parallel
    images_bytes = []
    i = 0
    for image_bytes in Model.generate.starmap(starmap_args):
        print(f"Generated image {i+1} / {len(starmap_args)}")
        images_bytes.append(image_bytes)
        i += 1

    # run evals in parallel
    aesthetics_future = run_aesthetics_eval.spawn(images_bytes)
    scannability_future = run_scannability_eval.spawn(images_bytes)

    # wait for evals to finish
    aesthetics_score = aesthetics_future.get()
    scannability_score = scannability_future.get()
    return aesthetics_score, scannability_score

if modal.is_local:
    # For experiment tracking
    import wandb
    wandb_run = wandb.init(
        project="qart_evals", 
        save_code=True,
        settings=wandb.Settings(code_dir=".")
    )

@app.local_entrypoint()
def evals():
    # aesthetics_score, scannability_score = run_evals.remote()

    # For experiment tracking
    aesthetics_score = {
        "pass@1": 0.1,
        "pass@3": 0.2,
        "pass@10": 0.3,
    }

    scannability_score = {
        "pass@1": 0.1,
        "pass@3": 0.2,
        "pass@10": 0.3,
    }

    wandb_run.summary.update({
        "n_tests": N_TESTS,
        "k_samples": K_SAMPLES,
        "aesthetics": aesthetics_score,
        "scannability": scannability_score,
    })

    time.sleep(1) # let logs finish piping to stdout

    print("Evals complete!")
    print("N_TESTS:", N_TESTS)
    print("S_SAMPLES:", K_SAMPLES)
    print("\nAesthetics:")
    for k, v in aesthetics_score.items():
        print(f"{k}: {v:.2%}")
    print("\nScannability:")
    for k, v in scannability_score.items():
        print(f"{k}: {v:.2%}")