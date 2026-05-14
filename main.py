import subprocess
import sys
import os

def run_script(script_path):
    # Set PYTHONPATH to the project root so all imports work
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, script_path], check=True, env=env)

if __name__ == "__main__":
    # print("=== Step 1: Train GAN ===")
    # run_script("src/gan/gan_train.py")
    #
    # print("\n=== Step 2: Generate Synthetic Data (GAN) ===")
    # run_script("src/gan_generate.py")
    #
    # print("\n=== Step 2b: Traditional Augmentation for D5 ===")
    # run_script("src/traditional_augment.py")

    print("\n=== Step 3: Train Classifiers on D1–D5 ===")
    run_script("src/model/classifier_train.py")

    print("\n=== Step 4: Evaluate on Test Set ===")
    run_script("src/evaluation/evaluate.py")

    print("\n=== Step 5: Compute FID ===")
    run_script("src/evaluation/fid_score.py")

    print("\n🎉 Pipeline complete. Results in 'outputs/' and 'experiments/'.")